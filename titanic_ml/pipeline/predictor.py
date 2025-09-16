"""Prediction and evaluation pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from ..core.models import PredictionResult
from ..core.preprocessor import TitanicPreprocessor
from ..core.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class TitanicPredictor:
    """Complete prediction pipeline for Titanic dataset."""
    
    def __init__(self, use_polars: bool = True):
        self.use_polars = use_polars
        self.preprocessor: Optional[TitanicPreprocessor] = None
        self.trainer: Optional[ModelTrainer] = None
        
    def load_pipeline(self, model_path: Path, preprocessor_path: Optional[Path] = None) -> None:
        """Load trained model and preprocessor."""
        model_path = Path(model_path)
        
        # Load model
        self.trainer = ModelTrainer.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load preprocessor if available
        if preprocessor_path and preprocessor_path.exists():
            import joblib
            preprocessor_data = joblib.load(preprocessor_path)
            self.preprocessor = preprocessor_data['preprocessor']
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            # Create new preprocessor (will need to be fitted)
            self.preprocessor = TitanicPreprocessor(use_polars=self.use_polars)
            logger.warning("No preprocessor found, using default settings")
    
    def predict_single(self, passenger_data: Dict) -> PredictionResult:
        """Make prediction for a single passenger."""
        if not self.trainer or not self.preprocessor:
            raise ValueError("Pipeline not loaded. Call load_pipeline first.")
        
        # Convert to DataFrame
        if self.use_polars:
            df = pl.DataFrame([passenger_data])
        else:
            df = pd.DataFrame([passenger_data])
        
        # Preprocess
        x_processed = self.preprocessor.transform(df)
        
        # Predict
        prediction = self.trainer.predict(x_processed)[0]
        probability = self.trainer.predict_proba(x_processed)[0]
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(probability - 0.5) * 2
        
        return PredictionResult(
            passenger_id=passenger_data.get('PassengerId', 0),
            survival_probability=float(probability),
            prediction=int(prediction),
            confidence=float(confidence)
        )
    
    def predict_batch(
        self, 
        df: Union[pd.DataFrame, pl.DataFrame]
    ) -> List[PredictionResult]:
        """Make predictions for multiple passengers."""
        if not self.trainer or not self.preprocessor:
            raise ValueError("Pipeline not loaded. Call load_pipeline first.")
        
        # Preprocess data
        x_processed = self.preprocessor.transform(df)
        
        # Make predictions
        predictions = self.trainer.predict(x_processed)
        probabilities = self.trainer.predict_proba(x_processed)
        
        # Get passenger IDs
        if self.use_polars and isinstance(df, pl.DataFrame):
            passenger_ids = df.get_column('PassengerId').to_list()
        else:
            passenger_ids = df['PassengerId'].tolist()
        
        # Create results
        results = []
        for i, (passenger_id, pred, prob) in enumerate(zip(passenger_ids, predictions, probabilities)):
            confidence = abs(prob - 0.5) * 2
            results.append(PredictionResult(
                passenger_id=int(passenger_id),
                survival_probability=float(prob),
                prediction=int(pred),
                confidence=float(confidence)
            ))
        
        return results
    
    def create_submission_file(
        self, 
        test_df: Union[pd.DataFrame, pl.DataFrame], 
        output_path: Path
    ) -> None:
        """Create Kaggle submission file."""
        predictions = self.predict_batch(test_df)
        
        # Create submission DataFrame
        submission_data = {
            'PassengerId': [pred.passenger_id for pred in predictions],
            'Survived': [pred.prediction for pred in predictions]
        }
        
        if self.use_polars:
            submission_df = pl.DataFrame(submission_data)
            submission_df.write_csv(output_path)
        else:
            submission_df = pd.DataFrame(submission_data)
            submission_df.to_csv(output_path, index=False)
        
        logger.info(f"Submission file created: {output_path}")


class ModelEvaluator:
    """Model evaluation and performance analysis."""
    
    def __init__(self):
        self.results: Dict = {}
        
    def evaluate_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Comprehensive evaluation of predictions."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        self.results['evaluation'] = metrics
        return metrics
    
    def analyze_feature_importance(
        self, 
        trainer: ModelTrainer, 
        feature_names: List[str],
        top_n: int = 10
    ) -> Dict[str, float]:
        """Analyze feature importance."""
        importance_dict = trainer.get_feature_importance(feature_names)
        
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top N features
        top_features = dict(sorted_features[:top_n])
        
        self.results['feature_importance'] = {
            'top_features': top_features,
            'all_features': importance_dict
        }
        
        return top_features
    
    def analyze_prediction_confidence(
        self, 
        predictions: List[PredictionResult]
    ) -> Dict[str, float]:
        """Analyze prediction confidence distribution."""
        confidences = [pred.confidence for pred in predictions]
        probabilities = [pred.survival_probability for pred in predictions]
        
        confidence_stats = {
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
        }
        
        probability_stats = {
            'mean_probability': float(np.mean(probabilities)),
            'median_probability': float(np.median(probabilities)),
            'std_probability': float(np.std(probabilities)),
        }
        
        # Confidence bins
        high_confidence = sum(1 for c in confidences if c > 0.8)
        medium_confidence = sum(1 for c in confidences if 0.5 < c <= 0.8)
        low_confidence = sum(1 for c in confidences if c <= 0.5)
        
        confidence_distribution = {
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence,
            'high_confidence_pct': high_confidence / len(predictions) * 100,
            'medium_confidence_pct': medium_confidence / len(predictions) * 100,
            'low_confidence_pct': low_confidence / len(predictions) * 100,
        }
        
        analysis = {
            **confidence_stats,
            **probability_stats,
            **confidence_distribution
        }
        
        self.results['confidence_analysis'] = analysis
        return analysis
    
    def generate_report(self, output_path: Optional[Path] = None) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': self.results
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report