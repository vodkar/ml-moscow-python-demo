"""Complete ML pipeline orchestrator."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl

from ..core.data_loader import DataLoader
from ..core.models import ModelConfig, PipelineConfig, TrainingMetrics
from ..core.preprocessor import TitanicPreprocessor
from ..core.trainer import ModelTrainer, EnsembleTrainer
from .predictor import ModelEvaluator, TitanicPredictor

logger = logging.getLogger(__name__)


class TitanicMLPipeline:
    """Complete ML pipeline for Titanic survival prediction."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(use_polars=config.use_polars)
        self.preprocessor = TitanicPreprocessor(use_polars=config.use_polars)
        self.trainer: Optional[ModelTrainer] = None
        self.evaluator = ModelEvaluator()
        
        # Create output directories
        self.config.model_output_path.mkdir(parents=True, exist_ok=True)
        
    def run_full_pipeline(
        self, 
        model_configs: list[ModelConfig],
        save_models: bool = True,
        create_submission: bool = True
    ) -> Dict:
        """Run the complete ML pipeline."""
        logger.info("Starting full ML pipeline...")
        
        # Step 1: Load and analyze data
        train_df, test_df, submission_df = self._load_data()
        dataset_info = self.data_loader.get_dataset_info(train_df)
        
        logger.info(f"Dataset loaded: {dataset_info.total_records} training records")
        logger.info(f"Missing values: {dataset_info.missing_values}")
        
        # Step 2: Preprocess data
        x_train, y_train = self.preprocessor.fit_transform(train_df)
        x_test = self.preprocessor.transform(test_df)
        
        logger.info(f"Data preprocessed: {x_train.shape[1]} features")
        
        # Step 3: Train models
        results = {}
        
        if len(model_configs) == 1:
            # Single model training
            self.trainer = ModelTrainer(model_configs[0])
            model, metrics = self.trainer.train(
                x_train, y_train, 
                self.preprocessor.get_feature_importance_names()
            )
            results['single_model'] = {
                'model_type': model_configs[0].model_type,
                'metrics': metrics.model_dump(),
                'feature_importance': self.trainer.get_feature_importance(
                    self.preprocessor.get_feature_importance_names()
                )
            }
        else:
            # Ensemble training
            ensemble = EnsembleTrainer(model_configs)
            metrics_list = ensemble.train(
                x_train, y_train,
                self.preprocessor.get_feature_importance_names()
            )
            
            results['ensemble'] = {
                'models': [
                    {
                        'model_type': config.model_type,
                        'metrics': metrics.model_dump()
                    }
                    for config, metrics in zip(model_configs, metrics_list)
                ],
                'ensemble_weights': ensemble.weights
            }
            
            # Use ensemble for predictions
            self.trainer = ensemble
        
        # Step 4: Evaluate model
        if hasattr(self.trainer, 'predict'):
            # Create validation split for evaluation
            from sklearn.model_selection import train_test_split
            x_tr, x_val, y_tr, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            val_predictions = self.trainer.predict(x_val)
            val_probabilities = self.trainer.predict_proba(x_val)
            
            evaluation_metrics = self.evaluator.evaluate_predictions(
                y_val, val_predictions, val_probabilities
            )
            results['evaluation'] = evaluation_metrics
        
        # Step 5: Save models and preprocessor
        if save_models:
            self._save_pipeline()
        
        # Step 6: Create predictions and submission
        if create_submission:
            predictor = TitanicPredictor(use_polars=self.config.use_polars)
            predictor.preprocessor = self.preprocessor
            predictor.trainer = self.trainer
            
            submission_path = self.config.model_output_path / "submission.csv"
            predictor.create_submission_file(test_df, submission_path)
            results['submission_path'] = str(submission_path)
        
        # Step 7: Generate comprehensive report
        report_path = self.config.model_output_path / "pipeline_report.json"
        results['pipeline_config'] = self.config.model_dump()
        results['dataset_info'] = dataset_info.model_dump()
        
        final_report = self.evaluator.generate_report(report_path)
        final_report.update(results)
        
        logger.info("Pipeline completed successfully!")
        return final_report
    
    def _load_data(self) -> Tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame, Optional[pd.DataFrame | pl.DataFrame]]:
        """Load training and test data."""
        return self.data_loader.load_titanic_data(self.config.data_path)
    
    def _save_pipeline(self) -> None:
        """Save trained models and preprocessor."""
        # Save main model
        if isinstance(self.trainer, ModelTrainer):
            model_path = self.config.model_output_path / "model.joblib"
            self.trainer.save_model(model_path)
        elif isinstance(self.trainer, EnsembleTrainer):
            # Save ensemble models
            for i, trainer in enumerate(self.trainer.trainers):
                model_path = self.config.model_output_path / f"ensemble_model_{i}.joblib"
                trainer.save_model(model_path)
            
            # Save ensemble metadata
            ensemble_path = self.config.model_output_path / "ensemble_metadata.joblib"
            ensemble_data = {
                'weights': self.trainer.weights,
                'model_configs': [trainer.config for trainer in self.trainer.trainers],
                'n_models': len(self.trainer.trainers)
            }
            joblib.dump(ensemble_data, ensemble_path)
        
        # Save preprocessor
        preprocessor_path = self.config.model_output_path / "preprocessor.joblib"
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.preprocessor.get_feature_importance_names()
        }
        joblib.dump(preprocessor_data, preprocessor_path)
        
        logger.info(f"Pipeline saved to {self.config.model_output_path}")
    
    def load_trained_pipeline(self, model_path: Path) -> None:
        """Load previously trained pipeline."""
        model_path = Path(model_path)
        
        # Check if it's an ensemble
        ensemble_metadata_path = model_path / "ensemble_metadata.joblib"
        if ensemble_metadata_path.exists():
            # Load ensemble
            ensemble_data = joblib.load(ensemble_metadata_path)
            
            trainers = []
            for i in range(ensemble_data['n_models']):
                trainer_path = model_path / f"ensemble_model_{i}.joblib"
                trainer = ModelTrainer.load_model(trainer_path)
                trainers.append(trainer)
            
            self.trainer = EnsembleTrainer(ensemble_data['model_configs'])
            self.trainer.trainers = trainers
            self.trainer.weights = ensemble_data['weights']
        else:
            # Load single model
            single_model_path = model_path / "model.joblib"
            self.trainer = ModelTrainer.load_model(single_model_path)
        
        # Load preprocessor
        preprocessor_path = model_path / "preprocessor.joblib"
        if preprocessor_path.exists():
            preprocessor_data = joblib.load(preprocessor_path)
            self.preprocessor = preprocessor_data['preprocessor']
        
        logger.info(f"Pipeline loaded from {model_path}")
    
    def predict_from_file(self, input_path: Path, output_path: Path) -> None:
        """Make predictions from CSV file."""
        if not self.trainer or not self.preprocessor:
            raise ValueError("Pipeline not loaded. Call load_trained_pipeline first.")
        
        # Load data
        test_df = self.data_loader.load_csv(input_path, validate_schema=False)
        
        # Create predictor and make predictions
        predictor = TitanicPredictor(use_polars=self.config.use_polars)
        predictor.preprocessor = self.preprocessor
        predictor.trainer = self.trainer
        
        # Create submission file
        predictor.create_submission_file(test_df, output_path)
    
    def get_model_summary(self) -> Dict:
        """Get summary of trained model(s)."""
        if not self.trainer:
            return {"error": "No model trained"}
        
        summary = {}
        
        if isinstance(self.trainer, ModelTrainer):
            summary = {
                'type': 'single_model',
                'model_type': self.trainer.config.model_type,
                'parameters': self.trainer.best_params,
                'metrics': self.trainer.training_metrics.model_dump() if self.trainer.training_metrics else None
            }
        elif isinstance(self.trainer, EnsembleTrainer):
            summary = {
                'type': 'ensemble',
                'n_models': len(self.trainer.trainers),
                'model_types': [trainer.config.model_type for trainer in self.trainer.trainers],
                'weights': self.trainer.weights,
                'individual_metrics': [
                    trainer.training_metrics.model_dump() if trainer.training_metrics else None
                    for trainer in self.trainer.trainers
                ]
            }
        
        return summary