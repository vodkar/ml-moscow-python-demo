"""Prediction and evaluation system."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)


class PredictionConfig(BaseModel):
    """Configuration for predictions."""

    model_path: Path = Field(default=Path("artifacts/model.joblib"))
    preprocessor_path: Path = Field(default=Path("artifacts/preprocessor.joblib"))
    output_path: Path = Field(default=Path("artifacts/predictions.csv"))
    probability_threshold: float = Field(default=0.5)


class ModelPredictor:
    """Model predictor with evaluation capabilities."""

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None

    def load_artifacts(self) -> None:
        """Load trained model and preprocessor."""
        import joblib

        from .models import ModelConfig

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        if not self.config.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {self.config.preprocessor_path}"
            )

        logger.info(f"Loading model from {self.config.model_path}")
        self.model = joblib.load(self.config.model_path)

        logger.info(f"Loading preprocessor from {self.config.preprocessor_path}")
        self.preprocessor = joblib.load(self.config.preprocessor_path)

        logger.info("Model and preprocessor loaded successfully")

    def predict(self, X: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
        if self.model is None or self.preprocessor is None:
            self.load_artifacts()

        # Keep passenger IDs for output
        passenger_ids = X.select("PassengerId").to_numpy().flatten()

        # Preprocess data
        X_processed = self.preprocessor.transform(X)
        X_np = X_processed.select(
            [col for col in X_processed.columns if col != "PassengerId"]
        ).to_numpy()

        # Make predictions
        predictions = self.model.predict(X_np)
        probabilities = self.model.predict_proba(X_np)[:, 1]

        logger.info(f"Generated predictions for {len(predictions)} samples")

        return predictions, probabilities

    def predict_and_save(self, X: pl.DataFrame) -> pl.DataFrame:
        """Make predictions and save to CSV."""
        predictions, probabilities = self.predict(X)

        # Create results DataFrame
        results = pl.DataFrame(
            {
                "PassengerId": X.select("PassengerId").to_series(),
                "Survived": predictions,
                "Probability": probabilities,
            }
        )

        # Save predictions
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        results.write_csv(self.config.output_path)

        logger.info(f"Predictions saved to {self.config.output_path}")

        return results

    def evaluate_model(self, X: pl.DataFrame, y_true: pl.Series) -> Dict[str, Any]:
        """Evaluate model performance on labeled data."""
        predictions, probabilities = self.predict(X)
        y_true_np = y_true.to_numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_true_np, predictions)
        roc_auc = roc_auc_score(y_true_np, probabilities)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_np, predictions, average="binary"
        )

        # Confusion matrix
        cm = confusion_matrix(y_true_np, predictions)

        # Classification report
        class_report = classification_report(
            y_true_np,
            predictions,
            target_names=["Not Survived", "Survived"],
            output_dict=True,
        )

        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "n_samples": len(y_true_np),
            "n_positive": int(y_true_np.sum()),
            "n_negative": int(len(y_true_np) - y_true_np.sum()),
        }

        logger.info(f"Model Evaluation Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")

        return metrics

    def get_prediction_summary(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Get summary statistics of predictions."""
        return {
            "total_predictions": len(predictions),
            "survived_predictions": int(predictions.sum()),
            "not_survived_predictions": int(len(predictions) - predictions.sum()),
            "survival_rate": float(predictions.mean()),
        }
