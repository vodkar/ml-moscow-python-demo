"""Evaluation metrics and model performance utilities."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ClassificationMetrics(BaseModel):
    """Container for classification metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None = None  # Optional in case of single class
    
    def to_dict(self) -> dict[str, float | None]:
        """Convert metrics to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
        }


class ConfusionMatrixMetrics(BaseModel):
    """Container for confusion matrix and derived metrics."""
    
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    @property
    def sensitivity(self) -> float:
        """Calculate sensitivity (recall/true positive rate)."""
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def specificity(self) -> float:
        """Calculate specificity (true negative rate)."""
        return self.true_negatives / (self.true_negatives + self.false_positives)
    
    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def to_dict(self) -> dict[str, int | float]:
        """Convert to dictionary format."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
        }


class ModelPerformanceReport(BaseModel):
    """Comprehensive model performance report."""
    
    model_name: str
    dataset_name: str
    metrics: ClassificationMetrics
    confusion_matrix: ConfusionMatrixMetrics
    classification_report: dict[str, Any]
    feature_importance: dict[str, float] | None = None
    cross_validation_scores: list[float] | None = None
    
    def save_to_json(self, file_path: Path) -> None:
        """Save the performance report to a JSON file."""
        report_data = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "metrics": self.metrics.to_dict(),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "classification_report": self.classification_report,
            "feature_importance": self.feature_importance,
            "cross_validation_scores": self.cross_validation_scores,
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(report_data, file, indent=2, ensure_ascii=False)
        
        logger.info(f"Performance report saved to {file_path}")


class ModelMetrics:
    """Utility class for calculating and managing model evaluation metrics."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray | pl.Series,
        y_pred: np.ndarray | pl.Series,
        y_pred_proba: np.ndarray | pl.Series | None = None,
    ) -> ClassificationMetrics:
        """Calculate comprehensive classification metrics.
        
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC).
            
        Returns:
            ClassificationMetrics object containing all calculated metrics.
        """
        # Convert Polars Series to numpy if needed
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pl.Series):
            y_pred = y_pred.to_numpy()
        if isinstance(y_pred_proba, pl.Series):
            y_pred_proba = y_pred_proba.to_numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except ValueError as exception:
                logger.warning(f"Could not calculate ROC-AUC: {exception}")
        
        logger.info(f"Classification metrics calculated - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
        )
    
    @staticmethod
    def calculate_confusion_matrix_metrics(
        y_true: np.ndarray | pl.Series,
        y_pred: np.ndarray | pl.Series,
    ) -> ConfusionMatrixMetrics:
        """Calculate confusion matrix and derived metrics.
        
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            
        Returns:
            ConfusionMatrixMetrics object.
        """
        # Convert Polars Series to numpy if needed
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pl.Series):
            y_pred = y_pred.to_numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        
        # For binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # For multiclass, we'll use macro-averages
            tp = np.diag(cm).sum()
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            
            # Convert to integers for the model
            tp, fp, fn, tn = int(tp), int(fp.sum()), int(fn.sum()), int(tn)
        
        return ConfusionMatrixMetrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )
    
    @staticmethod
    def generate_classification_report(
        y_true: np.ndarray | pl.Series,
        y_pred: np.ndarray | pl.Series,
        target_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate detailed classification report.
        
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            target_names: Names of target classes.
            
        Returns:
            Dictionary containing the classification report.
        """
        # Convert Polars Series to numpy if needed
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()
        if isinstance(y_pred, pl.Series):
            y_pred = y_pred.to_numpy()
        
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        
        logger.debug("Classification report generated")
        return report
    
    @staticmethod
    def create_performance_report(
        model_name: str,
        dataset_name: str,
        y_true: np.ndarray | pl.Series,
        y_pred: np.ndarray | pl.Series,
        y_pred_proba: np.ndarray | pl.Series | None = None,
        feature_importance: dict[str, float] | None = None,
        cv_scores: list[float] | None = None,
        target_names: list[str] | None = None,
    ) -> ModelPerformanceReport:
        """Create a comprehensive performance report.
        
        Args:
            model_name: Name of the model.
            dataset_name: Name of the dataset.
            y_true: True target values.
            y_pred: Predicted target values.
            y_pred_proba: Predicted probabilities (optional).
            feature_importance: Dictionary of feature importances (optional).
            cv_scores: Cross-validation scores (optional).
            target_names: Names of target classes (optional).
            
        Returns:
            ModelPerformanceReport object.
        """
        metrics = ModelMetrics.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        cm_metrics = ModelMetrics.calculate_confusion_matrix_metrics(y_true, y_pred)
        classification_report = ModelMetrics.generate_classification_report(
            y_true, y_pred, target_names
        )
        
        report = ModelPerformanceReport(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            confusion_matrix=cm_metrics,
            classification_report=classification_report,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
        )
        
        logger.info(f"Performance report created for model '{model_name}' on dataset '{dataset_name}'")
        return report