"""Machine learning model training and evaluation module."""

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from titanic_ml.config import ModelConfig
from titanic_ml.utils.metrics import ModelMetrics, ModelPerformanceReport

logger = logging.getLogger(__name__)

ModelType = Literal["random_forest", "xgboost", "logistic_regression"]


class TitanicModel:
    """Titanic survival prediction model with multiple algorithm support."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model with configuration.
        
        Args:
            config: Model configuration object.
        """
        self.config = config
        self.model: RandomForestClassifier | XGBClassifier | LogisticRegression | None = None
        self.feature_names_: list[str] = []
        self.feature_importance_: dict[str, float] | None = None
        self.is_fitted_ = False
        
        logger.info(f"TitanicModel initialized with model_type='{config.model_type}'")
    
    def _create_model(self) -> RandomForestClassifier | XGBClassifier | LogisticRegression:
        """Create the model instance based on configuration.
        
        Returns:
            Configured model instance.
        """
        if self.config.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                random_state=self.config.random_state,
                eval_metric="logloss",  # Suppress warnings
                n_jobs=-1,  # Use all available cores
            )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif self.config.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=self.config.lr_max_iter,
                C=self.config.lr_c,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def fit(
        self, 
        training_features: pl.DataFrame | np.ndarray, 
        training_targets: pl.Series | np.ndarray
    ) -> "TitanicModel":
        """Fit the model on training data.
        
        Args:
            training_features: Training feature matrix.
            training_targets: Training target vector.
            
        Returns:
            Fitted model instance.
        """
        logger.info(f"Training {self.config.model_type} model")
        
        # Convert Polars to numpy if needed
        if isinstance(training_features, pl.DataFrame):
            self.feature_names_ = training_features.columns
            features_array = training_features.to_numpy()
        else:
            features_array = training_features
        
        if isinstance(training_targets, pl.Series):
            targets_array = training_targets.to_numpy()
        else:
            targets_array = training_targets
        
        # Create and fit the model
        self.model = self._create_model()
        self.model.fit(features_array, targets_array)
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        self.is_fitted_ = True
        logger.info(f"Model training completed. Features: {len(self.feature_names_)}")
        
        return self
    
    def predict(self, features: pl.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            features: Feature matrix for prediction.
            
        Returns:
            Predicted class labels.
        """
        if not self.is_fitted_ or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert Polars to numpy if needed
        if isinstance(features, pl.DataFrame):
            features_array = features.to_numpy()
        else:
            features_array = features
        
        predictions = self.model.predict(features_array)
        logger.debug(f"Predictions made for {len(predictions)} samples")
        
        return predictions
    
    def predict_proba(self, features: pl.DataFrame | np.ndarray) -> np.ndarray:
        """Make probability predictions on new data.
        
        Args:
            features: Feature matrix for prediction.
            
        Returns:
            Predicted class probabilities.
        """
        if not self.is_fitted_ or self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert Polars to numpy if needed
        if isinstance(features, pl.DataFrame):
            features_array = features.to_numpy()
        else:
            features_array = features
        
        probabilities = self.model.predict_proba(features_array)
        logger.debug(f"Probability predictions made for {len(probabilities)} samples")
        
        return probabilities
    
    def evaluate(
        self, 
        test_features: pl.DataFrame | np.ndarray, 
        test_targets: pl.Series | np.ndarray,
        dataset_name: str = "test",
    ) -> ModelPerformanceReport:
        """Evaluate the model on test data.
        
        Args:
            test_features: Test feature matrix.
            test_targets: Test target vector.
            dataset_name: Name of the dataset for reporting.
            
        Returns:
            Comprehensive performance report.
        """
        if not self.is_fitted_ or self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info(f"Evaluating model on {dataset_name} dataset")
        
        # Make predictions
        predictions = self.predict(test_features)
        probabilities = None
        
        try:
            prob_scores = self.predict_proba(test_features)
            # For binary classification, use positive class probabilities
            probabilities = prob_scores[:, 1] if prob_scores.shape[1] == 2 else None
        except Exception as exception:
            logger.warning(f"Could not get prediction probabilities: {exception}")
        
        # Convert targets to numpy if needed
        if isinstance(test_targets, pl.Series):
            targets_array = test_targets.to_numpy()
        else:
            targets_array = test_targets
        
        # Create performance report
        performance_report = ModelMetrics.create_performance_report(
            model_name=self.config.model_type,
            dataset_name=dataset_name,
            y_true=targets_array,
            y_pred=predictions,
            y_pred_proba=probabilities,
            feature_importance=self.feature_importance_,
            target_names=["Not Survived", "Survived"],
        )
        
        logger.info(f"Model evaluation completed. Accuracy: {performance_report.metrics.accuracy:.4f}")
        return performance_report
    
    def cross_validate(
        self, 
        features: pl.DataFrame | np.ndarray, 
        targets: pl.Series | np.ndarray,
        cv_folds: int = 5,
        scoring: str = "accuracy",
    ) -> tuple[float, float, list[float]]:
        """Perform cross-validation on the dataset.
        
        Args:
            features: Feature matrix.
            targets: Target vector.
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric for cross-validation.
            
        Returns:
            Tuple of (mean_score, std_score, all_scores).
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Convert Polars to numpy if needed
        if isinstance(features, pl.DataFrame):
            features_array = features.to_numpy()
        else:
            features_array = features
        
        if isinstance(targets, pl.Series):
            targets_array = targets.to_numpy()
        else:
            targets_array = targets
        
        # Create model for cross-validation
        model = self._create_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, features_array, targets_array, cv=cv_folds, scoring=scoring, n_jobs=-1
        )
        
        mean_score = float(np.mean(cv_scores))
        std_score = float(np.std(cv_scores))
        
        logger.info(f"Cross-validation completed. {scoring}: {mean_score:.4f} (+/- {std_score:.4f})")
        return mean_score, std_score, cv_scores.tolist()
    
    def train_with_validation(
        self,
        features: pl.DataFrame | np.ndarray,
        targets: pl.Series | np.ndarray,
        validation_split: float = 0.2,
    ) -> tuple["TitanicModel", ModelPerformanceReport]:
        """Train the model with automatic train/validation split.
        
        Args:
            features: Full feature matrix.
            targets: Full target vector.
            validation_split: Fraction of data to use for validation.
            
        Returns:
            Tuple of (fitted_model, validation_performance_report).
        """
        logger.info(f"Training model with validation split: {validation_split}")
        
        # Convert to numpy arrays
        if isinstance(features, pl.DataFrame):
            self.feature_names_ = features.columns
            features_array = features.to_numpy()
        else:
            features_array = features
        
        if isinstance(targets, pl.Series):
            targets_array = targets.to_numpy()
        else:
            targets_array = targets
        
        # Split data
        train_features, val_features, train_targets, val_targets = train_test_split(
            features_array,
            targets_array,
            test_size=validation_split,
            random_state=self.config.random_state,
            stratify=targets_array,  # Maintain class distribution
        )
        
        logger.info(f"Split data - Train: {len(train_features)}, Validation: {len(val_features)}")
        
        # Train model
        self.model = self._create_model()
        self.model.fit(train_features, train_targets)
        self._extract_feature_importance()
        self.is_fitted_ = True
        
        # Evaluate on validation set
        val_performance = self.evaluate(val_features, val_targets, "validation")
        
        return self, val_performance
    
    def save(self, file_path: Path) -> None:
        """Save the fitted model to disk.
        
        Args:
            file_path: Path to save the model.
        """
        if not self.is_fitted_ or self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the entire TitanicModel instance
        model_data = {
            "model": self.model,
            "config": self.config,
            "feature_names": self.feature_names_,
            "feature_importance": self.feature_importance_,
            "is_fitted": self.is_fitted_,
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path) -> "TitanicModel":
        """Load a fitted model from disk.
        
        Args:
            file_path: Path to the saved model.
            
        Returns:
            Loaded model instance.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model_data = joblib.load(file_path)
        
        # Create new instance
        instance = cls(model_data["config"])
        instance.model = model_data["model"]
        instance.feature_names_ = model_data["feature_names"]
        instance.feature_importance_ = model_data["feature_importance"]
        instance.is_fitted_ = model_data["is_fitted"]
        
        logger.info(f"Model loaded from {file_path}")
        return instance
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from the fitted model."""
        if self.model is None:
            return
        
        # Get feature importance based on model type
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if len(self.feature_names_) == len(importances):
                self.feature_importance_ = dict(zip(self.feature_names_, importances))
                logger.debug("Feature importance extracted")
            else:
                logger.warning("Mismatch between feature names and importances")
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficients
            coefficients = np.abs(self.model.coef_[0])  # Assuming binary classification
            if len(self.feature_names_) == len(coefficients):
                self.feature_importance_ = dict(zip(self.feature_names_, coefficients))
                logger.debug("Feature coefficients extracted")
        else:
            logger.warning("Model does not provide feature importance")
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None if not available.
        """
        return self.feature_importance_.copy() if self.feature_importance_ else None
    
    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive information about the model.
        
        Returns:
            Dictionary containing model metadata.
        """
        return {
            "model_type": self.config.model_type,
            "is_fitted": self.is_fitted_,
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_.copy(),
            "has_feature_importance": self.feature_importance_ is not None,
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else vars(self.config),
        }