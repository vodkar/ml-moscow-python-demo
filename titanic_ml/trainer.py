"""Model training module with multiple algorithms and hyperparameter optimization."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Final

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Model type constants
MODEL_TYPES: Final[list[str]] = [
    "logistic_regression",
    "random_forest", 
    "xgboost",
    "lightgbm",
    "svm"
]


class TrainerConfig(BaseModel):
    """Configuration for model trainer."""
    
    model_type: str = Field(default="xgboost", description="Type of model to train")
    test_size: float = Field(default=0.2, description="Proportion of data for testing")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    hyperparameter_optimization: bool = Field(default=True, description="Enable hyperparameter optimization")
    optimization_trials: int = Field(default=100, description="Number of optimization trials")
    model_save_path: Path = Field(default=Path("models"), description="Path to save trained models")
    
    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Model type must be one of {MODEL_TYPES}")
        
        if not 0.1 <= self.test_size <= 0.5:
            raise ValueError("Test size must be between 0.1 and 0.5")
        
        if self.cv_folds < 2:
            raise ValueError("CV folds must be at least 2")


class ModelPerformance(BaseModel):
    """Model performance metrics."""
    
    accuracy: float = Field(description="Accuracy score")
    roc_auc: float = Field(description="ROC AUC score") 
    cv_mean: float = Field(description="Cross-validation mean score")
    cv_std: float = Field(description="Cross-validation standard deviation")
    confusion_matrix: list[list[int]] = Field(description="Confusion matrix")
    classification_report: dict[str, Any] = Field(description="Classification report")


class TitanicTrainer:
    """ML model trainer for Titanic survival prediction.
    
    Supports multiple algorithms with hyperparameter optimization using Optuna.
    Provides comprehensive model evaluation and saving capabilities.
    """
    
    def __init__(self, config: TrainerConfig) -> None:
        """Initialize trainer with configuration.
        
        Args:
            config: TrainerConfig instance with training parameters
        """
        self.config = config
        self.model: Any = None
        self.feature_names: list[str] = []
        self.performance: ModelPerformance | None = None
        self.best_params: dict[str, Any] = {}
        
        # Create model save directory
        self.config.model_save_path.mkdir(parents=True, exist_ok=True)
    
    def _create_base_model(self) -> Any:
        """Create base model instance without hyperparameters.
        
        Returns:
            Base model instance
        """
        if self.config.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "xgboost":
            return xgb.XGBClassifier(
                random_state=self.config.random_state,
                eval_metric="logloss",
                verbosity=0
            )
        elif self.config.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                random_state=self.config.random_state,
                verbosity=-1,
                force_row_wise=True
            )
        elif self.config.model_type == "svm":
            return SVC(
                random_state=self.config.random_state,
                probability=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _get_hyperparameter_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter search space for each model type.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters for the trial
        """
        if self.config.model_type == "logistic_regression":
            return {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": trial.suggest_categorical("solver", ["liblinear", "saga"])
            }
        elif self.config.model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            }
        elif self.config.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10)
            }
        elif self.config.model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10)
            }
        elif self.config.model_type == "svm":
            return {
                "C": trial.suggest_float("C", 0.1, 100, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            }
        else:
            return {}
    
    def _objective(self, trial: optuna.Trial, features: np.ndarray, target: np.ndarray) -> float:
        """Objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            features: Feature matrix
            target: Target vector
            
        Returns:
            Cross-validation score to maximize
        """
        # Get hyperparameters for this trial
        params = self._get_hyperparameter_space(trial)
        
        # Create model with trial parameters
        model = self._create_base_model()
        model.set_params(**params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model,
            features,
            target,
            cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state),
            scoring="accuracy",
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def _optimize_hyperparameters(self, features: np.ndarray, target: np.ndarray) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna.
        
        Args:
            features: Feature matrix
            target: Target vector
            
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Optimizing hyperparameters for {self.config.model_type}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, features, target),
            n_trials=self.config.optimization_trials,
            show_progress_bar=False
        )
        
        logger.info(f"Best CV score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params
    
    def _prepare_features_and_target(self, dataframe: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare features and target arrays from DataFrame.
        
        Args:
            dataframe: Input DataFrame with features and target
            
        Returns:
            Tuple of (features, target) as numpy arrays
        """
        # Separate features and target
        if "Survived" not in dataframe.columns:
            raise ValueError("Training data must contain 'Survived' column")
        
        # Get feature columns (excluding target and ID)
        feature_columns = [col for col in dataframe.columns if col not in ["Survived", "PassengerId"]]
        self.feature_names = feature_columns
        
        # Convert to numpy arrays
        features = dataframe.select(feature_columns).to_numpy()
        target = dataframe.select("Survived").to_numpy().ravel()
        
        logger.info(f"Prepared features: {features.shape}, target: {target.shape}")
        logger.info(f"Feature names: {self.feature_names}")
        
        return features, target
    
    def _evaluate_model(self, model: Any, features: np.ndarray, target: np.ndarray) -> ModelPerformance:
        """Evaluate model performance using multiple metrics.
        
        Args:
            model: Trained model
            features: Feature matrix
            target: Target vector
            
        Returns:
            ModelPerformance object with evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Predictions
        predictions = model.predict(features)
        prediction_probabilities = model.predict_proba(features)[:, 1]
        
        # Basic metrics
        accuracy = accuracy_score(target, predictions)
        roc_auc = roc_auc_score(target, prediction_probabilities)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model,
            features,
            target,
            cv=StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state),
            scoring="accuracy",
            n_jobs=-1
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(target, predictions).tolist()
        
        # Classification report
        class_report = classification_report(target, predictions, output_dict=True)
        
        performance = ModelPerformance(
            accuracy=accuracy,
            roc_auc=roc_auc,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return performance
    
    def train(self, train_data: pl.DataFrame) -> None:
        """Train the model on provided data.
        
        Args:
            train_data: Training DataFrame with features and target
        """
        logger.info(f"Training {self.config.model_type} model")
        
        # Prepare data
        features, target = self._prepare_features_and_target(train_data)
        
        # Optimize hyperparameters if enabled
        if self.config.hyperparameter_optimization:
            self.best_params = self._optimize_hyperparameters(features, target)
        else:
            self.best_params = {}
        
        # Create and train final model
        self.model = self._create_base_model()
        if self.best_params:
            self.model.set_params(**self.best_params)
        
        logger.info("Training final model")
        self.model.fit(features, target)
        
        # Evaluate performance
        self.performance = self._evaluate_model(self.model, features, target)
        
        logger.info("Training completed successfully")
    
    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        """Make predictions on test data.
        
        Args:
            test_data: Test DataFrame with features
            
        Returns:
            Prediction array
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making predictions on test data")
        
        # Prepare features (ensure same columns as training)
        missing_features = set(self.feature_names) - set(test_data.columns)
        if missing_features:
            raise ValueError(f"Test data missing features: {missing_features}")
        
        # Select and order features consistently with training
        features = test_data.select(self.feature_names).to_numpy()
        
        # Make predictions
        predictions = self.model.predict(features)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_proba(self, test_data: pl.DataFrame) -> np.ndarray:
        """Make probability predictions on test data.
        
        Args:
            test_data: Test DataFrame with features
            
        Returns:
            Prediction probability array
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making probability predictions on test data")
        
        # Prepare features
        features = test_data.select(self.feature_names).to_numpy()
        
        # Make probability predictions
        probabilities = self.model.predict_proba(features)
        
        return probabilities
    
    def save_model(self, filename: str | None = None) -> Path:
        """Save trained model to disk.
        
        Args:
            filename: Optional filename for saved model
            
        Returns:
            Path to saved model file
            
        Raises:
            ValueError: If model hasn't been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        if filename is None:
            filename = f"titanic_{self.config.model_type}_model.joblib"
        
        model_path = self.config.model_save_path / filename
        
        # Save model and metadata
        model_data = {
            "model": self.model,
            "config": self.config.model_dump(),
            "feature_names": self.feature_names,
            "performance": self.performance.model_dump() if self.performance else None,
            "best_params": self.best_params
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: Path | str) -> None:
        """Load trained model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.best_params = model_data.get("best_params", {})
        
        if model_data.get("performance"):
            self.performance = ModelPerformance(**model_data["performance"])
        
        logger.info("Model loaded successfully")
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores, or None if not available
        """
        if self.model is None:
            return None
        
        importance_dict = None
        
        if hasattr(self.model, "feature_importances_"):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficient values
            importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        
        if importance_dict:
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict


def create_trainer(
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    hyperparameter_optimization: bool = True,
    optimization_trials: int = 100,
    model_save_path: str | Path = "models"
) -> TitanicTrainer:
    """Create a configured trainer instance.
    
    Args:
        model_type: Type of model to train
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        cv_folds: Number of cross-validation folds
        hyperparameter_optimization: Enable hyperparameter optimization
        optimization_trials: Number of optimization trials
        model_save_path: Path to save trained models
        
    Returns:
        Configured TitanicTrainer instance
    """
    config = TrainerConfig(
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
        cv_folds=cv_folds,
        hyperparameter_optimization=hyperparameter_optimization,
        optimization_trials=optimization_trials,
        model_save_path=Path(model_save_path)
    )
    return TitanicTrainer(config)