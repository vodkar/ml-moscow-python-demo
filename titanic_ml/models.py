"""Model training and optimization pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import optuna
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score


class ModelConfig(BaseModel):
    """Configuration for model training."""

    model_type: str = Field(
        default="xgboost", description="Model type: xgboost or random_forest"
    )
    cv_folds: int = Field(default=5)
    optimization_trials: int = Field(default=100)
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2)
    model_save_path: Path = Field(default=Path("artifacts/model.joblib"))
    preprocessor_save_path: Path = Field(default=Path("artifacts/preprocessor.joblib"))


class ModelTrainer:
    """Model trainer with hyperparameter optimization."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model: Optional[Any] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Optional[np.ndarray] = None

        # Ensure artifacts directory exists
        self.config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.preprocessor_save_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_model(self, params: Dict[str, Any]) -> Any:
        """Get model instance with specified parameters."""
        if self.config.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    random_state=self.config.random_state,
                    eval_metric="logloss",
                    **params,
                )
            except ImportError:
                raise ImportError(
                    "XGBoost is not properly installed. Please install libomp: brew install libomp"
                )
        elif self.config.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=self.config.random_state, **params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space."""
        if self.config.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
        elif self.config.model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
            }

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function."""
        params = self._get_param_space(trial)
        model = self._get_model(params)

        # Use stratified k-fold cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        return scores.mean()

    def optimize_hyperparameters(self, X: pl.DataFrame, y: pl.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        logger.info(
            f"Starting hyperparameter optimization with {self.config.optimization_trials} trials"
        )

        # Convert to numpy for sklearn compatibility
        X_np = X.select([col for col in X.columns if col != "PassengerId"]).to_numpy()
        y_np = y.to_numpy()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X_np, y_np),
            n_trials=self.config.optimization_trials,
            show_progress_bar=True,
        )

        logger.info(f"Best trial score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        self.best_params = study.best_params
        return self.best_params

    def train_model(
        self, X: pl.DataFrame, y: pl.Series, optimize: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """Train model with optional hyperparameter optimization."""
        logger.info(f"Training {self.config.model_type} model")

        # Convert to numpy for sklearn compatibility
        X_np = X.select([col for col in X.columns if col != "PassengerId"]).to_numpy()
        y_np = y.to_numpy()

        if optimize:
            self.optimize_hyperparameters(X, y)
            model = self._get_model(self.best_params)
        else:
            # Use default parameters
            default_params = {}
            if self.config.model_type == "xgboost":
                default_params = {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                }
            elif self.config.model_type == "random_forest":
                default_params = {"n_estimators": 100, "max_depth": 10}

            model = self._get_model(default_params)

        # Train the model
        model.fit(X_np, y_np)
        self.model = model

        # Evaluate using cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        cv_scores = cross_val_score(
            model, X_np, y_np, cv=cv, scoring="accuracy", n_jobs=-1
        )
        self.cv_scores = cv_scores

        metrics = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }

        logger.info(
            f"Cross-validation accuracy: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
        )

        return model, metrics

    def save_model(self, preprocessor: Any) -> None:
        """Save trained model and preprocessor."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        logger.info(f"Saving model to {self.config.model_save_path}")
        joblib.dump(self.model, self.config.model_save_path)

        logger.info(f"Saving preprocessor to {self.config.preprocessor_save_path}")
        joblib.dump(preprocessor, self.config.preprocessor_save_path)

    def load_model(self) -> Tuple[Any, Any]:
        """Load saved model and preprocessor."""
        logger.info(f"Loading model from {self.config.model_save_path}")
        model = joblib.load(self.config.model_save_path)

        logger.info(f"Loading preprocessor from {self.config.preprocessor_save_path}")
        preprocessor = joblib.load(self.config.preprocessor_save_path)

        self.model = model
        return model, preprocessor

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("No model available. Train a model first.")

        if hasattr(self.model, "feature_importances_"):
            return dict(
                zip(
                    [
                        f"feature_{i}"
                        for i in range(len(self.model.feature_importances_))
                    ],
                    self.model.feature_importances_,
                )
            )
        else:
            return {}
