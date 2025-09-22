"""Ensemble models for Titanic survival prediction."""

from typing import Any, Final

import joblib
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


class ModelConfig(BaseModel):
    """Configuration for ensemble models."""

    use_random_forest: bool = Field(default=True)
    use_xgboost: bool = Field(default=True)
    use_logistic_regression: bool = Field(default=True)
    cv_folds: int = Field(default=5, ge=3, le=10)
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)


class TitanicEnsemble:
    """Ensemble model for Titanic survival prediction."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize ensemble model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.models: dict[str, Any] = {}
        self.ensemble_model: VotingClassifier | None = None
        self.fitted_: bool = False

    def _create_base_models(self) -> dict[str, Any]:
        """Create base models for ensemble.

        Returns:
            Dictionary of base models
        """
        base_models = {}

        if self.config.use_random_forest:
            base_models["rf"] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

        if self.config.use_xgboost:
            base_models["xgb"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                eval_metric="logloss",
            )

        if self.config.use_logistic_regression:
            base_models["lr"] = LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )

        return base_models

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TitanicEnsemble":
        """Fit the ensemble model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Fitted ensemble model
        """
        self.models = self._create_base_models()

        estimators = [(name, model) for name, model in self.models.items()]

        self.ensemble_model = VotingClassifier(
            estimators=estimators, voting="soft", n_jobs=self.config.n_jobs
        )

        self.ensemble_model.fit(X_train, y_train)
        self.fitted_ = True

        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions using the ensemble model.

        Args:
            X_test: Test features

        Returns:
            Predictions

        Raises:
            ValueError: If the model hasn't been fitted
        """
        if not self.fitted_ or self.ensemble_model is None:
            raise ValueError("Ensemble model must be fitted before making predictions")

        predictions = self.ensemble_model.predict(X_test)
        return pd.Series(predictions, index=X_test.index)

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities using the ensemble model.

        Args:
            X_test: Test features

        Returns:
            Class probabilities

        Raises:
            ValueError: If the model hasn't been fitted
        """
        if not self.fitted_ or self.ensemble_model is None:
            raise ValueError("Ensemble model must be fitted before making predictions")

        probabilities = self.ensemble_model.predict_proba(X_test)
        return pd.DataFrame(
            probabilities,
            index=X_test.index,
            columns=[f"class_{i}" for i in range(probabilities.shape[1])],
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """Evaluate the ensemble model performance.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            Evaluation metrics

        Raises:
            ValueError: If the model hasn't been fitted
        """
        if not self.fitted_:
            raise ValueError("Ensemble model must be fitted before evaluation")

        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        report = classification_report(y_test, predictions, output_dict=True)

        return {
            "accuracy": accuracy,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "classification_report": report,
        }

    def cross_validate(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, float]:
        """Perform cross-validation on the ensemble model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Cross-validation scores

        Raises:
            ValueError: If the model hasn't been fitted
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model must be created before cross-validation")

        cv_scores = cross_val_score(
            self.ensemble_model,
            X_train,
            y_train,
            cv=self.config.cv_folds,
            scoring="accuracy",
        )

        return {
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
            "individual_scores": cv_scores.tolist(),
        }

    def save_model(self, filepath: str) -> None:
        """Save the trained ensemble model to disk.

        Args:
            filepath: Path where to save the model

        Raises:
            ValueError: If the model hasn't been fitted
        """
        if not self.fitted_ or self.ensemble_model is None:
            raise ValueError("Ensemble model must be fitted before saving")

        model_data = {
            "ensemble_model": self.ensemble_model,
            "config": self.config.model_dump(),
            "fitted_": self.fitted_,
        }

        joblib.dump(model_data, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "TitanicEnsemble":
        """Load a trained ensemble model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded ensemble model
        """
        model_data = joblib.load(filepath)

        config = ModelConfig(**model_data["config"])
        ensemble = cls(config)
        ensemble.ensemble_model = model_data["ensemble_model"]
        ensemble.fitted_ = model_data["fitted_"]

        return ensemble


# Default model configuration
DEFAULT_MODEL_CONFIG: Final[ModelConfig] = ModelConfig()
