"""Training pipeline for Titanic ML models."""

from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from titanic_ml.data.loader import DataLoader
from titanic_ml.models.ensemble import TitanicEnsemble
from titanic_ml.preprocessing.features import TitanicFeatureEngineer


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    validation_size: float = Field(default=0.1, ge=0.05, le=0.3)
    random_state: int = Field(default=42)
    target_column: str = Field(default="Survived")
    save_model_path: str = Field(default="models/titanic_ensemble.joblib")
    save_preprocessor_path: str = Field(default="models/feature_engineer.joblib")


class ModelTrainer:
    """Model training pipeline for Titanic dataset."""

    def __init__(
        self,
        data_loader: DataLoader,
        feature_engineer: TitanicFeatureEngineer,
        model: TitanicEnsemble,
        config: TrainingConfig,
    ) -> None:
        """Initialize model trainer.

        Args:
            data_loader: Data loader instance
            feature_engineer: Feature engineering transformer
            model: Ensemble model for training
            config: Training configuration
        """
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.model = model
        self.config = config
        self.training_metrics: dict[str, Any] = {}

    def _prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Prepare features by selecting relevant columns.

        Args:
            dataframe: Input dataframe

        Returns:
            DataFrame with selected features
        """
        feature_columns = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "FamilySize",
            "IsAlone",
            "Title",
            "AgeBin",
            "FareBin",
        ]

        available_columns = [col for col in feature_columns if col in dataframe.columns]
        return dataframe[available_columns]

    def train(self) -> dict[str, Any]:
        """Train the model end-to-end.

        Returns:
            Training metrics and results
        """
        # Load training data
        train_dataframe = self.data_loader.load_train_data()

        # Convert to pandas if using polars
        if hasattr(train_dataframe, "to_pandas"):
            train_dataframe = train_dataframe.to_pandas()

        # Separate features and target
        target_data = train_dataframe[self.config.target_column]
        feature_data = train_dataframe.drop(columns=[self.config.target_column])

        # Fit and transform features
        self.feature_engineer.fit(feature_data)
        transformed_features = self.feature_engineer.transform(feature_data)

        # Select final features
        final_features = self._prepare_features(transformed_features)

        # Split data into train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            final_features,
            target_data,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=target_data,
        )

        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        # Train ensemble model
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        validation_metrics = self.model.evaluate(X_val, y_val)

        # Cross-validation
        cv_results = self.model.cross_validate(X_train, y_train)

        # Test set evaluation
        test_metrics = self.model.evaluate(X_test, y_test)

        # Store metrics
        self.training_metrics = {
            "validation_metrics": validation_metrics,
            "cross_validation": cv_results,
            "test_metrics": test_metrics,
            "dataset_info": {
                "total_samples": len(train_dataframe),
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "feature_count": len(final_features.columns),
                "target_distribution": target_data.value_counts().to_dict(),
            },
        }

        # Save model and preprocessor
        self._save_artifacts()

        return self.training_metrics

    def _save_artifacts(self) -> None:
        """Save trained model and preprocessor artifacts."""
        # Create directories if they don't exist
        model_path = Path(self.config.save_model_path)
        preprocessor_path = Path(self.config.save_preprocessor_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and preprocessor
        self.model.save_model(str(model_path))

        # Save feature engineer using joblib directly
        import joblib

        joblib.dump(self.feature_engineer, str(preprocessor_path))

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Get feature importance from the trained ensemble model.

        Returns:
            DataFrame with feature importances if available
        """
        if not self.model.fitted_ or self.model.ensemble_model is None:
            return None

        importances = []
        feature_names = []

        # Extract feature importances from base models
        for name, estimator in self.model.ensemble_model.named_estimators_.items():
            if hasattr(estimator, "feature_importances_"):
                importance_dict = {
                    "model": name,
                    "importances": estimator.feature_importances_.tolist(),
                }
                importances.append(importance_dict)

        if not importances:
            return None

        # Get feature names from the last training
        if hasattr(self, "_last_feature_names"):
            feature_names = self._last_feature_names
        else:
            # Fallback to generic feature names
            num_features = len(importances[0]["importances"])
            feature_names = [f"feature_{i}" for i in range(num_features)]

        # Create DataFrame
        importance_dataframe = pd.DataFrame(
            {
                "feature": feature_names,
                **{imp["model"]: imp["importances"] for imp in importances},
            }
        )

        return importance_dataframe


# Default training configuration
DEFAULT_TRAINING_CONFIG: Final[TrainingConfig] = TrainingConfig()
