"""Prediction pipeline for Titanic ML models."""

from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from titanic_ml.models.ensemble import TitanicEnsemble
from titanic_ml.preprocessing.features import TitanicFeatureEngineer


class PredictionConfig(BaseModel):
    """Configuration for model prediction."""

    model_path: str = Field(default="models/titanic_ensemble.joblib")
    preprocessor_path: str = Field(default="models/feature_engineer.joblib")
    output_probabilities: bool = Field(default=False)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class TitanicPredictor:
    """Prediction pipeline for Titanic survival prediction."""

    def __init__(self, config: PredictionConfig) -> None:
        """Initialize predictor with configuration.

        Args:
            config: Prediction configuration
        """
        self.config = config
        self.model: TitanicEnsemble | None = None
        self.feature_engineer: TitanicFeatureEngineer | None = None
        self.loaded_: bool = False

    def load_models(self) -> None:
        """Load trained model and preprocessor from disk.

        Raises:
            FileNotFoundError: If model files are not found
        """
        model_path = Path(self.config.model_path)
        preprocessor_path = Path(self.config.preprocessor_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        # Load model and preprocessor
        self.model = TitanicEnsemble.load_model(str(model_path))
        self.feature_engineer = joblib.load(str(preprocessor_path))
        
        self.loaded_ = True

    def _prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Prepare features by selecting relevant columns.

        Args:
            dataframe: Input dataframe

        Returns:
            DataFrame with selected features
        """
        feature_columns = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", 
            "Fare", "Embarked", "FamilySize", "IsAlone", 
            "Title", "AgeBin", "FareBin"
        ]
        
        available_columns = [col for col in feature_columns if col in dataframe.columns]
        return dataframe[available_columns]

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on input data.

        Args:
            dataframe: Input dataframe with passenger features

        Returns:
            DataFrame with predictions and optional probabilities

        Raises:
            ValueError: If models are not loaded
        """
        if not self.loaded_ or self.model is None or self.feature_engineer is None:
            raise ValueError("Models must be loaded before making predictions")

        # Transform features
        transformed_features = self.feature_engineer.transform(dataframe)
        final_features = self._prepare_features(transformed_features)

        # Make predictions
        predictions = self.model.predict(final_features)
        
        result_dataframe = pd.DataFrame({
            "PassengerId": dataframe.index if "PassengerId" not in dataframe.columns else dataframe["PassengerId"],
            "Survived": predictions
        })

        # Add probabilities if requested
        if self.config.output_probabilities:
            probabilities = self.model.predict_proba(final_features)
            result_dataframe["Survival_Probability"] = probabilities["class_1"]
            result_dataframe["Confidence"] = probabilities.max(axis=1)
            
            # Add confidence-based prediction
            high_confidence_mask = result_dataframe["Confidence"] >= self.config.confidence_threshold
            result_dataframe["High_Confidence"] = high_confidence_mask

        return result_dataframe

    def predict_single(self, passenger_data: dict[str, Any]) -> dict[str, Any]:
        """Make prediction for a single passenger.

        Args:
            passenger_data: Dictionary with passenger features

        Returns:
            Dictionary with prediction results

        Raises:
            ValueError: If models are not loaded
        """
        if not self.loaded_:
            raise ValueError("Models must be loaded before making predictions")

        # Convert to DataFrame
        single_passenger_df = pd.DataFrame([passenger_data])
        
        # Make prediction
        prediction_result = self.predict(single_passenger_df)
        
        # Convert to dictionary
        result_dict = prediction_result.iloc[0].to_dict()
        
        return result_dict

    def explain_prediction(self, dataframe: pd.DataFrame) -> pd.DataFrame | None:
        """Get feature importance explanation for predictions.

        Args:
            dataframe: Input dataframe

        Returns:
            DataFrame with feature importance if available

        Raises:
            ValueError: If models are not loaded
        """
        if not self.loaded_ or self.model is None:
            raise ValueError("Models must be loaded before explanation")

        # Transform features to get the same features used in training
        if self.feature_engineer is not None:
            transformed_features = self.feature_engineer.transform(dataframe)
            final_features = self._prepare_features(transformed_features)
        else:
            final_features = dataframe

        # Get feature importance from the model
        if hasattr(self.model, 'get_feature_importance'):
            return None  # Would need to implement feature importance extraction
        
        return None

    def batch_predict(self, input_file: str, output_file: str) -> None:
        """Make batch predictions from CSV file.

        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file

        Raises:
            ValueError: If models are not loaded
            FileNotFoundError: If input file is not found
        """
        if not self.loaded_:
            raise ValueError("Models must be loaded before making predictions")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load data
        input_dataframe = pd.read_csv(input_path)
        
        # Make predictions
        predictions = self.predict(input_dataframe)
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information

        Raises:
            ValueError: If models are not loaded
        """
        if not self.loaded_ or self.model is None:
            raise ValueError("Models must be loaded to get model information")

        model_info = {
            "model_type": "TitanicEnsemble",
            "is_fitted": self.model.fitted_,
            "base_models": [],
        }

        if self.model.ensemble_model is not None:
            for name, estimator in self.model.ensemble_model.named_estimators_.items():
                model_info["base_models"].append({
                    "name": name,
                    "type": type(estimator).__name__
                })

        return model_info


# Default prediction configuration
DEFAULT_PREDICTION_CONFIG: Final[PredictionConfig] = PredictionConfig()