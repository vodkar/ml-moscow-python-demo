"""Prediction pipeline for making predictions on new Titanic data."""

import logging
from pathlib import Path
from typing import Any

import polars as pl

from titanic_ml.core.data_loader import DataLoader
from titanic_ml.core.model import TitanicModel
from titanic_ml.core.preprocessor import TitanicPreprocessor

logger = logging.getLogger(__name__)


class TitanicPredictor:
    """End-to-end prediction pipeline for Titanic survival prediction."""
    
    def __init__(
        self,
        model: TitanicModel | None = None,
        preprocessor: TitanicPreprocessor | None = None,
        data_loader: DataLoader | None = None,
    ) -> None:
        """Initialize the predictor with components.
        
        Args:
            model: Trained model instance.
            preprocessor: Fitted preprocessor instance.
            data_loader: Data loader instance.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.data_loader = data_loader or DataLoader(cache_enabled=True)
        
        logger.info("TitanicPredictor initialized")
    
    def load_model_and_preprocessor(
        self, 
        model_path: Path, 
        preprocessor_path: Path
    ) -> "TitanicPredictor":
        """Load trained model and preprocessor from disk.
        
        Args:
            model_path: Path to saved model.
            preprocessor_path: Path to saved preprocessor.
            
        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading model from {model_path}")
        self.model = TitanicModel.load(model_path)
        
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = TitanicPreprocessor.load(preprocessor_path)
        
        logger.info("Model and preprocessor loaded successfully")
        return self
    
    def predict_from_file(
        self, 
        input_file: Path, 
        output_file: Path | None = None
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Make predictions on data from a CSV file.
        
        Args:
            input_file: Path to input CSV file.
            output_file: Optional path to save predictions.
            
        Returns:
            Tuple of (predictions_dataframe, prediction_metadata).
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before making predictions")
        
        logger.info(f"Making predictions on data from {input_file}")
        
        # Load data
        raw_data, dataset_info = self.data_loader.load_test_data(input_file)
        logger.info(f"Loaded {dataset_info.shape[0]} samples for prediction")
        
        # Make predictions
        predictions_df, metadata = self.predict_from_dataframe(raw_data)
        
        # Save predictions if output path specified
        if output_file is not None:
            self._save_predictions(predictions_df, output_file)
        
        return predictions_df, metadata
    
    def predict_from_dataframe(self, dataframe: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Make predictions on a Polars DataFrame.
        
        Args:
            dataframe: Input DataFrame.
            
        Returns:
            Tuple of (predictions_dataframe, prediction_metadata).
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before making predictions")
        
        logger.info(f"Making predictions on DataFrame with shape {dataframe.shape}")
        
        # Preprocess data
        processed_data = self.preprocessor.transform(dataframe)
        logger.debug(f"Data preprocessed. Shape: {processed_data.shape}")
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        # Get prediction probabilities if available
        try:
            probabilities = self.model.predict_proba(processed_data)
            # For binary classification, get probability of survival (positive class)
            survival_probabilities = probabilities[:, 1] if probabilities.shape[1] == 2 else None
        except Exception as exception:
            logger.warning(f"Could not get prediction probabilities: {exception}")
            survival_probabilities = None
        
        # Create results DataFrame
        results_df = pl.DataFrame({
            "PassengerId": dataframe["PassengerId"],
            "Survived": predictions,
        })
        
        # Add probabilities if available
        if survival_probabilities is not None:
            results_df = results_df.with_columns([
                pl.Series("SurvivalProbability", survival_probabilities)
            ])
        
        # Prepare metadata
        metadata = {
            "input_shape": dataframe.shape,
            "processed_shape": processed_data.shape,
            "n_predictions": len(predictions),
            "survival_rate": float(predictions.mean()),
            "has_probabilities": survival_probabilities is not None,
            "model_type": self.model.config.model_type,
            "feature_count": len(self.model.feature_names_),
        }
        
        logger.info(f"Predictions completed. Survival rate: {metadata['survival_rate']:.2%}")
        return results_df, metadata
    
    def predict_single_passenger(self, passenger_data: dict[str, Any]) -> dict[str, Any]:
        """Make a prediction for a single passenger.
        
        Args:
            passenger_data: Dictionary containing passenger information.
            
        Returns:
            Dictionary containing prediction results.
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before making predictions")
        
        logger.debug("Making prediction for single passenger")
        
        # Convert to DataFrame
        passenger_df = pl.DataFrame([passenger_data])
        
        # Make prediction
        predictions_df, metadata = self.predict_from_dataframe(passenger_df)
        
        # Extract single result
        result = {
            "passenger_id": passenger_data.get("PassengerId", "Unknown"),
            "prediction": int(predictions_df["Survived"][0]),
            "survival_prediction": "Survived" if predictions_df["Survived"][0] == 1 else "Did not survive",
        }
        
        # Add probability if available
        if "SurvivalProbability" in predictions_df.columns:
            prob = float(predictions_df["SurvivalProbability"][0])
            result["survival_probability"] = prob
            result["confidence"] = "High" if prob > 0.7 or prob < 0.3 else "Medium"
        
        logger.debug(f"Single passenger prediction: {result['survival_prediction']}")
        return result
    
    def batch_predict_with_analysis(
        self, 
        input_file: Path, 
        output_dir: Path
    ) -> dict[str, Any]:
        """Perform batch prediction with comprehensive analysis.
        
        Args:
            input_file: Path to input CSV file.
            output_dir: Directory to save results and analysis.
            
        Returns:
            Dictionary containing analysis results.
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before making predictions")
        
        logger.info(f"Performing batch prediction with analysis on {input_file}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Make predictions
        predictions_df, metadata = self.predict_from_file(
            input_file, 
            output_dir / "predictions.csv"
        )
        
        # Perform analysis
        analysis = self._analyze_predictions(predictions_df)
        
        # Save analysis
        analysis_file = output_dir / "prediction_analysis.json"
        self._save_analysis(analysis, analysis_file)
        
        # Combine results
        results = {
            "predictions_file": str(output_dir / "predictions.csv"),
            "analysis_file": str(analysis_file),
            "metadata": metadata,
            "analysis": analysis,
        }
        
        logger.info(f"Batch prediction and analysis completed. Results saved to {output_dir}")
        return results
    
    def _save_predictions(self, predictions_df: pl.DataFrame, output_file: Path) -> None:
        """Save predictions to a CSV file.
        
        Args:
            predictions_df: DataFrame containing predictions.
            output_file: Path to save the predictions.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.write_csv(output_file)
        logger.info(f"Predictions saved to {output_file}")
    
    def _analyze_predictions(self, predictions_df: pl.DataFrame) -> dict[str, Any]:
        """Analyze prediction results.
        
        Args:
            predictions_df: DataFrame containing predictions.
            
        Returns:
            Dictionary containing analysis results.
        """
        total_passengers = len(predictions_df)
        survivors = predictions_df["Survived"].sum()
        non_survivors = total_passengers - survivors
        survival_rate = float(survivors / total_passengers) if total_passengers > 0 else 0.0
        
        analysis = {
            "total_passengers": total_passengers,
            "predicted_survivors": int(survivors),
            "predicted_non_survivors": int(non_survivors),
            "survival_rate": survival_rate,
        }
        
        # Add probability analysis if available
        if "SurvivalProbability" in predictions_df.columns:
            probs = predictions_df["SurvivalProbability"]
            analysis.update({
                "avg_survival_probability": float(probs.mean()),
                "min_survival_probability": float(probs.min()),
                "max_survival_probability": float(probs.max()),
                "high_confidence_predictions": int((probs > 0.8).sum() + (probs < 0.2).sum()),
                "low_confidence_predictions": int(((probs >= 0.4) & (probs <= 0.6)).sum()),
            })
        
        logger.debug(f"Prediction analysis completed: {analysis}")
        return analysis
    
    def _save_analysis(self, analysis: dict[str, Any], output_file: Path) -> None:
        """Save analysis results to a JSON file.
        
        Args:
            analysis: Analysis dictionary.
            output_file: Path to save the analysis.
        """
        import json
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(analysis, file, indent=2, ensure_ascii=False)
        
        logger.debug(f"Analysis saved to {output_file}")
    
    def get_prediction_summary(
        self, 
        predictions_df: pl.DataFrame
    ) -> dict[str, Any]:
        """Get a summary of predictions.
        
        Args:
            predictions_df: DataFrame containing predictions.
            
        Returns:
            Dictionary containing prediction summary.
        """
        return self._analyze_predictions(predictions_df)
    
    def validate_input_data(self, dataframe: pl.DataFrame) -> dict[str, Any]:
        """Validate input data for prediction.
        
        Args:
            dataframe: Input DataFrame to validate.
            
        Returns:
            Dictionary containing validation results.
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Check required columns
        required_columns = {"PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"}
        missing_columns = required_columns - set(dataframe.columns)
        
        if missing_columns:
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
            validation_results["is_valid"] = False
        
        # Check for empty data
        if len(dataframe) == 0:
            validation_results["errors"].append("Input data is empty")
            validation_results["is_valid"] = False
        
        # Check for duplicate PassengerIds
        if "PassengerId" in dataframe.columns:
            duplicate_ids = dataframe["PassengerId"].is_duplicated().sum()
            if duplicate_ids > 0:
                validation_results["warnings"].append(f"Found {duplicate_ids} duplicate PassengerIds")
        
        # Check for excessive missing values
        for column in dataframe.columns:
            null_percentage = dataframe[column].null_count() / len(dataframe)
            if null_percentage > 0.8:  # More than 80% missing
                validation_results["warnings"].append(
                    f"Column '{column}' has {null_percentage:.1%} missing values"
                )
        
        logger.debug(f"Input validation completed: {validation_results}")
        return validation_results