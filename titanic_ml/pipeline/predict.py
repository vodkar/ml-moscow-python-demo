"""Prediction pipeline for making batch predictions on new data."""

import logging
from pathlib import Path
from typing import Any

import polars as pl

from titanic_ml.config import AppConfig, PredictionConfig
from titanic_ml.core.predictor import TitanicPredictor

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """End-to-end prediction pipeline for batch predictions."""
    
    def __init__(self, config: AppConfig) -> None:
        """Initialize the prediction pipeline.
        
        Args:
            config: Application configuration object.
        """
        self.config = config
        self.predictor = TitanicPredictor()
        
        logger.info("PredictionPipeline initialized")
    
    def run_batch_prediction(
        self, 
        input_file: Path | None = None,
        output_file: Path | None = None,
        with_analysis: bool = True,
    ) -> dict[str, Any]:
        """Run batch prediction on test data.
        
        Args:
            input_file: Optional custom input file path.
            output_file: Optional custom output file path.
            with_analysis: Whether to include comprehensive analysis.
            
        Returns:
            Dictionary containing prediction results and metadata.
        """
        logger.info("Starting batch prediction pipeline")
        
        # Use provided paths or config paths
        data_file = input_file or self.config.data.test_file
        result_file = output_file or self.config.prediction.output_path
        
        # Load model and preprocessor
        self.predictor.load_model_and_preprocessor(
            self.config.prediction.model_path,
            self.config.prediction.preprocessor_path,
        )
        
        # Run predictions with optional analysis
        if with_analysis:
            results = self.predictor.batch_predict_with_analysis(
                data_file, 
                result_file.parent
            )
        else:
            predictions_df, metadata = self.predictor.predict_from_file(
                data_file, 
                result_file
            )
            results = {
                "predictions_file": str(result_file),
                "metadata": metadata,
                "analysis": self.predictor.get_prediction_summary(predictions_df),
            }
        
        logger.info("Batch prediction pipeline completed successfully")
        return results
    
    def predict_single_sample(
        self, 
        passenger_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Make prediction for a single passenger.
        
        Args:
            passenger_data: Dictionary containing passenger information.
            
        Returns:
            Dictionary containing prediction result.
        """
        logger.info("Making single passenger prediction")
        
        # Load model and preprocessor if not already loaded
        if self.predictor.model is None or self.predictor.preprocessor is None:
            self.predictor.load_model_and_preprocessor(
                self.config.prediction.model_path,
                self.config.prediction.preprocessor_path,
            )
        
        # Make prediction
        result = self.predictor.predict_single_passenger(passenger_data)
        
        logger.info(f"Single prediction completed: {result['survival_prediction']}")
        return result
    
    def validate_input_file(self, input_file: Path) -> dict[str, Any]:
        """Validate input file for prediction.
        
        Args:
            input_file: Path to input file.
            
        Returns:
            Dictionary containing validation results.
        """
        logger.info(f"Validating input file: {input_file}")
        
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Check if file exists
        if not input_file.exists():
            validation_results["errors"].append(f"Input file not found: {input_file}")
            validation_results["is_valid"] = False
            return validation_results
        
        # Load and validate data
        try:
            from titanic_ml.core.data_loader import DataLoader
            data_loader = DataLoader()
            test_data, _ = data_loader.load_test_data(input_file)
            
            # Use predictor's validation method
            data_validation = self.predictor.validate_input_data(test_data)
            
            validation_results.update(data_validation)
            
        except Exception as exception:
            validation_results["errors"].append(f"Error loading/validating data: {str(exception)}")
            validation_results["is_valid"] = False
        
        logger.info(f"Input validation completed: {'Valid' if validation_results['is_valid'] else 'Invalid'}")
        return validation_results
    
    def compare_predictions(
        self, 
        predictions_file: Path,
        ground_truth_file: Path,
    ) -> dict[str, Any]:
        """Compare predictions against ground truth (if available).
        
        Args:
            predictions_file: Path to predictions CSV.
            ground_truth_file: Path to ground truth CSV.
            
        Returns:
            Dictionary containing comparison results.
        """
        logger.info(f"Comparing predictions against ground truth")
        
        try:
            # Load predictions and ground truth
            predictions_df = pl.read_csv(predictions_file)
            ground_truth_df = pl.read_csv(ground_truth_file)
            
            # Merge on PassengerId
            merged_df = predictions_df.join(
                ground_truth_df, 
                on="PassengerId", 
                how="inner",
                suffix="_true"
            )
            
            if len(merged_df) == 0:
                return {
                    "error": "No matching PassengerIds found between predictions and ground truth"
                }
            
            # Calculate metrics
            from titanic_ml.utils.metrics import ModelMetrics
            
            y_true = merged_df["Survived_true"].to_numpy()
            y_pred = merged_df["Survived"].to_numpy()
            
            metrics = ModelMetrics.calculate_classification_metrics(y_true, y_pred)
            confusion_matrix = ModelMetrics.calculate_confusion_matrix_metrics(y_true, y_pred)
            
            comparison_results = {
                "total_samples": len(merged_df),
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "confusion_matrix": confusion_matrix.to_dict(),
                "correct_predictions": int((y_true == y_pred).sum()),
                "incorrect_predictions": int((y_true != y_pred).sum()),
            }
            
            logger.info(f"Prediction comparison completed. Accuracy: {metrics.accuracy:.4f}")
            return comparison_results
            
        except Exception as exception:
            logger.error(f"Error comparing predictions: {str(exception)}")
            return {"error": str(exception)}
    
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information.
        """
        if self.predictor.model is None:
            # Try to load model if not already loaded
            try:
                self.predictor.load_model_and_preprocessor(
                    self.config.prediction.model_path,
                    self.config.prediction.preprocessor_path,
                )
            except Exception as exception:
                return {"error": f"Could not load model: {str(exception)}"}
        
        return self.predictor.model.get_model_info()
    
    def validate_configuration(self) -> dict[str, Any]:
        """Validate the prediction configuration.
        
        Returns:
            Dictionary containing validation results.
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Check if model files exist
        if not self.config.prediction.model_path.exists():
            validation_results["errors"].append(f"Model file not found: {self.config.prediction.model_path}")
            validation_results["is_valid"] = False
        
        if not self.config.prediction.preprocessor_path.exists():
            validation_results["errors"].append(f"Preprocessor file not found: {self.config.prediction.preprocessor_path}")
            validation_results["is_valid"] = False
        
        # Check if test file exists (if using default)
        if not self.config.data.test_file.exists():
            validation_results["warnings"].append(f"Default test file not found: {self.config.data.test_file}")
        
        # Check if output directory is writable
        try:
            self.config.prediction.output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            validation_results["errors"].append("Cannot create output directory")
            validation_results["is_valid"] = False
        
        logger.debug(f"Configuration validation completed: {validation_results}")
        return validation_results
    
    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the prediction pipeline.
        
        Returns:
            Dictionary containing pipeline metadata.
        """
        info = {
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else vars(self.config),
            "model_loaded": self.predictor.model is not None,
            "preprocessor_loaded": self.predictor.preprocessor is not None,
        }
        
        # Add model info if available
        if self.predictor.model is not None:
            info["model_info"] = self.predictor.model.get_model_info()
        
        return info