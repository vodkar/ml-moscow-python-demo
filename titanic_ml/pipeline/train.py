"""Training pipeline for end-to-end model training workflow."""

import logging
from pathlib import Path
from typing import Any

import polars as pl

from titanic_ml.config import AppConfig, TrainingConfig
from titanic_ml.core.data_loader import DataLoader
from titanic_ml.core.model import TitanicModel
from titanic_ml.core.preprocessor import TitanicPreprocessor
from titanic_ml.utils.metrics import ModelPerformanceReport

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for Titanic survival prediction model."""
    
    def __init__(self, config: AppConfig) -> None:
        """Initialize the training pipeline.
        
        Args:
            config: Application configuration object.
        """
        self.config = config
        self.data_loader = DataLoader(cache_enabled=True)
        self.preprocessor = TitanicPreprocessor(
            handle_missing=True,
            create_features=True,
            scale_features=True,
            encode_categorical=True,
        )
        self.model: TitanicModel | None = None
        
        logger.info("TrainingPipeline initialized")
    
    def run_full_pipeline(self) -> dict[str, Any]:
        """Run the complete training pipeline.
        
        Returns:
            Dictionary containing training results and metadata.
        """
        logger.info("Starting full training pipeline")
        
        # Step 1: Load data
        train_data, train_info = self.data_loader.load_train_data(self.config.data.train_file)
        logger.info(f"Training data loaded: {train_info.shape}")
        
        # Step 2: Preprocess data
        processed_features, processed_targets = self._preprocess_data(train_data)
        
        # Step 3: Train model with validation
        model, validation_report = self._train_model_with_validation(
            processed_features, processed_targets
        )
        
        # Step 4: Perform cross-validation
        cv_mean, cv_std, cv_scores = model.cross_validate(
            processed_features,
            processed_targets,
            cv_folds=self.config.training.cross_validation_folds,
        )
        
        # Step 5: Save model and preprocessor
        model_path, preprocessor_path = self._save_artifacts(model)
        
        # Step 6: Save performance report
        validation_report.cross_validation_scores = cv_scores
        report_path = self._save_performance_report(validation_report)
        
        # Compile results
        results = {
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "performance_report_path": str(report_path),
            "training_data_info": train_info.model_dump() if hasattr(train_info, "model_dump") else train_info.__dict__,
            "validation_metrics": validation_report.metrics.to_dict(),
            "cross_validation": {
                "mean_score": cv_mean,
                "std_score": cv_std,
                "individual_scores": cv_scores,
            },
            "model_info": model.get_model_info(),
            "feature_importance": model.get_feature_importance(),
            "preprocessor_features": self.preprocessor.get_feature_names(),
        }
        
        logger.info(f"Training pipeline completed successfully. Validation accuracy: {validation_report.metrics.accuracy:.4f}")
        return results
    
    def train_and_evaluate(
        self, 
        train_file: Path | None = None,
        validation_split: float = 0.2,
    ) -> tuple[TitanicModel, ModelPerformanceReport]:
        """Train model and evaluate performance.
        
        Args:
            train_file: Optional custom training file path.
            validation_split: Fraction of data for validation.
            
        Returns:
            Tuple of (trained_model, performance_report).
        """
        logger.info("Starting training and evaluation")
        
        # Use provided file or config file
        data_file = train_file or self.config.data.train_file
        
        # Load data
        train_data, _ = self.data_loader.load_train_data(data_file)
        
        # Preprocess data
        processed_features, processed_targets = self._preprocess_data(train_data)
        
        # Train model with validation
        model, validation_report = self._train_model_with_validation(
            processed_features, processed_targets, validation_split
        )
        
        return model, validation_report
    
    def hyperparameter_tuning(self, param_grid: dict[str, list[Any]]) -> dict[str, Any]:
        """Perform hyperparameter tuning using grid search.
        
        Args:
            param_grid: Dictionary of parameters to tune.
            
        Returns:
            Dictionary containing tuning results.
        """
        from sklearn.model_selection import GridSearchCV
        
        logger.info("Starting hyperparameter tuning")
        
        # Load and preprocess data
        train_data, _ = self.data_loader.load_train_data(self.config.data.train_file)
        processed_features, processed_targets = self._preprocess_data(train_data)
        
        # Convert to numpy for sklearn
        features_array = processed_features.to_numpy()
        targets_array = processed_targets.to_numpy()
        
        # Create base model
        base_model = TitanicModel(self.config.model)
        sklearn_model = base_model._create_model()
        
        # Perform grid search
        grid_search = GridSearchCV(
            sklearn_model,
            param_grid,
            cv=self.config.training.cross_validation_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        
        grid_search.fit(features_array, targets_array)
        
        # Extract results
        tuning_results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "best_score_std": float(grid_search.cv_results_["std_test_score"][grid_search.best_index_]),
            "all_results": {
                "params": grid_search.cv_results_["params"],
                "mean_test_scores": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_scores": grid_search.cv_results_["std_test_score"].tolist(),
            },
        }
        
        logger.info(f"Hyperparameter tuning completed. Best score: {tuning_results['best_score']:.4f}")
        return tuning_results
    
    def _preprocess_data(self, train_data: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
        """Preprocess training data.
        
        Args:
            train_data: Raw training data.
            
        Returns:
            Tuple of (processed_features, targets).
        """
        logger.info("Preprocessing training data")
        
        # Separate features and target
        targets = train_data["Survived"]
        
        # Fit and transform preprocessor
        self.preprocessor.fit(train_data, target="Survived")
        processed_features = self.preprocessor.transform(train_data)
        
        # Remove target column if it exists in processed features
        if "Survived" in processed_features.columns:
            processed_features = processed_features.drop("Survived")
        
        logger.info(f"Preprocessing completed. Features: {processed_features.shape[1]}")
        return processed_features, targets
    
    def _train_model_with_validation(
        self,
        features: pl.DataFrame,
        targets: pl.Series,
        validation_split: float = 0.2,
    ) -> tuple[TitanicModel, ModelPerformanceReport]:
        """Train model with automatic validation split.
        
        Args:
            features: Processed feature matrix.
            targets: Target vector.
            validation_split: Fraction for validation.
            
        Returns:
            Tuple of (trained_model, validation_report).
        """
        logger.info(f"Training model with {validation_split:.1%} validation split")
        
        # Create and train model
        self.model = TitanicModel(self.config.model)
        model, validation_report = self.model.train_with_validation(
            features, targets, validation_split
        )
        
        logger.info(f"Model training completed. Validation accuracy: {validation_report.metrics.accuracy:.4f}")
        return model, validation_report
    
    def _save_artifacts(self, model: TitanicModel) -> tuple[Path, Path]:
        """Save trained model and preprocessor.
        
        Args:
            model: Trained model instance.
            
        Returns:
            Tuple of (model_path, preprocessor_path).
        """
        logger.info("Saving model artifacts")
        
        # Save model
        model_path = self.config.training.model_save_path / "titanic_model.joblib"
        model.save(model_path)
        
        # Save preprocessor
        preprocessor_path = self.config.training.model_save_path / "preprocessor.joblib"
        self.preprocessor.save(preprocessor_path)
        
        logger.info(f"Artifacts saved - Model: {model_path}, Preprocessor: {preprocessor_path}")
        return model_path, preprocessor_path
    
    def _save_performance_report(self, report: ModelPerformanceReport) -> Path:
        """Save performance report to disk.
        
        Args:
            report: Performance report to save.
            
        Returns:
            Path where report was saved.
        """
        report_path = self.config.training.metrics_save_path / "performance_report.json"
        report.save_to_json(report_path)
        
        logger.info(f"Performance report saved to {report_path}")
        return report_path
    
    def validate_configuration(self) -> dict[str, Any]:
        """Validate the training configuration.
        
        Returns:
            Dictionary containing validation results.
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Check if training data exists
        if not self.config.data.train_file.exists():
            validation_results["errors"].append(f"Training file not found: {self.config.data.train_file}")
            validation_results["is_valid"] = False
        
        # Check model configuration
        if self.config.model.test_size <= 0 or self.config.model.test_size >= 1:
            validation_results["errors"].append("test_size must be between 0 and 1")
            validation_results["is_valid"] = False
        
        if self.config.training.cross_validation_folds < 2:
            validation_results["warnings"].append("cross_validation_folds should be at least 2")
        
        # Check if output directories are writable
        try:
            self.config.training.model_save_path.mkdir(parents=True, exist_ok=True)
            self.config.training.metrics_save_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            validation_results["errors"].append("Cannot create output directories")
            validation_results["is_valid"] = False
        
        logger.debug(f"Configuration validation completed: {validation_results}")
        return validation_results
    
    def get_pipeline_info(self) -> dict[str, Any]:
        """Get information about the training pipeline.
        
        Returns:
            Dictionary containing pipeline metadata.
        """
        return {
            "config": self.config.model_dump() if hasattr(self.config, "model_dump") else vars(self.config),
            "data_loader_cache_info": self.data_loader.get_cache_info(),
            "preprocessor_fitted": hasattr(self.preprocessor, "feature_names_") and bool(self.preprocessor.feature_names_),
            "model_fitted": self.model is not None and self.model.is_fitted_,
        }