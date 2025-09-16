"""Tests for the configuration module."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from titanic_ml.config import AppConfig, DataConfig, ModelConfig, PredictionConfig, TrainingConfig


class TestDataConfig:
    """Test cases for DataConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DataConfig()
        
        assert config.train_file == Path("data/train.csv")
        assert config.test_file == Path("data/test.csv")
        assert config.submission_file == Path("data/gender_submission.csv")
    
    def test_custom_paths(self) -> None:
        """Test custom file paths."""
        with tempfile.NamedTemporaryFile(suffix='.csv') as train_file, \
             tempfile.NamedTemporaryFile(suffix='.csv') as test_file, \
             tempfile.NamedTemporaryFile(suffix='.csv') as submission_file:
            
            config = DataConfig(
                train_file=Path(train_file.name),
                test_file=Path(test_file.name),
                submission_file=Path(submission_file.name)
            )
            
            assert config.train_file == Path(train_file.name)
            assert config.test_file == Path(test_file.name)
            assert config.submission_file == Path(submission_file.name)


class TestModelConfig:
    """Test cases for ModelConfig."""
    
    def test_default_config(self) -> None:
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.model_type == "xgboost"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.xgb_n_estimators == 100
    
    def test_valid_model_types(self) -> None:
        """Test valid model types."""
        for model_type in ["xgboost", "random_forest", "logistic_regression"]:
            config = ModelConfig(model_type=model_type)
            assert config.model_type == model_type
    
    def test_parameter_validation(self) -> None:
        """Test parameter validation."""
        # Test valid ranges
        config = ModelConfig(
            test_size=0.3,
            xgb_n_estimators=200,
            xgb_max_depth=10,
            xgb_learning_rate=0.05
        )
        assert config.test_size == 0.3
        assert config.xgb_n_estimators == 200
        
        # Test invalid ranges
        with pytest.raises(ValidationError):
            ModelConfig(test_size=1.5)  # > 1.0
        
        with pytest.raises(ValidationError):
            ModelConfig(xgb_learning_rate=2.0)  # > 1.0


class TestTrainingConfig:
    """Test cases for TrainingConfig."""
    
    def test_default_config(self) -> None:
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.model_save_path == Path("models")
        assert config.metrics_save_path == Path("metrics")
        assert config.cross_validation_folds == 5
        assert config.enable_feature_importance is True
    
    def test_directory_creation(self) -> None:
        """Test that directories are created during validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "test_models"
            metrics_path = temp_path / "test_metrics"
            
            config = TrainingConfig(
                model_save_path=model_path,
                metrics_save_path=metrics_path
            )
            
            assert model_path.exists()
            assert metrics_path.exists()


class TestPredictionConfig:
    """Test cases for PredictionConfig."""
    
    def test_default_config(self) -> None:
        """Test default prediction configuration."""
        config = PredictionConfig()
        
        assert config.model_path == Path("models/titanic_model.joblib")
        assert config.preprocessor_path == Path("models/preprocessor.joblib")
        assert config.output_path == Path("output/predictions.csv")


class TestAppConfig:
    """Test cases for the main AppConfig."""
    
    def test_default_config(self) -> None:
        """Test default application configuration."""
        config = AppConfig()
        
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.prediction, PredictionConfig)
        assert config.log_level == "INFO"
    
    def test_nested_config_updates(self) -> None:
        """Test updating nested configurations."""
        config = AppConfig()
        
        # Update model config
        config.model.model_type = "random_forest"
        assert config.model.model_type == "random_forest"
        
        # Update training config
        config.training.cross_validation_folds = 10
        assert config.training.cross_validation_folds == 10
    
    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = AppConfig(log_level=level)
            assert config.log_level == level
    
    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            AppConfig(invalid_field="value")