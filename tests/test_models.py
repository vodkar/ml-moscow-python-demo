"""Tests for Pydantic models."""

import pytest
from pathlib import Path
from titanic_ml.core.models import (
    PassengerRecord, ModelConfig, TrainingMetrics, 
    PredictionResult, DatasetInfo, PipelineConfig
)


class TestPassengerRecord:
    """Test PassengerRecord model."""
    
    def test_valid_passenger_record(self):
        """Test creating valid passenger record."""
        record = PassengerRecord(
            PassengerId=1,
            Pclass=1,
            Name="Test, Mr. John",
            Sex="male",
            Age=25.0,
            SibSp=1,
            Parch=0,
            Ticket="12345",
            Fare=50.0,
            Cabin="A1",
            Embarked="S",
            Survived=1
        )
        
        assert record.PassengerId == 1
        assert record.Pclass == 1
        assert record.Sex == "male"
        assert record.Age == 25.0
        assert record.Survived == 1
    
    def test_passenger_record_with_optional_fields(self):
        """Test passenger record with optional fields as None."""
        record = PassengerRecord(
            PassengerId=1,
            Pclass=3,
            Name="Test, Mr. John",
            Sex="female",
            Age=None,  # Optional
            SibSp=0,
            Parch=0,
            Ticket="12345",
            Fare=None,  # Optional
            Cabin=None,  # Optional
            Embarked=None,  # Optional
            Survived=None  # Optional for test data
        )
        
        assert record.Age is None
        assert record.Fare is None
        assert record.Cabin is None
        assert record.Embarked is None
        assert record.Survived is None
    
    def test_invalid_pclass(self):
        """Test invalid passenger class."""
        with pytest.raises(ValueError):
            PassengerRecord(
                PassengerId=1,
                Pclass=4,  # Invalid, should be 1-3
                Name="Test, Mr. John",
                Sex="male",
                Age=25.0,
                SibSp=0,
                Parch=0,
                Ticket="12345",
                Fare=50.0
            )
    
    def test_invalid_age(self):
        """Test invalid age."""
        with pytest.raises(ValueError):
            PassengerRecord(
                PassengerId=1,
                Pclass=1,
                Name="Test, Mr. John",
                Sex="male",
                Age=-5.0,  # Invalid, should be >= 0
                SibSp=0,
                Parch=0,
                Ticket="12345",
                Fare=50.0
            )
    
    def test_invalid_survived(self):
        """Test invalid survived value."""
        with pytest.raises(ValueError):
            PassengerRecord(
                PassengerId=1,
                Pclass=1,
                Name="Test, Mr. John",
                Sex="male",
                Age=25.0,
                SibSp=0,
                Parch=0,
                Ticket="12345",
                Fare=50.0,
                Survived=2  # Invalid, should be 0 or 1
            )


class TestModelConfig:
    """Test ModelConfig model."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.model_type == "xgboost"
        assert config.hyperparameters == {}
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert config.test_size == 0.2
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            model_type="lightgbm",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
            cv_folds=10,
            random_state=123,
            test_size=0.3
        )
        
        assert config.model_type == "lightgbm"
        assert config.hyperparameters == {"n_estimators": 100, "max_depth": 5}
        assert config.cv_folds == 10
        assert config.random_state == 123
        assert config.test_size == 0.3


class TestTrainingMetrics:
    """Test TrainingMetrics model."""
    
    def test_valid_training_metrics(self):
        """Test valid training metrics."""
        metrics = TrainingMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            roc_auc=0.92,
            cv_scores=[0.84, 0.86, 0.83, 0.87, 0.85]
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.83
        assert metrics.recall == 0.87
        assert metrics.f1_score == 0.85
        assert metrics.roc_auc == 0.92
        assert len(metrics.cv_scores) == 5
    
    def test_invalid_metric_values(self):
        """Test invalid metric values (outside 0-1 range)."""
        with pytest.raises(ValueError):
            TrainingMetrics(
                accuracy=1.5,  # Invalid, should be <= 1
                precision=0.83,
                recall=0.87,
                f1_score=0.85,
                roc_auc=0.92,
                cv_scores=[0.84, 0.86, 0.83, 0.87, 0.85]
            )


class TestPredictionResult:
    """Test PredictionResult model."""
    
    def test_valid_prediction_result(self):
        """Test valid prediction result."""
        result = PredictionResult(
            passenger_id=123,
            survival_probability=0.75,
            prediction=1,
            confidence=0.5
        )
        
        assert result.passenger_id == 123
        assert result.survival_probability == 0.75
        assert result.prediction == 1
        assert result.confidence == 0.5
    
    def test_invalid_prediction_result(self):
        """Test invalid prediction result."""
        with pytest.raises(ValueError):
            PredictionResult(
                passenger_id=123,
                survival_probability=1.5,  # Invalid, should be <= 1
                prediction=1,
                confidence=0.5
            )


class TestDatasetInfo:
    """Test DatasetInfo model."""
    
    def test_dataset_info(self):
        """Test dataset info creation."""
        info = DatasetInfo(
            total_records=891,
            features=["Age", "Sex", "Pclass", "Fare"],
            missing_values={"Age": 177, "Cabin": 687},
            survival_rate=0.384,
            class_distribution={"0": 549, "1": 342}
        )
        
        assert info.total_records == 891
        assert len(info.features) == 4
        assert info.missing_values["Age"] == 177
        assert info.survival_rate == 0.384
        assert info.class_distribution["0"] == 549


class TestPipelineConfig:
    """Test PipelineConfig model."""
    
    def test_pipeline_config_with_string_paths(self):
        """Test pipeline config with string paths."""
        config = PipelineConfig(
            data_path="data",
            model_output_path="models",
            use_polars=True,
            enable_hyperparameter_tuning=True,
            n_trials=50
        )
        
        assert isinstance(config.data_path, Path)
        assert isinstance(config.model_output_path, Path)
        assert str(config.data_path) == "data"
        assert str(config.model_output_path) == "models"
        assert config.use_polars is True
        assert config.enable_hyperparameter_tuning is True
        assert config.n_trials == 50
    
    def test_pipeline_config_with_path_objects(self):
        """Test pipeline config with Path objects."""
        data_path = Path("test_data")
        model_path = Path("test_models")
        
        config = PipelineConfig(
            data_path=data_path,
            model_output_path=model_path
        )
        
        assert config.data_path == data_path
        assert config.model_output_path == model_path
    
    def test_pipeline_config_defaults(self):
        """Test pipeline config default values."""
        config = PipelineConfig(
            data_path=Path("data")
        )
        
        assert config.model_output_path == Path("models")
        assert config.use_polars is True
        assert config.enable_hyperparameter_tuning is True
        assert config.n_trials == 100