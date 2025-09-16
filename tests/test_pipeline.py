"""Tests for the Titanic ML application."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from titanic_ml.core.config import DataConfig, ModelConfig, get_config
from titanic_ml.core.data import DataProcessor
from titanic_ml.core.features import FeatureEngineer
from titanic_ml.core.model import ModelTrainer
from titanic_ml.pipeline.prediction import PredictionPipeline
from titanic_ml.pipeline.training import TrainingPipeline


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 3, 1, 2],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath",
            "Allen, Mr. William Henry"
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "Cabin": [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", "S"]
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Create sample test data for testing."""
    data = {
        "PassengerId": [892, 893, 894],
        "Pclass": [3, 3, 2],
        "Name": [
            "Kelly, Mr. James",
            "Wilkes, Mrs. James",
            "Myles, Mr. Thomas Francis"
        ],
        "Sex": ["male", "female", "male"],
        "Age": [34.5, 47.0, 62.0],
        "SibSp": [0, 1, 0],
        "Parch": [0, 0, 0],
        "Ticket": ["330911", "363272", "240276"],
        "Fare": [7.8292, 7.0, 9.6875],
        "Cabin": [None, None, None],
        "Embarked": ["Q", "S", "Q"]
    }
    return pl.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestDataConfig:
    """Test data configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.train_path == Path("data/train.csv")
        assert config.test_path == Path("data/test.csv")
        assert config.test_size == 0.2
        assert config.random_state == 42
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = DataConfig(test_size=0.3, random_state=123)
        assert config.test_size == 0.3
        assert config.random_state == 123
        
        # Invalid test_size
        with pytest.raises(ValueError):
            DataConfig(test_size=0.05)  # Too small
        
        with pytest.raises(ValueError):
            DataConfig(test_size=0.8)   # Too large


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert "random_forest" in config.algorithms
        assert "logistic_regression" in config.algorithms
        assert config.n_estimators == 100
        assert config.cv_folds == 5


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_init(self, temp_dir):
        """Test DataProcessor initialization."""
        config = DataConfig(train_path=temp_dir / "train.csv")
        processor = DataProcessor(config)
        assert processor.config == config
    
    def test_validate_data_training(self, sample_train_data):
        """Test training data validation."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        # Should not raise exception for valid data
        processor.validate_data(sample_train_data, is_training=True)
    
    def test_validate_data_test(self, sample_test_data):
        """Test test data validation."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        # Should not raise exception for valid test data
        processor.validate_data(sample_test_data, is_training=False)
    
    def test_validate_data_missing_columns(self, sample_train_data):
        """Test validation with missing columns."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        # Remove required column
        invalid_data = sample_train_data.drop("Age")
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor.validate_data(invalid_data, is_training=True)
    
    def test_split_data(self, sample_train_data):
        """Test data splitting."""
        config = DataConfig(test_size=0.4, random_state=42)
        processor = DataProcessor(config)
        
        train_df, val_df = processor.split_data(sample_train_data)
        
        # Check splits
        assert len(train_df) + len(val_df) == len(sample_train_data)
        assert len(val_df) > 0
        assert len(train_df) > 0
        
        # Check no overlap
        train_ids = set(train_df["PassengerId"].to_list())
        val_ids = set(val_df["PassengerId"].to_list())
        assert len(train_ids.intersection(val_ids)) == 0
    
    def test_get_data_info(self, sample_train_data):
        """Test data info extraction."""
        config = DataConfig()
        processor = DataProcessor(config)
        
        info = processor.get_data_info(sample_train_data)
        
        assert "shape" in info
        assert "columns" in info
        assert "survival_rate" in info
        assert info["shape"] == (5, 12)


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.label_encoders == {}
        assert not engineer.is_fitted
    
    def test_extract_title(self, sample_train_data):
        """Test title extraction."""
        engineer = FeatureEngineer()
        result = engineer.extract_title(sample_train_data)
        
        assert "Title" in result.columns
        titles = result["Title"].to_list()
        assert "Mr" in titles
        assert "Mrs" in titles
        assert "Miss" in titles
    
    def test_create_family_features(self, sample_train_data):
        """Test family feature creation."""
        engineer = FeatureEngineer()
        result = engineer.create_family_features(sample_train_data)
        
        expected_cols = ["FamilySize", "IsAlone", "FamilySizeCategory"]
        for col in expected_cols:
            assert col in result.columns
        
        # Check family size calculation
        family_sizes = result["FamilySize"].to_list()
        expected_family_sizes = [2, 2, 1, 2, 1]  # SibSp + Parch + 1
        assert family_sizes == expected_family_sizes
    
    def test_create_age_features(self, sample_train_data):
        """Test age feature creation."""
        engineer = FeatureEngineer()
        # Need to extract titles first as age features depend on it
        data_with_titles = engineer.extract_title(sample_train_data)
        result = engineer.create_age_features(data_with_titles)
        
        assert "AgeCategory" in result.columns
        assert "AgeBin" in result.columns
        
        # Check no missing ages
        assert result["Age"].null_count() == 0
    
    def test_fit_transform(self, sample_train_data):
        """Test fit and transform on training data."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_train_data)
        
        # Check that we have the expected columns
        feature_names = engineer.get_feature_names()
        for feature in feature_names:
            if feature in result.columns:
                assert result[feature].null_count() == 0  # No missing values
        
        assert engineer.is_fitted
    
    def test_transform_without_fit(self, sample_test_data):
        """Test transform without fitting first."""
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="must be fitted"):
            engineer.transform(sample_test_data)
    
    def test_fit_transform_then_transform(self, sample_train_data, sample_test_data):
        """Test full feature engineering pipeline."""
        engineer = FeatureEngineer()
        
        # Fit on training data
        train_result = engineer.fit_transform(sample_train_data)
        
        # Transform test data
        test_result = engineer.transform(sample_test_data)
        
        # Check that both have similar structure
        train_features = set(train_result.columns)
        test_features = set(test_result.columns)
        
        # Test should have all features except Survived
        expected_diff = {"Survived"}
        assert train_features - test_features == expected_diff


class TestModelTrainer:
    """Test model training functionality."""
    
    def test_init(self):
        """Test ModelTrainer initialization."""
        config = ModelConfig()
        trainer = ModelTrainer(config)
        assert trainer.config == config
        assert trainer.best_model is None
    
    def test_get_models(self):
        """Test model creation."""
        config = ModelConfig(algorithms=["random_forest", "logistic_regression"])
        trainer = ModelTrainer(config)
        
        models = trainer.get_models()
        assert "random_forest" in models
        assert "logistic_regression" in models
        assert "xgboost" not in models
    
    def test_prepare_data(self, sample_train_data):
        """Test data preparation for modeling."""
        config = ModelConfig()
        trainer = ModelTrainer(config)
        
        # First need to process features
        engineer = FeatureEngineer()
        processed_data = engineer.fit_transform(sample_train_data)
        trainer.feature_engineer = engineer
        
        X, y, feature_names = trainer.prepare_data(processed_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(X) == len(y) == len(processed_data)


class TestTrainingPipeline:
    """Test training pipeline functionality."""
    
    @patch('titanic_ml.pipeline.training.DataProcessor')
    @patch('titanic_ml.pipeline.training.ModelTrainer')
    def test_init(self, mock_trainer, mock_processor):
        """Test TrainingPipeline initialization."""
        pipeline = TrainingPipeline()
        assert pipeline.config is not None
        mock_processor.assert_called_once()
        mock_trainer.assert_called_once()
    
    def test_init_with_config_override(self):
        """Test initialization with config override."""
        override = {"algorithms": ["random_forest"]}
        pipeline = TrainingPipeline(override)
        assert pipeline.config is not None


class TestPredictionPipeline:
    """Test prediction pipeline functionality."""
    
    def test_init_no_model(self, temp_dir):
        """Test initialization when model doesn't exist."""
        fake_model_path = temp_dir / "nonexistent_model.joblib"
        
        with pytest.raises(FileNotFoundError):
            PredictionPipeline(fake_model_path)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = get_config()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
    
    def test_feature_names_consistency(self):
        """Test that feature names are consistent across modules."""
        engineer = FeatureEngineer()
        feature_names = engineer.get_feature_names()
        
        # Should be a non-empty list of strings
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


if __name__ == "__main__":
    pytest.main([__file__])