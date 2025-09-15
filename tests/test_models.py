"""Tests for model training functionality."""

import tempfile
from pathlib import Path

import pytest

from titanic_ml.data import DataConfig, DataLoader
from titanic_ml.models import ModelConfig, ModelTrainer
from titanic_ml.preprocessing import DataPreprocessor, PreprocessingConfig


class TestModelTrainer:
    """Test model training functionality."""

    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for testing."""
        # Load and preprocess real data
        data_config = DataConfig()
        loader = DataLoader(data_config)
        train_df, _ = loader.load_data()

        preprocessing_config = PreprocessingConfig()
        preprocessor = DataPreprocessor(preprocessing_config)
        X_train, y_train = preprocessor.fit_transform(train_df)

        return X_train, y_train, preprocessor

    def test_model_config_defaults(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.model_type == "xgboost"
        assert config.cv_folds == 5
        assert config.optimization_trials == 100
        assert config.random_state == 42

    def test_model_trainer_initialization(self):
        """Test model trainer initialization."""
        config = ModelConfig()
        trainer = ModelTrainer(config)
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.best_params is None

    def test_get_model_xgboost(self):
        """Test getting XGBoost model instance."""
        config = ModelConfig(model_type="xgboost")
        trainer = ModelTrainer(config)

        params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        model = trainer._get_model(params)

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_get_model_random_forest(self):
        """Test getting Random Forest model instance."""
        config = ModelConfig(model_type="random_forest")
        trainer = ModelTrainer(config)

        params = {"n_estimators": 100, "max_depth": 10}
        model = trainer._get_model(params)

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises error."""
        config = ModelConfig(model_type="invalid_model")
        trainer = ModelTrainer(config)

        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer._get_model({})

    def test_train_model_without_optimization(self, sample_processed_data):
        """Test training model without hyperparameter optimization."""
        X_train, y_train, preprocessor = sample_processed_data

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(
                optimization_trials=10,  # Small number for faster tests
                model_save_path=Path(temp_dir) / "model.joblib",
                preprocessor_save_path=Path(temp_dir) / "preprocessor.joblib",
            )
            trainer = ModelTrainer(config)

            model, metrics = trainer.train_model(X_train, y_train, optimize=False)

            assert model is not None
            assert trainer.model is not None
            assert "cv_mean" in metrics
            assert "cv_std" in metrics
            assert "cv_scores" in metrics
            assert metrics["cv_mean"] > 0
            assert metrics["cv_mean"] <= 1

    def test_train_model_with_optimization(self, sample_processed_data):
        """Test training model with hyperparameter optimization."""
        X_train, y_train, preprocessor = sample_processed_data

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(
                optimization_trials=5,  # Small number for faster tests
                model_save_path=Path(temp_dir) / "model.joblib",
                preprocessor_save_path=Path(temp_dir) / "preprocessor.joblib",
            )
            trainer = ModelTrainer(config)

            model, metrics = trainer.train_model(X_train, y_train, optimize=True)

            assert model is not None
            assert trainer.model is not None
            assert trainer.best_params is not None
            assert "cv_mean" in metrics
            assert "cv_std" in metrics
            assert metrics["cv_mean"] > 0
            assert metrics["cv_mean"] <= 1

    def test_save_and_load_model(self, sample_processed_data):
        """Test saving and loading model."""
        X_train, y_train, preprocessor = sample_processed_data

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(
                optimization_trials=5,
                model_save_path=Path(temp_dir) / "model.joblib",
                preprocessor_save_path=Path(temp_dir) / "preprocessor.joblib",
            )
            trainer = ModelTrainer(config)

            # Train and save model
            model, metrics = trainer.train_model(X_train, y_train, optimize=False)
            trainer.save_model(preprocessor)

            # Verify files exist
            assert config.model_save_path.exists()
            assert config.preprocessor_save_path.exists()

            # Load model
            new_trainer = ModelTrainer(config)
            loaded_model, loaded_preprocessor = new_trainer.load_model()

            assert loaded_model is not None
            assert loaded_preprocessor is not None
            assert new_trainer.model is not None

    def test_get_feature_importance(self, sample_processed_data):
        """Test getting feature importance."""
        X_train, y_train, preprocessor = sample_processed_data

        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(
                model_save_path=Path(temp_dir) / "model.joblib",
                preprocessor_save_path=Path(temp_dir) / "preprocessor.joblib",
            )
            trainer = ModelTrainer(config)

            # Should raise error before training
            with pytest.raises(ValueError, match="No model available"):
                trainer.get_feature_importance()

            # Train model
            model, metrics = trainer.train_model(X_train, y_train, optimize=False)

            # Should work after training
            importance = trainer.get_feature_importance()
            assert isinstance(importance, dict)
            assert len(importance) > 0
