"""Tests for the trainer module."""

from pathlib import Path

import polars as pl
import pytest

from titanic_ml.trainer import (MODEL_TYPES, TitanicTrainer, TrainerConfig,
                                create_trainer)


class TestTrainerConfig:
    """Test TrainerConfig class."""
    
    def test_default_config(self, temp_model_dir: Path) -> None:
        """Test default configuration values."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        
        assert config.model_type == "xgboost"
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.cv_folds == 5
        assert config.hyperparameter_optimization is True
        assert config.optimization_trials == 100
        assert config.model_save_path == temp_model_dir
    
    def test_invalid_model_type(self, temp_model_dir: Path) -> None:
        """Test validation of invalid model type."""
        with pytest.raises(ValueError, match="Model type must be one of"):
            TrainerConfig(model_type="invalid_model", model_save_path=temp_model_dir)
    
    def test_invalid_test_size(self, temp_model_dir: Path) -> None:
        """Test validation of invalid test size."""
        with pytest.raises(ValueError, match="Test size must be between"):
            TrainerConfig(test_size=0.8, model_save_path=temp_model_dir)
    
    def test_invalid_cv_folds(self, temp_model_dir: Path) -> None:
        """Test validation of invalid CV folds."""
        with pytest.raises(ValueError, match="CV folds must be at least"):
            TrainerConfig(cv_folds=1, model_save_path=temp_model_dir)


class TestTitanicTrainer:
    """Test TitanicTrainer class."""
    
    def test_initialization(self, temp_model_dir: Path) -> None:
        """Test trainer initialization."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        assert trainer.config == config
        assert trainer.model is None
        assert trainer.feature_names == []
        assert trainer.performance is None
        assert trainer.best_params == {}
    
    def test_create_base_models(self, temp_model_dir: Path) -> None:
        """Test creation of different base models."""
        for model_type in MODEL_TYPES:
            config = TrainerConfig(model_type=model_type, model_save_path=temp_model_dir)
            trainer = TitanicTrainer(config)
            
            model = trainer._create_base_model()
            assert model is not None
            
            # Check that model has fit method
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
    
    def test_unsupported_model_type(self, temp_model_dir: Path) -> None:
        """Test error for unsupported model type."""
        # Create config with invalid model type (bypassing validation)
        config = TrainerConfig.__new__(TrainerConfig)
        config.model_type = "unsupported_model"
        config.model_save_path = temp_model_dir
        
        trainer = TitanicTrainer(config)
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer._create_base_model()
    
    @pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
    def test_train_simple_models(self, sample_train_data: pl.DataFrame, temp_model_dir: Path, model_type: str) -> None:
        """Test training simple models without hyperparameter optimization."""
        from titanic_ml.feature_engineer import create_feature_engineer

        # Prepare data
        feature_engineer = create_feature_engineer()
        processed_data = feature_engineer.fit_transform(sample_train_data)
        
        # Train model
        config = TrainerConfig(
            model_type=model_type,
            hyperparameter_optimization=False,
            model_save_path=temp_model_dir
        )
        trainer = TitanicTrainer(config)
        trainer.train(processed_data)
        
        # Check that model was trained
        assert trainer.model is not None
        assert len(trainer.feature_names) > 0
        assert trainer.performance is not None
        
        # Check performance metrics
        assert 0 <= trainer.performance.accuracy <= 1
        assert 0 <= trainer.performance.roc_auc <= 1
        assert trainer.performance.cv_mean > 0
    
    def test_train_missing_survived_column(self, sample_test_data: pl.DataFrame, temp_model_dir: Path) -> None:
        """Test training with missing Survived column."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        with pytest.raises(ValueError, match="Training data must contain 'Survived' column"):
            trainer.train(sample_test_data)
    
    def test_predict_without_training(self, sample_test_data: pl.DataFrame, temp_model_dir: Path) -> None:
        """Test prediction without training."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            trainer.predict(sample_test_data)
    
    def test_predict_proba_without_training(self, sample_test_data: pl.DataFrame, temp_model_dir: Path) -> None:
        """Test probability prediction without training."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            trainer.predict_proba(sample_test_data)
    
    def test_save_without_training(self, temp_model_dir: Path) -> None:
        """Test saving without training."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            trainer.save_model()
    
    def test_full_training_pipeline(self, sample_train_data: pl.DataFrame, sample_test_data: pl.DataFrame, temp_model_dir: Path) -> None:
        """Test complete training and prediction pipeline."""
        from titanic_ml.feature_engineer import create_feature_engineer

        # Prepare training data
        feature_engineer = create_feature_engineer()
        processed_train_data = feature_engineer.fit_transform(sample_train_data)
        
        # Train model
        config = TrainerConfig(
            model_type="logistic_regression",
            hyperparameter_optimization=False,
            model_save_path=temp_model_dir
        )
        trainer = TitanicTrainer(config)
        trainer.train(processed_train_data)
        
        # Save model
        model_path = trainer.save_model()
        assert model_path.exists()
        
        # Prepare test data
        processed_test_data = feature_engineer.transform(sample_test_data)
        
        # Make predictions
        predictions = trainer.predict(processed_test_data)
        assert len(predictions) == len(sample_test_data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Make probability predictions
        probabilities = trainer.predict_proba(processed_test_data)
        assert probabilities.shape[0] == len(sample_test_data)
        assert probabilities.shape[1] == 2  # Binary classification
    
    def test_load_nonexistent_model(self, temp_model_dir: Path) -> None:
        """Test loading non-existent model."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        nonexistent_path = temp_model_dir / "nonexistent.joblib"
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            trainer.load_model(nonexistent_path)
    
    def test_save_and_load_model(self, sample_train_data: pl.DataFrame, temp_model_dir: Path) -> None:
        """Test saving and loading model."""
        from titanic_ml.feature_engineer import create_feature_engineer

        # Prepare data and train model
        feature_engineer = create_feature_engineer()
        processed_data = feature_engineer.fit_transform(sample_train_data)
        
        config = TrainerConfig(
            model_type="logistic_regression",
            hyperparameter_optimization=False,
            model_save_path=temp_model_dir
        )
        trainer = TitanicTrainer(config)
        trainer.train(processed_data)
        
        # Save model
        model_path = trainer.save_model()
        
        # Create new trainer and load model
        new_trainer = TitanicTrainer(config)
        new_trainer.load_model(model_path)
        
        # Check that model was loaded correctly
        assert new_trainer.model is not None
        assert new_trainer.feature_names == trainer.feature_names
        assert new_trainer.performance is not None
    
    def test_get_feature_importance(self, sample_train_data: pl.DataFrame, temp_model_dir: Path) -> None:
        """Test getting feature importance."""
        from titanic_ml.feature_engineer import create_feature_engineer

        # Prepare data and train model
        feature_engineer = create_feature_engineer()
        processed_data = feature_engineer.fit_transform(sample_train_data)
        
        config = TrainerConfig(
            model_type="random_forest",
            hyperparameter_optimization=False,
            model_save_path=temp_model_dir
        )
        trainer = TitanicTrainer(config)
        trainer.train(processed_data)
        
        # Get feature importance
        importance = trainer.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, dict)
        assert len(importance) == len(trainer.feature_names)
        
        # Check that all values are non-negative
        assert all(val >= 0 for val in importance.values())
    
    def test_get_feature_importance_without_training(self, temp_model_dir: Path) -> None:
        """Test getting feature importance without training."""
        config = TrainerConfig(model_save_path=temp_model_dir)
        trainer = TitanicTrainer(config)
        
        importance = trainer.get_feature_importance()
        assert importance is None


class TestCreateTrainer:
    """Test create_trainer factory function."""
    
    def test_create_with_defaults(self) -> None:
        """Test creating trainer with default parameters."""
        trainer = create_trainer()
        
        assert isinstance(trainer, TitanicTrainer)
        assert trainer.config.model_type == "xgboost"
        assert trainer.config.hyperparameter_optimization is True
    
    def test_create_with_custom_params(self, temp_model_dir: Path) -> None:
        """Test creating trainer with custom parameters."""
        trainer = create_trainer(
            model_type="random_forest",
            hyperparameter_optimization=False,
            cv_folds=3,
            model_save_path=temp_model_dir
        )
        
        assert isinstance(trainer, TitanicTrainer)
        assert trainer.config.model_type == "random_forest"
        assert trainer.config.hyperparameter_optimization is False
        assert trainer.config.cv_folds == 3
        assert trainer.config.model_save_path == temp_model_dir