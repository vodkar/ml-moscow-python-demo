"""Tests for model functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from titanic_ml.models.ensemble import ModelConfig, TitanicEnsemble


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample training data for testing."""
    features = pd.DataFrame({
        "Pclass": [1, 2, 3, 1, 2],
        "Sex": [0, 1, 1, 0, 1],  # Encoded: 0=male, 1=female
        "Age": [25.0, 30.0, 35.0, 40.0, 28.0],
        "Fare": [50.0, 25.0, 10.0, 75.0, 30.0],
        "FamilySize": [1, 2, 1, 3, 1],
        "IsAlone": [1, 0, 1, 0, 1],
    })
    target = pd.Series([1, 0, 0, 1, 0])
    return features, target


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelConfig()
        assert config.use_random_forest is True
        assert config.use_xgboost is True
        assert config.use_logistic_regression is True
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert config.n_jobs == -1

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ModelConfig(
            use_random_forest=False,
            use_xgboost=True,
            use_logistic_regression=False,
            cv_folds=3,
            random_state=123,
            n_jobs=1
        )
        assert config.use_random_forest is False
        assert config.use_xgboost is True
        assert config.use_logistic_regression is False
        assert config.cv_folds == 3
        assert config.random_state == 123
        assert config.n_jobs == 1


class TestTitanicEnsemble:
    """Tests for TitanicEnsemble."""

    def test_initialization(self) -> None:
        """Test ensemble model initialization."""
        config = ModelConfig()
        ensemble = TitanicEnsemble(config)
        assert ensemble.config == config
        assert len(ensemble.models) == 0
        assert ensemble.ensemble_model is None
        assert ensemble.fitted_ is False

    def test_create_base_models(self) -> None:
        """Test base model creation."""
        config = ModelConfig()
        ensemble = TitanicEnsemble(config)
        
        base_models = ensemble._create_base_models()
        
        # All models should be created by default
        assert "rf" in base_models
        assert "xgb" in base_models
        assert "lr" in base_models
        assert len(base_models) == 3

    def test_create_base_models_selective(self) -> None:
        """Test selective base model creation."""
        config = ModelConfig(
            use_random_forest=True,
            use_xgboost=False,
            use_logistic_regression=True
        )
        ensemble = TitanicEnsemble(config)
        
        base_models = ensemble._create_base_models()
        
        assert "rf" in base_models
        assert "xgb" not in base_models
        assert "lr" in base_models
        assert len(base_models) == 2

    def test_fit_predict_pipeline(self, sample_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test complete fit-predict pipeline."""
        X_train, y_train = sample_training_data
        
        config = ModelConfig(cv_folds=2)  # Reduce folds for faster testing
        ensemble = TitanicEnsemble(config)
        
        # Fit the ensemble
        fitted_ensemble = ensemble.fit(X_train, y_train)
        assert fitted_ensemble.fitted_ is True
        assert fitted_ensemble.ensemble_model is not None
        
        # Make predictions
        predictions = fitted_ensemble.predict(X_train)
        assert len(predictions) == len(X_train)
        assert predictions.dtype in ['int64', 'int32', 'object']  # Classification output
        
        # Make probability predictions
        probabilities = fitted_ensemble.predict_proba(X_train)
        assert probabilities.shape == (len(X_train), 2)  # Binary classification

    def test_evaluate(self, sample_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test model evaluation."""
        X_train, y_train = sample_training_data
        
        config = ModelConfig(cv_folds=2)
        ensemble = TitanicEnsemble(config)
        ensemble.fit(X_train, y_train)
        
        # Evaluate on the same data (not ideal but for testing)
        metrics = ensemble.evaluate(X_train, y_train)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "classification_report" in metrics
        
        # Metrics should be between 0 and 1
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_cross_validate(self, sample_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test cross-validation."""
        X_train, y_train = sample_training_data
        
        config = ModelConfig(cv_folds=2)
        ensemble = TitanicEnsemble(config)
        
        # Create base models for cross-validation
        ensemble.models = ensemble._create_base_models()
        from sklearn.ensemble import VotingClassifier
        estimators = [(name, model) for name, model in ensemble.models.items()]
        ensemble.ensemble_model = VotingClassifier(
            estimators=estimators, voting="soft", n_jobs=config.n_jobs
        )
        
        cv_results = ensemble.cross_validate(X_train, y_train)
        
        assert "mean_accuracy" in cv_results
        assert "std_accuracy" in cv_results
        assert "individual_scores" in cv_results
        
        assert 0.0 <= cv_results["mean_accuracy"] <= 1.0
        assert cv_results["std_accuracy"] >= 0.0
        assert len(cv_results["individual_scores"]) == config.cv_folds

    def test_save_load_model(self, sample_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test model saving and loading."""
        X_train, y_train = sample_training_data
        
        config = ModelConfig(cv_folds=2)
        ensemble = TitanicEnsemble(config)
        ensemble.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.joblib"
            
            # Save the model
            ensemble.save_model(str(model_path))
            assert model_path.exists()
            
            # Load the model
            loaded_ensemble = TitanicEnsemble.load_model(str(model_path))
            assert loaded_ensemble.fitted_ is True
            assert loaded_ensemble.ensemble_model is not None
            
            # Test that loaded model makes same predictions
            original_predictions = ensemble.predict(X_train)
            loaded_predictions = loaded_ensemble.predict(X_train)
            pd.testing.assert_series_equal(original_predictions, loaded_predictions)

    def test_predict_without_fit(self, sample_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
        """Test that prediction fails without fitting first."""
        X_train, _ = sample_training_data
        
        config = ModelConfig()
        ensemble = TitanicEnsemble(config)
        
        with pytest.raises(ValueError, match="Ensemble model must be fitted"):
            ensemble.predict(X_train)
        
        with pytest.raises(ValueError, match="Ensemble model must be fitted"):
            ensemble.predict_proba(X_train)
        
        with pytest.raises(ValueError, match="Ensemble model must be fitted"):
            ensemble.evaluate(X_train, pd.Series([0, 1, 0, 1, 0]))

    def test_save_without_fit(self) -> None:
        """Test that saving fails without fitting first."""
        config = ModelConfig()
        ensemble = TitanicEnsemble(config)
        
        with pytest.raises(ValueError, match="Ensemble model must be fitted"):
            ensemble.save_model("test.joblib")