"""Tests for data preprocessing functionality."""

import numpy as np
import polars as pl
import pytest

from titanic_ml.data import DataConfig, DataLoader
from titanic_ml.preprocessing import DataPreprocessor, PreprocessingConfig


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 5],
                "Survived": [0, 1, 1, 0, 1],
                "Pclass": [3, 1, 3, 1, 2],
                "Name": [
                    "Mr. John Smith",
                    "Mrs. Jane Doe",
                    "Miss Mary Johnson",
                    "Mr. Bob Wilson",
                    "Master. Tom Brown",
                ],
                "Sex": ["male", "female", "female", "male", "male"],
                "Age": [22.0, 38.0, None, 35.0, 4.0],
                "SibSp": [1, 1, 0, 1, 0],
                "Parch": [0, 0, 0, 0, 1],
                "Ticket": ["A123", "PC456", "STON789", "SC012", "CA345"],
                "Fare": [7.25, 71.28, 7.93, None, 8.05],
                "Cabin": [None, "C85", None, "C123", None],
                "Embarked": ["S", "C", "S", None, "S"],
            }
        )

    def test_preprocessing_config_defaults(self):
        """Test default preprocessing configuration."""
        config = PreprocessingConfig()
        assert config.target_column == "Survived"
        assert config.id_column == "PassengerId"
        assert "Sex" in config.categorical_features
        assert "Embarked" in config.categorical_features
        assert "Age" in config.numerical_features
        assert "Fare" in config.numerical_features

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)
        assert preprocessor.config == config
        assert not preprocessor._is_fitted
        assert len(preprocessor.label_encoders) == 0

    def test_fit_transform(self, sample_data):
        """Test fitting and transforming training data."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)

        X, y = preprocessor.fit_transform(sample_data)

        assert isinstance(X, pl.DataFrame)
        assert isinstance(y, pl.Series)
        assert preprocessor._is_fitted
        assert len(preprocessor.label_encoders) > 0
        assert X.shape[0] == sample_data.shape[0]
        assert len(y) == sample_data.shape[0]

        # Check that features are created
        assert "FamilySize" in X.columns
        assert "IsAlone" in X.columns
        assert "Title" in X.columns
        assert "FarePerPerson" in X.columns
        assert "AgeGroup" in X.columns

    def test_transform_without_fitting_raises_error(self, sample_data):
        """Test that transform raises error when not fitted."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)

        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_data)

    def test_feature_engineering(self, sample_data):
        """Test feature engineering functionality."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)

        # Create features
        df_with_features = preprocessor._create_features(sample_data)

        assert "FamilySize" in df_with_features.columns
        assert "IsAlone" in df_with_features.columns
        assert "Title" in df_with_features.columns
        assert "FarePerPerson" in df_with_features.columns
        assert "AgeGroup" in df_with_features.columns

        # Check family size calculation
        family_sizes = df_with_features.select("FamilySize").to_numpy().flatten()
        expected_family_sizes = [2, 2, 1, 2, 2]  # SibSp + Parch + 1
        np.testing.assert_array_equal(family_sizes, expected_family_sizes)

        # Check is_alone calculation
        is_alone = df_with_features.select("IsAlone").to_numpy().flatten()
        expected_is_alone = [0, 0, 1, 0, 0]  # Only passenger 3 is alone
        np.testing.assert_array_equal(is_alone, expected_is_alone)

    def test_missing_value_handling(self, sample_data):
        """Test missing value handling."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)

        # Create features first, then handle missing values
        df_with_features = preprocessor._create_features(sample_data)
        df_filled = preprocessor._handle_missing_values(
            df_with_features, is_training=True
        )

        # Check that Age and Fare nulls are filled
        assert df_filled.select(pl.col("Age").is_null().sum()).item() == 0
        assert df_filled.select(pl.col("Fare").is_null().sum()).item() == 0
        assert df_filled.select(pl.col("Embarked").is_null().sum()).item() == 0

    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)

        # Should raise error before fitting
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.get_feature_names()

        # Fit preprocessor
        preprocessor.fit_transform(sample_data)

        # Should work after fitting
        feature_names = preprocessor.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "FamilySize" in feature_names
        assert "IsAlone" in feature_names

    def test_full_pipeline_with_real_data(self):
        """Test full preprocessing pipeline with real data."""
        # Load real data
        config = DataConfig()
        loader = DataLoader(config)
        train_df, test_df = loader.load_data()

        # Preprocess
        preprocessing_config = PreprocessingConfig()
        preprocessor = DataPreprocessor(preprocessing_config)

        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test = preprocessor.transform(test_df)

        # Verify shapes and types
        assert isinstance(X_train, pl.DataFrame)
        assert isinstance(X_test, pl.DataFrame)
        assert isinstance(y_train, pl.Series)

        assert X_train.shape[0] == train_df.shape[0]
        assert X_test.shape[0] == test_df.shape[0]
        assert len(y_train) == train_df.shape[0]

        # Check that feature columns are consistent
        train_features = [col for col in X_train.columns if col != "PassengerId"]
        test_features = [col for col in X_test.columns if col != "PassengerId"]
        assert train_features == test_features
