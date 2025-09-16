"""Tests for feature engineering functionality."""

import pandas as pd
import pytest

from titanic_ml.preprocessing.features import FeatureConfig, TitanicFeatureEngineer


@pytest.fixture
def sample_titanic_data() -> pd.DataFrame:
    """Create sample Titanic data for testing."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "Allen, Mr. William Henry"
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, None],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "Cabin": [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", None]
    })


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureConfig()
        assert config.handle_missing_values is True
        assert config.create_family_size is True
        assert config.extract_title_from_name is True
        assert config.scale_numerical_features is True
        assert config.encode_categorical_features is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FeatureConfig(
            handle_missing_values=False,
            create_family_size=False,
            extract_title_from_name=False,
            scale_numerical_features=False,
            encode_categorical_features=False
        )
        assert config.handle_missing_values is False
        assert config.create_family_size is False
        assert config.extract_title_from_name is False
        assert config.scale_numerical_features is False
        assert config.encode_categorical_features is False


class TestTitanicFeatureEngineer:
    """Tests for TitanicFeatureEngineer."""

    def test_initialization(self) -> None:
        """Test feature engineer initialization."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        assert engineer.config == config
        assert len(engineer.label_encoders) == 0
        assert engineer.scaler is None
        assert engineer.fitted_ is False

    def test_extract_title(self) -> None:
        """Test title extraction from names."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        
        assert engineer._extract_title("Braund, Mr. Owen Harris") == "Mr"
        assert engineer._extract_title("Cumings, Mrs. John Bradley") == "Mrs"
        assert engineer._extract_title("Heikkinen, Miss. Laina") == "Miss"
        assert engineer._extract_title("Smith, Master. John") == "Master"
        assert engineer._extract_title("Jones, Dr. Mary") == "Rare"
        assert engineer._extract_title("Invalid Name") == "Unknown"

    def test_handle_missing_values(self, sample_titanic_data: pd.DataFrame) -> None:
        """Test missing value handling."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        
        processed_data = engineer._handle_missing_values(sample_titanic_data)
        
        # Age should be filled with group median
        assert processed_data["Age"].isna().sum() == 0
        
        # Embarked should be filled with mode
        assert processed_data["Embarked"].isna().sum() == 0
        
        # Fare should have no missing values in this sample
        assert processed_data["Fare"].isna().sum() == 0

    def test_create_derived_features(self, sample_titanic_data: pd.DataFrame) -> None:
        """Test derived feature creation."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        
        # First handle missing values
        processed_data = engineer._handle_missing_values(sample_titanic_data)
        derived_data = engineer._create_derived_features(processed_data)
        
        # Check family size features
        assert "FamilySize" in derived_data.columns
        assert "IsAlone" in derived_data.columns
        expected_family_sizes = [2, 2, 1, 2, 1]  # SibSp + Parch + 1
        assert derived_data["FamilySize"].tolist() == expected_family_sizes
        
        # Check title extraction
        assert "Title" in derived_data.columns
        expected_titles = ["Mr", "Mrs", "Miss", "Mrs", "Mr"]
        assert derived_data["Title"].tolist() == expected_titles

    def test_fit_transform_pipeline(self, sample_titanic_data: pd.DataFrame) -> None:
        """Test complete fit-transform pipeline."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        
        # Fit the transformer
        fitted_engineer = engineer.fit(sample_titanic_data)
        assert fitted_engineer.fitted_ is True
        assert len(fitted_engineer.label_encoders) > 0
        assert fitted_engineer.scaler is not None
        
        # Transform the data
        transformed_data = fitted_engineer.transform(sample_titanic_data)
        
        # Check that required columns exist
        expected_columns = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
            "Embarked", "FamilySize", "IsAlone", "Title", "AgeBin", "FareBin"
        ]
        for column in expected_columns:
            if column in sample_titanic_data.columns or column in ["FamilySize", "IsAlone", "Title", "AgeBin", "FareBin"]:
                assert column in transformed_data.columns

    def test_transform_without_fit(self, sample_titanic_data: pd.DataFrame) -> None:
        """Test that transform fails without fitting first."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        
        with pytest.raises(ValueError, match="Feature engineer must be fitted"):
            engineer.transform(sample_titanic_data)

    def test_get_feature_names(self) -> None:
        """Test feature name extraction."""
        config = FeatureConfig()
        engineer = TitanicFeatureEngineer(config)
        
        feature_names = engineer.get_feature_names()
        
        expected_base_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        expected_derived_features = ["FamilySize", "IsAlone", "Title", "AgeBin", "FareBin"]
        
        for feature in expected_base_features:
            assert feature in feature_names
        for feature in expected_derived_features:
            assert feature in feature_names

    def test_partial_feature_config(self, sample_titanic_data: pd.DataFrame) -> None:
        """Test feature engineering with partial configuration."""
        config = FeatureConfig(
            create_family_size=False,
            extract_title_from_name=False
        )
        engineer = TitanicFeatureEngineer(config)
        
        fitted_engineer = engineer.fit(sample_titanic_data)
        transformed_data = fitted_engineer.transform(sample_titanic_data)
        
        # These features should not be created
        assert "FamilySize" not in transformed_data.columns
        assert "IsAlone" not in transformed_data.columns
        assert "Title" not in transformed_data.columns