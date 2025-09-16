"""Tests for the feature engineer module."""

from pathlib import Path

import polars as pl
import pytest

from titanic_ml.feature_engineer import (FeatureEngineerConfig,
                                         TitanicFeatureEngineer,
                                         create_feature_engineer)


class TestFeatureEngineerConfig:
    """Test FeatureEngineerConfig class."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FeatureEngineerConfig()
        
        assert config.fill_missing_age is True
        assert config.fill_missing_embarked is True
        assert config.fill_missing_fare is True
        assert config.create_title_feature is True
        assert config.create_family_features is True
        assert config.create_age_bands is True
        assert config.create_fare_bands is True
        assert config.create_cabin_features is True
        assert config.drop_original_features is True


class TestTitanicFeatureEngineer:
    """Test TitanicFeatureEngineer class."""
    
    def test_initialization(self) -> None:
        """Test feature engineer initialization."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer._trained_fill_values == {}
        assert engineer._is_fitted is False
    
    def test_extract_title(self, sample_train_data: pl.DataFrame) -> None:
        """Test title extraction from names."""
        config = FeatureEngineerConfig(create_title_feature=True)
        engineer = TitanicFeatureEngineer(config)
        
        result = engineer._extract_title(sample_train_data)
        
        assert "Title" in result.columns
        # All names have "Mr." so should be mapped to "Mr"
        titles = result.select("Title").unique().to_series().to_list()
        assert "Mr" in titles
    
    def test_create_family_features(self, sample_train_data: pl.DataFrame) -> None:
        """Test family feature creation."""
        config = FeatureEngineerConfig(create_family_features=True)
        engineer = TitanicFeatureEngineer(config)
        
        result = engineer._create_family_features(sample_train_data)
        
        expected_columns = ["FamilySize", "IsAlone", "FamilySizeCategory"]
        for col in expected_columns:
            assert col in result.columns
        
        # Check family size calculation
        family_sizes = result.select("FamilySize").to_series()
        assert family_sizes.min() >= 1  # At least the person themselves
    
    def test_create_cabin_features(self, sample_train_data: pl.DataFrame) -> None:
        """Test cabin feature creation."""
        config = FeatureEngineerConfig(create_cabin_features=True)
        engineer = TitanicFeatureEngineer(config)
        
        result = engineer._create_cabin_features(sample_train_data)
        
        expected_columns = ["CabinDeck", "HasCabin", "CabinCount"]
        for col in expected_columns:
            assert col in result.columns
        
        # Check that Unknown is used for null values
        cabin_decks = result.select("CabinDeck").unique().to_series().to_list()
        assert "Unknown" in cabin_decks
    
    def test_create_age_bands(self, sample_train_data: pl.DataFrame) -> None:
        """Test age band creation."""
        config = FeatureEngineerConfig(create_age_bands=True)
        engineer = TitanicFeatureEngineer(config)
        
        # First fill missing ages
        filled_data = engineer._fill_missing_values(sample_train_data, is_training=True)
        result = engineer._create_age_bands(filled_data)
        
        assert "AgeBand" in result.columns
        
        # Check that age bands are created
        age_bands = result.select("AgeBand").drop_nulls().unique().to_series().to_list()
        assert len(age_bands) > 0
    
    def test_create_fare_bands(self, sample_train_data: pl.DataFrame) -> None:
        """Test fare band creation."""
        config = FeatureEngineerConfig(create_fare_bands=True)
        engineer = TitanicFeatureEngineer(config)
        
        # First fill missing fares
        filled_data = engineer._fill_missing_values(sample_train_data, is_training=True)
        result = engineer._create_fare_bands(filled_data)
        
        assert "FareBand" in result.columns
        
        # Check that fare bands are created
        fare_bands = result.select("FareBand").drop_nulls().unique().to_series().to_list()
        assert len(fare_bands) > 0
    
    def test_fill_missing_values_training(self, sample_train_data: pl.DataFrame) -> None:
        """Test missing value filling for training data."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        result = engineer._fill_missing_values(sample_train_data, is_training=True)
        
        # Check that missing values are filled
        assert result.select("Age").null_count().item() == 0
        assert result.select("Fare").null_count().item() == 0
        assert result.select("Embarked").null_count().item() == 0
        
        # Check that fill values are stored
        assert "age_overall" in engineer._trained_fill_values
        assert "fare" in engineer._trained_fill_values
        assert "embarked" in engineer._trained_fill_values
    
    def test_fill_missing_values_inference(self, sample_train_data: pl.DataFrame, sample_test_data: pl.DataFrame) -> None:
        """Test missing value filling for inference data."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        # First train on training data
        engineer._fill_missing_values(sample_train_data, is_training=True)
        
        # Then apply to test data
        result = engineer._fill_missing_values(sample_test_data, is_training=False)
        
        # Check that missing values are filled using stored values
        assert result.select("Age").null_count().item() == 0
        assert result.select("Fare").null_count().item() == 0
        assert result.select("Embarked").null_count().item() == 0
    
    def test_encode_categorical_features(self, sample_train_data: pl.DataFrame) -> None:
        """Test categorical feature encoding."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        # Add some derived features first
        data_with_features = engineer._extract_title(sample_train_data)
        data_with_features = engineer._create_family_features(data_with_features)
        data_with_features = engineer._create_cabin_features(data_with_features)
        
        result = engineer._encode_categorical_features(data_with_features)
        
        # Check that encoded features are created
        expected_encoded = ["Sex_male", "Embarked_S", "Embarked_C", "Embarked_Q"]
        for col in expected_encoded:
            assert col in result.columns
        
        # Check that values are 0 or 1
        sex_male_values = result.select("Sex_male").unique().to_series().to_list()
        assert all(val in [0, 1] for val in sex_male_values)
    
    def test_drop_original_features(self, sample_train_data: pl.DataFrame) -> None:
        """Test dropping of original features."""
        config = FeatureEngineerConfig(drop_original_features=True)
        engineer = TitanicFeatureEngineer(config)
        
        # Add derived features first
        data_with_features = engineer._extract_title(sample_train_data)
        data_with_features = engineer._create_family_features(data_with_features)
        data_with_features = engineer._create_cabin_features(data_with_features)
        data_with_features = engineer._encode_categorical_features(data_with_features)
        
        result = engineer._drop_original_features(data_with_features)
        
        # Check that original categorical features are dropped
        dropped_features = ["Name", "Ticket", "Sex", "Embarked", "Cabin"]
        for col in dropped_features:
            assert col not in result.columns
    
    def test_fit_transform(self, sample_train_data: pl.DataFrame) -> None:
        """Test complete fit_transform pipeline."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        result = engineer.fit_transform(sample_train_data)
        
        # Check that engineer is fitted
        assert engineer._is_fitted is True
        
        # Check that result is a DataFrame
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == sample_train_data.shape[0]
        
        # Check that target column is preserved
        assert "Survived" in result.columns
        assert "PassengerId" in result.columns
    
    def test_transform_without_fit(self, sample_test_data: pl.DataFrame) -> None:
        """Test that transform fails without fitting."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        with pytest.raises(ValueError, match="Feature engineer must be fitted"):
            engineer.transform(sample_test_data)
    
    def test_transform_after_fit(self, sample_train_data: pl.DataFrame, sample_test_data: pl.DataFrame) -> None:
        """Test transform after fitting."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        # Fit on training data
        train_result = engineer.fit_transform(sample_train_data)
        
        # Transform test data
        test_result = engineer.transform(sample_test_data)
        
        # Check that both have the same columns (except target)
        train_columns = set(train_result.columns) - {"Survived"}
        test_columns = set(test_result.columns) - {"Survived"}
        assert train_columns == test_columns
    
    def test_get_feature_names(self, sample_train_data: pl.DataFrame) -> None:
        """Test getting feature names."""
        config = FeatureEngineerConfig()
        engineer = TitanicFeatureEngineer(config)
        
        feature_names = engineer.get_feature_names(sample_train_data)
        
        # Check that feature names are returned
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Check that target and ID are excluded
        assert "Survived" not in feature_names
        assert "PassengerId" not in feature_names


class TestCreateFeatureEngineer:
    """Test create_feature_engineer factory function."""
    
    def test_create_with_defaults(self) -> None:
        """Test creating feature engineer with default parameters."""
        engineer = create_feature_engineer()
        
        assert isinstance(engineer, TitanicFeatureEngineer)
        assert engineer.config.fill_missing_age is True
        assert engineer.config.create_title_feature is True
    
    def test_create_with_custom_params(self) -> None:
        """Test creating feature engineer with custom parameters."""
        engineer = create_feature_engineer(
            fill_missing_age=False,
            create_title_feature=False,
            create_family_features=False
        )
        
        assert isinstance(engineer, TitanicFeatureEngineer)
        assert engineer.config.fill_missing_age is False
        assert engineer.config.create_title_feature is False
        assert engineer.config.create_family_features is False