"""Tests for data preprocessing functionality."""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from titanic_ml.core.preprocessor import TitanicPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 
                'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 
                'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, None, 35.0, 35.0],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.28, 7.92, 53.1, None],
        'Cabin': [None, 'C85', None, 'C123', None],
        'Embarked': ['S', 'C', 'S', 'S', None],
        'Survived': [0, 1, 1, 1, 0]
    })


@pytest.fixture
def test_data():
    """Create test data without target variable."""
    return pd.DataFrame({
        'PassengerId': [6, 7, 8],
        'Pclass': [2, 3, 1],
        'Name': ['Test, Mr. One', 'Test, Mrs. Two', 'Test, Miss Three'],
        'Sex': ['male', 'female', 'male'],
        'Age': [25.0, None, 30.0],
        'SibSp': [0, 1, 2],
        'Parch': [1, 0, 1],
        'Ticket': ['12345', '67890', 'ABCDE'],
        'Fare': [15.0, 10.0, 50.0],
        'Cabin': ['A1', None, 'B2'],
        'Embarked': ['S', 'C', 'Q']
    })


class TestTitanicPreprocessor:
    """Test cases for TitanicPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        assert preprocessor.use_polars is False
        assert preprocessor.label_encoders == {}
        assert preprocessor.scaler is not None
        assert preprocessor.imputers == {}
        assert preprocessor.feature_names == []
        assert preprocessor.is_fitted is False
    
    def test_fit_transform(self, sample_data):
        """Test fitting and transforming training data."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        X, y = preprocessor.fit_transform(sample_data)
        
        # Check output shapes
        assert X.shape[0] == 5  # 5 samples
        assert X.shape[1] > 0   # Some features created
        assert y.shape[0] == 5  # 5 target values
        assert preprocessor.is_fitted is True
        
        # Check target values
        assert list(y) == [0, 1, 1, 1, 0]
        
        # Check feature names are stored
        assert len(preprocessor.feature_names) > 0
    
    def test_transform_without_fitting(self, sample_data):
        """Test that transform fails when not fitted."""
        preprocessor = TitanicPreprocessor()
        
        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            preprocessor.transform(sample_data)
    
    def test_transform_after_fitting(self, sample_data, test_data):
        """Test transforming new data after fitting."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        # Fit on training data
        X_train, y_train = preprocessor.fit_transform(sample_data)
        
        # Transform test data
        X_test = preprocessor.transform(test_data)
        
        # Check shapes match
        assert X_test.shape[1] == X_train.shape[1]
        assert X_test.shape[0] == 3  # 3 test samples
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering creates expected features."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        X, y = preprocessor.fit_transform(sample_data)
        
        # Check that engineered features exist in feature names
        feature_names = preprocessor.get_feature_importance_names()
        
        # Should have family-related features
        assert any('FamilySize' in name for name in feature_names)
        assert any('IsAlone' in name for name in feature_names)
    
    def test_missing_value_handling(self, sample_data):
        """Test missing value imputation."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        # Original data has missing values
        assert sample_data['Age'].isna().any()
        assert sample_data['Fare'].isna().any()
        assert sample_data['Embarked'].isna().any()
        
        X, y = preprocessor.fit_transform(sample_data)
        
        # After preprocessing, no missing values should remain
        assert not np.isnan(X).any()
    
    def test_categorical_encoding(self, sample_data):
        """Test categorical variable encoding."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        X, y = preprocessor.fit_transform(sample_data)
        
        # Check that label encoders were created
        assert len(preprocessor.label_encoders) > 0
        
        # Sex should be encoded
        assert 'Sex' in preprocessor.label_encoders
        sex_encoder = preprocessor.label_encoders['Sex']
        assert isinstance(sex_encoder, LabelEncoder)
        assert set(sex_encoder.classes_) == {'female', 'male'}
    
    def test_numerical_scaling(self, sample_data):
        """Test numerical feature scaling."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        X, y = preprocessor.fit_transform(sample_data)
        
        # Check that scaler was fitted
        assert hasattr(preprocessor.scaler, 'mean_')
        assert hasattr(preprocessor.scaler, 'scale_')
    
    def test_feature_names_consistency(self, sample_data, test_data):
        """Test that feature names remain consistent between train/test."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        X_train, _ = preprocessor.fit_transform(sample_data)
        X_test = preprocessor.transform(test_data)
        
        train_feature_names = preprocessor.get_feature_importance_names()
        
        # Should have same number of features
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] == len(train_feature_names)
    
    def test_unknown_category_handling(self, sample_data):
        """Test handling of unknown categories in test data."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        # Fit on training data
        X_train, _ = preprocessor.fit_transform(sample_data)
        
        # Create test data with unknown category
        test_data_unknown = pd.DataFrame({
            'PassengerId': [999],
            'Pclass': [1],
            'Name': ['Unknown, Mr. Test'],
            'Sex': ['unknown_gender'],  # Unknown category
            'Age': [30.0],
            'SibSp': [0],
            'Parch': [0],
            'Ticket': ['TEST123'],
            'Fare': [50.0],
            'Cabin': ['X1'],
            'Embarked': ['Z']  # Unknown category
        })
        
        # Should handle unknown categories gracefully
        X_test = preprocessor.transform(test_data_unknown)
        assert X_test.shape[1] == X_train.shape[1]
        assert X_test.shape[0] == 1
    
    def test_get_feature_importance_names(self, sample_data):
        """Test getting feature names for importance analysis."""
        preprocessor = TitanicPreprocessor(use_polars=False)
        
        # Before fitting, should return empty list
        assert preprocessor.get_feature_importance_names() == []
        
        # After fitting, should return feature names
        X, y = preprocessor.fit_transform(sample_data)
        feature_names = preprocessor.get_feature_importance_names()
        
        assert len(feature_names) == X.shape[1]
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)