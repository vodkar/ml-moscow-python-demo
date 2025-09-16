"""Tests for data loading functionality."""

import pytest
import pandas as pd
import polars as pl
from pathlib import Path
import tempfile

from titanic_ml.core.data_loader import DataLoader
from titanic_ml.core.models import DatasetInfo


@pytest.fixture
def sample_titanic_data():
    """Create sample Titanic data for testing."""
    data = {
        'PassengerId': [1, 2, 3, 4, 5],
        'Pclass': [3, 1, 3, 1, 3],
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 
                'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 
                'Allen, Mr. William Henry'],
        'Sex': ['male', 'female', 'female', 'female', 'male'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
        'SibSp': [1, 1, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05],
        'Cabin': [None, 'C85', None, 'C123', None],
        'Embarked': ['S', 'C', 'S', 'S', 'S'],
        'Survived': [0, 1, 1, 1, 0]
    }
    return data


@pytest.fixture
def temp_csv_file(sample_titanic_data):
    """Create temporary CSV file for testing."""
    df = pd.DataFrame(sample_titanic_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink()


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init_polars(self):
        """Test DataLoader initialization with Polars."""
        loader = DataLoader(use_polars=True)
        assert loader.use_polars is True
        assert loader._cache == {}
    
    def test_init_pandas(self):
        """Test DataLoader initialization with Pandas."""
        loader = DataLoader(use_polars=False)
        assert loader.use_polars is False
        assert loader._cache == {}
    
    def test_load_csv_pandas(self, temp_csv_file):
        """Test loading CSV with Pandas."""
        loader = DataLoader(use_polars=False)
        df = loader.load_csv(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'PassengerId' in df.columns
        assert 'Survived' in df.columns
    
    def test_load_csv_polars(self, temp_csv_file):
        """Test loading CSV with Polars."""
        loader = DataLoader(use_polars=True)
        df = loader.load_csv(temp_csv_file)
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        assert 'PassengerId' in df.columns
        assert 'Survived' in df.columns
    
    def test_load_csv_with_cache(self, temp_csv_file):
        """Test caching functionality."""
        loader = DataLoader(use_polars=False)
        
        # First load
        df1 = loader.load_csv(temp_csv_file, cache_key="test_data")
        assert "test_data" in loader._cache
        
        # Second load should use cache
        df2 = loader.load_csv(temp_csv_file, cache_key="test_data")
        
        # Should be the same object from cache
        assert df1 is df2
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = DataLoader()
        non_existent_path = Path("non_existent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load_csv(non_existent_path)
    
    def test_validate_titanic_schema_valid(self, temp_csv_file):
        """Test schema validation with valid data."""
        loader = DataLoader()
        # Should not raise exception
        df = loader.load_csv(temp_csv_file, validate_schema=True)
        assert len(df) == 5
    
    def test_get_dataset_info_pandas(self, sample_titanic_data):
        """Test dataset info generation with Pandas."""
        loader = DataLoader(use_polars=False)
        df = pd.DataFrame(sample_titanic_data)
        
        info = loader.get_dataset_info(df)
        
        assert isinstance(info, DatasetInfo)
        assert info.total_records == 5
        assert 'PassengerId' in info.features
        assert info.survival_rate == 0.6  # 3 out of 5 survived
        assert info.class_distribution == {0: 2, 1: 3}
    
    def test_get_dataset_info_polars(self, sample_titanic_data):
        """Test dataset info generation with Polars."""
        loader = DataLoader(use_polars=True)
        df = pl.DataFrame(sample_titanic_data)
        
        info = loader.get_dataset_info(df)
        
        assert isinstance(info, DatasetInfo)
        assert info.total_records == 5
        assert 'PassengerId' in info.features
        assert info.survival_rate == 0.6  # 3 out of 5 survived
    
    def test_clear_cache(self, temp_csv_file):
        """Test cache clearing."""
        loader = DataLoader()
        
        # Load data to populate cache
        loader.load_csv(temp_csv_file, cache_key="test")
        assert len(loader._cache) == 1
        
        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0