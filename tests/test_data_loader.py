"""Tests for the data loading module."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from titanic_ml.core.data_loader import DataLoader


class TestDataLoader:
    """Test cases for the DataLoader class."""
    
    @pytest.fixture
    def sample_train_data(self) -> str:
        """Create sample training data CSV content."""
        return """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S"""
    
    @pytest.fixture
    def sample_test_data(self) -> str:
        """Create sample test data CSV content."""
        return """PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
893,3,"Wilkes, Mrs. James (Ellen Needs)",female,47,1,0,363272,7,,S"""
    
    @pytest.fixture
    def temp_train_file(self, sample_train_data: str) -> Path:
        """Create temporary training file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as file:
            file.write(sample_train_data)
            return Path(file.name)
    
    @pytest.fixture
    def temp_test_file(self, sample_test_data: str) -> Path:
        """Create temporary test file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as file:
            file.write(sample_test_data)
            return Path(file.name)
    
    def test_init(self) -> None:
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.cache_enabled is True
        assert loader._cached_datasets == {}
        
        loader_no_cache = DataLoader(cache_enabled=False)
        assert loader_no_cache.cache_enabled is False
    
    def test_load_train_data(self, temp_train_file: Path) -> None:
        """Test loading training data."""
        loader = DataLoader()
        dataframe, dataset_info = loader.load_train_data(temp_train_file)
        
        # Check DataFrame
        assert isinstance(dataframe, pl.DataFrame)
        assert dataframe.shape == (3, 12)  # 3 rows, 12 columns
        assert "Survived" in dataframe.columns
        assert "PassengerId" in dataframe.columns
        
        # Check dataset info
        assert dataset_info.shape == (3, 12)
        assert len(dataset_info.columns) == 12
        assert isinstance(dataset_info.missing_values, dict)
        assert dataset_info.memory_usage_mb > 0
        
        # Clean up
        temp_train_file.unlink()
    
    def test_load_test_data(self, temp_test_file: Path) -> None:
        """Test loading test data."""
        loader = DataLoader()
        dataframe, dataset_info = loader.load_test_data(temp_test_file)
        
        # Check DataFrame
        assert isinstance(dataframe, pl.DataFrame)
        assert dataframe.shape == (2, 11)  # 2 rows, 11 columns (no Survived)
        assert "Survived" not in dataframe.columns
        assert "PassengerId" in dataframe.columns
        
        # Check dataset info
        assert dataset_info.shape == (2, 11)
        
        # Clean up
        temp_test_file.unlink()
    
    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent file raises error."""
        loader = DataLoader()
        nonexistent_file = Path("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load_train_data(nonexistent_file)
        
        with pytest.raises(FileNotFoundError):
            loader.load_test_data(nonexistent_file)
    
    def test_cache_functionality(self, temp_train_file: Path) -> None:
        """Test caching functionality."""
        loader = DataLoader(cache_enabled=True)
        
        # First load
        df1, _ = loader.load_train_data(temp_train_file)
        assert len(loader._cached_datasets) == 1
        
        # Second load should use cache
        df2, _ = loader.load_train_data(temp_train_file)
        assert df1.equals(df2)
        assert len(loader._cached_datasets) == 1
        
        # Check cache info
        cache_info = loader.get_cache_info()
        assert len(cache_info) == 1
        
        # Clear cache
        loader.clear_cache()
        assert len(loader._cached_datasets) == 0
        
        # Clean up
        temp_train_file.unlink()
    
    def test_validate_data_consistency(self, temp_train_file: Path, temp_test_file: Path) -> None:
        """Test data consistency validation."""
        loader = DataLoader()
        
        train_df, _ = loader.load_train_data(temp_train_file)
        test_df, _ = loader.load_test_data(temp_test_file)
        
        issues = loader.validate_data_consistency(train_df, test_df)
        
        assert isinstance(issues, dict)
        assert "warnings" in issues
        assert "errors" in issues
        
        # Clean up
        temp_train_file.unlink()
        temp_test_file.unlink()