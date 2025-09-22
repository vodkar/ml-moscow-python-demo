"""Tests for data loading functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from titanic_ml.data.loader import DataConfig, DataLoader


@pytest.fixture
def sample_csv_data() -> str:
    """Create sample CSV data for testing."""
    return """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S"""


@pytest.fixture
def temp_data_dir(sample_csv_data: str) -> Path:
    """Create temporary directory with sample data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir)
        
        # Create train.csv
        train_file = data_dir / "train.csv"
        train_file.write_text(sample_csv_data)
        
        # Create test.csv (without Survived column)
        test_data = sample_csv_data.replace("Survived,", "").replace(",0", "").replace(",1", "")
        test_file = data_dir / "test.csv"
        test_file.write_text(test_data)
        
        yield data_dir


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DataConfig()
        assert config.data_path == Path("data")
        assert config.use_polars is True
        assert config.cache_data is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DataConfig(
            data_path=Path("/custom/path"),
            use_polars=False,
            cache_data=False
        )
        assert config.data_path == Path("/custom/path")
        assert config.use_polars is False
        assert config.cache_data is False


class TestDataLoader:
    """Tests for DataLoader."""

    def test_initialization(self, temp_data_dir: Path) -> None:
        """Test data loader initialization."""
        config = DataConfig(data_path=temp_data_dir)
        loader = DataLoader(config)
        assert loader.config == config
        assert len(loader._cache) == 0

    def test_load_csv_with_pandas(self, temp_data_dir: Path) -> None:
        """Test CSV loading with pandas backend."""
        config = DataConfig(data_path=temp_data_dir, use_polars=False)
        loader = DataLoader(config)
        
        dataframe = loader.load_csv("train.csv")
        assert isinstance(dataframe, pd.DataFrame)
        assert len(dataframe) == 3
        assert "PassengerId" in dataframe.columns
        assert "Survived" in dataframe.columns

    def test_load_train_data(self, temp_data_dir: Path) -> None:
        """Test loading training data."""
        config = DataConfig(data_path=temp_data_dir, use_polars=False)
        loader = DataLoader(config)
        
        train_data = loader.load_train_data()
        assert isinstance(train_data, pd.DataFrame)
        assert len(train_data) == 3
        assert "Survived" in train_data.columns

    def test_load_test_data(self, temp_data_dir: Path) -> None:
        """Test loading test data."""
        config = DataConfig(data_path=temp_data_dir, use_polars=False)
        loader = DataLoader(config)
        
        test_data = loader.load_test_data()
        assert isinstance(test_data, pd.DataFrame)
        assert len(test_data) == 3

    def test_caching_behavior(self, temp_data_dir: Path) -> None:
        """Test data caching functionality."""
        config = DataConfig(data_path=temp_data_dir, cache_data=True, use_polars=False)
        loader = DataLoader(config)
        
        # First load should cache the data
        data1 = loader.load_csv("train.csv")
        assert len(loader._cache) == 1
        
        # Second load should use cached data
        data2 = loader.load_csv("train.csv")
        assert data1 is data2  # Same object reference
        
        # Clear cache and verify
        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_file_not_found(self, temp_data_dir: Path) -> None:
        """Test handling of non-existent files."""
        config = DataConfig(data_path=temp_data_dir)
        loader = DataLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")

    def test_no_caching(self, temp_data_dir: Path) -> None:
        """Test behavior when caching is disabled."""
        config = DataConfig(data_path=temp_data_dir, cache_data=False, use_polars=False)
        loader = DataLoader(config)
        
        # Load data without caching
        loader.load_csv("train.csv")
        assert len(loader._cache) == 0