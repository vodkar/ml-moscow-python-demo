"""Tests for the data loader module."""

from pathlib import Path

import polars as pl
import pytest

from titanic_ml.data_loader import (DataLoaderConfig, TitanicDataLoader,
                                    create_data_loader)


class TestDataLoaderConfig:
    """Test DataLoaderConfig class."""
    
    def test_default_config(self, temp_data_dir: Path) -> None:
        """Test default configuration values."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        
        assert config.data_path == temp_data_dir
        assert config.train_file == "train.csv"
        assert config.test_file == "test.csv"
        assert config.lazy_loading is True
        assert config.chunk_size is None
    
    def test_invalid_data_path(self) -> None:
        """Test validation of invalid data path."""
        with pytest.raises(ValueError, match="Data path does not exist"):
            DataLoaderConfig(data_path=Path("nonexistent_path"))


class TestTitanicDataLoader:
    """Test TitanicDataLoader class."""
    
    def test_initialization(self, temp_data_dir: Path) -> None:
        """Test data loader initialization."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        assert loader.config == config
        assert loader._train_data is None
        assert loader._test_data is None
    
    def test_file_paths(self, temp_data_dir: Path) -> None:
        """Test file path properties."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        assert loader.train_file_path == temp_data_dir / "train.csv"
        assert loader.test_file_path == temp_data_dir / "test.csv"
    
    def test_load_train_data(self, temp_data_dir: Path, sample_train_data: pl.DataFrame) -> None:
        """Test loading training data."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        train_data = loader.load_train_data()
        
        # Check basic properties
        assert isinstance(train_data, pl.DataFrame)
        assert train_data.shape[0] == sample_train_data.shape[0]
        assert "Survived" in train_data.columns
        assert "PassengerId" in train_data.columns
        
        # Test caching
        train_data_cached = loader.load_train_data()
        assert train_data.equals(train_data_cached)
    
    def test_load_test_data(self, temp_data_dir: Path, sample_test_data: pl.DataFrame) -> None:
        """Test loading test data."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        test_data = loader.load_test_data()
        
        # Check basic properties
        assert isinstance(test_data, pl.DataFrame)
        assert test_data.shape[0] == sample_test_data.shape[0]
        assert "Survived" not in test_data.columns
        assert "PassengerId" in test_data.columns
        
        # Test caching
        test_data_cached = loader.load_test_data()
        assert test_data.equals(test_data_cached)
    
    def test_force_reload(self, temp_data_dir: Path) -> None:
        """Test force reload functionality."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        # Load data initially
        train_data1 = loader.load_train_data()
        
        # Force reload
        train_data2 = loader.load_train_data(force_reload=True)
        
        # Data should be the same but force reload was called
        assert train_data1.equals(train_data2)
    
    def test_missing_file_error(self, temp_data_dir: Path) -> None:
        """Test error when data file is missing."""
        config = DataLoaderConfig(
            data_path=temp_data_dir,
            train_file="nonexistent.csv"
        )
        loader = TitanicDataLoader(config)
        
        with pytest.raises(FileNotFoundError, match="Training file not found"):
            loader.load_train_data()
    
    def test_dataset_info(self, temp_data_dir: Path) -> None:
        """Test dataset information retrieval."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        # Initially no info
        info = loader.get_dataset_info()
        assert info == {}
        
        # Load data and check info
        loader.load_train_data()
        loader.load_test_data()
        
        info = loader.get_dataset_info()
        assert "train" in info
        assert "test" in info
        
        assert "rows" in info["train"]
        assert "columns" in info["train"]
        assert "memory_mb" in info["train"]
        assert "null_values" in info["train"]
    
    def test_clear_cache(self, temp_data_dir: Path) -> None:
        """Test cache clearing functionality."""
        config = DataLoaderConfig(data_path=temp_data_dir)
        loader = TitanicDataLoader(config)
        
        # Load data
        loader.load_train_data()
        loader.load_test_data()
        
        assert loader._train_data is not None
        assert loader._test_data is not None
        
        # Clear cache
        loader.clear_cache()
        
        assert loader._train_data is None
        assert loader._test_data is None


class TestCreateDataLoader:
    """Test create_data_loader factory function."""
    
    def test_create_with_string_path(self, temp_data_dir: Path) -> None:
        """Test creating data loader with string path."""
        loader = create_data_loader(str(temp_data_dir))
        
        assert isinstance(loader, TitanicDataLoader)
        assert loader.config.data_path == temp_data_dir
    
    def test_create_with_path_object(self, temp_data_dir: Path) -> None:
        """Test creating data loader with Path object."""
        loader = create_data_loader(temp_data_dir)
        
        assert isinstance(loader, TitanicDataLoader)
        assert loader.config.data_path == temp_data_dir
    
    def test_create_with_default_path(self) -> None:
        """Test creating data loader with default path."""
        # This will fail due to validation, but tests the default
        with pytest.raises(ValueError):
            create_data_loader()