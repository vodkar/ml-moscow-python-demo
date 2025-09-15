"""Tests for data loading functionality."""

from pathlib import Path

import polars as pl
import pytest

from titanic_ml.data import DataConfig, DataLoader


class TestDataLoader:
    """Test data loading functionality."""

    def test_data_config_defaults(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.data_dir == Path("data")
        assert config.train_file == "train.csv"
        assert config.test_file == "test.csv"
        assert config.chunk_size == 10000

    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        config = DataConfig()
        loader = DataLoader(config)
        assert loader.config == config

    def test_load_train_data(self):
        """Test loading training data."""
        config = DataConfig()
        loader = DataLoader(config)
        train_df = loader.load_train_data()

        assert isinstance(train_df, pl.DataFrame)
        assert train_df.shape[0] > 0
        assert "PassengerId" in train_df.columns
        assert "Survived" in train_df.columns
        assert "Pclass" in train_df.columns

    def test_load_test_data(self):
        """Test loading test data."""
        config = DataConfig()
        loader = DataLoader(config)
        test_df = loader.load_test_data()

        assert isinstance(test_df, pl.DataFrame)
        assert test_df.shape[0] > 0
        assert "PassengerId" in test_df.columns
        assert "Pclass" in test_df.columns
        assert "Survived" not in test_df.columns

    def test_load_data(self):
        """Test loading both datasets."""
        config = DataConfig()
        loader = DataLoader(config)
        train_df, test_df = loader.load_data()

        assert isinstance(train_df, pl.DataFrame)
        assert isinstance(test_df, pl.DataFrame)
        assert train_df.shape[0] > 0
        assert test_df.shape[0] > 0

    def test_get_data_info(self):
        """Test getting data information."""
        config = DataConfig()
        loader = DataLoader(config)
        info = loader.get_data_info()

        assert "train_shape" in info
        assert "test_shape" in info
        assert "train_columns" in info
        assert "test_columns" in info
        assert "train_dtypes" in info
        assert "test_dtypes" in info

        assert len(info["train_shape"]) == 2
        assert len(info["test_shape"]) == 2
        assert info["train_shape"][0] > 0
        assert info["test_shape"][0] > 0
