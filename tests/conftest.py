"""Test configuration and fixtures for the test suite."""

import tempfile
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_train_data() -> pl.DataFrame:
    """Create sample training data for testing."""
    return pl.DataFrame({
        "PassengerId": range(1, 101),
        "Survived": [0, 1] * 50,
        "Pclass": [1, 2, 3] * 33 + [1],
        "Name": [f"Person {i}, Mr. John" for i in range(100)],
        "Sex": ["male", "female"] * 50,
        "Age": [25.0, 30.0, None, 35.0] * 25,
        "SibSp": [0, 1, 2] * 33 + [0],
        "Parch": [0, 1] * 50,
        "Ticket": [f"TICKET{i}" for i in range(100)],
        "Fare": [10.0, 20.0, None, 30.0] * 25,
        "Cabin": [None, "A1", "B2", None] * 25,
        "Embarked": ["S", "C", "Q", None] * 25,
    })


@pytest.fixture
def sample_test_data() -> pl.DataFrame:
    """Create sample test data for testing."""
    return pl.DataFrame({
        "PassengerId": range(101, 151),
        "Pclass": [1, 2, 3] * 16 + [1, 2],
        "Name": [f"TestPerson {i}, Mrs. Jane" for i in range(50)],
        "Sex": ["female", "male"] * 25,
        "Age": [28.0, 32.0, None, 40.0] * 12 + [25.0, 30.0],
        "SibSp": [0, 1] * 25,
        "Parch": [0, 2] * 25,
        "Ticket": [f"TESTTICKET{i}" for i in range(50)],
        "Fare": [15.0, 25.0, None, 35.0] * 12 + [20.0, 40.0],
        "Cabin": [None, "C3", None, "D4"] * 12 + [None, "E5"],
        "Embarked": ["S", "C"] * 25,
    })


@pytest.fixture
def temp_data_dir(sample_train_data: pl.DataFrame, sample_test_data: pl.DataFrame) -> Path:
    """Create temporary directory with sample CSV files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir)
        
        # Write CSV files
        sample_train_data.write_csv(data_path / "train.csv")
        sample_test_data.write_csv(data_path / "test.csv")
        
        yield data_path


@pytest.fixture
def temp_model_dir() -> Path:
    """Create temporary directory for model files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)