"""Efficient data loading module for Titanic dataset using Polars."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Define column schema for validation
REQUIRED_COLUMNS: Final[set[str]] = {
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}

TRAIN_REQUIRED_COLUMNS: Final[set[str]] = REQUIRED_COLUMNS | {"Survived"}


class DataLoaderConfig(BaseModel):
    """Configuration for data loader."""

    data_path: Path = Field(default=Path("data"), description="Path to data directory")
    train_file: str = Field(default="train.csv", description="Training data filename")
    test_file: str = Field(default="test.csv", description="Test data filename")
    lazy_loading: bool = Field(
        default=True, description="Use lazy loading for large datasets"
    )
    chunk_size: int | None = Field(
        default=None, description="Chunk size for processing large files"
    )

    @field_validator("data_path")
    @classmethod
    def validate_data_path(cls, path_value: Path) -> Path:
        """Validate that data path exists."""
        if not path_value.exists():
            raise ValueError(f"Data path does not exist: {path_value}")
        return path_value


class TitanicDataLoader:
    """Efficient data loader for Titanic dataset using Polars.

    Designed to handle large datasets efficiently with lazy loading
    and memory-optimized operations.
    """

    def __init__(self, config: DataLoaderConfig) -> None:
        """Initialize data loader with configuration.

        Args:
            config: DataLoaderConfig instance with loading parameters
        """
        self.config = config
        self._train_data: pl.DataFrame | None = None
        self._test_data: pl.DataFrame | None = None

    @property
    def train_file_path(self) -> Path:
        """Get full path to training file."""
        return self.config.data_path / self.config.train_file

    @property
    def test_file_path(self) -> Path:
        """Get full path to test file."""
        return self.config.data_path / self.config.test_file

    def _validate_schema(
        self, dataframe: pl.DataFrame, required_columns: set[str]
    ) -> None:
        """Validate that dataframe contains required columns.

        Args:
            dataframe: DataFrame to validate
            required_columns: Set of required column names

        Raises:
            ValueError: If required columns are missing
        """
        actual_columns = set(dataframe.columns)
        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Found columns: {actual_columns}"
            )

    def _optimize_dtypes(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Optimize column data types for memory efficiency.

        Args:
            dataframe: Input DataFrame

        Returns:
            DataFrame with optimized dtypes
        """
        # Define optimal dtypes for each column
        dtype_mapping = {
            "PassengerId": pl.UInt32,
            "Survived": pl.UInt8,  # Only for training data
            "Pclass": pl.UInt8,
            "Sex": pl.Categorical,
            "Age": pl.Float32,
            "SibSp": pl.UInt8,
            "Parch": pl.UInt8,
            "Fare": pl.Float32,
            "Cabin": pl.Categorical,
            "Embarked": pl.Categorical,
        }

        # Apply dtype conversions only for existing columns
        cast_expressions = []
        for column_name, dtype in dtype_mapping.items():
            if column_name in dataframe.columns:
                cast_expressions.append(pl.col(column_name).cast(dtype))

        if cast_expressions:
            dataframe = dataframe.with_columns(cast_expressions)

        return dataframe

    def load_train_data(self, force_reload: bool = False) -> pl.DataFrame:
        """Load training data with caching.

        Args:
            force_reload: Force reloading data even if cached

        Returns:
            Training DataFrame

        Raises:
            FileNotFoundError: If training file doesn't exist
            ValueError: If data validation fails
        """
        if self._train_data is not None and not force_reload:
            logger.info("Returning cached training data")
            return self._train_data

        if not self.train_file_path.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_file_path}")

        logger.info(f"Loading training data from {self.train_file_path}")

        if self.config.lazy_loading:
            lazy_frame = pl.scan_csv(self.train_file_path)
            dataframe = lazy_frame.collect()
        else:
            dataframe = pl.read_csv(self.train_file_path)

        # Validate schema
        self._validate_schema(dataframe, TRAIN_REQUIRED_COLUMNS)

        # Optimize dtypes
        dataframe = self._optimize_dtypes(dataframe)

        # Cache the data
        self._train_data = dataframe

        logger.info(
            f"Loaded training data: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns"
        )
        logger.info(f"Memory usage: {dataframe.estimated_size('mb'):.2f} MB")

        return dataframe

    def load_test_data(self, force_reload: bool = False) -> pl.DataFrame:
        """Load test data with caching.

        Args:
            force_reload: Force reloading data even if cached

        Returns:
            Test DataFrame

        Raises:
            FileNotFoundError: If test file doesn't exist
            ValueError: If data validation fails
        """
        if self._test_data is not None and not force_reload:
            logger.info("Returning cached test data")
            return self._test_data

        if not self.test_file_path.exists():
            raise FileNotFoundError(f"Test file not found: {self.test_file_path}")

        logger.info(f"Loading test data from {self.test_file_path}")

        if self.config.lazy_loading:
            lazy_frame = pl.scan_csv(self.test_file_path)
            dataframe = lazy_frame.collect()
        else:
            dataframe = pl.read_csv(self.test_file_path)

        # Validate schema (test data doesn't have 'Survived' column)
        self._validate_schema(dataframe, REQUIRED_COLUMNS)

        # Optimize dtypes
        dataframe = self._optimize_dtypes(dataframe)

        # Cache the data
        self._test_data = dataframe

        logger.info(
            f"Loaded test data: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns"
        )
        logger.info(f"Memory usage: {dataframe.estimated_size('mb'):.2f} MB")

        return dataframe

    def get_dataset_info(self) -> dict[str, dict[str, str | int | float]]:
        """Get information about loaded datasets.

        Returns:
            Dictionary with dataset statistics
        """
        info = {}

        if self._train_data is not None:
            info["train"] = {
                "rows": self._train_data.shape[0],
                "columns": self._train_data.shape[1],
                "memory_mb": round(self._train_data.estimated_size("mb"), 2),
                "null_values": self._train_data.null_count().sum_horizontal().item(),
            }

        if self._test_data is not None:
            info["test"] = {
                "rows": self._test_data.shape[0],
                "columns": self._test_data.shape[1],
                "memory_mb": round(self._test_data.estimated_size("mb"), 2),
                "null_values": self._test_data.null_count().sum_horizontal().item(),
            }

        return info

    def clear_cache(self) -> None:
        """Clear cached data to free memory."""
        logger.info("Clearing data cache")
        self._train_data = None
        self._test_data = None


def create_data_loader(data_path: str | Path = "data") -> TitanicDataLoader:
    """Create a configured data loader instance.

    Args:
        data_path: Path to data directory

    Returns:
        Configured TitanicDataLoader instance
    """
    config = DataLoaderConfig(data_path=Path(data_path))
    return TitanicDataLoader(config)
