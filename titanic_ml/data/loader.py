"""Efficient data loading with support for multiple backends."""

from pathlib import Path
from typing import Final

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data loading."""

    data_path: Path = Field(default=Path("data"), description="Path to data directory")
    use_polars: bool = Field(
        default=True, description="Use Polars for faster data loading"
    )
    cache_data: bool = Field(default=True, description="Cache loaded data")


class DataLoader:
    """High-performance data loader with multiple backend support."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize data loader with configuration.

        Args:
            config: Data loading configuration
        """
        self.config = config
        self._cache: dict[str, pd.DataFrame | pl.DataFrame] = {}

    def load_csv(
        self, filename: str, use_cache: bool = True
    ) -> pd.DataFrame | pl.DataFrame:
        """Load CSV file with optimal performance.

        Args:
            filename: Name of CSV file to load
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        cache_key = f"{filename}_{self.config.use_polars}"

        if use_cache and self.config.cache_data and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self.config.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        if self.config.use_polars:
            dataframe = pl.read_csv(file_path)
        else:
            dataframe = pd.read_csv(file_path)

        if self.config.cache_data:
            self._cache[cache_key] = dataframe

        return dataframe

    def load_train_data(self) -> pd.DataFrame | pl.DataFrame:
        """Load training data."""
        return self.load_csv("train.csv")

    def load_test_data(self) -> pd.DataFrame | pl.DataFrame:
        """Load test data."""
        return self.load_csv("test.csv")

    def clear_cache(self) -> None:
        """Clear data cache."""
        self._cache.clear()


# Global constants
DEFAULT_DATA_CONFIG: Final[DataConfig] = DataConfig()
