"""Efficient data ingestion and loading using Polars."""

from pathlib import Path
from typing import Tuple

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data loading."""

    data_dir: Path = Field(default=Path("data"))
    train_file: str = Field(default="train.csv")
    test_file: str = Field(default="test.csv")
    chunk_size: int = Field(default=10000, description="Chunk size for large datasets")


class DataLoader:
    """Efficient data loader using Polars for scalability."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._validate_files()

    def _validate_files(self) -> None:
        """Validate that required data files exist."""
        train_path = self.config.data_dir / self.config.train_file
        test_path = self.config.data_dir / self.config.test_file

        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")

        logger.info(f"Data files validated: {train_path}, {test_path}")

    def load_train_data(self) -> pl.DataFrame:
        """Load training data with optimized schema inference."""
        train_path = self.config.data_dir / self.config.train_file

        logger.info(f"Loading training data from {train_path}")

        df = pl.read_csv(
            train_path,
            dtypes={
                "PassengerId": pl.Int32,
                "Survived": pl.Int8,
                "Pclass": pl.Int8,
                "Age": pl.Float32,
                "SibSp": pl.Int8,
                "Parch": pl.Int8,
                "Fare": pl.Float32,
            },
            null_values=["", "NA", "NULL"],
        )

        logger.info(
            f"Loaded training data: {df.shape[0]:,} rows, {df.shape[1]} columns"
        )
        return df

    def load_test_data(self) -> pl.DataFrame:
        """Load test data with optimized schema inference."""
        test_path = self.config.data_dir / self.config.test_file

        logger.info(f"Loading test data from {test_path}")

        df = pl.read_csv(
            test_path,
            dtypes={
                "PassengerId": pl.Int32,
                "Pclass": pl.Int8,
                "Age": pl.Float32,
                "SibSp": pl.Int8,
                "Parch": pl.Int8,
                "Fare": pl.Float32,
            },
            null_values=["", "NA", "NULL"],
        )

        logger.info(f"Loaded test data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load both training and test data."""
        train_df = self.load_train_data()
        test_df = self.load_test_data()
        return train_df, test_df

    def get_data_info(self) -> dict:
        """Get basic information about the datasets."""
        train_df, test_df = self.load_data()

        return {
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
            "train_columns": train_df.columns,
            "test_columns": test_df.columns,
            "train_dtypes": {
                col: str(dtype) for col, dtype in zip(train_df.columns, train_df.dtypes)
            },
            "test_dtypes": {
                col: str(dtype) for col, dtype in zip(test_df.columns, test_df.dtypes)
            },
        }
