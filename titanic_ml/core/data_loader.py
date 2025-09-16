"""Efficient data loading module using Polars for high performance."""

import logging
from pathlib import Path
from typing import Final

import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Feature columns in the Titanic dataset
FEATURE_COLUMNS: Final[list[str]] = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", 
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
]

TRAIN_COLUMNS: Final[list[str]] = FEATURE_COLUMNS
TEST_COLUMNS: Final[list[str]] = [col for col in FEATURE_COLUMNS if col != "Survived"]


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""
    
    shape: tuple[int, int]
    columns: list[str] 
    missing_values: dict[str, int]
    dtypes: dict[str, str]
    memory_usage_mb: float


class DataLoader:
    """Efficient data loader using Polars for fast data operations."""
    
    def __init__(self, cache_enabled: bool = True) -> None:
        """Initialize the data loader.
        
        Args:
            cache_enabled: Whether to enable caching of loaded datasets.
        """
        self.cache_enabled = cache_enabled
        self._cached_datasets: dict[str, pl.DataFrame] = {}
        logger.info(f"DataLoader initialized with caching={'enabled' if cache_enabled else 'disabled'}")
    
    def load_train_data(self, file_path: Path) -> tuple[pl.DataFrame, DatasetInfo]:
        """Load training data with Survived target column.
        
        Args:
            file_path: Path to the training CSV file.
            
        Returns:
            Tuple of (DataFrame, DatasetInfo) containing the loaded data and metadata.
            
        Raises:
            FileNotFoundError: If the training file doesn't exist.
            ValueError: If required columns are missing.
        """
        logger.info(f"Loading training data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training file not found: {file_path}")
        
        cache_key = f"train_{file_path.stem}"
        
        if self.cache_enabled and cache_key in self._cached_datasets:
            logger.info("Using cached training data")
            dataframe = self._cached_datasets[cache_key]
        else:
            # Use Polars for efficient CSV reading with optimized dtypes
            dataframe = pl.read_csv(
                file_path,
                try_parse_dates=False,  # Keep dates as strings for now
                null_values=["", "NA", "NULL", "null"],
                ignore_errors=True,  # Handle malformed rows gracefully
            )
            
            # Validate required columns are present
            missing_columns = set(TRAIN_COLUMNS) - set(dataframe.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in training data: {missing_columns}")
            
            if self.cache_enabled:
                self._cached_datasets[cache_key] = dataframe
                logger.info(f"Training data cached with key: {cache_key}")
        
        dataset_info = self._generate_dataset_info(dataframe, "training")
        logger.info(f"Training data loaded: {dataset_info.shape[0]} rows, {dataset_info.shape[1]} columns")
        
        return dataframe, dataset_info
    
    def load_test_data(self, file_path: Path) -> tuple[pl.DataFrame, DatasetInfo]:
        """Load test data without Survived target column.
        
        Args:
            file_path: Path to the test CSV file.
            
        Returns:
            Tuple of (DataFrame, DatasetInfo) containing the loaded data and metadata.
            
        Raises:
            FileNotFoundError: If the test file doesn't exist.
            ValueError: If required columns are missing.
        """
        logger.info(f"Loading test data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        cache_key = f"test_{file_path.stem}"
        
        if self.cache_enabled and cache_key in self._cached_datasets:
            logger.info("Using cached test data")
            dataframe = self._cached_datasets[cache_key]
        else:
            # Use Polars for efficient CSV reading
            dataframe = pl.read_csv(
                file_path,
                try_parse_dates=False,
                null_values=["", "NA", "NULL", "null"],
                ignore_errors=True,
            )
            
            # Validate required columns are present (excluding Survived for test data)
            missing_columns = set(TEST_COLUMNS) - set(dataframe.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in test data: {missing_columns}")
            
            if self.cache_enabled:
                self._cached_datasets[cache_key] = dataframe
                logger.info(f"Test data cached with key: {cache_key}")
        
        dataset_info = self._generate_dataset_info(dataframe, "test")
        logger.info(f"Test data loaded: {dataset_info.shape[0]} rows, {dataset_info.shape[1]} columns")
        
        return dataframe, dataset_info
    
    def load_submission_template(self, file_path: Path) -> pl.DataFrame:
        """Load submission template for Kaggle submission format.
        
        Args:
            file_path: Path to the submission template CSV file.
            
        Returns:
            DataFrame containing PassengerId and Survived columns.
            
        Raises:
            FileNotFoundError: If the submission file doesn't exist.
        """
        logger.info(f"Loading submission template from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Submission template not found: {file_path}")
        
        submission_df = pl.read_csv(file_path)
        
        required_columns = {"PassengerId", "Survived"}
        if not required_columns.issubset(set(submission_df.columns)):
            raise ValueError(f"Submission template must contain columns: {required_columns}")
        
        logger.info(f"Submission template loaded: {submission_df.shape[0]} rows")
        return submission_df
    
    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        cache_count = len(self._cached_datasets)
        self._cached_datasets.clear()
        logger.info(f"Cleared {cache_count} cached datasets")
    
    def get_cache_info(self) -> dict[str, tuple[int, int]]:
        """Get information about cached datasets.
        
        Returns:
            Dictionary mapping cache keys to (rows, columns) tuples.
        """
        return {
            key: (df.shape[0], df.shape[1]) 
            for key, df in self._cached_datasets.items()
        }
    
    def _generate_dataset_info(self, dataframe: pl.DataFrame, dataset_type: str) -> DatasetInfo:
        """Generate comprehensive information about a dataset.
        
        Args:
            dataframe: The Polars DataFrame to analyze.
            dataset_type: Type of dataset (e.g., 'training', 'test').
            
        Returns:
            DatasetInfo object with dataset metadata.
        """
        # Calculate missing values per column
        missing_values = {}
        for column in dataframe.columns:
            null_count = dataframe.select(pl.col(column).is_null().sum()).item()
            if null_count > 0:
                missing_values[column] = null_count
        
        # Get data types
        dtypes = {col: str(dtype) for col, dtype in zip(dataframe.columns, dataframe.dtypes)}
        
        # Estimate memory usage (Polars doesn't have direct method, so we estimate)
        estimated_memory_mb = dataframe.estimated_size() / (1024 * 1024)
        
        dataset_info = DatasetInfo(
            shape=dataframe.shape,
            columns=list(dataframe.columns),
            missing_values=missing_values,
            dtypes=dtypes,
            memory_usage_mb=round(estimated_memory_mb, 2)
        )
        
        logger.debug(f"{dataset_type.capitalize()} dataset info generated: {dataset_info}")
        return dataset_info
    
    def validate_data_consistency(self, train_df: pl.DataFrame, test_df: pl.DataFrame) -> dict[str, list[str]]:
        """Validate consistency between training and test datasets.
        
        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            
        Returns:
            Dictionary containing any consistency issues found.
        """
        issues = {"warnings": [], "errors": []}
        
        # Check for column differences (excluding Survived which is only in train)
        train_cols = set(train_df.columns) - {"Survived"}
        test_cols = set(test_df.columns)
        
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        
        if missing_in_test:
            issues["errors"].append(f"Columns missing in test data: {missing_in_test}")
        
        if extra_in_test:
            issues["warnings"].append(f"Extra columns in test data: {extra_in_test}")
        
        # Check for dtype consistency in common columns
        common_columns = train_cols & test_cols
        for column in common_columns:
            train_dtype = train_df.select(pl.col(column)).dtypes[0]
            test_dtype = test_df.select(pl.col(column)).dtypes[0]
            
            if train_dtype != test_dtype:
                issues["warnings"].append(
                    f"Column '{column}' has different dtypes: train={train_dtype}, test={test_dtype}"
                )
        
        if issues["warnings"] or issues["errors"]:
            logger.warning(f"Data consistency issues found: {issues}")
        else:
            logger.info("No data consistency issues found")
        
        return issues