"""Efficient data loading and validation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import polars as pl
from pydantic import ValidationError

from .models import DatasetInfo, PassengerRecord

logger = logging.getLogger(__name__)


class DataLoader:
    """High-performance data loader with validation and caching."""
    
    def __init__(self, use_polars: bool = True):
        self.use_polars = use_polars
        self._cache: Dict[str, Union[pd.DataFrame, pl.DataFrame]] = {}
        
    def load_csv(
        self, 
        file_path: Path, 
        cache_key: Optional[str] = None,
        validate_schema: bool = True
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load CSV file efficiently with optional caching and validation."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        cache_key = cache_key or str(file_path)
        
        if cache_key in self._cache:
            logger.info(f"Loading from cache: {cache_key}")
            return self._cache[cache_key]
            
        logger.info(f"Loading data from: {file_path}")
        
        if self.use_polars:
            df = pl.read_csv(file_path, null_values=["", "NA", "NULL"])
        else:
            df = pd.read_csv(file_path, na_values=["", "NA", "NULL"])
            
        if validate_schema:
            self._validate_titanic_schema(df)
            
        self._cache[cache_key] = df
        logger.info(f"Loaded {len(df)} records")
        
        return df
    
    def load_titanic_data(
        self, 
        data_dir: Path
    ) -> Tuple[Union[pd.DataFrame, pl.DataFrame], 
               Union[pd.DataFrame, pl.DataFrame], 
               Optional[Union[pd.DataFrame, pl.DataFrame]]]:
        """Load Titanic train, test, and optional submission data."""
        data_dir = Path(data_dir)
        
        train_df = self.load_csv(data_dir / "train.csv", "train")
        test_df = self.load_csv(data_dir / "test.csv", "test", validate_schema=False)
        
        submission_file = data_dir / "gender_submission.csv"
        submission_df = None
        if submission_file.exists():
            submission_df = self.load_csv(submission_file, "submission", validate_schema=False)
            
        return train_df, test_df, submission_df
    
    def validate_records(self, df: Union[pd.DataFrame, pl.DataFrame]) -> List[PassengerRecord]:
        """Validate individual records against Pydantic model."""
        if self.use_polars and isinstance(df, pl.DataFrame):
            records_dict = df.to_pandas().to_dict("records")
        else:
            records_dict = df.to_dict("records")
            
        valid_records = []
        invalid_count = 0
        
        for record in records_dict:
            try:
                valid_records.append(PassengerRecord(**record))
            except ValidationError as e:
                invalid_count += 1
                logger.warning(f"Invalid record {record.get('PassengerId', 'unknown')}: {e}")
                
        logger.info(f"Validated {len(valid_records)} records, {invalid_count} invalid")
        return valid_records
    
    def get_dataset_info(self, df: Union[pd.DataFrame, pl.DataFrame]) -> DatasetInfo:
        """Generate dataset statistics and information."""
        if self.use_polars and isinstance(df, pl.DataFrame):
            total_records = len(df)
            features = df.columns
            
            missing_values = {}
            for col in features:
                null_count = df.filter(pl.col(col).is_null()).height
                if null_count > 0:
                    missing_values[col] = null_count
                    
            survival_rate = None
            class_distribution = None
            if "Survived" in features:
                survival_stats = df.group_by("Survived").agg(pl.count().alias("count"))
                class_distribution = dict(zip(
                    survival_stats["Survived"].to_list(),
                    survival_stats["count"].to_list()
                ))
                total_with_survival = sum(class_distribution.values())
                survival_rate = class_distribution.get(1, 0) / total_with_survival
                
        else:
            total_records = len(df)
            features = df.columns.tolist()
            missing_values = df.isnull().sum().to_dict()
            missing_values = {k: int(v) for k, v in missing_values.items() if v > 0}
            
            survival_rate = None
            class_distribution = None
            if "Survived" in features:
                class_distribution = df["Survived"].value_counts().to_dict()
                survival_rate = class_distribution.get(1, 0) / len(df)
                
        return DatasetInfo(
            total_records=total_records,
            features=features,
            missing_values=missing_values,
            survival_rate=survival_rate,
            class_distribution=class_distribution
        )
    
    def _validate_titanic_schema(self, df: Union[pd.DataFrame, pl.DataFrame]) -> None:
        """Validate that the DataFrame has expected Titanic columns."""
        required_columns = {"PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"}
        
        if self.use_polars and isinstance(df, pl.DataFrame):
            columns = set(df.columns)
        else:
            columns = set(df.columns)
            
        missing_columns = required_columns - columns
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        logger.info("Schema validation passed")
        
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.info("Cache cleared")