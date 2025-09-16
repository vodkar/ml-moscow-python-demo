"""Efficient data processing using Polars for scalability."""

from pathlib import Path
from typing import Optional, Tuple

import polars as pl
from rich.console import Console

from .config import DataConfig

console = Console()


class DataProcessor:
    """High-performance data processor using Polars."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_train_data(self) -> pl.DataFrame:
        """Load training data."""
        console.print(f"Loading training data from {self.config.train_path}")
        
        if not self.config.train_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.config.train_path}")
            
        return pl.read_csv(self.config.train_path)
    
    def load_test_data(self) -> pl.DataFrame:
        """Load test data."""
        console.print(f"Loading test data from {self.config.test_path}")
        
        if not self.config.test_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.config.test_path}")
            
        return pl.read_csv(self.config.test_path)
    
    def validate_data(self, df: pl.DataFrame, is_training: bool = True) -> None:
        """Validate data integrity and structure."""
        console.print("Validating data...")
        
        # Check required columns
        required_cols = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        if is_training:
            required_cols.append("Survived")
            
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types and basic statistics
        console.print(f"Data shape: {df.shape}")
        console.print(f"Missing values per column:")
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                console.print(f"  {col}: {null_count}")
    
    def split_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Split data into training and validation sets."""
        console.print(f"Splitting data with test_size={self.config.test_size}")
        
        # Stratified split by survival rate
        survived_1 = df.filter(pl.col("Survived") == 1)
        survived_0 = df.filter(pl.col("Survived") == 0)
        
        # Calculate split sizes
        val_size_1 = int(len(survived_1) * self.config.test_size)
        val_size_0 = int(len(survived_0) * self.config.test_size)
        
        # Sample validation sets
        val_1 = survived_1.sample(n=val_size_1, seed=self.config.random_state)
        val_0 = survived_0.sample(n=val_size_0, seed=self.config.random_state)
        
        # Create validation set
        val_df = pl.concat([val_1, val_0]).sample(fraction=1.0, seed=self.config.random_state)
        
        # Create training set (remaining data)
        val_ids = set(val_df["PassengerId"].to_list())
        train_df = df.filter(~pl.col("PassengerId").is_in(val_ids))
        
        console.print(f"Training set: {len(train_df)} samples")
        console.print(f"Validation set: {len(val_df)} samples")
        
        return train_df, val_df
    
    def get_data_info(self, df: pl.DataFrame) -> dict:
        """Get comprehensive data information."""
        info = {
            "shape": df.shape,
            "columns": df.columns,
            "dtypes": dict(zip(df.columns, [str(dtype) for dtype in df.dtypes])),
            "null_counts": {col: df[col].null_count() for col in df.columns},
            "memory_usage": df.estimated_size("mb")
        }
        
        if "Survived" in df.columns:
            survival_rate = df["Survived"].mean()
            info["survival_rate"] = survival_rate
            info["class_distribution"] = df["Survived"].value_counts().sort("Survived")
            
        return info