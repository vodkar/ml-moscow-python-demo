"""Configuration models using Pydantic for type validation."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""
    
    train_file: Path = Field(default=Path("data/train.csv"))
    test_file: Path = Field(default=Path("data/test.csv"))
    submission_file: Path = Field(default=Path("data/gender_submission.csv"))
    
    @field_validator("train_file", "test_file", "submission_file")
    @classmethod
    def validate_file_exists(cls, file_path: Path) -> Path:
        """Validate that data files exist."""
        if not file_path.exists():
            raise ValueError(f"Data file does not exist: {file_path}")
        return file_path


class ModelConfig(BaseModel):
    """Configuration for model training and hyperparameters."""
    
    model_type: Literal["random_forest", "xgboost", "logistic_regression"] = "xgboost"
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)
    
    # XGBoost specific parameters
    xgb_n_estimators: int = Field(default=100, ge=10, le=1000)
    xgb_max_depth: int = Field(default=6, ge=3, le=20)
    xgb_learning_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    
    # Random Forest specific parameters
    rf_n_estimators: int = Field(default=100, ge=10, le=1000)
    rf_max_depth: int | None = Field(default=None)
    rf_min_samples_split: int = Field(default=2, ge=2)
    
    # Logistic Regression specific parameters
    lr_max_iter: int = Field(default=1000, ge=100, le=10000)
    lr_c: float = Field(default=1.0, ge=0.001, le=100.0)


class TrainingConfig(BaseModel):
    """Configuration for the training pipeline."""
    
    model_save_path: Path = Field(default=Path("models"))
    metrics_save_path: Path = Field(default=Path("metrics"))
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    enable_feature_importance: bool = Field(default=True)
    
    @field_validator("model_save_path", "metrics_save_path")
    @classmethod
    def create_directory(cls, directory_path: Path) -> Path:
        """Create directory if it doesn't exist."""
        directory_path.mkdir(parents=True, exist_ok=True)
        return directory_path


class PredictionConfig(BaseModel):
    """Configuration for prediction pipeline."""
    
    model_path: Path = Field(default=Path("models/titanic_model.joblib"))
    preprocessor_path: Path = Field(default=Path("models/preprocessor.joblib"))
    output_path: Path = Field(default=Path("output/predictions.csv"))
    
    @field_validator("model_path", "preprocessor_path")
    @classmethod
    def validate_model_file_exists(cls, file_path: Path) -> Path:
        """Validate that model files exist."""
        if not file_path.exists():
            raise ValueError(f"Model file does not exist: {file_path}")
        return file_path
    
    @field_validator("output_path")
    @classmethod
    def create_output_directory(cls, output_path: Path) -> Path:
        """Create output directory if it doesn't exist."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path


class AppConfig(BaseModel):
    """Main application configuration."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    
    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        """Pydantic model configuration."""
        
        extra = "forbid"  # Forbid extra fields
        use_enum_values = True  # Use enum values instead of enum objects