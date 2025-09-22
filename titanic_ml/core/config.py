"""Configuration management for the Titanic ML application."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """Configuration for data processing."""

    train_path: Path = Field(default=Path("data/train.csv"))
    test_path: Path = Field(default=Path("data/test.csv"))
    submission_path: Path = Field(default=Path("data/gender_submission.csv"))
    output_dir: Path = Field(default=Path("outputs"))

    # Data processing parameters
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = Field(default=42)


class ModelConfig(BaseModel):
    """Configuration for model training."""

    # Model selection
    algorithms: list[str] = Field(default=["random_forest", "logistic_regression"])

    # Model parameters
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: Optional[int] = Field(default=10, ge=3, le=20)
    learning_rate: float = Field(default=0.1, ge=0.01, le=1.0)

    # Training parameters
    cv_folds: int = Field(default=5, ge=3, le=10)
    scoring: str = Field(default="accuracy")
    n_jobs: int = Field(default=-1)
    random_state: int = Field(default=42)

    # Model persistence
    model_path: Path = Field(default=Path("outputs/model.joblib"))


class AppConfig(BaseSettings):
    """Main application configuration."""

    # Environment
    environment: Literal["development", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Components
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    class Config:
        env_prefix = "TITANIC_ML_"
        env_nested_delimiter = "__"


def get_config() -> AppConfig:
    """Get application configuration."""
    return AppConfig()
