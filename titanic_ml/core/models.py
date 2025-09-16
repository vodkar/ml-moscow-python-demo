"""Data models for Titanic ML application."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Sex(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"


class Embarked(str, Enum):
    """Port of embarkation enumeration."""
    CHERBOURG = "C"
    QUEENSTOWN = "Q" 
    SOUTHAMPTON = "S"


class PassengerRecord(BaseModel):
    """Individual passenger record."""
    PassengerId: int = Field(description="Passenger identifier")
    Pclass: int = Field(ge=1, le=3, description="Ticket class")
    Name: str = Field(description="Passenger name")
    Sex: str = Field(description="Gender")
    Age: Optional[float] = Field(ge=0, le=120, description="Age in years")
    SibSp: int = Field(ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(ge=0, description="Number of parents/children aboard")
    Ticket: str = Field(description="Ticket number")
    Fare: Optional[float] = Field(ge=0, description="Passenger fare")
    Cabin: Optional[str] = Field(description="Cabin number")
    Embarked: Optional[str] = Field(description="Port of embarkation")
    Survived: Optional[int] = Field(ge=0, le=1, description="Survival (0=No, 1=Yes)")


class ModelConfig(BaseModel):
    """Model training configuration."""
    model_type: str = Field(default="xgboost")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    cv_folds: int = Field(default=5, ge=2)
    random_state: int = Field(default=42)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class TrainingMetrics(BaseModel):
    """Model training metrics."""
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1) 
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    cv_scores: List[float] = Field(description="Cross-validation scores")


class PredictionResult(BaseModel):
    """Prediction result for a single passenger."""
    passenger_id: int
    survival_probability: float = Field(ge=0, le=1)
    prediction: int = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class DatasetInfo(BaseModel):
    """Dataset information and statistics."""
    total_records: int = Field(ge=0)
    features: List[str]
    missing_values: Dict[str, int] = Field(default_factory=dict)
    survival_rate: Optional[float] = Field(ge=0, le=1)
    class_distribution: Optional[Dict[str, int]] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    data_path: Path
    model_output_path: Path = Field(default_factory=lambda: Path("models"))
    use_polars: bool = Field(default=True, description="Use Polars for data processing")
    enable_hyperparameter_tuning: bool = Field(default=True)
    n_trials: int = Field(default=100, ge=10)
    
    class Config:
        arbitrary_types_allowed = True
        
    @field_validator("data_path", "model_output_path")
    @classmethod
    def validate_paths(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v