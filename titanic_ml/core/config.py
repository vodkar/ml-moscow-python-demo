from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Paths(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    train_csv: Path = Field(default_factory=lambda: Path("data/train.csv"))
    test_csv: Path = Field(default_factory=lambda: Path("data/test.csv"))
    artifacts_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    model_path: Path = Field(default_factory=lambda: Path("artifacts/model.joblib"))
    features_path: Path = Field(
        default_factory=lambda: Path("artifacts/feature_columns.json")
    )
    predictions_path: Path = Field(
        default_factory=lambda: Path("artifacts/predictions.csv")
    )


class TrainConfig(BaseModel):
    target: str = "Survived"
    id_col: str = "PassengerId"
    numeric_features: list[str] = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]
    categorical_features: list[str] = [
        "Pclass",
        "Sex",
        "Embarked",
        "Cabin",
        "Ticket",
    ]
    drop_features: list[str] = ["Name"]
    test_size: float = 0.2
    random_state: int = 42


class AppConfig(BaseModel):
    paths: Paths = Paths()
    train: TrainConfig = TrainConfig()

    @staticmethod
    def ensure_dirs(paths: Paths) -> None:
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        paths.data_dir.mkdir(parents=True, exist_ok=True)
