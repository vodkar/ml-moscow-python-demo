from __future__ import annotations

"""Configuration and constants for the Titanic ML application.

This module centralizes file paths and hyperparameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"

TRAIN_FILE_NAME: Final[str] = "train.csv"
TEST_FILE_NAME: Final[str] = "test.csv"

# Model file names
MODEL_FILE_NAME: Final[str] = "model.joblib"
PREPROCESSOR_FILE_NAME: Final[str] = "preprocessor.joblib"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training configuration with hyperparameters.

    Attributes:
        n_estimators: Number of trees for RandomForestClassifier.
        max_depth: Maximum tree depth; None means unlimited.
        random_state: Seed for reproducibility.
        n_jobs: Parallel workers for model training.
        test_size: Fraction for validation split.
    """

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42
    n_jobs: int = -1
    test_size: float = 0.2


def ensure_artifacts_dir() -> Path:
    """Ensure artifacts directory exists and return its path.

    Returns:
        Path: Path to artifacts directory.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
