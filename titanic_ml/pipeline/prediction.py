from __future__ import annotations

"""Prediction pipeline for Titanic ML application."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ..core.config import ARTIFACTS_DIR, MODEL_FILE_NAME
from ..core.features import PASSENGER_ID_COL, select_features


def predict_batch(
    input_csv: Path | str, model_path: Path | None = None
) -> pd.DataFrame:
    """Run batch prediction on an input CSV using the persisted model.

    Args:
        input_csv: Path to CSV with Titanic features.
        model_path: Optional explicit model path; falls back to artifacts dir.

    Returns:
        DataFrame: DataFrame with PassengerId and predicted Survived.
    """

    resolved_model_path: Path = model_path or (ARTIFACTS_DIR / MODEL_FILE_NAME)
    pipeline: Pipeline = joblib.load(resolved_model_path)

    df: pd.DataFrame = pd.read_csv(Path(input_csv))
    X = select_features(df)
    preds = pipeline.predict(X)  # numpy array-like
    out_df = pd.DataFrame(
        {
            PASSENGER_ID_COL: df[PASSENGER_ID_COL].astype(int),
            "Survived": pd.Series(preds, dtype=int),
        }
    )
    return out_df
