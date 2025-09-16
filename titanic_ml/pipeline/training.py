from __future__ import annotations

"""Training pipeline for Titanic ML application."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..core.config import (DATA_DIR, MODEL_FILE_NAME, PREPROCESSOR_FILE_NAME,
                           TRAIN_FILE_NAME, TrainConfig, ensure_artifacts_dir)
from ..core.data import read_csv_frame
from ..core.features import TARGET_COL, build_preprocessor, select_features
from ..core.model import build_classifier


def train_model(config: TrainConfig | None = None) -> dict[str, float]:
    """Train the model and persist artifacts.

    Args:
        config: Optional TrainConfig. Defaults are used if None.

    Returns:
        dict[str, float]: Metrics including accuracy and ROC-AUC on validation set.
    """

    cfg: TrainConfig = config or TrainConfig()

    # Load data
    train_path: Path = DATA_DIR / TRAIN_FILE_NAME
    df: pd.DataFrame = read_csv_frame(train_path)

    # Split features/target
    y = df[TARGET_COL].astype(int)
    X = select_features(df.drop(columns=[TARGET_COL]))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    clf = build_classifier(cfg)

    pipeline: Pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    acc: float = float(accuracy_score(y_val, y_pred))
    roc_auc: float = float(roc_auc_score(y_val, y_proba))

    # Persist artifacts
    out_dir: Path = ensure_artifacts_dir()
    joblib.dump(pipeline, out_dir / MODEL_FILE_NAME)

    # Also persist standalone preprocessor if needed for batch inference flexibility
    joblib.dump(preprocessor, out_dir / PREPROCESSOR_FILE_NAME)

    return {"accuracy": acc, "roc_auc": roc_auc}
