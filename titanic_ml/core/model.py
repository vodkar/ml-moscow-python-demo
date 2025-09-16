from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .features import FeatureEncoder


@dataclass
class TrainedArtifacts:
    encoder_path: Path
    model_path: Path


class TitanicModel:
    def __init__(self, encoder: FeatureEncoder | None = None):
        self.encoder = encoder
        self.model = HistGradientBoostingClassifier(
            max_depth=None,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=42,
        )

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "TitanicModel":
        if self.encoder is None:
            raise ValueError("Encoder must be set before training")
        Xenc = self.encoder.fit_transform(X)
        self.model.fit(Xenc.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xenc = self.encoder.transform(X)
        return self.model.predict(Xenc.to_numpy())

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        Xenc = self.encoder.transform(X)
        proba = self.model.predict_proba(Xenc.to_numpy())
        # predict_proba returns (n,2); take class 1 probability
        return proba[:, 1]

    def save(self, model_path: Path) -> None:
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, model_path: Path, encoder: FeatureEncoder) -> "TitanicModel":
        model = joblib.load(model_path)
        inst = cls(encoder)
        inst.model = model
        return inst


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict:
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return metrics
