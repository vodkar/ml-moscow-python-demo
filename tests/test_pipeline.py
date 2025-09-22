from __future__ import annotations

from pathlib import Path

from titanic_ml.core.config import DATA_DIR, TrainConfig
from titanic_ml.pipeline.prediction import predict_batch
from titanic_ml.pipeline.training import train_model


def test_training_and_prediction(tmp_path: Path) -> None:
    # Train
    metrics = train_model(TrainConfig(n_estimators=50, test_size=0.25, random_state=0))
    assert metrics["accuracy"] > 0.6

    # Predict on provided test.csv
    test_csv = DATA_DIR / "test.csv"
    assert test_csv.exists()
    preds = predict_batch(test_csv)
    assert {"PassengerId", "Survived"}.issubset(set(preds.columns))
    assert len(preds) > 0
