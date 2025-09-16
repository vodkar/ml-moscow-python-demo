from pathlib import Path

import polars as pl

from titanic_ml.core.config import AppConfig
from titanic_ml.core.data import load_train_data, train_valid_split
from titanic_ml.core.features import FeatureEncoder
from titanic_ml.core.model import TitanicModel, evaluate


def test_end_to_end_training():
    cfg = AppConfig()
    X, y = load_train_data(Path("data/train.csv"), cfg.train.target)
    Xtr, Xva, ytr, yva = train_valid_split(X, y)
    enc = FeatureEncoder(cfg.train.numeric_features, cfg.train.categorical_features, cfg.train.drop_features)
    model = TitanicModel(enc)
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    metrics = evaluate(yva.to_numpy(), pred)
    assert metrics["accuracy"] > 0.7
