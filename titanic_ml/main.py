from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from .core.config import AppConfig
from .core.data import load_test_data, load_train_data, train_valid_split
from .core.features import FeatureEncoder
from .core.model import TitanicModel, evaluate

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def train(
    train_csv: Path = typer.Option(Path("data/train.csv"), help="Path to train CSV"),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"), help="Where to save artifacts"
    ),
):
    cfg = AppConfig()
    cfg.paths.train_csv = train_csv
    cfg.paths.artifacts_dir = artifacts_dir
    AppConfig.ensure_dirs(cfg.paths)

    console.rule("Load data")
    X, y = load_train_data(cfg.paths.train_csv, cfg.train.target)
    Xtr, Xva, ytr, yva = train_valid_split(
        X, y, test_size=cfg.train.test_size, random_state=cfg.train.random_state
    )

    console.rule("Fit features + model")
    encoder = FeatureEncoder(
        cfg.train.numeric_features,
        cfg.train.categorical_features,
        cfg.train.drop_features,
    )
    model = TitanicModel(encoder)
    model.fit(Xtr, ytr)

    console.rule("Validate")
    pred = model.predict(Xva)
    proba = None
    try:
        proba = model.predict_proba(Xva)
    except Exception:
        pass
    metrics = evaluate(yva.to_numpy(), pred, proba)

    console.print(metrics)
    console.rule("Save artifacts")
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = cfg.paths.artifacts_dir / "feature_columns.json"
    model_path = cfg.paths.artifacts_dir / "model.joblib"
    encoder.dump(encoder_path)
    model.save(model_path)
    console.print(f"Saved encoder -> {encoder_path}")
    console.print(f"Saved model   -> {model_path}")


@app.command()
def evaluate_cmd(
    train_csv: Path = typer.Option(Path("data/train.csv"), help="Path to train CSV"),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"), help="Artifacts dir with model"
    ),
):
    cfg = AppConfig()
    cfg.paths.train_csv = train_csv
    cfg.paths.artifacts_dir = artifacts_dir
    AppConfig.ensure_dirs(cfg.paths)

    X, y = load_train_data(cfg.paths.train_csv, cfg.train.target)
    Xtr, Xva, ytr, yva = train_valid_split(
        X, y, test_size=cfg.train.test_size, random_state=cfg.train.random_state
    )
    encoder = FeatureEncoder.load(cfg.paths.artifacts_dir / "feature_columns.json")
    model = TitanicModel.load(cfg.paths.artifacts_dir / "model.joblib", encoder)
    pred = model.predict(Xva)
    proba = None
    try:
        proba = model.predict_proba(Xva)
    except Exception:
        pass
    metrics = evaluate(yva.to_numpy(), pred, proba)
    table = Table(title="Evaluation")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


@app.command()
def predict(
    test_csv: Path = typer.Option(Path("data/test.csv"), help="Path to test CSV"),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"), help="Artifacts dir with model"
    ),
    out_csv: Path = typer.Option(
        Path("artifacts/predictions.csv"), help="Path to save predictions"
    ),
    id_col: str = typer.Option("PassengerId", help="ID column name in test.csv"),
):
    cfg = AppConfig()
    cfg.paths.test_csv = test_csv
    cfg.paths.artifacts_dir = artifacts_dir
    cfg.paths.predictions_path = out_csv
    AppConfig.ensure_dirs(cfg.paths)

    df = load_test_data(cfg.paths.test_csv)
    encoder = FeatureEncoder.load(cfg.paths.artifacts_dir / "feature_columns.json")
    model = TitanicModel.load(cfg.paths.artifacts_dir / "model.joblib", encoder)
    proba = model.predict_proba(df)
    # ensure id exists
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not in test df")
    out = pl.DataFrame({id_col: df[id_col], "Survived": (proba >= 0.5).astype(int)})
    cfg.paths.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(cfg.paths.predictions_path)
    console.print(f"Saved predictions -> {cfg.paths.predictions_path}")


if __name__ == "__main__":
    app()
