from __future__ import annotations

"""CLI for Titanic ML application using Typer."""

from pathlib import Path
from typing import Optional

import typer

from .core.config import TrainConfig
from .pipeline.prediction import predict_batch
from .pipeline.training import train_model

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def train(
    n_estimators: int = typer.Option(200, help="Number of trees for RandomForest"),
    max_depth: Optional[int] = typer.Option(None, help="Max depth of trees"),
    test_size: float = typer.Option(0.2, help="Validation size fraction"),
    random_state: int = typer.Option(42, help="Random seed"),
) -> None:
    """Train the model and print evaluation metrics."""

    cfg = TrainConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        test_size=test_size,
        random_state=random_state,
    )
    metrics = train_model(cfg)
    typer.echo(
        f"Accuracy: {metrics['accuracy']:.4f}; ROC-AUC: {metrics['roc_auc']:.4f}"
    )


@app.command()
def predict(
    input_csv: Path = typer.Argument(..., exists=True, readable=True),
    output_csv: Path = typer.Option(Path("predictions.csv"), help="Output CSV path"),
) -> None:
    """Run batch prediction and save to CSV."""

    out_df = predict_batch(input_csv)
    out_df.to_csv(output_csv, index=False)
    typer.echo(f"Saved predictions to {output_csv}")


def main() -> None:
    """Entrypoint for console script."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
