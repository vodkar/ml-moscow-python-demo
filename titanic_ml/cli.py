"""Command-line interface for the Titanic ML application."""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from .data import DataConfig, DataLoader
from .models import ModelConfig, ModelTrainer
from .prediction import ModelPredictor, PredictionConfig
from .preprocessing import DataPreprocessor, PreprocessingConfig

app = typer.Typer(help="Titanic survival prediction ML application")
console = Console()


@app.command()
def train(
    data_dir: Path = typer.Option(
        Path("data"), help="Directory containing training data"
    ),
    model_type: str = typer.Option(
        "xgboost", help="Model type: xgboost or random_forest"
    ),
    optimize: bool = typer.Option(True, help="Whether to optimize hyperparameters"),
    trials: int = typer.Option(100, help="Number of optimization trials"),
    output_dir: Path = typer.Option(
        Path("artifacts"), help="Directory to save model artifacts"
    ),
) -> None:
    """Train the Titanic survival prediction model."""
    console.print("[bold green]Starting model training...[/bold green]")

    try:
        # Configure logging
        logger.add("training.log", rotation="10 MB")

        # Load data
        data_config = DataConfig(data_dir=data_dir)
        data_loader = DataLoader(data_config)
        train_df, _ = data_loader.load_data()

        console.print(f"[blue]Loaded training data: {train_df.shape[0]:,} rows[/blue]")

        # Preprocess data
        preprocessing_config = PreprocessingConfig()
        preprocessor = DataPreprocessor(preprocessing_config)
        X_train, y_train = preprocessor.fit_transform(train_df)

        console.print(f"[blue]Preprocessed data: {X_train.shape[1]} features[/blue]")

        # Train model
        model_config = ModelConfig(
            model_type=model_type,
            optimization_trials=trials,
            model_save_path=output_dir / "model.joblib",
            preprocessor_save_path=output_dir / "preprocessor.joblib",
        )
        trainer = ModelTrainer(model_config)

        model, metrics = trainer.train_model(X_train, y_train, optimize=optimize)

        # Save artifacts
        trainer.save_model(preprocessor)

        # Display results
        table = Table(title="Training Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model Type", model_type)
        table.add_row(
            "CV Accuracy", f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
        )
        table.add_row("Hyperparameter Optimization", "Yes" if optimize else "No")
        if optimize:
            table.add_row("Optimization Trials", str(trials))

        console.print(table)
        console.print(
            f"[bold green]Model saved to {model_config.model_save_path}[/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def predict(
    data_path: Path = typer.Option(Path("data/test.csv"), help="Path to test data"),
    model_path: Path = typer.Option(
        Path("artifacts/model.joblib"), help="Path to trained model"
    ),
    preprocessor_path: Path = typer.Option(
        Path("artifacts/preprocessor.joblib"), help="Path to preprocessor"
    ),
    output_path: Path = typer.Option(
        Path("artifacts/predictions.csv"), help="Output path for predictions"
    ),
) -> None:
    """Make predictions on test data."""
    console.print("[bold green]Starting prediction...[/bold green]")

    try:
        # Load test data
        from .data import DataConfig, DataLoader

        data_config = DataConfig(data_dir=data_path.parent, test_file=data_path.name)
        data_loader = DataLoader(data_config)
        _, test_df = data_loader.load_data()

        console.print(f"[blue]Loaded test data: {test_df.shape[0]:,} rows[/blue]")

        # Make predictions
        prediction_config = PredictionConfig(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            output_path=output_path,
        )
        predictor = ModelPredictor(prediction_config)
        results = predictor.predict_and_save(test_df)

        # Display summary
        predictions = results.select("Survived").to_numpy().flatten()
        summary = predictor.get_prediction_summary(predictions)

        table = Table(title="Prediction Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Predictions", str(summary["total_predictions"]))
        table.add_row("Predicted Survived", str(summary["survived_predictions"]))
        table.add_row(
            "Predicted Not Survived", str(summary["not_survived_predictions"])
        )
        table.add_row("Survival Rate", f"{summary['survival_rate']:.2%}")

        console.print(table)
        console.print(f"[bold green]Predictions saved to {output_path}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Prediction failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    data_path: Path = typer.Option(Path("data/train.csv"), help="Path to labeled data"),
    model_path: Path = typer.Option(
        Path("artifacts/model.joblib"), help="Path to trained model"
    ),
    preprocessor_path: Path = typer.Option(
        Path("artifacts/preprocessor.joblib"), help="Path to preprocessor"
    ),
) -> None:
    """Evaluate model performance on labeled data."""
    console.print("[bold green]Starting model evaluation...[/bold green]")

    try:
        # Load data
        from .data import DataConfig, DataLoader

        data_config = DataConfig(data_dir=data_path.parent, train_file=data_path.name)
        data_loader = DataLoader(data_config)
        eval_df, _ = data_loader.load_data()

        # Extract target
        y_true = eval_df.select("Survived").to_series()
        X_eval = eval_df.drop("Survived")

        console.print(f"[blue]Loaded evaluation data: {eval_df.shape[0]:,} rows[/blue]")

        # Evaluate model
        prediction_config = PredictionConfig(
            model_path=model_path, preprocessor_path=preprocessor_path
        )
        predictor = ModelPredictor(prediction_config)
        metrics = predictor.evaluate_model(X_eval, y_true)

        # Display results
        table = Table(title="Model Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        table.add_row("ROC AUC", f"{metrics['roc_auc']:.4f}")
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall", f"{metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")
        table.add_row("Total Samples", str(metrics["n_samples"]))
        table.add_row("Positive Samples", str(metrics["n_positive"]))
        table.add_row("Negative Samples", str(metrics["n_negative"]))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server"),
    port: int = typer.Option(8000, help="Port to bind the server"),
    model_path: Path = typer.Option(
        Path("artifacts/model.joblib"), help="Path to trained model"
    ),
    preprocessor_path: Path = typer.Option(
        Path("artifacts/preprocessor.joblib"), help="Path to preprocessor"
    ),
) -> None:
    """Start the prediction API server."""
    console.print(f"[bold green]Starting API server on {host}:{port}...[/bold green]")

    try:
        import uvicorn

        from .api import create_app

        app_instance = create_app(model_path, preprocessor_path)
        uvicorn.run(app_instance, host=host, port=port)

    except Exception as e:
        console.print(f"[bold red]Server failed to start: {e}[/bold red]")
        raise typer.Exit(1)


def train_command() -> None:
    """Entry point for training command."""
    app()


def predict_command() -> None:
    """Entry point for prediction command."""
    app()


def serve_command() -> None:
    """Entry point for serve command."""
    app()


if __name__ == "__main__":
    app()
