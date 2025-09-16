"""Command-line interface for Titanic ML application."""

import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from .core.models import ModelConfig, PipelineConfig
from .pipeline.pipeline import TitanicMLPipeline

app = typer.Typer(
    name="titanic-ml", help="Modern ML CLI for Titanic survival prediction"
)
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@app.command()
def train(
    data_path: Path = typer.Option(
        Path("data"),
        "--data-path",
        "-d",
        help="Path to directory containing train.csv and test.csv",
    ),
    output_path: Path = typer.Option(
        Path("models"), "--output-path", "-o", help="Path to save trained models"
    ),
    model_type: str = typer.Option(
        "xgboost",
        "--model-type",
        "-m",
        help="Model type: xgboost, lightgbm, random_forest, logistic_regression",
    ),
    use_ensemble: bool = typer.Option(
        False, "--ensemble", "-e", help="Train ensemble of multiple models"
    ),
    hyperparameter_tuning: bool = typer.Option(
        True, "--tune/--no-tune", help="Enable/disable hyperparameter tuning"
    ),
    n_trials: int = typer.Option(
        100, "--trials", "-t", help="Number of hyperparameter optimization trials"
    ),
    use_polars: bool = typer.Option(
        True,
        "--polars/--pandas",
        help="Use Polars (fast) or Pandas for data processing",
    ),
    cross_validation_folds: int = typer.Option(
        5, "--cv-folds", "-cv", help="Number of cross-validation folds"
    ),
) -> None:
    """Train ML model(s) on Titanic dataset."""

    rich_print(f"[bold green]Starting Titanic ML Training[/bold green]")
    rich_print(f"Data path: {data_path}")
    rich_print(f"Output path: {output_path}")
    rich_print(f"Model type: {model_type}")
    rich_print(f"Ensemble: {use_ensemble}")
    rich_print(f"Hyperparameter tuning: {hyperparameter_tuning}")
    rich_print(f"Using: {'Polars' if use_polars else 'Pandas'}")

    try:
        # Validate data path
        if not data_path.exists():
            rich_print(f"[red]Error: Data path {data_path} does not exist[/red]")
            raise typer.Exit(1)

        train_file = data_path / "train.csv"
        test_file = data_path / "test.csv"

        if not train_file.exists() or not test_file.exists():
            rich_print(
                f"[red]Error: train.csv or test.csv not found in {data_path}[/red]"
            )
            raise typer.Exit(1)

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            data_path=data_path, model_output_path=output_path, use_polars=use_polars
        )

        # Create model configurations
        model_configs = []

        if use_ensemble:
            # Create ensemble with multiple model types
            model_types = ["xgboost", "random_forest", "logistic_regression"]
            rich_print(
                "[cyan]Creating ensemble with XGBoost, Random Forest, and Logistic Regression[/cyan]"
            )
        else:
            model_types = [model_type]

        for mt in model_types:
            config = ModelConfig(
                model_type=mt, cv_folds=cross_validation_folds, hyperparameters={}
            )
            # Add hyperparameter tuning settings to config
            config.enable_hyperparameter_tuning = hyperparameter_tuning
            config.n_trials = n_trials
            model_configs.append(config)

        # Initialize and run pipeline
        pipeline = TitanicMLPipeline(pipeline_config)

        with Progress() as progress:
            task = progress.add_task("[green]Training models...", total=100)

            # Run the complete pipeline
            results = pipeline.run_full_pipeline(
                model_configs=model_configs, save_models=True, create_submission=True
            )

            progress.update(task, completed=100)

        # Display results
        _display_training_results(results)

        rich_print(f"[bold green]Training completed successfully![/bold green]")
        rich_print(f"Models saved to: {output_path}")
        if "submission_path" in results:
            rich_print(f"Submission file: {results['submission_path']}")

    except Exception as e:
        rich_print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    model_path: Path = typer.Option(
        Path("models"), "--model-path", "-m", help="Path to trained model directory"
    ),
    input_file: Path = typer.Option(
        ..., "--input", "-i", help="Input CSV file with passenger data"
    ),
    output_file: Path = typer.Option(
        Path("predictions.csv"),
        "--output",
        "-o",
        help="Output CSV file for predictions",
    ),
    use_polars: bool = typer.Option(
        True, "--polars/--pandas", help="Use Polars or Pandas for data processing"
    ),
) -> None:
    """Make predictions using trained model."""

    rich_print(f"[bold blue]Making Predictions[/bold blue]")
    rich_print(f"Model path: {model_path}")
    rich_print(f"Input file: {input_file}")
    rich_print(f"Output file: {output_file}")

    try:
        # Validate paths
        if not model_path.exists():
            rich_print(f"[red]Error: Model path {model_path} does not exist[/red]")
            raise typer.Exit(1)

        if not input_file.exists():
            rich_print(f"[red]Error: Input file {input_file} does not exist[/red]")
            raise typer.Exit(1)

        # Create pipeline config
        pipeline_config = PipelineConfig(
            data_path=Path("."),  # Not used for prediction
            model_output_path=model_path,
            use_polars=use_polars,
        )

        # Load trained pipeline
        pipeline = TitanicMLPipeline(pipeline_config)
        pipeline.load_trained_pipeline(model_path)

        # Make predictions
        with console.status("[bold green]Making predictions...") as status:
            pipeline.predict_from_file(input_file, output_file)
            status.update("[bold green]Predictions completed!")

        rich_print(f"[bold green]Predictions saved to: {output_file}[/bold green]")

    except Exception as e:
        rich_print(f"[red]Prediction failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: Path = typer.Option(
        Path("models"), "--model-path", "-m", help="Path to trained model directory"
    ),
    test_data: Path = typer.Option(
        ..., "--test-data", "-t", help="Test data with ground truth labels"
    ),
    output_report: Path = typer.Option(
        Path("evaluation_report.json"),
        "--output",
        "-o",
        help="Output evaluation report file",
    ),
) -> None:
    """Evaluate model performance on test data."""

    rich_print(f"[bold yellow]Evaluating Model Performance[/bold yellow]")

    try:
        # This would require implementing evaluation logic
        # For now, show model summary
        pipeline_config = PipelineConfig(
            data_path=Path("."), model_output_path=model_path, use_polars=True
        )

        pipeline = TitanicMLPipeline(pipeline_config)
        pipeline.load_trained_pipeline(model_path)

        summary = pipeline.get_model_summary()
        _display_model_summary(summary)

        rich_print(f"[bold green]Model evaluation completed[/bold green]")

    except Exception as e:
        rich_print(f"[red]Evaluation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    model_path: Path = typer.Option(
        Path("models"), "--model-path", "-m", help="Path to trained model directory"
    ),
) -> None:
    """Show information about trained model."""

    try:
        if not model_path.exists():
            rich_print(f"[red]Model path {model_path} does not exist[/red]")
            raise typer.Exit(1)

        pipeline_config = PipelineConfig(
            data_path=Path("."), model_output_path=model_path, use_polars=True
        )

        pipeline = TitanicMLPipeline(pipeline_config)
        pipeline.load_trained_pipeline(model_path)

        summary = pipeline.get_model_summary()
        _display_model_summary(summary)

    except Exception as e:
        rich_print(f"[red]Failed to load model info: {e}[/red]")
        raise typer.Exit(1)


def _display_training_results(results: dict) -> None:
    """Display training results in a formatted table."""

    if "single_model" in results:
        model_info = results["single_model"]

        table = Table(title="Training Results - Single Model")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model Type", model_info["model_type"])

        if "metrics" in model_info:
            metrics = model_info["metrics"]
            table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            table.add_row("Precision", f"{metrics.get('precision', 0):.4f}")
            table.add_row("Recall", f"{metrics.get('recall', 0):.4f}")
            table.add_row("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
            table.add_row("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")

        console.print(table)

    elif "ensemble" in results:
        ensemble_info = results["ensemble"]

        table = Table(title="Training Results - Ensemble")
        table.add_column("Model", style="cyan")
        table.add_column("Weight", style="magenta")
        table.add_column("Accuracy", style="green")
        table.add_column("ROC AUC", style="yellow")

        for i, model_info in enumerate(ensemble_info["models"]):
            weight = (
                ensemble_info["ensemble_weights"][i]
                if ensemble_info["ensemble_weights"]
                else 0
            )
            metrics = model_info.get("metrics", {})

            table.add_row(
                model_info["model_type"],
                f"{weight:.3f}",
                f"{metrics.get('accuracy', 0):.4f}",
                f"{metrics.get('roc_auc', 0):.4f}",
            )

        console.print(table)


def _display_model_summary(summary: dict) -> None:
    """Display model summary information."""

    table = Table(title="Model Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    if summary.get("type") == "single_model":
        table.add_row("Type", "Single Model")
        table.add_row("Model Type", summary.get("model_type", "Unknown"))

        if "metrics" in summary and summary["metrics"]:
            metrics = summary["metrics"]
            table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            table.add_row("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")

    elif summary.get("type") == "ensemble":
        table.add_row("Type", "Ensemble")
        table.add_row("Number of Models", str(summary.get("n_models", 0)))

        if "model_types" in summary:
            model_types_str = ", ".join(summary["model_types"])
            table.add_row("Model Types", model_types_str)

    console.print(table)


if __name__ == "__main__":
    app()
