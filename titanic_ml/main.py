"""Command-line interface for the Titanic ML application."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from .core.config import get_config
from .pipeline.prediction import PredictionPipeline
from .pipeline.training import TrainingPipeline

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Titanic ML: High-performance ML application for Titanic dataset analysis and prediction."""
    pass


@cli.command()
@click.option(
    "--tune-hyperparameters/--no-tune",
    default=True,
    help="Enable or disable hyperparameter tuning"
)
@click.option(
    "--algorithms",
    multiple=True,
    default=["random_forest", "xgboost", "logistic_regression"],
    help="Algorithms to train (can specify multiple)"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for models and results"
)
def train(tune_hyperparameters: bool, algorithms: tuple, output_dir: Optional[Path]):
    """Train machine learning models on the Titanic dataset."""
    console.print("[bold blue]🚢 Starting Titanic ML Training[/bold blue]")
    
    # Prepare config override
    config_override = {}
    if algorithms:
        config_override["algorithms"] = list(algorithms)
    if output_dir:
        config_override["output_dir"] = output_dir
    
    try:
        # Initialize and run training pipeline
        pipeline = TrainingPipeline(config_override)
        results = pipeline.run(tune_hyperparameters=tune_hyperparameters)
        
        # Display results summary
        _display_training_results(results)
        
        console.print(f"\\n[bold green]✅ Training completed successfully![/bold green]")
        console.print(f"Best model: {results['best_model']}")
        console.print(f"Model saved to: {results['model_path']}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Training failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--test-file",
    type=click.Path(exists=True, path_type=Path),
    default="data/test.csv",
    help="Path to test data file"
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model file"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for predictions"
)
def predict(test_file: Path, model_path: Optional[Path], output: Optional[Path]):
    """Make predictions on test data."""
    console.print("[bold blue]🔮 Making Predictions[/bold blue]")
    
    try:
        # Initialize prediction pipeline
        pipeline = PredictionPipeline(model_path)
        
        # Make predictions
        output_path = pipeline.predict_from_file(test_file, output)
        
        console.print(f"\\n[bold green]✅ Predictions completed![/bold green]")
        console.print(f"Results saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Prediction failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--pclass",
    type=click.IntRange(1, 3),
    required=True,
    help="Passenger class (1, 2, or 3)"
)
@click.option(
    "--sex",
    type=click.Choice(["male", "female"]),
    required=True,
    help="Sex of passenger"
)
@click.option(
    "--age",
    type=float,
    required=True,
    help="Age of passenger"
)
@click.option(
    "--sibsp",
    type=int,
    default=0,
    help="Number of siblings/spouses aboard"
)
@click.option(
    "--parch",
    type=int,
    default=0,
    help="Number of parents/children aboard"
)
@click.option(
    "--fare",
    type=float,
    required=True,
    help="Passenger fare"
)
@click.option(
    "--embarked",
    type=click.Choice(["S", "C", "Q"]),
    default="S",
    help="Port of embarkation"
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model file"
)
def predict_single(pclass: int, sex: str, age: float, sibsp: int, parch: int, 
                  fare: float, embarked: str, model_path: Optional[Path]):
    """Make prediction for a single passenger."""
    console.print("[bold blue]👤 Single Passenger Prediction[/bold blue]")
    
    # Prepare passenger data
    passenger_data = {
        "PassengerId": 999999,  # Dummy ID
        "Pclass": pclass,
        "Name": "Doe, Mr. John",  # Dummy name
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "DUMMY",  # Dummy ticket
        "Fare": fare,
        "Cabin": None,
        "Embarked": embarked
    }
    
    try:
        # Initialize prediction pipeline
        pipeline = PredictionPipeline(model_path)
        
        # Make prediction
        result = pipeline.predict_single(passenger_data)
        
        # Display result
        _display_single_prediction(passenger_data, result)
        
    except Exception as e:
        console.print(f"[bold red]❌ Prediction failed: {e}[/bold red]")
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--data-file",
    type=click.Path(exists=True, path_type=Path),
    default="data/train.csv",
    help="Path to data file for analysis"
)
def analyze(data_file: Path):
    """Analyze the Titanic dataset."""
    console.print("[bold blue]📊 Dataset Analysis[/bold blue]")
    
    try:
        import polars as pl

        # Load data
        df = pl.read_csv(data_file)
        console.print(f"\\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic statistics
        _display_data_analysis(df)
        
    except Exception as e:
        console.print(f"[bold red]❌ Analysis failed: {e}[/bold red]")
        raise click.ClickException(str(e))


def _display_training_results(results: Dict[str, Any]) -> None:
    """Display training results in a formatted table."""
    table = Table(title="Training Results")
    table.add_column("Model", style="cyan")
    table.add_column("Validation Accuracy", style="green")
    table.add_column("CV Mean", style="yellow")
    table.add_column("CV Std", style="magenta")
    
    training_results = results.get("training_results", {})
    for model_name, model_results in training_results.items():
        val_acc = model_results.get("val_accuracy", 0)
        cv_mean = model_results.get("train_metrics", {}).get("cv_mean", 0)
        cv_std = model_results.get("train_metrics", {}).get("cv_std", 0)
        
        table.add_row(
            model_name,
            f"{val_acc:.4f}",
            f"{cv_mean:.4f}",
            f"{cv_std:.4f}"
        )
    
    console.print(table)
    
    # Display feature importance
    feature_importance = results.get("feature_importance", {})
    if feature_importance:
        console.print("\\n[bold]Top 5 Most Important Features:[/bold]")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            console.print(f"{i+1}. {feature}: {importance:.4f}")


def _display_single_prediction(passenger_data: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Display single passenger prediction result."""
    # Passenger info table
    info_table = Table(title="Passenger Information")
    info_table.add_column("Attribute", style="cyan")
    info_table.add_column("Value", style="white")
    
    display_fields = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    for field in display_fields:
        info_table.add_row(field, str(passenger_data[field]))
    
    console.print(info_table)
    
    # Prediction result
    prediction = "Survived" if result["prediction"] == 1 else "Did not survive"
    prob = result.get("survival_probability")
    
    console.print(f"\\n[bold]Prediction:[/bold] {prediction}")
    if prob is not None:
        console.print(f"[bold]Survival Probability:[/bold] {prob:.3f}")
    console.print(f"[bold]Model Used:[/bold] {result['model_used']}")


def _display_data_analysis(df) -> None:
    """Display basic data analysis."""
    # Import polars for type hints
    import polars as pl

    # Basic info
    console.print(f"Dataset shape: {df.shape}")
    
    # Missing values
    console.print("\\n[bold]Missing Values:[/bold]")
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            console.print(f"  {col}: {null_count}")
    
    # Survival statistics (if present)
    if "Survived" in df.columns:
        survival_rate = df["Survived"].mean()
        console.print(f"\\n[bold]Survival Rate:[/bold] {survival_rate:.3f}")
        
        # Survival by class
        survival_by_class = df.group_by("Pclass").agg([
            pl.col("Survived").mean().alias("survival_rate"),
            pl.col("Survived").count().alias("count")
        ]).sort("Pclass")
        
        console.print("\\n[bold]Survival by Class:[/bold]")
        for row in survival_by_class.iter_rows(named=True):
            console.print(f"  Class {row['Pclass']}: {row['survival_rate']:.3f} ({row['count']} passengers)")
    
    # Age statistics
    if "Age" in df.columns:
        age_stats = df["Age"].describe()
        console.print(f"\\n[bold]Age Statistics:[/bold]")
        console.print(f"  Mean: {df['Age'].mean():.1f}")
        console.print(f"  Median: {df['Age'].median():.1f}")
        console.print(f"  Min: {df['Age'].min():.1f}")
        console.print(f"  Max: {df['Age'].max():.1f}")


if __name__ == "__main__":
    cli()