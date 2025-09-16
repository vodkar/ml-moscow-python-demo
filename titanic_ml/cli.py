"""Command-line interface for Titanic ML application."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from titanic_ml.data.loader import DEFAULT_DATA_CONFIG, DataConfig, DataLoader
from titanic_ml.models.ensemble import DEFAULT_MODEL_CONFIG, ModelConfig, TitanicEnsemble
from titanic_ml.preprocessing.features import DEFAULT_FEATURE_CONFIG, FeatureConfig, TitanicFeatureEngineer
from titanic_ml.prediction.predictor import DEFAULT_PREDICTION_CONFIG, PredictionConfig, TitanicPredictor
from titanic_ml.training.trainer import DEFAULT_TRAINING_CONFIG, ModelTrainer, TrainingConfig

app = typer.Typer(
    name="titanic-ml",
    help="High-performance ML application for Titanic survival prediction",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def train(
    data_path: Annotated[Optional[str], typer.Option(help="Path to data directory")] = None,
    model_output: Annotated[Optional[str], typer.Option(help="Path to save trained model")] = None,
    config_file: Annotated[Optional[str], typer.Option(help="Path to configuration JSON file")] = None,
    test_size: Annotated[Optional[float], typer.Option(help="Test set size (0.1-0.5)")] = None,
    use_polars: Annotated[bool, typer.Option(help="Use Polars for faster data loading")] = True,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
) -> None:
    """Train the Titanic survival prediction model."""
    console.print("[bold green]Starting Titanic ML Model Training[/bold green]")
    
    # Load configuration
    if config_file:
        with open(config_file, 'r') as config_json:
            config_data = json.load(config_json)
        data_config = DataConfig(**config_data.get("data", {}))
        model_config = ModelConfig(**config_data.get("model", {}))
        feature_config = FeatureConfig(**config_data.get("features", {}))
        training_config = TrainingConfig(**config_data.get("training", {}))
    else:
        data_config = DEFAULT_DATA_CONFIG
        model_config = DEFAULT_MODEL_CONFIG
        feature_config = DEFAULT_FEATURE_CONFIG
        training_config = DEFAULT_TRAINING_CONFIG

    # Override with command line arguments
    if data_path:
        data_config.data_path = Path(data_path)
    if model_output:
        training_config.save_model_path = model_output
    if test_size:
        training_config.test_size = test_size
    data_config.use_polars = use_polars

    # Initialize components
    data_loader = DataLoader(data_config)
    feature_engineer = TitanicFeatureEngineer(feature_config)
    model = TitanicEnsemble(model_config)
    trainer = ModelTrainer(data_loader, feature_engineer, model, training_config)

    # Train model with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training model...", total=None)
        
        try:
            training_results = trainer.train()
            progress.update(task, description="Training completed!")
        except Exception as training_exception:
            console.print(f"[bold red]Training failed: {training_exception}[/bold red]")
            raise typer.Exit(1) from training_exception

    # Display results
    if verbose:
        _display_training_results(training_results)
    else:
        validation_accuracy = training_results["validation_metrics"]["accuracy"]
        test_accuracy = training_results["test_metrics"]["accuracy"]
        console.print(f"✅ Training completed!")
        console.print(f"📊 Validation accuracy: {validation_accuracy:.3f}")
        console.print(f"🎯 Test accuracy: {test_accuracy:.3f}")

    console.print(f"💾 Model saved to: {training_config.save_model_path}")


@app.command()
def predict(
    input_file: Annotated[str, typer.Argument(help="Path to input CSV file")],
    output_file: Annotated[Optional[str], typer.Option(help="Path to save predictions")] = None,
    model_path: Annotated[Optional[str], typer.Option(help="Path to trained model")] = None,
    probabilities: Annotated[bool, typer.Option(help="Include prediction probabilities")] = False,
    confidence_threshold: Annotated[Optional[float], typer.Option(help="Confidence threshold (0.0-1.0)")] = None,
) -> None:
    """Make predictions on new data."""
    console.print("[bold blue]Making Titanic Survival Predictions[/bold blue]")
    
    # Configure predictor
    prediction_config = DEFAULT_PREDICTION_CONFIG
    if model_path:
        prediction_config.model_path = model_path
    if probabilities:
        prediction_config.output_probabilities = True
    if confidence_threshold:
        prediction_config.confidence_threshold = confidence_threshold

    predictor = TitanicPredictor(prediction_config)
    
    # Load models
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        
        try:
            predictor.load_models()
            progress.update(task, description="Models loaded!")
        except Exception as load_exception:
            console.print(f"[bold red]Failed to load models: {load_exception}[/bold red]")
            raise typer.Exit(1) from load_exception

    # Make predictions
    if not output_file:
        output_file = str(Path(input_file).with_suffix(".predictions.csv"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Making predictions...", total=None)
        
        try:
            predictor.batch_predict(input_file, output_file)
            progress.update(task, description="Predictions completed!")
        except Exception as prediction_exception:
            console.print(f"[bold red]Prediction failed: {prediction_exception}[/bold red]")
            raise typer.Exit(1) from prediction_exception

    console.print(f"✅ Predictions saved to: {output_file}")


@app.command()
def single_predict(
    pclass: Annotated[int, typer.Option(help="Passenger class (1, 2, or 3)")],
    sex: Annotated[str, typer.Option(help="Gender (male/female)")],
    age: Annotated[float, typer.Option(help="Age in years")],
    sibsp: Annotated[int, typer.Option(help="Number of siblings/spouses aboard")] = 0,
    parch: Annotated[int, typer.Option(help="Number of parents/children aboard")] = 0,
    fare: Annotated[float, typer.Option(help="Passenger fare")] = 32.0,
    embarked: Annotated[str, typer.Option(help="Port of embarkation (C/Q/S)")] = "S",
    model_path: Annotated[Optional[str], typer.Option(help="Path to trained model")] = None,
) -> None:
    """Make prediction for a single passenger."""
    console.print("[bold cyan]Single Passenger Prediction[/bold cyan]")
    
    # Configure predictor
    prediction_config = DEFAULT_PREDICTION_CONFIG
    prediction_config.output_probabilities = True
    if model_path:
        prediction_config.model_path = model_path

    predictor = TitanicPredictor(prediction_config)
    
    # Load models
    try:
        predictor.load_models()
    except Exception as load_exception:
        console.print(f"[bold red]Failed to load models: {load_exception}[/bold red]")
        raise typer.Exit(1) from load_exception

    # Prepare passenger data
    passenger_data = {
        "Pclass": pclass,
        "Sex": sex.lower(),
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked.upper()
    }

    # Make prediction
    try:
        result = predictor.predict_single(passenger_data)
    except Exception as prediction_exception:
        console.print(f"[bold red]Prediction failed: {prediction_exception}[/bold red]")
        raise typer.Exit(1) from prediction_exception

    # Display results
    survived = "Yes" if result["Survived"] == 1 else "No"
    survival_prob = result.get("Survival_Probability", 0.5)
    confidence = result.get("Confidence", 0.5)
    
    console.print(f"🚢 Passenger Details:")
    console.print(f"   Class: {pclass}, Sex: {sex}, Age: {age}")
    console.print(f"   Family: {sibsp} siblings/spouses, {parch} parents/children")
    console.print(f"   Fare: ${fare:.2f}, Embarked: {embarked}")
    console.print()
    console.print(f"📊 Prediction: [bold]{'Survived' if result['Survived'] == 1 else 'Did not survive'}[/bold]")
    console.print(f"🎯 Survival probability: {survival_prob:.1%}")
    console.print(f"🔍 Confidence: {confidence:.1%}")


@app.command()
def info(
    model_path: Annotated[Optional[str], typer.Option(help="Path to trained model")] = None,
) -> None:
    """Display information about the trained model."""
    console.print("[bold magenta]Model Information[/bold magenta]")
    
    prediction_config = DEFAULT_PREDICTION_CONFIG
    if model_path:
        prediction_config.model_path = model_path

    predictor = TitanicPredictor(prediction_config)
    
    try:
        predictor.load_models()
        model_info = predictor.get_model_info()
    except Exception as load_exception:
        console.print(f"[bold red]Failed to load model: {load_exception}[/bold red]")
        raise typer.Exit(1) from load_exception

    console.print(f"Model Type: {model_info['model_type']}")
    console.print(f"Model Status: {'✅ Fitted' if model_info['is_fitted'] else '❌ Not fitted'}")
    
    if model_info['base_models']:
        table = Table(title="Base Models")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        
        for base_model in model_info['base_models']:
            table.add_row(base_model['name'], base_model['type'])
        
        console.print(table)


def _display_training_results(results: dict) -> None:
    """Display detailed training results."""
    console.print("\n[bold green]Training Results[/bold green]")
    
    # Dataset info
    dataset_info = results["dataset_info"]
    console.print(f"📊 Dataset: {dataset_info['total_samples']} total samples")
    console.print(f"   Train: {dataset_info['training_samples']}, "
                  f"Validation: {dataset_info['validation_samples']}, "
                  f"Test: {dataset_info['test_samples']}")
    console.print(f"   Features: {dataset_info['feature_count']}")
    
    # Performance metrics
    val_metrics = results["validation_metrics"]
    test_metrics = results["test_metrics"]
    cv_results = results["cross_validation"]
    
    table = Table(title="Model Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Validation", style="green")
    table.add_column("Test", style="yellow")
    table.add_column("CV Mean ± Std", style="magenta")
    
    table.add_row(
        "Accuracy",
        f"{val_metrics['accuracy']:.3f}",
        f"{test_metrics['accuracy']:.3f}",
        f"{cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}"
    )
    table.add_row(
        "Precision",
        f"{val_metrics['precision']:.3f}",
        f"{test_metrics['precision']:.3f}",
        "-"
    )
    table.add_row(
        "Recall",
        f"{val_metrics['recall']:.3f}",
        f"{test_metrics['recall']:.3f}",
        "-"
    )
    table.add_row(
        "F1-Score",
        f"{val_metrics['f1_score']:.3f}",
        f"{test_metrics['f1_score']:.3f}",
        "-"
    )
    
    console.print(table)


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()