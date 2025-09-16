"""Command-line interface for the Titanic ML application."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from titanic_ml.config import AppConfig
from titanic_ml.pipeline.predict import PredictionPipeline
from titanic_ml.pipeline.train import TrainingPipeline
from titanic_ml.utils.logging import setup_logger

app = typer.Typer(
    name="titanic-ml",
    help="Modern ML application for Titanic survival prediction",
    add_completion=False,
)
console = Console()

# Global configuration
global_config: AppConfig | None = None


def get_config(config_file: Optional[Path] = None) -> AppConfig:
    """Get or create application configuration."""
    global global_config
    
    if global_config is None:
        if config_file and config_file.exists():
            # Load from file if provided
            with open(config_file, "r", encoding="utf-8") as file:
                config_data = json.load(file)
                global_config = AppConfig(**config_data)
        else:
            # Use default configuration
            global_config = AppConfig()
    
    return global_config


def init_logging_and_config(log_level: str = "INFO", log_file: Optional[Path] = None, config_file: Optional[Path] = None) -> None:
    """Initialize logging and configuration."""
    # Validate log level
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        log_level = "INFO"
    
    from typing import cast, Literal
    valid_log_level = cast(Literal["DEBUG", "INFO", "WARNING", "ERROR"], log_level)
    setup_logger(
        name="titanic_ml",
        level=valid_log_level,
        log_file=log_file,
        use_rich=True,
    )
    
    # Initialize configuration
    get_config(config_file)


@app.command("train")
def train_model(
    model_type: str = typer.Option(
        "xgboost", "--model", "-m",
        help="Model type: xgboost, random_forest, logistic_regression"
    ),
    train_file: Optional[Path] = typer.Option(
        None, "--train-file", "-t", help="Path to training CSV file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for model artifacts"
    ),
    validation_split: float = typer.Option(
        0.2, "--val-split", help="Validation split ratio (0.1-0.5)"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level"
    ),
) -> None:
    """Train a new model on the Titanic dataset."""
    init_logging_and_config(log_level)
    console.print(Panel.fit("🚢 Starting Titanic Model Training", style="bold blue"))
    
    try:
        # Get configuration
        config = get_config()
        
        # Update model configuration
        config.model.model_type = model_type
        config.model.test_size = validation_split
        
        # Update paths if provided
        if train_file:
            config.data.train_file = train_file
        if output_dir:
            config.training.model_save_path = output_dir / "models"
            config.training.metrics_save_path = output_dir / "metrics"
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(config)
        
        # Validate configuration
        validation_results = pipeline.validate_configuration()
        if not validation_results["is_valid"]:
            console.print("❌ Configuration validation failed:", style="red")
            for error in validation_results["errors"]:
                console.print(f"  • {error}", style="red")
            raise typer.Exit(1)
        
        # Display warnings
        if validation_results["warnings"]:
            console.print("⚠️  Configuration warnings:", style="yellow")
            for warning in validation_results["warnings"]:
                console.print(f"  • {warning}", style="yellow")
        
        # Run training
        console.print(f"🔧 Training {model_type} model...")
        results = pipeline.run_full_pipeline()
        
        # Display results
        display_training_results(results)
        
        console.print("✅ Training completed successfully!", style="green")
        
    except Exception as exception:
        console.print(f"❌ Training failed: {str(exception)}", style="red")
        raise typer.Exit(1)


@app.command("predict")
def predict_data(
    input_file: Path = typer.Argument(..., help="Path to input CSV file"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to output CSV file"
    ),
    model_file: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to trained model file"
    ),
    preprocessor_file: Optional[Path] = typer.Option(
        None, "--preprocessor", "-p", help="Path to preprocessor file"
    ),
    with_analysis: bool = typer.Option(
        True, "--analysis/--no-analysis", help="Include prediction analysis"
    ),
) -> None:
    """Make predictions on new data."""
    console.print(Panel.fit("🔮 Making Titanic Survival Predictions", style="bold green"))
    
    try:
        # Get configuration
        config = get_config()
        
        # Update paths if provided
        if model_file:
            config.prediction.model_path = model_file
        if preprocessor_file:
            config.prediction.preprocessor_path = preprocessor_file
        if output_file:
            config.prediction.output_path = output_file
        else:
            config.prediction.output_path = Path("output") / "predictions.csv"
        
        # Initialize prediction pipeline
        pipeline = PredictionPipeline(config)
        
        # Validate configuration
        validation_results = pipeline.validate_configuration()
        if not validation_results["is_valid"]:
            console.print("❌ Configuration validation failed:", style="red")
            for error in validation_results["errors"]:
                console.print(f"  • {error}", style="red")
            raise typer.Exit(1)
        
        # Validate input file
        input_validation = pipeline.validate_input_file(input_file)
        if not input_validation["is_valid"]:
            console.print("❌ Input validation failed:", style="red")
            for error in input_validation["errors"]:
                console.print(f"  • {error}", style="red")
            raise typer.Exit(1)
        
        # Display warnings
        if input_validation["warnings"]:
            console.print("⚠️  Input warnings:", style="yellow")
            for warning in input_validation["warnings"]:
                console.print(f"  • {warning}", style="yellow")
        
        # Run predictions
        console.print(f"🔮 Making predictions on {input_file}...")
        results = pipeline.run_batch_prediction(
            input_file=input_file,
            output_file=config.prediction.output_path,
            with_analysis=with_analysis,
        )
        
        # Display results
        display_prediction_results(results)
        
        console.print("✅ Predictions completed successfully!", style="green")
        
    except Exception as exception:
        console.print(f"❌ Prediction failed: {str(exception)}", style="red")
        raise typer.Exit(1)


@app.command("single")
def predict_single(
    passenger_id: int = typer.Argument(..., help="Passenger ID"),
    pclass: int = typer.Option(..., "--class", help="Passenger class (1, 2, or 3)"),
    name: str = typer.Option(..., "--name", help="Passenger name"),
    sex: str = typer.Option(..., "--sex", help="Sex (male or female)"),
    age: float = typer.Option(..., "--age", help="Age in years"),
    sibsp: int = typer.Option(0, "--siblings", help="Number of siblings/spouses"),
    parch: int = typer.Option(0, "--parents", help="Number of parents/children"),
    ticket: str = typer.Option("UNKNOWN", "--ticket", help="Ticket number"),
    fare: float = typer.Option(..., "--fare", help="Passenger fare"),
    cabin: Optional[str] = typer.Option(None, "--cabin", help="Cabin number"),
    embarked: str = typer.Option("S", "--embarked", help="Port of embarkation (C, Q, S)"),
    model_file: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to trained model file"
    ),
    preprocessor_file: Optional[Path] = typer.Option(
        None, "--preprocessor", "-p", help="Path to preprocessor file"
    ),
) -> None:
    """Predict survival for a single passenger."""
    console.print(Panel.fit("👤 Single Passenger Prediction", style="bold cyan"))
    
    try:
        # Get configuration
        config = get_config()
        
        # Update paths if provided
        if model_file:
            config.prediction.model_path = model_file
        if preprocessor_file:
            config.prediction.preprocessor_path = preprocessor_file
        
        # Prepare passenger data
        passenger_data = {
            "PassengerId": passenger_id,
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": ticket,
            "Fare": fare,
            "Cabin": cabin or "",
            "Embarked": embarked,
        }
        
        # Initialize prediction pipeline
        pipeline = PredictionPipeline(config)
        
        # Make prediction
        console.print("🔮 Making prediction...")
        result = pipeline.predict_single_sample(passenger_data)
        
        # Display result
        display_single_prediction_result(passenger_data, result)
        
    except Exception as exception:
        console.print(f"❌ Prediction failed: {str(exception)}", style="red")
        raise typer.Exit(1)


@app.command("evaluate")
def evaluate_model(
    predictions_file: Path = typer.Argument(..., help="Path to predictions CSV file"),
    ground_truth_file: Path = typer.Argument(..., help="Path to ground truth CSV file"),
) -> None:
    """Evaluate predictions against ground truth."""
    console.print(Panel.fit("📊 Evaluating Model Performance", style="bold magenta"))
    
    try:
        # Get configuration and initialize pipeline
        config = get_config()
        pipeline = PredictionPipeline(config)
        
        # Compare predictions
        console.print("📊 Comparing predictions with ground truth...")
        results = pipeline.compare_predictions(predictions_file, ground_truth_file)
        
        if "error" in results:
            console.print(f"❌ Evaluation failed: {results['error']}", style="red")
            raise typer.Exit(1)
        
        # Display evaluation results
        display_evaluation_results(results)
        
        console.print("✅ Evaluation completed successfully!", style="green")
        
    except Exception as exception:
        console.print(f"❌ Evaluation failed: {str(exception)}", style="red")
        raise typer.Exit(1)


@app.command("info")
def show_info(
    model_file: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to model file"
    ),
    show_config: bool = typer.Option(
        False, "--config", help="Show configuration details"
    ),
) -> None:
    """Show information about the application or model."""
    console.print(Panel.fit("ℹ️  Titanic ML Information", style="bold blue"))
    
    try:
        config = get_config()
        
        if model_file:
            # Show model information
            config.prediction.model_path = model_file
            pipeline = PredictionPipeline(config)
            model_info = pipeline.get_model_info()
            
            if "error" in model_info:
                console.print(f"❌ Could not load model: {model_info['error']}", style="red")
                return
            
            display_model_info(model_info)
        
        if show_config:
            # Show configuration
            display_config_info(config)
        
        if not model_file and not show_config:
            # Show general application info
            display_app_info()
    
    except Exception as exception:
        console.print(f"❌ Error showing info: {str(exception)}", style="red")
        raise typer.Exit(1)


def display_training_results(results: dict[str, Any]) -> None:
    """Display training results in a formatted table."""
    table = Table(title="Training Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    # Add key metrics
    metrics = results["validation_metrics"]
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("Recall", f"{metrics['recall']:.4f}")
    table.add_row("F1-Score", f"{metrics['f1_score']:.4f}")
    
    # Add cross-validation results
    cv_results = results["cross_validation"]
    table.add_row("CV Mean Score", f"{cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    # Add model info
    model_info = results["model_info"]
    table.add_row("Model Type", model_info["model_type"])
    table.add_row("Feature Count", str(model_info["n_features"]))
    
    console.print(table)
    
    # Show file paths
    console.print(f"📁 Model saved to: {results['model_path']}", style="dim")
    console.print(f"📁 Preprocessor saved to: {results['preprocessor_path']}", style="dim")
    console.print(f"📁 Report saved to: {results['performance_report_path']}", style="dim")


def display_prediction_results(results: dict[str, Any]) -> None:
    """Display prediction results."""
    metadata = results.get("metadata", {})
    analysis = results.get("analysis", {})
    
    table = Table(title="Prediction Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    # Add prediction info
    if metadata:
        table.add_row("Total Passengers", str(metadata.get("n_predictions", "Unknown")))
        table.add_row("Survival Rate", f"{metadata.get('survival_rate', 0):.1%}")
        table.add_row("Model Type", metadata.get("model_type", "Unknown"))
    
    if analysis:
        table.add_row("Predicted Survivors", str(analysis.get("predicted_survivors", "Unknown")))
        table.add_row("Predicted Non-Survivors", str(analysis.get("predicted_non_survivors", "Unknown")))
    
    console.print(table)
    console.print(f"📁 Predictions saved to: {results['predictions_file']}", style="dim")


def display_single_prediction_result(passenger_data: dict[str, Any], result: dict[str, Any]) -> None:
    """Display single prediction result."""
    # Passenger info
    info_table = Table(title="Passenger Information", show_header=True)
    info_table.add_column("Attribute", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Name", passenger_data["Name"])
    info_table.add_row("Class", str(passenger_data["Pclass"]))
    info_table.add_row("Sex", passenger_data["Sex"])
    info_table.add_row("Age", str(passenger_data["Age"]))
    info_table.add_row("Fare", f"${passenger_data['Fare']:.2f}")
    
    console.print(info_table)
    
    # Prediction result
    prediction_panel = Panel.fit(
        f"🔮 Prediction: [bold]{result['survival_prediction']}[/bold]\n"
        + (f"📊 Probability: {result.get('survival_probability', 'N/A'):.1%}\n" if 'survival_probability' in result else "")
        + (f"🎯 Confidence: {result.get('confidence', 'N/A')}" if 'confidence' in result else ""),
        title="Prediction Result",
        style="green" if result["prediction"] == 1 else "red"
    )
    
    console.print(prediction_panel)


def display_evaluation_results(results: dict[str, Any]) -> None:
    """Display evaluation results."""
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total Samples", str(results["total_samples"]))
    table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    table.add_row("Precision", f"{results['precision']:.4f}")
    table.add_row("Recall", f"{results['recall']:.4f}")
    table.add_row("F1-Score", f"{results['f1_score']:.4f}")
    table.add_row("Correct Predictions", str(results["correct_predictions"]))
    table.add_row("Incorrect Predictions", str(results["incorrect_predictions"]))
    
    console.print(table)


def display_model_info(model_info: dict[str, Any]) -> None:
    """Display model information."""
    table = Table(title="Model Information", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Model Type", model_info["model_type"])
    table.add_row("Is Fitted", "✅" if model_info["is_fitted"] else "❌")
    table.add_row("Feature Count", str(model_info["n_features"]))
    table.add_row("Has Feature Importance", "✅" if model_info["has_feature_importance"] else "❌")
    
    console.print(table)


def display_config_info(config: AppConfig) -> None:
    """Display configuration information."""
    console.print("📋 Configuration Details:", style="bold")
    console.print(f"  Training File: {config.data.train_file}", style="dim")
    console.print(f"  Test File: {config.data.test_file}", style="dim")
    console.print(f"  Model Type: {config.model.model_type}", style="dim")
    console.print(f"  Log Level: {config.log_level}", style="dim")


def display_app_info() -> None:
    """Display general application information."""
    console.print("🚢 Titanic ML Application", style="bold blue")
    console.print("Modern machine learning pipeline for Titanic survival prediction")
    console.print("\nAvailable commands:")
    console.print("  • train    - Train a new model")
    console.print("  • predict  - Make batch predictions")
    console.print("  • single   - Predict for single passenger")
    console.print("  • evaluate - Evaluate model performance")
    console.print("  • info     - Show information")
    console.print("\nUse --help with any command for more details.")


if __name__ == "__main__":
    app()