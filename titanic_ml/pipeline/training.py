"""Training pipeline for the Titanic ML application."""

from pathlib import Path
from typing import Any, Dict

import polars as pl
from rich.console import Console

from ..core.config import get_config
from ..core.data import DataProcessor
from ..core.features import FeatureEngineer
from ..core.model import ModelTrainer

console = Console()


class TrainingPipeline:
    """Complete training pipeline for the Titanic ML application."""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config = get_config()
        if config_override:
            # Simple config override (in production, use more sophisticated merging)
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.data_processor = DataProcessor(self.config.data)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(self.config.model)
        
    def run(self, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        console.print("[bold blue]Starting Titanic ML Training Pipeline[/bold blue]")
        
        # Create output directory
        self.config.data.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load and validate data
        console.print("\n[bold green]Step 1: Loading and validating data[/bold green]")
        train_data = self.data_processor.load_train_data()
        self.data_processor.validate_data(train_data, is_training=True)
        
        # Get data info
        data_info = self.data_processor.get_data_info(train_data)
        console.print(f"Dataset shape: {data_info['shape']}")
        console.print(f"Survival rate: {data_info.get('survival_rate', 'N/A'):.3f}")
        
        # Step 2: Feature engineering
        console.print("\n[bold green]Step 2: Feature engineering[/bold green]")
        processed_data = self.feature_engineer.fit_transform(train_data)
        
        # Step 3: Split data
        console.print("\n[bold green]Step 3: Splitting data[/bold green]")
        train_df, val_df = self.data_processor.split_data(processed_data)
        
        # Step 4: Train models
        console.print("\n[bold green]Step 4: Training models[/bold green]")
        # Transfer the fitted feature engineer to the model trainer
        self.model_trainer.feature_engineer = self.feature_engineer
        training_results = self.model_trainer.train_models(
            train_df, val_df, tune_hyperparameters=tune_hyperparameters
        )
        
        # Step 5: Feature importance
        console.print("\n[bold green]Step 5: Analyzing feature importance[/bold green]")
        feature_names = self.feature_engineer.get_feature_names()
        feature_importance = self.model_trainer.generate_feature_importance(feature_names)
        
        # Step 6: Save model
        console.print("\n[bold green]Step 6: Saving model[/bold green]")
        model_path = self.model_trainer.save_model()
        
        # Prepare results
        results = {
            "data_info": data_info,
            "training_results": training_results,
            "feature_importance": feature_importance,
            "model_path": str(model_path),
            "best_model": self.model_trainer.best_model_name,
            "config": self.config
        }
        
        console.print(f"\n[bold blue]Training completed! Best model: {self.model_trainer.best_model_name}[/bold blue]")
        console.print(f"Model saved to: {model_path}")
        
        return results