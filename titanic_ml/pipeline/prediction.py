"""Prediction pipeline for the Titanic ML application."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from rich.console import Console

from ..core.config import get_config
from ..core.data import DataProcessor
from ..core.model import ModelTrainer

console = Console()


class PredictionPipeline:
    """Complete prediction pipeline for the Titanic ML application."""

    def __init__(self, model_path: Optional[Path] = None):
        self.config = get_config()
        self.data_processor = DataProcessor(self.config.data)
        self.model_trainer = ModelTrainer(self.config.model)

        # Load trained model
        model_load_path = model_path or self.config.model.model_path
        self.model_trainer.load_model(model_load_path)

    def predict_from_file(
        self, test_file_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """Make predictions on test data from file."""
        console.print(f"[bold blue]Making predictions on {test_file_path}[/bold blue]")

        # Load test data
        test_data = pl.read_csv(test_file_path)
        console.print(f"Loaded {len(test_data)} test samples")

        # Validate test data structure
        self.data_processor.validate_data(test_data, is_training=False)

        # Make predictions
        predictions = self.model_trainer.predict(test_data)
        probabilities = None

        try:
            probabilities = self.model_trainer.predict_proba(test_data)
        except ValueError:
            console.print("Model does not support probability predictions")

        # Create submission dataframe
        passenger_ids = test_data["PassengerId"].to_list()
        submission_df = pl.DataFrame(
            {"PassengerId": passenger_ids, "Survived": predictions.tolist()}
        )

        # Add probabilities if available
        if probabilities is not None:
            submission_df = submission_df.with_columns(
                [pl.Series("Survival_Probability", probabilities[:, 1].tolist())]
            )

        # Save predictions
        if output_path is None:
            output_path = self.config.data.output_dir / "predictions.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df.write_csv(output_path)

        console.print(f"Predictions saved to {output_path}")
        console.print(f"Predicted survival rate: {np.mean(predictions):.3f}")

        return output_path

    def predict_single(self, passenger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single passenger."""
        # Convert to DataFrame
        df = pl.DataFrame([passenger_data])

        # Make prediction
        prediction = self.model_trainer.predict(df)[0]

        try:
            probabilities = self.model_trainer.predict_proba(df)[0]
            survival_probability = probabilities[1]
        except ValueError:
            survival_probability = None

        result = {
            "passenger_id": passenger_data.get("PassengerId", "Unknown"),
            "prediction": int(prediction),
            "survival_probability": float(survival_probability)
            if survival_probability is not None
            else None,
            "model_used": self.model_trainer.best_model_name,
        }

        return result

    def batch_predict(
        self, passenger_list: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Make predictions for multiple passengers."""
        console.print(f"Making batch predictions for {len(passenger_list)} passengers")

        # Convert to DataFrame
        df = pl.DataFrame(passenger_list)

        # Make predictions
        predictions = self.model_trainer.predict(df)

        try:
            probabilities = self.model_trainer.predict_proba(df)
            survival_probabilities = probabilities[:, 1]
        except ValueError:
            survival_probabilities = [None] * len(predictions)

        results = []
        for i, passenger_data in enumerate(passenger_list):
            result = {
                "passenger_id": passenger_data.get("PassengerId", f"Unknown_{i}"),
                "prediction": int(predictions[i]),
                "survival_probability": float(survival_probabilities[i])
                if survival_probabilities[i] is not None
                else None,
                "model_used": self.model_trainer.best_model_name,
            }
            results.append(result)

        return results
