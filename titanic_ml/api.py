"""FastAPI server for Titanic survival predictions."""

from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from .prediction import ModelPredictor, PredictionConfig


class PassengerData(BaseModel):
    """Input schema for passenger data."""

    PassengerId: int = Field(..., description="Passenger ID")
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Name: str = Field(..., description="Passenger name")
    Sex: str = Field(..., description="Passenger sex (male/female)")
    Age: float = Field(None, ge=0, le=120, description="Passenger age")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Ticket: str = Field(..., description="Ticket number")
    Fare: float = Field(None, ge=0, description="Passenger fare")
    Cabin: str = Field(None, description="Cabin number")
    Embarked: str = Field(None, description="Port of embarkation (C/Q/S)")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    PassengerId: int
    Survived: int = Field(..., description="Predicted survival (0 or 1)")
    Probability: float = Field(..., ge=0, le=1, description="Survival probability")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


def create_app(model_path: Path, preprocessor_path: Path) -> FastAPI:
    """Create FastAPI application with loaded model."""

    app = FastAPI(
        title="Titanic Survival Prediction API",
        description="API for predicting passenger survival on the Titanic",
        version="1.0.0",
    )

    # Load model on startup
    prediction_config = PredictionConfig(
        model_path=model_path, preprocessor_path=preprocessor_path
    )
    predictor = ModelPredictor(prediction_config)

    @app.on_event("startup")
    async def startup_event():
        """Load model artifacts on startup."""
        try:
            predictor.load_artifacts()
            logger.info("Model artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Titanic Survival Prediction API",
            "version": "1.0.0",
            "endpoints": {
                "/predict": "Single passenger prediction",
                "/predict_batch": "Batch passenger predictions",
                "/health": "Health check",
            },
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model_loaded": predictor.model is not None}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(passenger: PassengerData):
        """Predict survival for a single passenger."""
        try:
            # Convert to DataFrame
            data_dict = passenger.dict()
            df = pl.DataFrame([data_dict])

            # Make prediction
            predictions, probabilities = predictor.predict(df)

            return PredictionResponse(
                PassengerId=passenger.PassengerId,
                Survived=int(predictions[0]),
                Probability=float(probabilities[0]),
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.post("/predict_batch", response_model=BatchPredictionResponse)
    async def predict_batch(passengers: List[PassengerData]):
        """Predict survival for multiple passengers."""
        try:
            # Convert to DataFrame
            data_dicts = [passenger.dict() for passenger in passengers]
            df = pl.DataFrame(data_dicts)

            # Make predictions
            predictions, probabilities = predictor.predict(df)

            # Create response
            prediction_responses = [
                PredictionResponse(
                    PassengerId=passenger.PassengerId,
                    Survived=int(pred),
                    Probability=float(prob),
                )
                for passenger, pred, prob in zip(passengers, predictions, probabilities)
            ]

            # Calculate summary
            summary = predictor.get_prediction_summary(predictions)

            return BatchPredictionResponse(
                predictions=prediction_responses, summary=summary
            )

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Batch prediction failed: {str(e)}"
            )

    return app
