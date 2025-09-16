"""Model training and evaluation system."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from rich.console import Console
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, cross_val_score

from .config import ModelConfig
from .features import FeatureEngineer

console = Console()


class ModelTrainer:
    """Model training and evaluation system."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.feature_engineer = FeatureEngineer()
        
    def get_models(self) -> Dict[str, Any]:
        """Get available models with default parameters."""
        models = {}
        
        if "random_forest" in self.config.algorithms:
            models["random_forest"] = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        
        if "xgboost" in self.config.algorithms:
            try:
                import xgboost as xgb
                models["xgboost"] = xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    eval_metric='logloss'
                )
            except ImportError:
                console.print("Warning: XGBoost is not available. Skipping XGBoost model.")
        
        if "logistic_regression" in self.config.algorithms:
            models["logistic_regression"] = LogisticRegression(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_iter=1000
            )
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """Get hyperparameter grids for tuning."""
        grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["liblinear", "lbfgs"],
                "penalty": ["l2"]
            }
        }
        
        # Only add XGBoost grid if XGBoost is available
        try:
            import xgboost
            grids["xgboost"] = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            }
        except ImportError:
            pass
        
        return grids
    
    def prepare_data(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for modeling."""
        feature_names = self.feature_engineer.get_feature_names()
        
        # Convert to pandas for sklearn compatibility
        df_pandas = df.to_pandas()
        
        # Get features and target
        X = df_pandas[feature_names].values
        y = df_pandas["Survived"].values
        
        return X, y, feature_names
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate model using cross-validation."""
        console.print(f"Evaluating {model_name}...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X, y, 
            cv=self.config.cv_folds, 
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs
        )
        
        # Fit model for additional metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_accuracy": accuracy_score(y, y_pred)
        }
        
        console.print(f"{model_name} - CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        return metrics
    
    def tune_hyperparameters(self, model: Any, X: np.ndarray, y: np.ndarray, 
                           param_grid: Dict[str, List[Any]], model_name: str) -> Any:
        """Tune hyperparameters using grid search."""
        console.print(f"Tuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        console.print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        console.print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_models(self, train_df: pl.DataFrame, val_df: pl.DataFrame, 
                    tune_hyperparameters: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate all models."""
        console.print("Starting model training...")
        
        # Prepare training data
        X_train, y_train, feature_names = self.prepare_data(train_df)
        X_val, y_val, _ = self.prepare_data(val_df)
        
        # Get models and hyperparameter grids
        models = self.get_models()
        param_grids = self.get_hyperparameter_grids()
        
        results = {}
        
        for model_name, model in models.items():
            console.print(f"\n--- Training {model_name} ---")
            
            # Tune hyperparameters if requested
            if tune_hyperparameters and model_name in param_grids:
                tuned_model = self.tune_hyperparameters(
                    model, X_train, y_train, param_grids[model_name], model_name
                )
                self.models[model_name] = tuned_model
            else:
                model.fit(X_train, y_train)
                self.models[model_name] = model
            
            # Evaluate model
            train_metrics = self.evaluate_model(self.models[model_name], X_train, y_train, f"{model_name}_train")
            
            # Validation metrics
            y_val_pred = self.models[model_name].predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            results[model_name] = {
                "model": self.models[model_name],
                "train_metrics": train_metrics,
                "val_accuracy": val_accuracy,
                "feature_names": feature_names
            }
            
            console.print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]["val_accuracy"])
        self.best_model = results[best_model_name]["model"]
        self.best_model_name = best_model_name
        
        console.print(f"\nBest model: {best_model_name} (Val Accuracy: {results[best_model_name]['val_accuracy']:.4f})")
        
        return results
    
    def generate_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Generate feature importance from the best model."""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Get feature importance
        if hasattr(self.best_model, "feature_importances_"):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importance = np.abs(self.best_model.coef_[0])
        else:
            return {}
        
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        console.print("\nTop 10 Feature Importances:")
        for i, (feature, importance) in enumerate(list(sorted_features.items())[:10]):
            console.print(f"{i+1}. {feature}: {importance:.4f}")
        
        return sorted_features
    
    def save_model(self, output_path: Optional[Path] = None) -> Path:
        """Save the best model and feature engineer."""
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        save_path = output_path or self.config.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and feature engineer together
        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_engineer": self.feature_engineer
        }
        
        joblib.dump(model_data, save_path)
        console.print(f"Model saved to {save_path}")
        
        return save_path
    
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load a saved model and feature engineer."""
        load_path = model_path or self.config.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        model_data = joblib.load(load_path)
        self.best_model = model_data["model"]
        self.best_model_name = model_data["model_name"]
        self.feature_engineer = model_data["feature_engineer"]
        
        console.print(f"Model loaded from {load_path}")
    
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.best_model is None:
            raise ValueError("No trained model available for prediction")
        
        # Transform features
        df_processed = self.feature_engineer.transform(df)
        feature_names = self.feature_engineer.get_feature_names()
        
        # Prepare data
        df_pandas = df_processed.to_pandas()
        X = df_pandas[feature_names].values
        
        # Make predictions
        predictions = self.best_model.predict(X)
        
        return predictions
    
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """Make probability predictions on new data."""
        if self.best_model is None:
            raise ValueError("No trained model available for prediction")
        
        if not hasattr(self.best_model, "predict_proba"):
            raise ValueError("Model does not support probability predictions")
        
        # Transform features
        df_processed = self.feature_engineer.transform(df)
        feature_names = self.feature_engineer.get_feature_names()
        
        # Prepare data
        df_pandas = df_processed.to_pandas()
        X = df_pandas[feature_names].values
        
        # Make probability predictions
        probabilities = self.best_model.predict_proba(X)
        
        return probabilities