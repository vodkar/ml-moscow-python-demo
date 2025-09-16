"""Model training and hyperparameter optimization."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

from .models import ModelConfig, TrainingMetrics

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Advanced model trainer with hyperparameter optimization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.best_params: Dict[str, Any] = {}
        self.training_metrics: Optional[TrainingMetrics] = None
        
    def train(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[Any, TrainingMetrics]:
        """Train model with hyperparameter optimization."""
        logger.info(f"Starting training with {self.config.model_type}")
        
        # Split training data for validation
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_train
        )
        
        # Hyperparameter optimization
        if hasattr(self.config, 'enable_hyperparameter_tuning') and self.config.enable_hyperparameter_tuning:
            logger.info("Running hyperparameter optimization...")
            self.best_params = self._optimize_hyperparameters(x_tr, y_tr)
        else:
            self.best_params = self.config.hyperparameters
        
        # Train final model with best parameters
        self.model = self._create_model(self.best_params)
        self.model.fit(x_tr, y_tr)
        
        # Evaluate model
        metrics = self._evaluate_model(self.model, x_val, y_val, x_train, y_train)
        self.training_metrics = metrics
        
        logger.info(f"Training completed. Validation accuracy: {metrics.accuracy:.4f}")
        return self.model, metrics
    
    def _create_model(self, params: Dict[str, Any]):
        """Create model instance with parameters."""
        base_params = {
            'random_state': self.config.random_state,
            'n_jobs': -1
        }
        
        if self.config.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ValueError("XGBoost not available. Install with: pip install xgboost")
            return XGBClassifier(**{**base_params, **params})
        elif self.config.model_type == 'lightgbm':
            if not HAS_LIGHTGBM:
                raise ValueError("LightGBM not available. Install with: pip install lightgbm")
            return LGBMClassifier(**{**base_params, **params, 'verbosity': -1})
        elif self.config.model_type == 'random_forest':
            return RandomForestClassifier(**{**base_params, **params})
        elif self.config.model_type == 'logistic_regression':
            return LogisticRegression(**{**base_params, **params, 'max_iter': 1000})
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _optimize_hyperparameters(self, x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        study = optuna.create_study(direction='maximize', verbosity=0)
        
        def objective(trial):
            params = self._suggest_hyperparameters(trial)
            model = self._create_model(params)
            
            # Cross-validation score
            scores = cross_val_score(
                model, x_train, y_train,
                cv=self.config.cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )
            return scores.mean()
        
        n_trials = getattr(self.config, 'n_trials', 100)
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters found: {study.best_params}")
        return study.best_params
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization."""
        if self.config.model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
        elif self.config.model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            }
        elif self.config.model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
        elif self.config.model_type == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            }
        else:
            return {}
    
    def _evaluate_model(
        self, 
        model: Any, 
        x_val: np.ndarray, 
        y_val: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> TrainingMetrics:
        """Evaluate model performance."""
        # Validation predictions
        y_pred = model.predict(x_val)
        y_pred_proba = model.predict_proba(x_val)[:, 1]
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            model, x_train, y_train,
            cv=self.config.cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        return TrainingMetrics(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred),
            recall=recall_score(y_val, y_pred),
            f1_score=f1_score(y_val, y_pred),
            roc_auc=roc_auc_score(y_val, y_pred_proba),
            cv_scores=cv_scores.tolist()
        )
    
    def save_model(self, file_path: Path) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'best_params': self.best_params,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: Path) -> 'ModelTrainer':
        """Load trained model from file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model_data = joblib.load(file_path)
        
        trainer = cls(model_data['config'])
        trainer.model = model_data['model']
        trainer.best_params = model_data.get('best_params', {})
        trainer.training_metrics = model_data.get('training_metrics')
        
        logger.info(f"Model loaded from {file_path}")
        return trainer
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(x_test)[:, 1]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        
        if feature_names and len(feature_names) == len(importances):
            return dict(zip(feature_names, importances))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}


class EnsembleTrainer:
    """Ensemble model trainer for improved performance."""
    
    def __init__(self, model_configs: List[ModelConfig]):
        self.model_configs = model_configs
        self.trainers: List[ModelTrainer] = []
        self.weights: Optional[List[float]] = None
        
    def train(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[TrainingMetrics]:
        """Train ensemble of models."""
        logger.info(f"Training ensemble with {len(self.model_configs)} models")
        
        metrics_list = []
        
        for i, config in enumerate(self.model_configs):
            logger.info(f"Training model {i+1}/{len(self.model_configs)}: {config.model_type}")
            
            trainer = ModelTrainer(config)
            model, metrics = trainer.train(x_train, y_train, feature_names)
            
            self.trainers.append(trainer)
            metrics_list.append(metrics)
            
        # Calculate ensemble weights based on validation performance
        self.weights = [metrics.roc_auc for metrics in metrics_list]
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
        logger.info(f"Ensemble training completed. Weights: {self.weights}")
        return metrics_list
    
    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Get ensemble prediction probabilities."""
        if not self.trainers:
            raise ValueError("Ensemble not trained. Call train() first.")
        
        predictions = []
        for trainer in self.trainers:
            predictions.append(trainer.predict_proba(x_test))
        
        # Weighted average of predictions
        ensemble_proba = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_proba
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Get ensemble predictions."""
        proba = self.predict_proba(x_test)
        return (proba >= 0.5).astype(int)