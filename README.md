# Titanic ML: Modern Machine Learning CLI Application

A high-performance, modern machine learning application for analyzing the Titanic dataset and predicting passenger survival. Built with cutting-edge tools and designed for scalability, efficiency, and ease of use.

## 🚀 Features

- **Modern Tech Stack**: Built with Polars for blazing-fast data processing, Pydantic for robust data validation, and Typer for beautiful CLI interfaces
- **Multiple ML Algorithms**: Support for Logistic Regression, Random Forest, XGBoost, LightGBM, and SVM
- **Hyperparameter Optimization**: Automated tuning using Optuna for optimal model performance
- **Advanced Feature Engineering**: Automated creation of derived features including family size, titles, cabin analysis, and age/fare bands
- **Efficient Data Processing**: Memory-optimized loading and processing for large datasets using Polars
- **Rich CLI Interface**: Beautiful command-line interface with progress bars, tables, and colored output
- **Comprehensive Evaluation**: Multiple metrics including accuracy, ROC AUC, cross-validation, and feature importance
- **Model Persistence**: Save and load trained models with metadata
- **Type Safety**: Full type hints and validation throughout the codebase

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Install from Source

```bash
git clone https://github.com/vodkar/ml-moscow-python-demo.git
cd ml-moscow-python-demo
pip install -e .
```

### Development Installation

For development with testing dependencies:

```bash
pip install -e ".[dev]"
```

## 🎯 Quick Start

### 1. Explore the Dataset

```bash
titanic-ml info
```

This command displays comprehensive information about the training and test datasets, including:

- Dataset dimensions and memory usage
- Column types and null value statistics
- Survival rate statistics

### 2. Train a Model

Train a basic logistic regression model:

```bash
titanic-ml train --model-type logistic_regression
```

Train an XGBoost model with hyperparameter optimization:

```bash
titanic-ml train --model-type xgboost --optimize --trials 50
```

### 3. Make Predictions

```bash
titanic-ml predict models/titanic_xgboost_model.joblib --probabilities --output-file predictions.csv
```

### 4. Evaluate Model Performance

```bash
titanic-ml evaluate models/titanic_xgboost_model.joblib
```

## 📋 CLI Commands

### `titanic-ml info`

Display comprehensive dataset information including statistics, column details, and survival rates.

**Options:**

- `--data-path PATH`: Path to data directory (default: data)

### `titanic-ml train`

Train machine learning models with various algorithms and configurations.

**Options:**

- `--model-type TEXT`: Model type (logistic_regression, random_forest, xgboost, lightgbm, svm)
- `--output-dir PATH`: Directory to save trained model (default: models)
- `--optimize/--no-optimize`: Enable hyperparameter optimization (default: True)
- `--trials INTEGER`: Number of optimization trials (default: 100)
- `--cv-folds INTEGER`: Cross-validation folds (default: 5)
- `--random-state INTEGER`: Random state for reproducibility (default: 42)
- `--verbose`: Enable verbose logging

**Examples:**

```bash
# Quick training without optimization
titanic-ml train --model-type random_forest --no-optimize

# Optimized XGBoost with custom settings
titanic-ml train --model-type xgboost --trials 200 --cv-folds 10

# Train all models (run multiple times with different model types)
for model in logistic_regression random_forest xgboost lightgbm; do
    titanic-ml train --model-type $model --trials 50
done
```

### `titanic-ml predict`

Generate predictions using trained models.

**Arguments:**

- `MODEL_PATH`: Path to trained model file

**Options:**

- `--data-path PATH`: Path to data directory (default: data)
- `--output-file PATH`: Output CSV file for predictions
- `--probabilities`: Include prediction probabilities
- `--verbose`: Enable verbose logging

**Examples:**

```bash
# Basic prediction
titanic-ml predict models/titanic_xgboost_model.joblib

# Prediction with probabilities saved to file
titanic-ml predict models/titanic_xgboost_model.joblib --probabilities --output-file results.csv

# Prediction from custom data path
titanic-ml predict models/model.joblib --data-path /path/to/custom/data
```

### `titanic-ml evaluate`

Evaluate trained model performance on training data.

**Arguments:**

- `MODEL_PATH`: Path to trained model file

**Options:**

- `--data-path PATH`: Path to data directory (default: data)
- `--verbose`: Enable verbose logging

## 🔧 Architecture

### Core Components

1. **Data Loader** (`titanic_ml.data_loader`)
   - Efficient CSV loading with Polars
   - Automatic data type optimization
   - Memory usage monitoring
   - Schema validation

2. **Feature Engineer** (`titanic_ml.feature_engineer`)
   - Missing value imputation
   - Title extraction from names
   - Family size calculations
   - Cabin deck analysis
   - Age and fare band creation
   - One-hot encoding
   - Feature selection

3. **Trainer** (`titanic_ml.trainer`)
   - Multiple ML algorithm support
   - Hyperparameter optimization with Optuna
   - Cross-validation evaluation
   - Model persistence
   - Feature importance analysis

4. **CLI Interface** (`titanic_ml.cli`)
   - Rich terminal interface
   - Progress tracking
   - Formatted output tables
   - Error handling

### Feature Engineering Pipeline

The application automatically creates the following features:

1. **Title Features**: Extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
2. **Family Features**: Family size, alone status, family size categories
3. **Cabin Features**: Cabin deck, cabin availability, cabin count
4. **Age Bands**: Child, Teen, Young Adult, Adult, Senior
5. **Fare Bands**: Low, Medium, High, Very High
6. **Encoded Features**: One-hot encoded categorical variables

### Supported Models

- **Logistic Regression**: Fast linear model good for baseline performance
- **Random Forest**: Robust ensemble method with feature importance
- **XGBoost**: Gradient boosting with excellent performance
- **LightGBM**: Fast gradient boosting with low memory usage
- **SVM**: Support Vector Machine with RBF kernel

## 📊 Performance

The application achieves competitive performance on the Titanic dataset:

- **Logistic Regression**: ~82-84% accuracy
- **Random Forest**: ~83-86% accuracy  
- **XGBoost**: ~85-88% accuracy (with optimization)
- **LightGBM**: ~84-87% accuracy
- **SVM**: ~81-84% accuracy

Performance metrics include:

- Accuracy
- ROC AUC
- Cross-validation scores
- Confusion matrix
- Feature importance rankings

## 🧪 Testing

The project includes comprehensive tests for all components:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=titanic_ml --cov-report=html

# Run specific test modules
pytest tests/test_data_loader.py
pytest tests/test_feature_engineer.py
pytest tests/test_trainer.py
```

## 🏗️ Development

### Project Structure

```
titanic_ml/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── data_loader.py           # Data loading and validation
├── feature_engineer.py      # Feature engineering pipeline
└── trainer.py              # Model training and evaluation

tests/
├── conftest.py              # Test configuration and fixtures
├── test_data_loader.py      # Data loader tests
├── test_feature_engineer.py # Feature engineering tests
└── test_trainer.py          # Trainer tests

data/
├── train.csv                # Training dataset
├── test.csv                 # Test dataset
└── gender_submission.csv    # Sample submission format
```

### Code Quality

The project maintains high code quality standards:

- **Type Safety**: Full type hints using modern Python syntax
- **Data Validation**: Pydantic models for configuration validation
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Detailed docstrings in Google style
- **Testing**: Unit and integration tests with >80% coverage
- **Formatting**: Black code formatting and Ruff linting

### Configuration

All components are configurable through Pydantic models:

```python
from titanic_ml.data_loader import DataLoaderConfig
from titanic_ml.feature_engineer import FeatureEngineerConfig
from titanic_ml.trainer import TrainerConfig

# Custom data loader configuration
data_config = DataLoaderConfig(
    data_path=Path("custom/data"),
    lazy_loading=True,
    chunk_size=1000
)

# Custom feature engineering configuration
feature_config = FeatureEngineerConfig(
    fill_missing_age=True,
    create_title_feature=True,
    create_family_features=True,
    drop_original_features=True
)

# Custom trainer configuration
trainer_config = TrainerConfig(
    model_type="xgboost",
    hyperparameter_optimization=True,
    optimization_trials=100,
    cv_folds=5
)
```

## 📈 Performance Optimization

### Memory Efficiency

- **Polars**: Lightning-fast DataFrame operations with optimized memory usage
- **Data Types**: Automatic optimization of column data types (UInt8, Float32, Categorical)
- **Lazy Loading**: Optional lazy evaluation for large datasets
- **Chunked Processing**: Support for processing large files in chunks

### Speed Optimizations

- **Vectorized Operations**: All feature engineering uses vectorized Polars operations
- **Parallel Processing**: Multi-threading support for cross-validation and hyperparameter optimization
- **Efficient Algorithms**: Modern ML libraries with optimized implementations
- **Caching**: Intelligent caching of processed data and fitted transformers

### Scalability

The application is designed to handle large datasets efficiently:

- Memory-mapped file reading for huge datasets
- Incremental processing capabilities
- Distributed computing support (future enhancement)
- Efficient model serialization

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Ensure code quality with `black` and `ruff`
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/vodkar/ml-moscow-python-demo.git
cd ml-moscow-python-demo
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Check code quality
black titanic_ml tests
ruff check titanic_ml tests
mypy titanic_ml
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/c/titanic) for providing the Titanic dataset
- [Polars](https://github.com/pola-rs/polars) for blazing-fast data processing
- [Optuna](https://github.com/optuna/optuna) for hyperparameter optimization
- [Rich](https://github.com/Textualize/rich) for beautiful terminal interfaces
- [Typer](https://github.com/tiangolo/typer) for modern CLI development

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/vodkar/ml-moscow-python-demo/issues) page
2. Create a new issue with detailed information
3. Include error messages, logs, and system information
4. Provide steps to reproduce the problem

---

**Built with ❤️ for the ML Moscow community**
