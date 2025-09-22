# Titanic ML CLI

Fast, typed, and modular ML application for Titanic survival prediction.

## Features

- Efficient CSV ingestion with optional chunking
- Robust preprocessing (imputation, OHE, scaling)
- RandomForest classifier with metrics (Accuracy, ROC-AUC)
- Persisted artifacts with joblib
- Simple CLI via Typer: train and predict

## Install

Requires Python 3.12+.

```bash
pip install -e .
```

## Quick start

- Train:

```bash
titanic-ml train --n-estimators 200 --max-depth 12 --test-size 0.2 --random-state 42
```

- Predict:

```bash
titanic-ml predict data/test.csv --output-csv predictions.csv
```

Artifacts are stored in `artifacts/`.

## Deployment

- PyPI/Editable install via `pyproject.toml` exposes `titanic-ml` CLI
- Optional Dockerfile can be added to containerize the app
- Stateless execution: mount data and artifacts volumes for batch runs
