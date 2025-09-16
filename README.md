# Titanic ML CLI

A fast, scalable CLI application to analyze and predict survival on the Titanic dataset using Polars and scikit-learn.

## Features

- Efficient data processing with Polars
- Robust feature engineering with consistent one-hot encoding
- Gradient boosting model (HistGradientBoostingClassifier)
- Typer CLI with train/evaluate/predict commands
- Artifacts saving (encoder + model) and CSV predictions
- Pytest tests

## Install

- Python 3.10–3.12 is supported
- Install dependencies and project in editable mode

```bash
uv pip install -e .
```

If uv is not available, use pip:

```bash
pip install -e .
```

## Usage

- Train model and save artifacts:

```bash
python -m titanic_ml.main train --train-csv data/train.csv --artifacts-dir artifacts
```

- Evaluate on the validation split:

```bash
python -m titanic_ml.main evaluate-cmd --train-csv data/train.csv --artifacts-dir artifacts
```

- Predict for test.csv and save predictions:

```bash
python -m titanic_ml.main predict --test-csv data/test.csv --artifacts-dir artifacts --out-csv artifacts/predictions.csv
```

You can also use the installed script once installed:

```bash
titanic-ml train
```

## Tests

```bash
pytest -q
```

## Deployment

- Containerize with Docker:

```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["titanic-ml", "train", "--train-csv", "data/train.csv", "--artifacts-dir", "artifacts"]
```

- Or run as a job in CI using the commands above. Artifacts are saved to `artifacts/`.

## Notes

- The pipeline assumes standard Kaggle Titanic CSV schema. Adjust feature lists in `titanic_ml/core/config.py` if needed.
- For very large datasets, Polars scales out-of-core; consider chunked reading and parquet.
