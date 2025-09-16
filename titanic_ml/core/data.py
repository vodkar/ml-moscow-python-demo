from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl


def read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, ignore_errors=True)


def load_train_data(path: Path, target: str) -> tuple[pl.DataFrame, pl.Series]:
    df = read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    y = df[target]
    X = df.drop([target])
    return X, y


def load_test_data(path: Path) -> pl.DataFrame:
    return read_csv(path)


def train_valid_split(
    X: pl.DataFrame,
    y: pl.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    n = X.height
    n_valid = int(n * test_size)
    # permutation-based split for reproducibility
    rng = np.random.default_rng(seed=random_state)
    order = rng.permutation(n).tolist()
    valid_idx = set(order[:n_valid])
    mask_valid = pl.Series([i in valid_idx for i in range(n)])
    X_train = X.filter(~mask_valid)
    y_train = y.filter(~mask_valid)
    X_valid = X.filter(mask_valid)
    y_valid = y.filter(mask_valid)
    return X_train, X_valid, y_train, y_valid
