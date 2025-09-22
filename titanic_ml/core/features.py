from __future__ import annotations

"""Feature engineering utilities for the Titanic dataset."""

from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define feature groups
PASSENGER_ID_COL: Final[str] = "PassengerId"
TARGET_COL: Final[str] = "Survived"

CATEGORICAL_COLS: Final[tuple[str, ...]] = ("Sex", "Embarked", "Pclass")
NUMERIC_COLS: Final[tuple[str, ...]] = ("Age", "SibSp", "Parch", "Fare")

SELECTED_FEATURES: Final[tuple[str, ...]] = CATEGORICAL_COLS + NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline for categorical and numeric features.

    Returns:
        ColumnTransformer: A transformer that imputes, encodes, and scales features.
    """

    categorical_pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=float
                ),
            ),
        ]
    )

    numeric_pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLS)),
            ("numeric", numeric_pipeline, list(NUMERIC_COLS)),
        ],
        remainder="drop",
    )
    return preprocessor


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and return the model features from a DataFrame.

    Args:
        df: Input DataFrame with raw columns.

    Returns:
        DataFrame: DataFrame containing selected features only.
    """

    return df.loc[:, list(SELECTED_FEATURES)]
