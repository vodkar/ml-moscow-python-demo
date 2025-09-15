"""Efficient data preprocessing pipeline using Polars."""

from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""

    target_column: str = Field(default="Survived")
    id_column: str = Field(default="PassengerId")
    categorical_features: List[str] = Field(default=["Sex", "Embarked"])
    numerical_features: List[str] = Field(
        default=["Age", "Fare", "Pclass", "SibSp", "Parch"]
    )
    features_to_drop: List[str] = Field(default=["Name", "Ticket", "Cabin"])
    age_fill_strategy: str = Field(
        default="median", description="Strategy to fill missing ages"
    )
    fare_fill_strategy: str = Field(
        default="median", description="Strategy to fill missing fares"
    )


class DataPreprocessor:
    """Efficient data preprocessor using Polars."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Any] = {}
        self._is_fitted = False

    def _create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create engineered features."""
        logger.info("Creating engineered features")

        return df.with_columns(
            [
                # Family size features
                (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
                (pl.col("SibSp") + pl.col("Parch") == 0).cast(pl.Int8).alias("IsAlone"),
                # Title extraction from Name
                pl.col("Name").str.extract(r", ([A-Za-z]+)\.").alias("Title"),
                # Fare per person
                (pl.col("Fare") / (pl.col("SibSp") + pl.col("Parch") + 1)).alias(
                    "FarePerPerson"
                ),
                # Age groups
                pl.when(pl.col("Age") <= 16)
                .then(pl.lit("Child"))
                .when(pl.col("Age") <= 32)
                .then(pl.lit("Young"))
                .when(pl.col("Age") <= 48)
                .then(pl.lit("Middle"))
                .when(pl.col("Age") <= 64)
                .then(pl.lit("Senior"))
                .otherwise(pl.lit("Elder"))
                .alias("AgeGroup"),
            ]
        )

    def _handle_missing_values(
        self, df: pl.DataFrame, is_training: bool = True
    ) -> pl.DataFrame:
        """Handle missing values efficiently."""
        logger.info("Handling missing values")

        if is_training:
            # Calculate statistics for training data
            self.feature_stats["age_median"] = df.select(pl.col("Age").median()).item()
            self.feature_stats["fare_median"] = df.select(
                pl.col("Fare").median()
            ).item()
            self.feature_stats["embarked_mode"] = df.select(
                pl.col("Embarked").mode().first()
            ).item()

        # Fill missing values
        df = df.with_columns(
            [
                pl.col("Age").fill_null(self.feature_stats["age_median"]),
                pl.col("Fare").fill_null(self.feature_stats["fare_median"]),
                pl.col("Embarked").fill_null(self.feature_stats["embarked_mode"]),
            ]
        )

        # Handle title grouping
        common_titles = ["Mr", "Mrs", "Miss", "Master"]
        df = df.with_columns(
            [
                pl.when(pl.col("Title").is_in(common_titles))
                .then(pl.col("Title"))
                .otherwise(pl.lit("Other"))
                .alias("Title")
            ]
        )

        return df

    def _encode_categorical_features(
        self, df: pl.DataFrame, is_training: bool = True
    ) -> pl.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features")

        categorical_cols = self.config.categorical_features + ["Title", "AgeGroup"]

        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    # Fit label encoder
                    unique_values = df.select(pl.col(col)).unique().to_numpy().flatten()
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(unique_values)

                # Transform
                values = df.select(pl.col(col)).to_numpy().flatten()
                encoded_values = self.label_encoders[col].transform(values)
                df = df.with_columns([pl.Series(name=col, values=encoded_values)])

        return df

    def _select_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select relevant features for modeling."""
        features_to_keep = (
            self.config.numerical_features
            + self.config.categorical_features
            + ["FamilySize", "IsAlone", "Title", "FarePerPerson", "AgeGroup"]
        )

        # Keep target if present
        if self.config.target_column in df.columns:
            features_to_keep.append(self.config.target_column)

        # Keep ID column
        features_to_keep.append(self.config.id_column)

        # Filter to existing columns
        features_to_keep = [col for col in features_to_keep if col in df.columns]

        logger.info(f"Selected features: {features_to_keep}")
        return df.select(features_to_keep)

    def fit_transform(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.Series]:
        """Fit preprocessor on training data and transform."""
        logger.info("Fitting preprocessor on training data")

        # Create features
        df = self._create_features(df)

        # Handle missing values
        df = self._handle_missing_values(df, is_training=True)

        # Encode categorical features
        df = self._encode_categorical_features(df, is_training=True)

        # Select features
        df = self._select_features(df)

        self._is_fitted = True
        logger.info("Preprocessor fitted successfully")

        # Separate features and target
        target = df.select(pl.col(self.config.target_column)).to_series()
        features = df.drop([self.config.target_column])

        return features, target

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform new data using fitted preprocessor."""
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")

        logger.info("Transforming data")

        # Create features
        df = self._create_features(df)

        # Handle missing values
        df = self._handle_missing_values(df, is_training=False)

        # Encode categorical features
        df = self._encode_categorical_features(df, is_training=False)

        # Select features
        df = self._select_features(df)

        return df

    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        features = (
            self.config.numerical_features
            + self.config.categorical_features
            + ["FamilySize", "IsAlone", "Title", "FarePerPerson", "AgeGroup"]
        )
        return features
