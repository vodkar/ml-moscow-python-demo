"""Feature engineering module for Titanic dataset preprocessing."""

from __future__ import annotations

import logging
import re
from typing import Final

import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants for feature engineering
TITLE_MAPPING: Final[dict[str, str]] = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Countess": "Rare",
    "Ms": "Miss",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Mme": "Mrs",
    "Capt": "Rare",
    "Sir": "Rare",
}

FAMILY_SIZE_MAPPING: Final[dict[str, int]] = {"Alone": 0, "Small": 1, "Large": 2}

AGE_BANDS: Final[list[float]] = [0, 12, 18, 35, 60, 100]
AGE_BAND_LABELS: Final[list[str]] = ["Child", "Teen", "Young_Adult", "Adult", "Senior"]

FARE_BANDS: Final[list[float]] = [0, 7.91, 14.454, 31, 1000]
FARE_BAND_LABELS: Final[list[str]] = ["Low", "Medium", "High", "Very_High"]


class FeatureEngineerConfig(BaseModel):
    """Configuration for feature engineering."""

    fill_missing_age: bool = Field(default=True, description="Fill missing age values")
    fill_missing_embarked: bool = Field(
        default=True, description="Fill missing embarked values"
    )
    fill_missing_fare: bool = Field(
        default=True, description="Fill missing fare values"
    )
    create_title_feature: bool = Field(
        default=True, description="Extract title from name"
    )
    create_family_features: bool = Field(
        default=True, description="Create family size features"
    )
    create_age_bands: bool = Field(default=True, description="Create age bands")
    create_fare_bands: bool = Field(default=True, description="Create fare bands")
    create_cabin_features: bool = Field(
        default=True, description="Extract cabin deck and number"
    )
    drop_original_features: bool = Field(
        default=True, description="Drop original features after transformation"
    )


class TitanicFeatureEngineer:
    """Feature engineering pipeline for Titanic dataset.

    Handles missing values, creates derived features, and prepares data for ML models.
    Designed for efficient processing of large datasets using Polars.
    """

    def __init__(self, config: FeatureEngineerConfig) -> None:
        """Initialize feature engineer with configuration.

        Args:
            config: FeatureEngineerConfig instance with processing parameters
        """
        self.config = config
        self._trained_fill_values: dict[str, float | str] = {}
        self._is_fitted = False

    def _extract_title(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Extract title from name and map to standard categories.

        Args:
            dataframe: Input DataFrame with Name column

        Returns:
            DataFrame with Title column added
        """
        if not self.config.create_title_feature:
            return dataframe

        logger.info("Extracting titles from names")

        # Extract title using regex pattern
        title_pattern = r", ([A-Za-z]+)\."

        return dataframe.with_columns(
            [
                pl.col("Name")
                .cast(pl.Utf8)  # Ensure it's string type
                .str.extract(title_pattern, 1)
                .map_elements(
                    lambda x: TITLE_MAPPING.get(x, "Rare"), return_dtype=pl.Utf8
                )
                .alias("Title")
            ]
        )

    def _create_family_features(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Create family size and related features.

        Args:
            dataframe: Input DataFrame with SibSp and Parch columns

        Returns:
            DataFrame with family features added
        """
        if not self.config.create_family_features:
            return dataframe

        logger.info("Creating family size features")

        return dataframe.with_columns(
            [
                # Family size = siblings/spouses + parents/children + self
                (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
                # Is alone flag
                ((pl.col("SibSp") + pl.col("Parch")) == 0).alias("IsAlone"),
            ]
        ).with_columns(
            [
                # Family size category
                pl.when(pl.col("FamilySize") == 1)
                .then(pl.lit("Alone"))
                .when((pl.col("FamilySize") >= 2) & (pl.col("FamilySize") <= 4))
                .then(pl.lit("Small"))
                .otherwise(pl.lit("Large"))
                .alias("FamilySizeCategory")
            ]
        )

    def _create_cabin_features(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Extract cabin deck and cabin number features.

        Args:
            dataframe: Input DataFrame with Cabin column

        Returns:
            DataFrame with cabin features added
        """
        if not self.config.create_cabin_features:
            return dataframe

        logger.info("Creating cabin features")

        return dataframe.with_columns(
            [
                # Cabin deck (first letter) - convert to string first
                pl.col("Cabin")
                .cast(pl.Utf8)
                .str.slice(0, 1)
                .fill_null("Unknown")
                .alias("CabinDeck"),
                # Has cabin flag
                pl.col("Cabin").is_not_null().alias("HasCabin"),
                # Cabin count (number of cabins for passenger)
                pl.col("Cabin")
                .cast(pl.Utf8)
                .str.split(" ")
                .list.len()
                .fill_null(0)
                .alias("CabinCount"),
            ]
        )

    def _create_age_bands(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Create age band categories.

        Args:
            dataframe: Input DataFrame with Age column

        Returns:
            DataFrame with age bands added
        """
        if not self.config.create_age_bands:
            return dataframe

        logger.info("Creating age bands")

        # Create age bands using manual conditions
        return dataframe.with_columns(
            [
                pl.when(pl.col("Age") <= 12)
                .then(pl.lit("Child"))
                .when((pl.col("Age") > 12) & (pl.col("Age") <= 18))
                .then(pl.lit("Teen"))
                .when((pl.col("Age") > 18) & (pl.col("Age") <= 35))
                .then(pl.lit("Young_Adult"))
                .when((pl.col("Age") > 35) & (pl.col("Age") <= 60))
                .then(pl.lit("Adult"))
                .when(pl.col("Age") > 60)
                .then(pl.lit("Senior"))
                .otherwise(None)
                .alias("AgeBand")
            ]
        )

    def _create_fare_bands(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Create fare band categories.

        Args:
            dataframe: Input DataFrame with Fare column

        Returns:
            DataFrame with fare bands added
        """
        if not self.config.create_fare_bands:
            return dataframe

        logger.info("Creating fare bands")

        # Create fare bands using manual conditions
        return dataframe.with_columns(
            [
                pl.when(pl.col("Fare") <= 7.91)
                .then(pl.lit("Low"))
                .when((pl.col("Fare") > 7.91) & (pl.col("Fare") <= 14.454))
                .then(pl.lit("Medium"))
                .when((pl.col("Fare") > 14.454) & (pl.col("Fare") <= 31))
                .then(pl.lit("High"))
                .when(pl.col("Fare") > 31)
                .then(pl.lit("Very_High"))
                .otherwise(None)
                .alias("FareBand")
            ]
        )

    def _fill_missing_values(
        self, dataframe: pl.DataFrame, is_training: bool = True
    ) -> pl.DataFrame:
        """Fill missing values using appropriate strategies.

        Args:
            dataframe: Input DataFrame
            is_training: Whether this is training data (to compute fill values)

        Returns:
            DataFrame with missing values filled
        """
        logger.info("Filling missing values")

        expressions = []

        # Fill missing Age values
        if self.config.fill_missing_age and "Age" in dataframe.columns:
            if is_training:
                # Use median age grouped by Pclass and Sex for more accurate imputation
                age_fill_value = (
                    dataframe.group_by(["Pclass", "Sex"])
                    .agg(pl.col("Age").median().alias("median_age"))
                    .to_dict(as_series=False)
                )
                self._trained_fill_values["age_by_group"] = age_fill_value

                # Also store overall median as fallback
                overall_age_median = dataframe.select(pl.col("Age").median()).item()
                self._trained_fill_values["age_overall"] = overall_age_median

            # For now, use overall median (can be enhanced to use group-based later)
            age_fill = self._trained_fill_values.get("age_overall", 29.0)
            expressions.append(pl.col("Age").fill_null(age_fill))

        # Fill missing Embarked values
        if self.config.fill_missing_embarked and "Embarked" in dataframe.columns:
            if is_training:
                embarked_mode = dataframe.select(
                    pl.col("Embarked").mode().first()
                ).item()
                self._trained_fill_values["embarked"] = embarked_mode

            embarked_fill = self._trained_fill_values.get("embarked", "S")
            expressions.append(pl.col("Embarked").fill_null(embarked_fill))

        # Fill missing Fare values
        if self.config.fill_missing_fare and "Fare" in dataframe.columns:
            if is_training:
                fare_median = dataframe.select(pl.col("Fare").median()).item()
                self._trained_fill_values["fare"] = fare_median

            fare_fill = self._trained_fill_values.get("fare", 14.4542)
            expressions.append(pl.col("Fare").fill_null(fare_fill))

        if expressions:
            dataframe = dataframe.with_columns(expressions)

        return dataframe

    def _encode_categorical_features(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Encode categorical features for ML models.

        Args:
            dataframe: Input DataFrame

        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features")

        expressions = []

        # Encode Sex as binary
        if "Sex" in dataframe.columns:
            expressions.append(
                pl.when(pl.col("Sex") == "male").then(1).otherwise(0).alias("Sex_male")
            )

        # Encode Embarked as one-hot
        if "Embarked" in dataframe.columns:
            for port in ["C", "Q", "S"]:
                expressions.append(
                    pl.when(pl.col("Embarked") == port)
                    .then(1)
                    .otherwise(0)
                    .alias(f"Embarked_{port}")
                )

        # Encode Title as one-hot
        if "Title" in dataframe.columns:
            for title in ["Mr", "Miss", "Mrs", "Master", "Rare"]:
                expressions.append(
                    pl.when(pl.col("Title") == title)
                    .then(1)
                    .otherwise(0)
                    .alias(f"Title_{title}")
                )

        # Encode FamilySizeCategory
        if "FamilySizeCategory" in dataframe.columns:
            for category in ["Alone", "Small", "Large"]:
                expressions.append(
                    pl.when(pl.col("FamilySizeCategory") == category)
                    .then(1)
                    .otherwise(0)
                    .alias(f"Family_{category}")
                )

        # Encode CabinDeck
        if "CabinDeck" in dataframe.columns:
            cabin_decks = ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]
            for deck in cabin_decks:
                expressions.append(
                    pl.when(pl.col("CabinDeck") == deck)
                    .then(1)
                    .otherwise(0)
                    .alias(f"Deck_{deck}")
                )

        # Encode AgeBand
        if "AgeBand" in dataframe.columns:
            for band in AGE_BAND_LABELS:
                expressions.append(
                    pl.when(pl.col("AgeBand") == band)
                    .then(1)
                    .otherwise(0)
                    .alias(f"Age_{band}")
                )

        # Encode FareBand
        if "FareBand" in dataframe.columns:
            for band in FARE_BAND_LABELS:
                expressions.append(
                    pl.when(pl.col("FareBand") == band)
                    .then(1)
                    .otherwise(0)
                    .alias(f"Fare_{band}")
                )

        if expressions:
            dataframe = dataframe.with_columns(expressions)

        return dataframe

    def _drop_original_features(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Drop original features that have been transformed.

        Args:
            dataframe: Input DataFrame

        Returns:
            DataFrame with original features removed
        """
        if not self.config.drop_original_features:
            return dataframe

        logger.info("Dropping original features")

        columns_to_drop = []

        # Always drop these non-predictive columns
        for col in ["Name", "Ticket"]:
            if col in dataframe.columns:
                columns_to_drop.append(col)

        # Drop original categorical columns if we've encoded them
        for col in ["Sex", "Embarked", "Cabin"]:
            if col in dataframe.columns:
                columns_to_drop.append(col)

        # Drop intermediate columns
        for col in ["Title", "FamilySizeCategory", "CabinDeck", "AgeBand", "FareBand"]:
            if col in dataframe.columns:
                columns_to_drop.append(col)

        if columns_to_drop:
            dataframe = dataframe.drop(columns_to_drop)

        return dataframe

    def fit_transform(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Fit the feature engineer on training data and transform it.

        Args:
            dataframe: Training DataFrame

        Returns:
            Transformed DataFrame
        """
        logger.info("Fitting and transforming training data")

        # Reset fitted state
        self._trained_fill_values = {}

        # Apply transformations in sequence
        dataframe = self._fill_missing_values(dataframe, is_training=True)
        dataframe = self._extract_title(dataframe)
        dataframe = self._create_family_features(dataframe)
        dataframe = self._create_cabin_features(dataframe)
        dataframe = self._create_age_bands(dataframe)
        dataframe = self._create_fare_bands(dataframe)
        dataframe = self._encode_categorical_features(dataframe)
        dataframe = self._drop_original_features(dataframe)

        self._is_fitted = True
        logger.info(f"Feature engineering complete. Final shape: {dataframe.shape}")

        return dataframe

    def transform(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Transform new data using fitted parameters.

        Args:
            dataframe: DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If feature engineer hasn't been fitted
        """
        if not self._is_fitted:
            raise ValueError(
                "Feature engineer must be fitted before transforming new data"
            )

        logger.info("Transforming new data")

        # Apply transformations in same sequence as fit_transform
        dataframe = self._fill_missing_values(dataframe, is_training=False)
        dataframe = self._extract_title(dataframe)
        dataframe = self._create_family_features(dataframe)
        dataframe = self._create_cabin_features(dataframe)
        dataframe = self._create_age_bands(dataframe)
        dataframe = self._create_fare_bands(dataframe)
        dataframe = self._encode_categorical_features(dataframe)
        dataframe = self._drop_original_features(dataframe)

        logger.info(f"Transformation complete. Final shape: {dataframe.shape}")

        return dataframe

    def get_feature_names(self, dataframe: pl.DataFrame) -> list[str]:
        """Get names of features after transformation.

        Args:
            dataframe: Original DataFrame

        Returns:
            List of feature names after transformation
        """
        # Create a copy and transform to get feature names
        temp_df = dataframe.clone()

        if not self._is_fitted:
            temp_df = self.fit_transform(temp_df)
        else:
            temp_df = self.transform(temp_df)

        # Remove target column if present
        feature_columns = [
            col for col in temp_df.columns if col not in ["Survived", "PassengerId"]
        ]

        return feature_columns


def create_feature_engineer(
    fill_missing_age: bool = True,
    fill_missing_embarked: bool = True,
    fill_missing_fare: bool = True,
    create_title_feature: bool = True,
    create_family_features: bool = True,
    create_age_bands: bool = True,
    create_fare_bands: bool = True,
    create_cabin_features: bool = True,
    drop_original_features: bool = True,
) -> TitanicFeatureEngineer:
    """Create a configured feature engineer instance.

    Args:
        fill_missing_age: Fill missing age values
        fill_missing_embarked: Fill missing embarked values
        fill_missing_fare: Fill missing fare values
        create_title_feature: Extract title from name
        create_family_features: Create family size features
        create_age_bands: Create age bands
        create_fare_bands: Create fare bands
        create_cabin_features: Extract cabin features
        drop_original_features: Drop original features after transformation

    Returns:
        Configured TitanicFeatureEngineer instance
    """
    config = FeatureEngineerConfig(
        fill_missing_age=fill_missing_age,
        fill_missing_embarked=fill_missing_embarked,
        fill_missing_fare=fill_missing_fare,
        create_title_feature=create_title_feature,
        create_family_features=create_family_features,
        create_age_bands=create_age_bands,
        create_fare_bands=create_fare_bands,
        create_cabin_features=create_cabin_features,
        drop_original_features=drop_original_features,
    )
    return TitanicFeatureEngineer(config)
