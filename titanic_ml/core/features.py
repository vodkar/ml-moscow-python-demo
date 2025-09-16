"""Feature engineering pipeline for the Titanic dataset."""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from rich.console import Console
from sklearn.preprocessing import LabelEncoder, StandardScaler

console = Console()


class FeatureEngineer:
    """Feature engineering for Titanic dataset."""

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_title(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract title from name."""
        console.print("Extracting titles from names...")

        def get_title(name: str) -> str:
            """Extract title from name string."""
            title_search = re.search(r" ([A-Za-z]+)\.", name)
            if title_search:
                title = title_search.group(1)
                # Group rare titles
                if title in [
                    "Lady",
                    "Countess",
                    "Capt",
                    "Col",
                    "Don",
                    "Dr",
                    "Major",
                    "Rev",
                    "Sir",
                    "Jonkheer",
                    "Dona",
                ]:
                    return "Rare"
                elif title in ["Mlle", "Ms"]:
                    return "Miss"
                elif title == "Mme":
                    return "Mrs"
                else:
                    return title
            return "Unknown"

        return df.with_columns(
            [
                pl.col("Name")
                .map_elements(get_title, return_dtype=pl.Utf8)
                .alias("Title")
            ]
        )

    def create_family_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create family-related features."""
        console.print("Creating family features...")

        return df.with_columns(
            [
                # Family size
                (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
                # Is alone
                ((pl.col("SibSp") + pl.col("Parch")) == 0)
                .cast(pl.Int32)
                .alias("IsAlone"),
            ]
        ).with_columns(
            [
                # Family size category
                pl.when(pl.col("FamilySize") == 1)
                .then(pl.lit("Single"))
                .when(pl.col("FamilySize").is_between(2, 4))
                .then(pl.lit("Small"))
                .when(pl.col("FamilySize").is_between(5, 7))
                .then(pl.lit("Medium"))
                .otherwise(pl.lit("Large"))
                .alias("FamilySizeCategory")
            ]
        )

    def create_age_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create age-related features."""
        console.print("Creating age features...")

        # Fill missing ages with median by title and pclass
        df_with_age = df.with_columns(
            [
                pl.col("Age")
                .fill_null(pl.col("Age").median().over(["Title", "Pclass"]))
                .fill_null(pl.col("Age").median())
            ]
        )

        return df_with_age.with_columns(
            [
                # Age categories
                pl.when(pl.col("Age") <= 16)
                .then(pl.lit("Child"))
                .when(pl.col("Age").is_between(17, 32))
                .then(pl.lit("Young_Adult"))
                .when(pl.col("Age").is_between(33, 48))
                .then(pl.lit("Adult"))
                .when(pl.col("Age").is_between(49, 64))
                .then(pl.lit("Middle_Aged"))
                .otherwise(pl.lit("Senior"))
                .alias("AgeCategory"),
                # Age bins
                (pl.col("Age") // 10).cast(pl.Int32).alias("AgeBin"),
            ]
        )

    def create_fare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create fare-related features."""
        console.print("Creating fare features...")

        # Fill missing fares with median by pclass
        df_with_fare = df.with_columns(
            [
                pl.col("Fare")
                .fill_null(pl.col("Fare").median().over("Pclass"))
                .fill_null(pl.col("Fare").median())
            ]
        )

        return df_with_fare.with_columns(
            [
                # Fare per person
                (pl.col("Fare") / pl.col("FamilySize")).alias("FarePerPerson"),
                # Fare categories
                pl.when(pl.col("Fare") <= 7.91)
                .then(pl.lit("Low"))
                .when(pl.col("Fare").is_between(7.92, 14.45))
                .then(pl.lit("Medium"))
                .when(pl.col("Fare").is_between(14.46, 31.0))
                .then(pl.lit("High"))
                .otherwise(pl.lit("Very_High"))
                .alias("FareCategory"),
            ]
        )

    def create_cabin_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create cabin-related features."""
        console.print("Creating cabin features...")

        def get_cabin_deck(cabin: str) -> str:
            """Extract deck from cabin."""
            if cabin and isinstance(cabin, str) and cabin.strip():
                return cabin[0]
            return "Unknown"

        return df.with_columns(
            [
                # Has cabin
                pl.col("Cabin").is_not_null().cast(pl.Int32).alias("HasCabin"),
                # Cabin deck
                pl.col("Cabin")
                .map_elements(get_cabin_deck, return_dtype=pl.Utf8)
                .alias("CabinDeck"),
            ]
        )

    def encode_categorical_features(
        self, df: pl.DataFrame, is_training: bool = True
    ) -> pl.DataFrame:
        """Encode categorical features."""
        console.print("Encoding categorical features...")

        categorical_columns = [
            "Sex",
            "Embarked",
            "Title",
            "FamilySizeCategory",
            "AgeCategory",
            "FareCategory",
            "CabinDeck",
        ]

        df_encoded = df.clone()

        # Fill missing embarked with mode
        df_encoded = df_encoded.with_columns(
            [
                pl.col("Embarked").fill_null("S")  # Most common embarkation point
            ]
        )

        for col in categorical_columns:
            if col in df_encoded.columns:
                if is_training:
                    # Fit and transform
                    unique_values = df_encoded[col].unique().drop_nulls().to_list()
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(unique_values)

                if col in self.label_encoders:
                    # Transform
                    values = df_encoded[col].to_list()
                    encoded_values = []

                    for val in values:
                        if val in self.label_encoders[col].classes_:
                            encoded_values.append(
                                self.label_encoders[col].transform([val])[0]
                            )
                        else:
                            # Handle unseen categories
                            encoded_values.append(0)

                    df_encoded = df_encoded.with_columns(
                        [pl.Series(f"{col}_encoded", encoded_values)]
                    )

        return df_encoded

    def select_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select final features for modeling."""
        console.print("Selecting features for modeling...")

        feature_columns = [
            "PassengerId",  # Keep for identification
            "Pclass",
            "Sex_encoded",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked_encoded",
            "Title_encoded",
            "FamilySize",
            "IsAlone",
            "FamilySizeCategory_encoded",
            "AgeCategory_encoded",
            "AgeBin",
            "FarePerPerson",
            "FareCategory_encoded",
            "HasCabin",
            "CabinDeck_encoded",
        ]

        # Add target if present
        if "Survived" in df.columns:
            feature_columns.insert(1, "Survived")

        # Filter columns that exist in the dataframe
        available_columns = [col for col in feature_columns if col in df.columns]

        return df.select(available_columns)

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform training data."""
        console.print("Fitting and transforming training data...")

        # Apply all transformations
        df_processed = self.extract_title(df)
        df_processed = self.create_family_features(df_processed)
        df_processed = self.create_age_features(df_processed)
        df_processed = self.create_fare_features(df_processed)
        df_processed = self.create_cabin_features(df_processed)
        df_processed = self.encode_categorical_features(df_processed, is_training=True)
        df_processed = self.select_features(df_processed)

        self.is_fitted = True
        return df_processed

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform test data using fitted encoders."""
        if not self.is_fitted:
            raise ValueError(
                "FeatureEngineer must be fitted before transforming test data"
            )

        console.print("Transforming test data...")

        # Apply all transformations
        df_processed = self.extract_title(df)
        df_processed = self.create_family_features(df_processed)
        df_processed = self.create_age_features(df_processed)
        df_processed = self.create_fare_features(df_processed)
        df_processed = self.create_cabin_features(df_processed)
        df_processed = self.encode_categorical_features(df_processed, is_training=False)
        df_processed = self.select_features(df_processed)

        return df_processed

    def get_feature_names(self) -> List[str]:
        """Get list of feature names for modeling."""
        return [
            "Pclass",
            "Sex_encoded",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked_encoded",
            "Title_encoded",
            "FamilySize",
            "IsAlone",
            "FamilySizeCategory_encoded",
            "AgeCategory_encoded",
            "AgeBin",
            "FarePerPerson",
            "FareCategory_encoded",
            "HasCabin",
            "CabinDeck_encoded",
        ]
