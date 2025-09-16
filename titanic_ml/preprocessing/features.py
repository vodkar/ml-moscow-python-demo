"""Feature engineering and preprocessing pipeline."""

import re
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


class PreprocessingMode(StrEnum):
    """Preprocessing mode enum."""

    TRAIN = "train"
    INFERENCE = "inference"


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""

    handle_missing_values: bool = Field(default=True)
    create_family_size: bool = Field(default=True)
    extract_title_from_name: bool = Field(default=True)
    scale_numerical_features: bool = Field(default=True)
    encode_categorical_features: bool = Field(default=True)


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer for Titanic dataset."""

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize feature engineer.

        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None
        self.fitted_: bool = False

    def _extract_title(self, name: str) -> str:
        """Extract title from passenger name.

        Args:
            name: Passenger name

        Returns:
            Extracted title
        """
        title_search = re.search(r" ([A-Za-z]+)\\.", name)
        if title_search:
            title = title_search.group(1)
            # Normalize rare titles
            if title in ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]:
                return "Rare"
            elif title in ["Mlle", "Ms"]:
                return "Miss"
            elif title == "Mme":
                return "Mrs"
            return title
        return "Unknown"

    def _handle_missing_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Args:
            dataframe: Input dataframe

        Returns:
            DataFrame with handled missing values
        """
        processed_df = dataframe.copy()

        # Fill Age with median grouped by Pclass and Sex
        if "Age" in processed_df.columns:
            processed_df["Age"] = processed_df.groupby(["Pclass", "Sex"])["Age"].transform(
                lambda group: group.fillna(group.median())
            )

        # Fill Embarked with mode
        if "Embarked" in processed_df.columns:
            processed_df["Embarked"] = processed_df["Embarked"].fillna(
                processed_df["Embarked"].mode()[0]
            )

        # Fill Fare with median grouped by Pclass
        if "Fare" in processed_df.columns:
            processed_df["Fare"] = processed_df.groupby("Pclass")["Fare"].transform(
                lambda group: group.fillna(group.median())
            )

        return processed_df

    def _create_derived_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create derived features.

        Args:
            dataframe: Input dataframe

        Returns:
            DataFrame with new features
        """
        processed_df = dataframe.copy()

        if self.config.create_family_size and "SibSp" in processed_df.columns and "Parch" in processed_df.columns:
            processed_df["FamilySize"] = processed_df["SibSp"] + processed_df["Parch"] + 1
            processed_df["IsAlone"] = (processed_df["FamilySize"] == 1).astype(int)

        if self.config.extract_title_from_name and "Name" in processed_df.columns:
            processed_df["Title"] = processed_df["Name"].apply(self._extract_title)

        # Age groups
        if "Age" in processed_df.columns:
            processed_df["AgeBin"] = pd.cut(
                processed_df["Age"], bins=[0, 12, 20, 40, 120], labels=["Child", "Teen", "Adult", "Elder"]
            )

        # Fare bins
        if "Fare" in processed_df.columns:
            processed_df["FareBin"] = pd.qcut(
                processed_df["Fare"], q=4, labels=["Low", "Medium-Low", "Medium-High", "High"]
            )

        return processed_df

    def fit(self, dataframe: pd.DataFrame, y: pd.Series | None = None) -> "TitanicFeatureEngineer":
        """Fit the feature engineer.

        Args:
            dataframe: Training dataframe
            y: Target values (ignored)

        Returns:
            Fitted feature engineer
        """
        processed_df = dataframe.copy()

        if self.config.handle_missing_values:
            processed_df = self._handle_missing_values(processed_df)

        processed_df = self._create_derived_features(processed_df)

        # Fit label encoders for categorical features
        if self.config.encode_categorical_features:
            categorical_cols = ["Sex", "Embarked", "Title", "AgeBin", "FareBin"]
            for column in categorical_cols:
                if column in processed_df.columns:
                    self.label_encoders[column] = LabelEncoder()
                    self.label_encoders[column].fit(processed_df[column].astype(str))

        # Fit scaler for numerical features
        if self.config.scale_numerical_features:
            numerical_cols = ["Age", "Fare", "FamilySize"]
            available_cols = [col for col in numerical_cols if col in processed_df.columns]
            if available_cols:
                self.scaler = StandardScaler()
                self.scaler.fit(processed_df[available_cols])

        self.fitted_ = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe.

        Args:
            dataframe: Input dataframe

        Returns:
            Transformed dataframe

        Raises:
            ValueError: If the transformer hasn't been fitted
        """
        if not self.fitted_:
            raise ValueError("Feature engineer must be fitted before transform")

        processed_df = dataframe.copy()

        if self.config.handle_missing_values:
            processed_df = self._handle_missing_values(processed_df)

        processed_df = self._create_derived_features(processed_df)

        # Apply label encoding
        if self.config.encode_categorical_features:
            for column, encoder in self.label_encoders.items():
                if column in processed_df.columns:
                    processed_df[column] = encoder.transform(processed_df[column].astype(str))

        # Apply scaling
        if self.config.scale_numerical_features and self.scaler is not None:
            numerical_cols = ["Age", "Fare", "FamilySize"]
            available_cols = [col for col in numerical_cols if col in processed_df.columns]
            if available_cols:
                processed_df[available_cols] = self.scaler.transform(processed_df[available_cols])

        return processed_df

    def get_feature_names(self) -> list[str]:
        """Get list of feature names after transformation.

        Returns:
            List of feature column names
        """
        base_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        
        derived_features = []
        if self.config.create_family_size:
            derived_features.extend(["FamilySize", "IsAlone"])
        if self.config.extract_title_from_name:
            derived_features.append("Title")
        
        derived_features.extend(["AgeBin", "FareBin"])
        
        return base_features + derived_features


# Default feature engineering configuration
DEFAULT_FEATURE_CONFIG: Final[FeatureConfig] = FeatureConfig()