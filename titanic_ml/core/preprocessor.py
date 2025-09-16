"""Data preprocessing module for Titanic dataset with efficient Polars operations."""

import logging
import re
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Constants for feature engineering
TITLE_MAPPING: Final[dict[str, int]] = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5,
    "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Countess": 8,
    "Ms": 2, "Lady": 8, "Jonkheer": 8, "Don": 8, "Dona": 8,
    "Mme": 3, "Capt": 7, "Sir": 8
}

FAMILY_SIZE_BINS: Final[list[int]] = [0, 1, 4, 11]
FAMILY_SIZE_LABELS: Final[list[str]] = ["Alone", "Small", "Large"]

AGE_BINS: Final[list[int]] = [0, 12, 18, 35, 60, 100]
AGE_LABELS: Final[list[str]] = ["Child", "Teen", "YoungAdult", "MiddleAged", "Senior"]

FARE_BINS: Final[list[int]] = [0, 7.91, 14.45, 31.0, 512.33]
FARE_LABELS: Final[list[str]] = ["Low", "Medium", "High", "VeryHigh"]


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """Comprehensive preprocessor for Titanic dataset using Polars for efficiency."""
    
    def __init__(
        self,
        handle_missing: bool = True,
        create_features: bool = True,
        scale_features: bool = True,
        encode_categorical: bool = True,
    ) -> None:
        """Initialize the preprocessor with configuration options.
        
        Args:
            handle_missing: Whether to handle missing values.
            create_features: Whether to create engineered features.
            scale_features: Whether to scale numerical features.
            encode_categorical: Whether to encode categorical features.
        """
        self.handle_missing = handle_missing
        self.create_features = create_features
        self.scale_features = scale_features
        self.encode_categorical = encode_categorical
        
        # Initialize transformers
        self.age_imputer: SimpleImputer | None = None
        self.fare_imputer: SimpleImputer | None = None
        self.embarked_imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        
        # Store feature names for consistency
        self.feature_names_: list[str] = []
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        
        logger.info(f"TitanicPreprocessor initialized with options: "
                   f"missing={handle_missing}, features={create_features}, "
                   f"scale={scale_features}, encode={encode_categorical}")
    
    def fit(self, dataframe: pl.DataFrame, target: str | None = None) -> "TitanicPreprocessor":
        """Fit the preprocessor on the training data.
        
        Args:
            dataframe: Input DataFrame.
            target: Name of target column (optional).
            
        Returns:
            Fitted preprocessor instance.
        """
        logger.info(f"Fitting preprocessor on data with shape {dataframe.shape}")
        
        # Create a copy to avoid modifying original data
        df_processed = dataframe.clone()
        
        # Apply feature engineering first
        if self.create_features:
            df_processed = self._engineer_features(df_processed)
        
        # Convert to pandas for sklearn compatibility during fitting
        pandas_df = df_processed.to_pandas()
        
        # Initialize imputers for missing values
        if self.handle_missing:
            self._fit_imputers(pandas_df)
        
        # Store feature information
        self._identify_feature_types(pandas_df, target)
        
        # Fit encoders for categorical features
        if self.encode_categorical:
            self._fit_label_encoders(pandas_df)
        
        # Fit scaler for numerical features
        if self.scale_features:
            self._fit_scaler(pandas_df)
        
        logger.info(f"Preprocessor fitted successfully. Features: {len(self.feature_names_)}")
        return self
    
    def transform(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Transform the input data using fitted preprocessors.
        
        Args:
            dataframe: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame.
        """
        logger.info(f"Transforming data with shape {dataframe.shape}")
        
        # Create a copy to avoid modifying original data
        df_processed = dataframe.clone()
        
        # Apply feature engineering
        if self.create_features:
            df_processed = self._engineer_features(df_processed)
        
        # Convert to pandas for sklearn transformations
        pandas_df = df_processed.to_pandas()
        
        # Handle missing values
        if self.handle_missing:
            pandas_df = self._transform_missing_values(pandas_df)
        
        # Encode categorical features
        if self.encode_categorical:
            pandas_df = self._transform_categorical_features(pandas_df)
        
        # Select and reorder features to match training
        pandas_df = self._select_features(pandas_df)
        
        # Scale numerical features
        if self.scale_features:
            pandas_df = self._transform_numerical_features(pandas_df)
        
        # Convert back to Polars
        result_df = pl.from_pandas(pandas_df)
        
        logger.info(f"Data transformed successfully. Output shape: {result_df.shape}")
        return result_df
    
    def save(self, file_path: Path) -> None:
        """Save the fitted preprocessor to disk.
        
        Args:
            file_path: Path to save the preprocessor.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path) -> "TitanicPreprocessor":
        """Load a fitted preprocessor from disk.
        
        Args:
            file_path: Path to the saved preprocessor.
            
        Returns:
            Loaded preprocessor instance.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {file_path}")
        
        preprocessor = joblib.load(file_path)
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor
    
    def _engineer_features(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Create engineered features from the raw data.
        
        Args:
            dataframe: Input DataFrame.
            
        Returns:
            DataFrame with engineered features.
        """
        df_features = dataframe.clone()
        
        # Extract title from name
        df_features = df_features.with_columns([
            pl.col("Name").str.extract(r", (.*?)\.").alias("Title")
        ])
        
        # Map titles to numerical values using Polars mapping
        title_mapping_expr = pl.col("Title").replace(TITLE_MAPPING, default=1)
        df_features = df_features.with_columns([
            title_mapping_expr.alias("Title_Numeric")
        ])
        
        # Family size features
        df_features = df_features.with_columns([
            (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
            (pl.col("SibSp") + pl.col("Parch") == 0).cast(pl.Int32).alias("IsAlone")
        ])
        
        # Categorize family size
        family_size_conditions = [
            pl.when(pl.col("FamilySize") == 1).then(pl.lit("Alone")),
            pl.when(pl.col("FamilySize").is_between(2, 4)).then(pl.lit("Small")),
            pl.when(pl.col("FamilySize") > 4).then(pl.lit("Large"))
        ]
        df_features = df_features.with_columns([
            pl.coalesce(family_size_conditions).alias("FamilySize_Category")
        ])
        
        # Extract deck from cabin (first letter)
        df_features = df_features.with_columns([
            pl.col("Cabin").str.slice(0, 1).alias("Deck")
        ])
        
        # Age categories (will be applied after imputation)
        df_features = df_features.with_columns([
            pl.when(pl.col("Age").is_null()).then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("Age") <= 12).then(pl.lit("Child"))
                .when(pl.col("Age") <= 18).then(pl.lit("Teen"))
                .when(pl.col("Age") <= 35).then(pl.lit("YoungAdult"))
                .when(pl.col("Age") <= 60).then(pl.lit("MiddleAged"))
                .otherwise(pl.lit("Senior"))
            ).alias("Age_Category")
        ])
        
        # Fare categories (will be applied after imputation)
        df_features = df_features.with_columns([
            pl.when(pl.col("Fare").is_null()).then(pl.lit(None))
            .otherwise(
                pl.when(pl.col("Fare") <= 7.91).then(pl.lit("Low"))
                .when(pl.col("Fare") <= 14.45).then(pl.lit("Medium"))
                .when(pl.col("Fare") <= 31.0).then(pl.lit("High"))
                .otherwise(pl.lit("VeryHigh"))
            ).alias("Fare_Category")
        ])
        
        # Calculate ticket frequency (how many passengers have the same ticket)
        ticket_counts = df_features.group_by("Ticket").agg(
            pl.count().alias("TicketFrequency")
        )
        df_features = df_features.join(ticket_counts, on="Ticket", how="left")
        
        logger.debug(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def _fit_imputers(self, pandas_df: pd.DataFrame) -> None:
        """Fit imputers for missing values.
        
        Args:
            pandas_df: Pandas DataFrame for sklearn compatibility.
        """
        # Age imputation using median
        if "Age" in pandas_df.columns and pandas_df["Age"].isna().any():
            self.age_imputer = SimpleImputer(strategy="median")
            self.age_imputer.fit(pandas_df[["Age"]])
            logger.debug("Age imputer fitted")
        
        # Fare imputation using median
        if "Fare" in pandas_df.columns and pandas_df["Fare"].isna().any():
            self.fare_imputer = SimpleImputer(strategy="median")
            self.fare_imputer.fit(pandas_df[["Fare"]])
            logger.debug("Fare imputer fitted")
        
        # Embarked imputation using most frequent
        if "Embarked" in pandas_df.columns and pandas_df["Embarked"].isna().any():
            self.embarked_imputer = SimpleImputer(strategy="most_frequent")
            self.embarked_imputer.fit(pandas_df[["Embarked"]])
            logger.debug("Embarked imputer fitted")
    
    def _identify_feature_types(self, pandas_df: pd.DataFrame, target: str | None = None) -> None:
        """Identify and store feature types.
        
        Args:
            pandas_df: Pandas DataFrame.
            target: Target column name to exclude.
        """
        # Exclude target and non-predictive columns
        excluded_columns = {"PassengerId", "Name", "Ticket", "Cabin"}
        if target:
            excluded_columns.add(target)
        
        all_columns = set(pandas_df.columns) - excluded_columns
        
        # Identify numerical features
        numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns
        self.numeric_features_ = [col for col in numeric_columns if col in all_columns]
        
        # Identify categorical features
        categorical_columns = pandas_df.select_dtypes(include=["object", "category"]).columns
        self.categorical_features_ = [col for col in categorical_columns if col in all_columns]
        
        # All features for final selection
        self.feature_names_ = self.numeric_features_ + self.categorical_features_
        
        logger.debug(f"Feature types identified - Numeric: {len(self.numeric_features_)}, "
                    f"Categorical: {len(self.categorical_features_)}")
    
    def _fit_label_encoders(self, pandas_df: pd.DataFrame) -> None:
        """Fit label encoders for categorical features.
        
        Args:
            pandas_df: Pandas DataFrame.
        """
        for column in self.categorical_features_:
            if column in pandas_df.columns:
                # Handle missing values by filling with "Unknown"
                series_filled = pandas_df[column].fillna("Unknown")
                
                self.label_encoders[column] = LabelEncoder()
                self.label_encoders[column].fit(series_filled)
                logger.debug(f"Label encoder fitted for column: {column}")
    
    def _fit_scaler(self, pandas_df: pd.DataFrame) -> None:
        """Fit scaler for numerical features.
        
        Args:
            pandas_df: Pandas DataFrame.
        """
        if self.numeric_features_:
            numeric_data = pandas_df[self.numeric_features_].fillna(0)
            self.scaler = StandardScaler()
            self.scaler.fit(numeric_data)
            logger.debug(f"Scaler fitted for {len(self.numeric_features_)} numerical features")
    
    def _transform_missing_values(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputers.
        
        Args:
            pandas_df: Input DataFrame.
            
        Returns:
            DataFrame with imputed values.
        """
        df_imputed = pandas_df.copy()
        
        # Apply age imputation
        if self.age_imputer is not None and "Age" in df_imputed.columns:
            df_imputed["Age"] = self.age_imputer.transform(df_imputed[["Age"]])[:, 0]
            
            # Update age categories after imputation using pandas cut
            df_imputed["Age_Category"] = pd.cut(
                df_imputed["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
            ).astype(str)
        
        # Apply fare imputation
        if self.fare_imputer is not None and "Fare" in df_imputed.columns:
            df_imputed["Fare"] = self.fare_imputer.transform(df_imputed[["Fare"]])[:, 0]
            
            # Update fare categories after imputation using pandas cut
            df_imputed["Fare_Category"] = pd.cut(
                df_imputed["Fare"], bins=FARE_BINS, labels=FARE_LABELS, include_lowest=True
            ).astype(str)
        
        # Apply embarked imputation
        if self.embarked_imputer is not None and "Embarked" in df_imputed.columns:
            df_imputed["Embarked"] = self.embarked_imputer.transform(df_imputed[["Embarked"]])[:, 0]
        
        # Fill remaining missing values in other categorical columns
        for column in self.categorical_features_:
            if column in df_imputed.columns:
                df_imputed[column] = df_imputed[column].fillna("Unknown")
        
        logger.debug("Missing values transformed")
        return df_imputed
    
    def _transform_categorical_features(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders.
        
        Args:
            pandas_df: Input DataFrame.
            
        Returns:
            DataFrame with encoded categorical features.
        """
        df_encoded = pandas_df.copy()
        
        for column in self.categorical_features_:
            if column in df_encoded.columns and column in self.label_encoders:
                # Handle unseen categories
                series_filled = df_encoded[column].fillna("Unknown")
                
                # Transform known categories, map unknown to 0
                encoder = self.label_encoders[column]
                encoded_values = []
                
                for value in series_filled:
                    if value in encoder.classes_:
                        encoded_values.append(encoder.transform([value])[0])
                    else:
                        encoded_values.append(0)  # Default for unknown categories
                
                df_encoded[column] = encoded_values
                logger.debug(f"Categorical feature '{column}' encoded")
        
        return df_encoded
    
    def _select_features(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """Select and reorder features to match training data.
        
        Args:
            pandas_df: Input DataFrame.
            
        Returns:
            DataFrame with selected features.
        """
        # Select only the features that were present during training
        available_features = [col for col in self.feature_names_ if col in pandas_df.columns]
        
        if len(available_features) != len(self.feature_names_):
            missing_features = set(self.feature_names_) - set(available_features)
            logger.warning(f"Missing features in transform: {missing_features}")
        
        df_selected = pandas_df[available_features].copy()
        logger.debug(f"Features selected: {len(available_features)}")
        
        return df_selected
    
    def _transform_numerical_features(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using fitted scaler.
        
        Args:
            pandas_df: Input DataFrame.
            
        Returns:
            DataFrame with scaled numerical features.
        """
        if self.scaler is not None and self.numeric_features_:
            df_scaled = pandas_df.copy()
            
            # Get numerical columns that are available
            available_numeric = [col for col in self.numeric_features_ if col in df_scaled.columns]
            
            if available_numeric:
                # Fill missing values with 0 before scaling
                numeric_data = df_scaled[available_numeric].fillna(0)
                
                # Apply scaling
                scaled_data = self.scaler.transform(numeric_data)
                
                # Update the dataframe
                for i, column in enumerate(available_numeric):
                    df_scaled[column] = scaled_data[:, i]
                
                logger.debug(f"Numerical features scaled: {len(available_numeric)}")
            
            return df_scaled
        
        return pandas_df
    
    def get_feature_names(self) -> list[str]:
        """Get the names of features output by the preprocessor.
        
        Returns:
            List of feature names.
        """
        return self.feature_names_.copy()