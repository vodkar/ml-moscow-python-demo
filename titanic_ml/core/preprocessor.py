"""Data preprocessing and feature engineering."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


class TitanicPreprocessor:
    """Advanced preprocessing pipeline for Titanic data."""

    def __init__(self, use_polars: bool = True):
        self.use_polars = use_polars
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.imputers: Dict[str, Union[SimpleImputer, KNNImputer]] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit_transform(
        self, df: Union[pd.DataFrame, pl.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data."""
        if self.use_polars and isinstance(df, pl.DataFrame):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df.copy()

        X, y = self._extract_features_target(df_pandas)
        X_processed = self._fit_transform_features(X)

        self.is_fitted = True
        logger.info(f"Preprocessor fitted with {X_processed.shape[1]} features")
        return X_processed, y

    def transform(self, df: Union[pd.DataFrame, pl.DataFrame]) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        if self.use_polars and isinstance(df, pl.DataFrame):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df.copy()

        X, _ = self._extract_features_target(df_pandas, has_target=False)
        X_processed = self._transform_features(X)

        return X_processed

    def _extract_features_target(
        self, df: pd.DataFrame, has_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Extract features and target variable."""
        features_to_drop = ["PassengerId", "Name", "Ticket"]

        if has_target and "Survived" in df.columns:
            y = df["Survived"].values
            features_to_drop.append("Survived")
        else:
            y = None

        X = df.drop(columns=features_to_drop, errors="ignore")
        return X, y

    def _fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform features."""
        X = X.copy()

        # Feature engineering
        X = self._engineer_features(X)

        # Handle missing values
        X = self._fit_transform_missing_values(X)

        # Encode categorical variables
        X = self._fit_transform_categorical(X)

        # Scale numerical features
        X = self._fit_transform_numerical(X)

        self.feature_names = list(X.columns)
        return X.values

    def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessor."""
        X = X.copy()

        # Feature engineering
        X = self._engineer_features(X)

        # Handle missing values
        X = self._transform_missing_values(X)

        # Encode categorical variables
        X = self._transform_categorical(X)

        # Scale numerical features
        X = self._transform_numerical(X)

        # Ensure same column order as training
        X = X.reindex(columns=self.feature_names, fill_value=0)

        return X.values

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        X = X.copy()

        # Family size
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        # Title extraction
        if "Name" in X.columns:
            X["Title"] = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
            # Group rare titles
            title_counts = X["Title"].value_counts()
            rare_titles = title_counts[title_counts < 5].index
            X["Title"] = X["Title"].replace(rare_titles, "Rare")

        # Age groups
        if "Age" in X.columns:
            X["AgeGroup"] = pd.cut(
                X["Age"],
                bins=[0, 12, 18, 35, 60, 100],
                labels=["Child", "Teen", "Adult", "Middle", "Senior"],
                include_lowest=True,
            )

        # Fare per person
        if "Fare" in X.columns:
            X["FarePerPerson"] = X["Fare"] / X["FamilySize"]

        # Cabin features
        if "Cabin" in X.columns:
            X["HasCabin"] = X["Cabin"].notna().astype(int)
            X["CabinDeck"] = X["Cabin"].str[0].fillna("Unknown")

        return X

    def _fit_transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit imputers and transform missing values."""
        X = X.copy()

        # Numerical columns - use KNN imputation
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            self.imputers["numerical"] = KNNImputer(n_neighbors=5)
            X[numerical_cols] = self.imputers["numerical"].fit_transform(
                X[numerical_cols]
            )

        # Categorical columns - use mode imputation
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                self.imputers[col] = SimpleImputer(strategy="most_frequent")
                X[col] = self.imputers[col].fit_transform(X[[col]]).ravel()

        return X

    def _transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputers."""
        X = X.copy()

        # Numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols and "numerical" in self.imputers:
            X[numerical_cols] = self.imputers["numerical"].transform(X[numerical_cols])

        # Categorical columns
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        for col in categorical_cols:
            if col in self.imputers:
                X[col] = self.imputers[col].transform(X[[col]]).ravel()

        return X

    def _fit_transform_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders and transform categorical variables."""
        X = X.copy()

        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        return X

    def _transform_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables using fitted encoders."""
        X = X.copy()

        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            if col in self.label_encoders:
                # Handle unknown categories
                X[col] = X[col].astype(str)
                known_labels = set(self.label_encoders[col].classes_)
                X[col] = X[col].apply(lambda x: x if x in known_labels else "Unknown")

                # Add 'Unknown' to encoder if not present
                if "Unknown" not in known_labels:
                    current_classes = list(self.label_encoders[col].classes_)
                    current_classes.append("Unknown")
                    self.label_encoders[col].classes_ = np.array(current_classes)

                X[col] = self.label_encoders[col].transform(X[col])

        return X

    def _fit_transform_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform numerical features."""
        X = X.copy()

        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        return X

    def _transform_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features using fitted scaler."""
        X = X.copy()

        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])

        return X

    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for importance analysis."""
        return self.feature_names.copy() if self.feature_names else []
