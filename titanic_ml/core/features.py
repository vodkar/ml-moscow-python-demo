from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl


class FeatureEncoder:
    """Fit-once, apply-many encoder built on Polars.

    - Fills missing values
    - Builds deterministic one-hot for categoricals
    - Keeps consistent columns between train and inference
    """

    def __init__(self, numeric: list[str], categorical: list[str], drop: list[str] | None = None):
        self.numeric = numeric
        self.categorical = categorical
        self.drop = drop or []
        self.columns_: list[str] | None = None

    def fit(self, df: pl.DataFrame) -> "FeatureEncoder":
        df = df.drop([c for c in self.drop if c in df.columns])
        df = self._fill_missing(df)
        cat_dummies = self._one_hot(df)
        num_df = df.select([c for c in self.numeric if c in df.columns])
        final = pl.concat([num_df, cat_dummies], how="horizontal")
        self.columns_ = final.columns
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.drop([c for c in self.drop if c in df.columns])
        df = self._fill_missing(df)
        cat_dummies = self._one_hot(df)
        num_df = df.select([c for c in self.numeric if c in df.columns])
        out = pl.concat([num_df, cat_dummies], how="horizontal")
        # align columns
        for col in self.columns_:
            if col not in out.columns:
                out = out.with_columns(pl.lit(0).cast(pl.Float64).alias(col))
        # keep only known columns and order
        out = out.select(self.columns_)
        return out

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)

    def _fill_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        fill_exprs = []
        for c in self.numeric:
            if c in df.columns:
                fill_exprs.append(pl.col(c).fill_null(pl.col(c).median()))
        for c in self.categorical:
            if c in df.columns:
                fill_exprs.append(pl.col(c).fill_null("unknown"))
        if not fill_exprs:
            return df
        return df.with_columns(fill_exprs)

    def _one_hot(self, df: pl.DataFrame) -> pl.DataFrame:
        cat_cols = [c for c in self.categorical if c in df.columns]
        if not cat_cols:
            return pl.DataFrame()
        # cast to string to ensure categories are strings
        df_cast = df.with_columns([pl.col(c).cast(pl.Utf8) for c in cat_cols])
        dummies = df_cast.select(cat_cols).to_dummies(separator="=")
        # convert to float for sklearn compatibility
        dummies = dummies.select([pl.col(c).cast(pl.Float64).alias(c) for c in dummies.columns])
        return dummies

    # Persistence
    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({
            "numeric": self.numeric,
            "categorical": self.categorical,
            "drop": self.drop,
            "columns": self.columns_,
        }))

    @classmethod
    def load(cls, path: Path) -> "FeatureEncoder":
        data = json.loads(path.read_text())
        enc = cls(data["numeric"], data["categorical"], data.get("drop", []))
        enc.columns_ = data.get("columns")
        return enc
