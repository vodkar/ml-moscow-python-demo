from __future__ import annotations

"""Data ingestion utilities for Titanic dataset.

Provides typed helpers to read full DataFrames or iterate chunks for large files.
"""

from pathlib import Path
from typing import Final, Iterator, cast

import pandas as pd

DEFAULT_COLUMNS: Final[tuple[str, ...]] = (
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
)


def read_csv_frame(
    path: Path | str, usecols: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Read a CSV into a DataFrame with selected columns and dtypes.

    Args:
        path: CSV path.
        usecols: Optional tuple of columns; defaults to DEFAULT_COLUMNS.

    Returns:
        DataFrame with selected columns and dtypes enforced.
    """

    selected: list[str] = list(usecols if usecols is not None else DEFAULT_COLUMNS)
    # Cast to satisfy type checker about read_csv return.
    return cast(pd.DataFrame, pd.read_csv(Path(path), usecols=selected))


def iter_csv_chunks(
    path: Path | str, chunk_size: int, usecols: tuple[str, ...] | None = None
) -> Iterator[pd.DataFrame]:
    """Iterate over CSV in chunks to handle large datasets.

    Args:
        path: CSV path.
        chunk_size: Number of rows per chunk.
        usecols: Optional tuple of columns; defaults to DEFAULT_COLUMNS.

    Yields:
        DataFrames of size up to chunk_size.
    """

    selected: list[str] = list(usecols if usecols is not None else DEFAULT_COLUMNS)
    reader = pd.read_csv(Path(path), usecols=selected, chunksize=chunk_size)
    for chunk in cast(Iterator[pd.DataFrame], reader):
        yield chunk
