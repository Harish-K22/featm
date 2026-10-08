from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd


class SemanticType(str, Enum):
    CONTINUOUS = "continuous"
    COUNT = "count"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    HIGH_CARDINALITY = "high_cardinality_categorical"
    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    semantic_type: SemanticType
    dtype: str
    null_fraction: float
    unique_fraction: float
    n_unique: int


@dataclass(frozen=True)
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: tuple[ColumnProfile, ...]
    sampled_rows: int

    def names_for(self, *types: SemanticType) -> list[str]:
        allowed = set(types)
        return [c.name for c in self.columns if c.semantic_type in allowed]


_ID_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid|key$)", re.I)


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(200)
    if len(sample) < 3:
        return False
    # Avoid treating arbitrary numeric strings as datetimes.
    if sample.str.fullmatch(r"\d+(\.\d+)?").mean() > 0.8:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    return bool(parsed.notna().mean() >= 0.85)


def _looks_text(s: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return False
    return float(sample.str.len().mean()) >= 30 or float(sample.str.contains(r"\s", regex=True).mean()) >= 0.65


def infer_schema(df: pd.DataFrame, sample_rows: int = 20_000) -> DatasetProfile:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("infer_schema currently expects a pandas DataFrame")
    sample = df if len(df) <= sample_rows else df.sample(sample_rows, random_state=0)
    profiles: list[ColumnProfile] = []
    n = max(len(sample), 1)

    for name in df.columns:
        s = sample[name]
        nonnull = s.dropna()
        n_unique = int(nonnull.nunique(dropna=True))
        unique_fraction = n_unique / max(len(nonnull), 1)
        null_fraction = float(s.isna().mean())

        if _looks_datetime(s):
            sem = SemanticType.DATETIME
        elif pd.api.types.is_bool_dtype(s) or n_unique <= 2:
            sem = SemanticType.BINARY
        elif pd.api.types.is_integer_dtype(s):
            if _ID_RE.search(str(name)) and unique_fraction > 0.8:
                sem = SemanticType.IDENTIFIER
            elif n_unique <= max(20, int(0.01 * n)):
                sem = SemanticType.COUNT
            elif unique_fraction > 0.995 and _ID_RE.search(str(name)):
                sem = SemanticType.IDENTIFIER
            else:
                sem = SemanticType.CONTINUOUS
        elif pd.api.types.is_numeric_dtype(s):
            sem = SemanticType.CONTINUOUS
        elif _looks_text(s):
            sem = SemanticType.TEXT
        else:
            if unique_fraction > 0.98 and n_unique > 100:
                sem = SemanticType.IDENTIFIER
            elif n_unique > 10_000 or unique_fraction > 0.5:
                sem = SemanticType.HIGH_CARDINALITY
            else:
                sem = SemanticType.CATEGORICAL

        profiles.append(
            ColumnProfile(
                name=str(name),
                semantic_type=sem,
                dtype=str(s.dtype),
                null_fraction=null_fraction,
                unique_fraction=unique_fraction,
                n_unique=n_unique,
            )
        )

    return DatasetProfile(
        n_rows=len(df),
        n_columns=df.shape[1],
        columns=tuple(profiles),
        sampled_rows=len(sample),
    )
