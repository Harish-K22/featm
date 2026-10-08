from __future__ import annotations

import numpy as np
import pandas as pd

from ...registry import Cost, register_feature


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


@register_feature(
    name="tabular.interactions.product",
    family="interactions",
    modality="tabular",
    input_types=("numeric", "numeric"),
    arity=2,
    cost=Cost.LOW,
    streamable=True,
    description="Pairwise numeric product.",
    tags=("pairwise", "multiplicative"),
)
def product(a: pd.Series, b: pd.Series, *, fit: bool = False, state=None):
    return _num(a) * _num(b), None


@register_feature(
    name="tabular.interactions.ratio_safe",
    family="interactions",
    modality="tabular",
    input_types=("numeric", "numeric"),
    arity=2,
    cost=Cost.LOW,
    streamable=True,
    description="Pairwise numeric ratio with zero denominator mapped to NaN.",
    tags=("pairwise", "ratio"),
)
def ratio(a: pd.Series, b: pd.Series, *, fit: bool = False, state=None):
    den = _num(b).replace(0, np.nan)
    return _num(a) / den, None
