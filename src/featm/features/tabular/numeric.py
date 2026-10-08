from __future__ import annotations

import numpy as np
import pandas as pd

from ...registry import Cost, register_feature


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


@register_feature(
    name="tabular.numeric.signed_log1p",
    family="numeric",
    modality="tabular",
    input_types=("numeric",),
    cost=Cost.TINY,
    streamable=True,
    description="Sign-preserving log(1 + abs(x)).",
    tags=("log", "nonlinear", "heavy-tail"),
)
def signed_log1p(s: pd.Series, *, fit: bool = False, state=None):
    x = _numeric(s)
    return np.sign(x) * np.log1p(np.abs(x)), None


@register_feature(
    name="tabular.numeric.sqrt_abs",
    family="numeric",
    modality="tabular",
    input_types=("numeric",),
    cost=Cost.TINY,
    streamable=True,
    description="Square root of absolute magnitude.",
    tags=("root", "nonlinear"),
)
def sqrt_abs(s: pd.Series, *, fit: bool = False, state=None):
    x = _numeric(s)
    return np.sqrt(np.abs(x)), None


@register_feature(
    name="tabular.numeric.square",
    family="numeric",
    modality="tabular",
    input_types=("numeric",),
    cost=Cost.TINY,
    streamable=True,
    description="Second power.",
    tags=("polynomial",),
)
def square(s: pd.Series, *, fit: bool = False, state=None):
    x = _numeric(s)
    return x * x, None


@register_feature(
    name="tabular.numeric.reciprocal_safe",
    family="numeric",
    modality="tabular",
    input_types=("numeric",),
    cost=Cost.TINY,
    streamable=True,
    description="Safe reciprocal with zero mapped to NaN.",
    tags=("ratio", "inverse"),
)
def reciprocal_safe(s: pd.Series, *, fit: bool = False, state=None):
    x = _numeric(s)
    return 1.0 / x.replace(0, np.nan), None
