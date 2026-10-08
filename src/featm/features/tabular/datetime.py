from __future__ import annotations

import numpy as np
import pandas as pd

from ...registry import Cost, register_feature


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


for _name, _attr in [
    ("year", "year"),
    ("month", "month"),
    ("day", "day"),
    ("dayofweek", "dayofweek"),
    ("dayofyear", "dayofyear"),
    ("quarter", "quarter"),
]:
    def _make(name=_name, attr=_attr):
        @register_feature(
            name=f"tabular.datetime.{name}",
            family="datetime",
            modality="tabular",
            input_types=("datetime",),
            cost=Cost.TINY,
            streamable=True,
            description=f"Calendar component: {name}.",
            tags=("calendar",),
        )
        def _feature(s: pd.Series, *, fit: bool = False, state=None):
            x = _dt(s)
            return getattr(x.dt, attr).astype(float), None
        return _feature
    _make()


@register_feature(
    name="tabular.datetime.dayofweek_sin",
    family="datetime",
    modality="tabular",
    input_types=("datetime",),
    cost=Cost.TINY,
    streamable=True,
    description="Cyclical sine encoding of day of week.",
    tags=("cyclical",),
)
def dow_sin(s: pd.Series, *, fit: bool = False, state=None):
    x = _dt(s).dt.dayofweek.astype(float)
    return np.sin(2 * np.pi * x / 7.0), None


@register_feature(
    name="tabular.datetime.dayofweek_cos",
    family="datetime",
    modality="tabular",
    input_types=("datetime",),
    cost=Cost.TINY,
    streamable=True,
    description="Cyclical cosine encoding of day of week.",
    tags=("cyclical",),
)
def dow_cos(s: pd.Series, *, fit: bool = False, state=None):
    x = _dt(s).dt.dayofweek.astype(float)
    return np.cos(2 * np.pi * x / 7.0), None
