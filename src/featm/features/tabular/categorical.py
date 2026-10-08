from __future__ import annotations

import pandas as pd

from ...registry import Cost, register_feature


@register_feature(
    name="tabular.categorical.frequency",
    family="categorical",
    modality="tabular",
    input_types=("categorical",),
    cost=Cost.LOW,
    streamable=True,
    incremental=True,
    description="Training-set category frequency; unseen categories map to zero.",
    tags=("frequency", "high-cardinality", "encoding"),
)
def frequency(s: pd.Series, *, fit: bool = False, state=None):
    key = s.astype("string").fillna("<NA>")
    if fit or state is None:
        counts = key.value_counts(dropna=False)
        mapping = (counts / max(len(key), 1)).to_dict()
    else:
        mapping = state
    out = key.map(mapping).fillna(0.0).astype(float)
    return out, mapping
