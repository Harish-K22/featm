"""Shared public parameter, numeric-input and text contracts."""
from __future__ import annotations

import inspect
from collections import Counter
import math
import re
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


def forward_parameters(target, *, positional=1, bound=()):
    """Public names of an explicitly forwarded callable, excluding bound args."""
    return tuple(p.name for p in list(inspect.signature(target).parameters.values())[positional:]
                 if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
                 and not p.name.startswith("_") and p.name not in {*bound, "fit", "state"})


def validate_params(spec, params) -> None:
    """Validate user keys, independently of the internal fit/state protocol."""
    fn = spec.implementation
    parameters = list(inspect.signature(fn).parameters.values())
    inputs = spec.arity if spec.modality == "tabular" else 1
    allowed = set(getattr(fn, "__featm_parameters__", ()))
    allowed.update(spec.default_params)
    allowed.update(spec.required_params_for_batching)
    for parameter in parameters[inputs:]:
        if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            if not parameter.name.startswith("_") and parameter.name not in {"fit", "state"}:
                allowed.add(parameter.name)
    unknown = set(params or {}) - allowed
    if unknown:
        raise ValueError(f"{spec.name}: unknown or internal parameter(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def numeric_series(values, *, finite_fit=False):
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ValueError("Numeric feature received nonnumeric nonmissing values")
    if finite_fit and np.isinf(numeric.to_numpy()).any():
        raise ValueError("Scaler training data must not contain +inf or -inf")
    return numeric


def category_key(value: Any) -> str:
    """Typed category identity; integral floats and integers denote one category."""
    if value is None or value is pd.NA or (pd.api.types.is_scalar(value) and bool(pd.isna(value))):
        return "missing:"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{bool(value)}"
    if isinstance(value, Integral):
        return f"number:{int(value)}"
    if isinstance(value, Real):
        value = float(value)
        if math.isfinite(value) and value.is_integer():
            return f"number:{int(value)}"
        return f"number:{value.hex()}"
    if isinstance(value, str):
        return "str:" + value
    if isinstance(value, bytes):
        return "bytes:" + value.hex()
    raise ValueError(f"Unsupported category type: {type(value).__name__}")


def normalize_text(value: Any) -> str:
    if value is None or value is pd.NA or (pd.api.types.is_scalar(value) and bool(pd.isna(value))):
        return ""
    return str(value)


def word_tokens(value):
    return re.findall(r"\b[\w'-]+\b", normalize_text(value).lower(), flags=re.UNICODE)


def tfidf_names(vocab):
    """Keep readable names, adding vocabulary indices only where necessary."""
    bases = [re.sub(r"[^\w]+", "_", t)[:40] or "term" for t in vocab]
    counts = Counter(bases)
    reserved = set(bases)
    names = []
    for i, base in enumerate(bases):
        name = base
        if counts[base] > 1:
            name = f"{base}__term_{i}"
            while name in reserved:
                name += "_"
        names.append(name)
    return tuple(names)
