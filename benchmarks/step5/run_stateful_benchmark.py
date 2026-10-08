from __future__ import annotations

"""Step-5 stateful fit/transform benchmark (Section 5).

The Step-5 foundation's worker.py measures stateful features only via a
single-shot ``kernel_fit_transform`` lane (the raw module function's
small-data convenience path, which refits from scratch every call). That
does not exercise, or separately time, the REAL out-of-core two-pass
contract (``featm.state.StatefulEncoder``: ``init_state`` ->
``update_state`` -> ``finalize_state`` -> ``transform_batch``) that
``featm.fit()``/``.transform()`` actually use. This script benchmarks that
real contract directly, for all 7 stateful features, in three separately
timed lanes:

  stateful_fit             -- init_state + update_state + finalize_state
  stateful_frozen_transform -- transform_batch against an already-fitted,
                               already-finalized state (no refitting)
  stateful_fit_plus_transform -- StatefulEncoder.fit_transform_batch, the
                               single-shot combined path

Two orthogonal growth axes, per the task's explicit "do not benchmark only
repeated identical rows" requirement:

  A. row_growth   -- rows increase, category/vocabulary POOL size fixed
  B. state_growth -- rows ~fixed, category/vocabulary POOL size increases
                     (not meaningful for zscore/minmax, which hold O(1)
                     state regardless of cardinality -- explicitly marked
                     not_applicable rather than fabricating a curve)

Also measures, per case: peak RSS delta, a state-size proxy (len of
pickle.dumps(state) -- a generic, encoder-agnostic serialized-size measure),
and probes the StateBudgetExceededError boundary for the encoders that
support ``state_budget_bytes`` (category_frequency + the 4 text encoders).
"""

import argparse
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from featm.state import STATEFUL_ENCODERS, StateBudgetExceededError
from benchmarks.step5.worker import PeakRSS

SEED = 4242
rng = np.random.default_rng(SEED)

# feature -> (kind, transform_params)
FEATURES = {
    "tabular.scaling.zscore": ("numeric", None),
    "tabular.scaling.minmax": ("numeric", None),
    "tabular.encoding.category_frequency": ("categorical", None),
    "text.bm25.bm25": ("text", {"query": "feature graph kernel"}),
    "text.information.surprisal": ("text", None),
    "text.semantics.similarity": ("text", None),
    "text.tfidf.tfidf": ("text", None),
}

ROW_LADDER = [500, 2_000, 8_000, 32_000]
STATE_LADDER = [10, 100, 1_000, 10_000]
FIXED_POOL_FOR_ROW_GROWTH = 50   # categories / distinct words
FIXED_ROWS_FOR_STATE_GROWTH = 4_000


def _numeric_series(n: int) -> pd.Series:
    return pd.Series(rng.normal(size=n))


def _categorical_series(n: int, pool: int) -> pd.Series:
    cats = [f"cat_{i}" for i in range(pool)]
    return pd.Series(rng.choice(cats, size=n))


def _text_series(n: int, vocab: int) -> pd.Series:
    words = [f"word{i}" for i in range(max(1, vocab))]
    docs = [" ".join(rng.choice(words, size=6)) for _ in range(n)]
    return pd.Series(docs)


def _build_series(kind: str, n: int, pool: int) -> pd.Series:
    if kind == "numeric":
        return _numeric_series(n)
    if kind == "categorical":
        return _categorical_series(n, pool)
    return _text_series(n, pool)


def _state_bytes(state) -> int:
    try:
        return len(pickle.dumps(state))
    except Exception:
        return -1


def _median(times: list[float]) -> float:
    t = sorted(times)
    return t[len(t) // 2]


def _time_lane(fn, *, warmup: int = 1, reps: int = 3) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    times = []
    peak = 0
    for _ in range(reps):
        with PeakRSS() as rss:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        peak = max(peak, rss.delta)
    return _median(times), peak


def _bench_case(feature: str, kind: str, series: pd.Series, params, axis: str, axis_value: int) -> dict:
    encoder = STATEFUL_ENCODERS[feature]
    spec = featm.REGISTRY.get(feature)
    n = len(series)

    def do_fit():
        state = encoder.init_state(params=params)
        state = encoder.update_state(state, series, params=params)
        return encoder.finalize_state(state, params=params)

    fit_median, fit_peak = _time_lane(do_fit, warmup=1, reps=3)
    fitted_state = do_fit()
    state_bytes = _state_bytes(fitted_state)

    def do_transform():
        return encoder.transform_batch(series, fitted_state, params=params)

    transform_median, transform_peak = _time_lane(do_transform, warmup=1, reps=3)

    def do_fit_transform():
        return encoder.fit_transform_batch(series, params=params)

    combined_median, combined_peak = _time_lane(do_fit_transform, warmup=1, reps=3)

    return {
        "feature": feature, "modality": spec.modality, "family": spec.family, "kind": kind,
        "growth_axis": axis, "axis_value": axis_value, "rows": n,
        "stateful_fit_median_seconds": fit_median, "stateful_fit_peak_rss_delta_bytes": fit_peak,
        "stateful_frozen_transform_median_seconds": transform_median, "stateful_frozen_transform_peak_rss_delta_bytes": transform_peak,
        "stateful_fit_plus_transform_median_seconds": combined_median, "stateful_fit_plus_transform_peak_rss_delta_bytes": combined_peak,
        "fitted_state_bytes": state_bytes, "status": "ok", "error_type": "", "error_message": "",
    }


def _probe_budget_boundary(feature: str, kind: str, params) -> dict:
    """Grow the state pool at a small fixed byte budget until StateBudgetExceededError
    fires, recording the boundary. zscore/minmax (O(1) state, budget-agnostic) are
    reported not_applicable rather than a fabricated boundary."""
    encoder = STATEFUL_ENCODERS[feature]
    budget_bytes = 4_000
    if kind == "numeric":
        return {"feature": feature, "budget_bytes": budget_bytes, "boundary_pool_size": None,
                "status": "not_applicable", "notes": "O(1) state regardless of cardinality; state_budget_bytes has no effect"}
    for pool in (10, 50, 200, 1_000, 5_000, 20_000, 100_000):
        series = _build_series(kind, n=max(pool * 3, 200), pool=pool)
        try:
            state = encoder.init_state(state_budget_bytes=budget_bytes, params=params)
            encoder.update_state(state, series, params=params)
        except StateBudgetExceededError as exc:
            return {"feature": feature, "budget_bytes": budget_bytes, "boundary_pool_size": pool,
                    "status": "boundary_found", "notes": str(exc)[:200]}
    return {"feature": feature, "budget_bytes": budget_bytes, "boundary_pool_size": None,
            "status": "boundary_not_reached_within_probe_range", "notes": "no StateBudgetExceededError up to pool=100000"}


FIELDS = ["feature", "modality", "family", "kind", "growth_axis", "axis_value", "rows",
          "stateful_fit_median_seconds", "stateful_fit_peak_rss_delta_bytes",
          "stateful_frozen_transform_median_seconds", "stateful_frozen_transform_peak_rss_delta_bytes",
          "stateful_fit_plus_transform_median_seconds", "stateful_fit_plus_transform_peak_rss_delta_bytes",
          "fitted_state_bytes", "status", "error_type", "error_message"]

BOUNDARY_FIELDS = ["feature", "budget_bytes", "boundary_pool_size", "status", "notes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-transform-out", default="step5_results/step5_stateful_fit_transform_results.csv")
    ap.add_argument("--state-growth-out", default="step5_results/step5_stateful_state_growth_results.csv")
    args = ap.parse_args()

    rows: list[dict] = []
    for feature, (kind, params) in FEATURES.items():
        print(f"=== {feature} ({kind}) ===", flush=True)
        for n in ROW_LADDER:
            series = _build_series(kind, n=n, pool=FIXED_POOL_FOR_ROW_GROWTH)
            try:
                rec = _bench_case(feature, kind, series, params, "row_growth_fixed_pool", n)
            except Exception as exc:
                rec = {"feature": feature, "modality": "", "family": "", "kind": kind, "growth_axis": "row_growth_fixed_pool",
                       "axis_value": n, "rows": n, "status": "error", "error_type": type(exc).__name__, "error_message": str(exc)[:300]}
            rows.append(rec)
            print(f"  row_growth n={n} -> {rec['status']}", flush=True)

    fit_out = Path(args.fit_transform_out)
    fit_out.parent.mkdir(parents=True, exist_ok=True)
    with fit_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {fit_out} ({len(rows)} rows)")

    growth_rows: list[dict] = []
    for feature, (kind, params) in FEATURES.items():
        if kind == "numeric":
            growth_rows.append({"feature": feature, "modality": featm.REGISTRY.get(feature).modality,
                                "family": featm.REGISTRY.get(feature).family, "kind": kind,
                                "growth_axis": "state_growth_fixed_rows", "axis_value": 0, "rows": FIXED_ROWS_FOR_STATE_GROWTH,
                                "status": "not_applicable", "error_type": "", "error_message": "O(1) state regardless of cardinality"})
            continue
        for pool in STATE_LADDER:
            series = _build_series(kind, n=FIXED_ROWS_FOR_STATE_GROWTH, pool=pool)
            try:
                rec = _bench_case(feature, kind, series, params, "state_growth_fixed_rows", pool)
            except Exception as exc:
                rec = {"feature": feature, "modality": "", "family": "", "kind": kind, "growth_axis": "state_growth_fixed_rows",
                       "axis_value": pool, "rows": FIXED_ROWS_FOR_STATE_GROWTH, "status": "error",
                       "error_type": type(exc).__name__, "error_message": str(exc)[:300]}
            growth_rows.append(rec)
            print(f"  state_growth pool={pool} -> {rec['status']}", flush=True)

    boundary_rows = [_probe_budget_boundary(feature, kind, params) for feature, (kind, params) in FEATURES.items()]
    for r in boundary_rows:
        print(f"  budget boundary {r['feature']}: {r['status']} pool={r['boundary_pool_size']}", flush=True)

    growth_out = Path(args.state_growth_out)
    with growth_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(growth_rows)
    boundary_out = growth_out.with_name(growth_out.stem + "_budget_boundary.csv")
    with boundary_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BOUNDARY_FIELDS); w.writeheader(); w.writerows(boundary_rows)
    print(f"Wrote {growth_out} ({len(growth_rows)} rows) and {boundary_out} ({len(boundary_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
