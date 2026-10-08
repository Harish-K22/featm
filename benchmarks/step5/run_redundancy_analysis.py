from __future__ import annotations

"""Step-5 redundancy/stability analysis (Section 11).

For each utility-eligible family bundle already proven meaningful by
``utility_probes.py`` (same probes, same feature sets -- redundancy is
about whether a family's OWN output columns are informative and distinct
from each other, not a new dataset), expand every feature to its real
output columns, concatenate the whole bundle into one column-labeled
matrix, and report:

  - pairwise absolute Pearson and Spearman correlation (never collapsed to
    one scalar "redundancy score");
  - near-duplicate-pair fraction at |corr| >= 0.95 and >= 0.99;
  - constant / near-constant columns (std < 1e-9);
  - missing/non-finite rate per column.

Never collapses a vector feature to one arbitrary scalar before computing
this -- every real output column participates in the correlation matrix.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from benchmarks.step5.utility_probes import (
    _graph_topology_probe, _tabular_probe, _text_lexical_probe, _timeseries_probe,
)


def _bundle_matrix(probe) -> tuple[np.ndarray, list[str]]:
    """All of probe.features, ALL real output columns, concatenated -- no
    stateful/leakage concern here since this analyzes the WHOLE frame's own
    internal column redundancy, not a train/test predictive claim."""
    parts = []
    names = []
    for feature in probe.features:
        spec = featm.REGISTRY.get(feature)
        if spec.stateful:
            fitted = featm.fit(probe.frame, features=[feature], columns=probe.columns or None, errors="raise")
            out = fitted.transform(probe.frame, output="memory").to_pandas()
        else:
            out = featm.extract(probe.frame, features=[feature], columns=probe.columns or None, errors="raise", output="memory").to_pandas()
        for col in out.columns:
            parts.append(out[col].to_numpy(dtype=float))
            names.append(f"{feature}::{col}")
    return np.column_stack(parts), names


def _analyze(probe) -> dict:
    try:
        X, names = _bundle_matrix(probe)
    except Exception as exc:
        return {"probe": probe.name, "status": "not_utility_eligible", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
    n_rows, n_cols = X.shape
    finite = np.isfinite(X)
    nonfinite_rate = float(1 - finite.sum() / X.size) if X.size else 0.0
    stds = np.nanstd(np.where(finite, X, np.nan), axis=0)
    constant_cols = [names[i] for i in range(n_cols) if not np.isfinite(stds[i]) or stds[i] < 1e-9]

    df = pd.DataFrame(X, columns=names)
    pearson = df.corr(method="pearson").to_numpy()
    spearman = df.corr(method="spearman").to_numpy()
    iu = np.triu_indices(n_cols, k=1)
    n_pairs = len(iu[0])
    dup95 = dup99 = 0
    if n_pairs:
        abs_pearson = np.abs(pearson[iu])
        dup95 = int(np.sum(np.nan_to_num(abs_pearson, nan=0.0) >= 0.95))
        dup99 = int(np.sum(np.nan_to_num(abs_pearson, nan=0.0) >= 0.99))
    return {
        "probe": probe.name, "status": "ok", "n_rows": n_rows, "n_columns": n_cols,
        "n_pairs": n_pairs, "near_duplicate_pairs_0_95": dup95, "near_duplicate_pairs_0_99": dup99,
        "near_duplicate_fraction_0_95": (dup95 / n_pairs) if n_pairs else 0.0,
        "near_duplicate_fraction_0_99": (dup99 / n_pairs) if n_pairs else 0.0,
        "constant_or_near_constant_columns": len(constant_cols),
        "constant_column_names": ";".join(constant_cols)[:500],
        "nonfinite_rate": nonfinite_rate,
        "median_abs_pearson_offdiag": float(np.nanmedian(np.abs(pearson[iu]))) if n_pairs else None,
        "median_abs_spearman_offdiag": float(np.nanmedian(np.abs(spearman[iu]))) if n_pairs else None,
    }


FIELDS = ["probe", "status", "n_rows", "n_columns", "n_pairs", "near_duplicate_pairs_0_95", "near_duplicate_pairs_0_99",
          "near_duplicate_fraction_0_95", "near_duplicate_fraction_0_99", "constant_or_near_constant_columns",
          "constant_column_names", "nonfinite_rate", "median_abs_pearson_offdiag", "median_abs_spearman_offdiag", "reason"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_results/step5_redundancy_results.csv")
    args = ap.parse_args()
    probes = (_tabular_probe(), _timeseries_probe(), _text_lexical_probe(), _graph_topology_probe())
    rows = []
    for p in probes:
        print(f"redundancy: {p.name} ...", flush=True)
        rows.append(_analyze(p))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
