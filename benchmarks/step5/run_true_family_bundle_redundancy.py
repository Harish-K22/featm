from __future__ import annotations

"""Redundancy analysis on TRUE single-family bundles (v0.5.0 -> v0.5.1 audit-fix pass).

Audit finding: the v0.5.0 redundancy artifact analyzed the same cross-family
"probe bundles" Finding 3 renamed -- e.g. the graph row combined density,
average_degree, clustering, transitivity, and assortativity across FOUR
different graph families, so a correlation between density and
average_degree was reported as if it were within-family redundancy, when it
is cross-family. This script instead reuses
run_true_family_bundle_utility.py's exact eligibility/inclusion logic (same
included-features set, per family) and computes redundancy WITHIN each true
single-family bundle only.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from benchmarks.step5.run_true_family_bundle_utility import (
    FAMILY_BUNDLE_SPECS, _check_eligibility, _probe_columns_for, _BM25_PARAMS, _family_members, _try_probe_import,
)


def _expand_family_bundle(frame: pd.DataFrame, included: list[str], columns_by_feature: dict,
                           batch_rows: int | None = None) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    for feature in included:
        spec = featm.REGISTRY.get(feature)
        params = {feature: _BM25_PARAMS[feature]} if feature in _BM25_PARAMS else None
        if spec.stateful:
            fitted = featm.fit(frame, features=[feature], columns=columns_by_feature[feature], params=params, errors="raise", batch_rows=batch_rows)
            out = fitted.transform(frame, output="memory", batch_rows=batch_rows).to_pandas()
        else:
            out = featm.extract(frame, features=[feature], columns=columns_by_feature[feature], params=params, errors="raise", output="memory", batch_rows=batch_rows).to_pandas()
        for col in out.columns:
            parts.append(out[col].to_numpy(dtype=float))
            names.append(f"{feature}::{col}")
    return (np.column_stack(parts), names) if parts else (np.empty((len(frame), 0)), [])


def _analyze(name: str, modality: str, family: str, frame: pd.DataFrame, role_columns: dict,
             batch_rows: int | None = None) -> dict:
    members = _family_members(modality, family)
    included, columns_by_feature = [], {}
    for feature in members:
        ok, _, columns = _check_eligibility(feature, frame, role_columns, batch_rows=batch_rows)
        if ok:
            included.append(feature); columns_by_feature[feature] = columns
    base = {"dataset_or_probe": name, "modality": modality, "family": family,
            "included_features": ";".join(included), "n_features": len(included)}
    if not included:
        return {**base, "status": "not_utility_eligible", "reason": "no eligible members"}
    try:
        X, colnames = _expand_family_bundle(frame, included, columns_by_feature, batch_rows=batch_rows)
    except Exception as exc:
        return {**base, "status": "not_utility_eligible", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
    n_rows, n_cols = X.shape
    if n_cols == 0:
        return {**base, "status": "not_utility_eligible", "reason": "zero output columns"}
    finite = np.isfinite(X)
    nonfinite_rate = float(1 - finite.sum() / X.size) if X.size else 0.0
    stds = np.nanstd(np.where(finite, X, np.nan), axis=0)
    constant_cols = [colnames[i] for i in range(n_cols) if not np.isfinite(stds[i]) or stds[i] < 1e-9]
    df = pd.DataFrame(X, columns=colnames)
    pearson = df.corr(method="pearson").to_numpy()
    spearman = df.corr(method="spearman").to_numpy()
    iu = np.triu_indices(n_cols, k=1)
    n_pairs = len(iu[0])
    dup95 = dup99 = 0
    if n_pairs:
        abs_pearson = np.nan_to_num(np.abs(pearson[iu]), nan=0.0)
        dup95 = int(np.sum(abs_pearson >= 0.95)); dup99 = int(np.sum(abs_pearson >= 0.99))
    return {
        **base, "n_output_columns": n_cols, "n_pairs": n_pairs,
        "near_duplicate_pairs_0_95": dup95, "near_duplicate_pairs_0_99": dup99,
        "near_duplicate_fraction_0_95": (dup95 / n_pairs) if n_pairs else 0.0,
        "near_duplicate_fraction_0_99": (dup99 / n_pairs) if n_pairs else 0.0,
        "constant_or_near_constant_columns": len(constant_cols),
        "constant_column_names": ";".join(constant_cols)[:500],
        "nonfinite_rate": nonfinite_rate,
        "median_abs_pearson_offdiag": float(np.nanmedian(np.abs(pearson[iu]))) if n_pairs else None,
        "median_abs_spearman_offdiag": float(np.nanmedian(np.abs(spearman[iu]))) if n_pairs else None,
        "status": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_final_results/step5_controlled_family_bundle_redundancy_results.csv")
    args = ap.parse_args()

    up = _try_probe_import()
    probes_by_name = {
        "tabular_controlled": up._tabular_probe(),
        "timeseries_frequency_controlled": up._timeseries_probe(),
        "text_lexical_word_distribution_controlled": up._text_lexical_probe(),
        "graph_topology_class_controlled": up._graph_topology_probe(),
    }

    rows = []
    for name, modality, family, role_columns in FAMILY_BUNDLE_SPECS:
        probe = probes_by_name[name]
        print(f"redundancy: {name} ({modality}.{family}) ...", flush=True)
        rec = _analyze(name, modality, family, probe.frame, role_columns)
        rec["evidence_layer"] = "controlled"
        rows.append(rec)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
