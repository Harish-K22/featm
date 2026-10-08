from __future__ import annotations

"""True single-family utility bundle evaluator (v0.5.0 -> v0.5.1 audit-fix pass).

Audit finding: the v0.5.0 "family bundle" evaluator concatenated the probe's
manually-curated cross-family feature list (e.g. graph bundle =
assortativity;clustering;connectivity;degree) and called the result a
"family bundle" -- it is not. A family bundle, per the Step-5 contract, is
ALL utility-eligible members of ONE (modality, family), concatenated.

For each (probe/dataset, modality, family) entry below:
  1. enumerate every registry member of that EXACT family;
  2. determine eligibility by actually trying to extract/fit it on the real
     dataset (never a hand-maintained compatibility table) -- if it raises,
     or produces output not row-aligned with the input, it is excluded with
     the caught reason;
  3. include every eligible member, using ALL of its real output columns;
  4. fit stateful members train-fold-only (same leakage-safe pattern as
     utility_probes.py's _feature_matrix, reused directly);
  5. evaluate the concatenated matrix with the same fixed downstream
     model/CV policy as the rest of Layer E.

Each result row has exactly ONE family -- never a semicolon-joined list.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from benchmarks.step5.utility_probes import _score_matrix


def _try_probe_import():
    # utility_probes.py's real dataset builders are network/dependency-heavy;
    # import lazily so a family-bundle-only run doesn't require them.
    from benchmarks.step5 import utility_probes as up
    return up


# (probe_or_dataset_builder_name, modality, family, {role: column_name})
# role is only used for tabular features (numeric/categorical); other
# modalities' families take the feature's default single input column.
FAMILY_BUNDLE_SPECS = [
    ("tabular_controlled", "tabular", "scaling", {"numeric": "x"}),
    ("tabular_controlled", "tabular", "encoding", {"categorical": "cat"}),
    ("tabular_controlled", "tabular", "numeric", {"numeric": "x"}),
    ("timeseries_frequency_controlled", "timeseries", "statistical", {}),
    ("timeseries_frequency_controlled", "timeseries", "spectral", {}),
    ("text_lexical_word_distribution_controlled", "text", "tfidf", {}),
    ("text_lexical_word_distribution_controlled", "text", "lexical", {}),
    ("text_lexical_word_distribution_controlled", "text", "bm25", {}),
    ("graph_topology_class_controlled", "graph", "connectivity", {}),
    ("graph_topology_class_controlled", "graph", "degree", {}),
    ("graph_topology_class_controlled", "graph", "clustering", {}),
    ("graph_topology_class_controlled", "graph", "assortativity", {}),
]

_BM25_PARAMS = {"text.bm25.bm25": {"query": "feature graph kernel"}}


def _family_members(modality: str, family: str) -> list[str]:
    return sorted(s.name for s in featm.REGISTRY.all() if s.modality == modality and s.family == family)


def _probe_columns_for(feature: str, role_columns: dict) -> dict | None:
    spec = featm.REGISTRY.get(feature)
    if not role_columns:
        return None
    role = "categorical" if spec.input_types and spec.input_types[0] == "categorical" else "numeric"
    col = role_columns.get(role)
    return {feature: col} if col else None


def _check_eligibility(feature: str, frame: pd.DataFrame, role_columns: dict, batch_rows: int | None = None) -> tuple[bool, str, dict | None]:
    """Trial-run the feature on the REAL frame (never a hand-maintained
    compatibility table) to decide eligibility. Returns (eligible, reason, columns_used)."""
    spec = featm.REGISTRY.get(feature)
    params = {feature: _BM25_PARAMS[feature]} if feature in _BM25_PARAMS else None
    columns = _probe_columns_for(feature, role_columns)
    try:
        if spec.stateful:
            fitted = featm.fit(frame, features=[feature], columns=columns, params=params, errors="raise", batch_rows=batch_rows)
            out = fitted.transform(frame, output="memory", batch_rows=batch_rows).to_pandas()
        else:
            out = featm.extract(frame, features=[feature], columns=columns, params=params, errors="raise", output="memory", batch_rows=batch_rows).to_pandas()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:200]}", columns
    if out.shape[0] != len(frame):
        return False, f"output not row-aligned: {out.shape[0]} rows for {len(frame)} input rows", columns
    if out.shape[1] == 0:
        return False, "zero output columns", columns
    return True, "", columns


def _feature_matrix_true(feature: str, columns: dict | None, params: dict | None, train: pd.DataFrame, test: pd.DataFrame,
                          batch_rows: int | None = None):
    spec = featm.REGISTRY.get(feature)
    if spec.stateful:
        fitted = featm.fit(train, features=[feature], columns=columns, params=params, errors="raise", batch_rows=batch_rows)
        Xtr = fitted.transform(train, output="memory", batch_rows=batch_rows).to_pandas().to_numpy()
        Xte = fitted.transform(test, output="memory", batch_rows=batch_rows).to_pandas().to_numpy()
    else:
        Xtr = featm.extract(train, features=[feature], columns=columns, params=params, errors="raise", output="memory", batch_rows=batch_rows).to_pandas().to_numpy()
        Xte = featm.extract(test, features=[feature], columns=columns, params=params, errors="raise", output="memory", batch_rows=batch_rows).to_pandas().to_numpy()
    return Xtr, Xte


def evaluate_true_family_bundle(name: str, task: str, frame: pd.DataFrame, y: np.ndarray,
                                 modality: str, family: str, role_columns: dict, folds: int = 5,
                                 batch_rows: int | None = None) -> dict:
    members = _family_members(modality, family)
    included, excluded, columns_by_feature = [], [], {}
    n_output_columns = 0
    for feature in members:
        ok, reason, columns = _check_eligibility(feature, frame, role_columns, batch_rows=batch_rows)
        if ok:
            included.append(feature)
            columns_by_feature[feature] = columns
        else:
            excluded.append((feature, reason))

    base = {
        "dataset_or_probe": name, "modality": modality, "family": family,
        "n_registry_family_features": len(members),
        "included_features": ";".join(included), "n_included_features": len(included),
        "excluded_features": ";".join(f for f, _ in excluded),
        "exclusion_reasons": " | ".join(f"{f}: {r}" for f, r in excluded),
        "folds": folds,
    }
    if not included:
        return {**base, "n_output_columns": 0, "status": "not_utility_eligible",
                "reason": "no eligible members in this family for this dataset"}

    try:
        from sklearn.model_selection import KFold, StratifiedKFold
        splitter = StratifiedKFold(folds, shuffle=True, random_state=123) if task == "classification" else KFold(folds, shuffle=True, random_state=123)
        split_args = (frame, y) if task == "classification" else (frame,)
        scores, rmses, r2s = [], [], []
        n_cols_seen = 0
        for tr, te in splitter.split(*split_args):
            train = frame.iloc[tr].reset_index(drop=True)
            test = frame.iloc[te].reset_index(drop=True)
            Xtr_parts, Xte_parts = [], []
            for feature in included:
                params = {feature: _BM25_PARAMS[feature]} if feature in _BM25_PARAMS else None
                Xtr, Xte = _feature_matrix_true(feature, columns_by_feature[feature], params, train, test, batch_rows=batch_rows)
                Xtr_parts.append(Xtr); Xte_parts.append(Xte)
            Xtr_all = np.concatenate(Xtr_parts, axis=1); Xte_all = np.concatenate(Xte_parts, axis=1)
            n_cols_seen = Xtr_all.shape[1]
            if task == "classification":
                scores.append(_score_matrix(Xtr_all, Xte_all, y[tr], y[te]))
            else:
                from sklearn.impute import SimpleImputer
                from sklearn.linear_model import Ridge
                from sklearn.metrics import mean_squared_error, r2_score
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
                model.fit(Xtr_all, y[tr]); p = model.predict(Xte_all)
                rmses.append(float(mean_squared_error(y[te], p) ** 0.5)); r2s.append(float(r2_score(y[te], p)))
        if task == "classification":
            return {**base, "n_output_columns": n_cols_seen, "metric": "roc_auc",
                    "mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)), "status": "ok"}
        return {**base, "n_output_columns": n_cols_seen, "metric": "rmse+r2",
                "mean_score": float(np.mean(rmses)), "std_score": float(np.std(rmses)),
                "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)), "status": "ok"}
    except Exception as exc:
        return {**base, "n_output_columns": 0, "status": "not_utility_eligible",
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_final_results/step5_controlled_family_bundle_utility_results.csv")
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
        print(f"family bundle: {name} ({modality}.{family}) ...", flush=True)
        rec = evaluate_true_family_bundle(name, probe.task, probe.frame, probe.y, modality, family, role_columns)
        rec["evidence_layer"] = "controlled"
        rows.append(rec)
        print(f"  -> included={rec['n_included_features']}/{rec['n_registry_family_features']} status={rec['status']} score={rec.get('mean_score')}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
