from __future__ import annotations

"""Real public-dataset true single-family utility bundles (v0.5.1 -> v0.5.2
audit-fix pass, Finding 4).

Audit finding: the "true family bundle" evaluator (see
run_true_family_bundle_utility.py) existed ONLY for the 4 synthetic
controlled probes. Real public datasets only ever had per-feature
individual utility plus hand-picked cross-family "probe bundles" (e.g.
credit-g's bundle field reads "encoding;numeric;scaling" -- a
cross-family selection, not a family bundle, even though it was stored
next to the word "bundle").

This module reuses evaluate_true_family_bundle(...) UNCHANGED (it is
already generic over any (frame, y, task) triple) and applies it to the
real RealDataset objects built in run_real_dataset_utility.py, using the
minimum per-dataset family coverage specified by the audit:
  - credit-g:            tabular.scaling / tabular.encoding / tabular.numeric
  - California Housing:  tabular.scaling / tabular.numeric
  - 20 Newsgroups:       text.tfidf / text.lexical
  - Digits:              image.color / image.shape / image.texture
  - GunPoint:            timeseries.statistical / timeseries.spectral
  - MUTAG:               graph.connectivity / graph.degree / graph.clustering
  - FSDD:                audio.waveform

Same rules as the controlled version: enumerate ALL registry members of
the exact (modality, family), decide eligibility by actually trying the
extraction/fit on the real frame, record every exclusion with its reason,
use all real output columns, fit stateful members train-fold-only, use
the same downstream model/CV policy, and never report more than one
family per row.
"""

import argparse
import csv
from pathlib import Path

from benchmarks.step5.run_real_dataset_utility import (
    _california_housing,
    _credit_g,
    _digits_binary,
    _fsdd_audio,
    _gunpoint_timeseries,
    _mutag_graph,
    _newsgroups_binary,
)
from benchmarks.step5.run_true_family_bundle_utility import evaluate_true_family_bundle

# (dataset_builder, [(modality, family, role_columns)], batch_rows)
# role_columns picks ONE representative column per role (numeric/categorical)
# so every member of a family trials against the same input column -- the
# same pattern the controlled probes already use (a single "x"/"cat" column
# per role, not a bespoke column per feature).
DATASET_FAMILY_SPECS = [
    (_credit_g, [
        ("tabular", "scaling", {"numeric": "duration", "categorical": "checking_status"}),
        ("tabular", "encoding", {"numeric": "duration", "categorical": "checking_status"}),
        ("tabular", "numeric", {"numeric": "duration", "categorical": "checking_status"}),
    ], None),
    (_california_housing, [
        ("tabular", "scaling", {"numeric": "MedInc"}),
        ("tabular", "numeric", {"numeric": "MedInc"}),
    ], None),
    (_newsgroups_binary, [
        ("text", "tfidf", {}),
        ("text", "lexical", {}),
    ], None),
    (_digits_binary, [
        ("image", "color", {}),
        ("image", "shape", {}),
        ("image", "texture", {}),
    ], None),
    (_gunpoint_timeseries, [
        ("timeseries", "statistical", {}),
        ("timeseries", "spectral", {}),
    ], None),
    (_mutag_graph, [
        ("graph", "connectivity", {}),
        ("graph", "degree", {}),
        ("graph", "clustering", {}),
    ], None),
    (_fsdd_audio, [
        ("audio", "waveform", {}),
    ], 1),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_final_results/step5_real_dataset_family_bundle_utility_results.csv")
    args = ap.parse_args()

    rows = []
    for builder, family_specs, batch_rows in DATASET_FAMILY_SPECS:
        print(f"loading {builder.__name__} ...", flush=True)
        ds = builder()
        for modality, family, role_columns in family_specs:
            print(f"family bundle: {ds.name} ({modality}.{family}) ...", flush=True)
            rec = evaluate_true_family_bundle(ds.name, ds.task, ds.frame, ds.y, modality, family,
                                               role_columns, batch_rows=batch_rows)
            rec["evidence_layer"] = "real_dataset"
            rows.append(rec)
            print(f"  -> included={rec['n_included_features']}/{rec['n_registry_family_features']} "
                  f"status={rec['status']} score={rec.get('mean_score')}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
