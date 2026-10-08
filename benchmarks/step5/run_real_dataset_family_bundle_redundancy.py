from __future__ import annotations

"""Real public-dataset within-family redundancy (v0.5.1 -> v0.5.2 audit-fix
pass, Finding 5).

Audit finding: family redundancy analysis (see
run_true_family_bundle_redundancy.py) existed ONLY for the 4 controlled
probes -- real datasets had no within-true-family redundancy evidence at
all. This module applies the exact same _analyze(...) logic to the same
real (dataset, modality, family) triples used in
run_real_dataset_family_bundle_utility.py, so every row here traces back
to a real true-family-bundle definition rather than merely avoiding a
semicolon in the family field: the included/excluded feature sets are
recomputed via the identical trial-based _check_eligibility(...) call
with the same role_columns/batch_rows per dataset+family, which is
deterministic, so they always agree with the utility artifact's
included_features for the same (dataset, family) key.
"""

import argparse
import csv
from pathlib import Path

from benchmarks.step5.run_real_dataset_family_bundle_utility import DATASET_FAMILY_SPECS
from benchmarks.step5.run_true_family_bundle_redundancy import _analyze


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_final_results/step5_real_dataset_family_bundle_redundancy_results.csv")
    args = ap.parse_args()

    rows = []
    for builder, family_specs, batch_rows in DATASET_FAMILY_SPECS:
        print(f"loading {builder.__name__} ...", flush=True)
        ds = builder()
        for modality, family, role_columns in family_specs:
            print(f"redundancy: {ds.name} ({modality}.{family}) ...", flush=True)
            rec = _analyze(ds.name, modality, family, ds.frame, role_columns, batch_rows=batch_rows)
            rec["evidence_layer"] = "real_dataset"
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
