from __future__ import annotations

"""Aggregates step5_engine_feature_results.csv into step5_engine_family_summary.csv.

Family aggregation is honest median-of-raw-rows across every feature in the
family (never a single representative), matching summarize.py's philosophy
for the CPU-lane results.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="step5_results/step5_engine_feature_results.csv")
    ap.add_argument("--out", default="step5_results/step5_engine_family_summary.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    by_feat = defaultdict(list)
    for r in rows:
        by_feat[r["feature"]].append(r)
    fam = defaultdict(list)
    for feat, rs in by_feat.items():
        first = rs[0]
        fam[(first["modality"], first["family"])].append((feat, rs))

    out_rows = []
    for (mod, fa), items in sorted(fam.items()):
        one_batch = [float(r["median_seconds"]) for _, rs in items for r in rs
                    if r["status"] == "ok" and r["lane"] == "engine_extract_one_batch" and r["median_seconds"]]
        four_way = [float(r["median_seconds"]) for _, rs in items for r in rs
                   if r["status"] == "ok" and r["lane"] == "engine_extract_4way" and r["median_seconds"]]
        out_rows.append({
            "modality": mod, "family": fa, "features": len(items),
            "features_with_any_ok": sum(1 for _, rs in items if any(r["status"] == "ok" for r in rs)),
            "median_one_batch_seconds": (sum(one_batch) / len(one_batch)) if one_batch else "",
            "median_4way_seconds": (sum(four_way) / len(four_way)) if four_way else "",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["modality", "family"])
        w.writeheader(); w.writerows(out_rows)
    print(f"Wrote {out} ({len(out_rows)} families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
