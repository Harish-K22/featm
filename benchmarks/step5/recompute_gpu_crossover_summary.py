from __future__ import annotations

"""Recomputes step5_gpu_validation_results.json's crossover_summary from the
EXISTING raw step5_gpu_crossover_results.csv, fixing a real classification
bug found by an independent audit (v0.5.0 -> v0.5.1 audit-fix pass):

The original notebook logic (scripts/build_step5_gpu_crossover_notebook.py,
Section G) looked for the first adjacent pair of measured points that BOTH
show speedup >= 1.0 and reported the SMALLER of the two as
"measured_crossover_elements" with crossover_status="measured" -- but if the
very FIRST two measured points already both win, there is no observed LOSING
point below them, so there is no measured bracket at all. That case was
mislabeled "measured" instead of being reported as left-censored (GPU wins
everywhere actually measured; the true crossover, if any, is smaller than
the smallest workload this run tried).

Independent audit of the real P100 data: 13 of 22 GPU-capable features win
at every measured point (left-censored); 9 have a genuine losing point
below a stable two-point win (a real measured bracket). No new GPU run is
needed to fix this -- it is a pure summary-logic bug over already-real data.

Four honest statuses, replacing the old bare "measured":
  measured_bracket                        -- a genuine losing point exists
                                             below a stable two-point win
  left_censored_gpu_wins_at_smallest_measured_point
  right_censored_gpu_never_wins_in_measured_range
  unstable_no_two_point_crossover         -- wins/losses alternate with no
                                             stable two-point transition
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _classify(pts: list[tuple[int, float]]) -> dict:
    pts = sorted(pts)
    all_win = all(s >= 1.0 for _, s in pts)
    all_lose = all(s < 1.0 for _, s in pts)
    if all_win:
        return {"crossover_status": "left_censored_gpu_wins_at_smallest_measured_point",
                "measured_crossover_elements": None,
                "crossover_upper_bound_elements": pts[0][0],
                "crossover_lower_bound_elements": None}
    if all_lose:
        return {"crossover_status": "right_censored_gpu_never_wins_in_measured_range",
                "measured_crossover_elements": None,
                "crossover_upper_bound_elements": None,
                "crossover_lower_bound_elements": pts[-1][0]}
    # A genuine bracket requires an observed LOSING point immediately followed
    # by two consecutive winning points (so the loss->win transition is real,
    # not a single noisy crossing) -- report the losing/winning boundary.
    for i in range(len(pts) - 1):
        e_i, s_i = pts[i]
        e_j, s_j = pts[i + 1]
        if s_i < 1.0 and s_j >= 1.0:
            # confirm the win is stable: this winning point (or the next one,
            # if it exists) must not immediately lose again
            stable = i + 2 >= len(pts) or pts[i + 2][1] >= 1.0
            if stable:
                return {"crossover_status": "measured_bracket",
                        "measured_crossover_elements": e_j,
                        "last_losing_elements": e_i,
                        "first_stable_winning_elements": e_j,
                        "crossover_upper_bound_elements": None,
                        "crossover_lower_bound_elements": None}
    return {"crossover_status": "unstable_no_two_point_crossover",
            "measured_crossover_elements": None,
            "crossover_upper_bound_elements": None,
            "crossover_lower_bound_elements": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossover-csv", default="step5_results/step5_gpu_crossover_results.csv")
    ap.add_argument("--validation-json", default="step5_results/step5_gpu_validation_results.json")
    ap.add_argument("--out", default="step5_results/step5_gpu_validation_results.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.crossover_csv, encoding="utf-8")))
    by_feat = defaultdict(list)
    thresholds = {}
    for r in rows:
        by_feat[r["feature"]].append((int(r["rows"]) * int(r["width"]), float(r["speedup_e2e"])))
        thresholds[r["feature"]] = int(r["gpu_min_elements_shipped"])

    summary = []
    counts = defaultdict(int)
    for feat, pts in sorted(by_feat.items()):
        cls = _classify(pts)
        counts[cls["crossover_status"]] += 1
        summary.append({"feature": feat, "gpu_min_elements_shipped": thresholds[feat],
                         "measured_points": len(pts), **cls})
        print(f"{feat:45s} {cls['crossover_status']}")

    print()
    print("Corrected classification counts:", dict(counts))

    validation = json.loads(Path(args.validation_json).read_text(encoding="utf-8")) if Path(args.validation_json).exists() else {}
    validation["crossover_summary"] = summary
    validation["crossover_classification_counts"] = dict(counts)
    validation["crossover_classification_fix_note"] = (
        "Recomputed by recompute_gpu_crossover_summary.py (v0.5.1 audit-fix pass) from the same raw "
        "step5_gpu_crossover_results.csv the original v0.5.0 notebook produced -- no new GPU run. Fixes a "
        "left-censoring bug: a feature that wins at its two SMALLEST measured points was previously "
        "mislabeled crossover_status='measured' with measured_crossover_elements=smallest_measured_point, "
        "even though no losing point was ever observed below it."
    )
    out = Path(args.out)
    out.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
