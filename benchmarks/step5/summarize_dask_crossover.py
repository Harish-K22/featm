from __future__ import annotations

"""Derives per-(feature, provider, worker_count) Dask vs. local crossover
status from the multi-size step5_dask_crossover_results.csv (Section 8 /
v0.5.1 audit-fix Finding 6). Uses the same left/right-censoring-aware
classification as the GPU crossover fix (recompute_gpu_crossover_summary.py)
-- a feature that is faster (or slower) than local at every measured size is
reported as censored, not as an invented exact crossover.

No speedup target: "Dask never beats local in the measured range" is a
valid, useful status, not a failure.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _classify(pts: list[tuple[int, float]]) -> dict:
    pts = sorted(pts)
    all_win = all(s >= 1.0 for _, s in pts)
    all_lose = all(s < 1.0 for _, s in pts)
    if all_win:
        return {"crossover_status": "dask_always_beats_local_in_measured_range",
                "crossover_upper_bound_n_rows": pts[0][0]}
    if all_lose:
        return {"crossover_status": "dask_never_beats_local_in_measured_range",
                "crossover_lower_bound_n_rows": pts[-1][0]}
    for i in range(len(pts) - 1):
        e_i, s_i = pts[i]; e_j, s_j = pts[i + 1]
        stable = (i + 2 >= len(pts)) or (pts[i + 2][1] >= 1.0)
        if s_i < 1.0 and s_j >= 1.0 and stable:
            return {"crossover_status": "measured_bracket", "last_losing_n_rows": e_i, "first_stable_winning_n_rows": e_j}
    return {"crossover_status": "unstable_no_two_point_crossover"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="step5_final_results/step5_dask_crossover_results.csv")
    ap.add_argument("--out", default="step5_final_results/step5_dask_crossover_summary.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, encoding="utf-8")))
    by_key = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["feature"], r["partition_provider"], r["worker_count"])
        by_key[key].append((int(r["n_rows"]), float(r["speedup_dask_over_local"])))

    out_rows = []
    for (feature, provider, worker_count), pts in sorted(by_key.items()):
        cls = _classify(pts)
        out_rows.append({"feature": feature, "partition_provider": provider, "worker_count": worker_count,
                          "measured_sizes": len(pts), "measured_n_rows": ";".join(str(n) for n, _ in sorted(pts)),
                          **cls})
        print(feature, provider, worker_count, "->", cls["crossover_status"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in out_rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(out_rows)
    print(f"Wrote {out} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
