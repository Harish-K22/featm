from __future__ import annotations

"""Step-5 metadata-vs-observed audit (Section 12).

Purely diagnostic: compares declared registry metadata (cost/complexity/
memory_profile/gpu_available) against what Layer A (CPU) and Layer C (GPU)
actually measured, and flags contradictions for human review. This script
NEVER edits registry metadata itself -- per the task's explicit rule, any
metadata correction needs a reproducer, >=3 meaningful sizes, an observed
slope, and an implementation-level explanation, and must be a separate,
disclosed change, never auto-applied here.

Noise-floor guard: a feature whose median latency never exceeds
``MIN_RELIABLE_SECONDS`` at any measured scale gets its empirical exponent
marked low_confidence rather than compared against declared complexity --
sub-millisecond measurements are dominated by fixed overhead/timer
granularity, not the algorithm's real scaling behavior.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

MIN_RELIABLE_SECONDS = 0.001  # 1ms noise floor

_COMPLEXITY_EXPONENT_RANGE = {
    "O(n)": (0.5, 1.5), "O(n log n)": (0.6, 1.8), "O(n^2)": (1.5, 2.5), "O(n^3)": (2.3, 3.7),
}
_MEMORY_EXPONENT_RANGE = {
    "constant": (-0.5, 0.6), "linear": (0.5, 1.6), "log_linear": (0.5, 1.8), "quadratic": (1.4, 2.6),
}


def _f(v):
    try:
        return float(v)
    except Exception:
        return math.nan


def _slope(pts: list[tuple[float, float]]) -> float:
    pts = sorted((x, y) for x, y in pts if x > 0 and y > 0)
    if len(pts) < 2:
        return math.nan
    slopes = [math.log(y2 / y1) / math.log(x2 / x1) for (x1, y1), (x2, y2) in zip(pts, pts[1:]) if x2 > x1]
    return statistics.median(slopes) if slopes else math.nan


def _load_cpu_rows(path: str) -> dict[str, list[dict]]:
    by_feat = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        by_feat[r["feature"]].append(r)
    return by_feat


def _load_gpu_crossover(path: str | None) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(r["feature"], []).append(r)
    return out


def _contradiction(observed: float, declared_range: tuple[float, float] | None) -> str:
    if declared_range is None or math.isnan(observed):
        return ""
    lo, hi = declared_range
    if observed < lo - 0.3 or observed > hi + 0.3:
        return f"observed {observed:.2f} outside expected [{lo},{hi}]"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-results", default="step5_results/step5_cpu_system_feature_results.csv")
    ap.add_argument("--gpu-crossover", default="step5_results/step5_gpu_crossover_results.csv")
    ap.add_argument("--out", default="step5_results/step5_metadata_observed_audit.csv")
    args = ap.parse_args()

    by_feat = _load_cpu_rows(args.cpu_results)
    gpu_by_feat = _load_gpu_crossover(args.gpu_crossover)

    rows = []
    for feature, recs in sorted(by_feat.items()):
        ok = [r for r in recs if r["status"] == "ok"]
        first = recs[0]
        declared_cost = first.get("cost", "")
        declared_complexity = first.get("complexity", "")
        declared_memory = first.get("memory_profile", "")
        gpu_available = first.get("gpu_available", "") in ("True", "true", "1")

        time_pts = [(_f(r["input_elements"]), _f(r["median_seconds"])) for r in ok]
        rss_pts = [(_f(r["input_elements"]), _f(r["peak_rss_delta_bytes"])) for r in ok]
        max_median = max((_f(r["median_seconds"]) for r in ok), default=math.nan)
        low_confidence = math.isnan(max_median) or max_median < MIN_RELIABLE_SECONDS

        observed_time_exp = _slope(time_pts)
        observed_rss_exp = _slope(rss_pts)

        time_note = "" if low_confidence else _contradiction(observed_time_exp, _COMPLEXITY_EXPONENT_RANGE.get(declared_complexity))
        rss_note = "" if low_confidence else _contradiction(observed_rss_exp, _MEMORY_EXPONENT_RANGE.get(declared_memory))

        gpu_note = ""
        measured_crossover = ""
        if gpu_available:
            crossover_recs = gpu_by_feat.get(feature, [])
            if crossover_recs:
                wins = [r for r in crossover_recs if _f(r.get("speedup_e2e", "0")) >= 1.0]
                measured_crossover = "measured" if crossover_recs else ""
                if not wins:
                    gpu_note = "gpu_available=True but no measured workload showed speedup_e2e>=1.0 in the crossover sweep"
            else:
                gpu_note = "gpu_available=True but no Layer-C crossover evidence found for this feature"

        notes = "; ".join(n for n in (
            "low_confidence_sub_ms_noise" if low_confidence else "",
            time_note, rss_note, gpu_note,
        ) if n)

        rows.append({
            "feature": feature, "declared_cost": declared_cost, "declared_complexity": declared_complexity,
            "observed_time_exponent": round(observed_time_exp, 3) if not math.isnan(observed_time_exp) else "",
            "declared_memory_profile": declared_memory,
            "observed_rss_exponent": round(observed_rss_exp, 3) if not math.isnan(observed_rss_exp) else "",
            "max_median_seconds_observed": round(max_median, 6) if not math.isnan(max_median) else "",
            "noise_floor_low_confidence": low_confidence,
            "gpu_available_declared": gpu_available, "gpu_crossover_evidence": measured_crossover,
            "contradiction_notes": notes,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["feature", "declared_cost", "declared_complexity", "observed_time_exponent", "declared_memory_profile",
              "observed_rss_exponent", "max_median_seconds_observed", "noise_floor_low_confidence",
              "gpu_available_declared", "gpu_crossover_evidence", "contradiction_notes"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    n_flagged = sum(1 for r in rows if r["contradiction_notes"] and "low_confidence_sub_ms_noise" != r["contradiction_notes"])
    n_with_any_note = sum(1 for r in rows if r["contradiction_notes"])
    print(f"Wrote {out} ({len(rows)} features, {n_with_any_note} with any note, {n_flagged} with a real (non-noise-only) flag)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
