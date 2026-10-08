from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import featm
from featm.registry import Cost
from benchmarks.step5.provenance import make_provenance, warn_if_mixed_checksums


# Adaptive ladders are deliberately conservative.  Claude's real-environment
# completion pass may calibrate these, but any change must be persisted in the
# run manifest so benchmark comparisons remain reproducible.
SCALES_BY_COST = {
    Cost.TINY: [1, 4, 16],
    Cost.LOW: [1, 4, 16],
    Cost.MEDIUM: [1, 2, 4],
    Cost.HIGH: [1, 2],
    Cost.EXTREME: [1],
}
TIMEOUT_BY_COST = {
    Cost.TINY: 45,
    Cost.LOW: 45,
    Cost.MEDIUM: 90,
    Cost.HIGH: 180,
    Cost.EXTREME: 300,
}

FIELD_ORDER = [
    "feature","modality","family","subfamily","cost","complexity","memory_profile","batch_semantics",
    "stateful","gpu_available","lane","scale","scale_axis","logical_samples","input_elements","uses_dummy_model","status",
    "median_seconds","min_seconds","p90_seconds","peak_rss_delta_bytes","output_rows","output_cols","output_bytes",
    "nonfinite_count","deterministic","digest","error_type","error_message","python","platform",
    "run_id","environment_id","source_checksum_algorithm","featm_source_checksum",
    "measurement_harness_checksum","source_checksum_blake2b16","featm_version",
]


def _load_existing(out: Path) -> tuple[list[dict], set[tuple[str, int]]]:
    """Load a prior run's output CSV (if any) so a restarted run can skip
    (feature, scale) identities that already have a recorded result instead
    of silently overwriting them with a second, possibly-contradictory row."""
    if not out.exists():
        return [], set()
    with out.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done: set[tuple[str, int]] = set()
    for r in rows:
        try:
            done.add((r["feature"], int(r["scale"])))
        except (KeyError, ValueError, TypeError):
            continue
    return rows, done


def _run_one(root: Path, feature: str, scale: int, timeout: int, warmups: int, repeats: int, provenance: dict):
    cmd = [sys.executable, "-m", "benchmarks.step5.worker", "--feature", feature, "--scale", str(scale), "--warmups", str(warmups), "--repeats", str(repeats)]
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        cp = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        s = featm.REGISTRY.get(feature)
        rec = {
            "feature": feature, "modality": s.modality, "family": s.family, "subfamily": s.subfamily,
            "cost": s.cost.value, "complexity": s.complexity.value, "memory_profile": s.memory_profile.value,
            "batch_semantics": s.batch_semantics.value, "stateful": bool(s.stateful), "gpu_available": bool(s.gpu_available),
            "lane": "kernel_fit_transform" if s.stateful else "kernel", "scale": scale, "status": "timeout",
            "error_type": "TimeoutExpired", "error_message": f"timeout after {timeout}s",
        }
        return {**rec, **provenance}
    lines = [x.strip() for x in cp.stdout.splitlines() if x.strip().startswith("{")]
    if not lines:
        s = featm.REGISTRY.get(feature)
        rec = {"feature":feature,"modality":s.modality,"family":s.family,"cost":s.cost.value,"scale":scale,
                "status":"worker_crash","error_type":f"exit_{cp.returncode}","error_message":(cp.stderr or cp.stdout)[-500:]}
        return {**rec, **provenance}
    try:
        rec = json.loads(lines[-1])
    except Exception as exc:
        rec = {"feature":feature,"scale":scale,"status":"worker_crash","error_type":type(exc).__name__,"error_message":lines[-1][-500:]}
    return {**rec, **provenance}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_results/step5_system_feature_results.csv")
    # Audit finding (v0.5.0 -> v0.5.1): this used to default to a DIFFERENT
    # filename than --out ("step5_system_run_manifest.json" vs. e.g.
    # "step5_cpu_system_feature_results.csv"), so every invocation that
    # didn't pass --manifest explicitly (every single-feature resume rerun
    # in the v0.5.0 pass) silently overwrote that stray file with its own
    # tiny run's config, while the real, differently-named manifest sitting
    # next to --out went stale and was never updated to reflect later runs.
    # Default is now None and derived FROM --out below, so the manifest
    # always sits next to, and is named consistently with, the data it
    # actually describes.
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--feature", action="append", help="exact feature; repeatable")
    ap.add_argument("--modality", action="append")
    ap.add_argument("--family", action="append", help="family or modality.family")
    ap.add_argument("--max-features", type=int)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scale", type=int, action="append", help="override adaptive cost ladder")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing --out file and start over instead of resuming")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    specs = list(featm.REGISTRY.all())
    if args.feature:
        wanted=set(args.feature); specs=[s for s in specs if s.name in wanted]
    if args.modality:
        wanted=set(args.modality); specs=[s for s in specs if s.modality in wanted]
    if args.family:
        wanted=set(args.family); specs=[s for s in specs if s.family in wanted or f"{s.modality}.{s.family}" in wanted]
    specs=sorted(specs,key=lambda s:s.name)
    if args.max_features is not None: specs=specs[:args.max_features]

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    rows, done = ([], set()) if args.fresh else _load_existing(out)
    if rows:
        print(f"resume: {len(rows)} prior row(s) loaded from {out}, {len(done)} (feature,scale) identities already recorded and will be skipped")

    # Derive the manifest path from --out (e.g. "..._feature_results.csv" ->
    # "..._run_manifest.json") rather than a hardcoded default filename, so
    # a caller who only sets --out (as every prior invocation in this
    # project's history did) still gets a manifest that sits next to,
    # and stays in sync with, that same output file.
    manifest_path = args.manifest or str(out).replace("_feature_results.csv", "_run_manifest.json")
    if manifest_path == str(out):
        manifest_path = str(out.with_name(out.stem + "_run_manifest.json"))

    provenance = make_provenance(root, featm.__version__)
    if rows and not args.fresh:
        warn_if_mixed_checksums(rows, provenance["featm_source_checksum"])

    manifest=Path(manifest_path); manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({
        **provenance,
        "features_selected":len(specs), "warmups":args.warmups,"repeats":args.repeats,
        "scale_override":args.scale, "scales_by_cost":{k.value:v for k,v in SCALES_BY_COST.items()},
        "timeouts_by_cost_seconds":{k.value:v for k,v in TIMEOUT_BY_COST.items()}, "started_at_unix":time.time(),
        "python": sys.version, "platform": platform.platform(),
        "resumed_prior_rows":len(rows),
    },indent=2,sort_keys=True)+"\n",encoding="utf-8")

    for i,s in enumerate(specs,1):
        scales=args.scale or SCALES_BY_COST[s.cost]
        for scale in scales:
            if (s.name, scale) in done:
                print(f"[{i}/{len(specs)}] {s.name} scale={scale} -> skip (already recorded)")
                continue
            rec=_run_one(root,s.name,scale,TIMEOUT_BY_COST[s.cost],args.warmups,args.repeats,provenance)
            rows.append(rec)
            done.add((s.name, scale))
            print(f"[{i}/{len(specs)}] {s.name} scale={scale} -> {rec.get('status')}")
            # Persist after every case so interruption never loses prior evidence.
            fields=list(FIELD_ORDER)
            extras=sorted({k for r in rows for k in r if k not in fields})
            # Write to a temp file and atomically replace -- a truncate-in-place
            # rewrite on every single case leaves a window where a concurrent
            # reader (or a kill at the wrong instant) sees an empty/partial
            # file; os.replace() is atomic on both POSIX and Windows.
            tmp=out.with_suffix(out.suffix + ".tmp")
            with tmp.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=fields+extras,extrasaction="ignore");w.writeheader();w.writerows(rows)
            os.replace(tmp,out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
