from __future__ import annotations

"""Generates step5_environment_manifest.json (Section 13).

One manifest per local CPU run -- every CSV row is traceable back to this
environment via featm_version + platform + run timestamp. The real Kaggle
GPU environment is recorded separately inside
step5_gpu_validation_results.json's own "environment" block (Section 7's
own artifact), since it is a genuinely different machine; this manifest
does not overwrite or merge with that -- see docs/BENCHMARKS.md.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import featm
from benchmarks.step5.provenance import make_provenance


def _optional_versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "dask", "distributed", "pyarrow",
                "networkx", "torch", "transformers", "timm", "cupy", "psutil"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = None
    return out


def _cpu_model() -> str:
    try:
        if platform.system() == "Windows":
            r = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10)
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0]
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_results/step5_environment_manifest.json")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    provenance = make_provenance(root, featm.__version__)
    manifest = {
        **provenance,
        "python": sys.version,
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_cores": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "gpu_models": [],  # this manifest describes the LOCAL CPU-only run; see step5_gpu_validation_results.json for the real Kaggle GPU environment
        "optional_dependency_versions": _optional_versions(),
        "random_seed": args.seed,
        "generated_at_unix": time.time(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
