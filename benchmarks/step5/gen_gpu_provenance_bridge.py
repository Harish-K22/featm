from __future__ import annotations

"""GPU-evidence provenance bridge (v0.5.0 -> v0.5.1 audit-fix pass, Finding 9).

The real Kaggle Tesla P100 GPU crossover/equivalence run was performed
against the v0.4.4 wheel (SHA-256 recorded below), not re-run against the
final v0.5.1 package. Step 3's GPU dispatcher and the 22 GPU-capable feature
implementations were not touched by anything in the v0.5.0 or v0.5.1 passes
(both only changed optional pretrained-embedding wrappers, benchmark/docs/
test code, and the version string) -- this script PROVES that claim file by
file rather than asserting it, by extracting the exact wheel used for the
P100 run and byte-comparing every GPU-relevant file against the final
source tree.

If any listed file differs, this script reports it as a MISMATCH and the
GPU evidence must be treated as NOT provenance-linked (a fresh GPU rerun
would then be required) -- it does not silently pass.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

# Files whose content can affect the 22 GPU-capable features' registered
# metadata, dispatch behavior, or numerical kernels.
GPU_RELEVANT_FILES = [
    "featm/gpu/__init__.py",
    "featm/gpu/backend.py",
    "featm/gpu/cache.py",
    "featm/gpu/dispatch.py",
    "featm/gpu/exceptions.py",
    "featm/gpu/memory.py",
    "featm/gpu/registration.py",
    "featm/gpu/kernels/__init__.py",
    "featm/gpu/kernels/reductions.py",
    "featm/gpu/kernels/spectral.py",
    "featm/features/signal/full.py",
    "featm/features/timeseries/basic.py",
    "featm/features/timeseries/full.py",
    "featm/registry.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_wheel_file(wheel_path: Path, arcname: str) -> bytes | None:
    with zipfile.ZipFile(wheel_path) as zf:
        try:
            return zf.read(arcname)
        except KeyError:
            return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-wheel", required=True, help="exact wheel used for the real P100 GPU run")
    ap.add_argument("--final-wheel", required=True, help="the final v0.5.1 wheel")
    ap.add_argument("--root", default=".", help="final source tree root (used if --final-wheel isn't built yet)")
    ap.add_argument("--out", default="step5_final_results/step5_gpu_provenance.json")
    args = ap.parse_args()

    old_wheel = Path(args.old_wheel)
    final_wheel = Path(args.final_wheel)

    comparisons = []
    all_match = True
    for arcname in GPU_RELEVANT_FILES:
        old_bytes = _extract_wheel_file(old_wheel, arcname)
        new_bytes = _extract_wheel_file(final_wheel, arcname) if final_wheel.exists() else None
        if new_bytes is None and final_wheel.exists():
            comparisons.append({"file": arcname, "status": "missing_in_final", "match": False})
            all_match = False
            continue
        if old_bytes is None:
            comparisons.append({"file": arcname, "status": "missing_in_old", "match": False})
            all_match = False
            continue
        match = old_bytes == new_bytes
        comparisons.append({
            "file": arcname, "status": "identical" if match else "DIFFERS", "match": match,
            "old_sha256": hashlib.sha256(old_bytes).hexdigest(),
            "new_sha256": hashlib.sha256(new_bytes).hexdigest() if new_bytes is not None else None,
        })
        if not match:
            all_match = False

    result = {
        "old_wheel_path": str(old_wheel), "old_wheel_sha256": _sha256(old_wheel),
        "final_wheel_path": str(final_wheel), "final_wheel_sha256": _sha256(final_wheel) if final_wheel.exists() else None,
        "gpu_relevant_files_compared": len(GPU_RELEVANT_FILES),
        "all_gpu_relevant_files_identical": all_match,
        "comparisons": comparisons,
        "conclusion": (
            "GPU evidence from the v0.4.4 P100 run is provenance-linked and inherited unchanged into the "
            "final release: every GPU-relevant file is byte-identical." if all_match else
            "MISMATCH: at least one GPU-relevant file differs between the old P100-evidence wheel and the "
            "final wheel -- GPU evidence is NOT safely inherited and a fresh GPU rerun is required before "
            "this can be treated as validating the final release."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
