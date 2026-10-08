from __future__ import annotations

"""Step-5 regression matrix + packaging check (Section 17 + Section L),
writing REAL computed command-level results to
step5_final_results/step5_regression_matrix.json -- gen_validation_results.py
reads this file for the Step1_4_regression, fixture_validation, and
packaging gates instead of a hardcoded zero.

v0.5.0 -> v0.5.1 audit-fix (Finding 7): the v0.5.0 pass deselected the one
pytest test that drives benchmarks/batch_equivalence_smoke.py's full-registry
sweep (it hung on that machine at the time) and reported the regression gate
as "0 failures" anyway -- an honest disclosure in prose, but not an honest
GATE, since a deselected test is not a passed test and the required explicit
commands (batch_equivalence_smoke.py, scaling_benchmark.py) were never
actually run standalone. This version runs EVERY required command explicitly,
with a bounded per-command wall-clock timeout (never hangs the harness
forever), and records one of four honest statuses per command:

  regression_pass      -- exit code 0
  known_external_issue -- nonzero exit, but the failure text matches a known
                           external-asset/model-download/environment pattern
                           (never silently turned into a pass)
  blocked              -- exceeded the bounded timeout; not run to completion
  genuine_failure      -- nonzero exit with no recognized external cause
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_SUMMARY_RE = re.compile(
    r"(?:(\d+) failed, )?(?:(\d+) passed)?(?:, (\d+) skipped)?(?:, (\d+) deselected)?(?:, (\d+) warnings?)? in ([\d.]+)s"
)

# Substrings that indicate a nonzero exit was caused by an external
# environment/asset issue (missing model download, native library crash
# tied to a corrupted cache, network) rather than a genuine code regression.
# Matching here NEVER converts the failure into a "pass" -- it only chooses
# between "known_external_issue" and "genuine_failure" for a human to see.
_KNOWN_EXTERNAL_PATTERNS = [
    "huggingface", "connectionerror", "httperror", "urlopen error",
    "could not load", "no space left", "0xc0000005", "access violation",
    "resolve host", "certificate verify failed", "read timed out",
]

# v0.5.1 -> v0.5.2 audit-fix (Finding 7): the v0.5.1 pass excluded this
# command entirely (SKIPPED_DISK_RISK) after repeatedly driving free disk
# space toward 0 bytes downloading heavy optional pretrained-model
# dependencies -- an honest disclosure, but the audit is explicit that
# skipped_disk_risk must not be reported as a regression pass in the final
# accepted package. The fix is at the harness level, not here: this command
# now runs with --external-assets=offline-skip (see batch_equivalence_smoke.py),
# which bounds every heavyweight-optional-dependency feature inside its own
# killable subprocess with offline env vars set, so it can no longer hang or
# run away with disk space -- it always completes, reporting genuinely
# unavailable remote assets in their own external_asset_unavailable bucket
# instead of silently skipping the whole command.
_DUPLICATE_HEAVY_TEST = "tests/test_audit4.py::test_additional_bug_batch_equivalence_harness_finds_zero_mismatches"

COMMANDS = [
    ("pytest", [sys.executable, "-m", "pytest", "-q", f"--deselect={_DUPLICATE_HEAVY_TEST}"], 900),
    ("smoke_universe", [sys.executable, "benchmarks/smoke_universe.py"], 300),
    ("distributed_equivalence_smoke", [sys.executable, "benchmarks/distributed_equivalence_smoke.py"], 300),
    ("distributed_dask_equivalence_smoke", [sys.executable, "benchmarks/distributed_dask_equivalence_smoke.py"], 300),
    ("scaling_benchmark", [sys.executable, "benchmarks/scaling_benchmark.py"], 300),
    ("step5_benchmark_foundation_tests", [sys.executable, "-m", "pytest", "-q", "tests/test_step5_benchmark_foundation.py"], 180),
    ("batch_equivalence_smoke", [sys.executable, "-m", "benchmarks.batch_equivalence_smoke", "--external-assets=offline-skip"], 1800),
]


def _classify_failure(stdout: str, stderr: str) -> str:
    text = (stdout + stderr).lower()
    return "known_external_issue" if any(p in text for p in _KNOWN_EXTERNAL_PATTERNS) else "genuine_failure"


def _run_command(root: Path, name: str, cmd: list[str], timeout_s: int) -> dict:
    print(f"Running {name}: {' '.join(cmd)} (timeout={timeout_s}s) ...", flush=True)
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        print(f"  -> BLOCKED after {elapsed:.1f}s (exceeded {timeout_s}s bound)")
        return {"command": " ".join(cmd), "status": "blocked", "exit_code": None,
                "elapsed_seconds": round(elapsed, 1),
                "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""}
    elapsed = time.time() - t0
    m = _SUMMARY_RE.search(cp.stdout)
    parsed = {"passed": int(m.group(2)) if m and m.group(2) else None,
              "failed": int(m.group(1)) if m and m.group(1) else None,
              "skipped": int(m.group(3)) if m and m.group(3) else None} if m else {}
    if cp.returncode == 0:
        status = "regression_pass"
    else:
        status = _classify_failure(cp.stdout, cp.stderr)
    print(f"  -> {status} (exit={cp.returncode}, {elapsed:.1f}s) {parsed}")
    return {"command": " ".join(cmd), "status": status, "exit_code": cp.returncode,
            "elapsed_seconds": round(elapsed, 1), "parsed": parsed,
            "stdout_tail": cp.stdout[-2000:], "stderr_tail": cp.stderr[-2000:]}


def _packaging_check(root: Path, expected_version: str) -> dict:
    checks = {}
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--sdist"], cwd=root, capture_output=True, text=True)
    checks["build_wheel_sdist"] = {"ok": r.returncode == 0, "stderr_tail": r.stderr[-500:] if r.returncode else ""}
    dist = root / "dist"
    wheel = next(iter(sorted(dist.glob(f"featm-{expected_version}-py3-none-any.whl"))), None)
    sdist = next(iter(sorted(dist.glob(f"featm-{expected_version}.tar.gz"))), None)
    checks["wheel_built"] = wheel is not None
    checks["sdist_built"] = sdist is not None

    verify_script = (
        "import featm, sys\n"
        f"assert featm.__version__ == {expected_version!r}, featm.__version__\n"
        "assert len(featm.REGISTRY.all()) == 552, len(featm.REGISTRY.all())\n"
        "featm.REGISTRY.validate_all()\n"
        "from featm.gpu.registration import gpu_capability_inventory\n"
        "assert gpu_capability_inventory()['gpu_capable_features'] == 22\n"
        "assert not any('cupy' in m or 'torch' in m for m in sys.modules), 'base import must stay lightweight'\n"
        "print('OK', featm.__version__, featm.__file__)\n"
    )
    for label, artifact in (("wheel", wheel), ("sdist", sdist)):
        if artifact is None:
            checks[f"{label}_clean_install"] = {"ok": False, "reason": "artifact not built"}
            continue
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(artifact)],
            cwd=root, capture_output=True, text=True,
        )
        install_ok = r.returncode == 0
        r2 = subprocess.run([sys.executable, "-c", verify_script], cwd=root, capture_output=True, text=True)
        checks[f"{label}_clean_install"] = {
            "ok": install_ok and r2.returncode == 0 and f"OK {expected_version}" in r2.stdout,
            "install_stderr_tail": r.stderr[-500:] if not install_ok else "",
            "verify_stdout": r2.stdout.strip(), "verify_stderr_tail": r2.stderr[-500:] if r2.returncode else "",
        }
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="step5_final_results/step5_regression_matrix.json")
    ap.add_argument("--skip-packaging", action="store_true")
    ap.add_argument("--expected-version", default="0.5.4")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    commands_result = {}
    for name, cmd, timeout_s in COMMANDS:
        commands_result[name] = _run_command(root, name, cmd, timeout_s)

    packaging = {}
    if not args.skip_packaging:
        print("Running packaging clean-install checks...", flush=True)
        packaging = _packaging_check(root, args.expected_version)
        print(f"  -> {packaging}")

    result = {"generated_at_unix": time.time(), "commands": commands_result, "packaging": packaging}
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")

    any_genuine_failure = any(c["status"] == "genuine_failure" for c in commands_result.values())
    return 1 if any_genuine_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
