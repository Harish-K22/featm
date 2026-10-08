"""Generates notebooks/FEATM_STEP5_GPU_CROSSOVER.ipynb.

Step-5 Layer C (Section 7): real-CUDA CPU<->GPU crossover evidence for all
22 GPU-capable features. Follows the same generated-notebook pattern as
scripts/build_gpu_validation_notebook.py (Step 3) -- diffable .py source,
not a hand-edited .ipynb -- but is scoped to Section 7's specific asks
rather than re-deriving all of Step 3's acceptance gates:

  - all 22 GPU-capable features, workloads around/above the shipped
    gpu_min_elements threshold
  - explicit backend="cpu" and backend="gpu" (never "auto") for raw speed
  - CPU<->GPU numerical equivalence via each feature's declared tolerance
  - median warm end-to-end latency (transfer included) AND kernel-only
    latency, reported separately and clearly labeled
  - speedup = cpu_seconds / gpu_seconds
  - backend="auto" chosen-backend audit against the shipped threshold
  - a measured crossover interval per feature (bracketing size where
    speedup crosses 1.0x), derived from real repeated measurements, not
    assumed from the shipped threshold
  - every OOM/injection claim, if any is added later, must be labeled as
    controlled/injected, never physical -- this script does not inject any
    OOM (Section 7 does not require it; Step 3's Section G already covers
    OOM/retry mechanics and is not being re-litigated here)

Run this script (not the notebook) whenever the notebook needs to change.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


# ==========================================================================
md(
    "# FEATM Step 5 -- Layer C: Real CUDA CPU<->GPU Crossover (Section 7)\n"
    "\n"
    "Benchmarks all 22 GPU-capable features on real CUDA hardware (Kaggle T4/P100/L4). "
    "Local development only proves architecture/dispatch correctness -- **this notebook is the "
    "only place real GPU crossover numbers come from.** No number in the Step-5 GPU evidence "
    "artifacts is estimated, extrapolated, or reused from Step 3's separate validation run.\n"
    "\n"
    "**Requires a Kaggle GPU session** (Settings -> Accelerator -> GPU). Every one of the 22 "
    "GPU-capable features in this v0.4.4 registry is a closed-form reduction/spectral computation "
    "(no pretrained-model download involved), so no internet-fetched model weights are needed -- "
    "only the featm wheel itself (attached as a Kaggle Dataset input) and cupy-cuda12x.\n"
    "\n"
    "Outputs saved at the end: `step5_gpu_crossover_results.csv`, `step5_gpu_validation_results.json` "
    "-- download both and feed them into `benchmarks/step5/summarize.py`-adjacent aggregation."
)

# ==========================================================================
md("## Section A -- Environment")
code(
    "import sys, platform, subprocess, json, time\n"
    "\n"
    "print('Python:', sys.version)\n"
    "print('Platform:', platform.platform())\n"
    "try:\n"
    "    smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)\n"
    "    print(smi.stdout if smi.returncode == 0 else f'nvidia-smi failed: {smi.stderr}')\n"
    "except FileNotFoundError:\n"
    "    print('nvidia-smi not found -- no NVIDIA driver visible. This notebook requires a GPU session.')\n"
    "env_info = {'python': sys.version, 'platform': platform.platform(), 'started_at_unix': time.time()}\n"
)

# ==========================================================================
md("## Section B -- Install FEATM from the local wheel (NOT PyPI) and import")
code(
    "import glob, subprocess, sys, os\n"
    "\n"
    "search_roots = ['/kaggle/input', '/kaggle/working', '.']\n"
    "wheel = next((p for root in search_roots for p in glob.glob(f'{root}/**/featm-*-py3-none-any.whl', recursive=True)), None)\n"
    "sdist = next((p for root in search_roots for p in glob.glob(f'{root}/**/featm-*.tar.gz', recursive=True)), None)\n"
    "src_zip = next((p for root in search_roots for p in glob.glob(f'{root}/**/featm-*-src.zip', recursive=True)), None)\n"
    "print('wheel:', wheel); print('sdist:', sdist); print('src_zip:', src_zip)\n"
    "assert wheel or sdist or src_zip, (\n"
    "    'No featm wheel/sdist/src ZIP found under /kaggle/input, /kaggle/working, or . -- '\n"
    "    'upload one as a Kaggle Dataset and attach it to this notebook (Add Input) before running.'\n"
    ")\n"
)
code(
    "install_target = wheel or sdist\n"
    "if install_target is None and src_zip is not None:\n"
    "    import zipfile\n"
    "    extract_dir = '/kaggle/working/featm_src'\n"
    "    os.makedirs(extract_dir, exist_ok=True)\n"
    "    with zipfile.ZipFile(src_zip) as zf:\n"
    "        zf.extractall(extract_dir)\n"
    "    install_target = extract_dir\n"
    "print('Installing:', install_target)\n"
    "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', install_target], capture_output=True, text=True)\n"
    "print(r.stdout[-3000:]); print(r.stderr[-2000:])\n"
    "assert r.returncode == 0, 'featm install failed -- see output above'\n"
)
code(
    "try:\n"
    "    import cupy  # noqa: F401\n"
    "    print('CuPy already available.')\n"
    "except ImportError:\n"
    "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], capture_output=True, text=True)\n"
    "    print(r.stdout[-3000:]); print(r.stderr[-2000:])\n"
    "    import cupy  # noqa: F401\n"
    "    print('Installed CuPy', cupy.__version__)\n"
)
code(
    "import featm\n"
    "print('featm version:', featm.__version__)\n"
    "assert len(featm.REGISTRY.all()) == 552, f'expected 552 registered features, got {len(featm.REGISTRY.all())}'\n"
    "\n"
    "from featm.gpu import backend as gpu_backend\n"
    "gpu_backend.reset_probe_cache()\n"
    "assert gpu_backend.any_gpu_available(), 'No usable GPU backend detected -- check the Kaggle accelerator setting.'\n"
    "info = gpu_backend.device_info()\n"
    "print('Detected GPU backend:', info.backend, '| device:', info.name)\n"
    "print(f'VRAM total: {info.total_bytes/1e9:.2f} GB | free: {info.free_bytes/1e9:.2f} GB')\n"
    "env_info['gpu_name'] = info.name; env_info['gpu_backend'] = info.backend\n"
    "env_info['gpu_total_vram_bytes'] = info.total_bytes; env_info['cuda_runtime_version'] = info.cuda_version\n"
    "env_info['cupy_version'] = cupy.__version__\n"
)

# ==========================================================================
md("## Section C -- GPU capability inventory (must match the frozen 22-feature Step-5 baseline)")
code(
    "from featm.gpu.registration import gpu_capability_inventory, validate_gpu_registry\n"
    "\n"
    "validate_gpu_registry()\n"
    "inventory = gpu_capability_inventory()\n"
    "print(json.dumps(inventory, indent=2))\n"
    "assert inventory['gpu_capable_features'] == 22, f\"expected 22 GPU-capable features, got {inventory['gpu_capable_features']}\"\n"
    "gpu_features = sorted(inventory['gpu_capable_feature_names'])\n"
)

# ==========================================================================
md(
    "## Section D -- CPU<->GPU numerical equivalence (all 22 features)\n"
    "\n"
    "Uses each feature's own declared `gpu_numerical_tolerance` via the real dispatch validator "
    "(`dispatch_one_feature(..., validate=True)`), across varied synthetic datasets at two sizes."
)
code(
    "import numpy as np\n"
    "from featm.registry import REGISTRY\n"
    "from featm.gpu.dispatch import dispatch_one_feature\n"
    "\n"
    "rng = np.random.default_rng(42)\n"
    "\n"
    "def _datasets(n, t):\n"
    "    out = {}\n"
    "    out['random'] = rng.standard_normal((n, t))\n"
    "    out['constant'] = np.full((n, t), 3.5)\n"
    "    out['all_zero'] = np.zeros((n, t))\n"
    "    out['negative'] = -np.abs(rng.standard_normal((n, t)))\n"
    "    with_nan = rng.standard_normal((n, t)); with_nan[:, 0] = np.nan\n"
    "    out['with_nan'] = with_nan\n"
    "    out['float32_input'] = rng.standard_normal((n, t)).astype(np.float32)\n"
    "    return out\n"
    "\n"
    "equivalence_results = []\n"
    "for name in gpu_features:\n"
    "    spec = REGISTRY.get(name)\n"
    "    for size_label, (n, t) in {'small': (5, 32), 'medium': (500, 128)}.items():\n"
    "        for data_label, X in _datasets(n, t).items():\n"
    "            try:\n"
    "                _, dinfo = dispatch_one_feature(spec, X, backend='gpu', gpu_fallback='strict', validate=True)\n"
    "                passed = dinfo.validation['passed']; detail = dinfo.validation\n"
    "            except Exception as exc:\n"
    "                passed = False; detail = {'error': f'{type(exc).__name__}: {exc}'}\n"
    "            equivalence_results.append({'feature': name, 'size': size_label, 'data': data_label,\n"
    "                                         'input_dtype': str(X.dtype), 'passed': passed, 'detail': detail})\n"
    "\n"
    "eq_failures = [r for r in equivalence_results if not r['passed']]\n"
    "print(f\"CPU<->GPU equivalence: {len(equivalence_results) - len(eq_failures)}/{len(equivalence_results)} passed\")\n"
    "for r in eq_failures[:20]:\n"
    "    print(' FAIL', r['feature'], r['size'], r['data'], '->', r['detail'])\n"
    "cpu_gpu_equivalence_failures = len(eq_failures)\n"
)

# ==========================================================================
md(
    "## Section E -- Crossover sweep: explicit CPU vs explicit GPU, transfer-included AND kernel-only\n"
    "\n"
    "For every GPU-capable feature: a size ladder anchored around and above its shipped "
    "`gpu_min_elements` threshold (0.1x, 0.5x, 1x, 2x, 8x the threshold, each rounded to a whole "
    "row count at width=64), explicit `backend=\"cpu\"`/`backend=\"gpu\"` (never `\"auto\"`), warm "
    "median-of-N timing. Kernel-only timing additionally isolates `dispatch_one_feature` (no public-"
    "engine batching/report overhead) with an explicit device-synchronize before stopping the GPU "
    "kernel-only timer, reported as a SEPARATE column -- never blended into the end-to-end number."
)
code(
    "def _median_seconds(fn, *, warmup, reps):\n"
    "    for _ in range(warmup): fn()\n"
    "    times = []\n"
    "    for _ in range(reps):\n"
    "        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)\n"
    "    times.sort(); return times[len(times) // 2]\n"
    "\n"
    "def _median_seconds_gpu_sync(fn, *, warmup, reps):\n"
    "    for _ in range(warmup): fn()\n"
    "    times = []\n"
    "    for _ in range(reps):\n"
    "        t0 = time.perf_counter(); fn(); cupy.cuda.Stream.null.synchronize()\n"
    "        times.append(time.perf_counter() - t0)\n"
    "    times.sort(); return times[len(times) // 2]\n"
    "\n"
    "from featm.gpu import memory as gpu_memory\n"
    "\n"
    "WIDTH = 64\n"
    "crossover_rows = []\n"
    "for name in gpu_features:\n"
    "    spec = REGISTRY.get(name)\n"
    "    # spec.gpu_min_elements is the raw per-feature override and is often None\n"
    "    # (falls back to a family/global default) -- estimate_gpu_min_elements()\n"
    "    # is the SAME resolver backend='auto' itself uses, so this is the real\n"
    "    # shipped threshold, not just whatever happens to be set on the spec.\n"
    "    threshold_elems = gpu_memory.estimate_gpu_min_elements(spec)\n"
    "    row_targets = sorted({max(4, int(threshold_elems * m / WIDTH)) for m in (0.1, 0.5, 1.0, 2.0, 8.0)})\n"
    "    for n in row_targets:\n"
    "        X = rng.standard_normal((n, WIDTH)).astype(np.float64)\n"
    "        elements = n * WIDTH\n"
    "        reps = 3 if n > 200_000 else 5\n"
    "        # End-to-end (public engine, transfer included) -- what a real caller experiences.\n"
    "        cpu_e2e = _median_seconds(lambda: featm.extract(X, features=[name], backend='cpu', errors='raise'), warmup=1, reps=reps)\n"
    "        gpu_e2e = _median_seconds_gpu_sync(lambda: featm.extract(X, features=[name], backend='gpu', gpu_fallback='strict', errors='raise'), warmup=2, reps=reps)\n"
    "        # Kernel-only (dispatch_one_feature directly, no batching/report machinery), separately labeled.\n"
    "        cpu_kernel = _median_seconds(lambda: dispatch_one_feature(spec, X, backend='cpu'), warmup=1, reps=reps)\n"
    "        gpu_kernel = _median_seconds_gpu_sync(lambda: dispatch_one_feature(spec, X, backend='gpu', gpu_fallback='strict'), warmup=2, reps=reps)\n"
    "        speedup_e2e = cpu_e2e / gpu_e2e if gpu_e2e > 0 else float('inf')\n"
    "        speedup_kernel = cpu_kernel / gpu_kernel if gpu_kernel > 0 else float('inf')\n"
    "        row = {\n"
    "            'feature': name, 'modality': spec.modality, 'family': spec.family, 'rows': n, 'width': WIDTH,\n"
    "            'elements': elements, 'gpu_min_elements_shipped': threshold_elems,\n"
    "            'cpu_e2e_median_seconds': cpu_e2e, 'gpu_e2e_median_seconds': gpu_e2e, 'speedup_e2e': speedup_e2e,\n"
    "            'cpu_kernel_median_seconds': cpu_kernel, 'gpu_kernel_median_seconds': gpu_kernel, 'speedup_kernel': speedup_kernel,\n"
    "            'transfer_included_in_e2e': True, 'gpu_model': env_info.get('gpu_name'), 'reps': reps,\n"
    "        }\n"
    "        crossover_rows.append(row)\n"
    "        print(f\"{name:40s} n={n:>9,} elem={elements:>11,} speedup_e2e={speedup_e2e:>6.3f}x speedup_kernel={speedup_kernel:>6.3f}x\")\n"
)

# ==========================================================================
md(
    "## Section F -- `backend=\"auto\"` chosen-backend audit against the shipped threshold\n"
    "\n"
    "For each feature and each swept size, records what `backend=\"auto\"` ACTUALLY chose and whether "
    "that agrees with the shipped `gpu_min_elements` threshold (elements >= threshold -> expect gpu)."
)
code(
    "auto_audit_rows = []\n"
    "for name in gpu_features:\n"
    "    spec = REGISTRY.get(name)\n"
    "    threshold_elems = gpu_memory.estimate_gpu_min_elements(spec)\n"
    "    for n in sorted({max(4, int(threshold_elems * m / WIDTH)) for m in (0.1, 0.5, 1.0, 2.0, 8.0)}):\n"
    "        X = rng.standard_normal((n, WIDTH)).astype(np.float64)\n"
    "        r = featm.extract(X, features=[name], backend='auto', errors='raise')\n"
    "        chosen = r.attrs['backend_report'][name]['actual_backend']\n"
    "        elements = n * WIDTH\n"
    "        expected = 'gpu' if elements >= threshold_elems else 'cpu'\n"
    "        auto_audit_rows.append({'feature': name, 'rows': n, 'elements': elements,\n"
    "                                 'gpu_min_elements_shipped': threshold_elems, 'auto_chosen_backend': chosen,\n"
    "                                 'threshold_expected_backend': expected, 'agrees_with_threshold': chosen == expected})\n"
    "auto_disagreements = [r for r in auto_audit_rows if not r['agrees_with_threshold']]\n"
    "print(f\"backend=auto vs shipped-threshold agreement: {len(auto_audit_rows) - len(auto_disagreements)}/{len(auto_audit_rows)}\")\n"
    "for r in auto_disagreements[:20]:\n"
    "    print(' DISAGREE', r)\n"
)

# ==========================================================================
md(
    "## Section G -- Measured crossover interval per feature\n"
    "\n"
    "From Section E's real measured `speedup_e2e` points (never assumed from the shipped "
    "threshold), finds the smallest measured element count where speedup_e2e >= 1.0 AND stays "
    ">= 1.0 at the next larger measured point (two consecutive non-noisy wins), bracketed by the "
    "last losing point below it. If GPU never wins within the measured range, or wins everywhere "
    "measured, that is reported honestly rather than extrapolated."
)
code(
    "from collections import defaultdict\n"
    "by_feature = defaultdict(list)\n"
    "for r in crossover_rows:\n"
    "    by_feature[r['feature']].append((r['elements'], r['speedup_e2e']))\n"
    "\n"
    "crossover_summary = []\n"
    "for name, pts in by_feature.items():\n"
    "    pts = sorted(pts)\n"
    "    all_win = all(s >= 1.0 for _, s in pts)\n"
    "    all_lose = all(s < 1.0 for _, s in pts)\n"
    "    result = {}\n"
    "    if all_win:\n"
    "        crossing_label = 'left_censored_gpu_wins_at_smallest_measured_point'\n"
    "        result = {'crossover_upper_bound_elements': pts[0][0]}\n"
    "    elif all_lose:\n"
    "        crossing_label = 'right_censored_gpu_never_wins_in_measured_range'\n"
    "        result = {'crossover_lower_bound_elements': pts[-1][0]}\n"
    "    else:\n"
    "        crossing_label = 'unstable_no_two_point_crossover'\n"
    "        for i in range(len(pts) - 1):\n"
    "            e_i, s_i = pts[i]; e_j, s_j = pts[i + 1]\n"
    "            stable = (i + 2 >= len(pts)) or (pts[i + 2][1] >= 1.0)\n"
    "            if s_i < 1.0 and s_j >= 1.0 and stable:\n"
    "                crossing_label = 'measured_bracket'\n"
    "                result = {'last_losing_elements': e_i, 'first_stable_winning_elements': e_j}\n"
    "                break\n"
    "    crossing = result.get('first_stable_winning_elements')\n"
    "    spec = REGISTRY.get(name)\n"
    "    crossover_summary.append({'feature': name, 'measured_crossover_elements': crossing, 'crossover_status': crossing_label,\n"
    "                               'gpu_min_elements_shipped': gpu_memory.estimate_gpu_min_elements(spec),\n"
    "                               'measured_points': len(pts), **result})\n"
    "    print(name, '->', crossing_label, crossing, result)\n"
)

# ==========================================================================
md("## Section H -- Save results")
code(
    "import csv\n"
    "\n"
    "with open('step5_gpu_crossover_results.csv', 'w', newline='') as f:\n"
    "    fieldnames = list(crossover_rows[0].keys())\n"
    "    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(crossover_rows)\n"
    "print(f'Wrote step5_gpu_crossover_results.csv ({len(crossover_rows)} rows)')\n"
    "\n"
    "with open('step5_gpu_auto_audit_results.csv', 'w', newline='') as f:\n"
    "    w = csv.DictWriter(f, fieldnames=list(auto_audit_rows[0].keys())); w.writeheader(); w.writerows(auto_audit_rows)\n"
    "print(f'Wrote step5_gpu_auto_audit_results.csv ({len(auto_audit_rows)} rows)')\n"
    "\n"
    "with open('step5_gpu_equivalence_results.csv', 'w', newline='') as f:\n"
    "    fieldnames = ['feature', 'size', 'data', 'input_dtype', 'passed', 'rtol', 'atol', 'max_abs_error', 'max_rel_error', 'error']\n"
    "    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()\n"
    "    for r in equivalence_results:\n"
    "        d = r['detail']\n"
    "        w.writerow({'feature': r['feature'], 'size': r['size'], 'data': r['data'], 'input_dtype': r['input_dtype'],\n"
    "                    'passed': r['passed'], 'rtol': d.get('rtol'), 'atol': d.get('atol'),\n"
    "                    'max_abs_error': d.get('max_abs_error'), 'max_rel_error': d.get('max_rel_error'), 'error': d.get('error')})\n"
    "print(f'Wrote step5_gpu_equivalence_results.csv ({len(equivalence_results)} rows)')\n"
    "\n"
    "env_info['finished_at_unix'] = time.time()\n"
    "validation_results = {\n"
    "    'featm_version': featm.__version__, 'environment': env_info,\n"
    "    'gpu_capable_features_expected': 22, 'gpu_capable_features_found': inventory['gpu_capable_features'],\n"
    "    'cpu_gpu_equivalence_checks': len(equivalence_results), 'cpu_gpu_equivalence_failures': cpu_gpu_equivalence_failures,\n"
    "    'auto_backend_checks': len(auto_audit_rows), 'auto_backend_disagreements': len(auto_disagreements),\n"
    "    'crossover_rows': len(crossover_rows), 'crossover_summary': crossover_summary,\n"
    "    'step5_gpu_layer_acceptance': cpu_gpu_equivalence_failures == 0 and len(auto_disagreements) == 0,\n"
    "}\n"
    "with open('step5_gpu_validation_results.json', 'w') as f:\n"
    "    json.dump(validation_results, f, indent=2, sort_keys=True, default=str)\n"
    "print(json.dumps(validation_results, indent=2, sort_keys=True, default=str))\n"
)


def build_notebook() -> dict:
    cells = []
    for kind, text in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "FEATM_STEP5_GPU_CROSSOVER.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
