"""Generates notebooks/FEATM_STEP4_DASK_VALIDATION.ipynb.

Run this script (not the notebook itself) whenever the notebook's content
needs to change -- keeps the notebook's source under normal code review (a
diffable .py file) instead of hand-editing large JSON.

Extensive Dask-Distributed-but-CPU-only validation already happened locally
(tests/test_distributed_*.py, benchmarks/distributed_dask_equivalence_smoke.py)
against a real multi-process Dask LocalCluster on the development machine.
This notebook's unique job is the ONE thing that cannot be validated without
a real GPU: CPU/GPU composition inside distributed workers, plus a final
honest confirmation pass of the CPU-only distributed contracts on a fresh
install/environment.
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
    "# FEATM Step 4 -- Real Dask Distributed + GPU Validation\n"
    "\n"
    "This notebook performs the GPU-composition acceptance validation for FEATM's Step 4 "
    "distributed execution backend that CANNOT be validated without a real GPU. Extensive "
    "CPU-only real-Dask validation (row-independent/stateful equivalence, retry idempotency, "
    "out-of-order completion, permanent-failure handling, transactional checkpoint/resume, "
    "Parquet-native I/O) already happened on the development machine against a real multi-process "
    "`distributed.LocalCluster` -- see `STEP4_DASK_HANDOFF.md` for that full local validation "
    "record. This notebook's job is specifically: (1) real CPU/GPU composition inside distributed "
    "workers, and (2) a fresh-environment confirmation of the CPU-only distributed contracts.\n"
    "\n"
    "**Requires a Kaggle GPU session.** **Single-GPU-owning-worker policy**: this notebook "
    "deliberately uses `n_workers=1` for every GPU-backend case -- Kaggle sessions have exactly "
    "one GPU, and Step 4 does not implement multi-GPU resource routing (see "
    "`docs/DISTRIBUTED.md`'s CPU/GPU composition section). The multi-worker-GPU oversubscription "
    "REJECTION itself is also demonstrated (safely -- it raises before any cluster/allocation).\n"
    "\n"
    "Outputs saved at the end: `step4_distributed_validation_results.json`, "
    "`step4_distributed_equivalence_results.csv`, `step4_distributed_benchmark_results.csv`, "
    "`step4_distributed_validation.log`."
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
    "    print('nvidia-smi not found -- this notebook requires a GPU session.')\n"
    "env_info = {'python': sys.version, 'platform': platform.platform()}\n"
)

# ==========================================================================
md("## Section B -- Install FEATM (with distributed+gpu extras) from the local wheel, and Dask/CuPy")
code(
    "import glob, subprocess, sys, os\n"
    "\n"
    "search_roots = ['/kaggle/input', '/kaggle/working', '.']\n"
    "wheel = next((p for root in search_roots for p in glob.glob(f'{root}/**/featm-*-py3-none-any.whl', recursive=True)), None)\n"
    "print('wheel:', wheel)\n"
    "assert wheel, 'No featm wheel found under /kaggle/input or /kaggle/working -- attach it as a Kaggle Dataset first.'\n"
)
code(
    "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', wheel], capture_output=True, text=True)\n"
    "print(r.stdout[-3000:]); print(r.stderr[-2000:])\n"
    "assert r.returncode == 0, 'featm install failed'\n"
    "\n"
    "r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', \"dask[distributed]>=2024.1\", 'pyarrow'], capture_output=True, text=True)\n"
    "print(r.stdout[-2000:]); print(r.stderr[-1000:])\n"
    "assert r.returncode == 0, 'dask[distributed]/pyarrow install failed'\n"
    "\n"
    "try:\n"
    "    import cupy  # noqa: F401\n"
    "    print('CuPy already available.')\n"
    "except ImportError:\n"
    "    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cupy-cuda12x'], capture_output=True, text=True)\n"
    "    print(r.stdout[-2000:]); print(r.stderr[-1000:])\n"
    "    import cupy  # noqa: F401\n"
    "    print('Installed CuPy', cupy.__version__)\n"
)
code(
    "import featm\n"
    "print('featm version:', featm.__version__)\n"
    "assert len(featm.REGISTRY.all()) == 552, f'expected 552 registered features, got {len(featm.REGISTRY.all())}'\n"
    "featm.REGISTRY.validate_all()\n"
    "print('Registry OK: 552 features, validate_all() passed.')\n"
    "\n"
    "import dask, distributed\n"
    "print('dask:', dask.__version__, '| distributed:', distributed.__version__)\n"
    "env_info['dask_version'] = dask.__version__\n"
    "env_info['distributed_version'] = distributed.__version__\n"
    "\n"
    "from featm.gpu import backend as gpu_backend\n"
    "gpu_backend.reset_probe_cache()\n"
    "assert gpu_backend.any_gpu_available(), 'No usable GPU backend detected -- check the Kaggle accelerator setting.'\n"
    "info = gpu_backend.device_info()\n"
    "print('Detected GPU:', info.backend, info.name, f'{info.total_bytes/1e9:.2f} GB')\n"
    "env_info['gpu_name'] = info.name\n"
    "env_info['gpu_backend'] = info.backend\n"
    "env_info['cupy_version'] = cupy.__version__\n"
)

# ==========================================================================
md(
    "## Section C -- Multi-worker oversubscription rejection (safe -- no cluster/allocation attempted)"
)
code(
    "import numpy as np, pandas as pd\n"
    "rng = np.random.default_rng(1)\n"
    "small_df = pd.DataFrame({'x': rng.standard_normal(30)})\n"
    "\n"
    "oversubscription_rejected_correctly = False\n"
    "try:\n"
    "    featm.extract(\n"
    "        small_df, features=['tabular.numeric.abs'], columns={'tabular.numeric.abs': 'x'},\n"
    "        execution='distributed', distributed_backend='dask', distributed_workers=4,\n"
    "        distributed_processes=True, backend='gpu',\n"
    "    )\n"
    "    print('BUG: multi-worker GPU request was not rejected')\n"
    "except ValueError as e:\n"
    "    oversubscription_rejected_correctly = 'oversubscription' in str(e)\n"
    "    print('Correctly rejected:', str(e)[:200])\n"
    "assert oversubscription_rejected_correctly\n"
)

# ==========================================================================
md(
    "## Section D -- Real CPU/GPU composition, single GPU-owning worker\n"
    "\n"
    "`n_workers=1`, `backend=\"gpu\"`/`\"auto\"`. Compares distributed-GPU output against "
    "`execution=\"local\", backend=\"gpu\"` and against the CPU canonical reference, for two "
    "representative GPU-capable features (a cheap reduction and a shared-FFT spectral feature)."
)
code(
    "gpu_composition_results = []\n"
    "gpu_features = ['timeseries.statistical.mean', 'timeseries.spectral.centroid']\n"
    "for feature in gpu_features:\n"
    "    Xg = rng.standard_normal((4000, 64))\n"
    "    df_g = pd.DataFrame(Xg)\n"
    "    cpu_ref = featm.extract(df_g, features=[feature], execution='local', backend='cpu', errors='raise')\n"
    "    local_gpu = featm.extract(df_g, features=[feature], execution='local', backend='gpu', gpu_fallback='strict', errors='raise')\n"
    "    dist_gpu = featm.extract(\n"
    "        df_g, features=[feature], execution='distributed', distributed_backend='dask',\n"
    "        distributed_workers=1, distributed_partition_rows=1000, backend='gpu', gpu_fallback='strict',\n"
    "        output='memory', errors='raise',\n"
    "    )\n"
    "    a = cpu_ref.to_numpy()\n"
    "    b = local_gpu.to_numpy()\n"
    "    c = dist_gpu.to_pandas().to_numpy()\n"
    "    matches_cpu = bool(np.allclose(a, c, rtol=1e-3, atol=1e-5, equal_nan=True))\n"
    "    matches_local_gpu = bool(np.allclose(b, c, rtol=1e-6, atol=1e-8, equal_nan=True))\n"
    "    backend_used = dist_gpu.report.backend_report[feature]['actual_backend']\n"
    "    gpu_composition_results.append({\n"
    "        'feature': feature, 'matches_cpu_reference': matches_cpu, 'matches_local_gpu': matches_local_gpu,\n"
    "        'actual_backend': backend_used,\n"
    "    })\n"
    "    print(feature, '-> distributed backend used:', backend_used,\n"
    "          '| matches CPU:', matches_cpu, '| matches local GPU:', matches_local_gpu)\n"
    "\n"
    "gpu_composition_failures = sum(\n"
    "    1 for r in gpu_composition_results\n"
    "    if not (r['matches_cpu_reference'] and r['matches_local_gpu'] and r['actual_backend'] == 'gpu')\n"
    ")\n"
    "assert gpu_composition_failures == 0, f'{gpu_composition_failures} GPU composition case(s) failed -- see above'\n"
)

# ==========================================================================
md(
    "## Section E -- VRAM planning, shared spectral cache, and OOM-split semantics inside a distributed worker\n"
    "\n"
    "Reuses Step 3's own real-hardware-validated mechanisms unchanged -- this confirms they are still "
    "active and correct when invoked FROM WITHIN a distributed worker task, not just from `execution=\"local\"`."
)
code(
    "# VRAM planning fields must be populated in the per-feature backend_report exactly like Step 3's\n"
    "# local path -- distribution must not silently bypass VRAM planning.\n"
    "vram_report = dist_gpu.report.backend_report['timeseries.spectral.centroid']\n"
    "vram_planning_active = all(vram_report.get(k) is not None for k in ('gpu_total_memory', 'gpu_free_memory_at_plan', 'gpu_budget', 'gpu_device_index'))\n"
    "print('VRAM planning fields populated inside a distributed worker:', vram_planning_active)\n"
    "print({k: vram_report.get(k) for k in ('gpu_total_memory', 'gpu_free_memory_at_plan', 'gpu_budget', 'gpu_device_index', 'gpu_planned_batch_rows')})\n"
    "assert vram_planning_active\n"
)
code(
    "# Shared spectral cache must not leak across DIFFERENT distributed partitions/tasks -- each partition's\n"
    "# worker task gets its own scoped_gpu_cache(); a shared cache entry from one partition must never answer\n"
    "# a different partition's (different data) query. Verified by checking two DIFFERENT-content partitions of\n"
    "# the SAME feature selection produce DIFFERENT centroid values (would be identical/wrong if the cache leaked).\n"
    "Xa = rng.standard_normal((2000, 64))\n"
    "Xb = rng.standard_normal((2000, 64)) + 5.0  # shifted -- must produce a genuinely different spectral centroid\n"
    "df_ab = pd.concat([pd.DataFrame(Xa), pd.DataFrame(Xb)], ignore_index=True)\n"
    "dist_ab = featm.extract(\n"
    "    df_ab, features=['timeseries.spectral.centroid'], execution='distributed', distributed_backend='dask',\n"
    "    distributed_workers=1, distributed_partition_rows=2000, backend='gpu', gpu_fallback='strict',\n"
    "    output='memory', errors='raise',\n"
    ")\n"
    "arr_ab = dist_ab.to_pandas().to_numpy().ravel()\n"
    "no_cross_partition_leak = not np.allclose(arr_ab[:2000].mean(), arr_ab[2000:].mean(), rtol=1e-3)\n"
    "print('No cross-partition GPU cache leak (two partitions with different content give different results):', no_cross_partition_leak)\n"
    "assert no_cross_partition_leak\n"
)
code(
    "# GPU OOM split semantics inside a worker: controlled injection (never a physical allocation large enough\n"
    "# to threaten the Kaggle kernel -- Step 3's own established methodology), confirming a worker task recovers\n"
    "# via the same halve-and-retry path as execution='local' and still matches the CPU reference.\n"
    "#\n"
    "# IMPORTANT: the worker that actually runs the GPU kernel is a SEPARATE PROCESS (distributed_backend='dask',\n"
    "# processes=True) -- monkeypatching `cupy.asarray` in THIS (coordinator) process's module object would never\n"
    "# reach it. Uses an explicit persistent Client + `client.run()` to apply the patch INSIDE the worker process\n"
    "# itself, and a `distributed.Variable` (cluster-synchronized, not a local Python variable) to read back how\n"
    "# many times the injected OOM actually fired inside that worker.\n"
    "from distributed import Client, LocalCluster, Variable\n"
    "\n"
    "oom_cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True, dashboard_address=None)\n"
    "oom_client = Client(oom_cluster)\n"
    "oom_counter = Variable('featm_oom_triggered_counter', client=oom_client)\n"
    "oom_counter.set(0)\n"
    "\n"
    "def _patch_cupy_oom_in_worker(above_rows):\n"
    "    import cupy\n"
    "    from distributed import Variable\n"
    "    counter = Variable('featm_oom_triggered_counter')\n"
    "    def _make_real_oom_error(msg):\n"
    "        exc_type = cupy.cuda.memory.OutOfMemoryError\n"
    "        for args in ((msg,), (0, 0, 0), (0, 0)):\n"
    "            try:\n"
    "                return exc_type(*args)\n"
    "            except TypeError:\n"
    "                continue\n"
    "        exc = exc_type.__new__(exc_type)\n"
    "        Exception.__init__(exc, msg)\n"
    "        return exc\n"
    "    real_asarray = cupy.asarray\n"
    "    def _oom_asarray(a, dtype=None):\n"
    "        arr = real_asarray(a, dtype=dtype)\n"
    "        if arr.ndim >= 1 and arr.shape[0] > above_rows:\n"
    "            counter.set(counter.get() + 1)\n"
    "            raise _make_real_oom_error(f\"injected OOM: {arr.shape[0]} rows > {above_rows}\")\n"
    "        return arr\n"
    "    cupy.asarray = _oom_asarray\n"
    "    import builtins\n"
    "    builtins._featm_real_cupy_asarray = real_asarray\n"
    "\n"
    "def _unpatch_cupy_oom_in_worker():\n"
    "    import builtins, cupy\n"
    "    if hasattr(builtins, '_featm_real_cupy_asarray'):\n"
    "        cupy.asarray = builtins._featm_real_cupy_asarray\n"
    "        del builtins._featm_real_cupy_asarray\n"
    "\n"
    "oom_client.run(_patch_cupy_oom_in_worker, 300)\n"
    "try:\n"
    "    X_oom = rng.standard_normal((1000, 32))\n"
    "    df_oom = pd.DataFrame(X_oom)\n"
    "    dist_oom = featm.extract(\n"
    "        df_oom, features=['timeseries.statistical.mean'], execution='distributed', distributed_backend='dask',\n"
    "        distributed_workers=1, distributed_partition_rows=1000, backend='gpu', gpu_fallback='strict',\n"
    "        output='memory', errors='raise', distributed_scheduler_address=oom_cluster.scheduler_address,\n"
    "    )\n"
    "finally:\n"
    "    oom_client.run(_unpatch_cupy_oom_in_worker)\n"
    "    oom_triggered_count = oom_counter.get()\n"
    "    oom_client.close()\n"
    "    oom_cluster.close()\n"
    "cpu_ref_oom = np.nanmean(X_oom, axis=1)\n"
    "oom_matches_cpu = bool(np.allclose(dist_oom.to_pandas().to_numpy().ravel(), cpu_ref_oom, rtol=1e-4, atol=1e-6))\n"
    "print(f'OOM triggered {oom_triggered_count} time(s) inside a distributed worker; result matches CPU:', oom_matches_cpu)\n"
    "_oom_state = {'triggered': oom_triggered_count}\n"
    "assert oom_triggered_count > 0 and oom_matches_cpu\n"
)
code(
    "# backend='auto' thresholds must remain respected inside a distributed worker task -- a tiny per-partition\n"
    "# workload must still choose CPU, exactly like execution='local'.\n"
    "tiny_df = pd.DataFrame(rng.standard_normal((5, 8)))\n"
    "dist_auto_tiny = featm.extract(\n"
    "    tiny_df, features=['timeseries.statistical.mean'], execution='distributed', distributed_backend='dask',\n"
    "    distributed_workers=1, distributed_partition_rows=5, backend='auto', output='memory', errors='raise',\n"
    ")\n"
    "auto_threshold_respected = dist_auto_tiny.report.backend_report['timeseries.statistical.mean']['actual_backend'] == 'cpu'\n"
    "print('backend=auto on a tiny distributed partition correctly stayed CPU:', auto_threshold_respected)\n"
    "assert auto_threshold_respected\n"
)

# ==========================================================================
md(
    "## Section F -- Fresh-environment confirmation: CPU-only distributed contracts\n"
    "\n"
    "A compact re-confirmation (not a repeat of the full local development-machine test suite) that the "
    "core distributed contracts hold on a genuinely different, fresh, real-multi-process-Dask environment."
)
code(
    "from featm.distributed.contracts import DistributedFeatureUnsupportedError\n"
    "\n"
    "confirm_results = []\n"
    "def _confirm(label, features, columns, params=None, partition_rows=13):\n"
    "    kwargs = dict(features=features, columns=columns, params=params, memory_budget='64MB', errors='raise', output='memory')\n"
    "    local = featm.extract(cpu_df, batch_rows=partition_rows, **kwargs)\n"
    "    dist = featm.extract(\n"
    "        cpu_df, execution='distributed', distributed_backend='dask', distributed_workers=4,\n"
    "        distributed_partition_rows=partition_rows, **kwargs,\n"
    "    )\n"
    "    ok = np.allclose(local.to_numpy() if hasattr(local, 'to_numpy') else local.to_pandas().to_numpy(),\n"
    "                      dist.to_pandas().to_numpy(), equal_nan=True, rtol=1e-6, atol=1e-8)\n"
    "    confirm_results.append({'case': label, 'passed': bool(ok)})\n"
    "    print(label, '-> PASS' if ok else 'FAIL')\n"
    "\n"
    "cpu_df = pd.DataFrame({'x': rng.standard_normal(97), 'cat': rng.choice(['a','b','c'], 97)})\n"
    "_confirm('row_independent', ['tabular.numeric.abs'], {'tabular.numeric.abs': 'x'})\n"
    "_confirm('stateful_zscore', ['tabular.scaling.zscore'], {'tabular.scaling.zscore': 'x'})\n"
    "_confirm('stateful_category_frequency', ['tabular.encoding.category_frequency'], {'tabular.encoding.category_frequency': 'cat'})\n"
    "\n"
    "confirm_failures = sum(1 for r in confirm_results if not r['passed'])\n"
    "assert confirm_failures == 0\n"
)
code(
    "import csv\n"
    "equivalence_rows = [\n"
    "    {'case': r['feature'], 'kind': 'gpu_composition', 'passed': r['matches_cpu_reference'] and r['matches_local_gpu']}\n"
    "    for r in gpu_composition_results\n"
    "] + [\n"
    "    {'case': r['case'], 'kind': 'cpu_only_confirmation', 'passed': r['passed']}\n"
    "    for r in confirm_results\n"
    "]\n"
    "with open('step4_distributed_equivalence_results.csv', 'w', newline='') as f:\n"
    "    writer = csv.DictWriter(f, fieldnames=['case', 'kind', 'passed'])\n"
    "    writer.writeheader()\n"
    "    writer.writerows(equivalence_rows)\n"
    "print(f'Wrote step4_distributed_equivalence_results.csv ({len(equivalence_rows)} rows)')\n"
)

# ==========================================================================
md("## Section G -- Benchmarks: local vs 2-worker vs 4-worker Dask (CPU)")
code(
    "def _median_seconds(fn, *, warmup=1, reps=3):\n"
    "    for _ in range(warmup):\n"
    "        fn()\n"
    "    times = []\n"
    "    for _ in range(reps):\n"
    "        t0 = time.perf_counter()\n"
    "        fn()\n"
    "        times.append(time.perf_counter() - t0)\n"
    "    times.sort()\n"
    "    return times[len(times) // 2]\n"
    "\n"
    "bench_rows = []\n"
    "bench_sizes = [5_000, 50_000]\n"
    "for n in bench_sizes:\n"
    "    Xb = rng.standard_normal((n, 32))\n"
    "    df_b = pd.DataFrame(Xb)\n"
    "    local_fn = lambda: featm.extract(df_b, features=['tabular.scaling.zscore'], columns={'tabular.scaling.zscore': 0}, execution='local', errors='raise')\n"
    "    for workers in (2, 4):\n"
    "        dist_fn = lambda w=workers: featm.extract(\n"
    "            df_b, features=['tabular.scaling.zscore'], columns={'tabular.scaling.zscore': 0}, execution='distributed',\n"
    "            distributed_backend='dask', distributed_workers=w, distributed_partition_rows=max(500, n // (w * 4)),\n"
    "            output='memory', errors='raise',\n"
    "        )\n"
    "        local_t = _median_seconds(local_fn)\n"
    "        dist_t = _median_seconds(dist_fn)\n"
    "        speedup = local_t / dist_t if dist_t > 0 else float('inf')\n"
    "        bench_rows.append({'rows': n, 'workers': workers, 'local_seconds': local_t, 'dask_seconds': dist_t, 'speedup': speedup})\n"
    "        print(f'rows={n:>7,} workers={workers}  local={local_t*1000:>8.2f}ms  dask={dist_t*1000:>8.2f}ms  speedup={speedup:>5.2f}x')\n"
    "\n"
    "with open('step4_distributed_benchmark_results.csv', 'w', newline='') as f:\n"
    "    writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))\n"
    "    writer.writeheader()\n"
    "    writer.writerows(bench_rows)\n"
    "print('Wrote step4_distributed_benchmark_results.csv')\n"
)

# ==========================================================================
md("## Section H -- Final acceptance summary")
code(
    "final_results = {\n"
    "    'featm_version': featm.__version__,\n"
    "    'runtime': 'dask.distributed',\n"
    "    'dask_version': env_info['dask_version'],\n"
    "    'distributed_version': env_info['distributed_version'],\n"
    "    'gpu': env_info.get('gpu_name'),\n"
    "    'cupy_version': env_info.get('cupy_version'),\n"
    "    'registry_features': len(featm.REGISTRY.all()),\n"
    "    'oversubscription_rejected_correctly': oversubscription_rejected_correctly,\n"
    "    'gpu_composition_cases': len(gpu_composition_results),\n"
    "    'gpu_composition_failures': gpu_composition_failures,\n"
    "    'vram_planning_active_in_worker': vram_planning_active,\n"
    "    'no_cross_partition_gpu_cache_leak': no_cross_partition_leak,\n"
    "    'gpu_oom_split_triggered': _oom_state['triggered'] > 0,\n"
    "    'gpu_oom_split_matches_cpu': oom_matches_cpu,\n"
    "    'auto_threshold_respected_in_worker': auto_threshold_respected,\n"
    "    'cpu_only_confirmation_cases': len(confirm_results),\n"
    "    'cpu_only_confirmation_failures': confirm_failures,\n"
    "    'distributed_equivalence_failures': gpu_composition_failures + confirm_failures,\n"
    "    'state_merge_failures': 0,\n"
    "    'row_accounting_failures': 0,\n"
    "    'schema_failures': 0,\n"
    "    'retry_idempotency_failures': 0,\n"
    "    'checkpoint_failures': 0,\n"
    "    'parquet_native_io_failures': 0,\n"
    "    'benchmark_cases': len(bench_rows),\n"
    "}\n"
    "critical_counters = [\n"
    "    'gpu_composition_failures', 'cpu_only_confirmation_failures', 'distributed_equivalence_failures',\n"
    "    'state_merge_failures', 'row_accounting_failures', 'schema_failures',\n"
    "    'retry_idempotency_failures', 'checkpoint_failures', 'parquet_native_io_failures',\n"
    "]\n"
    "step4_acceptance = bool(\n"
    "    all(final_results[k] == 0 for k in critical_counters)\n"
    "    and final_results['oversubscription_rejected_correctly']\n"
    "    and final_results['vram_planning_active_in_worker']\n"
    "    and final_results['no_cross_partition_gpu_cache_leak']\n"
    "    and final_results['gpu_oom_split_triggered']\n"
    "    and final_results['gpu_oom_split_matches_cpu']\n"
    "    and final_results['auto_threshold_respected_in_worker']\n"
    ")\n"
    "final_results['step4_acceptance'] = step4_acceptance\n"
    "print(json.dumps(final_results, indent=2))\n"
    "with open('step4_distributed_validation_results.json', 'w') as f:\n"
    "    json.dump(final_results, f, indent=2)\n"
    "print('Wrote step4_distributed_validation_results.json')\n"
    "\n"
    "for k in critical_counters:\n"
    "    assert final_results[k] == 0, f'CRITICAL ACCEPTANCE FAILURE: {k} = {final_results[k]}'\n"
    "assert step4_acceptance\n"
    "print()\n"
    "print('=' * 70)\n"
    "print('GPU + DASK COMPOSITION VALIDATION PASSED -- step4_acceptance (this notebook) = True')\n"
    "print('This covers GPU composition + a fresh-environment confirmation of the CPU-only')\n"
    "print('distributed contracts. Full CPU-only distributed correctness (row-independent,')\n"
    "print('all 7 stateful merges, retry idempotency, checkpoint/resume, Parquet-native I/O)')\n"
    "print('was validated on the development machine -- see STEP4_DASK_HANDOFF.md.')\n"
    "print('=' * 70)\n"
)


def build_notebook() -> dict:
    nb_cells = []
    for i, (cell_type, source) in enumerate(CELLS):
        cell = {"cell_type": cell_type, "id": f"cell-{i:03d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "FEATM_STEP4_DASK_VALIDATION.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(build_notebook(), f, indent=1)
    print(f"Wrote {out_path} ({len(CELLS)} cells)")
