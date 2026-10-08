"""Finding 12 (independent audit, v0.4.0 -> v0.4.1): a persistent-cluster distributed
benchmark that actually measures steady-state parallel throughput.

The v0.4.0 notebook's Section G benchmark called ``featm.extract(..., execution=
"distributed", distributed_backend="dask", ...)`` with no ``distributed_scheduler_address``
on every single timed repetition -- so ``distributed_extract`` constructed a BRAND NEW
``LocalCluster`` from scratch on every call. The reported numbers (Dask taking 1.5-2.4
SECONDS vs local's 4-6 MILLISECONDS) were honestly disclosed as a methodology artifact
(cluster-startup overhead dominating), never claimed as a real performance characteristic
-- but a benchmark that only measures its own startup cost is not informative about
whether Step 4's distributed execution provides genuine parallelism at all.

This script creates ONE persistent ``LocalCluster``/``Client`` PER worker count, reused
across every repetition and every case, and reports cold-start cost SEPARATELY from
steady-state per-call time -- so "is Dask ever actually faster, and at what point" can be
answered honestly, instead of "how long does starting a cluster take."

Three axes, matching the audit's explicit requirement (§12):
  - a CPU-heavy ROW_INDEPENDENT feature (timeseries.spectral.centroid)
  - a stateful two-pass feature (tabular.scaling.zscore)
  - Parquet-native file-backed input (tabular.numeric.abs over a real .parquet file)

Usage::

    python benchmarks/distributed_persistent_cluster_benchmark.py [--rows N] [--reps N]

No speedup target is asserted -- workload-dependent speedup is expected, not required.
The only thing this script insists on is that timings reflect steady-state per-call cost,
not cluster construction.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import featm


def _median_seconds(fn, *, warmup: int, reps: int) -> tuple[float, float]:
    """Returns (cold_first_call_seconds, median_of_remaining_reps_seconds)."""
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    for _ in range(max(0, warmup - 1)):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return cold, statistics.median(times) if times else float("nan")


def _cpu_heavy_case(rows: int) -> tuple[pd.DataFrame, str, dict, str]:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(np.vstack([rng.standard_normal(64) for _ in range(rows)]))
    return df, "timeseries.spectral.centroid", {}, "int-index"


def _stateful_case(rows: int) -> tuple[pd.DataFrame, str, dict, str]:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.standard_normal(rows)})
    return df, "tabular.scaling.zscore", {"tabular.scaling.zscore": "x"}, "x"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=[20_000, 100_000])
    ap.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    try:
        from distributed import Client, LocalCluster
    except ImportError:
        print("SKIP: distributed is not installed (pip install 'featm[distributed]')")
        return 0

    rows_list: list[int] = args.rows
    tmpdir = Path(tempfile.mkdtemp(prefix="featm-persist-bench-"))
    results: list[dict] = []

    for n in rows_list:
        cpu_df, cpu_feature, cpu_columns, cpu_col_name = _cpu_heavy_case(n)
        stateful_df, stateful_feature, stateful_columns, _ = _stateful_case(n)
        parquet_path = tmpdir / f"data_{n}.parquet"
        pd.DataFrame({"x": np.random.default_rng(3).standard_normal(n)}).to_parquet(parquet_path)

        cases = [
            ("cpu_heavy_row_independent", cpu_df, cpu_feature, cpu_columns if cpu_col_name != "int-index" else {cpu_feature: 0}),
            ("stateful_two_pass", stateful_df, stateful_feature, stateful_columns),
            ("parquet_native_io", str(parquet_path), "tabular.numeric.abs", {"tabular.numeric.abs": "x"}),
        ]

        for case_name, data, feature, columns in cases:
            local_fn = lambda d=data, f=feature, c=columns: featm.extract(
                d, features=[f], columns=c, execution="local", errors="raise",
            )
            local_cold, local_steady = _median_seconds(local_fn, warmup=args.warmup, reps=args.reps)
            print(f"[{case_name}] rows={n:>7,}  local  steady={local_steady*1000:>9.2f}ms")

            for workers in args.workers:
                partition_rows = max(500, n // (workers * 4))
                cluster = LocalCluster(n_workers=workers, threads_per_worker=1, processes=True, dashboard_address=None)
                client = Client(cluster)
                try:
                    dist_fn = lambda d=data, f=feature, c=columns, w=workers, pr=partition_rows: featm.extract(
                        d, features=[f], columns=c, execution="distributed", distributed_backend="dask",
                        distributed_workers=w, distributed_partition_rows=pr, output="memory", errors="raise",
                        distributed_scheduler_address=cluster.scheduler_address,
                    )
                    dist_cold, dist_steady = _median_seconds(dist_fn, warmup=args.warmup, reps=args.reps)
                finally:
                    client.close()
                    cluster.close()

                speedup = local_steady / dist_steady if dist_steady > 0 else float("nan")
                row = {
                    "case": case_name, "rows": n, "workers": workers,
                    "local_steady_seconds": round(local_steady, 6),
                    "dask_cold_first_call_seconds": round(dist_cold, 6),
                    "dask_steady_seconds": round(dist_steady, 6),
                    "steady_state_speedup": round(speedup, 4) if speedup == speedup else None,
                }
                results.append(row)
                print(
                    f"[{case_name}] rows={n:>7,} workers={workers}  "
                    f"dask_cold={dist_cold*1000:>9.2f}ms  dask_steady={dist_steady*1000:>9.2f}ms  "
                    f"speedup(steady)={speedup:>6.2f}x"
                )

    out_path = Path("step4_distributed_persistent_cluster_benchmark_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path} ({len(results)} rows)")
    print(
        "\nNote: 'dask_cold_first_call_seconds' is the first call against an ALREADY-CONSTRUCTED "
        "persistent cluster (still pays task-graph submission + worker warm-up, but NOT cluster "
        "startup, which happens once outside all timing here). 'dask_steady_seconds' is the median "
        "of subsequent calls against the SAME cluster -- this is the number that answers 'is Dask "
        "genuinely faster for this workload,' not cluster construction time."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
