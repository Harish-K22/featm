from __future__ import annotations

"""Step-5 Layer D: persistent-cluster Dask crossover benchmark (Section 8).

Reuses the real ``distributed.LocalCluster(processes=True)`` +
``featm.extract(execution="distributed", distributed_backend="dask", ...)``
path the Step-4 foundation (``benchmarks/distributed_dask_equivalence_smoke.py``)
already proved correct -- this script is purely about TIMING, not semantic
equivalence, which is why it is a separate script rather than an extension of
that one.

Deliberately separates, per Section 8's explicit requirement:
  - cluster_startup_seconds: LocalCluster construction + Client connect, timed
    ONCE per (worker_count) cluster, never blended into any per-call number.
  - cold_first_call_seconds: the FIRST distributed extract() call on a given
    (cluster, feature, source-kind) combination -- includes one-time Dask
    graph/worker-import warmup that a genuinely steady-state number must not
    include.
  - steady_state_median_seconds: median of REPEATS calls after the cold call,
    on the SAME already-warm cluster/workers.
  - local_steady_state_median_seconds: the same feature/data run locally
    (execution="local"), for the "is Dask faster than local here" question.

Benchmarks representative row-independent features (tabular, timeseries) and
all 7 stateful features (all Source-representable via DATASET_CONTEXT_ROW_ALIGNED,
confirmed against the registry), across both the coordinator-streamed path (a
plain in-memory DataFrame) and the Parquet-native path (a real .parquet file,
so ``DaskParquetPartitionProvider`` is genuinely exercised, not simulated),
at worker counts 1/2/4.

No mandatory speedup target: "Dask is slower than local below N rows" is a
valid, useful, HONEST conclusion this script is expected to be able to reach
and report as such, not something to tune the workload size to avoid.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from featm.registry import REGISTRY

SEED = 2026
rng = np.random.default_rng(SEED)

REPEATS = 3  # steady-state repeats, AFTER the separately-timed cold first call


def _tabular_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "num": rng.normal(size=n),
        "cat": rng.choice(["a", "b", "c", "d"], size=n),
    })


def _text_df(n: int) -> pd.DataFrame:
    words = ["feature", "graph", "signal", "matrix", "vector", "kernel", "tensor", "sample"]
    docs = [" ".join(rng.choice(words, size=6)) for _ in range(n)]
    return pd.DataFrame({"text": docs})


def _timeseries_df(n: int, width: int = 32) -> pd.DataFrame:
    arr = rng.normal(size=(n, width))
    return pd.DataFrame(arr, columns=[f"t{i}" for i in range(width)])


# (feature, dataframe_builder, columns, params) -- representative
# row-independent features plus all 7 stateful features. All 7 stateful
# features are DATASET_CONTEXT_ROW_ALIGNED (confirmed against the live
# registry, not assumed), which is Source-representable.
CASES = [
    ("tabular.numeric.abs", _tabular_df, {"tabular.numeric.abs": "num"}, None),
    ("timeseries.statistical.mean", _timeseries_df, {}, None),
    ("tabular.scaling.zscore", _tabular_df, {"tabular.scaling.zscore": "num"}, None),
    ("tabular.scaling.minmax", _tabular_df, {"tabular.scaling.minmax": "num"}, None),
    ("tabular.encoding.category_frequency", _tabular_df, {"tabular.encoding.category_frequency": "cat"}, None),
    ("text.bm25.bm25", _text_df, {}, {"text.bm25.bm25": {"query": "feature graph"}}),
    ("text.tfidf.tfidf", _text_df, {}, None),
    ("text.information.surprisal", _text_df, {}, None),
    ("text.semantics.similarity", _text_df, {}, None),
]

FIELDS = [
    "feature", "modality", "family", "worker_count", "partition_provider", "partition_rows",
    "n_rows", "cluster_startup_seconds", "cold_first_call_seconds", "steady_state_median_seconds",
    "steady_state_min_seconds", "rows_per_sec", "local_steady_state_median_seconds", "speedup_dask_over_local",
    "actual_worker_count", "status", "error_type", "error_message",
]


def _median(times: list[float]) -> float:
    t = sorted(times)
    return t[len(t) // 2]


def _run_case(scheduler_address: str, worker_count: int, feature: str, df: pd.DataFrame,
              columns: dict, params: dict | None, partition_rows: int, source, source_label: str) -> dict:
    spec = REGISTRY.get(feature)
    base = {"feature": feature, "modality": spec.modality, "family": spec.family, "worker_count": worker_count,
            "partition_provider": source_label, "partition_rows": partition_rows, "n_rows": len(df), "status": "error"}
    kwargs = dict(features=[feature], columns=columns or None, params=params, output="memory", errors="raise")
    try:
        def dask_call():
            return featm.extract(
                source, execution="distributed", distributed_backend="dask", distributed_workers=worker_count,
                distributed_partition_rows=partition_rows, distributed_scheduler_address=scheduler_address,
                distributed_processes=True, **kwargs,
            )

        t0 = time.perf_counter()
        first = dask_call()
        cold_first_call_seconds = time.perf_counter() - t0
        actual_worker_count = first.report.distributed_report.get("actual_worker_count")
        partition_provider = first.report.distributed_report.get("partition_provider")

        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            dask_call()
            times.append(time.perf_counter() - t0)
        steady_median = _median(times)

        local_times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            featm.extract(df, batch_rows=partition_rows, **kwargs)
            local_times.append(time.perf_counter() - t0)
        local_median = _median(local_times)

        base.update(
            status="ok",
            cluster_startup_seconds=None,  # filled in by caller (measured once per cluster, not per case)
            cold_first_call_seconds=cold_first_call_seconds,
            steady_state_median_seconds=steady_median,
            steady_state_min_seconds=min(times),
            rows_per_sec=len(df) / steady_median if steady_median > 0 else float("inf"),
            local_steady_state_median_seconds=local_median,
            speedup_dask_over_local=(local_median / steady_median) if steady_median > 0 else float("inf"),
            actual_worker_count=actual_worker_count,
            partition_provider=partition_provider,
            error_type="", error_message="",
        )
    except (ImportError, ModuleNotFoundError) as exc:
        base.update(status="optional_dependency_unavailable", error_type=type(exc).__name__, error_message=str(exc)[:500])
    except Exception as exc:
        base.update(status="error", error_type=type(exc).__name__, error_message=str(exc)[:500])
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="step5_final_results/step5_dask_crossover_results.csv")
    # Audit finding (v0.5.0 -> v0.5.1): a single workload size cannot support a
    # crossover/overhead conclusion at all -- it can only report "faster" or
    # "slower" AT that one size. A size ladder is required so a genuine
    # crossover (or its absence) can be observed across the measured range,
    # the same way Layer A's CPU scale ladder and Layer C's GPU crossover
    # sweep both already work.
    ap.add_argument("--n-rows", type=int, action="append", default=None,
                     help="workload size ladder -- large enough for Dask overhead to be visible; repeatable")
    ap.add_argument("--partition-rows", type=int, default=5_000)
    ap.add_argument("--workers", type=int, action="append", default=None)
    args = ap.parse_args()

    from distributed import Client, LocalCluster

    worker_counts = args.workers or [1, 2, 4]
    n_rows_ladder = args.n_rows or [40_000, 160_000, 640_000]
    partition_rows = args.partition_rows

    tmp_dir = Path("step5_final_results/_dask_crossover_parquet")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for worker_count in worker_counts:
        print(f"=== worker_count={worker_count}: starting persistent LocalCluster ===", flush=True)
        t0 = time.perf_counter()
        cluster = LocalCluster(n_workers=worker_count, threads_per_worker=1, processes=True, dashboard_address=None)
        client = Client(cluster)
        cluster_startup_seconds = time.perf_counter() - t0
        print(f"cluster startup: {cluster_startup_seconds:.3f}s | {client}", flush=True)
        try:
            # The SAME persistent cluster is reused across every size in the
            # ladder for this worker_count -- never recreated per measurement.
            for n in n_rows_ladder:
                for feature, builder, columns, params in CASES:
                    df = builder(n)
                    parquet_path = tmp_dir / f"{feature.replace('.', '_')}_{n}.parquet"
                    # A single physical row group larger than partition_rows makes
                    # DaskParquetPartitionProvider refuse (correctly) and fall back
                    # to coordinator_streamed (ParquetRowGroupTooLargeError) -- write
                    # with row groups at/under partition_rows so the parquet-native
                    # path is genuinely exercised, not silently downgraded.
                    df.to_parquet(parquet_path, row_group_size=min(partition_rows, max(1, n // 4)))

                    for source, label in ((df, "coordinator_streamed"), (str(parquet_path), "parquet_native")):
                        print(f"[{worker_count}w, n={n}] {feature} via {label} ...", flush=True)
                        rec = _run_case(cluster.scheduler_address, worker_count, feature, df, columns, params,
                                         partition_rows, source, label)
                        rec["cluster_startup_seconds"] = cluster_startup_seconds
                        rows.append(rec)
                        print(f"    -> {rec['status']} steady_state={rec.get('steady_state_median_seconds')} "
                              f"speedup={rec.get('speedup_dask_over_local')} provider={rec.get('partition_provider')}", flush=True)
                        tmp = out.with_suffix(out.suffix + ".tmp")
                        with tmp.open("w", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                            w.writeheader(); w.writerows(rows)
                        os.replace(tmp, out)
        finally:
            client.close()
            cluster.close()

    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
