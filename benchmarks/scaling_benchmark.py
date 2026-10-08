"""Reproducible Step 2 out-of-core scaling benchmark.

Generates a synthetic dataset as a chunked, streamed sequence of Parquet
row-groups -- never holding the whole dataset in one in-memory DataFrame --
then runs ``featm.extract()`` out-of-core under a few different budget/output
configurations, recording wall time, the batch size the planner chose,
declared budgets, retries/failures, output size, and whether output ended up
memory- or disk-backed.

This is a correctness/behavior benchmark, not a maximum-throughput benchmark:
it exists to make scaling claims reproducible and falsifiable rather than
asserted. Row count defaults to a moderate size and is a command-line flag so
it can be run at whatever size is safe on the local machine; do not raise it
to the point of exhausting local disk/RAM.

Usage:
    python benchmarks/scaling_benchmark.py --rows 1000000
    python benchmarks/scaling_benchmark.py --rows 1000000 --outdir ./bench_out
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from featm.resources import current_process_peak_rss_bytes, current_process_rss_bytes


def generate_parquet(path: Path, n_rows: int, *, chunk_rows: int = 100_000, seed: int = 0) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    writer = None
    written = 0
    try:
        while written < n_rows:
            take = min(chunk_rows, n_rows - written)
            df = pd.DataFrame(
                {
                    "price": rng.normal(100.0, 25.0, take),
                    "quantity": rng.integers(1, 200, take).astype(float),
                    "city": rng.choice(["NYC", "LA", "SF", "CHI", "HOU", "PHX"], take),
                    "score": rng.normal(0.0, 1.0, take),
                }
            )
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema)
            writer.write_table(table)
            written += take
    finally:
        if writer is not None:
            writer.close()


def run_case(path: Path, *, memory_budget, output, batch_rows, checkpoint=None, mode="exact") -> dict:
    rss_before = current_process_rss_bytes()
    peak_before = current_process_peak_rss_bytes()
    t0 = time.time()
    result = featm.extract(
        path,
        features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        memory_budget=memory_budget,
        output=output,
        batch_rows=batch_rows,
        checkpoint=checkpoint,
        mode=mode,
    )
    elapsed = time.time() - t0
    rss_after = current_process_rss_bytes()
    peak_after = current_process_peak_rss_bytes()
    return {
        "elapsed_seconds": round(elapsed, 3),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        # Peak process RSS since process start (Windows: native peak_wset, not a
        # before/after snapshot -- catches a spike even if it doesn't persist).
        # peak_before is the baseline already spent on interpreter/import/data
        # generation before this case runs; peak_after is the running total.
        "peak_rss_before_case_bytes": peak_before,
        "peak_rss_after_case_bytes": peak_after,
        "peak_rss_delta_for_this_case_bytes": max(peak_after - peak_before, 0),
        "output_shape": list(result.shape),
        "output_backend": result.report.output_backend,
        "batch_rows_used": result.report.batch_rows,
        "n_batches": result.report.n_batches,
        "memory_budget_bytes": result.report.memory_budget_bytes,
        "disk_budget_bytes": result.report.disk_budget_bytes,
        "retries": result.report.retries,
        "batch_reductions": result.report.batch_reductions,
        "failed_features": result.report.failed_features,
        "completed_features": result.report.completed_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="featm-bench-"))
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / "synthetic.parquet"

    print(f"Generating {args.rows:,} row synthetic Parquet dataset at {data_path} (streamed, chunked writes)...")
    t0 = time.time()
    generate_parquet(data_path, args.rows)
    gen_elapsed = time.time() - t0
    size_mb = data_path.stat().st_size / 1e6
    print(f"  done in {gen_elapsed:.1f}s, file size {size_mb:.1f} MB")

    results: dict[str, dict] = {}

    print("\n[Case 1] auto memory budget, auto output (typical default usage)")
    results["auto"] = run_case(data_path, memory_budget=None, output="auto", batch_rows=None)
    print(json.dumps(results["auto"], indent=2, default=str))

    print("\n[Case 2] tiny declared working-memory budget -> forces disk-backed output")
    # Deliberately smaller than the estimated output size (n_rows * n_output_columns * 8
    # bytes) so output='auto' spills to disk instead of keeping the small result in memory.
    results["tiny_budget_forced_disk"] = run_case(data_path, memory_budget="200KB", output="auto", batch_rows=None)
    print(json.dumps(results["tiny_budget_forced_disk"], indent=2, default=str))

    print("\n[Case 3] explicit disk output + checkpoint directory")
    ckpt = outdir / "checkpoint"
    results["checkpointed_disk"] = run_case(data_path, memory_budget=None, output="disk", batch_rows=200_000, checkpoint=ckpt)
    print(json.dumps(results["checkpointed_disk"], indent=2, default=str))

    summary = {
        "dataset_rows": args.rows,
        "dataset_file_size_mb": round(size_mb, 2),
        "dataset_generation_seconds": round(gen_elapsed, 3),
        "cases": results,
        "note": (
            "rss_*_bytes and peak_rss_*_bytes are total process RSS (interpreter + all "
            "imported libraries + everything else the process holds), not the same "
            "quantity as the declared memory_budget, which bounds only the executor's own "
            "working buffers. peak_rss_*_bytes is the OS-tracked peak working set since "
            "process start (Windows peak_wset), not a before/after snapshot, so it also "
            "catches a transient spike that a snapshot could miss. See docs/SCALING.md."
        ),
    }
    summary_path = outdir / "benchmark_results.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results written to {summary_path}")


if __name__ == "__main__":
    main()
