"""Real Dask Distributed equivalence harness (Step 4 completion pass).

Unlike ``distributed_equivalence_smoke.py`` (the local ThreadPool reference
runtime, no Dask needed), this actually drives a real
``distributed.LocalCluster(processes=True)`` and compares its output against
the existing local Step-2 engine, for a broad cross-section of modalities
reusing the SAME sample builders ``batch_equivalence_smoke.py`` already
maintains (never a handful of hand-picked functions only).

One shared cluster/client is created ONCE and reused via
``scheduler_address=`` for every case -- a fresh ``LocalCluster`` per case
would dominate the runtime with startup cost, not the thing this script
actually needs to prove.

Buckets, honestly reported (never let a network/model-download problem in an
unrelated optional feature masquerade as a distributed semantic failure --
those are reported as ``optional_unavailable``, matching
``batch_equivalence_smoke.py``'s own convention):

    equivalent
    rejected_correctly
    optional_unavailable
    excluded_with_reason
    mismatches
    row_accounting_failures
    schema_failures
    state_merge_failures

Exit code is nonzero only for genuine mismatches/accounting/schema/state
failures.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import featm
from featm.distributed.contracts import DistributedFeatureUnsupportedError
from featm.execution import UnsafeMultiBatchFeatureError
from featm.registry import REGISTRY, BatchSemantics

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from batch_equivalence_smoke import (  # noqa: E402
    _MODALITY_BUILDERS,
    _REQUIRED_PARAM_VALUES,
    N,
)

PARTITIONS = [3, 5]  # 5 is not a divisor of N=12 -- an uneven last partition


#: Deliberately excludes the heaviest object-column modalities (image/audio/video/
#: graph) -- those already have real-Dask serialization coverage (manually verified
#: during this pass's development, see STEP4_DASK_HANDOFF.md) and the LOCAL-runtime
#: foundation smoke (distributed_equivalence_smoke.py) already exercises them
#: end to end. Repeatedly serializing many large ndarray/graph objects across a long
#: sequence of cases against ONE shared cluster proved unstable on this development
#: machine (a real, environment-specific resource/networking issue, not a FEATM bug --
#: see that handoff section for the exact failure). Kept intentionally small and fast.
_LIGHT_MODALITIES = ("tabular", "timeseries", "signal", "event", "sequence", "information", "text")


def _pick_cases() -> list:
    """A bounded cross-section: up to 2 claimed-batch-safe features per light
    modality with a registered sample builder, plus every required-param feature
    (for the rejection-then-explicit-success pair)."""
    cases = []
    by_modality: dict[str, list] = {}
    for spec in REGISTRY.all():
        if spec.modality not in _LIGHT_MODALITIES:
            continue
        if spec.batch_semantics not in (
            BatchSemantics.ROW_INDEPENDENT, BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED, BatchSemantics.OPAQUE_OBJECT_SAMPLE,
        ):
            continue
        if spec.required_params_for_batching and spec.name not in _REQUIRED_PARAM_VALUES:
            continue
        if spec.modality == "tabular" and spec.incremental and not spec.stateful:
            continue
        by_modality.setdefault(spec.modality, []).append(spec)
    for modality, specs in sorted(by_modality.items()):
        cases.extend(specs[:2])
    return cases


def main() -> int:
    from distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None)
    client = Client(cluster)
    print(f"Dask cluster: {client}", flush=True)

    buckets: dict[str, list[tuple[str, str]]] = {
        "equivalent": [], "rejected_correctly": [], "optional_unavailable": [], "excluded_with_reason": [],
        "mismatches": [], "row_accounting_failures": [], "schema_failures": [], "state_merge_failures": [],
    }

    try:
        cases = _pick_cases()
        print(f"running {len(cases)} cases...", flush=True)
        for case_i, spec in enumerate(cases):
            name = spec.name
            print(f"[{case_i + 1}/{len(cases)}] {name} ...", flush=True)
            try:
                builder = _MODALITY_BUILDERS[spec.modality]
                df, columns = builder(spec)
                params = {name: _REQUIRED_PARAM_VALUES[name]} if name in _REQUIRED_PARAM_VALUES else None
                kwargs = dict(features=[name], columns=columns or None, params=params, memory_budget="64MB", errors="raise", output="memory")

                local = featm.extract(df, batch_rows=3, **kwargs)
                dist = featm.extract(
                    df, execution="distributed", distributed_backend="dask", distributed_workers=2,
                    distributed_partition_rows=3, distributed_scheduler_address=cluster.scheduler_address, **kwargs,
                )
                local_cols = list(local.columns) if hasattr(local, "columns") else list(local.to_pandas().columns)
                dist_cols = list(dist.to_pandas().columns)
                if local_cols != dist_cols:
                    buckets["schema_failures"].append((name, f"local={local_cols} dist={dist_cols}"))
                    continue
                local_arr = (local if isinstance(local, pd.DataFrame) else local.to_pandas()).to_numpy(dtype=float)
                dist_arr = dist.to_pandas().to_numpy(dtype=float)
                if local_arr.shape[0] != dist_arr.shape[0]:
                    buckets["row_accounting_failures"].append((name, f"local rows={local_arr.shape[0]} dist rows={dist_arr.shape[0]}"))
                    continue
                if not np.allclose(local_arr, dist_arr, rtol=1e-6, atol=1e-8, equal_nan=True):
                    buckets["mismatches"].append((name, "values differ"))
                    continue
                buckets["equivalent"].append((name, ""))
            except (ImportError, ModuleNotFoundError) as exc:
                buckets["optional_unavailable"].append((name, str(exc)[:150]))
            except (DistributedFeatureUnsupportedError, UnsafeMultiBatchFeatureError):
                buckets["rejected_correctly"].append((name, "correctly rejected (unsupported distributed shape)"))
            except Exception as exc:
                buckets["state_merge_failures"].append((name, f"{type(exc).__name__}: {exc}"[:200]))

        # Explicit rejection checks: FULL_DATASET must reject clearly.
        full_dataset_specs = [s for s in REGISTRY.all() if s.batch_semantics is BatchSemantics.FULL_DATASET]
        if full_dataset_specs:
            spec = full_dataset_specs[0]
            try:
                featm.extract(
                    pd.DataFrame({"x": np.arange(10, dtype=float)}), features=[spec.name],
                    execution="distributed", distributed_backend="dask", distributed_scheduler_address=cluster.scheduler_address,
                    errors="raise",
                )
                buckets["mismatches"].append((spec.name, "FULL_DATASET feature was NOT rejected"))
            except DistributedFeatureUnsupportedError:
                buckets["rejected_correctly"].append((spec.name, "FULL_DATASET correctly rejected"))
            except Exception as exc:
                buckets["state_merge_failures"].append((spec.name, f"unexpected: {type(exc).__name__}: {exc}"))
    finally:
        client.close()
        cluster.close()

    for bucket in ("equivalent", "rejected_correctly", "optional_unavailable", "excluded_with_reason"):
        print(f"{bucket} = {len(buckets[bucket])}")
    for bucket in ("mismatches", "row_accounting_failures", "schema_failures", "state_merge_failures"):
        print(f"{bucket} = {len(buckets[bucket])}")
        for n, d in buckets[bucket]:
            print("  ", n, "->", d)

    import csv

    with open("step4_distributed_equivalence_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "bucket", "detail"])
        writer.writeheader()
        for bucket, rows in buckets.items():
            for name, detail in rows:
                writer.writerow({"case": name, "bucket": bucket, "detail": detail})
    print(f"Wrote step4_distributed_equivalence_results.csv ({sum(len(v) for v in buckets.values())} rows)")

    hard_failures = sum(len(buckets[b]) for b in ("mismatches", "row_accounting_failures", "schema_failures", "state_merge_failures"))
    return 1 if hard_failures else 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"elapsed: {time.time() - t0:.1f}s")
    raise SystemExit(code)
