"""Step 4 completion pass: REAL Dask Distributed runtime tests.

Skipped entirely if Dask is not installed. These specifically exercise
behavior the local ThreadPool reference runtime CANNOT: real process
serialization, real scheduler completion ordering, real task retry, and a
real permanent-failure exception shape -- exactly the class of bug this
completion pass found that ``tests/test_distributed_foundation.py`` (local
runtime only) could not.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("distributed")

import featm
from featm.distributed.contracts import DistributedFeatureError, DistributedTaskError, Partition
from featm.distributed.checkpoint import DistributedCheckpointManager
from featm.distributed.dask_runtime import DaskDistributedRuntime
from featm.distributed.execution import _ordered_results, _ReorderGate


@pytest.fixture(scope="module")
def dask_runtime():
    rt = DaskDistributedRuntime(n_workers=4, threads_per_worker=1, processes=True)
    yield rt
    rt.close()


def test_row_independent_matches_local_on_real_dask_uneven_partitions(dask_runtime):
    rng = np.random.default_rng(20)
    df = pd.DataFrame({"x": rng.standard_normal(237)})
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    dask_r = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="dask", distributed_workers=4, distributed_partition_rows=17,
        output="memory", errors="raise",
    )
    a = local_r.to_numpy()
    b = dask_r.to_pandas().to_numpy()
    assert np.allclose(a, b)


def test_all_seven_stateful_encoders_match_local_on_real_dask(dask_runtime):
    rng = np.random.default_rng(21)
    num_df = pd.DataFrame({"x": rng.standard_normal(200), "cat": rng.choice(["a", "b", "c"], 200)})
    docs = [f"the quick brown fox number {i} extra padding words here" for i in range(200)]
    docs[0] = "UNIQUE_FIRST_DOC_MARKER special"
    text_df = pd.DataFrame({"doc": docs})

    cases = [
        ("tabular.scaling.zscore", num_df, "x", None),
        ("tabular.scaling.minmax", num_df, "x", None),
        ("tabular.encoding.category_frequency", num_df, "cat", None),
        ("text.information.surprisal", text_df, "doc", None),
        ("text.bm25.bm25", text_df, "doc", {"text.bm25.bm25": {"query": "fox"}}),
        ("text.tfidf.tfidf", text_df, "doc", None),
        ("text.semantics.similarity", text_df, "doc", None),
    ]
    for feature, df, col, params in cases:
        local_r = featm.extract(df, features=[feature], columns={feature: col}, execution="local", batch_rows=19, params=params, errors="raise")
        dask_r = featm.extract(
            df, features=[feature], columns={feature: col}, execution="distributed", distributed_backend="dask",
            distributed_workers=4, distributed_partition_rows=19, output="memory", params=params, errors="raise",
        )
        a = (local_r.to_pandas() if hasattr(local_r, "to_pandas") else local_r).to_numpy(dtype=float)
        b = dask_r.to_pandas().to_numpy(dtype=float)
        assert a.shape == b.shape, feature
        assert np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-8), feature


def test_out_of_order_completion_reassembles_correct_source_order(dask_runtime):
    """Forces partition 0 to be the SLOWEST to complete -- real scheduler out-of-order
    delivery, not assumed. Regression for the ordering contract every stateful
    encoder (especially similarity's first-document semantics) depends on."""
    def slow_for_zero(part):
        if part.index == 0:
            time.sleep(2)
        return part.index * 10

    parts = [Partition(i, None) for i in range(6)]
    stream = dask_runtime.map_stream(slow_for_zero, parts, max_in_flight=6, retries=0)
    ordered = list(_ordered_results(stream))
    assert ordered == [0, 10, 20, 30, 40, 50]


def test_permanent_task_failure_raises_distributed_task_error_not_silent_passthrough(dask_runtime):
    """Regression for a real bug found on real Dask: on this environment's installed
    distributed version, `as_completed(..., raise_errors=False)` hands back a
    `(exc_type, exc, traceback)` 3-tuple on permanent failure, not a bare exception
    instance -- a naive `isinstance(result, BaseException)` check silently misses this
    and lets the error tuple flow downstream as if it were real partition output."""
    def always_fails(part):
        raise RuntimeError(f"permanent failure for partition {part.index}")

    parts = [Partition(i, None) for i in range(3)]
    with pytest.raises(DistributedTaskError) as exc_info:
        list(dask_runtime.map_stream(always_fails, parts, max_in_flight=3, retries=1))
    assert isinstance(exc_info.value.original_exception, RuntimeError)


def test_task_retry_is_idempotent_on_real_dask(dask_runtime, tmp_path):
    """A partition fails on its first REAL (separate-process) attempt and succeeds on
    retry, using a filesystem marker keyed by partition ID (workers are separate
    processes -- no shared memory to coordinate the injected failure otherwise)."""
    marker_dir = tmp_path / "retry_markers"
    marker_dir.mkdir()
    marker_path = str(marker_dir)

    # Must be a real top-level, picklable function -- a closure over `marker_path`
    # captured via default argument (safe to pickle: a plain string).
    def flaky(part, _marker_dir=marker_path):
        import os

        marker = os.path.join(_marker_dir, f"part-{part.index}.attempt")
        attempt = int(open(marker).read()) + 1 if os.path.exists(marker) else 1
        with open(marker, "w") as f:
            f.write(str(attempt))
        if part.index == 1 and attempt == 1:
            raise RuntimeError("intentional first-attempt failure")
        return part.index * 100

    parts = [Partition(i, None) for i in range(4)]
    stream = dask_runtime.map_stream(flaky, parts, max_in_flight=4, retries=2)
    ordered = list(_ordered_results(stream))
    assert sorted(ordered) == [0, 100, 200, 300]
    assert int((marker_dir / "part-1.attempt").read_text()) == 2, "must have retried exactly once, not looped"


def test_errors_raise_propagates_feature_failure_on_priming_partition_real_dask(dask_runtime):
    """Regression: distributed transform used to ALWAYS silently degrade a feature
    failure to NaN-fill regardless of errors=, even on the very first (schema-priming)
    partition -- violating the documented errors="raise" contract every other featm
    entry point honors. A missing required transform-time param (bm25's `query`) is a
    reliable, real trigger."""
    docs = [f"doc number {i} with extra padding text here" for i in range(60)]
    df = pd.DataFrame({"doc": docs})
    with pytest.raises(DistributedFeatureError, match="query"):
        featm.extract(
            df, features=["text.bm25.bm25"], columns={"text.bm25.bm25": "doc"},
            execution="distributed", distributed_backend="dask", distributed_workers=2,
            distributed_partition_rows=10, output="memory", errors="raise",
        )


def test_hash_bucket_features_are_process_stable_on_real_dask(dask_runtime):
    """Regression for a real bug real multi-process Dask validation found:
    text.character.character_ngrams, text.lexical.ngrams, and
    sequence.ngrams.ngrams used the builtin hash() to bucket n-grams -- str
    hashing is randomized PER PROCESS by default (PYTHONHASHSEED), so two
    DIFFERENT partitions computed on two DIFFERENT Dask worker PROCESSES
    silently bucketed the identical n-gram differently. Invisible on the
    local ThreadPool runtime (one process, one hash seed) and invisible on
    every prior single-process CPU run -- only a real multi-process Dask
    comparison against a single-process local reference exposes it. Fixed by
    switching to featm.features._utils.content_seed (already used for
    DeepWalk/EEMD's process-independent RNG seeding)."""
    rng = np.random.default_rng(23)
    words = ["apple", "banana", "car", "road", "fruit", "engine", "yellow", "tropical"]
    docs = [" ".join(rng.choice(words, size=4)) for _ in range(24)]
    df = pd.DataFrame({"text": docs})
    seqs = pd.DataFrame({"s": (["aabb", "abab", "baba", "bbaa", "aaab", "baaa"] * 4)})

    cases = [
        ("text.character.character_ngrams", df, "text"),
        ("text.lexical.ngrams", df, "text"),
        ("sequence.ngrams.ngrams", seqs, "s"),
    ]
    for feature, data, col in cases:
        local_r = featm.extract(data, features=[feature], columns={feature: col}, execution="local", batch_rows=5, errors="raise")
        dask_r = featm.extract(
            data, features=[feature], columns={feature: col}, execution="distributed", distributed_backend="dask",
            distributed_workers=4, distributed_partition_rows=5, output="memory", errors="raise",
        )
        a = local_r.to_pandas().to_numpy(dtype=float) if hasattr(local_r, "to_pandas") else local_r.to_numpy(dtype=float)
        b = dask_r.to_pandas().to_numpy(dtype=float)
        assert np.allclose(a, b), f"{feature}: process-unstable hash bucketing regression"


def test_reorder_buffer_bounded_on_real_dask_slow_partition_zero(dask_runtime):
    """Finding 2 (independent audit), confirmed on REAL multi-process Dask (not
    just the local ThreadPool reference): with max_in_flight=4 and partition 0
    artificially slow among 60 fast partitions, the reassembly buffer must stay
    O(max_in_flight), not grow to include the whole dataset."""
    def slow_for_zero(part):
        if part.index == 0:
            time.sleep(1.5)
        return part.index

    n = 60
    parts = [Partition(i, None) for i in range(n)]
    max_in_flight = 4
    gate = _ReorderGate(window=max_in_flight)
    stats: dict = {}
    stream = dask_runtime.map_stream(
        slow_for_zero, parts, max_in_flight=max_in_flight, retries=0, should_submit_more=gate.should_submit_more,
    )
    ordered = list(_ordered_results(stream, gate=gate, stats=stats))
    assert ordered == list(range(n))
    observed = stats["max_reorder_buffer_observed"]
    assert observed <= 4 * max_in_flight, (
        f"real-Dask reorder buffer grew to {observed} with max_in_flight={max_in_flight} and {n} partitions"
    )


def test_checkpoint_resume_lazy_and_row_independent_matches_local_on_real_dask(tmp_path):
    """Finding 3 (independent audit), confirmed on real Dask: a checkpointed
    distributed run, resumed, reuses every committed shard lazily and still
    matches the local reference exactly."""
    rng = np.random.default_rng(24)
    df = pd.DataFrame({"x": rng.standard_normal(150)})
    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="dask", distributed_workers=3,
        distributed_partition_rows=11, output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    first = featm.extract(df, **kwargs)
    second = featm.extract(df, **kwargs)  # resume against a fully-done checkpoint
    a = local_r.to_numpy()
    b = second.to_pandas().to_numpy()
    assert np.allclose(a, b)
    assert second.report.distributed_report["transform_partitions_computed_this_run"] == 0
    assert second.report.distributed_report["checkpoint_resumed"] is True
    mgr = DistributedCheckpointManager(str(ckpt_dir))
    assert mgr.validate_committed_shards_integrity() == []


def test_partial_fit_failure_not_committed_under_errors_raise_on_real_dask(tmp_path):
    """Finding 1 (independent audit), confirmed on real Dask: the exact reproducer
    from the audit report -- a partition whose fit produces a feature error under
    errors="raise" must not become a reusable committed checkpoint entry, and the
    same failure must reproduce on resume, not silently vanish into an incomplete
    finalized state."""
    long_value = "X" * 20_000
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["a", long_value, "b", "c"],
    })
    features = ["tabular.scaling.zscore", "tabular.encoding.category_frequency"]
    columns = {"tabular.scaling.zscore": "num", "tabular.encoding.category_frequency": "cat"}
    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=features, columns=columns, execution="distributed", distributed_backend="dask",
        distributed_workers=2, distributed_partition_rows=1, memory_budget=65_536,
        checkpoint=str(ckpt_dir), errors="raise", output="memory",
    )
    with pytest.raises(DistributedFeatureError):
        featm.extract(df, **kwargs)
    mgr = DistributedCheckpointManager(str(ckpt_dir))
    assert 1 not in mgr.fit_committed_indices()
    assert not mgr.fit_done()
    with pytest.raises(DistributedFeatureError):
        featm.extract(df, **kwargs)  # resumed: must reproduce the SAME failure, not succeed silently
    assert not mgr.fit_done()


def test_distributed_report_reflects_actual_scheduler_truth_on_real_dask(dask_runtime):
    """Finding 9 (independent audit): the report must reflect the SCHEDULER'S
    real address/worker count, not just echo the caller's input hints."""
    rng = np.random.default_rng(25)
    df = pd.DataFrame({"x": rng.standard_normal(40)})
    r = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="dask", distributed_workers=4, distributed_partition_rows=10,
        output="memory", errors="raise",
    )
    dr = r.report.distributed_report
    assert dr["actual_scheduler_address"] is not None
    assert dr["actual_scheduler_address"] != dr["scheduler_address"]  # caller passed None; the real address is not None
    assert dr["actual_worker_count"] == 4
    assert dr["actual_threads_by_worker"] is not None and len(dr["actual_threads_by_worker"]) == 4


def test_gpu_capable_feature_dispatches_through_unchanged_step3_backend_on_real_dask(dask_runtime):
    """CPU-only sanity: proves backend="cpu" through a distributed worker produces
    IDENTICAL results to backend="cpu" locally for a feature that IS GPU-capable --
    confirming distribution never silently reroutes a GPU-eligible feature's CPU math.
    Real GPU composition is covered by tests/test_distributed_gpu.py (Kaggle-only)."""
    rng = np.random.default_rng(22)
    df = pd.DataFrame(np.vstack([rng.standard_normal(64) for _ in range(50)]))
    local_r = featm.extract(df, features=["timeseries.statistical.mean"], execution="local", backend="cpu", errors="raise")
    dask_r = featm.extract(
        df, features=["timeseries.statistical.mean"], execution="distributed", distributed_backend="dask",
        distributed_workers=2, distributed_partition_rows=10, backend="cpu", output="memory", errors="raise",
    )
    a = local_r.to_numpy()
    b = dask_r.to_pandas().to_numpy()
    assert np.allclose(a, b)


# ==========================================================================
# Finding 1 (final closure audit, v0.4.1 -> v0.4.2), real Dask: multiple
# individually-safe physical row groups must never combine past the
# resolved logical partition bound.
# ==========================================================================
def test_parquet_row_group_grouping_bounded_on_real_dask(dask_runtime, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from featm.distributed.partition_provider import DaskParquetPartitionProvider
    from featm.sources import ParquetSource

    rng = np.random.default_rng(26)
    rows_per_group = [30, 30, 30, 30]
    x = rng.standard_normal(sum(rows_per_group))
    offset = 0
    tables = []
    for n in rows_per_group:
        tables.append(pa.Table.from_pandas(pd.DataFrame({"x": x[offset : offset + n]}), preserve_index=False))
        offset += n
    path = tmp_path / "rg.parquet"
    writer = pq.ParquetWriter(path, tables[0].schema)
    for t in tables:
        writer.write_table(t)
    writer.close()

    provider = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=50)
    for part in provider.iter_partitions(["x"]):
        assert part.frame.expected_rows <= 50, f"partition {part.index} descriptor expected_rows={part.frame.expected_rows} > 50"

    local_r = featm.extract(
        pd.DataFrame({"x": x}), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local",
    )
    dask_r = featm.extract(
        str(path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="dask", distributed_workers=2, distributed_partition_rows=50,
        output="memory", errors="raise",
    )
    a = local_r.to_numpy()
    b = dask_r.to_pandas().to_numpy()
    assert np.allclose(a, b)
    assert dask_r.report.distributed_report["partition_provider"] == "parquet_native"


# ==========================================================================
# Finding 2 (final closure audit, v0.4.1 -> v0.4.2), real Dask: errors=
# "skip"/"warn" must not crash when every selected feature fails, whether
# during the fit phase or during transform priming.
# ==========================================================================
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_all_stateful_fit_failure_matches_local_on_real_dask(dask_runtime, policy):
    long_value = "X" * 20_000
    df = pd.DataFrame({"cat": ["a", long_value, "b", "c"]})
    local_r = featm.extract(
        df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="local", batch_rows=1, memory_budget=65_536, errors=policy,
    )
    dask_r = featm.extract(
        df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="distributed", distributed_backend="dask", distributed_workers=2, distributed_partition_rows=1,
        memory_budget=65_536, errors=policy, output="memory",
    )
    assert local_r.shape == dask_r.shape == (0, 0)
    assert dask_r.report.failed_features == local_r.report.failed_features == ["tabular.encoding.category_frequency"]


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_all_transform_priming_failure_matches_local_on_real_dask(dask_runtime, policy):
    docs = [f"doc number {i} with extra padding text here" for i in range(30)]
    df = pd.DataFrame({"doc": docs})
    dask_r = featm.extract(
        df, features=["text.bm25.bm25"], columns={"text.bm25.bm25": "doc"},
        execution="distributed", distributed_backend="dask", distributed_workers=2, distributed_partition_rows=6,
        output="memory", errors=policy,
    )
    assert dask_r.shape == (0, 0)
    assert dask_r.report.failed_features == ["text.bm25.bm25"]


def test_mixed_fail_survive_selection_on_real_dask(dask_runtime):
    docs = [f"doc number {i} with extra padding text here" for i in range(30)]
    df = pd.DataFrame({"doc": docs, "x": list(range(30))})
    dask_r = featm.extract(
        df, features=["text.bm25.bm25", "tabular.numeric.abs"],
        columns={"text.bm25.bm25": "doc", "tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="dask", distributed_workers=2, distributed_partition_rows=6,
        output="memory", errors="skip",
    )
    assert dask_r.shape == (30, 1)
    assert dask_r.report.failed_features == ["text.bm25.bm25"]
    assert np.allclose(dask_r.to_pandas()["tabular.numeric.abs"].to_numpy(), df["x"].abs().to_numpy())


# ==========================================================================
# Finding 3 (final closure audit, v0.4.1 -> v0.4.2), real Dask: explicitly
# invalid zero/negative public parameters must be rejected BEFORE any
# cluster/task submission, not silently truthiness-defaulted.
# ==========================================================================
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(distributed_max_in_flight=0),
        dict(distributed_max_in_flight=-1),
        dict(distributed_partition_rows=0),
        dict(distributed_workers=0),
        dict(distributed_threads_per_worker=0),
        dict(distributed_task_retries=-1),
    ],
)
def test_invalid_parameters_rejected_before_dask_cluster_creation(kwargs):
    """No dask_runtime fixture dependency deliberately -- this must fail during
    distributed_extract()'s own preflight, before a DaskDistributedRuntime (and
    therefore a real LocalCluster) is ever constructed."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        featm.extract(
            df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
            execution="distributed", distributed_backend="dask", **kwargs,
        )


# ==========================================================================
# Checkpoint budget identity closure (v0.4.2 -> v0.4.3), real Dask: a
# checkpoint's derived state_budget_bytes must be identity-protected, the
# same reproducer as the local-runtime test, confirmed on a real cluster.
# ==========================================================================
def test_resume_rejects_incompatible_state_budget_small_to_large_on_real_dask(dask_runtime, tmp_path):
    from featm.checkpoint import CheckpointIncompatibleError

    long_value = "X" * 20_000
    df = pd.DataFrame({"cat": ["a", long_value, "b", "c"]})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="distributed", distributed_backend="dask", distributed_workers=2, distributed_partition_rows=1,
        errors="skip", output="memory", checkpoint=str(ckpt),
    )
    first = featm.extract(df, memory_budget=65_536, **kwargs)
    assert first.shape == (0, 0)
    with pytest.raises(CheckpointIncompatibleError, match="state_budget_bytes"):
        featm.extract(df, memory_budget=1_048_576, **kwargs)
