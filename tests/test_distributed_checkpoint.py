"""Step 4 completion pass: transactional distributed checkpoint/resume.

Two kinds of coverage here:

- Precise, deterministic CRASH-WINDOW tests against
  :class:`DistributedCheckpointManager` directly -- simulating "the process
  died between these two specific file operations" exactly, which a live
  process-kill test cannot do reliably or repeatably.
- End-to-end resume tests through :func:`featm.extract` (using the
  Dask-independent local reference runtime, like
  ``tests/test_distributed_foundation.py``, so this file does not require
  Dask installed) proving a genuinely partial checkpoint resumes to the
  exact same result as an uninterrupted run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError
from featm.distributed.checkpoint import (
    DistributedCheckpointIdentity,
    DistributedCheckpointManager,
    DistributedCheckpointArtifactTooLargeError,
)


def _identity(**overrides) -> DistributedCheckpointIdentity:
    base = dict(
        featm_version=featm.__version__, source_fingerprint="src-abc", schema_fingerprint="schema-abc",
        features=["f.g.h"], params={}, partition_rows=10, mode="exact", errors="raise",
        columns={"f.g.h": ["str:x"]}, state_budget_bytes=16_384,
    )
    base.update(overrides)
    return DistributedCheckpointIdentity(**base)


# ==========================================================================
# Deterministic crash-window tests against DistributedCheckpointManager.
# ==========================================================================
def test_crash_after_state_file_write_before_manifest_commit_leaves_safe_orphan(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())

    # Simulate exactly "the data file was durably renamed into place, then the
    # process died before the manifest update" -- write the state file the same
    # way commit_fit_partition() does, but do NOT update the manifest.
    import pickle

    (mgr.fit_state_dir / "part-000000.pkl").write_bytes(pickle.dumps({"f.g.h": 42}))

    # The orphan file exists, but the manifest does not list it as committed.
    assert 0 not in mgr.fit_committed_indices()
    # A fresh manager reading the same directory must not be confused by the orphan.
    mgr2 = DistributedCheckpointManager(tmp_path)
    assert mgr2.fit_committed_indices() == set()
    # And a real commit for the same index must succeed cleanly (overwriting the orphan).
    mgr2.commit_fit_partition(0, {"f.g.h": 99})
    assert mgr2.load_fit_partition(0) == {"states": {"f.g.h": 99}, "feature_errors": {}}
    assert mgr2.fit_committed_indices() == {0}


def test_crash_during_manifest_write_leaves_old_manifest_valid(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_fit_partition(0, {"f.g.h": 1})
    before = mgr.load()

    # Simulate "the manifest's temp file was written but os.replace never
    # happened" -- leave a stray .tmp file next to the real manifest. The real
    # manifest, from the LAST successful atomic replace, must still be exactly
    # what it was before the interrupted write.
    tmp_manifest = mgr.manifest_path.with_suffix(mgr.manifest_path.suffix + ".tmp999999")
    tmp_manifest.write_text(json.dumps({"corrupted": True}))

    after = mgr.load()
    assert after == before, "the real manifest must be untouched by a stray/incomplete temp write"
    assert after["fit_committed_partitions"] == [0]


def test_retry_same_partition_twice_is_idempotent_not_double_counted(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_fit_partition(2, {"f.g.h": "first-attempt-state"})
    mgr.commit_fit_partition(2, {"f.g.h": "second-attempt-state"})  # a Dask retry succeeding twice
    assert mgr.fit_committed_indices() == {2}, "the SET of committed indices must not double-count a repeated commit"
    assert mgr.load_fit_partition(2)["states"] == {"f.g.h": "second-attempt-state"}, "the last write wins, deterministically"


def test_duplicate_transform_shard_commit_does_not_create_two_files(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    frame1 = pd.DataFrame({"c": [1.0, 2.0]})
    frame2 = pd.DataFrame({"c": [1.0, 2.0]})  # identical content, simulating a genuine retry recomputation
    mgr.commit_transform_partition(0, frame1)
    mgr.commit_transform_partition(0, frame2)
    assert mgr.transform_committed_indices() == {0}
    shards = list(mgr.output_dir.glob("part-000000*.parquet"))
    assert len(shards) == 1, f"expected exactly one committed shard file, found {shards}"


def test_resume_rejects_incompatible_source_fingerprint(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity(source_fingerprint="src-A"))
    mgr2 = DistributedCheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="source_fingerprint"):
        mgr2.initialize_or_resume(_identity(source_fingerprint="src-B"))


def test_resume_rejects_incompatible_partition_rows(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity(partition_rows=50))
    mgr2 = DistributedCheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="partition_rows"):
        mgr2.initialize_or_resume(_identity(partition_rows=25))


def test_resume_rejects_incompatible_features_or_params(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity(features=["a.b.c"]))
    mgr2 = DistributedCheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="features"):
        mgr2.initialize_or_resume(_identity(features=["x.y.z"]))


def test_completed_checkpoint_integrity_detects_missing_shard(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_transform_partition(0, pd.DataFrame({"c": [1.0]}))
    mgr.commit_transform_partition(1, pd.DataFrame({"c": [2.0]}))
    mgr.mark_transform_done()
    # Simulate an out-of-band deletion (disk cleanup tool, accidental rm) of a
    # shard the manifest still claims is committed.
    (mgr.output_dir / "part-000001.parquet").unlink()
    problems = mgr.validate_committed_shards_integrity()
    assert any("partition 1" in p and "missing" in p for p in problems)


def test_completed_checkpoint_integrity_detects_corrupt_shard(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_transform_partition(0, pd.DataFrame({"c": [1.0]}))
    mgr.mark_transform_done()
    (mgr.output_dir / "part-000000.parquet").write_bytes(b"not a real parquet file")
    problems = mgr.validate_committed_shards_integrity()
    assert any("partition 0" in p and "corrupt" in p for p in problems)


def test_completed_checkpoint_integrity_clean_when_all_shards_present(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_fit_partition(0, {"f.g.h": 1})
    mgr.commit_transform_partition(0, pd.DataFrame({"c": [1.0]}))
    assert mgr.validate_committed_shards_integrity() == []


# ==========================================================================
# End-to-end resume through featm.extract() (local reference runtime).
# ==========================================================================
def test_end_to_end_resume_from_partial_checkpoint_matches_uninterrupted_run(tmp_path):
    rng = np.random.default_rng(30)
    df = pd.DataFrame({"x": rng.standard_normal(200), "cat": rng.choice(["a", "b"], 200)})
    features = ["tabular.scaling.zscore", "tabular.encoding.category_frequency"]
    columns = {"tabular.scaling.zscore": "x", "tabular.encoding.category_frequency": "cat"}

    local_r = featm.extract(df, features=features, columns=columns, execution="local", batch_rows=20, errors="raise")

    ckpt_dir = tmp_path / "ckpt"
    full_r = featm.extract(
        df, features=features, columns=columns, execution="distributed", distributed_backend="local",
        distributed_workers=4, distributed_partition_rows=20, output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    a = local_r.to_pandas().to_numpy(dtype=float)
    b = full_r.to_pandas().to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)

    # Simulate a crash that lost SOME committed progress (both fit and transform).
    mgr = DistributedCheckpointManager(str(ckpt_dir))
    manifest = mgr.load()
    for idx in (1, 3):
        shard = mgr.output_dir / f"part-{idx:06d}.parquet"
        if shard.exists():
            shard.unlink()
    manifest["transform_committed_partitions"] = [i for i in manifest["transform_committed_partitions"] if i not in (1, 3)]
    manifest["transform_done"] = False
    for idx in (2,):
        state = mgr.fit_state_dir / f"part-{idx:06d}.pkl"
        if state.exists():
            state.unlink()
    manifest["fit_committed_partitions"] = [i for i in manifest["fit_committed_partitions"] if i not in (2,)]
    manifest["fit_done"] = False
    finalized = mgr.directory / "fit_finalized.pkl"
    if finalized.exists():
        finalized.unlink()
    with open(mgr.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    resumed_r = featm.extract(
        df, features=features, columns=columns, execution="distributed", distributed_backend="local",
        distributed_workers=4, distributed_partition_rows=20, output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    c = resumed_r.to_pandas().to_numpy(dtype=float)
    assert np.allclose(a, c, equal_nan=True)
    assert resumed_r.report.distributed_report["checkpoint_resumed"] is True


def test_resume_against_completed_checkpoint_is_fast_and_correct_zero_recompute(tmp_path):
    """A resume against an ALREADY fully-done checkpoint must not recompute the fit
    merge at all (fit_partitions_computed_this_run == 0) and must reuse every transform
    shard, while still returning the exact right values."""
    rng = np.random.default_rng(31)
    df = pd.DataFrame({"x": rng.standard_normal(120)})
    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, execution="distributed",
        distributed_backend="local", distributed_workers=4, distributed_partition_rows=15, output="memory",
        errors="raise", checkpoint=str(ckpt_dir),
    )
    first = featm.extract(df, **kwargs)
    second = featm.extract(df, **kwargs)
    assert np.allclose(first.to_pandas().to_numpy(), second.to_pandas().to_numpy())
    assert second.report.distributed_report["fit_partitions_computed_this_run"] == 0
    assert second.report.distributed_report["transform_partitions_computed_this_run"] == 0
    assert second.report.distributed_report["checkpoint_resumed"] is True


def test_incompatible_resume_via_public_extract_api_raises_clearly(tmp_path):
    rng = np.random.default_rng(32)
    df = pd.DataFrame({"x": rng.standard_normal(60)})
    ckpt_dir = tmp_path / "ckpt"
    featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="distributed",
        distributed_backend="local", distributed_partition_rows=10, output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(
            df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="distributed",
            distributed_backend="local", distributed_partition_rows=7, output="memory", errors="raise", checkpoint=str(ckpt_dir),
        )


# ==========================================================================
# Finding 6 (independent audit): mode/errors/typed-label identity coverage.
# ==========================================================================
def test_resume_rejects_incompatible_errors_policy(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity(errors="raise"))
    mgr2 = DistributedCheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="errors"):
        mgr2.initialize_or_resume(_identity(errors="skip"))


def test_resume_rejects_incompatible_mode(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity(mode="exact"))
    mgr2 = DistributedCheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="mode"):
        mgr2.initialize_or_resume(_identity(mode="approximate"))


def test_checkpoint_identity_uses_canonical_label_not_bare_str(tmp_path):
    """An int-labeled column ``0`` and a string-labeled column ``"0"`` are distinct
    pandas labels; a bare str() identity would treat them as the same checkpoint."""
    df = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0], "0": [10.0, 20.0, 30.0, 40.0]})
    ckpt_dir = tmp_path / "ckpt"
    featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": 0}, execution="distributed",
        distributed_backend="local", distributed_partition_rows=2, output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(
            df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "0"}, execution="distributed",
            distributed_backend="local", distributed_partition_rows=2, output="memory", errors="raise", checkpoint=str(ckpt_dir),
        )


# ==========================================================================
# Finding 5 (independent audit): content-checksum integrity, not just
# "opens cleanly" -- a valid-but-different pickle/parquet must be detected.
# ==========================================================================
def test_integrity_detects_fit_partition_tampered_with_different_valid_pickle(tmp_path):
    import pickle

    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_fit_partition(0, {"f.g.h": 1.0})
    assert mgr.validate_committed_shards_integrity() == []
    tampered = pickle.dumps({"states": {"f.g.h": 999.0}, "feature_errors": {}})
    (mgr.fit_state_dir / "part-000000.pkl").write_bytes(tampered)
    problems = mgr.validate_committed_shards_integrity()
    assert any("partition 0" in p and "checksum mismatch" in p for p in problems)


def test_integrity_detects_fit_finalized_tampered_with_different_valid_pickle(tmp_path):
    import pickle

    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_fit_finalized({"f.g.h": {"mean": 1.0}})
    mgr.mark_fit_done()
    assert mgr.validate_committed_shards_integrity() == []
    (mgr.directory / "fit_finalized.pkl").write_bytes(pickle.dumps({"f.g.h": {"mean": 999.0}}))
    problems = mgr.validate_committed_shards_integrity()
    assert any("fit_finalized.pkl" in p and "checksum mismatch" in p for p in problems)


def test_integrity_detects_transform_shard_tampered_with_different_valid_parquet_same_shape(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_transform_partition(0, pd.DataFrame({"c": [1.0, 2.0]}))
    assert mgr.validate_committed_shards_integrity() == []
    # Same schema, same row count, DIFFERENT values -- must still be caught.
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = mgr.output_dir / "part-000000.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"c": [999.0, 888.0]}), preserve_index=False), path)
    problems = mgr.validate_committed_shards_integrity()
    assert any("partition 0" in p and "checksum mismatch" in p for p in problems)


def test_integrity_detects_truncated_transform_shard(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_transform_partition(0, pd.DataFrame({"c": [1.0, 2.0, 3.0]}))
    path = mgr.output_dir / "part-000000.parquet"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    problems = mgr.validate_committed_shards_integrity()
    assert any("partition 0" in p for p in problems)


# ==========================================================================
# Finding 1 (independent audit): a partition with ANY feature error under
# errors="raise" must not become a reusable "successful" checkpoint entry.
# ==========================================================================
def test_partial_fit_failure_not_committed_under_errors_raise_and_reproduces_on_resume(tmp_path):
    long_value = "X" * 20_000
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["a", long_value, "b", "c"],
    })
    features = ["tabular.scaling.zscore", "tabular.encoding.category_frequency"]
    columns = {"tabular.scaling.zscore": "num", "tabular.encoding.category_frequency": "cat"}
    ckpt_dir = tmp_path / "ckpt"

    kwargs = dict(
        features=features, columns=columns, execution="distributed", distributed_backend="local",
        distributed_workers=2, distributed_partition_rows=1, memory_budget=65_536,
        checkpoint=str(ckpt_dir), errors="raise", output="memory",
    )

    from featm.distributed.contracts import DistributedFeatureError

    with pytest.raises(DistributedFeatureError):
        featm.extract(df, **kwargs)

    mgr = DistributedCheckpointManager(str(ckpt_dir))
    # Row index 1 (the long_value row) is its own partition (distributed_partition_rows=1)
    # and is the one whose category_frequency fit raises StateBudgetExceededError.
    # Under errors="raise" it must NOT become a committed, reusable "successful"
    # fit partition -- committing it (even partially, for zscore alone) would let
    # a resumed run skip re-fitting it and silently finalize a merged
    # category_frequency state missing this partition's contribution, with the
    # original error nowhere reproduced.
    assert 1 not in mgr.fit_committed_indices()
    # Every partition that WAS committed (0, 2, 3 -- the short-category rows) must
    # carry no feature_errors of its own; none of them is the one that failed.
    for idx in mgr.fit_committed_indices():
        data = mgr.load_fit_partition(idx)
        assert data["feature_errors"] == {}
    assert not mgr.fit_done()

    # The SAME checkpoint, resumed, must raise the SAME failure again --
    # not silently succeed with an incomplete merged state.
    with pytest.raises(DistributedFeatureError):
        featm.extract(df, **kwargs)
    assert not mgr.fit_done()


def test_partial_fit_failure_persisted_and_reproduced_under_errors_skip(tmp_path):
    """Under errors!='raise', a partition's partial fit failure is committed together
    with its feature_errors record, so an uninterrupted run and a resumed run agree
    on exactly which feature failed and why -- not just that a feature failed."""
    long_value = "X" * 20_000
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["a", long_value, "b", "c"],
    })
    features = ["tabular.scaling.zscore", "tabular.encoding.category_frequency"]
    columns = {"tabular.scaling.zscore": "num", "tabular.encoding.category_frequency": "cat"}

    uninterrupted = featm.extract(
        df, features=features, columns=columns, execution="distributed", distributed_backend="local",
        distributed_workers=2, distributed_partition_rows=1, memory_budget=65_536, errors="skip", output="memory",
    )

    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=features, columns=columns, execution="distributed", distributed_backend="local",
        distributed_workers=2, distributed_partition_rows=1, memory_budget=65_536,
        checkpoint=str(ckpt_dir), errors="skip", output="memory",
    )
    first = featm.extract(df, **kwargs)
    second = featm.extract(df, **kwargs)  # resume against a fully-done checkpoint

    # zscore succeeded in all three runs; category_frequency failed in all three,
    # and must be ABSENT from the output the same way in every run.
    assert list(uninterrupted.to_pandas().columns) == list(first.to_pandas().columns) == list(second.to_pandas().columns)
    assert "tabular.scaling.zscore" in "".join(uninterrupted.to_pandas().columns)
    assert not any("category_frequency" in c for c in second.to_pandas().columns)
    assert second.report.distributed_report["fit_partitions_computed_this_run"] == 0


# ==========================================================================
# Finding 3 (independent audit): checkpoint resume loads committed partitions
# LAZILY -- one at a time, at the moment each is needed -- not all upfront.
# ==========================================================================
def test_checkpoint_resume_loads_committed_partitions_lazily_not_all_upfront(tmp_path, monkeypatch):
    rng = np.random.default_rng(41)
    df = pd.DataFrame({"x": rng.standard_normal(200)})
    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="distributed",
        distributed_backend="local", distributed_workers=1, distributed_partition_rows=10, output="memory",
        errors="raise", checkpoint=str(ckpt_dir),
    )
    featm.extract(df, **kwargs)  # fully commits 20 transform shards

    load_order: list[int] = []
    original = DistributedCheckpointManager.load_transform_partition

    def _tracked(self, index):
        load_order.append(index)
        assert load_order == sorted(load_order), "checkpoint partitions must be loaded in logical order, one at a time"
        return original(self, index)

    monkeypatch.setattr(DistributedCheckpointManager, "load_transform_partition", _tracked)
    featm.extract(df, **kwargs)  # resume: every partition reused from checkpoint
    assert load_order == list(range(20)), "every committed partition must be loaded exactly once, in order"


# ==========================================================================
# Finding 2 (independent audit): the reorder/reassembly buffer is bounded by
# a small function of max_in_flight, not by the total partition count.
# ==========================================================================
def test_reorder_buffer_bounded_when_one_early_partition_is_slow(tmp_path):
    import time

    from featm.distributed.contracts import Partition
    from featm.distributed.execution import _ReorderGate, _ordered_results
    from featm.distributed.local import LocalThreadRuntime

    n = 1000
    parts = [Partition(i, i) for i in range(n)]

    def fn(part):
        if part.index == 0:
            time.sleep(0.6)
        else:
            time.sleep(0.001)
        return part.index

    runtime = LocalThreadRuntime(n_workers=2)
    try:
        max_in_flight = 2
        gate = _ReorderGate(window=max_in_flight)
        stats: dict = {}
        out = list(_ordered_results(
            runtime.map_stream(fn, parts, max_in_flight=max_in_flight, retries=0, should_submit_more=gate.should_submit_more),
            gate=gate, stats=stats,
        ))
    finally:
        runtime.close()

    assert out == list(range(n)), "delivery order must still be exactly correct"
    observed = stats["max_reorder_buffer_observed"]
    assert observed <= 4 * max_in_flight, (
        f"reorder buffer grew to {observed} with max_in_flight={max_in_flight} and 1000 partitions -- "
        "expected O(max_in_flight), not O(total partitions) (v0.4.0 grew this to 100/100 in the equivalent test)"
    )


# ==========================================================================
# Finding 13.3 / Finding 4 (final closure audit): a checkpoint artifact whose
# ON-DISK size is far larger than the configured budget is rejected as a
# disk-size guard -- streamed via pickle.dump, never via pickle.dumps, so
# committing it never held a second full in-memory copy in the first place.
# ==========================================================================
def test_commit_fit_partition_rejects_state_far_exceeding_budget(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    huge_state = {"f.g.h": "X" * 10_000}
    with pytest.raises(DistributedCheckpointArtifactTooLargeError):
        mgr.commit_fit_partition(0, huge_state, state_budget_bytes=100)
    # The oversized temp file must not be left behind, and nothing was committed.
    assert mgr.fit_committed_indices() == set()
    assert not any(mgr.fit_state_dir.glob("*.tmp*"))


def test_large_checkpoint_commit_load_integrity_never_use_full_buffer_pickle_or_read_bytes(tmp_path, monkeypatch):
    """Finding 4 (final closure audit): spy on pickle.dumps and Path.read_bytes across a
    full commit/load/integrity cycle with a deliberately large payload -- neither must be
    called anywhere in the production checkpoint code path. Atomic commit semantics,
    checksums, and tamper detection must all still work identically (no regression on
    Finding 5)."""
    import pickle as pickle_module

    from featm.distributed import checkpoint as ckpt_module

    dumps_calls: list[Any] = []
    orig_dumps = pickle_module.dumps

    def _spy_dumps(*a, **k):
        dumps_calls.append(1)
        return orig_dumps(*a, **k)

    monkeypatch.setattr(ckpt_module.pickle, "dumps", _spy_dumps)

    read_bytes_calls: list[str] = []
    orig_read_bytes = Path.read_bytes

    def _spy_read_bytes(self):
        read_bytes_calls.append(str(self))
        return orig_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _spy_read_bytes)

    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    large_state = {"f.g.h": "Y" * 200_000}
    mgr.commit_fit_partition(0, large_state)
    loaded = mgr.load_fit_partition(0)
    assert loaded["states"] == large_state

    mgr.commit_fit_finalized({"f.g.h": large_state})
    mgr.mark_fit_done()
    assert mgr.load_fit_finalized() == {"f.g.h": large_state}

    frame = pd.DataFrame({"c": list(range(5000))})
    mgr.commit_transform_partition(0, frame)
    assert len(mgr.load_transform_partition(0)) == 5000

    assert mgr.validate_committed_shards_integrity() == []

    assert dumps_calls == [], "pickle.dumps must not be used for checkpoint artifact commits -- use streaming pickle.dump"
    assert read_bytes_calls == [], f"Path.read_bytes must not be used for checkpoint artifacts: {read_bytes_calls}"

    # Now prove tamper detection (Finding 5) still works after the streaming rewrite.
    (mgr.fit_state_dir / "part-000000.pkl").write_bytes(
        orig_dumps({"states": {"f.g.h": "TAMPERED"}, "feature_errors": {}})
    )
    problems = mgr.validate_committed_shards_integrity()
    assert any("partition 0" in p and "checksum mismatch" in p for p in problems)


def test_commit_fit_partition_allows_state_within_budget(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    mgr.initialize_or_resume(_identity())
    mgr.commit_fit_partition(0, {"f.g.h": 1.0}, state_budget_bytes=10_000)
    assert mgr.fit_committed_indices() == {0}


# ==========================================================================
# Finding 7 (independent audit): checkpointing must fail clearly BEFORE any
# fit work starts if PyArrow (required for transform-shard commits) is
# unavailable -- not mid-run with a raw ModuleNotFoundError.
# ==========================================================================
def test_checkpoint_without_pyarrow_fails_clearly_before_any_work(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    rng = np.random.default_rng(43)
    df = pd.DataFrame({"x": rng.standard_normal(20)})
    with pytest.raises(ImportError, match=r"featm\[io,distributed\]"):
        featm.extract(
            df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="distributed",
            distributed_backend="local", distributed_partition_rows=5, output="memory", errors="raise",
            checkpoint=str(tmp_path / "ckpt"),
        )
    # No partial checkpoint directory/manifest should be left behind by a
    # preflight failure that happens before any work starts.
    assert not (tmp_path / "ckpt" / "distributed_checkpoint.json").exists()


# ==========================================================================
# Checkpoint budget identity closure (v0.4.2 -> v0.4.3): state_budget_bytes
# -- the NORMALIZED value derive_state_budget_bytes() actually hands to
# every stateful encoder -- must be part of checkpoint identity. For an
# exact stateful feature (category_frequency, TF-IDF/BM25/surprisal), this
# budget directly decides fit success/failure under errors="skip"/"warn",
# so a resume under a materially different budget must be rejected, not
# silently reused with stale pass/fail semantics.
# ==========================================================================
def _budget_reproducer_df():
    long_value = "X" * 20_000
    return pd.DataFrame({"cat": ["a", long_value, "b", "c"]})


_BUDGET_KWARGS = dict(
    features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
    execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=1,
    errors="skip", output="memory",
)


def test_budget_changes_result_without_checkpoint_sanity_check():
    """Establishes the premise the rest of this section depends on: the SAME
    feature/data genuinely produces a DIFFERENT result under different budgets."""
    df = _budget_reproducer_df()
    small = featm.extract(df, memory_budget=65_536, **_BUDGET_KWARGS)
    large = featm.extract(df, memory_budget=1_048_576, **_BUDGET_KWARGS)
    assert small.shape == (0, 0)
    assert small.report.failed_features == ["tabular.encoding.category_frequency"]
    assert large.shape == (4, 1)
    assert large.report.failed_features == []


def test_resume_rejects_incompatible_state_budget_small_to_large(tmp_path):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_BUDGET_KWARGS)
    assert first.shape == (0, 0)
    with pytest.raises(CheckpointIncompatibleError, match="state_budget_bytes"):
        featm.extract(df, memory_budget=1_048_576, checkpoint=str(ckpt), **_BUDGET_KWARGS)


def test_resume_rejects_incompatible_state_budget_large_to_small(tmp_path):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=1_048_576, checkpoint=str(ckpt), **_BUDGET_KWARGS)
    assert first.shape == (4, 1)
    with pytest.raises(CheckpointIncompatibleError, match="state_budget_bytes"):
        featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_BUDGET_KWARGS)


def test_resume_with_equivalent_derived_state_budget_remains_compatible(tmp_path):
    """Two different raw memory_budget= values that normalize to the SAME
    derive_state_budget_bytes() (and same partition_rows) must remain
    compatible -- only the derived value is identity-relevant, never the raw text."""
    from featm.resources import derive_state_budget_bytes, parse_size

    assert derive_state_budget_bytes(parse_size(65_536)) == derive_state_budget_bytes(parse_size(65_537))
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_BUDGET_KWARGS)
    second = featm.extract(df, memory_budget=65_537, checkpoint=str(ckpt), **_BUDGET_KWARGS)
    assert first.shape == second.shape == (0, 0)
    assert second.report.distributed_report["checkpoint_resumed"] is True


def test_stateless_only_checkpoint_not_invalidated_by_memory_budget_change(tmp_path):
    """A checkpoint for a run with NO stateful features must not be invalidated by an
    unrelated memory_budget change -- the result never depended on the state budget."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=1,
        errors="raise", output="memory", checkpoint=str(ckpt),
    )
    first = featm.extract(df, memory_budget=65_536, **kwargs)
    second = featm.extract(df, memory_budget=999_999_999, **kwargs)
    assert first.shape == second.shape == (4, 1)
    assert second.report.distributed_report["checkpoint_resumed"] is True


def test_stateless_checkpoint_identity_records_none_state_budget(tmp_path):
    mgr = DistributedCheckpointManager(tmp_path)
    identity = _identity(features=["a.b.c"], state_budget_bytes=None)
    manifest = mgr.initialize_or_resume(identity)
    assert manifest["state_budget_bytes"] is None
    mgr2 = DistributedCheckpointManager(tmp_path)
    mgr2.initialize_or_resume(_identity(features=["a.b.c"], state_budget_bytes=None))  # must not raise


def test_old_v2_manifest_without_state_budget_rejected_as_incompatible(tmp_path):
    """A checkpoint written under the OLD (v2) manifest schema -- before
    state_budget_bytes existed in identity at all -- must be rejected
    unambiguously on resume, not field-compared against a manifest missing a
    key the current schema requires (which could coincidentally "match" a
    stateless run's real None and mask exactly the stale-reuse bug this
    field exists to catch)."""
    mgr = DistributedCheckpointManager(tmp_path)
    v2_manifest = {
        "manifest_version": 2,
        "featm_version": featm.__version__, "source_fingerprint": "src-abc", "schema_fingerprint": "schema-abc",
        "features": ["f.g.h"], "params": {}, "partition_rows": 10, "mode": "exact", "errors": "raise",
        "columns": {"f.g.h": ["str:x"]},
        # deliberately no "state_budget_bytes" key -- the pre-v3 shape
        "fit_committed_partitions": [], "fit_partition_checksums": {}, "fit_finalized_checksum": None,
        "fit_failed_features": {}, "transform_committed_partitions": [], "transform_partition_checksums": {},
        "transform_partition_row_counts": {}, "fit_done": False, "transform_done": False,
    }
    mgr.manifest_path.write_text(json.dumps(v2_manifest), encoding="utf-8")
    mgr2 = DistributedCheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="manifest_version"):
        mgr2.initialize_or_resume(_identity(features=["f.g.h"], state_budget_bytes=16_384))


# ==========================================================================
# Successful all-fit-failed distributed checkpoint must mark transform_done
# (v0.4.2 left it False forever after a genuinely successful run with no
# transform phase -- inconsistent completion metadata, not a numerical bug).
# ==========================================================================
def test_all_fit_failed_successful_checkpoint_marks_transform_done(tmp_path):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_BUDGET_KWARGS)
    assert first.shape == (0, 0)
    mgr = DistributedCheckpointManager(str(ckpt))
    assert mgr.fit_done() is True
    assert mgr.transform_done() is True

    second = featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_BUDGET_KWARGS)
    assert second.shape == (0, 0)
    assert second.report.failed_features == ["tabular.encoding.category_frequency"]
    assert second.report.distributed_report["checkpoint_resumed"] is True
    assert second.report.distributed_report["fit_partitions_computed_this_run"] == 0
    assert second.report.distributed_report["transform_partitions_computed_this_run"] == 0
