"""Local checkpoint replay-history closure pass (v0.4.3 -> v0.4.4).

The v0.4.3 checkpoint-budget-identity closure pass's own handoff explicitly
acknowledged a residual local checkpoint gap and claimed it was not
reachable under a SAME-budget resume. That claim was false, independently
reproduced here exactly: a stateful feature that fails during fit under
``errors="skip"``/``"warn"`` is never persisted as failed anywhere in the
local ``CheckpointManager`` manifest (only successfully-fitted state files
are recorded) -- so a resume whose fit phase already completed (``manifest
.get("fit_state_files")`` truthy-guarded the old restore branch) silently
lost the fact that a feature had failed at all, once every selected
stateful feature failed (an empty ``fit_state_files`` dict is falsy, so the
whole restore branch never ran). The exact same class of gap existed for
transform-priming failures (persisted schema only, never the failure) and
for the zero-read "done" fast path (reconstructs completed features from
``primed_columns``, never historical failures).

Fixed by persisting ``fit_failed_features``/``transform_priming_failed_
features``/``degraded_batches_by_feature`` in the checkpoint manifest,
atomically alongside the progress/phase they belong to, and restoring them
on every resume path: mid-fit partial resume, mid-transform partial resume,
and the zero-read "done" fast path. ``CHECKPOINT_MANIFEST_VERSION`` bumped
1 -> 2 so an old (pre-fix) checkpoint is rejected as incompatible rather
than silently resumed into the same history-loss state this pass closes.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError
from featm.sources import CSVSource


@pytest.fixture
def _pyarrow():
    return pytest.importorskip("pyarrow")


def _budget_reproducer_df():
    long_value = "X" * 20_000
    return pd.DataFrame({"cat": ["a", long_value, "b", "c"]})


_CATFREQ_KWARGS = dict(
    features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
    execution="local", batch_rows=1, memory_budget=65_536,
)


# ==========================================================================
# A. all-stateful-fit-failed, same-budget completed (zero-read) resume.
# ==========================================================================
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_all_fit_failed_same_budget_completed_resume_preserves_failure(_pyarrow, tmp_path, policy):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    kwargs = dict(_CATFREQ_KWARGS, errors=policy, checkpoint=str(ckpt))

    first = featm.extract(df, **kwargs)
    assert first.shape == (0, 0)
    assert first.report.failed_features == ["tabular.encoding.category_frequency"]

    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is True

    second = featm.extract(df, **kwargs)
    assert second.shape == (0, 0)
    assert second.report.failed_features == ["tabular.encoding.category_frequency"]
    assert second.report.resumed is True
    # Zero-read: the source was never re-scanned.
    assert second.report.n_batches == manifest.get("transform_shards_written", 0)
    # The specific failure class (StateBudgetExceededError) must survive, not
    # be replaced by a generic message.
    assert any("StateBudgetExceededError" in w for w in second.report.warnings)


# ==========================================================================
# B. mixed stateful fit success + failure.
# ==========================================================================
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_mixed_fit_success_and_failure_resume_matches_fresh(_pyarrow, tmp_path, policy):
    long_value = "X" * 20_000
    df = pd.DataFrame({"cat": ["a", long_value, "b", "c"], "x": [1.0, 2.0, 3.0, 4.0]})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "x", "tabular.encoding.category_frequency": "cat"},
        execution="local", batch_rows=1, memory_budget=65_536, errors=policy, checkpoint=str(ckpt),
    )
    first = featm.extract(df, **kwargs)
    assert first.shape == (4, 1)
    assert first.report.failed_features == ["tabular.encoding.category_frequency"]
    assert first.report.completed_features == ["tabular.scaling.zscore"]

    second = featm.extract(df, **kwargs)
    assert second.shape == (4, 1)
    assert second.report.failed_features == ["tabular.encoding.category_frequency"]
    assert second.report.completed_features == ["tabular.scaling.zscore"]
    assert np.allclose(first.to_pandas().to_numpy(), second.to_pandas().to_numpy())
    # The failed feature must not become a generic "unfitted stateful" skip.
    assert not any(
        "no fitted state for this transform" in w for w in second.report.warnings
    ), "a genuinely-failed stateful feature must not be miscategorized as merely unfitted on resume"


# ==========================================================================
# C. transform-priming failure (deterministic: BM25 with no query=).
# ==========================================================================
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_transform_priming_failure_resume_matches_fresh(_pyarrow, tmp_path, policy):
    docs = [f"doc number {i} with extra padding text here" for i in range(10)]
    df = pd.DataFrame({"doc": docs})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["text.bm25.bm25"], columns={"text.bm25.bm25": "doc"},
        execution="local", batch_rows=2, errors=policy, checkpoint=str(ckpt),
    )
    first = featm.extract(df, **kwargs)
    assert first.shape == (0, 0)
    assert first.report.failed_features == ["text.bm25.bm25"]

    second = featm.extract(df, **kwargs)
    assert second.shape == (0, 0)
    assert second.report.failed_features == ["text.bm25.bm25"]
    assert second.report.resumed is True


# ==========================================================================
# E. partial checkpoint -- a REAL simulated crash (a flaky source that
# raises mid-scan, the same technique tests/test_execution.py already uses
# for its own interrupted-run tests), not a hand-edited manifest.
# ==========================================================================
def test_partial_fit_resume_after_simulated_crash_preserves_failure_and_does_not_reretry(_pyarrow, tmp_path):
    """Feature fails on batch 1 (the long-value row); the process is killed after
    batch 2 completes (simulated); resume must NOT re-attempt fitting the
    already-failed feature starting from batch 2 onward (which would skip
    right over the row that caused the original failure)."""
    long_value = "X" * 20_000
    df = pd.DataFrame({"cat": ["a", long_value, "b", "c", "d", "e"]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    orig_iter = CSVSource.iter_batches
    state = {"total": 0, "fail_after": 3}  # allow batches 0,1,2 through (3 total), crash on the 4th

    def flaky_iter(self, *, columns=None, batch_rows=50_000):
        for batch in orig_iter(self, columns=columns, batch_rows=batch_rows):
            state["total"] += 1
            if state["total"] > state["fail_after"]:
                raise RuntimeError("SIMULATED CRASH DURING FIT")
            yield batch

    flaky = CSVSource(csv_path)
    flaky.iter_batches = types.MethodType(flaky_iter, flaky)
    ckpt = tmp_path / "ckpt"
    kwargs = dict(_CATFREQ_KWARGS, errors="skip", checkpoint=str(ckpt))
    with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
        featm.extract(flaky, **kwargs)

    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["phase"] == "fit"
    assert manifest["fit_batches_completed"] == 3
    # The failure (batch index 1, the long-value row) must already be durably
    # committed even though the fit phase as a whole has not finished.
    assert "tabular.encoding.category_frequency" in manifest["fit_failed_features"]

    resumed = featm.extract(csv_path, **kwargs)
    uninterrupted = featm.extract(
        df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="local", batch_rows=1, memory_budget=65_536, errors="skip",
    )
    assert resumed.shape == uninterrupted.shape == (0, 0)
    assert resumed.report.failed_features == uninterrupted.report.failed_features == ["tabular.encoding.category_frequency"]


def test_partial_transform_resume_after_simulated_crash_preserves_priming_failure(_pyarrow, tmp_path):
    docs = [f"doc number {i} with extra padding text here" for i in range(10)]
    df = pd.DataFrame({"doc": docs, "x": list(range(10))})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    orig_iter = CSVSource.iter_batches
    # text.bm25.bm25 is STATEFUL -- it has its own fit-phase scan (5 batches:
    # 10 rows / batch_rows=2) that reads from this SAME flaky source BEFORE
    # the transform phase even starts. Let the whole fit phase through, then
    # exactly the transform priming batch (batch 0 of the transform phase),
    # THEN crash -- simulating "the process died right after committing the
    # priming failure, before any further transform batch."
    state = {"total": 0, "fail_after": 6}

    def flaky_iter(self, *, columns=None, batch_rows=50_000):
        for batch in orig_iter(self, columns=columns, batch_rows=batch_rows):
            state["total"] += 1
            if state["total"] > state["fail_after"]:
                raise RuntimeError("SIMULATED CRASH DURING TRANSFORM")
            yield batch

    flaky = CSVSource(csv_path)
    flaky.iter_batches = types.MethodType(flaky_iter, flaky)
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["text.bm25.bm25", "tabular.numeric.abs"], columns={"text.bm25.bm25": "doc", "tabular.numeric.abs": "x"},
        execution="local", batch_rows=2, errors="skip", output="disk", checkpoint=str(ckpt),
    )
    with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
        featm.extract(flaky, **kwargs)

    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is False
    assert "text.bm25.bm25" in manifest["transform_priming_failed_features"]

    resumed = featm.extract(csv_path, **kwargs)
    assert resumed.report.failed_features == ["text.bm25.bm25"]
    assert list(resumed.to_pandas().columns) == ["tabular.numeric.abs"]
    assert np.allclose(resumed.to_pandas()["tabular.numeric.abs"].to_numpy(), df["x"].abs().to_numpy())


# ==========================================================================
# F. errors="raise" must remain unchanged: failure is re-raised, never
# converted into checkpointed skip/warn semantics.
# ==========================================================================
def test_errors_raise_unaffected_by_checkpoint_and_reraises_identically_on_retry(_pyarrow, tmp_path):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    kwargs = dict(_CATFREQ_KWARGS, errors="raise", checkpoint=str(ckpt))
    with pytest.raises(Exception) as exc1:
        featm.extract(df, **kwargs)
    with pytest.raises(Exception) as exc2:
        featm.extract(df, **kwargs)
    assert type(exc1.value) is type(exc2.value)
    assert "StateBudgetExceededError" in type(exc1.value).__name__


# ==========================================================================
# Manifest schema version: an old (pre-fix) checkpoint must be rejected,
# not silently resumed into the same history-loss state.
# ==========================================================================
def test_old_manifest_version_rejected_as_incompatible(_pyarrow, tmp_path):
    from featm.checkpoint import CheckpointManager, CheckpointIdentity

    mgr = CheckpointManager(tmp_path)
    identity = CheckpointIdentity(
        featm_version=featm.__version__, source_fingerprint="src-abc", schema_fingerprint="schema-abc",
        features=["f.g.h"], params={},
        settings={"batch_rows": 1, "mode": "exact", "errors": "skip", "columns": {"f.g.h": ["str:x"]},
                  "approximate_sample_size": 200_000, "state_budget_bytes": 16_384},
    )
    v1_manifest = identity.to_dict()
    v1_manifest.update({
        "manifest_version": 1,  # the OLD schema, no failure-history keys at all
        "phase": "done", "fit_batches_completed": 1, "fit_accumulator_file": None, "fit_state_files": {},
        "transform_batches_completed": 1, "transform_shards_written": 1, "done": True,
    })
    mgr.manifest_path.write_text(json.dumps(v1_manifest), encoding="utf-8")

    mgr2 = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointIncompatibleError, match="manifest_version|manifest schema"):
        mgr2.initialize_or_resume(identity, has_fit_phase=True)
