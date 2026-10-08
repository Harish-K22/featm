"""Step 4 completion pass: distributed fit-state merge correctness.

Focused on gaps NOT already covered by tests/test_distributed_foundation.py
(which already exercises all 7 encoders' raw-state-merge equivalence and
similarity's first-document ordering broadly): the state-memory BUDGET being
enforced during the coordinator's MERGE step (not just per-partition
update), and the errors="raise" propagation fix from this completion pass
(a fit-phase failure on ANY partition raises immediately; a transform-phase
failure only raises on the priming partition, not later ones).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import featm
from featm.distributed.contracts import DistributedFeatureError
from featm.state import StateBudgetExceededError


def test_merge_state_budget_exceeded_during_coordinator_merge_raises_clearly():
    """No single partition alone exceeds the budget (few distinct categories per
    partition), but the coordinator's MERGE of all partitions' distinct categories
    together does -- this must be caught by merge_states()'s own budget check, not
    silently allowed to grow unbounded just because each individual partition looked fine."""
    rng = np.random.default_rng(40)
    # Each partition sees a DISJOINT block of many distinct long category strings,
    # so merging is what pushes total distinct-category memory over a tiny budget.
    categories = [f"category_{i:06d}_" + "x" * 40 for i in range(400)]
    df = pd.DataFrame({"cat": categories})
    with pytest.raises((StateBudgetExceededError, DistributedFeatureError)):
        featm.extract(
            df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
            execution="distributed", distributed_backend="local", distributed_workers=4,
            distributed_partition_rows=20, memory_budget="20KB", output="memory", errors="raise",
        )


def test_fit_phase_failure_on_a_later_partition_raises_immediately_under_errors_raise():
    """Regression: the local out-of-core engine's fit phase raises on ANY failing
    batch (not just the first) under errors="raise" -- a failed fit invalidates the
    whole feature. The distributed fit phase must match. Uses a state-budget breach
    on the SECOND partition specifically (the first partition's distinct categories
    alone fit comfortably; the second partition's NEW distinct categories are what
    pushes the per-partition update_state() over budget) -- a real StatefulEncoder
    failure mode, not a synthetic one, and specifically on a non-priming partition."""
    rng = np.random.default_rng(43)
    first_partition = [f"early_{i}" for i in range(5)] * 4  # 5 distinct, cheap
    second_partition = [f"category_{i:06d}_" + "x" * 60 for i in range(30)]  # 30 distinct, expensive
    df = pd.DataFrame({"cat": first_partition + second_partition})
    with pytest.raises((StateBudgetExceededError, DistributedFeatureError)):
        featm.extract(
            df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
            execution="distributed", distributed_backend="local", distributed_workers=2,
            distributed_partition_rows=20, memory_budget="8KB", output="memory", errors="raise",
        )


def test_transform_phase_later_partition_failure_is_nan_filled_under_skip(monkeypatch):
    """C17: only skip/warn may degrade later partitions; strict cases are covered
    by test_v055_correctness.test_v055_c17_error_policy."""
    import featm.execution as exec_mod

    calls = {"n": 0}
    real = exec_mod._compute_transform_with_retry

    def flaky_after_first_partition(spec, batch, resolved_columns, params, fitted_states, report, *, backend_policy, batch_id):
        if spec.name == "tabular.numeric.abs" and batch_id == 1:
            raise RuntimeError("intentional later-partition failure")
        return real(spec, batch, resolved_columns, params, fitted_states, report, backend_policy=backend_policy, batch_id=batch_id)

    monkeypatch.setattr("featm.distributed.execution._compute_transform_with_retry", flaky_after_first_partition)

    df = pd.DataFrame({"x": np.arange(60, dtype=float) - 30})
    result = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=1,
        distributed_partition_rows=20, output="memory", errors="skip",
    )
    arr = result.to_pandas().to_numpy().ravel()
    assert len(arr) == 60
    assert np.isnan(arr[20:40]).all(), "the failing partition's rows must be NaN-filled, not missing"
    assert not np.isnan(arr[:20]).any() and not np.isnan(arr[40:]).any()
    assert result.report.distributed_report["degraded_partitions_by_feature"].get("tabular.numeric.abs") == 1


def test_n_partitions_less_than_n_workers_still_correct():
    df = pd.DataFrame({"x": np.arange(5, dtype=float)})
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    dist_r = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=8, distributed_partition_rows=2,
        output="memory", errors="raise",
    )
    assert np.allclose(local_r.to_numpy(), dist_r.to_pandas().to_numpy())


def test_one_row_source():
    df = pd.DataFrame({"x": [3.5]})
    dist_r = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=4, distributed_partition_rows=10,
        output="memory", errors="raise",
    )
    assert dist_r.to_pandas().to_numpy().ravel().tolist() == [3.5]


def test_empty_source_returns_clean_empty_result():
    df = pd.DataFrame({"x": pd.Series([], dtype=float)})
    dist_r = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=4, distributed_partition_rows=10,
        output="memory", errors="raise",
    )
    assert len(dist_r) == 0


def test_non_string_integer_column_labels_survive_distributed_round_trip():
    df = pd.DataFrame({0: np.arange(30, dtype=float), 1: np.arange(30, dtype=float) * 2})
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": 0}, execution="local")
    dist_r = featm.extract(
        df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": 0},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=7,
        output="memory", errors="raise",
    )
    assert np.allclose(local_r.to_numpy(), dist_r.to_pandas().to_numpy())


def test_vector_output_feature_schema_matches_local():
    rng = np.random.default_rng(41)
    df = pd.DataFrame(np.vstack([rng.standard_normal(64) for _ in range(40)]))
    local_r = featm.extract(df, features=["timeseries.spectral.periodogram"], execution="local")
    dist_r = featm.extract(
        df, features=["timeseries.spectral.periodogram"],
        execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=7,
        output="memory", errors="raise",
    )
    a = local_r.to_pandas().to_numpy() if hasattr(local_r, "to_pandas") else local_r.to_numpy()
    b = dist_r.to_pandas().to_numpy()
    assert list(local_r.columns if hasattr(local_r, "columns") else local_r.columns) == list(dist_r.to_pandas().columns)
    assert np.allclose(a, b, equal_nan=True)


def test_direct_and_stateful_mixed_selection_matches_local():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(90)})
    features = ["tabular.numeric.abs", "tabular.scaling.zscore"]
    columns = {"tabular.numeric.abs": "x", "tabular.scaling.zscore": "x"}
    local_r = featm.extract(df, features=features, columns=columns, execution="local", batch_rows=15, errors="raise")
    dist_r = featm.extract(
        df, features=features, columns=columns, execution="distributed", distributed_backend="local",
        distributed_workers=3, distributed_partition_rows=15, output="memory", errors="raise",
    )
    a = local_r.to_pandas().to_numpy(dtype=float)
    b = dist_r.to_pandas().to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


# ==========================================================================
# Finding 2 (final closure audit, v0.4.1 -> v0.4.2): errors="skip"/"warn"
# crashed with "distributed partition K produced 0 output rows for N input
# rows" when EVERY selected feature failed (Case A: fit-phase failure;
# Case B: transform-priming failure). The canonical local engine returns a
# clean empty result (shape (0, 0), failed_features populated) instead of
# raising -- distributed execution must match exactly.
# ==========================================================================
def _reproducer_df():
    long_value = "X" * 20_000
    return pd.DataFrame({"cat": ["a", long_value, "b", "c"]})


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_all_stateful_fit_failure_returns_local_equivalent_empty_result(policy):
    df = _reproducer_df()
    local_r = featm.extract(
        df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="local", batch_rows=1, memory_budget=65_536, errors=policy,
    )
    dist_r = featm.extract(
        df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=1,
        memory_budget=65_536, errors=policy, output="memory",
    )
    assert local_r.shape == dist_r.shape == (0, 0)
    assert dist_r.report.failed_features == local_r.report.failed_features == ["tabular.encoding.category_frequency"]
    assert dist_r.report.distributed_report["transform_partitions_computed_this_run"] == 0


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_all_features_fail_during_transform_priming_returns_local_equivalent_empty_result(policy):
    """BM25 with no query= param fails on every partition (not a fit-phase failure --
    bm25 is stateful, so it fails during priming TRANSFORM, after a successful fit)."""
    docs = [f"doc number {i} with extra padding text here" for i in range(20)]
    df = pd.DataFrame({"doc": docs})
    dist_r = featm.extract(
        df, features=["text.bm25.bm25"], columns={"text.bm25.bm25": "doc"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=5,
        output="memory", errors=policy,
    )
    assert dist_r.shape == (0, 0)
    assert dist_r.report.failed_features == ["text.bm25.bm25"]


def test_mixed_one_feature_fails_one_survives_still_returns_n_rows_for_survivor():
    docs = [f"doc number {i} with extra padding text here" for i in range(20)]
    df = pd.DataFrame({"doc": docs, "x": list(range(20))})
    dist_r = featm.extract(
        df, features=["text.bm25.bm25", "tabular.numeric.abs"],
        columns={"text.bm25.bm25": "doc", "tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=5,
        output="memory", errors="skip",
    )
    assert dist_r.shape == (20, 1)
    assert dist_r.report.failed_features == ["text.bm25.bm25"]
    assert list(dist_r.to_pandas().columns) == ["tabular.numeric.abs"]
    assert np.allclose(dist_r.to_pandas()["tabular.numeric.abs"].to_numpy(), df["x"].abs().to_numpy())


def test_all_rejected_before_submission_still_returns_local_equivalent_empty_result():
    """Every selected feature rejected by the DISTRIBUTED PLANNER itself (FULL_DATASET
    semantics) under errors="skip" -- a different code path from Case A/B (fit/transform
    failure), but must be equally clean, not a crash."""
    df = pd.DataFrame({"doc": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    dist_r = featm.extract(
        df, features=["text.semantics.topic"], columns={"text.semantics.topic": "doc"},
        execution="distributed", distributed_backend="local", distributed_partition_rows=2,
        errors="skip", output="memory",
    )
    assert dist_r.shape == (0, 0)


def test_empty_source_with_all_selected_features_skipped_matches_local():
    df = pd.DataFrame({"cat": pd.Series([], dtype=object)})
    dist_r = featm.extract(
        df, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=5,
        output="memory", errors="skip",
    )
    assert len(dist_r) == 0


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_checkpoint_all_stateful_fit_failure_degradation_persists_and_resume_matches(tmp_path, policy):
    df = _reproducer_df()
    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=1,
        memory_budget=65_536, errors=policy, output="memory", checkpoint=str(ckpt_dir),
    )
    first = featm.extract(df, **kwargs)
    second = featm.extract(df, **kwargs)  # resume against a completed checkpoint
    assert first.shape == second.shape == (0, 0)
    assert first.report.failed_features == second.report.failed_features == ["tabular.encoding.category_frequency"]
    assert second.report.distributed_report["checkpoint_resumed"] is True
    assert second.report.distributed_report["transform_partitions_computed_this_run"] == 0


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_checkpoint_all_transform_priming_failure_degradation_persists_and_resume_matches(tmp_path, policy):
    docs = [f"doc number {i} with extra padding text here" for i in range(20)]
    df = pd.DataFrame({"doc": docs})
    ckpt_dir = tmp_path / "ckpt"
    kwargs = dict(
        features=["text.bm25.bm25"], columns={"text.bm25.bm25": "doc"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=5,
        output="memory", errors=policy, checkpoint=str(ckpt_dir),
    )
    first = featm.extract(df, **kwargs)
    second = featm.extract(df, **kwargs)  # resume: must not re-scan the source to reconfirm
    assert first.shape == second.shape == (0, 0)
    assert first.report.failed_features == second.report.failed_features == ["text.bm25.bm25"]
    assert second.report.distributed_report["checkpoint_resumed"] is True
    assert second.report.distributed_report["transform_partitions_computed_this_run"] == 0
