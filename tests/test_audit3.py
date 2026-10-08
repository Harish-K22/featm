"""Regression tests for the third independent audit / final closure pass.

Each section corresponds to one numbered finding in STEP2_AUDIT3_FIX_HANDOFF.md.
Tests that need a disk backend (Parquet, output="disk", checkpointing) request
the ``_pyarrow`` fixture; everything else must run in a base install.
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError
from featm.execution import UnsafeOutputModeError
from featm.matrix import CheckpointCorruptionError, DiskFeatureMatrixWriter
from featm.registry import Complexity
from featm.resources import ResourceBudget, ResourceBudgetTooSmallError, plan_batch_rows
from featm.sources import CSVSource


@pytest.fixture
def _pyarrow():
    return pytest.importorskip("pyarrow")


def _counting_source(source_cls, *args, **kwargs):
    src = source_cls(*args, **kwargs)
    original_iter = source_cls.iter_batches
    reads: list[int] = []

    def counting_iter(self, *, columns=None, batch_rows=50_000):
        for batch in original_iter(self, columns=columns, batch_rows=batch_rows):
            reads.append(len(batch))
            yield batch

    src.iter_batches = types.MethodType(counting_iter, src)
    return src, reads


# ==========================================================================
# Finding 1 -- checkpoint= + output="memory" produces a checkpoint that
# cannot resume
# ==========================================================================
def test_finding1_checkpoint_plus_output_memory_resumes_direct_feature(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(100.0)})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.numeric.abs"],
        columns={"tabular.numeric.abs": "x"},
        checkpoint=ckpt,
        output="memory",
        batch_rows=25,
        memory_budget="1MB",
    )
    first = featm.extract(X, **kwargs)
    assert first.shape == (100, 1)
    assert (ckpt / "output" / "manifest.json").exists()
    second = featm.extract(X, **kwargs)
    assert second.shape == (100, 1)
    assert second.report.resumed is True


def test_finding1_checkpoint_plus_output_memory_resumes_stateful_feature(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(200.0)})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.scaling.zscore"],
        columns={"tabular.scaling.zscore": "x"},
        checkpoint=ckpt,
        output="memory",
        batch_rows=40,
        memory_budget="1MB",
    )
    first = featm.extract(X, **kwargs)
    second = featm.extract(X, **kwargs)
    assert np.allclose(first.to_pandas().to_numpy(), second.to_pandas().to_numpy())


def test_finding1_second_call_is_zero_source_reads(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(150.0)})
    p = tmp_path / "d.csv"
    X.to_csv(p, index=False)
    ckpt = tmp_path / "ckpt"
    kwargs = dict(features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, checkpoint=ckpt, output="memory", batch_rows=30, memory_budget="1MB")
    featm.extract(p, **kwargs)
    src, reads = _counting_source(CSVSource, p)
    result = featm.extract(src, **kwargs)
    assert sum(reads) == 0
    assert result.shape == (150, 1)


# ==========================================================================
# Finding 2 -- checkpoint marked done before output finalized
# ==========================================================================
def test_finding2_crash_during_finalize_leaves_resumable_not_done_checkpoint(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(300.0)})
    ckpt = tmp_path / "ckpt"

    orig_finalize = DiskFeatureMatrixWriter.finalize

    def flaky_finalize(self, **kw):
        raise RuntimeError("SIMULATED CRASH during writer.finalize()")

    DiskFeatureMatrixWriter.finalize = flaky_finalize
    try:
        with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
            featm.extract(
                X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
                checkpoint=ckpt, output="disk", batch_rows=50, memory_budget="1MB",
            )
    finally:
        DiskFeatureMatrixWriter.finalize = orig_finalize

    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is False
    assert manifest["phase"] != "done"

    result = featm.extract(
        X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        checkpoint=ckpt, output="disk", batch_rows=50, memory_budget="1MB",
    )
    assert result.shape == (300, 1)
    clean = featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, output="memory", batch_rows=50, memory_budget="1MB")
    assert np.allclose(result.to_pandas().to_numpy(), clean.to_pandas().to_numpy())


# ==========================================================================
# Finding 3 -- checkpoint column/schema identity loses Python label type
# ==========================================================================
def test_finding3_int_vs_string_label_checkpoint_collision_rejected(_pyarrow, tmp_path):
    X = pd.DataFrame({0: [1, 2, 3], "0": [10, 20, 30]})
    ckpt = tmp_path / "ckpt"
    r1 = featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": 0}, checkpoint=ckpt, output="disk", batch_rows=2, memory_budget="1MB")
    assert list(r1.to_pandas().to_numpy().ravel()) == [1.0, 2.0, 3.0]
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "0"}, checkpoint=ckpt, output="disk", batch_rows=2, memory_budget="1MB")


def test_finding3_canonical_label_type_preserving():
    from featm.sources import canonical_label

    assert canonical_label(0) != canonical_label("0")
    assert canonical_label(0.0) != canonical_label(0)
    assert canonical_label(False) != canonical_label(0)
    assert canonical_label(None) == canonical_label(None)
    assert canonical_label((0, "a")) == canonical_label((0, "a"))
    assert canonical_label((0, "a")) != canonical_label(("0", "a"))
    with pytest.raises(TypeError):
        canonical_label(object())


# ==========================================================================
# Finding 4 -- public leakage-safe train/test fit/transform API
# ==========================================================================
def test_finding4_zscore_train_test_no_leakage():
    train = pd.DataFrame({"x": [0.0, 10.0]})
    test = pd.DataFrame({"x": [100.0, 110.0]})
    fitted = featm.fit(train, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, memory_budget="1MB")
    train_out = fitted.transform(train, output="memory").to_pandas().to_numpy().ravel()
    test_out = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.allclose(train_out, [-1.0, 1.0])
    assert np.allclose(test_out, [19.0, 21.0])


def test_finding4_minmax_train_test_no_leakage():
    train = pd.DataFrame({"x": [0.0, 100.0]})
    test = pd.DataFrame({"x": [50.0, -10.0]})
    fitted = featm.fit(train, features=["tabular.scaling.minmax"], columns={"tabular.scaling.minmax": "x"}, memory_budget="1MB")
    out = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.allclose(out, [0.5, -0.1])


def test_finding4_category_frequency_unseen_test_category_maps_to_zero():
    train = pd.DataFrame({"c": ["a", "a", "b", "b", "b", "c"]})
    test = pd.DataFrame({"c": ["a", "b", "c", "never_seen"]})
    fitted = featm.fit(train, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"}, memory_budget="1MB")
    out = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.isclose(out[0], 2 / 6)
    assert np.isclose(out[1], 3 / 6)
    assert np.isclose(out[2], 1 / 6)
    assert out[3] == 0.0


def test_finding4_transform_twice_is_stable_and_state_unmutated():
    train = pd.DataFrame({"c": ["a", "a", "b"]})
    test = pd.DataFrame({"c": ["a", "b", "unseen"]})
    fitted = featm.fit(train, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"}, memory_budget="1MB")
    out1 = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    out2 = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.array_equal(out1, out2)
    # transforming a THIRD, different source must not perturb the frozen state either
    fitted.transform(pd.DataFrame({"c": ["z"] * 500}), output="memory")
    out3 = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.array_equal(out1, out3)


def test_finding4_multi_batch_out_of_core_fit_and_transform(_pyarrow, tmp_path):
    train = pd.DataFrame({"x": np.arange(5000, dtype=float)})
    p = tmp_path / "train.csv"
    train.to_csv(p, index=False)
    fitted = featm.fit(p, features=["tabular.scaling.minmax"], columns={"tabular.scaling.minmax": "x"}, batch_rows=500, memory_budget="1MB")
    test = pd.DataFrame({"x": [0.0, 4999.0, -100.0, 6000.0]})
    out = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], 1.0)
    assert out[2] < 0
    assert out[3] > 1


def test_finding4_save_and_reload_fitted_artifact_matches(tmp_path):
    train = pd.DataFrame({"x": [0.0, 10.0]})
    test = pd.DataFrame({"x": [100.0, 110.0]})
    fitted = featm.fit(train, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, memory_budget="1MB")
    path = tmp_path / "artifact.pkl"
    fitted.save(path)
    reloaded = featm.load_fitted(path)
    out1 = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    out2 = reloaded.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.array_equal(out1, out2)


def test_finding4_direct_features_alongside_stateful_still_work():
    train = pd.DataFrame({"x": [0.0, 10.0], "y": [-5.0, 5.0]})
    test = pd.DataFrame({"x": [100.0, 110.0], "y": [1.0, -1.0]})
    fitted = featm.fit(
        train, features=["tabular.scaling.zscore", "tabular.numeric.abs"],
        columns={"tabular.scaling.zscore": "x", "tabular.numeric.abs": "y"},
        memory_budget="1MB",
    )
    out = fitted.transform(test, output="memory").to_pandas()
    assert out.shape == (2, 2)
    assert np.allclose(out["tabular.numeric.abs"].to_numpy(), [1.0, 1.0])


def test_finding4_transform_of_unfitted_stateful_feature_is_isolated_not_crash():
    """A feature selected for transform but never successfully fitted (e.g. too few train rows)
    must be isolated per errors=, not silently dropped or a hard crash."""
    from featm.execution import UnsafeMultiBatchFeatureError
    from featm.extractor import FittedFeatureExtractor

    fitted = FittedFeatureExtractor(
        features=["tabular.scaling.zscore"], modality=None,
        columns={"tabular.scaling.zscore": "x"}, params=None, mode="exact", errors="raise",
        memory_budget="1MB", disk_budget=None, cache_budget=None, approximate_sample_size=200_000,
        fitted_states={}, failed_stateful=set(), fit_report=None,
    )
    test = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(UnsafeMultiBatchFeatureError):
        fitted.transform(test, output="memory")

    fitted_warn = FittedFeatureExtractor(
        features=["tabular.scaling.zscore"], modality=None,
        columns={"tabular.scaling.zscore": "x"}, params=None, mode="exact", errors="skip",
        memory_budget="1MB", disk_budget=None, cache_budget=None, approximate_sample_size=200_000,
        fitted_states={}, failed_stateful=set(), fit_report=None,
    )
    result = fitted_warn.transform(test, output="memory")
    assert result.shape == (0, 0)


# ==========================================================================
# Finding 5 -- exact category-frequency fit state can grow beyond budget
# ==========================================================================
def test_finding5_high_cardinality_exact_fails_safely_under_tiny_budget():
    from featm.state import StateBudgetExceededError

    n = 5_000
    X = pd.DataFrame({"c": [f"unique_category_{i:06d}" for i in range(n)]})
    result = featm.extract(
        X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
        memory_budget="64KB", output="memory", batch_rows=100, mode="exact", errors="warn",
    )
    assert "tabular.encoding.category_frequency" in result.report.failed_features
    assert any("StateBudgetExceededError" in w for w in result.report.warnings)


def test_finding5_high_cardinality_exact_raises_under_errors_raise():
    from featm.state import StateBudgetExceededError

    n = 5_000
    X = pd.DataFrame({"c": [f"unique_category_{i:06d}" for i in range(n)]})
    with pytest.raises(StateBudgetExceededError):
        featm.extract(
            X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
            memory_budget="64KB", output="memory", batch_rows=100, mode="exact", errors="raise",
        )


def test_finding5_low_cardinality_exact_still_works_at_same_tiny_budget():
    n = 5_000
    X = pd.DataFrame({"c": [f"cat_{i % 3}" for i in range(n)]})
    result = featm.extract(
        X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
        memory_budget="64KB", output="memory", batch_rows=100, mode="exact", errors="raise",
    )
    assert result.shape == (n, 1)


# ==========================================================================
# Finding 6 -- approximate reservoir capacity ignores memory_budget
# ==========================================================================
def test_finding6_reservoir_capacity_clamped_to_state_budget():
    n = 5_000
    X = pd.DataFrame({"c": [f"cat_{i % 5000}" for i in range(n)]})
    result = featm.extract(
        X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
        memory_budget="64KB", output="memory", batch_rows=100, mode="approximate",
    )
    info = result.report.approximations_used["tabular.encoding.category_frequency"]
    assert info["sample_size_capacity_requested"] == 200_000
    assert info["sample_size_capacity_effective"] < 200_000
    assert info["sample_size"] <= info["sample_size_capacity_effective"]


def test_finding6_reservoir_capacity_not_clamped_when_budget_is_generous():
    n = 2_000
    X = pd.DataFrame({"c": [f"cat_{i % 50}" for i in range(n)]})
    result = featm.extract(
        X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
        memory_budget="1GB", output="memory", batch_rows=200, mode="approximate", approximate_sample_size=1000,
    )
    info = result.report.approximations_used["tabular.encoding.category_frequency"]
    assert info["sample_size_capacity_effective"] == 1000


# ==========================================================================
# Finding 7 -- batch planner hard floor can exceed memory budget
# ==========================================================================
def test_finding7_planner_never_returns_more_rows_than_budget_affords():
    budget = ResourceBudget.resolve(memory_budget="64KB")
    spec = featm.feature_info("timeseries.statistical.mean")
    rows = plan_batch_rows(schema_n_columns=100, specs=[spec], budget=budget)
    est = (100 + 1) * 8 * 4 * rows
    assert est <= budget.memory_bytes


def test_finding7_impossibly_tiny_budget_raises_actionable_error():
    budget = ResourceBudget.resolve(memory_budget="10B")
    spec = featm.feature_info("timeseries.statistical.mean")
    with pytest.raises(ResourceBudgetTooSmallError):
        plan_batch_rows(schema_n_columns=100, specs=[spec], budget=budget)


def test_finding7_explicit_batch_rows_override_still_honored_exactly():
    budget = ResourceBudget.resolve(memory_budget="10B")
    spec = featm.feature_info("timeseries.statistical.mean")
    rows = plan_batch_rows(schema_n_columns=100, specs=[spec], budget=budget, override=5000)
    assert rows == 5000


def test_finding7_generous_budget_still_uses_the_throughput_preferred_min_rows():
    budget = ResourceBudget.resolve(memory_budget="1GB")
    spec = featm.feature_info("timeseries.statistical.mean")
    rows = plan_batch_rows(schema_n_columns=100, specs=[spec], budget=budget)
    assert rows >= 256  # unaffected for a budget that comfortably affords the preferred floor


# ==========================================================================
# Finding 8 -- memory-budget semantics for explicit output="memory"
# ==========================================================================
def test_finding8_explicit_output_memory_respects_budget_by_default():
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float)})
    with pytest.raises(UnsafeOutputModeError):
        featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, memory_budget="1KB", output="memory", batch_rows=1000)


def test_finding8_explicit_output_memory_override_flag_accepts_the_risk():
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float)})
    result = featm.extract(
        X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, memory_budget="1KB",
        output="memory", batch_rows=1000, allow_memory_budget_override=True,
    )
    assert result.shape == (100_000, 1)
    assert any("override" in w for w in result.report.warnings)


def test_finding8_output_auto_still_switches_to_disk_instead_of_raising(_pyarrow):
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float)})
    result = featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, memory_budget="1KB", output="auto", batch_rows=1000)
    assert result.report.output_backend == "disk"
    assert result.shape == (100_000, 1)


def test_finding8_output_disk_unaffected_by_memory_budget(_pyarrow):
    X = pd.DataFrame({"x": np.arange(50_000, dtype=float)})
    result = featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, memory_budget="1KB", output="disk", batch_rows=1000)
    assert result.shape == (50_000, 1)


# ==========================================================================
# Finding 9 -- point-cloud Delaunay mis-curated as quadratic
# ==========================================================================
def test_finding9_pointcloud_delaunay_curated_linearithmic():
    spec = featm.feature_info("pointcloud.triangulation.delaunay")
    assert spec.complexity == Complexity.LINEARITHMIC


def test_finding9_pointcloud_delaunay_not_wrongly_rejected_at_realistic_scale():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20_000, 3)), columns=["x", "y", "z"])
    result = featm.extract(
        X, features=["pointcloud.triangulation.delaunay"], columns={"pointcloud.triangulation.delaunay": ["x", "y", "z"]},
        memory_budget="500MB", output="memory", errors="raise",
    )
    assert "pointcloud.triangulation.delaunay" in result.report.completed_features


def test_finding9_no_delaunay_voronoi_registration_drifts_into_conflicting_complexity():
    """Every Delaunay/Voronoi(-tagged) registration must agree on LINEARITHMIC complexity;
    catches exactly the class of drift this finding was about, for any future registration too."""
    specs = featm.REGISTRY.all()
    delaunay_voronoi = [s for s in specs if "delaunay" in s.tags or "voronoi" in s.tags]
    assert len(delaunay_voronoi) >= 5  # sanity: the families we know about are actually present
    mismatched = [s.name for s in delaunay_voronoi if s.complexity != Complexity.LINEARITHMIC]
    assert mismatched == []


def test_finding9_genuinely_quadratic_point_pattern_features_not_weakened():
    ripley = featm.feature_info("spatial.point_patterns.ripley_k")
    assert ripley.complexity == Complexity.QUADRATIC
    assert ripley.memory_estimator is not None


# ==========================================================================
# Finding 10 -- streamable/batchable semantics and the text whole-batch bug
# ==========================================================================
def test_finding10_text_modality_batches_correctly_out_of_core():
    n = 100
    tdf = pd.DataFrame({"t": [f"hello world number {i} more words here" for i in range(n)]})
    result = featm.extract(
        tdf, features=["text.lexical.avg_word_length"], memory_budget="50MB", output="memory", batch_rows=17, errors="raise",
    )
    assert result.shape == (n, 1)


def test_finding10_multi_nonnumeric_column_whole_batch_feature_raises_clearly():
    from featm.execution import ColumnResolutionError

    tdf = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    with pytest.raises(ColumnResolutionError):
        featm.extract(tdf, features=["text.lexical.avg_word_length"], memory_budget="50MB", output="memory", batch_rows=2, errors="raise")


def test_finding10_graph_feature_from_pandas_source_now_works_correctly():
    """A PandasSource with a genuine pre-loaded object column is the one case
    graph.* out-of-core is meant to work (see docs/SCALING.md's modality-support
    table) -- the Finding 10 fix (Series, not DataFrame, for a single non-numeric
    column) makes this actually succeed, not merely fail less confusingly."""
    pytest.importorskip("networkx")
    import networkx as nx

    graphs = [nx.erdos_renyi_graph(5, 0.5, seed=i) for i in range(20)]
    gdf = pd.DataFrame({"g": graphs})
    result = featm.extract(gdf, features=["graph.size.n_nodes"], columns={"graph.size.n_nodes": ["g"]}, memory_budget="50MB", output="memory", batch_rows=5, errors="raise")
    assert result.shape == (20, 1)
    expected = [g.number_of_nodes() for g in graphs]
    assert list(result.to_pandas().to_numpy().ravel()) == expected


def test_finding10_graph_feature_from_csv_fails_clearly_not_silently_wrong(_pyarrow, tmp_path):
    """A CSV cannot hold real Graph objects -- each row's value is a plain string.

    Third audit pass: this was isolated at COMPUTE time (the implementation's
    ``g.number_of_nodes()`` call raised on a plain str, caught and recorded as
    a per-feature failure) -- still "fail clearly, never silently wrong", but
    only discovered after a batch was actually read and the feature was
    actually invoked. Fifth audit pass, Finding 1: this exact CSV/graph
    mismatch is now the general OPAQUE_OBJECT_SAMPLE-vs-non-PandasSource
    guard in ``_prepare_run`` -- rejected PROACTIVELY, before any batch is
    read or the feature is invoked at all (``report.skipped_features``, not
    ``failed_features``), which is a strictly earlier and clearer rejection
    of the same underlying problem, not a weakened one. Updated per the
    audit's own instruction to explain exactly why a pre-existing test's
    assertion changed rather than leaving it silently adjusted."""
    csv_path = tmp_path / "g.csv"
    pd.DataFrame({"g": ["not_a_graph"] * 10}).to_csv(csv_path, index=False)
    result = featm.extract(csv_path, features=["graph.size.n_nodes"], columns={"graph.size.n_nodes": ["g"]}, memory_budget="50MB", output="memory", batch_rows=5, errors="warn")
    assert "graph.size.n_nodes" in result.report.skipped_features
    assert result.shape == (0, 0)


def test_finding10_streamable_field_not_consulted_by_out_of_core_engine():
    """streamable is a legacy tabular-only field; a non-tabular row-independent
    modality (timeseries) registered streamable=False must still batch correctly
    out-of-core -- the engine does not gate on it at all."""
    spec = featm.feature_info("timeseries.autocorrelation.acf")
    assert spec.streamable is False
    n, T = 2000, 16
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, T)), columns=[f"t{i}" for i in range(T)])
    result = featm.extract(X, features=["timeseries.autocorrelation.acf"], memory_budget="10MB", output="memory", batch_rows=100, errors="raise")
    assert result.shape[0] == n


# ==========================================================================
# Finding 11 -- completed-checkpoint integrity is structural, not content
# ==========================================================================
def test_finding11_missing_shard_still_detected(_pyarrow, tmp_path):
    X = pd.DataFrame({"price": np.arange(3000, dtype=float)})
    p = tmp_path / "d.csv"
    X.to_csv(p, index=False)
    ckpt = tmp_path / "ckpt"
    featm.extract(p, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)
    out_dir = ckpt / "output"
    shard = next(f for f in out_dir.iterdir() if f.name.startswith("part-"))
    shard.unlink()
    with pytest.raises(CheckpointCorruptionError):
        featm.extract(p, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)


def test_finding11_same_row_count_replaced_shard_detected_via_checksum(_pyarrow, tmp_path):
    X = pd.DataFrame({"price": np.arange(3000, dtype=float)})
    p = tmp_path / "d.csv"
    X.to_csv(p, index=False)
    ckpt = tmp_path / "ckpt"
    featm.extract(p, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)
    out_dir = ckpt / "output"
    shard_name = next(f for f in out_dir.iterdir() if f.name.startswith("part-")).name
    shard_path = out_dir / shard_name
    replacement = pd.DataFrame({"tabular.numeric.abs__000": np.zeros(500, dtype=float)})
    replacement.to_parquet(shard_path, index=False)
    with pytest.raises(CheckpointCorruptionError):
        featm.extract(p, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)


def test_finding11_manifest_records_shard_sizes_and_checksums(_pyarrow, tmp_path):
    X = pd.DataFrame({"price": np.arange(300.0)})
    p = tmp_path / "d.csv"
    X.to_csv(p, index=False)
    ckpt = tmp_path / "ckpt"
    featm.extract(p, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=100, output="disk", checkpoint=ckpt)
    manifest = json.loads((ckpt / "output" / "manifest.json").read_text())
    assert len(manifest["shard_byte_sizes"]) == len(manifest["shards"])
    assert len(manifest["shard_checksums"]) == len(manifest["shards"])
    assert all(isinstance(c, str) and len(c) == 32 for c in manifest["shard_checksums"])


def test_finding11_intact_checkpoint_still_zero_source_reads(_pyarrow, tmp_path):
    X = pd.DataFrame({"price": np.arange(3000, dtype=float)})
    p = tmp_path / "d.csv"
    X.to_csv(p, index=False)
    ckpt = tmp_path / "ckpt"
    featm.extract(p, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)
    src, reads = _counting_source(CSVSource, p)
    result = featm.extract(src, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)
    assert sum(reads) == 0
    assert result.shape == (3000, 1)


# ==========================================================================
# Finding 12 -- documentation inconsistencies
# ==========================================================================
def test_finding12_approx_note_does_not_claim_full_scan_is_skipped():
    from featm.state import _APPROX_NOTE

    assert "instead of scanning every row" not in _APPROX_NOTE
    assert "every fit row is still read/streamed" in _APPROX_NOTE.lower() or "every row" in _APPROX_NOTE.lower()


def test_finding12_sampling_module_docstring_does_not_claim_full_scan_is_skipped():
    import featm.sampling as sampling_mod

    assert "instead of scanning every row" not in (sampling_mod.__doc__ or "")


# ==========================================================================
# Additional audit: original Step-2 acceptance re-check (spot checks, not
# exhaustive -- full narrative is in STEP2_AUDIT3_FIX_HANDOFF.md)
# ==========================================================================
def test_acceptance_exact_mode_never_approximates_even_under_state_pressure():
    """mode='exact' must fail, never silently approximate, when state can't fit."""
    n = 5_000
    X = pd.DataFrame({"c": [f"unique_{i:06d}" for i in range(n)]})
    result = featm.extract(
        X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
        memory_budget="64KB", output="memory", batch_rows=100, mode="exact", errors="warn",
    )
    assert result.report.approximations_used == {}


def test_acceptance_cache_still_genuinely_per_batch(_pyarrow, tmp_path):
    n = 3000
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 16)), columns=[f"c{i}" for i in range(16)])
    p = tmp_path / "d.parquet"
    X.to_parquet(p, index=False)
    result = featm.extract(p, features=["timeseries.spectral.centroid"], batch_rows=500, output="memory")
    assert result.report.n_batches > 1
    assert result.report.cache_stats["misses"] == result.report.n_batches


def test_acceptance_registry_count_and_v01_names_unaffected():
    specs = featm.REGISTRY.all()
    assert len(specs) == 552
