"""Regression tests for the fourth independent audit / batch-semantics closure pass.

Each section corresponds to one numbered finding in STEP2_AUDIT4_FIX_HANDOFF.md.
Tests that need a disk backend (Parquet, output="disk", checkpointing) request
the ``_pyarrow`` fixture; everything else must run in a base install.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import featm
from featm.execution import (
    ColumnResolutionError,
    FeatureRowAlignmentError,
    FeatureSchemaDriftError,
    UnsafeMultiBatchFeatureError,
    UnsafeOutputModeError,
)
from featm.registry import BatchSemantics, FeatureSpec, REGISTRY
from featm.resources import ResourceBudgetTooSmallError, plan_batch_rows, ResourceBudget
from featm.sampling import ReservoirStateBudgetExceededError
from featm.state import StateBudgetExceededError


@pytest.fixture
def _pyarrow():
    return pytest.importorskip("pyarrow")


@pytest.fixture
def register_temp_feature():
    registered: list[str] = []

    def _register(spec: FeatureSpec) -> FeatureSpec:
        REGISTRY.register(spec)
        registered.append(spec.name)
        return spec

    yield _register
    for name in registered:
        REGISTRY._specs.pop(name, None)


TEXT_DOCS = [
    "apple apple red fruit",
    "apple green fruit",
    "banana yellow fruit",
    "banana banana tropical",
    "car engine road speed",
    "car wheel road road",
]


def _cmp_full_vs_batched(feature, *, params=None, source=None, columns=None):
    src = source if source is not None else pd.DataFrame({"text": TEXT_DOCS})
    n = len(src)
    full = featm.extract(src, features=[feature], params=params, columns=columns, batch_rows=n, output="memory", errors="warn")
    batched = featm.extract(src, features=[feature], params=params, columns=columns, batch_rows=2, output="memory", errors="warn")
    return full, batched


# ==========================================================================
# Finding 1 -- corpus/global text features are silently wrong across batches
# ==========================================================================
def test_finding1a_surprisal_batch_equivalent():
    full, batched = _cmp_full_vs_batched("text.information.surprisal")
    assert full.shape == batched.shape == (6, 1)
    assert np.allclose(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy())
    assert batched.report.failed_features == []


def test_finding1b_bm25_batch_equivalent():
    p = {"text.bm25.bm25": {"query": "fruit apple"}}
    full, batched = _cmp_full_vs_batched("text.bm25.bm25", params=p)
    assert np.allclose(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy())


def test_finding1c_tfidf_batch_equivalent_shape_and_values():
    p = {"text.tfidf.tfidf": {"max_features": 20}}
    full, batched = _cmp_full_vs_batched("text.tfidf.tfidf", params=p)
    assert full.shape == batched.shape
    assert not batched.to_pandas().isna().any().any()
    assert np.allclose(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy())


def test_finding1d_similarity_batch_equivalent_no_per_batch_reference():
    full, batched = _cmp_full_vs_batched("text.semantics.similarity")
    vals = batched.to_pandas().to_numpy().ravel()
    # Only the true first document (row 0) should score 1.0 -- rows 2 and 4
    # must NOT also score 1.0 (the pre-fix per-batch-local-reference bug).
    assert vals[0] == pytest.approx(1.0)
    assert vals[2] != pytest.approx(1.0)
    assert vals[4] != pytest.approx(1.0)
    assert np.allclose(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy())


def test_finding1e_topic_rejected_not_silently_wrong():
    df = pd.DataFrame({"text": TEXT_DOCS})
    result = featm.extract(
        df, features=["text.semantics.topic"], params={"text.semantics.topic": {"n_topics": 2}},
        batch_rows=2, output="memory", errors="warn",
    )
    assert result.shape == (0, 0)
    assert "text.semantics.topic" in result.report.skipped_features
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(
            df, features=["text.semantics.topic"], params={"text.semantics.topic": {"n_topics": 2}},
            batch_rows=2, output="memory", errors="raise",
        )


def test_finding1_corpus_features_integrate_with_public_fit_api():
    train = pd.DataFrame({"text": TEXT_DOCS})
    test = pd.DataFrame({"text": ["apple banana fruit", "car road"]})
    fitted = featm.fit(train, features=["text.tfidf.tfidf"], params={"text.tfidf.tfidf": {"max_features": 10}}, memory_budget="10MB")
    out1 = fitted.transform(test, output="memory")
    out2 = fitted.transform(test, output="memory")
    assert np.array_equal(out1.to_pandas().to_numpy(), out2.to_pandas().to_numpy())
    assert out1.shape[0] == 2


# ==========================================================================
# Finding 2 -- sequence features reset their implicit reference per batch
# ==========================================================================
SEQ = ["aaaa", "aaab", "bbbb", "bbba", "cccc", "ccca"]


def test_finding2_edit_distance_rejected_without_explicit_reference():
    df = pd.DataFrame({"s": SEQ})
    result = featm.extract(df, features=["sequence.distance.edit_distance"], batch_rows=2, output="memory", errors="warn")
    assert result.shape == (0, 0)
    assert "sequence.distance.edit_distance" in result.report.skipped_features
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(df, features=["sequence.distance.edit_distance"], batch_rows=2, output="memory", errors="raise")


def test_finding2_edit_distance_batch_equivalent_with_explicit_reference():
    df = pd.DataFrame({"s": SEQ})
    p = {"sequence.distance.edit_distance": {"reference": "aaaa"}}
    full = featm.extract(df, features=["sequence.distance.edit_distance"], params=p, batch_rows=6, output="memory")
    batched = featm.extract(df, features=["sequence.distance.edit_distance"], params=p, batch_rows=2, output="memory")
    assert np.array_equal(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy())


def test_finding2_alignment_rejected_without_explicit_reference():
    df = pd.DataFrame({"s": SEQ})
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(df, features=["sequence.alignment.alignment"], batch_rows=2, output="memory", errors="raise")


def test_finding2_single_batch_still_uses_convenient_default():
    """A run that fits in one batch is unaffected -- matches small-data default behavior exactly."""
    df = pd.DataFrame({"s": SEQ})
    result = featm.extract(df, features=["sequence.distance.edit_distance"], batch_rows=100, output="memory", errors="raise")
    assert result.shape == (6, 1)


# ==========================================================================
# Finding 3 -- event recency uses a batch-local global reference time
# ==========================================================================
EVENTS = [
    np.array([1, 2, 3.0]),
    np.array([2, 4, 10.0]),
    np.array([5, 6, 7.0]),
    np.array([1, 20, 21.0]),
]


def test_finding3_recency_rejected_without_explicit_reference_time():
    df = pd.DataFrame({"e": EVENTS})
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(df, features=["event.timing.recency"], batch_rows=2, output="memory", errors="raise")


def test_finding3_recency_batch_equivalent_with_explicit_reference_time():
    df = pd.DataFrame({"e": EVENTS})
    p = {"event.timing.recency": {"reference_time": 21.0}}
    full = featm.extract(df, features=["event.timing.recency"], params=p, batch_rows=4, output="memory")
    batched = featm.extract(df, features=["event.timing.recency"], params=p, batch_rows=2, output="memory")
    assert np.array_equal(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy())


# ==========================================================================
# Finding 4 -- structured cross-series / pair / triple / multimodal inputs
# ==========================================================================
def test_finding4_cross_correlation_structured_input_rejected():
    X = pd.DataFrame({"a": np.arange(8, dtype=float), "b": np.arange(8, dtype=float)[::-1]})
    result = featm.extract(
        X, features=["timeseries.cross_series.cross_correlation"],
        memory_budget="1MB", batch_rows=2, output="memory", errors="skip",
    )
    assert result.shape[0] == 0
    assert "timeseries.cross_series.cross_correlation" in result.report.skipped_features
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(
            X, features=["timeseries.cross_series.cross_correlation"],
            memory_budget="1MB", batch_rows=2, output="memory", errors="raise",
        )


def test_finding4_cross_correlation_still_works_in_memory_small_data():
    a = np.sin(np.linspace(0, 10, 50))
    b = np.cos(np.linspace(0, 10, 50))
    result = featm.extract((a, b), features=["timeseries.cross_series.cross_correlation"], errors="raise")
    assert result.shape[0] == 1


@pytest.mark.parametrize("feature", [
    "information.dependence.mutual_information",
    "information.divergence.kl",
    "multimodal.similarity.cross_modal",
])
def test_finding4_structured_sample_features_rejected_out_of_core(feature):
    spec = featm.feature_info(feature)
    assert spec.batch_semantics == BatchSemantics.STRUCTURED_SAMPLE
    X = pd.DataFrame({"a": np.arange(8, dtype=float), "b": np.arange(8, dtype=float)[::-1]})
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(X, features=[feature], memory_budget="1MB", batch_rows=2, output="memory", errors="raise")


def test_finding4_structured_sample_count_and_no_row_independent_regressions():
    structured = [s for s in REGISTRY.all() if s.batch_semantics == BatchSemantics.STRUCTURED_SAMPLE]
    names = {s.name for s in structured}
    assert len(structured) == 31
    for expected_family in (
        "information.dependence.mutual_information", "information.divergence.kl",
        "multimodal.similarity.cross_modal", "timeseries.cross_series.cross_correlation",
        "event.survival.survival", "signal.synchronization.phase_locking",
    ):
        assert expected_family in names


# ==========================================================================
# Finding 5 -- generic row-alignment invariant
# ==========================================================================
def test_finding5_row_alignment_violation_raises(register_temp_feature):
    def bad(x, **kw):
        return np.zeros(max(len(np.asarray(x)) - 1, 0))

    register_temp_feature(FeatureSpec(name="audit4.f5.bad_alignment", family="f5", modality="information", input_types=("distribution_or_samples",), implementation=bad))
    X = pd.DataFrame(np.random.default_rng(0).random((10, 4)))
    result = featm.extract(X, features=["audit4.f5.bad_alignment"], batch_rows=5, output="memory", errors="warn")
    assert "audit4.f5.bad_alignment" in result.report.failed_features
    with pytest.raises(FeatureRowAlignmentError):
        featm.extract(X, features=["audit4.f5.bad_alignment"], batch_rows=5, output="memory", errors="raise")


def test_finding5_inconsistent_column_row_counts_raises(register_temp_feature):
    def bad(x, **kw):
        n = len(np.asarray(x))
        return {"a": np.zeros(n), "b": np.zeros(max(n - 2, 0))}

    register_temp_feature(FeatureSpec(name="audit4.f5.bad_cols", family="f5", modality="information", input_types=("distribution_or_samples",), output_kind="vector", implementation=bad))
    X = pd.DataFrame(np.random.default_rng(0).random((10, 4)))
    with pytest.raises(FeatureRowAlignmentError):
        featm.extract(X, features=["audit4.f5.bad_cols"], batch_rows=5, output="memory", errors="raise")


def test_finding5_full_dataset_feature_legitimately_not_row_aligned():
    """A FULL_DATASET feature's single grown-to-whole-source batch may summarize
    many rows into fewer output rows -- this must NOT trip the row-alignment check."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["x", "y", "z"])
    result = featm.extract(
        X, features=["pointcloud.triangulation.delaunay"], columns={"pointcloud.triangulation.delaunay": ["x", "y", "z"]},
        memory_budget="200MB", output="memory", errors="raise",
    )
    assert result.shape[0] == 1  # one summary row for the whole point cloud, not 500


def test_finding5_row_independent_feature_unaffected_at_various_batch_sizes(register_temp_feature):
    def ok(x, **kw):
        return np.asarray(x, dtype=float) * 2

    register_temp_feature(FeatureSpec(name="audit4.f5.ok", family="f5", modality="tabular", input_types=("numeric",), implementation=ok))
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    for batch_rows in (1, 2, 7, 20):
        r = featm.extract(X, features=["audit4.f5.ok"], columns={"audit4.f5.ok": "x"}, batch_rows=batch_rows, output="memory", errors="raise")
        assert r.shape == (20, 1)


# ==========================================================================
# Finding 6 -- output schema must be stable across batches
# ==========================================================================
def test_finding6_schema_drift_raises_and_isolates(register_temp_feature):
    from featm.registry import FeatureOutput

    calls = {"n": 0}

    def drifting(x, **kw):
        calls["n"] += 1
        n = len(np.asarray(x))
        if calls["n"] == 1:
            return FeatureOutput(np.zeros((n, 2)), ("a", "b"))
        return FeatureOutput(np.zeros((n, 3)), ("a", "b", "c"))

    register_temp_feature(FeatureSpec(name="audit4.f6.drift", family="f6", modality="information", input_types=("distribution_or_samples",), output_kind="vector", implementation=drifting))
    X = pd.DataFrame(np.random.default_rng(0).random((10, 4)))
    result = featm.extract(X, features=["audit4.f6.drift"], batch_rows=5, output="memory", errors="warn")
    assert "audit4.f6.drift" in result.report.failed_features
    assert "audit4.f6.drift" not in result.report.completed_features

    calls["n"] = 0
    with pytest.raises(FeatureSchemaDriftError):
        featm.extract(X, features=["audit4.f6.drift"], batch_rows=5, output="memory", errors="raise")


def test_finding6_stateful_feature_naturally_has_frozen_schema():
    """A correctly fitted/stateful feature's schema is frozen by fit state, so it
    passes the schema-stability check by construction across any batch split."""
    df = pd.DataFrame({"text": TEXT_DOCS * 3})
    result = featm.extract(
        df, features=["text.tfidf.tfidf"], params={"text.tfidf.tfidf": {"max_features": 5}},
        batch_rows=3, output="memory", errors="raise",
    )
    assert result.shape == (18, 5)


# ==========================================================================
# Finding 7 -- registry-wide batch-semantics classification (spot checks;
# the exhaustive audit is narrated in STEP2_AUDIT4_FIX_HANDOFF.md)
# ==========================================================================
def test_finding7_registry_count_unaffected():
    assert len(REGISTRY.all()) == 552


def test_finding7_full_dataset_features_still_classified_correctly():
    for name in ("geometry.triangulation.delaunay_degree", "spatial.point_patterns.ripley_k", "text.semantics.topic"):
        spec = featm.feature_info(name)
        assert spec.batch_semantics == BatchSemantics.FULL_DATASET
        assert spec.requires_full_materialization is True


def test_finding7_row_independent_families_unaffected():
    for name in ("tabular.numeric.abs", "timeseries.statistical.mean", "signal.time_domain.rms", "event.timing.event_rate"):
        spec = featm.feature_info(name)
        assert spec.batch_semantics == BatchSemantics.ROW_INDEPENDENT


def test_finding7_stateful_text_features_classified_dataset_context():
    for name in ("text.information.surprisal", "text.bm25.bm25", "text.tfidf.tfidf", "text.semantics.similarity"):
        spec = featm.feature_info(name)
        assert spec.batch_semantics == BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED
        assert spec.stateful is True


# ==========================================================================
# Finding 9 -- checkpoint + output="memory" bypasses the hard memory budget
# ==========================================================================
def test_finding9_checkpoint_memory_materialize_respects_budget(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float)})
    ckpt = tmp_path / "ckpt"
    with pytest.raises(UnsafeOutputModeError):
        featm.extract(
            X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
            memory_budget="1KB", output="memory", checkpoint=ckpt, batch_rows=1000,
        )


def test_finding9_checkpoint_memory_override_accepts_risk(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float)})
    ckpt = tmp_path / "ckpt"
    result = featm.extract(
        X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        memory_budget="1KB", output="memory", checkpoint=ckpt, batch_rows=1000,
        allow_memory_budget_override=True,
    )
    assert result.shape == (100_000, 1)


def test_finding9_done_fast_path_also_enforces_budget(_pyarrow, tmp_path):
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float)})
    ckpt = tmp_path / "ckpt"
    featm.extract(
        X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        memory_budget="1KB", output="memory", checkpoint=ckpt, batch_rows=1000,
        allow_memory_budget_override=True,
    )
    with pytest.raises(UnsafeOutputModeError):
        featm.extract(
            X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
            memory_budget="1KB", output="memory", checkpoint=ckpt, batch_rows=1000,
        )


# ==========================================================================
# Finding 10 -- byte-aware input batch planning for strings/objects
# ==========================================================================
def test_finding10_planner_chooses_much_smaller_batch_for_long_strings():
    from featm.sources import PandasSource

    n = 200
    X = pd.DataFrame({"t": ["x" * 10_000 for _ in range(n)]})
    src = PandasSource(X)
    budget = ResourceBudget.resolve(memory_budget="64KB")
    spec = featm.feature_info("text.lexical.avg_word_length")
    real_bpr = src.estimated_bytes_per_row(["t"])
    assert real_bpr is not None and real_bpr > 5_000
    rows_old = plan_batch_rows(schema_n_columns=1, specs=[spec], budget=budget)
    rows_new = plan_batch_rows(schema_n_columns=1, specs=[spec], budget=budget, bytes_per_row_in=real_bpr)
    assert rows_new < rows_old / 10


def test_finding10_end_to_end_long_string_extraction_succeeds():
    n = 200
    X = pd.DataFrame({"t": ["x" * 10_000 for _ in range(n)]})
    result = featm.extract(X, features=["text.lexical.avg_word_length"], memory_budget="64KB", output="memory", errors="raise")
    assert result.shape == (n, 1)
    assert result.report.batch_rows is not None and result.report.batch_rows < 50


def test_finding10_pandas_source_byte_estimate_matches_deep_memory_usage():
    from featm.sources import PandasSource

    X = pd.DataFrame({"a": np.arange(100, dtype=float), "b": ["short"] * 100})
    src = PandasSource(X)
    est = src.estimated_bytes_per_row()
    actual = X.memory_usage(deep=True).sum() / len(X)
    assert est == int(actual)


# ==========================================================================
# Finding 11 -- state/reservoir byte caps use real per-item estimates
# ==========================================================================
def test_finding11_exact_frequency_long_strings_fail_safely_not_silently():
    n = 300
    X = pd.DataFrame({"c": [f"{i:06d}_" + ("x" * 10_000) for i in range(n)]})
    with pytest.raises(StateBudgetExceededError):
        featm.extract(
            X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
            memory_budget="64KB", output="memory", batch_rows=50, mode="exact", errors="raise",
        )


def test_finding11_approximate_reservoir_long_strings_fail_safely_not_silently():
    n = 500
    X = pd.DataFrame({"c": [f"{i:06d}_" + ("x" * 10_000) for i in range(n)]})
    with pytest.raises(ReservoirStateBudgetExceededError):
        featm.extract(
            X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
            memory_budget="64KB", output="memory", batch_rows=50, mode="approximate", errors="raise",
        )


def test_finding11_short_strings_still_work_at_the_same_tiny_budget():
    n = 300
    X = pd.DataFrame({"c": [f"cat_{i % 5}" for i in range(n)]})
    result = featm.extract(
        X, features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "c"},
        memory_budget="64KB", output="memory", batch_rows=50, mode="exact", errors="raise",
    )
    assert result.shape == (n, 1)


def test_finding11_reservoir_sampling_still_uniform_not_biased():
    """Sanity: with a budget generous relative to item size, the reservoir samples
    uniformly (no silent capacity shrink/bias introduced by byte-awareness)."""
    from featm.sampling import ReservoirSampler

    rng = np.random.default_rng(0)
    values = rng.normal(size=100_000)
    sampler, _ = ReservoirSampler.with_budget(1000, budget_bytes=10_000_000, bytes_per_item=50, seed=0)
    sampler.add_batch(values)
    sample = sampler.sample()
    assert len(sample) == 1000
    assert abs(sample.mean()) < 0.15  # should track the population mean (~0) reasonably closely


# ==========================================================================
# Finding 12 -- generic output-row/schema engine guardrails
# ==========================================================================
def test_finding12_row_aligned_feature_returning_fewer_rows_rejected(register_temp_feature):
    def bad(x, **kw):
        return np.zeros(max(len(np.asarray(x)) - 1, 0))

    register_temp_feature(FeatureSpec(name="audit4.f12.fewer_rows", family="f12", modality="tabular", input_types=("numeric",), implementation=bad))
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    with pytest.raises(FeatureRowAlignmentError):
        featm.extract(X, features=["audit4.f12.fewer_rows"], columns={"audit4.f12.fewer_rows": "x"}, batch_rows=3, output="memory", errors="raise")


def test_finding12_column_row_count_mismatch_rejected(register_temp_feature):
    def bad(x, **kw):
        n = len(np.asarray(x))
        return {"a": np.zeros(n), "b": np.zeros(max(n - 1, 0))}

    register_temp_feature(FeatureSpec(name="audit4.f12.col_mismatch", family="f12", modality="tabular", input_types=("numeric",), implementation=bad))
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    with pytest.raises(FeatureRowAlignmentError):
        featm.extract(X, features=["audit4.f12.col_mismatch"], columns={"audit4.f12.col_mismatch": "x"}, batch_rows=3, output="memory", errors="raise")


def test_finding12_schema_drift_across_batches_rejected(register_temp_feature):
    from featm.registry import FeatureOutput

    calls = {"n": 0}

    def drifting(x, **kw):
        calls["n"] += 1
        n = len(np.asarray(x))
        names = ("a",) if calls["n"] == 1 else ("b",)
        return FeatureOutput(np.zeros((n, 1)), names)

    register_temp_feature(FeatureSpec(name="audit4.f12.col_name_drift", family="f12", modality="tabular", input_types=("numeric",), output_kind="vector", implementation=drifting))
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    with pytest.raises(FeatureSchemaDriftError):
        featm.extract(X, features=["audit4.f12.col_name_drift"], columns={"audit4.f12.col_name_drift": "x"}, batch_rows=3, output="memory", errors="raise")


def test_finding12_row_independent_feature_works_at_batch_sizes_1_2_n(register_temp_feature):
    def ok(x, **kw):
        return np.asarray(x, dtype=float) + 1

    register_temp_feature(FeatureSpec(name="audit4.f12.row_independent", family="f12", modality="tabular", input_types=("numeric",), implementation=ok))
    X = pd.DataFrame({"x": np.arange(7, dtype=float)})
    for batch_rows in (1, 2, 7):
        r = featm.extract(X, features=["audit4.f12.row_independent"], columns={"audit4.f12.row_independent": "x"}, batch_rows=batch_rows, output="memory", errors="raise")
        assert np.allclose(r.to_pandas().to_numpy().ravel(), X["x"].to_numpy() + 1)


def test_finding12_non_batchable_feature_never_split_on_memory_error(register_temp_feature):
    calls = {"n": 0}

    def flaky_full(x, **kw):
        calls["n"] += 1
        a = np.asarray(x)
        if len(a) > 3:
            raise MemoryError("forced")
        return np.full(len(a), 1.0)

    register_temp_feature(FeatureSpec(name="audit4.f12.full_mat_flaky", family="f12", modality="geometry", input_types=("point_coordinates",), requires_full_materialization=True, implementation=flaky_full))
    X = pd.DataFrame({"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float)})
    result = featm.extract(X, features=["audit4.f12.full_mat_flaky"], columns={"audit4.f12.full_mat_flaky": ["x", "y"]}, memory_budget="1GB", output="memory", errors="warn")
    assert "audit4.f12.full_mat_flaky" in result.report.failed_features
    assert result.shape == (0, 0)


def test_finding12_stateful_feature_batch_splitting_remains_safe():
    df = pd.DataFrame({"x": np.arange(2000, dtype=float)})
    r1 = featm.extract(df, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, batch_rows=100, output="memory")
    r2 = featm.extract(df, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, batch_rows=333, output="memory")
    assert np.allclose(r1.to_pandas().to_numpy(), r2.to_pandas().to_numpy())


# ==========================================================================
# Additional bugs discovered by this pass's own automated batch-equivalence
# harness (benchmarks/batch_equivalence_smoke.py) -- not among the 12 named
# findings, found and fixed per the instruction to fix (not merely list) any
# additional real bug uncovered during this audit.
# ==========================================================================
@pytest.mark.parametrize("feature", [
    "tabular.numeric.rank", "tabular.numeric.percentile", "tabular.numeric.ecdf",
    "tabular.statistical.order_statistics", "tabular.categorical.entropy",
    "tabular.categorical.transition", "tabular.relational.entity_graph",
])
def test_additional_bug_whole_column_tabular_features_rejected_multi_batch(feature):
    """pandas .rank()/.value_counts()/whole-column entropy/.shift(1) all depend on
    the WHOLE column, not just the current row -- previously unflagged, unlike the
    already-correctly-rejected legacy incremental=True features."""
    spec = featm.feature_info(feature)
    assert spec.incremental is True
    n = 20
    if spec.arity == 2:
        df = pd.DataFrame({"a": [f"g{i%4}" for i in range(n)], "b": [f"h{i%3}" for i in range(n)]})
        columns = {feature: ["a", "b"]}
    elif spec.input_types == ("categorical",):
        df = pd.DataFrame({"x": [f"cat{i % 5}" for i in range(n)]})
        columns = {feature: "x"}
    else:
        df = pd.DataFrame({"x": np.linspace(-3, 3, n)})
        columns = {feature: "x"}
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(df, features=[feature], columns=columns, batch_rows=5, output="memory", errors="raise")
    # single-batch-safe run (whole dataset fits in one batch) still works correctly
    result = featm.extract(df, features=[feature], columns=columns, batch_rows=n, output="memory", errors="raise")
    assert result.shape[0] == n


def test_additional_bug_signal_ica_marked_full_materialization():
    """FastICA jointly decomposes ALL rows in a batch together -- the estimated
    unmixing matrix (and every row's output) depends on which other rows share
    its batch, not row-independent at all."""
    spec = featm.feature_info("signal.decomposition.ica")
    assert spec.requires_full_materialization is True
    assert spec.batch_semantics == BatchSemantics.FULL_DATASET


def test_additional_bug_signal_phase_locking_marked_structured_sample():
    spec = featm.feature_info("signal.synchronization.phase_locking")
    assert spec.batch_semantics == BatchSemantics.STRUCTURED_SAMPLE


def test_additional_bug_batch_equivalence_harness_finds_zero_mismatches():
    """Direct regression test mirroring benchmarks/batch_equivalence_smoke.py's
    exit code: the full registry-wide audit must report zero mismatches/row-
    alignment failures/schema-drift failures.

    Fifth audit pass, Finding 6: the parent pytest process sees ``src/``
    through pytest's own ``pythonpath = ["src"]`` ini option (a pytest-level
    import-path injection, not a real install), but a spawned CHILD process
    does not inherit that -- from a plain extracted source checkout with
    dependencies installed but featm itself not ``pip install -e .``-ed, this
    subprocess used to fail with ``ModuleNotFoundError: No module named
    'featm'`` even though every other test in the suite passed fine. Fixed by
    explicitly propagating the same ``src/`` path via the child's own
    ``PYTHONPATH``, reproducing exactly what pytest's ini option does for
    this process, rather than relying on the caller's ambient environment
    (an editable install, or a pre-set PYTHONPATH) to happen to make it work.
    """
    import os
    import pathlib
    import subprocess
    import sys as _sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_dir = str(repo_root / "src")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [_sys.executable, "-W", "ignore", "benchmarks/batch_equivalence_smoke.py"],
        capture_output=True, text=True, cwd=str(repo_root), env=env,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]


# ==========================================================================
# Finding 14 -- do not regress prior-pass fixes (spot checks)
# ==========================================================================
def test_finding14_leakage_safe_train_test_still_works():
    train = pd.DataFrame({"x": [0.0, 10.0]})
    test = pd.DataFrame({"x": [100.0, 110.0]})
    fitted = featm.fit(train, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, memory_budget="1MB")
    out = fitted.transform(test, output="memory").to_pandas().to_numpy().ravel()
    assert np.allclose(out, [19.0, 21.0])


def test_finding14_checkpoint_int_vs_str_label_identity_still_rejected(_pyarrow, tmp_path):
    from featm.checkpoint import CheckpointIncompatibleError

    X = pd.DataFrame({0: [1, 2, 3], "0": [10, 20, 30]})
    ckpt = tmp_path / "ckpt"
    featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": 0}, checkpoint=ckpt, output="disk", batch_rows=2, memory_budget="1MB")
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "0"}, checkpoint=ckpt, output="disk", batch_rows=2, memory_budget="1MB")


def test_finding14_cache_still_genuinely_per_batch(_pyarrow, tmp_path):
    n = 3000
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 16)), columns=[f"c{i}" for i in range(16)])
    p = tmp_path / "d.parquet"
    X.to_parquet(p, index=False)
    result = featm.extract(p, features=["timeseries.spectral.centroid"], batch_rows=500, output="memory")
    assert result.report.n_batches > 1
    assert result.report.cache_stats["misses"] == result.report.n_batches


def test_finding14_delaunay_voronoi_complexity_still_curated():
    for name in (
        "geometry.triangulation.delaunay_degree", "spatial.triangulation.delaunay",
        "spatial.tessellation.voronoi", "pointcloud.triangulation.delaunay",
    ):
        from featm.registry import Complexity

        assert featm.feature_info(name).complexity == Complexity.LINEARITHMIC


def test_finding14_all_v01_names_and_registry_count_preserved():
    assert len(REGISTRY.all()) == 552
