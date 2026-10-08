"""Regression tests for the second independent audit of the Step 2 engine.

Each section corresponds to one numbered finding in the audit. Tests that
need a disk backend (Parquet, output="disk", checkpointing) request the
``_pyarrow`` fixture; everything else must run in a base install.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointManager
from featm.execution import (
    ApproximateModeUnavailableError,
    ColumnResolutionError,
    RowCountConstraintError,
    UnsafeOutputModeError,
    _is_retry_split_safe,
    _resolve_effective_mode,
)
from featm.registry import Complexity, Cost, FeatureSpec, REGISTRY
from featm.sources import ArrowSource, CSVSource, PandasSource


@pytest.fixture
def _pyarrow():
    return pytest.importorskip("pyarrow")


@pytest.fixture
def register_temp_feature():
    """Register a synthetic FeatureSpec for the duration of one test only.

    REGISTRY is a process-wide singleton; without teardown, synthetic test
    features permanently inflate REGISTRY.all() and break the fixed-count
    compatibility checks in test_core.py (549 + 3).
    """
    registered: list[str] = []

    def _register(spec: FeatureSpec) -> FeatureSpec:
        REGISTRY.register(spec)
        registered.append(spec.name)
        return spec

    yield _register
    for name in registered:
        REGISTRY._specs.pop(name, None)


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
# Finding 1 -- MemoryError retry must not split a non-batchable feature
# ==========================================================================
def test_finding1_full_materialization_feature_not_split_on_memory_error(register_temp_feature):
    name = "audit2.f1.full_memory_retry"

    def fn(x, **kw):
        a = np.asarray(x)
        if len(a) > 50:
            raise MemoryError("forced")
        return np.full(len(a), len(a), dtype=float)

    register_temp_feature(
        FeatureSpec(name=name, family="f1", modality="geometry", input_types=("point_coordinates",), requires_full_materialization=True, cost=Cost.LOW, implementation=fn)
    )
    X = pd.DataFrame({"x": np.arange(100.0), "y": np.arange(100.0)})
    with pytest.raises(MemoryError):
        featm.extract(X, features=[name], columns={name: ["x", "y"]}, memory_budget="1GB", output="memory", errors="raise")

    # under errors="warn"/"skip" it must be isolated, never return a chunk-local [50.] answer
    result = featm.extract(X, features=[name], columns={name: ["x", "y"]}, memory_budget="1GB", output="memory", errors="warn")
    assert name in result.report.failed_features
    assert result.shape[1] == 0


def test_finding1_delaunay_not_retried_as_sub_triangulations(monkeypatch):
    """A real Delaunay feature forced to MemoryError must fail, never be split."""
    import featm.features.geometry.triangulation as tri_mod

    def boom(pts):
        raise MemoryError("forced allocation failure")

    monkeypatch.setattr(tri_mod, "_tri_data_impl", boom)
    pts = np.random.default_rng(0).normal(size=(120, 2))
    X = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1]})
    result = featm.extract(
        X, features=["geometry.triangulation.delaunay_degree"], columns={"geometry.triangulation.delaunay_degree": ["x", "y"]},
        memory_budget="1GB", output="memory", errors="warn",
    )
    assert "geometry.triangulation.delaunay_degree" in result.report.failed_features
    assert result.shape == (0, 0)


def test_finding1_legacy_single_shot_tabular_not_split_on_memory_error(register_temp_feature):
    calls = {"n": 0}

    def flaky_zscore(s, *, fit=False, state=None):
        calls["n"] += 1
        if len(s) > 300:
            raise MemoryError("forced")
        if fit or state is None:
            state = (s.mean(), s.std(ddof=0))
        mu, sd = state
        return (s - mu) / (sd if sd else np.nan), state

    name = "audit2.f1.legacy_flaky"
    register_temp_feature(FeatureSpec(name=name, family="f1", modality="tabular", input_types=("numeric",), incremental=True, cost=Cost.LOW, implementation=flaky_zscore))
    n = 1000
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    # single-batch-safe (batch_rows covers the whole n) but still forces MemoryError
    result = featm.extract(X, features=[name], columns={name: "x"}, batch_rows=n + 10, output="memory", errors="warn")
    assert name in result.report.failed_features
    assert result.shape == (0, 0)


def test_finding1_row_independent_feature_still_recovers_by_halving(register_temp_feature):
    def fragile(s, **kw):
        if len(s) > 300:
            raise MemoryError("forced")
        return s.astype(float) + 1.0

    name = "audit2.f1.row_independent"
    register_temp_feature(FeatureSpec(name=name, family="f1", modality="tabular", input_types=("numeric",), cost=Cost.LOW, implementation=fragile))
    n = 2000
    X = pd.DataFrame({"x": np.arange(n, dtype=float)})
    result = featm.extract(X, features=[name], columns={name: "x"}, batch_rows=n, output="memory", errors="warn")
    assert result.shape == (n, 1)
    assert sum(result.report.batch_reductions.values()) > 0
    assert np.allclose(result.to_pandas().iloc[:, 0].to_numpy(), X["x"].to_numpy() + 1.0)


def test_finding1_stateful_transform_still_allowed_to_split_after_fit():
    assert _is_retry_split_safe(featm.feature_info("tabular.scaling.zscore"), is_stateful_transform=True) is True
    assert _is_retry_split_safe(featm.feature_info("tabular.scaling.zscore"), is_stateful_transform=False) is True  # row-independent by default outside fit
    assert _is_retry_split_safe(featm.feature_info("geometry.triangulation.delaunay_degree"), is_stateful_transform=False) is False
    legacy = featm.feature_info("tabular.numeric.zscore")
    assert _is_retry_split_safe(legacy, is_stateful_transform=False) is False


# ==========================================================================
# Finding 2 -- output="auto" must account for the REAL output width
# ==========================================================================
def test_finding2_fft_reproducer_switches_to_disk_with_io(_pyarrow):
    n, T = 20_000, 64
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, T)), columns=[f"c{i}" for i in range(T)])
    F = featm.extract(X, features=["timeseries.spectral.fft"], memory_budget="2MB", output="auto")
    assert F.shape == (n, 32)
    assert F.is_disk_backed


def test_finding2_fft_reproducer_fails_safely_without_io(monkeypatch):
    from featm import execution

    monkeypatch.setattr(execution, "_io_backend_available", lambda: False)
    n, T = 20_000, 64
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, T)), columns=[f"c{i}" for i in range(T)])
    with pytest.raises(UnsafeOutputModeError):
        featm.extract(X, features=["timeseries.spectral.fft"], memory_budget="2MB", output="auto")


def test_finding2_parameterized_width_no_hardcoded_32(_pyarrow):
    n, T = 20_000, 64
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, T)), columns=[f"c{i}" for i in range(T)])
    # n_coeffs=4 -> 8 real columns, well within a 2MB budget -> must stay memory
    F = featm.extract(X, features=["timeseries.spectral.fft"], params={"timeseries.spectral.fft": {"n_coeffs": 4}}, memory_budget="2MB", output="auto")
    assert F.shape == (n, 8)
    assert not F.is_disk_backed


def test_finding2_multi_feature_combined_width_exceeds_pre_run_estimate(_pyarrow):
    n, T = 20_000, 64
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, T)), columns=[f"c{i}" for i in range(T)])
    F = featm.extract(X, features=["timeseries.spectral.fft", "timeseries.spectral.periodogram"], memory_budget="2MB", output="auto")
    assert F.shape == (n, 48)
    assert F.is_disk_backed


def test_finding2_scalar_output_stays_memory_when_it_genuinely_fits():
    n, T = 20_000, 64
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, T)), columns=[f"c{i}" for i in range(T)])
    F = featm.extract(X, features=["timeseries.statistical.mean"], memory_budget="2MB", output="auto")
    assert F.shape == (n, 1)
    assert not F.is_disk_backed


# ==========================================================================
# Finding 3 -- PandasSource with non-string / mixed column labels
# ==========================================================================
def test_finding3_integer_labeled_columns():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 8)))  # columns 0..7
    result = featm.extract(X, features=["timeseries.statistical.mean"], memory_budget="2MB", output="memory")
    assert result.shape == (100, 1)


def test_finding3_mixed_int_and_string_labels():
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], "a": [4.0, 5.0, 6.0]})
    src = PandasSource(X)
    assert src.schema().names() == [0, "a"]
    batch = next(src.iter_batches(columns=[0, "a"], batch_rows=10))
    assert list(batch.columns) == [0, "a"]


def test_finding3_explicit_columns_mapping_with_integer_labels():
    X = pd.DataFrame({0: [1.0, 2.0], 1: [3.0, 4.0]})
    result = featm.extract(X, features=["tabular.interactions.product"], columns={"tabular.interactions.product": [0, 1]}, output="memory")
    assert np.allclose(result.to_pandas().iloc[:, 0].to_numpy(), [3.0, 8.0])


def test_finding3_no_sorted_crash_on_mixed_label_types():
    from featm.execution import _ordered_unique_columns

    # sorted({0, "a"}) raises TypeError in plain Python; the schema-order helper must not.
    order = _ordered_unique_columns({0, "a"}, [0, "a", "b"])
    assert order == [0, "a"]


def test_finding3_checkpoint_identity_deterministic_with_int_labels(_pyarrow, tmp_path):
    X = pd.DataFrame({0: np.arange(200, dtype=float)})
    ckpt = tmp_path / "ckpt"
    result = featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": 0}, batch_rows=50, output="disk", checkpoint=ckpt)
    assert result.shape == (200, 1)
    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is True


# ==========================================================================
# Finding 4 -- explicit column mapping validated early
# ==========================================================================
def test_finding4_bad_explicit_column_isolated_under_skip():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "NOPE"}, output="memory", errors="skip")
    assert result.shape == (0, 0)
    assert "tabular.numeric.abs" in result.report.skipped_features


def test_finding4_bad_explicit_column_raises_under_raise():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ColumnResolutionError):
        featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "NOPE"}, output="memory", errors="raise")


def test_finding4_other_valid_feature_still_runs_alongside_bad_mapping():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = featm.extract(
        X, features=["tabular.numeric.abs", "tabular.scaling.zscore"],
        columns={"tabular.numeric.abs": "NOPE", "tabular.scaling.zscore": "b"},
        output="memory", errors="warn",
    )
    assert "tabular.scaling.zscore" in result.report.completed_features
    assert "tabular.numeric.abs" in result.report.skipped_features


# ==========================================================================
# Finding 5 -- API parity: empty selection and modality=
# ==========================================================================
def test_finding5_empty_feature_selection_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least one feature"):
        featm.extract(X, features=[], memory_budget="1MB", output="memory")


def test_finding5_modality_mismatch_rejected_in_large_engine():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="do not belong to modality"):
        featm.extract(X, features=["tabular.numeric.abs"], modality="image", columns={"tabular.numeric.abs": "a"}, output="memory")


def test_finding5_matching_modality_still_works():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = featm.extract(X, features=["tabular.numeric.abs"], modality="tabular", columns={"tabular.numeric.abs": "a"}, output="memory")
    assert result.shape == (3, 1)


# ==========================================================================
# Finding 6 -- mode="approximate" resolution for every selected feature
# ==========================================================================
def test_finding6_unsupported_direct_feature_approximate_mode_skipped_not_silently_exact():
    X = pd.DataFrame(np.arange(20).reshape(4, 5), columns=list("abcde"))
    result = featm.extract(X, features=["timeseries.statistical.min"], mode="approximate", output="memory", errors="warn")
    assert result.report.approximations_used == {}
    assert "timeseries.statistical.min" in result.report.skipped_features
    assert result.shape == (0, 0)


def test_finding6_unsupported_direct_feature_approximate_mode_raises_under_raise():
    X = pd.DataFrame(np.arange(20).reshape(4, 5), columns=list("abcde"))
    with pytest.raises(ApproximateModeUnavailableError):
        featm.extract(X, features=["timeseries.statistical.min"], mode="approximate", output="memory", errors="raise")


def test_finding6_max_rows_exact_enforced_even_with_approximate_available():
    spec = FeatureSpec(
        name="audit2.f6.maxrows", family="f6", modality="tabular", input_types=("numeric",),
        max_rows_exact=100, approximate_available=True, implementation=lambda s, **kw: (s, None),
    )
    with pytest.raises(RowCountConstraintError):
        _resolve_effective_mode(spec, "exact", 500)
    # auto mode falls back to approximate instead of rejecting, since it's available
    assert _resolve_effective_mode(spec, "auto", 500) == "approximate"
    assert _resolve_effective_mode(spec, "auto", 50) == "exact"


def test_finding6_stateful_exact_approx_auto_still_correct():
    X = pd.DataFrame({"x": np.arange(5000, dtype=float)})
    r_exact = featm.extract(X, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, mode="exact", output="memory")
    assert r_exact.report.approximations_used == {}
    r_approx = featm.extract(X, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, mode="approximate", output="memory")
    assert "tabular.scaling.zscore" in r_approx.report.approximations_used


# ==========================================================================
# Finding 7 -- cache scope per batch; tiny budgets never exceeded
# ==========================================================================
def test_finding7_cache_scope_is_per_batch_not_whole_run(_pyarrow, tmp_path):
    from featm.features import _utils

    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame(rng.normal(size=(n, 16)), columns=[f"c{i}" for i in range(16)])
    path = tmp_path / "d.parquet"
    X.to_parquet(path, index=False)

    calls = {"n": 0}
    original = _utils._spectral_parts_impl

    def counting(a):
        calls["n"] += 1
        return original(a)

    _utils._spectral_parts_impl = counting
    try:
        result = featm.extract(path, features=["timeseries.spectral.centroid", "timeseries.spectral.fft"], batch_rows=1000, output="memory")
    finally:
        _utils._spectral_parts_impl = original
    # One shared computation PER BATCH (fresh cache each batch), not once for the whole run.
    assert calls["n"] == result.report.n_batches
    assert result.report.n_batches > 1
    assert result.report.cache_stats["hits"] == result.report.n_batches
    assert result.report.cache_stats["misses"] == result.report.n_batches


@pytest.mark.parametrize("budget_str", ["32KB", "100KB"])
def test_finding7_cache_bytes_never_exceed_declared_memory_budget(budget_str):
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 32)), columns=[f"c{i}" for i in range(32)])
    result = featm.extract(X, features=["timeseries.spectral.centroid"], memory_budget=budget_str, output="memory")
    from featm.resources import parse_size

    assert result.report.cache_stats["bytes"] <= parse_size(budget_str)


def test_finding7_impossibly_tiny_budget_fails_safely_instead_of_silent_floor():
    """Third-audit-pass regression: a 1KB budget with 32 wide columns cannot fit even
    one row once a real (non-floored) per-row estimate is honored -- see
    test_audit3.py::test_finding7_* for the direct planner-level reproducer/fix.
    """
    from featm.resources import ResourceBudgetTooSmallError

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 32)), columns=[f"c{i}" for i in range(32)])
    with pytest.raises(ResourceBudgetTooSmallError):
        featm.extract(X, features=["timeseries.spectral.centroid"], memory_budget="1KB", output="memory")


def test_finding7_cache_stats_aggregated_across_batches(_pyarrow, tmp_path):
    n = 3000
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 16)), columns=[f"c{i}" for i in range(16)])
    path = tmp_path / "d.parquet"
    X.to_parquet(path, index=False)
    result = featm.extract(path, features=["timeseries.spectral.centroid"], batch_rows=500, output="memory")
    total_batches = result.report.n_batches
    assert total_batches > 1
    assert result.report.cache_stats["misses"] == total_batches  # one fresh miss per batch scope


# ==========================================================================
# Finding 8 -- checkpoint source identity for pandas / Arrow sources
# ==========================================================================
def test_finding8_pandas_fingerprint_detects_in_place_mutation():
    X = pd.DataFrame({"a": [1, 2, 3]})
    s = PandasSource(X)
    f1 = s.fingerprint()
    X.loc[1, "a"] = 999
    f2 = s.fingerprint()
    assert f1 != f2


def test_finding8_pandas_fingerprint_stable_for_unchanged_data():
    X = pd.DataFrame({"a": [1, 2, 3]})
    assert PandasSource(X).fingerprint() == PandasSource(X.copy()).fingerprint()


def test_finding8_arrow_fingerprint_reflects_file_set_not_just_path(_pyarrow, tmp_path):
    df = pd.DataFrame({"a": np.arange(100, dtype=float)})
    d = tmp_path / "dataset"
    d.mkdir()
    df.to_parquet(d / "part-0.parquet", index=False)
    fp1 = ArrowSource(d).fingerprint()
    df2 = pd.DataFrame({"a": np.arange(200, dtype=float)})
    df2.to_parquet(d / "part-0.parquet", index=False)
    fp2 = ArrowSource(d).fingerprint()
    assert fp1 != fp2


def test_finding8_checkpoint_uses_content_fingerprint_for_pandas(_pyarrow, tmp_path):
    """A checkpoint against a PandasSource must not be silently reused after mutation."""
    from featm.checkpoint import CheckpointIncompatibleError

    X = pd.DataFrame({"a": np.arange(500, dtype=float)})
    ckpt = tmp_path / "ckpt"
    featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "a"}, batch_rows=100, output="disk", checkpoint=ckpt)
    X.loc[0, "a"] = -999.0  # mutate after the checkpoint was written
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(X, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "a"}, batch_rows=100, output="disk", checkpoint=ckpt)


# ==========================================================================
# Finding 9 -- fit-phase checkpoint transactional consistency
# ==========================================================================
def test_finding9_crash_between_accumulator_snapshot_and_manifest_commit(_pyarrow, tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({"price": rng.normal(50, 5, n), "city": rng.choice(["A", "B"], n)})
    path = tmp_path / "d.csv"
    X.to_csv(path, index=False)
    ckpt = tmp_path / "ckpt"

    orig_save = CheckpointManager.save
    state = {"n": 0, "fail_at": 3}

    def flaky_save(self, manifest):
        state["n"] += 1
        if state["n"] == state["fail_at"]:
            raise RuntimeError("SIMULATED CRASH before manifest commit")
        return orig_save(self, manifest)

    CheckpointManager.save = flaky_save
    try:
        with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
            featm.extract(
                path, features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
                columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
                batch_rows=500, output="disk", checkpoint=ckpt,
            )
    finally:
        CheckpointManager.save = orig_save

    result = featm.extract(
        path, features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=500, output="disk", checkpoint=ckpt,
    )
    ckpt_clean = tmp_path / "ckpt_clean"
    clean = featm.extract(
        path, features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=500, output="disk", checkpoint=ckpt_clean,
    )
    a, b = result.to_pandas(), clean.to_pandas()
    assert a.shape == b.shape
    assert np.allclose(a.iloc[:, 0], b.iloc[:, 0])  # z-score matches exactly, not double-counted
    assert (a.iloc[:, 1].to_numpy() == b.iloc[:, 1].to_numpy()).all()  # category frequency matches exactly


# ==========================================================================
# Finding 10 -- completed checkpoint fast path validates output shards
# ==========================================================================
def test_finding10_missing_shard_detected_not_silently_reused(_pyarrow, tmp_path):
    from featm.matrix import CheckpointCorruptionError

    X = pd.DataFrame({"price": np.arange(3000, dtype=float)})
    path = tmp_path / "d.csv"
    X.to_csv(path, index=False)
    ckpt = tmp_path / "ckpt"
    featm.extract(path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)

    out_dir = ckpt / "output"
    shard = next(f for f in out_dir.iterdir() if f.name.startswith("part-"))
    shard.unlink()

    with pytest.raises(CheckpointCorruptionError):
        featm.extract(path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)


def test_finding10_intact_checkpoint_still_reused_with_zero_reads(_pyarrow, tmp_path):
    X = pd.DataFrame({"price": np.arange(3000, dtype=float)})
    path = tmp_path / "d.csv"
    X.to_csv(path, index=False)
    ckpt = tmp_path / "ckpt"
    featm.extract(path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)

    src, reads = _counting_source(CSVSource, path)
    result = featm.extract(src, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=500, output="disk", checkpoint=ckpt)
    assert sum(reads) == 0
    assert result.shape == (3000, 1)


# ==========================================================================
# Finding 11 -- full-materialization resource planning uses complexity/estimator
# ==========================================================================
def test_finding11_quadratic_synthetic_feature_rejected_before_huge_allocation(register_temp_feature):
    name = "audit2.f11.quadratic"
    register_temp_feature(
        FeatureSpec(
            name=name, family="f11", modality="geometry", input_types=("point_coordinates",),
            requires_full_materialization=True, cost=Cost.LOW,
            memory_estimator=lambda n: n * n * 8,  # simulated O(n^2) dense structure
            implementation=lambda x, **kw: np.full(len(np.asarray(x)), 1.0),
        )
    )
    # 100,000 points -> raw bytes ~1.6MB, but n^2*8 = 80GB: must be rejected, no huge allocation attempted.
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float), "y": np.arange(100_000, dtype=float)})
    result = featm.extract(X, features=[name], columns={name: ["x", "y"]}, memory_budget="2GB", output="memory", errors="warn")
    assert name in result.report.skipped_features
    assert any("estimated working memory" in w for w in result.report.warnings)


def test_finding11_linear_feature_at_same_scale_approved(register_temp_feature):
    name = "audit2.f11.linear"
    register_temp_feature(
        FeatureSpec(
            name=name, family="f11", modality="geometry", input_types=("point_coordinates",),
            requires_full_materialization=True, cost=Cost.LOW, complexity=Complexity.LINEAR,
            implementation=lambda x, **kw: np.full(len(np.asarray(x)), 1.0),
        )
    )
    X = pd.DataFrame({"x": np.arange(100_000, dtype=float), "y": np.arange(100_000, dtype=float)})
    result = featm.extract(X, features=[name], columns={name: ["x", "y"]}, memory_budget="2GB", output="memory", errors="raise")
    assert name in result.report.completed_features


def test_finding11_curated_quadratic_spatial_family_estimator_present():
    spec = featm.feature_info("spatial.point_patterns.ripley_k")
    assert spec.complexity == Complexity.QUADRATIC
    assert spec.memory_estimator is not None
    assert spec.memory_estimator(1000) == 1000 * 1000 * 8


def test_finding11_delaunay_not_wrongly_rejected_at_realistic_scale(_pyarrow, tmp_path):
    """Regression: a naive quadratic default would wrongly reject a real Delaunay run."""
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(50_000, 2))
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1]})
    path = tmp_path / "pts.parquet"
    df.to_parquet(path, index=False)
    result = featm.extract(
        path, features=["geometry.triangulation.delaunay_degree"], columns={"geometry.triangulation.delaunay_degree": ["x", "y"]},
        memory_budget="500MB", errors="raise",
    )
    assert "geometry.triangulation.delaunay_degree" in result.report.completed_features


# ==========================================================================
# Finding 12 -- IO backend checked before expensive work
# ==========================================================================
def test_finding12_output_disk_without_io_fails_before_any_read(monkeypatch, tmp_path):
    from featm import execution

    monkeypatch.setattr(execution, "_io_backend_available", lambda: False)
    X = pd.DataFrame({"price": np.arange(5000, dtype=float)})
    path = tmp_path / "d.csv"
    X.to_csv(path, index=False)
    src, reads = _counting_source(CSVSource, path)
    with pytest.raises(ImportError):
        featm.extract(src, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, output="disk", batch_rows=500)
    assert sum(reads) == 0


def test_finding12_auto_output_no_disk_backend_fails_before_stateful_fit_scan(monkeypatch):
    from featm import execution

    monkeypatch.setattr(execution, "_io_backend_available", lambda: False)
    X = pd.DataFrame({"price": np.arange(5000, dtype=float)})
    src, reads = _counting_source(PandasSource, X)
    with pytest.raises(ImportError):
        featm.extract(src, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, memory_budget="1KB", output="auto", batch_rows=500)
    assert sum(reads) == 0
