from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError
from featm.execution import (
    ApproximateModeUnavailableError,
    ColumnResolutionError,
    RowCountConstraintError,
    UnsafeMultiBatchFeatureError,
    UnsafeOutputModeError,
    _resolve_effective_mode,
)
from featm.matrix import FeatureMatrix
from featm.registry import Cost, FeatureSpec, REGISTRY
from featm.sources import CSVSource


@pytest.fixture
def _pyarrow():
    """Request this fixture to skip a test cleanly when pyarrow is absent.

    Only Parquet/disk-backed/checkpointed runs need this. CSV + memory-backed
    out-of-core extraction must keep working (and being tested) in a base
    install with no optional dependencies -- that is the whole point of
    keeping the base install lightweight.
    """
    return pytest.importorskip("pyarrow")


def _counting_csv_source(path):
    """A CSVSource wrapper that records how many rows each iter_batches() call reads."""
    src = CSVSource(path)
    original_iter = CSVSource.iter_batches
    reads: list[int] = []

    def counting_iter(self, *, columns=None, batch_rows=50_000):
        for batch in original_iter(self, columns=columns, batch_rows=batch_rows):
            reads.append(len(batch))
            yield batch

    src.iter_batches = types.MethodType(counting_iter, src)
    return src, reads


# --------------------------------------------------------------------------
# Basic out-of-core extraction over CSV (base install: no optional deps needed)
# --------------------------------------------------------------------------
def test_csv_extraction_returns_feature_matrix(csv_path, sample_tabular_df):
    result = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=700,
        output="memory",
    )
    assert isinstance(result, FeatureMatrix)
    assert result.shape == (len(sample_tabular_df), 2)
    assert result.report is not None
    assert set(result.report.completed_features) == {"tabular.scaling.zscore", "tabular.encoding.category_frequency"}


def test_small_in_memory_path_unchanged_return_type():
    """The plain small-data call must still return a bare pd.DataFrame (v0.1.0 behavior)."""
    x = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    out = featm.extract(x, features=["timeseries.statistical.mean"])
    assert isinstance(out, pd.DataFrame)
    assert not isinstance(out, FeatureMatrix)


def test_deterministic_column_order(csv_path):
    result = featm.extract(
        csv_path,
        features=["tabular.encoding.category_frequency", "tabular.scaling.zscore"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=1000,
        output="memory",
    )
    assert result.columns == ["tabular.encoding.category_frequency", "tabular.scaling.zscore"]


# --------------------------------------------------------------------------
# Item 1: no hidden full-source materialization / extra priming scan
# --------------------------------------------------------------------------
def test_direct_feature_extraction_reads_source_exactly_once(csv_path, sample_tabular_df):
    """A stateless (direct, one-pass-capable) extraction must scan the source exactly once."""
    src, reads = _counting_csv_source(csv_path)
    result = featm.extract(
        src, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"}, batch_rows=700, output="memory"
    )
    assert sum(reads) == len(sample_tabular_df)  # not 2x or 3x -- exactly the row count, once
    assert result.shape == (len(sample_tabular_df), 1)


def test_stateful_feature_extraction_reads_source_exactly_twice(csv_path, sample_tabular_df):
    """Fit + transform is two passes -- not three (the old bug added a third, discarded, priming pass)."""
    src, reads = _counting_csv_source(csv_path)
    result = featm.extract(
        src, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=700, output="memory"
    )
    assert sum(reads) == 2 * len(sample_tabular_df)
    assert result.shape == (len(sample_tabular_df), 1)


def test_no_whole_source_collection_in_execution_module():
    import inspect

    from featm import execution

    src = inspect.getsource(execution)
    assert "list(src.iter_batches" not in src
    assert "list(source.iter_batches" not in src


# --------------------------------------------------------------------------
# Item 2: batch-safety contract -- features whose math needs the whole dataset
# --------------------------------------------------------------------------
def test_full_dataset_feature_rejected_when_it_does_not_fit_one_batch(csv_path):
    """Small batch_rows with an unknown-row-count source: must reject, never chunk silently."""
    result = featm.extract(
        csv_path,
        features=["geometry.triangulation.delaunay_degree"],
        columns={"geometry.triangulation.delaunay_degree": ["price", "quantity"]},
        batch_rows=30,
        errors="warn",
        output="memory",
    )
    assert result.report.completed_features == []
    assert "geometry.triangulation.delaunay_degree" in result.report.skipped_features
    assert result.shape == (0, 0)


def test_full_dataset_feature_rejected_raises_with_errors_raise(csv_path):
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(
            csv_path,
            features=["geometry.triangulation.delaunay_degree"],
            columns={"geometry.triangulation.delaunay_degree": ["price", "quantity"]},
            batch_rows=30,
            errors="raise",
            output="memory",
        )


def test_full_dataset_feature_matches_whole_input_when_budget_allows(_pyarrow, parquet_path, sample_tabular_df):
    """With a large enough memory budget (known row count via Parquet), the engine grows the
    batch to cover the whole dataset and must produce the mathematically exact whole-input
    result -- this is the audit's core Delaunay-chunking regression."""
    pts = sample_tabular_df[["price", "quantity"]].to_numpy()
    normal = featm.extract(pts, features=["geometry.triangulation.delaunay_degree"])

    engine_result = featm.extract(
        parquet_path,
        features=["geometry.triangulation.delaunay_degree"],
        columns={"geometry.triangulation.delaunay_degree": ["price", "quantity"]},
        memory_budget="2GB",
        errors="raise",
    )
    a = normal.iloc[:, 0].to_numpy()
    b = engine_result.to_pandas().iloc[:, 0].to_numpy()
    assert np.array_equal(a, b)


def test_matrix_family_also_protected_from_silent_chunking(csv_path):
    """Representative coverage beyond geometry: matrix.* is also requires_full_materialization."""
    spec = featm.feature_info("matrix.algebra.determinant")
    assert spec.batchable is False
    result = featm.extract(
        csv_path,
        features=["matrix.algebra.determinant"],
        columns={"matrix.algebra.determinant": ["price", "quantity"]},
        batch_rows=30,
        errors="warn",
        output="memory",
    )
    assert "matrix.algebra.determinant" in result.report.skipped_features


@pytest.mark.parametrize(
    "name",
    [
        "geometry.triangulation.delaunay_degree",
        "spatial.neighbors.nearest_neighbor",
        "manifold.dimension.local_dimension",
        "topology.betti.numbers",
        "pointcloud.geometry.bbox_volume",
        "matrix.algebra.determinant",
    ],
)
def test_full_dataset_families_all_marked_not_batchable(name):
    spec = featm.feature_info(name)
    assert spec.batchable is False, f"{name} should be requires_full_materialization=True (not batchable)"


# --------------------------------------------------------------------------
# Item 3: tabular features with arity > 1
# --------------------------------------------------------------------------
def test_two_arg_interaction_feature_matches_reference_across_batches():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    result = featm.extract(
        df,
        features=["tabular.interactions.product"],
        columns={"tabular.interactions.product": ["a", "b"]},
        batch_rows=2,
        output="memory",
    )
    expected = (df["a"] * df["b"]).to_numpy()
    assert np.allclose(result.to_pandas().iloc[:, 0].to_numpy(), expected)


def test_two_arg_ratio_feature_matches_reference_across_batches():
    df = pd.DataFrame({"a": [10.0, 20.0, 0.0, 40.0, 50.0], "b": [2.0, 0.0, 5.0, 4.0, 10.0]})
    result = featm.extract(
        df, features=["tabular.interactions.ratio_safe"], columns={"tabular.interactions.ratio_safe": ["a", "b"]},
        batch_rows=2, output="memory",
    )
    expected = df["a"] / df["b"].replace(0, np.nan)
    got = result.to_pandas().iloc[:, 0].to_numpy()
    assert np.allclose(got, expected.to_numpy(), equal_nan=True)


def test_three_arg_interaction_feature_matches_reference_across_batches():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 2.0, 2.0, 2.0], "c": [1.0, 1.0, 2.0, 2.0]})
    result = featm.extract(
        df, features=["tabular.interactions.three_way"],
        columns={"tabular.interactions.three_way": ["a", "b", "c"]}, batch_rows=1, output="memory",
    )
    expected = (df["a"] * df["b"] * df["c"]).to_numpy()
    assert np.allclose(result.to_pandas().iloc[:, 0].to_numpy(), expected)


def test_wrong_arity_mapping_length_gives_clear_error():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ColumnResolutionError, match="arity=2"):
        featm.extract(
            df, features=["tabular.interactions.product"], columns={"tabular.interactions.product": ["a"]},
            batch_rows=1, errors="raise", output="memory",
        )


def test_arity_greater_than_one_ambiguous_without_explicit_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ColumnResolutionError, match="explicit"):
        featm.extract(df, features=["tabular.interactions.product"], errors="raise", output="memory")


def test_multi_arity_output_column_order_deterministic():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    result = featm.extract(
        df,
        features=["tabular.interactions.three_way", "tabular.interactions.product"],
        columns={"tabular.interactions.three_way": ["a", "b", "c"], "tabular.interactions.product": ["a", "b"]},
        batch_rows=1,
        output="memory",
    )
    assert result.columns == ["tabular.interactions.three_way", "tabular.interactions.product"]


# --------------------------------------------------------------------------
# Fault isolation (base install: CSV + output="memory")
# --------------------------------------------------------------------------
def test_permanently_failing_feature_isolated_with_warn(csv_path, register_temp_feature):
    def always_fails(s, **kwargs):
        raise ValueError("boom")

    register_temp_feature(
        FeatureSpec(name="test.exec.always_fails", family="exec", modality="tabular", input_types=("numeric",), implementation=always_fails)
    )
    result = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore", "test.exec.always_fails"],
        columns={"tabular.scaling.zscore": "price", "test.exec.always_fails": "price"},
        batch_rows=1000,
        errors="warn",
        output="memory",
    )
    assert "tabular.scaling.zscore" in result.report.completed_features
    assert "test.exec.always_fails" in result.report.failed_features
    assert result.shape[1] == 1  # the failed feature contributes no columns


def test_errors_raise_strict_mode_stops_the_run(csv_path, register_temp_feature):
    def always_fails(s, **kwargs):
        raise ValueError("boom")

    register_temp_feature(
        FeatureSpec(name="test.exec.always_fails2", family="exec", modality="tabular", input_types=("numeric",), implementation=always_fails)
    )
    with pytest.raises(ValueError, match="boom"):
        featm.extract(
            csv_path,
            features=["test.exec.always_fails2"],
            columns={"test.exec.always_fails2": "price"},
            batch_rows=1000,
            errors="raise",
            output="memory",
        )


def test_optional_dependency_failure_isolated(csv_path, register_temp_feature):
    def missing_dep(s, **kwargs):
        raise ImportError("Install 'nonexistent_package'")

    register_temp_feature(
        FeatureSpec(name="test.exec.missing_dep", family="exec", modality="tabular", input_types=("numeric",), implementation=missing_dep)
    )
    result = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore", "test.exec.missing_dep"],
        columns={"tabular.scaling.zscore": "price", "test.exec.missing_dep": "price"},
        batch_rows=1000,
        errors="warn",
        output="memory",
    )
    assert "test.exec.missing_dep" in result.report.optional_dependency_failures


def test_degraded_batch_fills_nan_and_keeps_schema(csv_path, register_temp_feature):
    calls = {"n": 0}

    def flaky_every_third_batch(s, **kwargs):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("intermittent failure")
        return s.astype(float) + 1.0

    register_temp_feature(
        FeatureSpec(name="test.exec.flaky", family="exec", modality="tabular", input_types=("numeric",), implementation=flaky_every_third_batch)
    )
    result = featm.extract(
        csv_path,
        features=["test.exec.flaky"],
        columns={"test.exec.flaky": "price"},
        batch_rows=500,
        errors="warn",
        output="memory",
    )
    pdf = result.to_pandas()
    assert pdf.shape[1] == 1  # schema stayed fixed despite some batches failing
    assert pdf.iloc[:, 0].isna().any()  # some rows are NaN
    assert pdf.iloc[:, 0].notna().any()  # but not all


# --------------------------------------------------------------------------
# Adaptive batch retry on MemoryError (base install)
# --------------------------------------------------------------------------
def test_adaptive_batch_retry_recovers_from_forced_memory_error(csv_path, sample_tabular_df, register_temp_feature):
    def fragile(s, **kwargs):
        if len(s) > 300:
            raise MemoryError("simulated allocation failure")
        return s.astype(float) + 1.0

    register_temp_feature(
        FeatureSpec(name="test.exec.fragile", family="exec", modality="tabular", input_types=("numeric",), implementation=fragile)
    )
    result = featm.extract(
        csv_path,
        features=["test.exec.fragile"],
        columns={"test.exec.fragile": "price"},
        batch_rows=2_000,
        errors="warn",
        output="memory",
    )
    assert result.shape == (len(sample_tabular_df), 1)
    assert sum(result.report.batch_reductions.values()) > 0
    assert sum(result.report.retries.values()) > 0
    expected = sample_tabular_df["price"].to_numpy() + 1.0
    assert np.allclose(result.to_pandas().iloc[:, 0].to_numpy(), expected)


def test_adaptive_retry_never_loops_forever_when_always_failing(csv_path, register_temp_feature):
    def always_memory_error(s, **kwargs):
        raise MemoryError("never works")

    register_temp_feature(
        FeatureSpec(name="test.exec.never_fits", family="exec", modality="tabular", input_types=("numeric",), implementation=always_memory_error)
    )
    result = featm.extract(
        csv_path,
        features=["test.exec.never_fits"],
        columns={"test.exec.never_fits": "price"},
        batch_rows=2_000,
        errors="warn",
        output="memory",
    )
    assert "test.exec.never_fits" in result.report.failed_features
    assert result.shape[1] == 0


# --------------------------------------------------------------------------
# Exact / approximate / auto protocol (base install)
# --------------------------------------------------------------------------
def test_exact_mode_never_approximates(csv_path):
    result = featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"},
        mode="exact", batch_rows=700, output="memory",
    )
    assert result.report.approximations_used == {}


def test_approximate_mode_reports_approximation_and_sample_size(csv_path):
    result = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore"],
        columns={"tabular.scaling.zscore": "price"},
        mode="approximate",
        batch_rows=700,
        approximate_sample_size=1_000,
        output="memory",
    )
    assert "tabular.scaling.zscore" in result.report.approximations_used
    info = result.report.approximations_used["tabular.scaling.zscore"]
    assert info["sample_size"] <= 1_000
    assert "population_size_seen" in info


def test_exact_mode_rejects_feature_without_exact_implementation():
    spec = FeatureSpec(
        name="test.exec.approx_only", family="exec", modality="tabular", input_types=("numeric",),
        exact_available=False, approximate_available=True, implementation=lambda s, **kw: (s, None),
    )
    with pytest.raises(ApproximateModeUnavailableError):
        _resolve_effective_mode(spec, "exact", None)


def test_auto_mode_prefers_exact_when_size_unknown_or_small():
    spec = FeatureSpec(
        name="test.exec.auto1", family="exec", modality="tabular", approximate_available=True, approximate_max_exact_rows=1000,
    )
    assert _resolve_effective_mode(spec, "auto", None) == "exact"
    assert _resolve_effective_mode(spec, "auto", 500) == "exact"
    assert _resolve_effective_mode(spec, "auto", 5000) == "approximate"


# --------------------------------------------------------------------------
# Legacy single-shot-fit tabular feature guard (base install)
# --------------------------------------------------------------------------
def test_legacy_incremental_feature_blocked_across_multiple_batches(csv_path):
    result = featm.extract(
        csv_path, features=["tabular.numeric.zscore"], columns={"tabular.numeric.zscore": "price"},
        batch_rows=500, errors="warn", output="memory",
    )
    assert result.report.completed_features == []
    assert "tabular.numeric.zscore" in result.report.skipped_features


def test_legacy_incremental_feature_allowed_when_single_batch_safe(_pyarrow, parquet_path, sample_tabular_df):
    big_batch = len(sample_tabular_df) + 100
    result = featm.extract(
        parquet_path, features=["tabular.numeric.zscore"], columns={"tabular.numeric.zscore": "price"},
        batch_rows=big_batch, errors="raise",
    )
    assert "tabular.numeric.zscore" in result.report.completed_features


@pytest.mark.parametrize("name", ["tabular.numeric.binning", "tabular.categorical.binary_encoding", "tabular.ordinal.cumulative"])
def test_newly_found_legacy_recompute_features_also_guarded(csv_path, name):
    """Legacy single-shot features found during the audit that were missing incremental=True."""
    spec = featm.feature_info(name)
    assert spec.incremental is True
    col = "price" if spec.input_types == ("numeric",) else "city"
    result = featm.extract(
        csv_path, features=[name], columns={name: col}, batch_rows=500, errors="warn", output="memory",
    )
    assert name in result.report.skipped_features, f"{name} should be guarded against unsafe multi-batch execution"


# --------------------------------------------------------------------------
# Item 9: min_rows / max_rows_exact enforcement
# --------------------------------------------------------------------------
def test_min_rows_enforced_when_row_count_known(_pyarrow, tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})  # only 2 points; geometry needs >= 3
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path, index=False)
    result = featm.extract(
        path, features=["geometry.triangulation.delaunay_degree"],
        columns={"geometry.triangulation.delaunay_degree": ["x", "y"]}, errors="warn",
    )
    assert "geometry.triangulation.delaunay_degree" in result.report.skipped_features


def test_min_rows_raises_with_errors_raise(_pyarrow, tmp_path):
    df = pd.DataFrame({"x": [1.0], "y": [3.0]})
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(RowCountConstraintError):
        featm.extract(
            path, features=["geometry.triangulation.delaunay_degree"],
            columns={"geometry.triangulation.delaunay_degree": ["x", "y"]}, errors="raise",
        )


# --------------------------------------------------------------------------
# Item 4: output='auto' must be safe when row count is unknown (CSV/JSONL)
# --------------------------------------------------------------------------
def test_unknown_row_count_csv_with_tiny_budget_does_not_use_unbounded_memory(_pyarrow, csv_path, sample_tabular_df):
    result = featm.extract(
        csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        memory_budget="64KB", output="auto",
    )
    assert result.report.output_backend == "disk"
    assert result.is_disk_backed
    assert result.shape == (len(sample_tabular_df), 1)


def test_unknown_row_count_jsonl_with_tiny_budget_is_also_safe(_pyarrow, jsonl_path, sample_tabular_df):
    result = featm.extract(
        jsonl_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        memory_budget="64KB", output="auto",
    )
    assert result.report.output_backend == "disk"
    assert result.shape == (len(sample_tabular_df), 1)


def test_explicit_output_memory_remains_available_as_override(csv_path):
    """Fifth audit pass, Finding 7: the exact warning WORDING changed (not the behavior) --
    the pre-fix text ("...This is an intentional override.") read as though memory_budget
    enforcement were disabled entirely for an unknown-row-count source, when it is actually
    only deferred to a per-batch check as real output accumulates (see docs/SCALING.md's
    "Output: memory- or disk-backed FeatureMatrix" section and the warning's own updated
    text). Assert on the accurate, current wording rather than the misleading pre-fix phrase."""
    result = featm.extract(
        csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        memory_budget="64KB", output="memory",
    )
    assert result.report.output_backend == "memory"
    assert any("enforced per-batch as real output actually accumulates" in w for w in result.report.warnings)


def test_auto_output_raises_when_no_disk_backend_and_unknown_row_count(csv_path, monkeypatch):
    """Base-install-safe: simulates 'no io backend available' regardless of what's actually
    installed in the test environment, so this test is meaningful with or without pyarrow."""
    from featm import execution

    monkeypatch.setattr(execution, "_io_backend_available", lambda: False)
    with pytest.raises(UnsafeOutputModeError):
        featm.extract(
            csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
            memory_budget="64KB", output="auto",
        )


# --------------------------------------------------------------------------
# Item 5: disk_budget enforcement
# --------------------------------------------------------------------------
def test_tiny_disk_budget_fails_deterministically(_pyarrow, csv_path):
    from featm.matrix import DiskBudgetExceededError

    with pytest.raises(DiskBudgetExceededError):
        featm.extract(
            csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
            batch_rows=500, output="disk", disk_budget="2KB",
        )


def test_sufficient_disk_budget_succeeds(_pyarrow, csv_path, sample_tabular_df):
    result = featm.extract(
        csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        batch_rows=500, output="disk", disk_budget="50MB",
    )
    assert result.shape == (len(sample_tabular_df), 1)


def test_disk_budget_failure_leaves_checkpoint_consistent(_pyarrow, tmp_path, csv_path):
    from featm.matrix import DiskBudgetExceededError

    ckpt = tmp_path / "ckpt"
    with pytest.raises(DiskBudgetExceededError):
        featm.extract(
            csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
            batch_rows=500, output="disk", disk_budget="2KB", checkpoint=ckpt,
        )
    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is False  # valid, consistent, honest about incompleteness

    # Resuming with a bigger budget must succeed and complete.
    result = featm.extract(
        csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        batch_rows=500, output="disk", disk_budget="50MB", checkpoint=ckpt,
    )
    assert result.report.completed_features == ["tabular.numeric.abs"]


# --------------------------------------------------------------------------
# Item 6: cache budget integrated with the overall memory budget
# --------------------------------------------------------------------------
def test_cache_budget_cannot_exceed_declared_memory_budget(csv_path):
    result = featm.extract(
        csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        memory_budget="100KB", cache_budget="50MB", output="memory",
    )
    assert any("clamped" in w for w in result.report.warnings)


def test_cache_budget_default_scales_down_with_tiny_memory_budget(csv_path):
    result = featm.extract(
        csv_path, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "price"},
        memory_budget="100KB", output="memory",
    )
    # Default cache budget must not silently sit at the flat 256MB default
    # when the whole declared working-memory budget is only 100KB.
    assert result.report.cache_stats.get("bytes", 0) <= 100 * 1024


# --------------------------------------------------------------------------
# Checkpoint / resume (item 8)
# --------------------------------------------------------------------------
def test_checkpoint_created_with_expected_contents(_pyarrow, tmp_path, csv_path):
    ckpt = tmp_path / "ckpt"
    featm.extract(
        csv_path,
        features=["tabular.scaling.zscore"],
        columns={"tabular.scaling.zscore": "price"},
        batch_rows=500,
        output="disk",
        checkpoint=ckpt,
    )
    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is True
    assert manifest["phase"] == "done"
    # Third-audit-pass Finding 3: column-label identity is now type-preserving
    # (canonical_label), not a bare str() -- "price" (a str label) encodes as
    # "str:price" so it can never collide with e.g. a same-spelled int label.
    assert manifest["settings"]["columns"] == {"tabular.scaling.zscore": ["str:price"]}
    assert (ckpt / "output" / "manifest.json").exists()
    assert (ckpt / "state").exists()


def test_checkpoint_incompatible_settings_rejected(_pyarrow, tmp_path, csv_path):
    ckpt = tmp_path / "ckpt"
    featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt,
    )
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(
            csv_path, features=["tabular.scaling.minmax"], columns={"tabular.scaling.minmax": "price"}, batch_rows=500,
            output="disk", checkpoint=ckpt,
        )


def test_checkpoint_rejects_reuse_when_only_column_mapping_changes(_pyarrow, tmp_path, csv_path):
    ckpt = tmp_path / "ckpt"
    featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt,
    )
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(
            csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "quantity"},
            batch_rows=500, output="disk", checkpoint=ckpt,
        )


def test_checkpoint_rejects_reuse_when_approximate_sample_size_changes(_pyarrow, tmp_path, csv_path):
    ckpt = tmp_path / "ckpt"
    featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt, approximate_sample_size=200_000,
    )
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(
            csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"},
            batch_rows=500, output="disk", checkpoint=ckpt, approximate_sample_size=999,
        )


def test_completed_checkpoint_reopens_with_zero_source_reads(_pyarrow, tmp_path, csv_path):
    ckpt = tmp_path / "ckpt"
    featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt,
    )
    src, reads = _counting_csv_source(csv_path)
    result = featm.extract(
        src, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt,
    )
    assert sum(reads) == 0
    assert result.report.resumed is True


def test_resume_from_transform_phase_does_not_rescan_completed_fit_phase(_pyarrow, tmp_path, csv_path, sample_tabular_df):
    # sample_tabular_df has 5000 rows; batch_rows=500 gives 10 batches per phase
    # (fit=10, transform=10). fail_after=13 lets the whole fit phase complete
    # (10 batches) then fails partway through transform (batch 4 of 10).
    orig_iter = CSVSource.iter_batches
    state = {"total": 0, "fail_after": 13}

    def flaky_iter(self, *, columns=None, batch_rows=50_000):
        for batch in orig_iter(self, columns=columns, batch_rows=batch_rows):
            state["total"] += 1
            if state["total"] > state["fail_after"]:
                raise RuntimeError("SIMULATED CRASH DURING TRANSFORM")
            yield batch

    flaky = CSVSource(csv_path)
    flaky.iter_batches = types.MethodType(flaky_iter, flaky)
    ckpt = tmp_path / "ckpt"
    with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
        featm.extract(
            flaky, features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
            columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
            batch_rows=500, output="disk", checkpoint=ckpt,
        )
    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["phase"] == "transform"  # fit already completed before the crash

    src, reads = _counting_csv_source(csv_path)
    result = featm.extract(
        src, features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=500, output="disk", checkpoint=ckpt,
    )
    # 10 batches read on resume (transform phase only, 5000 rows / 500 batch_rows)
    # -- not 20, which would mean the completed fit phase got rescanned too.
    assert len(reads) == 10
    assert sum(reads) == len(sample_tabular_df)
    assert result.shape == (len(sample_tabular_df), 2)


def test_interrupted_run_resumes_to_match_a_clean_run(_pyarrow, tmp_path, csv_path, sample_tabular_df):
    flaky = CSVSource(csv_path)
    original_iter = CSVSource.iter_batches
    call_state = {"n": 0}

    def flaky_iter(self, *, columns=None, batch_rows=50_000):
        for batch in original_iter(self, columns=columns, batch_rows=batch_rows):
            call_state["n"] += 1
            if call_state["n"] > 3:
                raise RuntimeError("SIMULATED CRASH")
            yield batch

    flaky.iter_batches = types.MethodType(flaky_iter, flaky)

    ckpt = tmp_path / "ckpt_interrupt"
    with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
        featm.extract(
            flaky,
            features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
            columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
            batch_rows=500,
            output="disk",
            checkpoint=ckpt,
        )
    manifest = json.loads((ckpt / "checkpoint.json").read_text())
    assert manifest["done"] is False

    resumed = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=500,
        output="disk",
        checkpoint=ckpt,
    )
    assert resumed.report.resumed is True

    ckpt_clean = tmp_path / "ckpt_clean"
    clean = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore", "tabular.encoding.category_frequency"],
        columns={"tabular.scaling.zscore": "price", "tabular.encoding.category_frequency": "city"},
        batch_rows=500,
        output="disk",
        checkpoint=ckpt_clean,
    )
    a = resumed.to_pandas()
    b = clean.to_pandas()
    assert a.shape == b.shape == (len(sample_tabular_df), 2)
    assert np.allclose(a.iloc[:, 0], b.iloc[:, 0])
    assert (a.iloc[:, 1].to_numpy() == b.iloc[:, 1].to_numpy()).all()


def test_resuming_a_completed_checkpoint_is_a_noop_reuse(_pyarrow, tmp_path, csv_path):
    ckpt = tmp_path / "ckpt_done"
    first = featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt,
    )
    second = featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=500,
        output="disk", checkpoint=ckpt,
    )
    assert second.report.resumed is True
    assert first.shape == second.shape


# --------------------------------------------------------------------------
# Parquet-specific (needs pyarrow via the parquet_path fixture itself)
# --------------------------------------------------------------------------
def test_parquet_extraction_matches_csv_extraction(csv_path, parquet_path):
    r_csv = featm.extract(
        csv_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=600, output="memory"
    )
    r_pq = featm.extract(
        parquet_path, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "price"}, batch_rows=600
    )
    a = r_csv.to_pandas().iloc[:, 0].to_numpy()
    b = r_pq.to_pandas().iloc[:, 0].to_numpy()
    assert np.allclose(a, b)


def test_whole_batch_nontabular_feature_over_out_of_core_source(_pyarrow, tmp_path):
    rng = np.random.default_rng(3)
    n = 2_000
    cols = {f"t{i:02d}": rng.normal(size=n) for i in range(16)}
    df = pd.DataFrame(cols)
    path = tmp_path / "series.parquet"
    df.to_parquet(path, index=False)

    result = featm.extract(path, features=["timeseries.statistical.mean"], batch_rows=333)
    assert result.shape == (n, 1)
    expected = df.to_numpy().mean(axis=1)
    assert np.allclose(result.to_pandas().iloc[:, 0].to_numpy(), expected)


def test_forced_disk_output_small_budget(_pyarrow, csv_path):
    result = featm.extract(
        csv_path,
        features=["tabular.scaling.zscore"],
        columns={"tabular.scaling.zscore": "price"},
        batch_rows=500,
        output="disk",
        memory_budget="1MB",
    )
    assert result.is_disk_backed
    assert result.report.output_backend == "disk"
