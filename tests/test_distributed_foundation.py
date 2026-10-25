from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import featm
from featm.distributed.contracts import DistributedFeatureUnsupportedError, DistributedRuntimeUnavailableError, Partition
from featm.distributed.execution import _ordered_results
from featm.distributed.local import LocalThreadRuntime
from featm.state import STATEFUL_ENCODERS


def _pdf(matrix):
    return matrix.to_pandas(allow_large=True)


def test_distributed_direct_matches_local_and_preserves_order():
    x = pd.DataFrame({"x": np.arange(37, dtype=float) - 10})
    kwargs = dict(features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, memory_budget="10MB", output="memory", errors="raise")
    local = featm.extract(x, batch_rows=5, **kwargs)
    dist = featm.extract(
        x, execution="distributed", distributed_backend="local", distributed_workers=4,
        distributed_partition_rows=5, **kwargs
    )
    pd.testing.assert_frame_equal(_pdf(dist), _pdf(local))
    assert len(dist) == len(x)
    assert dist.report.distributed_report["transform_partitions_computed_this_run"] == 8


def test_distributed_zscore_fit_merge_matches_local_two_pass():
    x = pd.DataFrame({"x": np.linspace(-20, 100, 61)})
    kwargs = dict(features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, memory_budget="10MB", output="memory", errors="raise")
    local = featm.extract(x, batch_rows=7, **kwargs)
    dist = featm.extract(
        x, execution="distributed", distributed_backend="local", distributed_workers=3,
        distributed_partition_rows=7, **kwargs
    )
    np.testing.assert_allclose(dist.to_numpy(), local.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
    assert dist.report.distributed_report["fit_partitions_computed_this_run"] == 9
    assert dist.report.distributed_report["transform_partitions_computed_this_run"] == 9


def test_distributed_tfidf_global_vocabulary_matches_local():
    docs = pd.DataFrame({"doc": [
        "alpha beta", "alpha gamma", "delta beta", "epsilon alpha", "gamma delta",
        "beta epsilon", "alpha alpha", "zeta beta", "eta theta", "alpha theta",
    ]})
    params = {"text.tfidf.tfidf": {"max_features": 16}}
    kwargs = dict(features=["text.tfidf.tfidf"], columns={"text.tfidf.tfidf": "doc"}, params=params, memory_budget="10MB", output="memory", errors="raise")
    local = featm.extract(docs, batch_rows=3, **kwargs)
    dist = featm.extract(docs, execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=3, **kwargs)
    assert dist.columns == local.columns
    np.testing.assert_allclose(dist.to_numpy(), local.to_numpy(), rtol=1e-12, atol=1e-12)


def test_distributed_similarity_preserves_first_document_in_source_order():
    docs = pd.DataFrame({"doc": ["THE FIRST DOCUMENT", "other text", "third text", "last text"]})
    kwargs = dict(features=["text.semantics.similarity"], columns={"text.semantics.similarity": "doc"}, memory_budget="10MB", output="memory", errors="raise")
    local = featm.extract(docs, batch_rows=2, **kwargs)
    dist = featm.extract(docs, execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=2, **kwargs)
    np.testing.assert_allclose(dist.to_numpy(), local.to_numpy(), rtol=1e-12, atol=1e-12)


def test_all_current_stateful_encoders_have_distributed_merge_equivalence():
    cases = {
        "tabular.scaling.zscore": (pd.Series([0., 10., 20., 30.]), {}),
        "tabular.scaling.minmax": (pd.Series([0., 10., 20., 30.]), {}),
        "tabular.encoding.category_frequency": (pd.Series(["a", "b", "a", "c"]), {}),
        "tabular.missingness.entropy": (pd.Series([1.0, float("nan"), None, 3.0]), {}),
        "text.information.surprisal": (pd.Series(["a b", "a c", "d e", "a d"]), {}),
        "text.bm25.bm25": (pd.Series(["a b", "a c", "d e", "a d"]), {"query": "a"}),
        "text.tfidf.tfidf": (pd.Series(["a b", "a c", "d e", "a d"]), {"max_features": 8}),
        "text.semantics.similarity": (pd.Series(["first doc", "other text", "third doc", "last"]), {}),
    }
    for name, (series, params) in cases.items():
        enc = STATEFUL_ENCODERS[name]
        assert enc.distributed_merge_supported, name
        full = enc.update_state(enc.init_state(params=params), series, params=params)
        left = enc.update_state(enc.init_state(params=params), series.iloc[:2], params=params)
        right = enc.update_state(enc.init_state(params=params), series.iloc[2:], params=params)
        merged = enc.merge_states(left, right, params=params)
        full_f = enc.finalize_state(full, params=params)
        merged_f = enc.finalize_state(merged, params=params)
        a = enc.transform_batch(series, full_f, params=params)
        b = enc.transform_batch(series, merged_f, params=params)
        aa = a.as_array() if hasattr(a, "as_array") else np.asarray(a)
        bb = b.as_array() if hasattr(b, "as_array") else np.asarray(b)
        np.testing.assert_allclose(aa, bb, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)


def test_required_reference_is_rejected_when_distribution_would_partition():
    x = pd.DataFrame({"s": ["aaaa", "aaab", "bbbb", "bbba"]})
    with pytest.raises(Exception, match="requires explicit param"):
        featm.extract(
            x, features=["sequence.distance.edit_distance"], columns={"sequence.distance.edit_distance": "s"},
            memory_budget="10MB", execution="distributed", distributed_backend="local",
            distributed_partition_rows=2, errors="raise", output="memory"
        )


def test_required_reference_becomes_safe_when_explicit():
    x = pd.DataFrame({"s": ["aaaa", "aaab", "bbbb", "bbba"]})
    params = {"sequence.distance.edit_distance": {"reference": "aaaa"}}
    dist = featm.extract(
        x, features=["sequence.distance.edit_distance"], columns={"sequence.distance.edit_distance": "s"}, params=params,
        memory_budget="10MB", execution="distributed", distributed_backend="local",
        distributed_partition_rows=2, errors="raise", output="memory"
    )
    assert list(dist.to_numpy().ravel()) == [0, 1, 4, 3]


def test_full_dataset_feature_rejected_in_distributed_mode():
    x = pd.DataFrame({"doc": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    with pytest.raises(DistributedFeatureUnsupportedError, match="whole-dataset"):
        featm.extract(
            x, features=["text.semantics.topic"], columns={"text.semantics.topic": "doc"},
            memory_budget="100MB", execution="distributed", distributed_backend="local",
            distributed_partition_rows=2, errors="raise", output="memory"
        )


def test_distributed_approximate_stateful_fit_is_rejected_not_naively_merged():
    x = pd.DataFrame({"x": np.arange(30, dtype=float)})
    with pytest.raises(DistributedFeatureUnsupportedError, match="supports state merging in exact mode only"):
        featm.extract(
            x, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"},
            mode="approximate", memory_budget="10MB", execution="distributed", distributed_backend="local",
            distributed_partition_rows=5, errors="raise", output="memory"
        )


def test_opaque_object_image_feature_can_partition_from_pandas_source():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"img": [rng.random((8, 8, 3)) for _ in range(7)]})
    kwargs = dict(features=["image.color.red_mean"], columns={"image.color.red_mean": "img"}, memory_budget="20MB", output="memory", errors="raise")
    local = featm.extract(x, batch_rows=2, **kwargs)
    dist = featm.extract(x, execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=2, **kwargs)
    np.testing.assert_allclose(dist.to_numpy(), local.to_numpy(), rtol=0, atol=0)


def test_ordered_result_buffer_reassembles_scheduler_out_of_order_results():
    stream = [(2, "c"), (0, "a"), (3, "d"), (1, "b")]
    assert list(_ordered_results(stream)) == ["a", "b", "c", "d"]


def test_local_runtime_retries_same_partition_idempotently():
    runtime = LocalThreadRuntime(n_workers=2)
    attempts = {}
    def worker(part):
        attempts[part.index] = attempts.get(part.index, 0) + 1
        if attempts[part.index] == 1:
            raise RuntimeError("transient")
        return part.index * 10
    try:
        parts = [Partition(i, pd.DataFrame({"x": [i]})) for i in range(4)]
        got = dict(runtime.map_stream(worker, parts, max_in_flight=2, retries=1))
    finally:
        runtime.close()
    assert got == {0: 0, 1: 10, 2: 20, 3: 30}
    assert attempts == {0: 2, 1: 2, 2: 2, 3: 2}


def test_memory_output_budget_is_not_silently_ignored():
    x = pd.DataFrame({"x": np.arange(2000, dtype=float)})
    with pytest.raises(MemoryError, match="distributed output='memory'"):
        featm.extract(
            x, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
            memory_budget="1KB", execution="distributed", distributed_backend="local",
            distributed_partition_rows=50, errors="raise", output="memory"
        )


def test_dask_request_fails_cleanly_when_optional_runtime_missing():
    try:
        import distributed  # noqa: F401
    except ImportError:
        x = pd.DataFrame({"x": [1., 2.]})
        with pytest.raises(DistributedRuntimeUnavailableError, match="featm\\[distributed\\]"):
            featm.extract(
                x, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
                memory_budget="10MB", execution="distributed", distributed_backend="dask",
                distributed_partition_rows=1, errors="raise", output="memory"
            )
    else:
        pytest.skip("Dask is installed in this environment; real adapter validation belongs to the completion pass")


def test_distributed_checkpoint_completes_and_matches_uncheckpointed_run(tmp_path):
    """Step 4 completion pass: distributed checkpoint/resume is implemented (see
    tests/test_distributed_checkpoint.py for the exhaustive crash-window/resume
    coverage) -- this is just a foundation-level smoke test that it does not
    raise and produces the same result as an uncheckpointed run, using the
    dask-independent local reference runtime like the rest of this file."""
    rng = np.random.default_rng(11)
    x = pd.DataFrame({"v": rng.standard_normal(40)})
    plain = featm.extract(
        x, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "v"},
        execution="distributed", distributed_backend="local", distributed_partition_rows=7,
    )
    checkpointed = featm.extract(
        x, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "v"},
        execution="distributed", distributed_backend="local", distributed_partition_rows=7,
        checkpoint=str(tmp_path / "ckpt"),
    )
    assert np.allclose(plain.to_pandas().to_numpy(), checkpointed.to_pandas().to_numpy())
    assert checkpointed.report.distributed_report["checkpoint_enabled"] is True


def test_rejected_feature_isolated_under_skip_while_valid_feature_runs():
    x = pd.DataFrame({"doc": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    result = featm.extract(
        x,
        features=["text.lexical.avg_word_length", "text.semantics.topic"],
        columns={"text.lexical.avg_word_length": "doc", "text.semantics.topic": "doc"},
        memory_budget="100MB",
        execution="distributed",
        distributed_backend="local",
        distributed_partition_rows=2,
        errors="skip",
        output="memory",
    )
    assert result.shape == (4, 1)
    assert "text.lexical.avg_word_length" in result.report.completed_features
    assert result.report.skipped_features.count("text.semantics.topic") == 1


def test_all_rejected_under_skip_returns_empty_matrix_with_report_not_crash():
    x = pd.DataFrame({"doc": ["alpha beta", "gamma delta"]})
    result = featm.extract(
        x,
        features=["text.semantics.topic"],
        columns={"text.semantics.topic": "doc"},
        memory_budget="100MB",
        execution="distributed",
        distributed_backend="local",
        distributed_partition_rows=1,
        errors="skip",
        output="memory",
    )
    assert result.shape == (0, 0)
    assert result.report.skipped_features == ["text.semantics.topic"]


def test_partition_feature_failure_isolated_and_filled_with_nan(register_temp_feature):
    from featm.registry import FeatureSpec, Cost
    name = "test.distributed.fail_on_later_partition"
    def fn(x):
        a = np.asarray(x, dtype=float).reshape(-1)
        if np.nanmax(a) >= 4:
            raise RuntimeError("later partition failure")
        return a * 2
    register_temp_feature(FeatureSpec(
        name=name, family="distributed_test", modality="timeseries",
        input_types=("numeric",), cost=Cost.LOW, implementation=fn,
    ))
    x = pd.DataFrame({"x": np.arange(6, dtype=float)})
    out = featm.extract(
        x, features=[name], columns={name: "x"}, memory_budget="10MB",
        execution="distributed", distributed_backend="local", distributed_workers=2,
        distributed_partition_rows=2, errors="skip", output="memory",
    )
    arr = out.to_numpy().ravel()
    np.testing.assert_array_equal(arr[:4], np.array([0., 2., 4., 6.]))
    assert np.isnan(arr[4:]).all()
    assert out.report.distributed_report["degraded_partitions_by_feature"][name] == 1
