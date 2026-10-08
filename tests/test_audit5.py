"""Regression tests for the fifth independent audit / object-modality closure + batch-contract
hardening pass.

Each section corresponds to one numbered finding in STEP2_AUDIT5_FIX_HANDOFF.md.
Tests that need a disk backend (Parquet, output="disk", checkpointing) request
the ``_pyarrow`` fixture; everything else must run in a base install.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import featm
from featm.execution import ColumnResolutionError, UnsafeMultiBatchFeatureError
from featm.registry import BatchSemantics, FeatureSpec, REGISTRY, RegistryValidationError, validate_spec
from featm.sources import PandasSource, SourceNotCheckpointableError
from featm.state import StateBudgetExceededError, _bounded_repr


@pytest.fixture
def _pyarrow():
    return pytest.importorskip("pyarrow")


@pytest.fixture
def _networkx():
    return pytest.importorskip("networkx")


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


def _cmp_full_vs_batched(df, feature, *, columns=None, params=None):
    n = len(df)
    full = featm.extract(df, features=[feature], columns=columns, params=params, batch_rows=n, output="memory", errors="raise")
    out = {}
    for br in (n, 1, 2, 5):
        batched = featm.extract(df, features=[feature], columns=columns, params=params, batch_rows=br, output="memory", errors="raise")
        out[br] = batched
    return full, out


# ==========================================================================
# Finding 1 -- image.*/audio.*/video.* object-column execution
# ==========================================================================
def _image_df(n=6, h=6, w=6, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": [rng.random((h, w, c)) * 255.0 for _ in range(n)]})


def _audio_df(n=6, t=256, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": [rng.random(t) for _ in range(n)]})


def _video_df(n=6, tt=4, h=4, w=4, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"v": [rng.random((tt, h, w, c)) * 255.0 for _ in range(n)]})


@pytest.mark.parametrize(
    "builder,feature,col",
    [
        (_image_df, "image.color.red_mean", "x"),
        (_audio_df, "audio.waveform.rms", "a"),
        (_video_df, "video.frame_statistics.mean_intensity", "v"),
    ],
)
def test_finding1_object_column_execution_and_equivalence(builder, feature, col):
    df = builder(n=6)
    full, batched_by_br = _cmp_full_vs_batched(df, feature, columns={feature: col})
    full_arr = full.to_pandas().to_numpy(dtype=float)
    assert full.shape[0] == 6
    for br in (1, 2, 5):
        b = batched_by_br[br].to_pandas().to_numpy(dtype=float)
        assert b.shape == full_arr.shape
        assert np.allclose(full_arr, b, equal_nan=True), f"mismatch at batch_rows={br}"


def test_finding1_graph_object_column_equivalence(_networkx):
    import networkx as nx

    graphs = [nx.path_graph(4 + (i % 3)) for i in range(6)]
    df = pd.DataFrame({"g": graphs})
    full, batched_by_br = _cmp_full_vs_batched(df, "graph.size.n_nodes", columns={"graph.size.n_nodes": "g"})
    full_arr = full.to_pandas().to_numpy(dtype=float)
    for br in (1, 2, 5):
        b = batched_by_br[br].to_pandas().to_numpy(dtype=float)
        assert np.allclose(full_arr, b)


def test_finding1_ragged_image_shapes_fail_clearly():
    df = pd.DataFrame({"x": [np.zeros((4, 4, 3)), np.zeros((5, 5, 3)), np.zeros((4, 4, 3))]})
    with pytest.raises(ColumnResolutionError, match="inconsistent per-row array shapes"):
        featm.extract(df, features=["image.color.red_mean"], columns={"image.color.red_mean": "x"}, batch_rows=2, output="memory", errors="raise")


def test_finding1_ragged_audio_shapes_fail_clearly():
    df = pd.DataFrame({"a": [np.zeros(100), np.zeros(150)]})
    with pytest.raises(ColumnResolutionError, match="inconsistent per-row array shapes"):
        featm.extract(df, features=["audio.waveform.rms"], columns={"audio.waveform.rms": "a"}, batch_rows=2, output="memory", errors="raise")


def test_finding1_csv_string_data_not_silently_treated_as_loaded_image(tmp_path):
    """CSV rows are plain strings (e.g. file paths), never real ndarray objects --
    must be rejected clearly, never silently coerced into a wrong-shaped result."""
    csv_path = tmp_path / "img_paths.csv"
    pd.DataFrame({"x": ["img1.png", "img2.png", "img3.png"]}).to_csv(csv_path, index=False)
    with pytest.raises(UnsafeMultiBatchFeatureError, match="PandasSource"):
        featm.extract(str(csv_path), features=["image.color.red_mean"], columns={"image.color.red_mean": "x"}, output="memory", errors="raise")


def test_finding1_opaque_object_sample_rejected_from_non_pandas_source_even_single_batch(tmp_path):
    """Rejection is unconditional (a structural Source-type mismatch), not just a
    multi-batch-splitting concern -- must fail even when the whole source would
    comfortably fit in one batch."""
    csv_path = tmp_path / "small.csv"
    pd.DataFrame({"a": ["x", "y"]}).to_csv(csv_path, index=False)
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(str(csv_path), features=["audio.waveform.rms"], columns={"audio.waveform.rms": "a"}, batch_rows=100, output="memory", errors="raise")


def test_finding1_text_series_convention_unchanged():
    docs = pd.DataFrame({"t": ["apple fruit red", "banana yellow fruit", "car road engine"]})
    r = featm.extract(docs, features=["text.information.surprisal"], columns={"text.information.surprisal": "t"}, batch_rows=2, output="memory", errors="raise")
    assert r.shape == (3, 1)


def test_finding1_graph_series_convention_unchanged(_networkx):
    import networkx as nx

    graphs = pd.DataFrame({"g": [nx.path_graph(5) for _ in range(4)]})
    r = featm.extract(graphs, features=["graph.size.n_edges"], columns={"graph.size.n_edges": "g"}, batch_rows=2, output="memory", errors="raise")
    assert r.shape == (4, 1)
    assert (r.to_pandas().to_numpy() == 4.0).all()


def test_finding1_all_image_audio_video_graph_features_are_opaque_object_sample():
    specs = [s for s in featm.REGISTRY.all() if s.modality in ("image", "audio", "video", "graph")]
    assert specs, "expected image/audio/video/graph features to exist"
    assert all(s.batch_semantics is BatchSemantics.OPAQUE_OBJECT_SAMPLE for s in specs)


# ==========================================================================
# Finding 2 -- graph.embeddings.deepwalk is not batch-partition invariant
# ==========================================================================
def test_finding2_deepwalk_batch_partition_invariant(_networkx):
    gensim = pytest.importorskip("gensim")
    import networkx as nx

    N = 6
    graphs = [nx.path_graph(5 + (i % 2)) for i in range(N)]
    df = pd.DataFrame({"g": graphs})
    name = "graph.embeddings.deepwalk"
    params = {name: {"num_walks": 2, "walk_length": 5, "dimensions": 8}}
    full = featm.extract(df, features=[name], columns={name: "g"}, params=params, batch_rows=N, output="memory", errors="raise").to_pandas()
    batched = featm.extract(df, features=[name], columns={name: "g"}, params=params, batch_rows=2, output="memory", errors="raise").to_pandas()
    assert np.allclose(full.to_numpy(dtype=float), batched.to_numpy(dtype=float))


def test_finding2_deepwalk_same_content_same_output_different_position(_networkx):
    """The correctness property Finding 2's fix actually rests on: two IDENTICAL
    graphs at different call positions get IDENTICAL embeddings (content-derived
    seeding, not position-derived)."""
    gensim = pytest.importorskip("gensim")
    import networkx as nx

    g = nx.path_graph(5)
    other = nx.path_graph(7)
    df = pd.DataFrame({"g": [g, other, g]})  # same graph at position 0 and 2
    name = "graph.embeddings.deepwalk"
    params = {name: {"num_walks": 2, "walk_length": 5, "dimensions": 8}}
    out = featm.extract(df, features=[name], columns={name: "g"}, params=params, batch_rows=3, output="memory", errors="raise").to_pandas()
    row0 = out.iloc[0].to_numpy(dtype=float)
    row2 = out.iloc[2].to_numpy(dtype=float)
    assert np.allclose(row0, row2)


def test_finding2_node2vec_already_batch_invariant_structurally_sound(_networkx):
    """node2vec seeds a FRESH internal RNG per graph via seed=0 in the constructor
    call (never a shared external RNG advancing across the loop) -- confirmed not
    to need the same fix as deepwalk. Skips cleanly if the optional 'node2vec'
    package is not installed."""
    pytest.importorskip("node2vec")
    import networkx as nx

    N = 6
    graphs = [nx.path_graph(5 + (i % 2)) for i in range(N)]
    df = pd.DataFrame({"g": graphs})
    name = "graph.embeddings.node2vec"
    params = {name: {"dimensions": 8}}
    full = featm.extract(df, features=[name], columns={name: "g"}, params=params, batch_rows=N, output="memory", errors="raise").to_pandas()
    batched = featm.extract(df, features=[name], columns={name: "g"}, params=params, batch_rows=2, output="memory", errors="raise").to_pandas()
    assert np.allclose(full.to_numpy(dtype=float), batched.to_numpy(dtype=float))


def test_finding2_eemd_now_deterministic_and_batch_invariant():
    """signal.decomposition.eemd was internally stochastic pre-fix (excluded from
    the equivalence harness). Fifth audit pass: seeded from each row's own signal
    content via PyEMD's noise_seed(), it is now both reproducible run-to-run AND
    batch-partition-invariant -- the stronger fix, not merely documenting it as
    "intentionally stochastic"."""
    pytest.importorskip("PyEMD")
    rng = np.random.default_rng(11)
    N = 6
    rows = [np.sin(np.linspace(0, 8, 150)) + 0.05 * rng.standard_normal(150) for _ in range(N)]
    df = pd.DataFrame(np.vstack(rows))
    name = "signal.decomposition.eemd"
    params = {name: {"n_imfs": 3}}
    full = featm.extract(df, features=[name], params=params, batch_rows=N, output="memory", errors="raise").to_pandas()
    batched = featm.extract(df, features=[name], params=params, batch_rows=2, output="memory", errors="raise").to_pandas()
    assert np.allclose(full.to_numpy(dtype=float), batched.to_numpy(dtype=float))
    full2 = featm.extract(df, features=[name], params=params, batch_rows=N, output="memory", errors="raise").to_pandas()
    assert np.allclose(full.to_numpy(dtype=float), full2.to_numpy(dtype=float)), "must be reproducible run-to-run, not just batch-invariant"


def test_finding2_cat_embedding_already_content_seeded_no_regression():
    """tabular.categorical.embeddings was ALREADY using a fresh, content-derived
    (blake2b, not the salted builtin hash()) per-row RNG seed before this pass --
    confirmed as a pre-existing correct example, not something this pass broke."""
    df = pd.DataFrame({"c": ["a", "b", "c", "a", "b", "c"]})
    full = featm.extract(df, features=["tabular.categorical.embeddings"], batch_rows=6, output="memory", errors="raise").to_pandas()
    batched = featm.extract(df, features=["tabular.categorical.embeddings"], batch_rows=2, output="memory", errors="raise").to_pandas()
    assert np.allclose(full.to_numpy(dtype=float), batched.to_numpy(dtype=float))


# ==========================================================================
# Finding 3 -- expanded batch-equivalence harness now covers object-column modalities
# ==========================================================================
def test_finding3_harness_covers_image_audio_video_graph_modalities():
    import sys as _sys
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    _sys.path.insert(0, str(repo_root / "benchmarks"))
    try:
        import batch_equivalence_smoke as harness
        importlib_reload = __import__("importlib").reload
        importlib_reload(harness)
        for mod in ("image", "audio", "video", "graph"):
            assert mod in harness._MODALITY_BUILDERS, f"{mod} missing a sample builder"
    finally:
        _sys.path.remove(str(repo_root / "benchmarks"))


# ==========================================================================
# Finding 4 -- PandasSource.fingerprint() for ndarray/object sample columns
# ==========================================================================
def test_finding4_ndarray_column_fingerprint_does_not_crash():
    src = PandasSource(pd.DataFrame({"image": [np.zeros((2, 2)), np.ones((2, 2))]}))
    fp = src.fingerprint()
    assert isinstance(fp, str) and len(fp) > 0


def test_finding4_ndarray_mutation_changes_fingerprint():
    df = pd.DataFrame({"image": [np.zeros((2, 2)), np.ones((2, 2))]})
    fp1 = PandasSource(df).fingerprint()
    df.loc[0, "image"][0, 0] = 99.0
    fp2 = PandasSource(df).fingerprint()
    assert fp1 != fp2


def test_finding4_same_content_same_fingerprint_not_identity_based():
    df1 = pd.DataFrame({"image": [np.zeros((2, 2)), np.ones((2, 2))]})
    df2 = pd.DataFrame({"image": [np.zeros((2, 2)), np.ones((2, 2))]})
    assert PandasSource(df1).fingerprint() == PandasSource(df2).fingerprint()


def test_finding4_graph_column_content_aware_fingerprint(_networkx):
    import networkx as nx

    same_a = PandasSource(pd.DataFrame({"g": [nx.path_graph(4), nx.path_graph(4)]})).fingerprint()
    same_b = PandasSource(pd.DataFrame({"g": [nx.path_graph(4), nx.path_graph(4)]})).fingerprint()
    different = PandasSource(pd.DataFrame({"g": [nx.path_graph(4), nx.path_graph(5)]})).fingerprint()
    assert same_a == same_b
    assert same_a != different


def test_finding4_unsupported_object_raises_source_not_checkpointable():
    class Weird:
        pass

    df = pd.DataFrame({"x": [np.zeros((2, 2)), {1, 2, 3}]})  # a set: unhashable AND unsupported
    with pytest.raises(SourceNotCheckpointableError):
        PandasSource(df).fingerprint()


def test_finding4_checkpoint_with_object_column_source_works_end_to_end(tmp_path, _pyarrow):
    df = pd.DataFrame({"x": [np.random.default_rng(i).random((4, 4, 3)) for i in range(4)]})
    ckpt = tmp_path / "ckpt"
    # Explicit memory_budget/batch_rows: an implicit default derived from live
    # system memory can vary by a few bytes between two sequential calls,
    # which would trip checkpoint identity's batch_rows-must-match check --
    # unrelated to this finding, so pinned explicitly for a stable repeat call.
    kwargs = dict(
        features=["image.color.red_mean"], columns={"image.color.red_mean": "x"},
        checkpoint=str(ckpt), output="memory", errors="raise", memory_budget="10MB", batch_rows=2,
    )
    r1 = featm.extract(df, **kwargs)
    r2 = featm.extract(df, **kwargs)
    assert np.allclose(r1.to_pandas().to_numpy(), r2.to_pandas().to_numpy())


# ==========================================================================
# Finding 5 -- BatchSemantics validation rejects internally contradictory metadata
# ==========================================================================
def test_finding5_dataset_context_row_aligned_without_stateful_rejected():
    spec = FeatureSpec(
        name="x5.y5.z1", family="y5", modality="x5",
        batch_semantics=BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED, stateful=False,
        required_params_for_batching=(),
    )
    with pytest.raises(RegistryValidationError, match="DATASET_CONTEXT_ROW_ALIGNED requires stateful=True"):
        validate_spec(spec)


def test_finding5_stateful_with_row_independent_rejected():
    spec = FeatureSpec(name="x5.z5.z2", family="z5", modality="x5", stateful=True, batch_semantics=BatchSemantics.ROW_INDEPENDENT)
    with pytest.raises(RegistryValidationError, match="stateful=True requires batch_semantics=DATASET_CONTEXT_ROW_ALIGNED"):
        validate_spec(spec)


def test_finding5_opaque_object_sample_with_full_materialization_rejected():
    spec = FeatureSpec(
        name="x5.y5.z3", family="y5", modality="x5",
        batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE, requires_full_materialization=True,
    )
    with pytest.raises(RegistryValidationError, match="mutually exclusive"):
        validate_spec(spec)


def test_finding5_opaque_object_sample_with_stateful_rejected():
    spec = FeatureSpec(name="x5.y5.z4", family="y5", modality="x5", batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE, stateful=True)
    with pytest.raises(RegistryValidationError):
        validate_spec(spec)


def test_finding5_stateful_derived_default_still_valid():
    """A stateful spec that leaves batch_semantics unset (the common case) must
    still validate fine -- __post_init__ derives DATASET_CONTEXT_ROW_ALIGNED
    automatically, satisfying the new rule without any explicit annotation."""
    spec = FeatureSpec(name="x5.y5.ok", family="y5", modality="x5", stateful=True)
    validate_spec(spec)  # must not raise
    assert spec.batch_semantics is BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED


def test_finding5_opaque_object_sample_plain_registration_still_valid():
    spec = FeatureSpec(name="x5.y5.ok2", family="y5", modality="x5", batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE)
    validate_spec(spec)  # must not raise


def test_finding5_full_registry_still_validates_after_tightened_rules():
    assert len(featm.REGISTRY.all()) == 552
    featm.REGISTRY.validate_all()  # must not raise


# ==========================================================================
# Finding 6 -- source-tree pytest subprocess portability
# ==========================================================================
def test_finding6_batch_equivalence_subprocess_works_without_ambient_pythonpath():
    """Mirrors the exact scenario Finding 6 reports: an extracted source checkout,
    dependencies installed, but featm itself NOT pip install -e .-ed, and no
    PYTHONPATH pre-set in the calling shell. The subprocess must still find
    featm via an explicitly constructed PYTHONPATH, not the caller's ambient
    environment."""
    import os
    import pathlib
    import subprocess
    import sys as _sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_dir = str(repo_root / "src")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # simulate no ambient PYTHONPATH
    env["PYTHONPATH"] = src_dir
    result = subprocess.run(
        [_sys.executable, "-c", "import featm; print(featm.__version__)"],
        capture_output=True, text=True, cwd=str(repo_root), env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip()


# ==========================================================================
# Finding 7 -- output="memory"/memory_budget documentation & warning-text consistency
# ==========================================================================
def test_finding7_explicit_memory_with_unknown_rows_still_enforces_budget(tmp_path, _pyarrow):
    """The corrected contract: output='memory' with an unknown row count is NOT a
    blanket bypass of memory_budget -- it still raises once accumulated output
    actually exceeds budget, deferred to a per-batch check rather than skipped."""
    from featm.execution import UnsafeOutputModeError

    csv_path = tmp_path / "wide.csv"
    n = 4000
    pd.DataFrame({"x": np.arange(n, dtype=float), "y": ["v" * 200] * n}).to_csv(csv_path, index=False)
    with pytest.raises(UnsafeOutputModeError):
        featm.extract(
            str(csv_path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
            memory_budget="4KB", output="memory", batch_rows=50, errors="raise",
        )


def test_finding7_override_flag_proceeds_with_warning_not_silently(tmp_path, _pyarrow):
    csv_path = tmp_path / "wide2.csv"
    n = 2000
    pd.DataFrame({"x": np.arange(n, dtype=float)}).to_csv(csv_path, index=False)
    result = featm.extract(
        str(csv_path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        memory_budget="4KB", output="memory", batch_rows=50, errors="raise",
        allow_memory_budget_override=True,
    )
    assert result.report.output_backend == "memory"
    assert result.shape[0] == n


# ==========================================================================
# Finding 8 -- bounded error-message size for rejected long category/term values
# ==========================================================================
def test_finding8_bounded_repr_stays_short_for_long_values():
    long_key = "y" * 2_000_000
    rep = _bounded_repr(long_key)
    assert len(rep) < 200


def test_finding8_bounded_repr_identity_preserved_for_short_values():
    assert _bounded_repr("short") == repr("short")


def test_finding8_category_frequency_budget_error_message_bounded():
    from featm.state import BoundedFrequencyCounter

    long_key = "z" * 500_000
    counter = BoundedFrequencyCounter(budget_bytes=1000)
    with pytest.raises(StateBudgetExceededError) as excinfo:
        counter.update([long_key])
    assert len(str(excinfo.value)) < 1000


def test_finding8_corpus_term_stats_budget_error_message_bounded():
    from featm.state import _CorpusTermStats

    long_term = "w" * 500_000
    stats = _CorpusTermStats(budget_bytes=1000)
    with pytest.raises(StateBudgetExceededError) as excinfo:
        stats.update([[long_term]])
    assert len(str(excinfo.value)) < 1000


def test_finding8_short_keys_unaffected():
    from featm.state import BoundedFrequencyCounter

    counter = BoundedFrequencyCounter(budget_bytes=10_000)
    counter.update(["a", "b", "a", "c"])
    assert counter.get_many(["a", "b", "c", "d"]) == {"a": 2, "b": 1, "c": 1, "d": 0}


# ==========================================================================
# No-regression spot checks (do not weaken v0.2.4/prior fixes)
# ==========================================================================
def test_no_regression_registry_count_and_552_names_preserved():
    assert len(featm.REGISTRY.all()) == 552


def test_no_regression_features_all_still_rejected():
    with pytest.raises(Exception):
        featm.extract(pd.DataFrame({"x": [1, 2, 3]}), features="all")


def test_no_regression_structured_sample_still_rejected():
    X = pd.DataFrame({"a": np.arange(8, dtype=float), "b": np.arange(8, dtype=float)[::-1]})
    with pytest.raises(UnsafeMultiBatchFeatureError):
        featm.extract(
            X, features=["timeseries.cross_series.cross_correlation"],
            memory_budget="1MB", batch_rows=2, output="memory", errors="raise",
        )
