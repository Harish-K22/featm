from __future__ import annotations

import numpy as np
import pytest

import featm
from featm.cache import SharedIntermediateCache, array_cache_key, current_cache, scoped_cache


def test_cache_none_outside_scope():
    assert current_cache() is None


def test_get_or_compute_caches_within_scope():
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return np.arange(10)

    with scoped_cache() as cache:
        a = cache.get_or_compute(("k",), compute)
        b = cache.get_or_compute(("k",), compute)
    assert calls["n"] == 1
    assert np.array_equal(a, b)


def test_fresh_scope_per_context_manager():
    with scoped_cache() as cache1:
        cache1.get_or_compute(("k",), lambda: 1)
        assert cache1.hits == 0 and cache1.misses == 1
    with scoped_cache() as cache2:
        assert cache2 is not cache1
        assert current_cache() is cache2


def test_no_leakage_across_scopes_even_with_same_key():
    with scoped_cache() as cache1:
        cache1.get_or_compute(("k",), lambda: "scope1-value")
    with scoped_cache() as cache2:
        # same key, different (fresh) scope: must recompute, not see scope1's value
        val = cache2.get_or_compute(("k",), lambda: "scope2-value")
    assert val == "scope2-value"


def test_lru_eviction_under_budget():
    cache = SharedIntermediateCache(max_bytes=100)

    def make(n):
        return lambda: np.zeros(n, dtype=np.uint8)

    cache.get_or_compute(("a",), make(60))
    cache.get_or_compute(("b",), make(60))  # forces eviction of "a" under the 100-byte budget
    assert cache.evictions >= 1
    assert cache.size_bytes <= 160  # bounded, not unbounded growth


def test_array_cache_key_content_addressed():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 4.0])
    assert array_cache_key("ns", a) == array_cache_key("ns", b)
    assert array_cache_key("ns", a) != array_cache_key("ns", c)


def test_spectral_parts_shared_across_selected_features():
    """timeseries.spectral.* features sharing one FFT computed once per extract() call."""
    from featm.features import _utils

    calls = {"n": 0}
    original = _utils._spectral_parts_impl

    def counting(a):
        calls["n"] += 1
        return original(a)

    _utils._spectral_parts_impl = counting
    try:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 64))
        featm.extract(
            x,
            features=[
                "timeseries.spectral.centroid",
                "timeseries.spectral.dominant_frequency",
                "timeseries.spectral.flatness",
                "timeseries.spectral.fft",
                "timeseries.spectral.periodogram",
                "timeseries.spectral.bandpower",
                "timeseries.spectral.cepstrum",
            ],
        )
        assert calls["n"] == 1
    finally:
        _utils._spectral_parts_impl = original


def test_delaunay_shared_across_selected_features():
    from featm.features.geometry import triangulation as tri_mod

    calls = {"n": 0}
    original = tri_mod._tri_data_impl

    def counting(pts):
        calls["n"] += 1
        return original(pts)

    tri_mod._tri_data_impl = counting
    try:
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
        featm.extract(
            pts,
            features=[
                "geometry.triangulation.delaunay_degree",
                "geometry.triangulation.mean_edge_length",
                "geometry.triangulation.circumradius",
                "geometry.triangulation.simplex_quality",
            ],
        )
        assert calls["n"] == 1
    finally:
        tri_mod._tri_data_impl = original


def test_graph_laplacian_shared_across_selected_features():
    nx = pytest.importorskip("networkx")
    from featm.features.graph import full as graph_mod

    calls = {"n": 0}
    original = graph_mod._lap_eigs_full_impl

    def counting(g):
        calls["n"] += 1
        return original(g)

    graph_mod._lap_eigs_full_impl = counting
    try:
        graphs = [nx.watts_strogatz_graph(14, 4, 0.2, seed=1)]
        featm.extract(
            graphs,
            features=["graph.spectral.eigenvalues", "graph.spectral.algebraic_connectivity", "graph.spectral.laplacian"],
        )
        assert calls["n"] == 1
    finally:
        graph_mod._lap_eigs_full_impl = original


def test_cache_key_metadata_matches_actual_sharing():
    specs = {s.name: s for s in featm.REGISTRY.search(modality="timeseries", family="spectral")}
    assert specs["timeseries.spectral.centroid"].cache_key == "timeseries.spectral_parts"
    assert specs["timeseries.spectral.fft"].cache_key == "timeseries.spectral_parts"
    # psd uses scipy.signal.welch directly, not the shared FFT -- must not claim it does
    assert specs["timeseries.spectral.psd"].cache_key is None
