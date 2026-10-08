"""Step 3 audit-fix pass local tests: GPU memory budget planning/estimation/cache-budget derivation.

Pure unit tests against featm.gpu.memory / featm.gpu.cache -- no fake-GPU
integration needed here (that lives in tests/test_gpu_dispatch.py, which
proves these functions are actually CALLED by production dispatch, not just
correct in isolation).
"""
from __future__ import annotations

import pytest

from featm.gpu import memory as gpu_memory
from featm.gpu.cache import gpu_cache_budget_bytes, _DEFAULT_GPU_CACHE_BUDGET_BYTES
from featm.registry import FeatureSpec


def _spec(**kwargs) -> FeatureSpec:
    kwargs.setdefault("name", "x.y.z")
    kwargs.setdefault("family", "y")
    kwargs.setdefault("modality", "x")
    return FeatureSpec(**kwargs)


# ==========================================================================
# estimate_gpu_bytes_per_row: family-aware, not one universal multiplier.
# ==========================================================================
def test_spectral_feature_estimate_exceeds_reduction_estimate_for_same_width():
    reduction_spec = _spec(name="timeseries.statistical.mean")
    spectral_spec = _spec(name="timeseries.spectral.centroid")
    reduction_bytes = gpu_memory.estimate_gpu_bytes_per_row(reduction_spec, n_columns_in=64, itemsize=4)
    spectral_bytes = gpu_memory.estimate_gpu_bytes_per_row(spectral_spec, n_columns_in=64, itemsize=4)
    assert spectral_bytes > reduction_bytes, (
        "a shared-FFT spectral kernel coexists with several more device-resident arrays per row "
        "than a plain reduction -- its estimate must not be the same universal multiplier"
    )


def test_estimate_scales_with_column_count_and_itemsize():
    spec = _spec(name="timeseries.statistical.mean")
    small = gpu_memory.estimate_gpu_bytes_per_row(spec, n_columns_in=16, itemsize=4)
    large = gpu_memory.estimate_gpu_bytes_per_row(spec, n_columns_in=64, itemsize=4)
    assert large > small
    f32 = gpu_memory.estimate_gpu_bytes_per_row(spec, n_columns_in=64, itemsize=4)
    f64 = gpu_memory.estimate_gpu_bytes_per_row(spec, n_columns_in=64, itemsize=8)
    assert f64 > f32


# ==========================================================================
# plan_gpu_batch_rows: the function Finding 1 found declared-but-never-called.
# ==========================================================================
def test_plan_gpu_batch_rows_returns_n_rows_when_budget_unavailable():
    budget = gpu_memory.GPUMemoryBudget.unavailable()
    spec = _spec(name="timeseries.statistical.mean")
    assert gpu_memory.plan_gpu_batch_rows(spec=spec, n_rows=1000, n_columns_in=64, budget=budget) == 1000


def test_plan_gpu_batch_rows_returns_zero_when_even_one_row_does_not_fit():
    spec = _spec(name="timeseries.statistical.mean")
    budget = gpu_memory.GPUMemoryBudget(
        backend="cupy", device_index=0, budget_bytes=1, total_bytes=16 * 1024**3,
        free_bytes_at_plan=8 * 1024**3, target_fraction=0.5,
    )
    assert gpu_memory.plan_gpu_batch_rows(spec=spec, n_rows=1000, n_columns_in=256, budget=budget) == 0


def test_plan_gpu_batch_rows_chunks_when_budget_fits_some_but_not_all_rows():
    spec = _spec(name="timeseries.statistical.mean")
    bytes_per_row = gpu_memory.estimate_gpu_bytes_per_row(spec, n_columns_in=64, itemsize=4)
    budget = gpu_memory.GPUMemoryBudget(
        backend="cupy", device_index=0, budget_bytes=bytes_per_row * 10, total_bytes=16 * 1024**3,
        free_bytes_at_plan=8 * 1024**3, target_fraction=0.5,
    )
    planned = gpu_memory.plan_gpu_batch_rows(spec=spec, n_rows=1000, n_columns_in=64, budget=budget)
    assert 0 < planned < 1000


def test_plan_gpu_batch_rows_never_exceeds_n_rows_even_with_huge_budget():
    spec = _spec(name="timeseries.statistical.mean")
    budget = gpu_memory.GPUMemoryBudget(
        backend="cupy", device_index=0, budget_bytes=100 * 1024**3, total_bytes=100 * 1024**3,
        free_bytes_at_plan=100 * 1024**3, target_fraction=0.5,
    )
    assert gpu_memory.plan_gpu_batch_rows(spec=spec, n_rows=50, n_columns_in=64, budget=budget) == 50


# ==========================================================================
# estimate_gpu_min_elements: Finding 10's workload-aware auto threshold.
# ==========================================================================
def test_estimate_gpu_min_elements_derives_from_gpu_min_rows_and_calibration_width():
    spec = _spec(name="timeseries.statistical.mean", gpu_min_rows=8_000)
    expected = 8_000 * gpu_memory.GPU_MIN_ROWS_CALIBRATION_SEQ_LEN
    assert gpu_memory.estimate_gpu_min_elements(spec) == expected


def test_estimate_gpu_min_elements_uses_explicit_override_when_set():
    spec = _spec(name="timeseries.statistical.mean", gpu_min_rows=8_000, gpu_min_elements=12345)
    assert gpu_memory.estimate_gpu_min_elements(spec) == 12345


def test_estimate_gpu_min_elements_falls_back_to_default_min_rows_when_unset():
    spec = _spec(name="timeseries.statistical.mean")  # gpu_min_rows=None
    expected = gpu_memory.DEFAULT_GPU_MIN_ROWS * gpu_memory.GPU_MIN_ROWS_CALIBRATION_SEQ_LEN
    assert gpu_memory.estimate_gpu_min_elements(spec) == expected


# ==========================================================================
# gpu_min_elements registry validation.
# ==========================================================================
def test_gpu_min_elements_must_be_non_negative():
    from featm.registry import RegistryValidationError, validate_spec

    spec = _spec(name="x.y.negmin", gpu_min_elements=-1)
    with pytest.raises(RegistryValidationError, match="gpu_min_elements"):
        validate_spec(spec)


def test_gpu_min_elements_none_and_zero_are_valid():
    from featm.registry import validate_spec

    validate_spec(_spec(name="x.y.noneval", gpu_min_elements=None))
    validate_spec(_spec(name="x.y.zeroval", gpu_min_elements=0))


# ==========================================================================
# GPU cache budget derivation: Finding 1's "cache can never exceed the
# declared GPU budget" requirement.
# ==========================================================================
def test_gpu_cache_budget_bytes_defaults_when_no_budget_resolved():
    assert gpu_cache_budget_bytes(None) == _DEFAULT_GPU_CACHE_BUDGET_BYTES


def test_gpu_cache_budget_bytes_never_exceeds_declared_gpu_budget():
    tiny_budget = gpu_memory.GPUMemoryBudget(
        backend="cupy", device_index=0, budget_bytes=1024, total_bytes=16 * 1024**3,
        free_bytes_at_plan=8 * 1024**3, target_fraction=0.5,
    )
    cap = gpu_cache_budget_bytes(tiny_budget)
    assert cap <= tiny_budget.budget_bytes
    assert cap <= _DEFAULT_GPU_CACHE_BUDGET_BYTES


def test_gpu_cache_budget_bytes_uses_default_when_budget_larger_than_default():
    huge_budget = gpu_memory.GPUMemoryBudget(
        backend="cupy", device_index=0, budget_bytes=100 * 1024**3, total_bytes=100 * 1024**3,
        free_bytes_at_plan=100 * 1024**3, target_fraction=0.5,
    )
    assert gpu_cache_budget_bytes(huge_budget) == _DEFAULT_GPU_CACHE_BUDGET_BYTES


# ==========================================================================
# Finding 9: GPU cache LRU eviction evicts only enough entries, not everything.
# ==========================================================================
def test_cache_eviction_evicts_only_enough_lru_entries_not_everything():
    from featm.gpu.cache import GPUIntermediateCache

    cache = GPUIntermediateCache(max_bytes=250)

    class _Arr:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    cache.get_or_compute(("oldest",), lambda: _Arr(100))
    cache.get_or_compute(("middle",), lambda: _Arr(100))
    # Adding a third 100-byte entry pushes total to 300 > 250 -- only the
    # OLDEST entry should be evicted (dropping to 200, under budget), not
    # every entry (the pre-fix bug: self._bytes never decremented in the
    # eviction loop, so the loop condition stayed true until the store was
    # completely empty).
    cache.get_or_compute(("newest",), lambda: _Arr(100))

    assert cache.entry_count == 2, f"expected exactly 1 eviction (oldest only), got {cache.entry_count} entries remaining"
    assert cache.evictions == 1
    assert cache.size_bytes == 200
    # The newer two entries must still be cache HITS (not recomputed).
    calls = {"n": 0}

    def _should_not_be_called():
        calls["n"] += 1
        return _Arr(100)

    cache.get_or_compute(("middle",), _should_not_be_called)
    cache.get_or_compute(("newest",), _should_not_be_called)
    assert calls["n"] == 0, "middle/newest entries should still be cached, not recomputed"


def test_cache_bytes_accounting_stays_correct_after_repeated_eviction():
    from featm.gpu.cache import GPUIntermediateCache

    cache = GPUIntermediateCache(max_bytes=150)

    class _Arr:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    for i in range(10):
        cache.get_or_compute((f"k{i}",), lambda: _Arr(50))
    # Only ever 3 entries (150 bytes) can coexist; size_bytes must reflect
    # exactly what remains, never drift from a broken eviction loop.
    assert cache.size_bytes <= 150
    assert cache.size_bytes == sum(50 for _ in range(cache.entry_count))
