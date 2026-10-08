"""Step 3 local tests: GPU dispatch control flow using a FAKE CuPy backend.

None of this exercises real CUDA numerics -- the fake backend below wraps
NumPy so every "GPU" computation is actually run on the CPU under a fake
module named ``cupy``. This is enough to test everything that is pure
CONTROL FLOW (which backend was chosen and why, OOM-retry halving and its
termination, split-safety enforcement, shared-cache call counting,
fallback-reason reporting) without claiming any of it validates real GPU
numerics -- that validation is Kaggle-only (see
notebooks/FEATM_STEP3_GPU_VALIDATION.ipynb).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import featm
from featm.gpu import backend as gpu_backend
from featm.gpu import dispatch as gpu_dispatch
from featm.gpu.cache import GPUIntermediateCache
from featm.gpu.exceptions import GPUOutOfMemoryError, GPUUnavailableError
from featm.gpu import memory as gpu_memory
from featm.registry import BatchSemantics, FeatureSpec, REGISTRY


class _FakeOutOfMemoryError(Exception):
    pass


class _FakeMemoryPool:
    def free_all_blocks(self):
        pass


class _FakeRuntime:
    @staticmethod
    def getDeviceCount():
        return 1

    @staticmethod
    def memGetInfo():
        return (8 * 1024**3, 16 * 1024**3)

    @staticmethod
    def getDeviceProperties(index):
        return {"name": b"Fake GPU"}

    @staticmethod
    def runtimeGetVersion():
        return 12000


class _FakeMemoryModule:
    OutOfMemoryError = _FakeOutOfMemoryError


class _FakeCudaModule:
    memory = _FakeMemoryModule
    runtime = _FakeRuntime

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False


class _FakeCupyModule:
    """Wraps NumPy under the name 'cupy' -- see module docstring for scope/limits."""

    __name__ = "cupy"
    cuda = _FakeCudaModule
    float32 = np.float32
    fft = np.fft

    #: When set, asarray() raises _FakeOutOfMemoryError for any batch with
    #: more rows than this -- the single injection point used to simulate a
    #: CUDA OOM at a controlled batch size.
    oom_above_rows: int | None = None
    oom_call_count: int = 0

    def asarray(self, a, dtype=None):
        arr = np.asarray(a, dtype=dtype)
        if self.oom_above_rows is not None and arr.ndim >= 1 and arr.shape[0] > self.oom_above_rows:
            self.oom_call_count += 1
            raise _FakeOutOfMemoryError(f"fake OOM: {arr.shape[0]} rows > {self.oom_above_rows}")
        return arr

    def asnumpy(self, a):
        return np.asarray(a)

    def get_default_memory_pool(self):
        return _FakeMemoryPool()

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Install a fake 'cupy' module and make featm.gpu.backend report it as available."""
    fake = _FakeCupyModule()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: True)
    monkeypatch.setattr(gpu_backend, "preferred_backend", lambda: "cupy")
    monkeypatch.setattr(gpu_backend, "require_gpu", lambda backend=None: "cupy")
    monkeypatch.setattr(gpu_dispatch.gpu_backend, "require_gpu", lambda backend=None: "cupy")
    monkeypatch.setattr(gpu_dispatch.gpu_backend, "xp_module", lambda backend: fake)
    monkeypatch.setattr(
        gpu_dispatch, "_oom_exception_types", lambda: (_FakeOutOfMemoryError,),
    )
    yield fake


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


def _timeseries(n=20, t=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([np.sin(np.linspace(0, 8, t)) + 0.05 * rng.standard_normal(t) for _ in range(n)])


# ==========================================================================
# Real feature dispatch through the fake backend -- exercises the actual
# gpu_capable() + kernel-lookup + dispatch machinery end to end (just not
# real CUDA numerics).
# ==========================================================================
def test_gpu_capable_reduction_feature_dispatches_to_gpu(fake_gpu):
    X = _timeseries(n=30)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict",
    )
    assert info.actual_backend == "gpu"
    assert info.gpu_attempted is True
    assert info.gpu_backend_name == "cupy"
    cpu_ref = np.nanmean(X, axis=1)
    assert np.allclose(np.asarray(result), cpu_ref, rtol=1e-4, atol=1e-6)


def test_gpu_capable_spectral_feature_dispatches_to_gpu(fake_gpu):
    X = _timeseries(n=30)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.spectral.centroid"), X, backend="gpu", gpu_fallback="strict",
    )
    assert info.actual_backend == "gpu"


def test_validate_true_cross_checks_against_cpu_and_passes(fake_gpu):
    X = _timeseries(n=15, seed=3)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.rms"), X, backend="gpu", gpu_fallback="strict", validate=True,
    )
    assert info.validation is not None
    assert info.validation["passed"] is True


def test_backend_auto_below_threshold_uses_cpu_no_gpu_attempted(fake_gpu):
    small_X = _timeseries(n=5)  # well below gpu_min_rows=8000 for reductions
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), small_X, backend="auto",
    )
    assert info.actual_backend == "cpu"
    assert info.gpu_attempted is False
    assert info.fallback_reason == "below_gpu_threshold"


def test_backend_auto_above_threshold_uses_gpu(fake_gpu, monkeypatch):
    spec = REGISTRY.get("timeseries.statistical.mean")
    monkeypatch.setattr(type(spec), "gpu_min_rows", 10, raising=False)  # unused; spec is frozen -- use n_rows_hint instead
    X = _timeseries(n=20)
    result, info = gpu_dispatch.dispatch_one_feature(
        spec, X, backend="auto", n_rows_hint=999_999,  # force "large enough" without needing a huge array
    )
    assert info.actual_backend == "gpu"


# ==========================================================================
# CUDA OOM retry: halves the batch, terminates, matches CPU, respects
# batch-splitting safety.
# ==========================================================================
def test_oom_retry_halves_batch_and_succeeds(fake_gpu):
    fake_gpu.oom_above_rows = 15  # first attempt (32 rows) OOMs, retries succeed once small enough
    X = _timeseries(n=32, seed=5)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict",
    )
    assert info.actual_backend == "gpu"
    assert info.gpu_retries >= 1
    cpu_ref = np.nanmean(X, axis=1)
    assert np.allclose(np.asarray(result), cpu_ref, rtol=1e-4, atol=1e-6)


def test_oom_retry_result_matches_cpu_reference_exactly_in_shape(fake_gpu):
    fake_gpu.oom_above_rows = 4
    X = _timeseries(n=17, seed=6)  # not a power of 2 -- exercises uneven halving
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.rms"), X, backend="gpu", gpu_fallback="strict",
    )
    assert np.asarray(result).shape == (17,)
    assert info.gpu_retries > 0


def test_oom_retry_is_bounded_never_infinite_loops(fake_gpu):
    """Even an OOM at every possible size must terminate (down to 1 row) -- no infinite loop."""
    fake_gpu.oom_above_rows = 0  # OOMs unconditionally for any n_rows > 0
    X = _timeseries(n=8, seed=7)
    with pytest.raises(GPUOutOfMemoryError):
        gpu_dispatch.dispatch_one_feature(
            REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict",
        )


def test_oom_after_retries_falls_back_to_cpu_when_policy_allows(fake_gpu):
    fake_gpu.oom_above_rows = 0
    X = _timeseries(n=8, seed=8)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="fallback",
    )
    assert info.cpu_fallback is True
    assert info.fallback_reason == "cuda_oom_after_retries"
    cpu_ref = np.nanmean(X, axis=1)
    assert np.allclose(np.asarray(result), cpu_ref)


def test_non_splittable_full_dataset_feature_never_splits_on_oom(fake_gpu, register_temp_feature):
    """A hypothetical FULL_DATASET GPU feature must NEVER be chunked to recover from OOM --
    this uses a synthetic spec/kernel since no real registered feature is both GPU-capable
    and FULL_DATASET in this milestone (registry.validate_spec forbids that combination for
    STRUCTURED_SAMPLE; this test proves the dispatch-level enforcement independently of that
    registry rule, using batch_semantics directly)."""
    spec = register_temp_feature(FeatureSpec(
        name="test.gpu.full_dataset_feature", family="gpu", modality="test",
        gpu_available=True, gpu_backend="cupy", requires_full_materialization=True,
        batch_semantics=BatchSemantics.FULL_DATASET,
    ))
    # Register a fake reduction kernel for this synthetic feature name so gpu_capable() is True.
    from featm.gpu.kernels import reductions as red_kernels

    red_kernels.REDUCTION_KERNELS[spec.name] = lambda a, xp: xp.nanmean(a, axis=1)
    try:
        fake_gpu.oom_above_rows = 0
        X = _timeseries(n=8, seed=9)
        with pytest.raises(GPUOutOfMemoryError, match="not batch-splittable"):
            gpu_dispatch.dispatch_one_feature(spec, X, backend="gpu", gpu_fallback="strict")
        assert fake_gpu.oom_call_count == 1, "must attempt exactly once, never split a FULL_DATASET feature"
    finally:
        del red_kernels.REDUCTION_KERNELS[spec.name]


# ==========================================================================
# Shared GPU intermediate: the spectral FFT primitive is computed once per
# batch even when several dependent features are selected together.
# ==========================================================================
def test_shared_gpu_fft_computed_once_for_multiple_dependent_features(fake_gpu, monkeypatch):
    """3 features sharing the SAME gpu_output_dtype (float32, the family default)
    must share exactly one cached FFT for the same batch_id.

    Deliberately picks 3 float32 features here, not just any 3 spectral
    features: `timeseries.spectral.fft`/`.cepstrum`/`.flatness` were changed
    to `gpu_output_dtype='float64'` after this milestone's real Kaggle
    validation found real precision issues in their log-domain/phase
    computations (see their registration comments in
    featm/features/timeseries/full.py and STEP3_GPU_HANDOFF.md §D) -- mixing
    a float64 feature into this test would correctly NOT share a cache entry
    with the float32 ones (see the companion test below), which would look
    like a cache-sharing regression here if this test still used one."""
    from featm.gpu.kernels import spectral as spectral_kernels

    call_count = {"n": 0}
    real_impl = spectral_kernels._spectral_parts_impl_gpu

    def counting_impl(a, xp):
        call_count["n"] += 1
        return real_impl(a, xp)

    monkeypatch.setattr(spectral_kernels, "_spectral_parts_impl_gpu", counting_impl)

    X = _timeseries(n=10, seed=10)
    from featm.gpu.cache import scoped_gpu_cache

    with scoped_gpu_cache():
        for name in ("timeseries.spectral.centroid", "timeseries.spectral.dominant_frequency", "timeseries.spectral.periodogram"):
            gpu_dispatch.dispatch_one_feature(REGISTRY.get(name), X, backend="gpu", gpu_fallback="strict", batch_id=42)
    assert call_count["n"] == 1, f"expected the shared FFT computed exactly once, got {call_count['n']}"


def test_mixed_dtype_spectral_features_correctly_use_separate_cache_entries(fake_gpu, monkeypatch):
    """A float64 feature (e.g. .flatness, .fft, .cepstrum) and a float32 feature
    selected together must NOT silently share one cached FFT tensor across
    incompatible precisions -- the cache key includes dtype specifically to
    prevent this, confirmed here rather than just asserted."""
    from featm.gpu.kernels import spectral as spectral_kernels

    call_count = {"n": 0}
    real_impl = spectral_kernels._spectral_parts_impl_gpu

    def counting_impl(a, xp):
        call_count["n"] += 1
        return real_impl(a, xp)

    monkeypatch.setattr(spectral_kernels, "_spectral_parts_impl_gpu", counting_impl)

    X = _timeseries(n=10, seed=10)
    from featm.gpu.cache import scoped_gpu_cache

    with scoped_gpu_cache():
        # C03 may select float64 for both features at dispatch. Check the cache
        # contract directly with genuinely different device-array dtypes.
        spectral_kernels.gpu_spectral_parts(X.astype(np.float32), np, batch_id=43)
        spectral_kernels.gpu_spectral_parts(X.astype(np.float64), np, batch_id=43)
    assert call_count["n"] == 2, f"expected two separate cache entries (different dtypes), got {call_count['n']}"


def test_gpu_cache_scope_releases_entries_on_exit(fake_gpu):
    cache = GPUIntermediateCache()
    cache.get_or_compute(("k",), lambda: np.zeros(100))
    assert cache.entry_count == 1
    cache.clear()
    assert cache.entry_count == 0
    assert cache.size_bytes == 0


def test_gpu_cache_never_leaks_between_batches(fake_gpu):
    """Two different batch_ids never hit each other's cached FFT."""
    from featm.gpu.cache import scoped_gpu_cache

    X1 = _timeseries(n=6, seed=11)
    X2 = _timeseries(n=6, seed=12)
    with scoped_gpu_cache() as cache1:
        r1, _ = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.spectral.centroid"), X1, backend="gpu", gpu_fallback="strict", batch_id=1)
    with scoped_gpu_cache() as cache2:
        r2, _ = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.spectral.centroid"), X2, backend="gpu", gpu_fallback="strict", batch_id=1)
    assert cache1 is not cache2
    assert not np.allclose(np.asarray(r1), np.asarray(r2))


# ==========================================================================
# Out-of-core (Source-backed) GPU dispatch integration.
# ==========================================================================
def test_out_of_core_gpu_backend_dispatches_per_batch(fake_gpu):
    df = pd.DataFrame(_timeseries(n=24, seed=13))
    result = featm.extract(
        df, features=["timeseries.statistical.mean"], backend="gpu", gpu_fallback="strict",
        batch_rows=6, output="memory", errors="raise",
    )
    backend_info = result.report.backend_report["timeseries.statistical.mean"]
    assert backend_info["actual_backend"] == "gpu"
    cpu_ref = featm.extract(df, features=["timeseries.statistical.mean"], backend="cpu", batch_rows=6, output="memory")
    assert np.allclose(result.to_pandas().to_numpy(), cpu_ref.to_pandas().to_numpy(), rtol=1e-4, atol=1e-6)


def test_out_of_core_gpu_batch_equivalence_across_partitions(fake_gpu):
    df = pd.DataFrame(_timeseries(n=24, seed=14))
    full = featm.extract(df, features=["timeseries.statistical.rms"], backend="gpu", gpu_fallback="strict", batch_rows=24, output="memory", errors="raise")
    batched = featm.extract(df, features=["timeseries.statistical.rms"], backend="gpu", gpu_fallback="strict", batch_rows=5, output="memory", errors="raise")
    assert np.allclose(full.to_pandas().to_numpy(), batched.to_pandas().to_numpy(), rtol=1e-4, atol=1e-6)


# ==========================================================================
# Audit-fix pass, Finding 1: GPU memory budget is actually enforced, not
# resolved-and-discarded.
# ==========================================================================
def test_tiny_gpu_budget_causes_proactive_chunking_for_safe_feature(fake_gpu, monkeypatch):
    calls: list[int] = []
    real_run = gpu_dispatch._run_gpu_kernel

    def counting_run(spec, a_np, xp_backend, *, cache_id, device_index, params):
        calls.append(len(a_np))
        return real_run(spec, a_np, xp_backend, cache_id=cache_id, device_index=device_index, params=params)

    monkeypatch.setattr(gpu_dispatch, "_run_gpu_kernel", counting_run)
    X = _timeseries(n=32, t=64, seed=20)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict",
        gpu_memory_budget="6KB",  # physical VRAM (fake: 8GB free) would easily fit the whole batch
    )
    assert len(calls) > 1, f"expected proactive VRAM-budget chunking (multiple kernel calls), got {calls}"
    assert info.gpu_planned_batch_rows is not None and 0 < info.gpu_planned_batch_rows < 32
    cpu_ref = np.nanmean(X, axis=1)
    assert np.allclose(np.asarray(result), cpu_ref, rtol=1e-4, atol=1e-6), "chunked GPU result must equal the CPU reference"


def test_strict_policy_rejects_before_allocation_when_one_row_does_not_fit(fake_gpu):
    X = _timeseries(n=10, t=64, seed=21)
    with pytest.raises(GPUOutOfMemoryError, match="single row"):
        gpu_dispatch.dispatch_one_feature(
            REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict",
            gpu_memory_budget=1,  # 1 byte -- cannot fit even one row
        )


def test_fallback_policy_uses_cpu_and_reports_insufficient_vram(fake_gpu):
    X = _timeseries(n=10, t=64, seed=22)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="fallback", gpu_memory_budget=1,
    )
    assert info.cpu_fallback is True
    assert info.fallback_reason == "insufficient_vram"
    assert info.actual_backend == "cpu"
    assert info.gpu_attempted is False, "must never attempt an allocation known in advance not to fit"
    cpu_ref = np.nanmean(X, axis=1)
    assert np.allclose(np.asarray(result), cpu_ref)


def test_non_splittable_feature_never_proactively_split_on_insufficient_budget(fake_gpu, register_temp_feature):
    """A FULL_DATASET feature whose full batch does not fit the declared budget must fail/fallback
    exactly as if nothing fits -- NEVER proactively chunked, even though the budget fits SOME rows."""
    spec = register_temp_feature(FeatureSpec(
        name="test.gpu.full_dataset_budget", family="gpu", modality="test",
        gpu_available=True, gpu_backend="cupy", requires_full_materialization=True,
        batch_semantics=BatchSemantics.FULL_DATASET,
        implementation=lambda x, **kw: np.nanmean(np.asarray(x), axis=1),
    ))
    from featm.gpu.kernels import reductions as red_kernels

    red_kernels.REDUCTION_KERNELS[spec.name] = lambda a, xp: xp.nanmean(a, axis=1)
    try:
        X = _timeseries(n=32, t=64, seed=23)
        bytes_per_row = gpu_memory.estimate_gpu_bytes_per_row(spec, n_columns_in=64, itemsize=4)
        budget_bytes = bytes_per_row * 8  # fits 8 of 32 rows -- neither zero nor all
        result, info = gpu_dispatch.dispatch_one_feature(
            spec, X, backend="gpu", gpu_fallback="fallback", gpu_memory_budget=budget_bytes,
        )
        assert info.gpu_attempted is False
        assert info.cpu_fallback is True
        assert info.fallback_reason == "insufficient_vram"
    finally:
        del red_kernels.REDUCTION_KERNELS[spec.name]


def test_gpu_report_fields_populated_on_successful_gpu_run(fake_gpu):
    X = _timeseries(n=10, seed=24)
    _, info = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict")
    d = info.as_dict()
    for k in ("gpu_total_memory", "gpu_free_memory_at_plan", "gpu_budget", "gpu_device_index", "gpu_planned_batch_rows"):
        assert d[k] is not None, f"{k} should be populated after a resolved GPU budget"


def test_plan_gpu_batch_rows_genuinely_called_by_production_dispatch(fake_gpu, monkeypatch):
    calls = []
    real_plan = gpu_dispatch.gpu_memory.plan_gpu_batch_rows

    def tracking_plan(**kwargs):
        calls.append(kwargs)
        return real_plan(**kwargs)

    monkeypatch.setattr(gpu_dispatch.gpu_memory, "plan_gpu_batch_rows", tracking_plan)
    X = _timeseries(n=10, seed=25)
    gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict")
    assert calls, "plan_gpu_batch_rows was never called by production dispatch -- it must not be dead code"


def test_gpu_cache_max_bytes_derived_from_declared_budget_end_to_end(fake_gpu, monkeypatch):
    """engine.extract() must size the shared GPU cache from gpu_memory_budget=, not a fixed 64MiB cap.

    ``extract()`` imports ``scoped_gpu_cache`` locally (``from .gpu.cache import
    gpu_cache_budget_bytes, scoped_gpu_cache``), so the only reliable patch point
    is the attribute on ``featm.gpu.cache`` itself -- the local import re-reads
    that attribute freshly on every call.
    """
    import featm.gpu.cache as gpu_cache_mod

    seen_max_bytes: list[int] = []
    real_scoped_gpu_cache = gpu_cache_mod.scoped_gpu_cache

    def tracking_scoped(*, max_bytes=gpu_cache_mod._DEFAULT_GPU_CACHE_BUDGET_BYTES):
        seen_max_bytes.append(max_bytes)
        return real_scoped_gpu_cache(max_bytes=max_bytes)

    monkeypatch.setattr(gpu_cache_mod, "scoped_gpu_cache", tracking_scoped)
    X = _timeseries(n=5, seed=26)
    featm.extract(X, features=["timeseries.statistical.mean"], backend="gpu", gpu_fallback="strict", gpu_memory_budget="4KB")
    assert seen_max_bytes, "scoped_gpu_cache was never opened with a resolved max_bytes"
    assert seen_max_bytes[-1] <= 4096, f"GPU cache budget {seen_max_bytes[-1]} exceeds the declared 4KB gpu_memory_budget"


# ==========================================================================
# Audit-fix pass, Finding 2: spectral shared-cache key collision after an
# OOM split. Reproduces the exact independently-reported 4-row case, plus
# regression coverage for vector features, odd splits, cross-feature
# sharing, and no leakage across outer batches.
# ==========================================================================
def test_spectral_oom_split_cache_collision_reproducer_fixed(fake_gpu):
    X = np.vstack([
        np.sin(np.linspace(0, 4, 64)),
        np.sin(np.linspace(0, 4, 64)),
        np.sin(np.linspace(0, 9, 64)),
        np.sin(np.linspace(0, 9, 64)),
    ])
    fake_gpu.oom_above_rows = 2  # first attempt (4 rows) OOMs; halves into two 2-row sub-batches
    spec = REGISTRY.get("timeseries.spectral.centroid")
    result, info = gpu_dispatch.dispatch_one_feature(spec, X, backend="gpu", gpu_fallback="strict")

    from featm.features._utils import spectral_parts

    _, mag, _, f = spectral_parts(X)
    den = mag.sum(1)
    cpu_ref = np.divide((mag * f).sum(1), den, out=np.zeros(len(mag)), where=den > 0)
    result_arr = np.asarray(result)
    assert np.allclose(result_arr, cpu_ref, rtol=1e-4, atol=1e-6), (
        f"GPU result after OOM split {result_arr} != CPU reference {cpu_ref} -- the second half "
        "likely reused the first half's cached FFT (Finding 2 regression)"
    )
    assert not np.isclose(result_arr[0], result_arr[2]), "rows 0/1 (freq k=4) and 2/3 (freq k=9) must differ -- equal values would mean a cache collision"


def test_vector_spectral_feature_after_oom_split_matches_cpu(fake_gpu):
    from featm.registry import FeatureOutput

    X = np.vstack([np.sin(np.linspace(0, k, 64)) for k in (2, 4, 6, 8)])
    fake_gpu.oom_above_rows = 2
    spec = REGISTRY.get("timeseries.spectral.periodogram")
    result, info = gpu_dispatch.dispatch_one_feature(spec, X, backend="gpu", gpu_fallback="strict")

    from featm.features._utils import spectral_parts

    _, _, p, _ = spectral_parts(X)
    cpu_ref = p[:, :16]
    result_arr = result.as_array() if isinstance(result, FeatureOutput) else np.asarray(result)
    assert np.allclose(result_arr, cpu_ref, rtol=1e-4, atol=1e-6)
    assert info.gpu_retries > 0


def test_odd_batch_size_oom_split_5_to_2_plus_3(fake_gpu):
    rng = np.random.default_rng(30)
    X = rng.standard_normal((5, 48))
    fake_gpu.oom_above_rows = 2
    spec = REGISTRY.get("timeseries.spectral.centroid")
    result, info = gpu_dispatch.dispatch_one_feature(spec, X, backend="gpu", gpu_fallback="strict")

    from featm.features._utils import spectral_parts

    _, mag, _, f = spectral_parts(X)
    den = mag.sum(1)
    cpu_ref = np.divide((mag * f).sum(1), den, out=np.zeros(len(mag)), where=den > 0)
    assert np.allclose(np.asarray(result), cpu_ref, rtol=1e-4, atol=1e-6)
    assert info.gpu_retries > 0


def test_multiple_spectral_features_oom_split_share_cache_and_are_correct(fake_gpu, monkeypatch):
    from featm.gpu.cache import scoped_gpu_cache
    from featm.gpu.kernels import spectral as spectral_kernels

    call_count = {"n": 0}
    real_impl = spectral_kernels._spectral_parts_impl_gpu

    def counting_impl(a, xp):
        call_count["n"] += 1
        return real_impl(a, xp)

    monkeypatch.setattr(spectral_kernels, "_spectral_parts_impl_gpu", counting_impl)

    X = np.vstack([np.sin(np.linspace(0, k, 64)) for k in (2, 4, 6, 8)])
    fake_gpu.oom_above_rows = 2
    with scoped_gpu_cache():
        r1, _ = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.spectral.centroid"), X, backend="gpu", gpu_fallback="strict", batch_id=99)
        r2, _ = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.spectral.dominant_frequency"), X, backend="gpu", gpu_fallback="strict", batch_id=99)

    # 2 features x 2 sub-ranges (rows 0-2, rows 2-4) each -- but each sub-range is
    # SHARED across the two features (same batch_id, same split path) -> 2 real computations, not 4.
    assert call_count["n"] == 2, f"expected the FFT computed once per SUB-RANGE shared across features, got {call_count['n']}"

    from featm.features._utils import spectral_parts

    _, mag, p, f = spectral_parts(X)
    den = mag.sum(1)
    cpu_centroid = np.divide((mag * f).sum(1), den, out=np.zeros(len(mag)), where=den > 0)
    cpu_domfreq = f[np.argmax(p, axis=1)]
    assert np.allclose(np.asarray(r1), cpu_centroid, rtol=1e-4, atol=1e-6)
    assert np.allclose(np.asarray(r2), cpu_domfreq, rtol=1e-4, atol=1e-6)


def test_no_cache_leakage_between_outer_batches_after_oom_split(fake_gpu):
    from featm.gpu.cache import scoped_gpu_cache

    X1 = np.vstack([np.sin(np.linspace(0, k, 64)) for k in (1, 2, 3, 4)])
    X2 = np.vstack([np.sin(np.linspace(0, k, 64)) for k in (5, 6, 7, 8)])
    fake_gpu.oom_above_rows = 2
    with scoped_gpu_cache() as c1:
        r1, _ = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.spectral.centroid"), X1, backend="gpu", gpu_fallback="strict", batch_id=7)
    fake_gpu.oom_above_rows = 2
    with scoped_gpu_cache() as c2:
        r2, _ = gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.spectral.centroid"), X2, backend="gpu", gpu_fallback="strict", batch_id=7)
    assert c1 is not c2
    assert not np.allclose(np.asarray(r1), np.asarray(r2))


# ==========================================================================
# Audit-fix pass, Finding 3: device= controls one coherent execution context.
# ==========================================================================
def test_device_context_entered_with_the_resolved_device_index(fake_gpu, monkeypatch):
    calls = []
    real_ctx = gpu_backend.device_context

    def tracking_ctx(backend, index):
        calls.append((backend, index))
        return real_ctx(backend, index)

    monkeypatch.setattr(gpu_dispatch.gpu_backend, "device_context", tracking_ctx)
    X = _timeseries(n=10, seed=27)
    gpu_dispatch.dispatch_one_feature(REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict", device="cuda:0")
    assert calls, "device_context was never entered -- kernel execution must run inside the planned device's context"
    assert all(c == ("cupy", 0) for c in calls), f"expected every device_context call to use ('cupy', 0), got {calls}"


def test_invalid_device_index_raises_clear_error(fake_gpu):
    X = _timeseries(n=5, seed=28)
    with pytest.raises(GPUUnavailableError, match="device index"):
        gpu_dispatch.dispatch_one_feature(
            REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="strict", device="cuda:5",
        )


def test_invalid_device_index_falls_back_to_cpu_under_fallback_policy(fake_gpu):
    X = _timeseries(n=5, seed=29)
    result, info = gpu_dispatch.dispatch_one_feature(
        REGISTRY.get("timeseries.statistical.mean"), X, backend="gpu", gpu_fallback="fallback", device="cuda:5",
    )
    assert info.cpu_fallback is True
    assert info.fallback_reason == "gpu_unavailable"


# ==========================================================================
# Audit-fix pass, Finding 4: gpu_backend is enforced -- a CuPy-authored
# kernel never silently routes through Torch merely because Torch CUDA is
# available. Deliberately does NOT use the fake_gpu fixture (which forces
# require_gpu() to always return "cupy" regardless of args) -- these tests
# monkeypatch cupy_available/torch_cuda_available directly so the REAL
# require_gpu() backend-selection logic runs.
# ==========================================================================
def test_cupy_spec_never_routes_through_torch_on_torch_only_machine(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: True)
    X = _timeseries(n=10, seed=32)
    spec = REGISTRY.get("timeseries.statistical.mean")
    assert spec.gpu_backend == "cupy"
    result, info = gpu_dispatch.dispatch_one_feature(spec, X, backend="gpu", gpu_fallback="fallback")
    assert info.actual_backend == "cpu"
    assert info.cpu_fallback is True
    assert info.fallback_reason == "gpu_unavailable"
    assert info.required_gpu_backend == "cupy"
    assert info.gpu_backend_name is None, "must never have selected torch as the actual GPU backend for a cupy-only spec"
    cpu_ref = np.nanmean(X, axis=1)
    assert np.allclose(np.asarray(result), cpu_ref)


def test_strict_raises_when_required_backend_unavailable_even_if_other_backend_present(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: True)
    X = _timeseries(n=10, seed=33)
    spec = REGISTRY.get("timeseries.statistical.mean")
    with pytest.raises(GPUUnavailableError):
        gpu_dispatch.dispatch_one_feature(spec, X, backend="gpu", gpu_fallback="strict")


# ==========================================================================
# Audit-fix pass, Finding 5 (general validator regression, not feature-specific):
# the CPU<->GPU validator must catch a genuine wrong answer, not just accept
# anything within a loosely-declared tolerance.
# ==========================================================================
def test_validate_against_cpu_catches_deliberate_magnitude_corruption():
    spec = REGISTRY.get("timeseries.statistical.mean")
    X = _timeseries(n=10, seed=34)
    cpu_result = np.nanmean(X, axis=1)
    corrupted = cpu_result.copy()
    corrupted[0] += 1.0
    v = gpu_dispatch._validate_against_cpu(spec, cpu_result, corrupted, spec.gpu_numerical_tolerance)
    assert v["passed"] is False


def test_fft_and_cepstrum_no_longer_gpu_capable():
    """Findings 5/6: both features had their GPU acceleration removed (loose/hard-to-trust
    tolerances) rather than shipped with a validator that could hide a real wrong answer."""
    assert gpu_dispatch.gpu_capable(REGISTRY.get("timeseries.spectral.fft")) is False
    assert gpu_dispatch.gpu_capable(REGISTRY.get("timeseries.spectral.cepstrum")) is False
    assert REGISTRY.get("timeseries.spectral.fft").gpu_available is False
    assert REGISTRY.get("timeseries.spectral.cepstrum").gpu_available is False
    # signal.frequency.fft (magnitude-only, no phase columns, no DC-bin instability) is unaffected.
    assert gpu_dispatch.gpu_capable(REGISTRY.get("signal.frequency.fft")) is True


# ==========================================================================
# Audit-fix pass, Finding 10: backend="auto" considers total workload
# (n_rows * n_columns), not row count alone.
# ==========================================================================
def test_backend_auto_uses_workload_elements_not_rows_alone(fake_gpu):
    # Deliberately derives sizes from the feature's ACTUAL registered
    # threshold rather than hardcoding element counts -- the auto-threshold
    # fix pass (v0.3.2) recalibrated gpu_min_elements from real Kaggle
    # measurements, so a hardcoded "1,000,000 elements should be GPU"
    # assumption here would silently go stale the next time the threshold is
    # recalibrated from new evidence.
    spec = REGISTRY.get("timeseries.statistical.mean")
    threshold = gpu_memory.estimate_gpu_min_elements(spec)

    # Few rows, huge width -> a genuinely large workload (comfortably above
    # threshold); must choose GPU despite a row count far below gpu_min_rows.
    wide_t = threshold // 4 + 1_000
    few_rows_long = np.tile(np.linspace(0, 1, wide_t), (4, 1))
    assert few_rows_long.size > threshold
    _, info_long = gpu_dispatch.dispatch_one_feature(spec, few_rows_long, backend="auto")
    assert info_long.actual_backend == "gpu", (
        f"expected element-count-aware auto to choose GPU for a {few_rows_long.shape} "
        f"({few_rows_long.size:,} elements > threshold {threshold:,}) workload, got {info_long.fallback_reason}"
    )

    # Many rows, tiny width -> a genuinely tiny workload (comfortably below
    # threshold); must stay CPU despite a row count far above gpu_min_rows.
    many_rows_short = np.random.default_rng(35).standard_normal((50_000, 2))
    assert many_rows_short.size < threshold
    _, info_short = gpu_dispatch.dispatch_one_feature(spec, many_rows_short, backend="auto")
    assert info_short.actual_backend == "cpu"
    assert info_short.fallback_reason == "below_gpu_threshold"


# ==========================================================================
# Auto-threshold fix pass (v0.3.2): the shipped gpu_min_elements/gpu_min_rows
# values must be genuinely performance-sane (calibrated against real Kaggle
# P100 timings -- see STEP3_GPU_AUTO_THRESHOLD_FIX_HANDOFF.md), and the
# dispatch-policy machinery itself (workload below/above threshold, explicit
# gpu_min_elements override, explicit backend="gpu" unaffected) is covered
# here with the fake backend -- no real GPU needed for policy/control-flow.
# ==========================================================================
def test_workload_below_calibrated_threshold_stays_cpu_under_auto(fake_gpu):
    spec = REGISTRY.get("timeseries.statistical.mean")
    threshold = gpu_memory.estimate_gpu_min_elements(spec)
    n, t = 4, max(2, (threshold // 8))  # comfortably below threshold
    X = _timeseries(n=n, t=t)
    assert X.size < threshold
    _, info = gpu_dispatch.dispatch_one_feature(spec, X, backend="auto")
    assert info.actual_backend == "cpu"
    assert info.fallback_reason == "below_gpu_threshold"


def test_workload_above_calibrated_threshold_uses_gpu_under_auto(fake_gpu):
    spec = REGISTRY.get("timeseries.statistical.mean")
    threshold = gpu_memory.estimate_gpu_min_elements(spec)
    n, t = 4, (threshold // 4) + 1_000  # comfortably above threshold
    X = _timeseries(n=n, t=t)
    assert X.size > threshold
    _, info = gpu_dispatch.dispatch_one_feature(spec, X, backend="auto")
    assert info.actual_backend == "gpu"


def test_explicit_gpu_min_elements_overrides_derived_value(fake_gpu, register_temp_feature):
    """A feature that declares its OWN gpu_min_elements must use that, not the
    gpu_min_rows-derived default -- proves the override path (Finding 10's
    original mechanism, exercised concretely here with a real dispatch call)."""
    spec = register_temp_feature(FeatureSpec(
        name="test.gpu.custom_threshold_feature", family="gpu", modality="test",
        gpu_available=True, gpu_backend="cupy", gpu_min_rows=8_000, gpu_min_elements=100,
    ))
    from featm.gpu.kernels import reductions as red_kernels

    red_kernels.REDUCTION_KERNELS[spec.name] = lambda a, xp: xp.nanmean(a, axis=1)
    try:
        assert gpu_memory.estimate_gpu_min_elements(spec) == 100
        # 3 rows x 50 columns = 150 elements: below the gpu_min_rows-derived
        # default (8000*64=512,000) but above the explicit override (100) --
        # only the override being consulted correctly can choose GPU here.
        X_small = _timeseries(n=3, t=50)
        assert X_small.size == 150
        _, info = gpu_dispatch.dispatch_one_feature(spec, X_small, backend="auto")
        assert info.actual_backend == "gpu", f"explicit gpu_min_elements override was not honored: {info.fallback_reason}"
    finally:
        del red_kernels.REDUCTION_KERNELS[spec.name]


def test_explicit_backend_gpu_ignores_auto_threshold_entirely(fake_gpu):
    """backend="gpu" (not "auto") must try GPU regardless of workload size --
    the calibrated threshold only governs backend="auto"'s CHOICE, never
    backend="gpu"'s explicit request."""
    spec = REGISTRY.get("timeseries.statistical.mean")
    threshold = gpu_memory.estimate_gpu_min_elements(spec)
    X_tiny = _timeseries(n=2, t=4)
    assert X_tiny.size < threshold
    _, info = gpu_dispatch.dispatch_one_feature(spec, X_tiny, backend="gpu", gpu_fallback="strict")
    assert info.actual_backend == "gpu"
    assert info.fallback_reason is None


def test_cheap_reduction_threshold_exceeds_the_old_v031_contradicted_default():
    """v0.3.2 recalibrated the 'cheap single-pass reduction' auto threshold
    (mean/std/min/max/range/rms/energy + the matching signal.time_domain
    features) from real Kaggle P100 evidence after the v0.3.1 benchmark
    showed backend="auto" would pick GPU at a workload (10,000 rows x 64 =
    640,000 elements) that measured 0.456x (SLOWER) on real hardware. The old
    gpu_min_rows=8_000 default (512,000 elements at the width=64 calibration)
    is provably too low for THIS shape; the shipped threshold must now
    exceed that contradicted value and must not accept the exact
    contradicted workload."""
    old_v031_threshold_elements = 8_000 * gpu_memory.GPU_MIN_ROWS_CALIBRATION_SEQ_LEN
    for name in (
        "timeseries.statistical.mean", "timeseries.statistical.std", "timeseries.statistical.min",
        "timeseries.statistical.max", "timeseries.statistical.range", "timeseries.statistical.rms",
        "timeseries.statistical.energy", "signal.time_domain.rms", "signal.time_domain.peak_to_peak",
        "signal.time_domain.crest_factor", "signal.time_domain.impulse_factor",
    ):
        spec = REGISTRY.get(name)
        threshold = gpu_memory.estimate_gpu_min_elements(spec)
        assert threshold > old_v031_threshold_elements, (
            f"{name}: gpu_min_elements={threshold:,} must exceed the old, real-hardware-contradicted "
            f"v0.3.1 default of {old_v031_threshold_elements:,} elements"
        )
        assert 640_000 < threshold, f"{name}: the exact contradicted 10,000x64 workload (640,000 elements) must now be below threshold"


def test_median_and_line_length_get_their_own_lower_measured_thresholds():
    """median (a sort/partition, not a single pass) and line_length (diff +
    abs + sum) were individually measured and showed a real, wide-margin GPU
    win far earlier than the cheap-reduction shape -- they are DELIBERATELY
    given their own, lower, thresholds rather than inheriting the more
    conservative shared one, so real GPU speedup is not left on the table
    for a shape real measurement says is safe much sooner."""
    cheap_threshold = gpu_memory.estimate_gpu_min_elements(REGISTRY.get("timeseries.statistical.mean"))
    for name in ("timeseries.statistical.median", "signal.time_domain.line_length"):
        spec = REGISTRY.get(name)
        threshold = gpu_memory.estimate_gpu_min_elements(spec)
        assert threshold < cheap_threshold, f"{name}: expected a lower, individually-measured threshold than the shared cheap-reduction one"


# ==========================================================================
# Audit-fix pass, Finding 11: mixed CPU/GPU usage across out-of-core batches
# is reported honestly, not collapsed to only the most recent batch.
# ==========================================================================
def test_execution_report_marks_mixed_backend_usage_across_batches():
    from featm.gpu.dispatch import BackendInfo
    from featm.report import ExecutionReport

    report = ExecutionReport()
    gpu_info = BackendInfo(requested_backend="auto", actual_backend="gpu", gpu_attempted=True).as_dict()
    cpu_info = BackendInfo(requested_backend="auto", actual_backend="cpu", fallback_reason="below_gpu_threshold").as_dict()
    report.record_backend_info("timeseries.statistical.mean", gpu_info)
    report.record_backend_info("timeseries.statistical.mean", gpu_info)
    report.record_backend_info("timeseries.statistical.mean", cpu_info)

    entry = report.backend_report["timeseries.statistical.mean"]
    assert entry["actual_backend"] == "mixed"
    assert entry["gpu_batches"] == 2
    assert entry["cpu_batches"] == 1
    assert set(entry["actual_backends_used"]) == {"gpu", "cpu"}
    assert "below_gpu_threshold" in entry["fallback_reasons_seen"]


def test_execution_report_stays_pure_gpu_when_only_gpu_batches_seen():
    from featm.gpu.dispatch import BackendInfo
    from featm.report import ExecutionReport

    report = ExecutionReport()
    gpu_info = BackendInfo(requested_backend="gpu", actual_backend="gpu", gpu_attempted=True).as_dict()
    report.record_backend_info("timeseries.statistical.mean", gpu_info)
    report.record_backend_info("timeseries.statistical.mean", gpu_info)
    entry = report.backend_report["timeseries.statistical.mean"]
    assert entry["actual_backend"] == "gpu"
    assert entry["gpu_batches"] == 2
    assert entry["cpu_batches"] == 0


# ==========================================================================
# Audit-fix pass, Finding 12: checkpoint/backend policy -- mixed CPU/GPU
# shard resume is intentionally supported now that every retained
# GPU-capable feature uses a tight declared tolerance (Findings 5/6 removed
# the two loose-tolerance features), and is tested directly rather than
# merely asserted.
# ==========================================================================
def test_mixed_backend_checkpoint_resume_stays_within_tolerance(fake_gpu, tmp_path, monkeypatch):
    from featm.sources import PandasSource

    df = pd.DataFrame(_timeseries(n=25, t=64, seed=50))
    original_iter = PandasSource.iter_batches
    state = {"n": 0}

    def flaky_iter(self, *, columns=None, batch_rows=50_000):
        for batch in original_iter(self, columns=columns, batch_rows=batch_rows):
            state["n"] += 1
            if state["n"] > 2:
                raise RuntimeError("SIMULATED CRASH")
            yield batch

    monkeypatch.setattr(PandasSource, "iter_batches", flaky_iter)
    ckpt = tmp_path / "ckpt_mixed_backend"
    with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
        featm.extract(
            df, features=["timeseries.statistical.mean"], backend="cpu",
            batch_rows=5, output="disk", checkpoint=ckpt, errors="raise",
        )

    monkeypatch.setattr(PandasSource, "iter_batches", original_iter)
    resumed = featm.extract(
        df, features=["timeseries.statistical.mean"], backend="gpu", gpu_fallback="strict",
        batch_rows=5, output="disk", checkpoint=ckpt, errors="raise",
    )
    assert resumed.report.resumed is True, "resuming under a DIFFERENT backend= must not be rejected by checkpoint identity"

    clean = featm.extract(df, features=["timeseries.statistical.mean"], backend="cpu", batch_rows=5, output="disk")
    assert np.allclose(resumed.to_pandas().to_numpy(), clean.to_pandas().to_numpy(), rtol=1e-4, atol=1e-6)
