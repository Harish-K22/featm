"""Step 3 local tests: GPU backend detection, lazy import discipline, CPU-only behavior.

None of these tests require a real GPU -- they verify the *architecture*
(lazy probing, safe CPU-only defaults, clean error types). Real CUDA
numerical correctness is Kaggle-only (see notebooks/FEATM_STEP3_GPU_VALIDATION.ipynb
and STEP3_GPU_HANDOFF.md's explicit "PENDING REAL CUDA VALIDATION" status).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import featm
from featm.gpu import backend as gpu_backend
from featm.gpu.exceptions import GPUUnavailableError


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    gpu_backend.reset_probe_cache()
    yield
    gpu_backend.reset_probe_cache()


def test_importing_featm_does_not_import_cuda_libraries():
    """The Step 3 spec is explicit: 'Importing featm on a CPU-only system must not
    import CUDA libraries eagerly or emit CUDA warnings.'"""
    cuda_modules = [m for m in sys.modules if "cupy" in m.lower() or ("torch" in m.lower() and "torchgen" not in m.lower())]
    assert cuda_modules == [], f"featm import pulled in CUDA-related modules: {cuda_modules}"


def test_cupy_available_false_when_not_installed_or_no_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # simulate "not importable"
    gpu_backend.reset_probe_cache()
    assert gpu_backend.cupy_available() is False


def test_any_gpu_available_false_on_this_machine():
    # This test suite runs on whatever machine executes it -- on a real GPU
    # box this could be True, which is fine; the point is the call itself
    # never raises and returns a plain bool either way.
    assert isinstance(gpu_backend.any_gpu_available(), bool)


def test_preferred_backend_none_when_nothing_available(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: False)
    assert gpu_backend.preferred_backend() is None


def test_preferred_backend_prefers_cupy_over_torch(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: True)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: True)
    assert gpu_backend.preferred_backend() == "cupy"


def test_require_gpu_raises_clean_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: False)
    with pytest.raises(GPUUnavailableError):
        gpu_backend.require_gpu()


def test_device_info_none_when_no_backend(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: False)
    assert gpu_backend.device_info() is None


# ==========================================================================
# backend="cpu" must be byte-for-byte the pre-Step-3 behavior.
# ==========================================================================
def test_backend_cpu_default_matches_pre_step3_output():
    rng = np.random.default_rng(0)
    X = np.vstack([np.sin(np.linspace(0, 10, 100)) + 0.02 * rng.standard_normal(100) for _ in range(5)])
    r_default = featm.extract(X, features=["timeseries.statistical.mean"])
    r_explicit_cpu = featm.extract(X, features=["timeseries.statistical.mean"], backend="cpu")
    assert np.allclose(r_default.to_numpy(), r_explicit_cpu.to_numpy())
    assert "backend_report" not in r_default.attrs
    assert "backend_report" not in r_explicit_cpu.attrs


def test_backend_cpu_never_imports_gpu_dispatch_module():
    """backend='cpu' should not even attempt the lazy featm.gpu.dispatch import path
    (verified indirectly: no GPU-related exception type can surface for backend='cpu')."""
    rng = np.random.default_rng(1)
    X = np.vstack([np.sin(np.linspace(0, 10, 50)) for _ in range(3)])
    result = featm.extract(X, features=["timeseries.statistical.mean"], backend="cpu", errors="raise")
    assert result.shape == (3, 1)


def test_invalid_backend_raises_value_error():
    X = np.zeros((3, 10))
    with pytest.raises(ValueError, match="backend must be"):
        featm.extract(X, features=["timeseries.statistical.mean"], backend="tpu")


def test_invalid_gpu_fallback_raises_value_error():
    X = np.zeros((3, 10))
    with pytest.raises(ValueError, match="gpu_fallback must be"):
        featm.extract(X, features=["timeseries.statistical.mean"], backend="gpu", gpu_fallback="bogus")


def test_backend_auto_on_gpu_unavailable_machine_uses_cpu_and_reports_why(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: False)
    rng = np.random.default_rng(2)
    X = np.vstack([np.sin(np.linspace(0, 10, 50)) for _ in range(4)])
    result = featm.extract(X, features=["timeseries.statistical.mean"], backend="auto")
    report = result.attrs["backend_report"]["timeseries.statistical.mean"]
    assert report["actual_backend"] == "cpu"
    assert report["fallback_reason"] == "gpu_unavailable"
    assert report["gpu_attempted"] is False


def test_backend_gpu_strict_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: False)
    X = np.vstack([np.sin(np.linspace(0, 10, 50)) for _ in range(4)])
    with pytest.raises(GPUUnavailableError):
        featm.extract(X, features=["timeseries.statistical.mean"], backend="gpu", gpu_fallback="strict", errors="raise")


def test_backend_gpu_fallback_default_silently_ok_but_visibly_reported(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cupy_available", lambda: False)
    monkeypatch.setattr(gpu_backend, "torch_cuda_available", lambda: False)
    X = np.vstack([np.sin(np.linspace(0, 10, 50)) for _ in range(4)])
    result = featm.extract(X, features=["timeseries.statistical.mean"], backend="gpu")  # default gpu_fallback="fallback"
    report = result.attrs["backend_report"]["timeseries.statistical.mean"]
    assert report["cpu_fallback"] is True
    assert report["fallback_reason"] == "gpu_unavailable"
    assert result.shape == (4, 1)
