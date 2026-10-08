"""Lazy GPU backend detection and device management (Step 3).

**Nothing in this module imports CuPy or Torch at module-import time.** Every
probe below is a function, called lazily, that attempts the import inside a
``try/except ImportError`` -- so ``import featm`` on a CPU-only machine with
no GPU extras installed never imports a CUDA library and never emits a CUDA
warning, exactly as the Step 3 spec requires ("Importing featm on a CPU-only
system must not import CUDA libraries eagerly or emit CUDA warnings"). Every
probe result is cached (per-process, since GPU availability cannot change
mid-process) so repeated dispatch decisions do not repeatedly re-import.

Two backends are supported this milestone: ``"cupy"`` (preferred -- a closer
NumPy-array-API match, used for every kernel in :mod:`featm.gpu.kernels`) and
``"torch"`` (present for completeness and for a future Torch-model-routing
feature; no kernel in this milestone requires it, so its absence never blocks
GPU dispatch for the covered feature set). Neither is a hard dependency of
``pip install featm``.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any


@functools.lru_cache(maxsize=1)
def cupy_available() -> bool:
    """Whether CuPy is importable AND reports at least one visible CUDA device.

    A plain ``import cupy`` can succeed even with no GPU driver present (the
    package itself is pure-Python-importable); the actual "is there a usable
    GPU" question needs ``cupy.cuda.runtime.getDeviceCount() > 0``. Both
    steps are wrapped, since a broken/mismatched CUDA install can raise
    something other than ``ImportError`` (e.g. an ``OSError`` for a missing
    ``.so``/``.dll``) -- any such failure means "not usable," not a crash.
    """
    try:
        import cupy  # noqa: F401
        import cupy.cuda.runtime as rt

        return rt.getDeviceCount() > 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    """Whether Torch is importable AND ``torch.cuda.is_available()``."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def any_gpu_available() -> bool:
    """Whether at least one supported GPU backend is usable right now."""
    return cupy_available() or torch_cuda_available()


def preferred_backend() -> str | None:
    """``"cupy"`` if available, else ``"torch"`` if available, else ``None``.

    CuPy is preferred: every GPU kernel this milestone ships
    (:mod:`featm.gpu.kernels`) is written against the CuPy array API, which
    mirrors NumPy closely enough that the CPU implementation and the GPU
    implementation read as the same algorithm -- important for the "CPU
    remains the mathematical reference" contract (Step 3 spec §8).
    """
    if cupy_available():
        return "cupy"
    if torch_cuda_available():
        return "torch"
    return None


@dataclass(frozen=True)
class DeviceInfo:
    """A snapshot of one GPU device's identity and current memory state."""

    backend: str
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    cuda_version: str | None = None
    driver_version: str | None = None


def parse_device(device: str | int | None) -> tuple[str, int]:
    """Normalize a ``device=`` argument to ``(kind, index)`` -- e.g. ``("cuda", 0)``.

    Accepts ``None`` (default device 0), an int (device index), or a string
    like ``"cuda:0"``/``"cuda"``/``"gpu:1"``. Single-GPU/current-device only
    (Step 3 explicitly excludes multi-GPU scheduling) -- this just lets a
    caller pin a specific device index when more than one GPU is visible.
    """
    if device is None:
        return "cuda", 0
    if isinstance(device, int):
        return "cuda", device
    s = str(device).lower()
    if ":" in s:
        kind, idx = s.split(":", 1)
        return kind, int(idx)
    return s, 0


def device_info(device: "str | int | None" = None, *, backend: str | None = None) -> DeviceInfo | None:
    """Query one device's name/total/free memory, or ``None`` if no GPU backend is usable.

    ``backend`` forces ``"cupy"``/``"torch"``; otherwise :func:`preferred_backend` decides.
    """
    be = backend or preferred_backend()
    if be is None:
        return None
    _, idx = parse_device(device)
    if be == "cupy":
        return _cupy_device_info(idx)
    if be == "torch":
        return _torch_device_info(idx)
    return None


def _cupy_device_info(index: int) -> DeviceInfo | None:
    try:
        import cupy

        with cupy.cuda.Device(index) as dev:
            free, total = cupy.cuda.runtime.memGetInfo()
            props = cupy.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            runtime_version = cupy.cuda.runtime.runtimeGetVersion()
            return DeviceInfo(
                backend="cupy", index=index, name=name, total_bytes=int(total), free_bytes=int(free),
                cuda_version=f"{runtime_version // 1000}.{(runtime_version % 1000) // 10}",
            )
    except Exception:
        return None


def _torch_device_info(index: int) -> DeviceInfo | None:
    try:
        import torch

        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        return DeviceInfo(
            backend="torch", index=index, name=props.name, total_bytes=int(total), free_bytes=int(free),
            cuda_version=torch.version.cuda,
        )
    except Exception:
        return None


def require_gpu(*, backend: str | None = None) -> str:
    """Return the usable backend name, or raise :class:`GPUUnavailableError`.

    The single point every strict-GPU code path funnels through, so the
    "no GPU available" error message is consistent everywhere it can occur.
    """
    from .exceptions import GPUUnavailableError

    be = backend or preferred_backend()
    if be is None or not (cupy_available() if be == "cupy" else torch_cuda_available()):
        raise GPUUnavailableError(
            "No usable GPU backend is available: neither CuPy with a visible CUDA device, nor "
            "Torch with torch.cuda.is_available()==True. Install featm[gpu-cupy] or featm[gpu-torch] "
            "on a machine with an NVIDIA GPU and a working CUDA driver, or use backend='cpu' (the "
            "default)."
        )
    return be


def xp_module(backend: str):
    """Return the CuPy/Torch module for ``backend`` (only called after :func:`require_gpu`)."""
    if backend == "cupy":
        import cupy

        return cupy
    if backend == "torch":
        import torch

        return torch
    raise ValueError(f"Unknown GPU backend: {backend!r}")


def device_count(backend: str) -> int:
    """Number of visible devices for ``backend`` -- used to validate an explicit ``device=`` index."""
    if backend == "cupy":
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount())
    if backend == "torch":
        import torch

        return int(torch.cuda.device_count())
    raise ValueError(f"Unknown GPU backend: {backend!r}")


def validate_device_index(backend: str, index: int) -> None:
    """Raise :class:`GPUUnavailableError` if ``index`` names a device that does not exist.

    Audit-fix pass, Finding 3: a caller can pass ``device="cuda:5"`` on a
    single-GPU box; without this check, planning/execution would silently
    query/target a nonexistent device and fail with a confusing low-level
    error deep inside CuPy/Torch instead of a clear, actionable one here.
    """
    from .exceptions import GPUUnavailableError

    count = device_count(backend)
    if index < 0 or index >= count:
        raise GPUUnavailableError(
            f"Requested GPU device index {index} for backend {backend!r}, but only {count} "
            f"device(s) are visible (valid indices: 0..{count - 1})." if count > 0 else
            f"Requested GPU device index {index} for backend {backend!r}, but no devices are visible."
        )


def device_context(backend: str, index: int):
    """One coherent execution context for ``backend``'s device ``index``.

    Audit-fix pass, Finding 3: everything that must agree on which physical
    GPU is in play -- VRAM planning/query, host->device transfer, kernel
    execution, cache allocations, output transfer, and OOM cleanup -- now
    runs inside this SAME context manager (see ``gpu/dispatch.py``'s
    ``_run_gpu_kernel``/``_release_gpu_scratch``), rather than planning
    against a caller-selected device while kernels silently execute on
    whatever the process's ambient current/default device happens to be.
    """
    if backend == "cupy":
        import cupy

        return cupy.cuda.Device(index)
    if backend == "torch":
        import torch

        return torch.cuda.device(index)
    raise ValueError(f"Unknown GPU backend: {backend!r}")


def reset_probe_cache() -> None:
    """Clear the cached availability probes -- for tests that monkeypatch backend availability."""
    cupy_available.cache_clear()
    torch_cuda_available.cache_clear()
