"""GPU-specific exception hierarchy (Step 3).

Distinct exception classes for distinct failure modes, per the Step 3
specification: "do not turn every GPU issue into a generic RuntimeError."
Each class documents when it is raised and how ``backend=``/``gpu_fallback=``
policy is expected to react to it -- the actual policy logic lives in
:mod:`featm.gpu.dispatch`, not here; these are just the vocabulary it uses.
"""
from __future__ import annotations


class GPUError(RuntimeError):
    """Base class for every Step 3 GPU-specific error."""


class GPUUnavailableError(GPUError):
    """No usable GPU backend (CuPy/Torch with CUDA) is importable/available on this machine.

    Raised by :func:`featm.gpu.backend.require_gpu` when ``backend="gpu"``
    (or ``"auto"`` after deciding GPU would otherwise be used) is requested
    but neither an optional GPU dependency is installed nor a CUDA device is
    visible. Under ``backend="gpu", gpu_fallback="strict"`` this propagates
    to the caller (or is recorded as a per-feature failure, per ``errors=``);
    under ``gpu_fallback="fallback"`` (the default) it is caught by the
    dispatcher and causes a CPU fallback with ``fallback_reason="gpu_unavailable"``.
    """


class GPUImplementationMissingError(GPUError):
    """The selected feature has no registered GPU implementation.

    Distinct from :class:`GPUUnavailableError`: a GPU may be present and
    usable, but this particular feature was never given a
    ``gpu_implementation`` (most of the registry, for this milestone -- see
    ``STEP3_GPU_HANDOFF.md`` for the exact covered feature list).
    ``fallback_reason="feature_not_gpu_capable"``.
    """


class GPUDtypeUnsupportedError(GPUError):
    """The input's dtype cannot be safely represented by the GPU kernel without silent precision loss.

    Raised (never silently downcast) when a feature's GPU implementation
    would need to coerce the input in a way that was not explicitly
    requested -- e.g. complex/object dtypes, or float64 input when
    ``gpu_output_dtype`` is declared ``float32`` and the caller did not
    opt into precision loss. ``fallback_reason="unsupported_dtype"``.
    """


class GPUOutOfMemoryError(GPUError):
    """A GPU kernel raised a real CUDA out-of-memory condition.

    :mod:`featm.gpu.dispatch` catches the backend-native OOM exception
    (``cupy.cuda.memory.OutOfMemoryError`` / ``torch.cuda.OutOfMemoryError``)
    and re-raises it as this common type so the rest of the engine (the
    batch-halving retry loop, the fallback-reason reporting) does not need
    to know which GPU library raised it.
    """


class GPUNumericalValidationError(GPUError):
    """A GPU result failed its declared CPU-vs-GPU numerical tolerance check.

    Only raised when explicit validation is requested (see
    ``featm.gpu.dispatch.dispatch_one_feature(..., validate=True)``, used by
    tests and the Kaggle notebook) -- normal production dispatch does not
    pay the cost of a redundant CPU computation on every call.
    """


class GPUBackendError(GPUError):
    """A GPU kernel raised an error that is not OOM, not a dtype problem, and not "unavailable".

    The catch-all for a genuine bug/failure inside a GPU implementation
    (e.g. a CuPy/Torch internal error unrelated to memory or dtype).
    ``fallback_reason="backend_error"``. Wraps the original exception via
    ``raise ... from exc`` so the underlying traceback is never lost.
    """
