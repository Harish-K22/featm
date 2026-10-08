"""GPU/CPU backend dispatch for one selected feature (Step 3).

:func:`dispatch_one_feature` is the single place that decides, for one
selected `FeatureSpec` and one batch of input, whether to run the CPU
reference implementation or a GPU kernel -- and, if GPU, drives the CUDA-OOM
halve-and-retry loop. Both the small-data path (`featm.engine.extract`) and
the out-of-core path (`featm.execution`) call this same function, so backend
policy can never diverge between them.

Backend policy (Step 3 spec §4):

- ``backend="cpu"`` -- never touches a GPU kernel, ever.
- ``backend="gpu"`` -- tries the GPU implementation; ``gpu_fallback="strict"``
  raises/propagates (per ``errors=``) if that is not possible for any reason;
  ``gpu_fallback="fallback"`` (default) falls back to CPU instead, recording
  exactly why.
- ``backend="auto"`` -- uses GPU only when a real CUDA device is available,
  the feature has a verified GPU implementation, AND the batch is large
  enough (``spec.gpu_min_rows``) for GPU execution to plausibly help;
  otherwise CPU, silently in the sense of "no error", but never silently in
  the sense of "unreported" -- see ``BackendInfo.fallback_reason``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from ..features._utils import as_2d_series
from ..registry import BatchSemantics, FeatureSpec
from . import backend as gpu_backend
from . import memory as gpu_memory
from .exceptions import (
    GPUBackendError,
    GPUDtypeUnsupportedError,
    GPUImplementationMissingError,
    GPUNumericalValidationError,
    GPUOutOfMemoryError,
    GPUUnavailableError,
)
from .kernels.reductions import REDUCTION_KERNELS
from .kernels.spectral import SPECTRAL_KERNELS, STANDALONE_SPECTRAL_KERNELS, VECTOR_SPECTRAL_FEATURES

#: BatchSemantics values for which splitting a GPU batch on OOM is
#: mathematically safe -- exactly Step 2's row-alignment-safe set (see
#: featm.execution._ROW_ALIGNED_BATCH_SEMANTICS), reused rather than
#: reinvented: a feature already proven safe to split for CPU MemoryError
#: retry is safe to split for GPU CUDA-OOM retry for the identical reason
#: (its result for one row does not depend on which other rows share its batch).
_GPU_SPLIT_SAFE_SEMANTICS = (BatchSemantics.ROW_INDEPENDENT, BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED)

#: Fallback reason vocabulary (Step 3 spec §12) -- kept as a module-level
#: tuple (not an Enum) so `BackendInfo.fallback_reason` stays a plain,
#: report/JSON-friendly string.
FALLBACK_REASONS = (
    "gpu_unavailable", "feature_not_gpu_capable", "below_gpu_threshold", "unsupported_dtype",
    "insufficient_vram", "cuda_oom_after_retries", "optional_gpu_dependency_missing", "backend_error",
)


@dataclass
class BackendInfo:
    """Per-feature backend outcome -- the fields the Step 3 spec requires the report to expose (§12)."""

    requested_backend: str
    actual_backend: str
    gpu_attempted: bool = False
    gpu_retries: int = 0
    cpu_fallback: bool = False
    fallback_reason: str | None = None
    gpu_time_seconds: float | None = None
    cpu_time_seconds: float | None = None
    gpu_backend_name: str | None = None  # "cupy" | "torch" | None
    input_dtype: str | None = None
    validation: "dict | None" = None  # populated only when validate=True
    #: Audit-fix pass, Finding 4: which backend this feature's kernel actually
    #: REQUIRES (``spec.gpu_backend``) -- reported even on failure, so a
    #: report can say e.g. "required cupy, only torch was available" instead
    #: of a bare "gpu_unavailable" that does not name what was missing.
    required_gpu_backend: str | None = None
    #: Audit-fix pass, Finding 1: real VRAM planning fields -- populated
    #: whenever a GPU backend/device was resolved at all (even on a fallback
    #: caused by insufficient VRAM), never merely on a successful GPU run.
    gpu_total_memory: int | None = None
    gpu_free_memory_at_plan: int | None = None
    gpu_budget: int | None = None
    gpu_device_index: int | None = None
    gpu_planned_batch_rows: int | None = None

    def as_dict(self) -> dict:
        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "gpu_attempted": self.gpu_attempted,
            "gpu_retries": self.gpu_retries,
            "cpu_fallback": self.cpu_fallback,
            "fallback_reason": self.fallback_reason,
            "gpu_time_seconds": self.gpu_time_seconds,
            "cpu_time_seconds": self.cpu_time_seconds,
            "gpu_backend_name": self.gpu_backend_name,
            "validation": self.validation,
            "input_dtype": self.input_dtype,
            "required_gpu_backend": self.required_gpu_backend,
            "gpu_total_memory": self.gpu_total_memory,
            "gpu_free_memory_at_plan": self.gpu_free_memory_at_plan,
            "gpu_budget": self.gpu_budget,
            "gpu_device_index": self.gpu_device_index,
            "gpu_planned_batch_rows": self.gpu_planned_batch_rows,
        }


def gpu_capable(spec: FeatureSpec) -> bool:
    """Whether this feature has a real, registered GPU kernel (not just ``gpu_available=True``)."""
    return bool(spec.gpu_available) and (
        spec.name in REDUCTION_KERNELS or spec.name in SPECTRAL_KERNELS or spec.name in STANDALONE_SPECTRAL_KERNELS
    )


def _oom_exception_types() -> tuple[type, ...]:
    """Backend-native OOM exception types, imported lazily (never at module load)."""
    types: list[type] = []
    try:
        import cupy

        types.append(cupy.cuda.memory.OutOfMemoryError)
    except Exception:
        pass
    try:
        import torch

        types.append(torch.cuda.OutOfMemoryError)
    except Exception:
        pass
    return tuple(types)


def _run_cpu(spec: FeatureSpec, X: Any, params: Mapping[str, Any]) -> Any:
    fn = spec.implementation
    kwargs = dict(spec.default_params)
    kwargs.update(params or {})
    result = fn(X, **kwargs)
    if spec.modality == "tabular" and isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    return result


def gpu_input_dtype(spec, values):
    sensitive = spec.name in SPECTRAL_KERNELS or spec.name in {
        "timeseries.statistical.std", "timeseries.statistical.range",
        "signal.time_domain.peak_to_peak", "signal.time_domain.line_length",
    }
    if spec.gpu_output_dtype == "float64" or (sensitive and np.asarray(values).dtype.itemsize > 4):
        return "float64"
    with np.errstate(over="ignore", under="ignore"):
        cast = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if np.any(finite & (~np.isfinite(cast) | ((cast == 0) & (values != 0)))):
        return "float64"
    return "float32"


def _run_gpu_kernel(
    spec: FeatureSpec, a_np: np.ndarray, xp_backend: str, *, cache_id: tuple, device_index: int, params: Mapping[str, Any],
) -> Any:
    """Transfer, compute, transfer back -- for exactly one (already-2D, float64) numpy batch.

    Raises the backend-native OOM exception type straight through (the
    caller -- :func:`_dispatch_gpu_with_retry` -- is the one that catches and
    reacts to it); any other backend exception is wrapped as
    :class:`GPUBackendError` so a genuine kernel bug is never confused with
    an OOM or a dtype problem.

    ``spec.gpu_output_dtype`` decides the device transfer dtype (Kaggle
    validation, first run: a real Tesla P100 run found ``timeseries.spectral.
    fft``'s phase columns and ``timeseries.spectral.cepstrum`` genuinely need
    float64 -- not a coding bug, a real precision-sensitivity investigation,
    see docs/SCALING.md's GPU section and STEP3_GPU_HANDOFF.md §D for the
    root-cause writeup). Every other kernel this milestone still computes in
    float32 by default -- this is not a blanket precision upgrade.

    ``cache_id`` (audit-fix pass, Finding 2) identifies this exact logical
    sub-batch for the shared GPU intermediate cache -- see
    :func:`_dispatch_gpu_with_retry`'s docstring for why a plain ``batch_id``
    is not unique enough once CUDA-OOM retry (or Finding 1's proactive VRAM
    chunking) splits one batch into several differently-ranged sub-batches
    that would otherwise collide on ``(batch_id, shape, dtype)`` alone.

    Runs inside ``gpu_backend.device_context(xp_backend, device_index)``
    (audit-fix pass, Finding 3) so the transfer, the kernel, and any
    shared-cache allocation the kernel triggers all happen on the SAME
    physical device that was planned against, not whatever device happens to
    be "current" for the process.
    """
    xp = gpu_backend.xp_module(xp_backend)
    device_dtype = getattr(xp, gpu_input_dtype(spec, a_np))

    with gpu_backend.device_context(xp_backend, device_index):
        device_arr = xp.asarray(a_np, dtype=device_dtype)

        kwargs = dict(spec.default_params)
        kwargs.update(params or {})

        if spec.name in REDUCTION_KERNELS:
            kernel = REDUCTION_KERNELS[spec.name]
            out = kernel(device_arr, xp)
            names = None
        elif spec.name in SPECTRAL_KERNELS:
            kernel = SPECTRAL_KERNELS[spec.name]
            raw = kernel(device_arr, xp, batch_id=cache_id, **kwargs)
            if spec.name in VECTOR_SPECTRAL_FEATURES:
                out, names = raw
            else:
                out, names = raw, None
        elif spec.name in STANDALONE_SPECTRAL_KERNELS:
            kernel = STANDALONE_SPECTRAL_KERNELS[spec.name]
            out = kernel(device_arr, xp)
            names = None
        else:
            raise GPUImplementationMissingError(f"{spec.name} has no registered GPU kernel")

        host_out = xp.asnumpy(out) if xp_backend == "cupy" else out.detach().cpu().numpy()
    return host_out.astype(np.float64, copy=False), names


def _dispatch_gpu_with_retry(
    spec: FeatureSpec, a_np: np.ndarray, xp_backend: str, *, params: Mapping[str, Any], info: BackendInfo,
    batch_id: int, device_index: int, row_offset: int = 0, max_retries: int = gpu_memory.MAX_GPU_OOM_RETRIES,
) -> Any:
    """Run the GPU kernel over ``a_np``, halving on CUDA OOM while mathematically safe.

    Splitting is only attempted when the feature's ``batch_semantics`` is in
    the row-alignment-safe set (see module docstring) -- exactly Step 2's
    rule, applied to VRAM instead of host RAM. A not-safe-to-split feature
    that OOMs is never chunked; the exception simply propagates to the
    caller, which treats it like any other GPU failure (fallback or raise,
    per policy) -- never silently recovered by changing what the feature
    computes.

    ``row_offset`` (audit-fix pass, Finding 2 -- the spectral cache-key
    collision bug) is this sub-batch's absolute row position within the
    ORIGINAL, un-split batch. A naive shared-cache key of
    ``(namespace, batch_id, shape, dtype)`` collides for a 50/50 OOM split:
    both halves have the SAME shape, so the second half would silently reuse
    the first half's cached FFT and return numerically wrong results for
    every row in the second half (independently reproduced: a controlled
    4-row OOM-above-2 test returned ``[x0, x0, x0, x0]`` instead of
    ``[x0, x0, x1, x1]``). Including ``(row_offset, row_offset + len(sub))``
    in the cache key makes every distinct sub-range its own cache entry,
    while still sharing across different FEATURES that hit the identical
    split path for the identical input (the offsets are a deterministic
    function of input size and the OOM trigger point, not random -- two
    spectral features selected together and OOMing at the same size derive
    the same tokens independently and therefore still share one cached FFT
    per sub-range, preserving the whole point of the shared-intermediate cache).
    """
    oom_types = _oom_exception_types()
    split_safe = spec.batch_semantics in _GPU_SPLIT_SAFE_SEMANTICS

    def _attempt(sub: np.ndarray, depth: int, offset: int) -> tuple[np.ndarray, list]:
        cache_id = (batch_id, offset, offset + len(sub))
        try:
            out, names = _run_gpu_kernel(spec, sub, xp_backend, cache_id=cache_id, device_index=device_index, params=params)
            return out, names
        except oom_types as exc:
            info.gpu_retries += 1
            _release_gpu_scratch(xp_backend, device_index)
            if not split_safe or len(sub) <= gpu_memory.MIN_GPU_RETRY_ROWS or depth >= max_retries:
                raise GPUOutOfMemoryError(
                    f"{spec.name}: CUDA out of memory on a batch of {len(sub)} rows"
                    + ("" if split_safe else " (not batch-splittable -- batch_semantics is not row-aligned-safe)")
                    + f", after {info.gpu_retries} retries."
                ) from exc
            mid = max(len(sub) // 2, gpu_memory.MIN_GPU_RETRY_ROWS)
            first, names1 = _attempt(sub[:mid], depth + 1, offset)
            second, names2 = _attempt(sub[mid:], depth + 1, offset + mid)
            if first.ndim == 1:
                return np.concatenate([first, second]), names1
            return np.concatenate([first, second], axis=0), names1

    return _attempt(a_np, 0, row_offset)


def _release_gpu_scratch(xp_backend: str, device_index: int = 0) -> None:
    """Best-effort release of pooled-but-unused device memory after an OOM, before retrying smaller.

    Deliberately NOT called after every ordinary kernel call (Step 3 spec
    §18: "do NOT call expensive global cache-clearing functions after every
    tiny kernel... that can destroy performance") -- only on the OOM
    recovery path, where giving the allocator a chance to reclaim
    already-freed-but-pooled blocks before a smaller retry is exactly the
    right trade. Runs inside the same selected-device context (Finding 3) so
    it frees the pool that was actually allocating, not the process's
    ambient current device.
    """
    try:
        with gpu_backend.device_context(xp_backend, device_index):
            if xp_backend == "cupy":
                import cupy

                cupy.get_default_memory_pool().free_all_blocks()
            elif xp_backend == "torch":
                import torch

                torch.cuda.empty_cache()
    except Exception:
        pass


def _validate_against_cpu(spec: FeatureSpec, cpu_result: Any, gpu_result: np.ndarray, tolerance: tuple[float, float]) -> dict:
    from ..engine import _expand_result

    cpu_expanded = _expand_result(spec.name, cpu_result)
    cpu_arr = np.column_stack(list(cpu_expanded.values())) if len(cpu_expanded) > 1 else next(iter(cpu_expanded.values()))
    cpu_arr = np.asarray(cpu_arr, dtype=float)
    gpu_arr = np.asarray(gpu_result, dtype=float)
    rtol, atol = tolerance
    if cpu_arr.shape != gpu_arr.shape:
        return {"passed": False, "max_abs_error": float("inf"), "max_rel_error": float("inf"), "nan_mask_matches": False, "rtol": rtol, "atol": atol}
    finite_mask = np.isfinite(cpu_arr) & np.isfinite(gpu_arr)
    nan_mask_matches = bool((np.isnan(cpu_arr) == np.isnan(gpu_arr)).all())
    max_abs = float(np.max(np.abs(cpu_arr[finite_mask] - gpu_arr[finite_mask]))) if finite_mask.any() else 0.0
    denom = np.abs(cpu_arr[finite_mask])
    max_rel = float(np.max(np.abs(cpu_arr[finite_mask] - gpu_arr[finite_mask]) / np.where(denom > atol, denom, 1))) if finite_mask.any() else 0.0
    nonfinite_matches = bool((np.isfinite(cpu_arr) == np.isfinite(gpu_arr)).all()
        and (np.isposinf(cpu_arr) == np.isposinf(gpu_arr)).all()
        and (np.isneginf(cpu_arr) == np.isneginf(gpu_arr)).all())
    passed = nan_mask_matches and nonfinite_matches and np.allclose(cpu_arr[finite_mask], gpu_arr[finite_mask], rtol=rtol, atol=atol)
    return {
        "passed": bool(passed), "max_abs_error": max_abs, "max_rel_error": max_rel,
        "nan_mask_matches": nan_mask_matches, "rtol": rtol, "atol": atol,
        "nonfinite_masks_match": nonfinite_matches,
    }


def dispatch_one_feature(
    spec: FeatureSpec,
    X: Any,
    *,
    backend: str = "cpu",
    gpu_fallback: str = "fallback",
    device: "str | int | None" = None,
    gpu_memory_budget: "str | int | None" = None,
    params: Mapping[str, Any] | None = None,
    batch_id: int = 0,
    n_rows_hint: int | None = None,
    validate: bool = False,
) -> tuple[Any, BackendInfo]:
    """Compute one selected feature over ``X``, choosing CPU or GPU per policy. Returns ``(result, info)``.

    ``result`` matches exactly what the CPU implementation would have
    returned for a CPU-backed call (a plain array, or a
    ``featm.registry.FeatureOutput``) -- callers (the small-data ``extract()``
    loop, the out-of-core ``_apply_direct_feature``) do not need to know
    which backend actually ran; only ``info`` differs.
    """
    if backend not in ("cpu", "gpu", "auto"):
        raise ValueError("backend must be cpu, gpu, or auto")
    if gpu_fallback not in ("fallback", "strict"):
        raise ValueError("gpu_fallback must be fallback or strict")
    params = params or {}
    info = BackendInfo(requested_backend=backend, actual_backend="cpu")

    if backend == "cpu":
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info

    # backend in ("gpu", "auto") from here on.
    if not gpu_capable(spec):
        info.fallback_reason = "feature_not_gpu_capable"
        if backend == "gpu" and gpu_fallback == "strict":
            raise GPUImplementationMissingError(
                f"{spec.name} has no GPU implementation and gpu_fallback='strict' was requested; "
                "pass gpu_fallback='fallback' to run it on CPU instead, or select backend='cpu'."
            )
        info.cpu_fallback = backend == "gpu"
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info

    info.required_gpu_backend = spec.gpu_backend

    xp_backend = None
    try:
        # Finding 4 (audit-fix pass): require the SPEC's declared backend,
        # not whatever featm.gpu.backend.preferred_backend() would pick.
        # Before this fix, a CuPy-authored kernel could be silently routed
        # through Torch on a Torch-only machine merely because SOME GPU
        # backend was available -- contradicting the registry's
        # gpu_backend="cupy" contract (real PyTorch is not a drop-in
        # CuPy-array-API replacement for every kernel here). ``spec.
        # gpu_backend`` is None only for a synthetic/test spec; that case
        # still falls through to whatever backend is preferred, matching the
        # pre-fix behavior for the (only) case where "any backend" was ever
        # actually intended.
        xp_backend = gpu_backend.require_gpu(backend=spec.gpu_backend)
    except GPUUnavailableError:
        info.fallback_reason = "gpu_unavailable"
        if backend == "gpu" and gpu_fallback == "strict":
            raise
        info.cpu_fallback = backend == "gpu"
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info
    info.gpu_backend_name = xp_backend

    # Finding 3 (audit-fix pass): validate the requested device index against
    # THIS backend before planning/executing anything against it -- an
    # out-of-range index must fail clearly here, not deep inside a CuPy/Torch
    # call with a confusing low-level error.
    _, device_index = gpu_backend.parse_device(device)
    try:
        gpu_backend.validate_device_index(xp_backend, device_index)
    except GPUUnavailableError:
        info.fallback_reason = "gpu_unavailable"
        if backend == "gpu" and gpu_fallback == "strict":
            raise
        info.cpu_fallback = backend == "gpu"
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info

    a2d = as_2d_series(X)  # reuses the exact CPU shape-validation/error messages
    n_rows = n_rows_hint if n_rows_hint is not None else a2d.shape[0]
    n_columns_in = a2d.shape[1]

    if backend == "auto":
        # Finding 10 (audit-fix pass): a row-only threshold ignores sequence
        # length/width entirely -- 10 rows x 1,000,000 steps is a huge
        # workload a row-only check would wrongly force onto CPU, while
        # 8,000 rows x 4 steps is tiny and would wrongly go to the GPU.
        # gpu.memory.estimate_gpu_min_elements derives an element-count
        # (rows * columns) threshold from the same gpu_min_rows the row-only
        # check used, calibrated at the real sequence width Step 3's Kaggle
        # benchmarks were measured at -- see that function's docstring.
        threshold_elements = gpu_memory.estimate_gpu_min_elements(spec)
        n_elements = n_rows * n_columns_in
        if n_elements < threshold_elements:
            info.fallback_reason = "below_gpu_threshold"
            t0 = time.perf_counter()
            result = _run_cpu(spec, X, params)
            info.cpu_time_seconds = time.perf_counter() - t0
            return result, info

    # Finding 1 (audit-fix pass): resolve AND ACTUALLY USE a real GPU memory
    # budget, against the same device that will execute the kernel. Before
    # this fix, `gpu_memory_budget=` was resolved and then thrown away --
    # the whole host batch was always sent to the GPU regardless of what was
    # declared, relying on physical CUDA OOM as the only backstop.
    budget = gpu_memory.GPUMemoryBudget.resolve(gpu_memory_budget=gpu_memory_budget, device=device_index, backend=xp_backend)
    info.gpu_total_memory = budget.total_bytes
    info.gpu_free_memory_at_plan = budget.free_bytes_at_plan
    info.gpu_budget = budget.budget_bytes
    info.gpu_device_index = budget.device_index

    info.input_dtype = gpu_input_dtype(spec, a2d)
    itemsize = 8 if info.input_dtype == "float64" else 4
    n_total = a2d.shape[0]
    planned_rows = gpu_memory.plan_gpu_batch_rows(
        spec=spec, n_rows=n_total, n_columns_in=n_columns_in, budget=budget, itemsize=itemsize,
    )
    info.gpu_planned_batch_rows = planned_rows
    split_safe = spec.batch_semantics in _GPU_SPLIT_SAFE_SEMANTICS

    if planned_rows == 0:
        # Not even one row's estimated working set fits the declared/resolved
        # budget -- never attempt a device allocation known in advance not to
        # fit; use the dedicated fallback reason, not a bare "backend_error"
        # discovered only after a failed allocation.
        info.fallback_reason = "insufficient_vram"
        if backend == "gpu" and gpu_fallback == "strict":
            raise GPUOutOfMemoryError(
                f"{spec.name}: estimated GPU working set for a single row exceeds the declared/resolved "
                f"GPU budget ({budget.budget_bytes} bytes); refusing to attempt GPU execution."
            )
        info.cpu_fallback = True
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info

    if planned_rows < n_total and not split_safe:
        # The full batch does not fit the declared budget, and this
        # feature's batch_semantics forbids splitting it (Step 2's
        # row-alignment-safety rule, reused for VRAM) -- never silently
        # chunk mathematics that is not safe to chunk. Fail/fallback
        # exactly as if it does not fit at all, rather than attempting a GPU
        # allocation known in advance to exceed the budget.
        info.fallback_reason = "insufficient_vram"
        if backend == "gpu" and gpu_fallback == "strict":
            raise GPUOutOfMemoryError(
                f"{spec.name}: estimated GPU working set for all {n_total} rows exceeds the declared/resolved "
                f"GPU budget, and batch_semantics={spec.batch_semantics.value!r} is not batch-splittable; "
                "refusing to attempt GPU execution."
            )
        info.cpu_fallback = True
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info

    info.gpu_attempted = True
    t0 = time.perf_counter()
    try:
        # Deliberately does NOT open its own scoped_gpu_cache() here -- doing
        # so would give every dispatch_one_feature() call its own fresh,
        # immediately-cleared cache scope, making the shared-FFT primitive
        # never actually shared across several features selected together
        # (a real bug this exact design caught in this pass's own local
        # tests: 3 spectral features computed the FFT 3 times instead of 1).
        # The scope is opened ONCE by the caller -- featm.engine.extract()'s
        # per-call scope, or featm.execution._run_transform_phase's
        # per-batch scope -- exactly mirroring how featm.cache.scoped_cache()
        # is opened once per call/batch on the CPU side, not once per feature.
        if planned_rows >= n_total:
            gpu_out, names = _dispatch_gpu_with_retry(
                spec, a2d, xp_backend, params=params, info=info, batch_id=batch_id, device_index=device_index,
            )
        else:
            # Finding 1: PROACTIVE chunking, sized to fit the declared VRAM
            # budget -- computed and applied BEFORE any allocation is
            # attempted, not only reactively after a CUDA OOM. Each chunk
            # still goes through the OOM-retry path independently (the
            # estimate is conservative, not a byte-exact guarantee -- see
            # estimate_gpu_bytes_per_row's docstring), and each chunk's cache
            # identity is its own absolute row range (Finding 2), so this
            # composes cleanly with the OOM-retry splitting above.
            chunks_out: list[np.ndarray] = []
            names = None
            start = 0
            while start < n_total:
                end = min(start + planned_rows, n_total)
                sub_out, sub_names = _dispatch_gpu_with_retry(
                    spec, a2d[start:end], xp_backend, params=params, info=info, batch_id=batch_id,
                    device_index=device_index, row_offset=start,
                )
                chunks_out.append(sub_out)
                if names is None:
                    names = sub_names
                start = end
            gpu_out = np.concatenate(chunks_out, axis=0) if chunks_out[0].ndim > 1 else np.concatenate(chunks_out)
        info.gpu_time_seconds = time.perf_counter() - t0
        info.actual_backend = "gpu"
    except GPUOutOfMemoryError:
        info.fallback_reason = "cuda_oom_after_retries"
        if backend == "gpu" and gpu_fallback == "strict":
            raise
        info.cpu_fallback = True
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info
    except (GPUDtypeUnsupportedError,) as exc:
        info.fallback_reason = "unsupported_dtype"
        if backend == "gpu" and gpu_fallback == "strict":
            raise
        info.cpu_fallback = True
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info
    except Exception as exc:
        info.fallback_reason = "backend_error"
        if backend == "gpu" and gpu_fallback == "strict":
            raise GPUBackendError(f"{spec.name}: GPU execution failed: {type(exc).__name__}: {exc}") from exc
        info.cpu_fallback = True
        t0 = time.perf_counter()
        result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t0
        return result, info

    if names is not None:
        from ..registry import FeatureOutput

        result: Any = FeatureOutput(gpu_out, names)
    else:
        result = gpu_out

    if validate:
        t1 = time.perf_counter()
        cpu_result = _run_cpu(spec, X, params)
        info.cpu_time_seconds = time.perf_counter() - t1
        info.validation = _validate_against_cpu(spec, cpu_result, gpu_out if names is None else np.asarray(result.as_array()), spec.gpu_numerical_tolerance)
        if not info.validation["passed"]:
            raise GPUNumericalValidationError(
                f"{spec.name}: GPU result outside declared tolerance {spec.gpu_numerical_tolerance} -- "
                f"max_abs_error={info.validation['max_abs_error']:.3g}, max_rel_error={info.validation['max_rel_error']:.3g}"
            )

    return result, info
