"""GPU memory (VRAM) budget and batch planning (Step 3).

A real, separate budget from host RAM/disk/cache -- Step 3 spec §10 is
explicit that these must never be conflated. :class:`GPUMemoryBudget` is the
VRAM analogue of :class:`featm.resources.ResourceBudget`; :func:`plan_gpu_batch_rows`
is the VRAM analogue of :func:`featm.resources.plan_batch_rows`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..resources import parse_size
from . import backend as gpu_backend

#: Never plan against 100% of reported free VRAM -- other processes on a
#: shared GPU (common on Kaggle), the CUDA context's own fixed overhead, and
#: memory-pool fragmentation all eat into "free" without featm's own
#: allocations growing. This is deliberately more conservative than the host
#: RAM planner's 0.65 default fraction: VRAM exhaustion is a hard kernel
#: failure (OOM), not graceful degradation, and is typically a much smaller
#: absolute pool (single-digit-to-double-digit GB) than host RAM.
DEFAULT_GPU_MEMORY_TARGET_FRACTION = 0.5
#: A GPU batch is halved on OOM down to this floor before giving up --
#: mirrors featm.execution._MIN_RETRY_ROWS's "shrink to 1 row before
#: declaring failure" policy, applied to the GPU retry loop.
MIN_GPU_RETRY_ROWS = 1
#: Bounded retry: halving from a few million rows reaches 1 row in ~20
#: halvings; this generous cap exists only to guarantee termination if a
#: batch size is somehow enormous, never to be the *expected* retry count
#: for a well-planned batch (Step 3 spec §11: "no infinite retry loops").
MAX_GPU_OOM_RETRIES = 24
#: Fallback threshold for backend="auto" when a feature does not declare its
#: own gpu_min_rows -- conservative on purpose (Step 3 spec §15: "avoid
#: transfer overhead traps... it is acceptable for v1 to use conservative
#: per-family thresholds"). Per-family defaults are set explicitly at
#: registration time in featm.gpu.registration; this constant is only the
#: last-resort fallback for a spec that somehow has neither.
DEFAULT_GPU_MIN_ROWS = 5_000
#: Audit-fix pass (Finding 10): every ``gpu_min_rows`` value currently set at
#: registration time was calibrated against real Kaggle Tesla P100 benchmarks
#: run at a FIXED sequence length of 64 columns/timesteps
#: (``scripts/build_gpu_validation_notebook.py`` Section J:
#: ``Xb = rng.standard_normal((n, 64))``). A row-only threshold silently
#: assumes every input has that same width, which is false in general (a
#: feature selected over 10 rows x 1,000,000 steps is a huge workload that a
#: row-only check would wrongly force onto CPU; 8,000 rows x 4 steps is a
#: tiny workload a row-only check would wrongly send to the GPU). Multiplying
#: a spec's ``gpu_min_rows`` by this constant recovers the ELEMENT-count
#: threshold that was actually being tested at calibration time, so
#: ``backend="auto"`` can compare ``n_rows * n_columns`` against it instead of
#: ``n_rows`` alone -- see ``estimate_gpu_min_elements``.
GPU_MIN_ROWS_CALIBRATION_SEQ_LEN = 64


@dataclass(frozen=True)
class GPUMemoryBudget:
    """A resolved VRAM working-memory envelope for one extraction run.

    ``total_bytes``/``free_bytes_at_plan`` are a snapshot taken once, at plan
    time (real free VRAM, not a fixed nominal card size) -- reported back to
    the caller (Step 3 spec §10: ``gpu_total_memory``/``gpu_free_memory_at_plan``)
    so a user can see exactly what the planner saw, not just the budget it derived.
    """

    backend: str | None
    device_index: int
    budget_bytes: int
    total_bytes: int | None
    free_bytes_at_plan: int | None
    target_fraction: float

    @classmethod
    def unavailable(cls) -> "GPUMemoryBudget":
        """A zero-budget placeholder for "no GPU backend usable" -- never used to plan a GPU batch."""
        return cls(backend=None, device_index=0, budget_bytes=0, total_bytes=None, free_bytes_at_plan=None, target_fraction=0.0)

    @property
    def available(self) -> bool:
        return self.backend is not None

    @classmethod
    def resolve(
        cls,
        *,
        gpu_memory_budget: "str | int | None" = None,
        device: "str | int | None" = None,
        backend: str | None = None,
        target_fraction: float = DEFAULT_GPU_MEMORY_TARGET_FRACTION,
    ) -> "GPUMemoryBudget":
        """Resolve a GPU budget: real free VRAM (with a safety margin) unless overridden explicitly.

        ``gpu_memory_budget="auto"`` (or ``None``) inspects real free VRAM via
        the active backend; an explicit size string (``"8GB"``) or int
        overrides it outright, still capped to what the device actually
        reports as free (a budget larger than physically available VRAM is
        not honored silently -- see the clamp below).
        """
        info = gpu_backend.device_info(device, backend=backend)
        if info is None:
            return cls.unavailable()
        requested = None if gpu_memory_budget in (None, "auto") else parse_size(gpu_memory_budget)
        if requested is None:
            budget = int(info.free_bytes * target_fraction)
        else:
            # An explicit request larger than currently-free VRAM is clamped,
            # not honored -- planning against memory that is not there would
            # just move the failure from "planning" to "the first allocation".
            budget = min(requested, info.free_bytes)
        return cls(
            backend=info.backend, device_index=info.index, budget_bytes=max(budget, 0),
            total_bytes=info.total_bytes, free_bytes_at_plan=info.free_bytes, target_fraction=target_fraction,
        )

    def report_fields(self) -> dict:
        return {
            "gpu_total_memory": self.total_bytes,
            "gpu_free_memory_at_plan": self.free_bytes_at_plan,
            "gpu_budget": self.budget_bytes,
            "gpu_backend": self.backend,
            "gpu_device_index": self.device_index,
        }


#: Feature names whose kernel computes a shared FFT (``featm.gpu.kernels.
#: spectral.gpu_spectral_parts``) -- these coexist with several more
#: same-length device arrays per row than a plain reduction does (see
#: :func:`estimate_gpu_bytes_per_row`'s docstring for the exact list), so a
#: single universal ``3 * (in + out)`` estimate materially UNDERESTIMATES
#: their real working set (Finding 1, audit-fix pass). Populated lazily (not
#: at import time) to avoid a hard import-order dependency between
#: ``gpu/memory.py`` and ``gpu/kernels/spectral.py``.
def _spectral_feature_names() -> frozenset[str]:
    from .kernels.spectral import SPECTRAL_KERNELS, STANDALONE_SPECTRAL_KERNELS

    return frozenset(SPECTRAL_KERNELS) | frozenset(STANDALONE_SPECTRAL_KERNELS)


def estimate_gpu_bytes_per_row(spec: Any, *, n_columns_in: int, itemsize: int) -> int:
    """Estimate device working-set bytes for ONE input row of ``spec``, family-aware.

    Not a single universal ``3 * (in + out)`` multiplier for every family
    (Finding 1): a plain elementwise reduction (``featm.gpu.kernels.
    reductions``) only ever holds the input row and a scalar output on-device
    at once, so ``3x`` (input + a little headroom for the reduction's own
    intermediate + output) is already conservative. A shared-FFT spectral
    kernel (``featm.gpu.kernels.spectral.gpu_spectral_parts``) coexists with
    several more same-length arrays per batch: the raw input, the
    mean-centered input, the complex ``rfft`` output (2 reals per complex
    element), the magnitude, the power, the frequency vector, and the
    feature's own output columns -- all at once, because
    :func:`featm.gpu.cache.scoped_gpu_cache` deliberately keeps them
    device-resident for reuse across every spectral feature selected
    together. Approximating that coexistence as roughly ``8 columns`` worth
    of ``n_columns_in``-sized arrays (input, centered, 2x complex-spectrum
    reals, magnitude, power, freqs, plus ~2 columns of output headroom) is
    conservative without being wasteful; used only to PLAN a batch size
    before allocating, never as a promise that this is a byte-exact CUDA
    working-set model (see :class:`GPUMemoryBudget`'s own docstring on that
    same honesty requirement).
    """
    n_columns_in = max(int(n_columns_in), 1)
    if spec.name in _spectral_feature_names():
        multiplier = 8
    else:
        multiplier = 3
    return max(n_columns_in * itemsize * multiplier, itemsize)


def estimate_gpu_min_elements(spec: Any) -> int:
    """Element-count (``n_rows * n_columns``) threshold for ``backend="auto"`` (Finding 10).

    Derived from ``spec.gpu_min_elements`` when a feature declares one
    explicitly, else from ``spec.gpu_min_rows`` (or
    :data:`DEFAULT_GPU_MIN_ROWS` when neither is set) multiplied by
    :data:`GPU_MIN_ROWS_CALIBRATION_SEQ_LEN` -- see that constant's docstring
    for exactly what was measured to produce it.
    """
    explicit = getattr(spec, "gpu_min_elements", None)
    if explicit is not None:
        return int(explicit)
    min_rows = spec.gpu_min_rows if spec.gpu_min_rows is not None else DEFAULT_GPU_MIN_ROWS
    return int(min_rows) * GPU_MIN_ROWS_CALIBRATION_SEQ_LEN


def plan_gpu_batch_rows(
    *, spec: Any, n_rows: int, n_columns_in: int, budget: GPUMemoryBudget, itemsize: int = 4,
) -> int:
    """Pick a GPU batch row count that proactively fits ``budget`` for ``spec``, before any allocation.

    Returns ``0`` when even a single row's estimated working set exceeds the
    budget (the caller's signal to treat this as ``fallback_reason=
    "insufficient_vram"`` -- see ``gpu/dispatch.py`` -- rather than attempting
    a GPU allocation that is already known not to fit). Returns ``n_rows``
    unchanged (no proactive constraint) when no real budget could be
    determined (``budget.available`` is False -- no usable GPU backend to
    query in the first place).

    This is the function the Step 3 audit-fix pass found declared but never
    called by production dispatch (Finding 1); it is now the single place
    that turns a resolved :class:`GPUMemoryBudget` into an actual row count,
    called from ``featm.gpu.dispatch.dispatch_one_feature`` before any
    device allocation happens, not merely after a CUDA OOM.
    """
    if not budget.available or budget.budget_bytes <= 0:
        return int(n_rows)
    bytes_per_row = estimate_gpu_bytes_per_row(spec, n_columns_in=n_columns_in, itemsize=itemsize)
    if bytes_per_row > budget.budget_bytes:
        return 0
    rows = budget.budget_bytes // bytes_per_row
    return max(1, min(int(n_rows), int(rows)))
