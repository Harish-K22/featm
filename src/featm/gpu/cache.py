"""Shared GPU-resident intermediate cache (Step 3).

The GPU analogue of :mod:`featm.cache`: several selected GPU-accelerated
spectral features (centroid, dominant frequency, flatness, ...) depend on
the same FFT of the same batch. Computing that FFT once per batch instead of
once per feature is the entire point of the "shared computational primitives"
strategy in the Step 3 spec -- doing it wrong (once per feature) would still
be *correct*, just needlessly slow, defeating a large fraction of any
expected GPU speedup for a multi-feature selection.

Design mirrors :mod:`featm.cache` deliberately (same scoping/eviction
shape), with two GPU-specific differences:

- entries are explicitly released (backend-appropriate: dropping the CuPy/
  Torch reference so the memory pool can reclaim it) when a scope closes,
  not just left for the interpreter's GC -- device memory is a much scarcer,
  harder-bounded resource than host RAM, so scope-exit cleanup is a
  correctness property here, not an optimization;
- size accounting uses each array's real device-resident byte count
  (``.nbytes`` for both CuPy and Torch tensors), so the GPU cache budget is
  actually VRAM-denominated, not host-RAM-denominated.
"""
from __future__ import annotations

import contextvars
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Iterator

#: Deliberately small relative to featm.cache's 256MB default -- VRAM is a
#: much scarcer resource, and this cache exists only to avoid recomputing an
#: FFT within ONE batch's feature selection, not to retain data across
#: batches (each batch gets its own fresh scope, same as the CPU cache).
_DEFAULT_GPU_CACHE_BUDGET_BYTES = 64 * 1024 * 1024

_CURRENT: contextvars.ContextVar["GPUIntermediateCache | None"] = contextvars.ContextVar(
    "featm_gpu_current_cache", default=None
)


def _device_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_device_nbytes(v) for v in value) or 64
    return 64


class GPUIntermediateCache:
    """A bounded, LRU-evicting cache of device-resident arrays for one GPU batch scope."""

    def __init__(self, max_bytes: int = _DEFAULT_GPU_CACHE_BUDGET_BYTES) -> None:
        self.max_bytes = max_bytes
        self._store: "OrderedDict[tuple, tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversized_skips = 0
        self._lock = threading.Lock()

    def get_or_compute(self, key: tuple, compute_fn: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._store.get(key)
            if hit is not None:
                self.hits += 1
                self._store.move_to_end(key)
                return hit[0]

        value = compute_fn()
        nbytes = max(_device_nbytes(value), 1)

        with self._lock:
            self.misses += 1
            if nbytes > self.max_bytes:
                self.oversized_skips += 1
                return value
            self._store[key] = (value, nbytes)
            self._store.move_to_end(key)
            self._bytes += nbytes
            self._evict_if_needed()
        return value

    def _evict_if_needed(self) -> None:
        """Evict oldest (LRU) entries ONE AT A TIME until back under budget.

        Audit-fix pass, Finding 9: the prior version never decremented
        ``self._bytes`` inside the loop, so once the cache crossed
        ``max_bytes`` the loop condition stayed true (self._bytes unchanged)
        until the store was completely empty -- evicting EVERY entry instead
        of only the oldest ones needed to fall back under budget, silently
        defeating the documented bounded-LRU behavior (not a numerical-output
        bug: an over-evicted cache just recomputes more than necessary, but
        it contradicts what this class is documented to do).
        """
        while self._bytes > self.max_bytes and self._store:
            _, (_, evicted_nbytes) = self._store.popitem(last=False)
            self._bytes -= evicted_nbytes
            self.evictions += 1

    def clear(self) -> None:
        """Drop every reference this scope holds, so the GPU allocator can reclaim the memory.

        Explicit clearing (rather than waiting for the scope's context
        manager to simply go out of scope and rely on Python's refcounting
        GC) matters here because a CuPy/Torch memory pool typically only
        returns freed blocks to *itself* for reuse, not to the OS/driver --
        the pool's own release/empty-cache call (done by the caller of a
        batch scope, not by this cache) is what actually matters for
        cross-batch VRAM growth; this method's job is just to make sure
        nothing in THIS cache is still holding a reference when that happens.
        """
        with self._lock:
            self._store.clear()
            self._bytes = 0

    @property
    def size_bytes(self) -> int:
        return self._bytes

    @property
    def entry_count(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
            "oversized_skips": self.oversized_skips, "entries": self.entry_count, "bytes": self._bytes,
        }


def current_gpu_cache() -> GPUIntermediateCache | None:
    return _CURRENT.get()


@contextmanager
def scoped_gpu_cache(*, max_bytes: int = _DEFAULT_GPU_CACHE_BUDGET_BYTES) -> Iterator[GPUIntermediateCache]:
    """Open one fresh GPU cache scope (one GPU batch), guaranteed to clear its own entries on exit.

    Never shared or nested across batches or datasets -- a fresh
    :class:`GPUIntermediateCache` every call, exactly like
    :func:`featm.cache.scoped_cache`, so a device-resident array computed
    for one batch can never be silently reused for a different batch's data.
    """
    c = GPUIntermediateCache(max_bytes=max_bytes)
    token = _CURRENT.set(c)
    try:
        yield c
    finally:
        _CURRENT.reset(token)
        c.clear()


def gpu_cache_budget_bytes(gpu_budget: Any, *, default_bytes: int = _DEFAULT_GPU_CACHE_BUDGET_BYTES) -> int:
    """Derive the shared-intermediate cache cap from a resolved ``GPUMemoryBudget`` (Finding 1).

    Audit-fix pass: previously the GPU cache always used a fixed 64MiB cap
    regardless of any declared ``gpu_memory_budget=`` -- a user declaring a
    1MB total GPU budget could still get a nominal 64MiB cache allowance.
    The cache cap is now clamped to never exceed the declared/resolved total
    GPU working-memory budget, so a tiny declared budget cannot create a
    larger cache allowance than the budget itself. Duck-typed on
    ``gpu_budget.available``/``gpu_budget.budget_bytes`` (accepts
    ``featm.gpu.memory.GPUMemoryBudget`` or ``None``) rather than importing
    that class directly, to avoid a cache.py <-> memory.py import cycle.
    """
    if gpu_budget is None or not gpu_budget.available or gpu_budget.budget_bytes <= 0:
        return default_bytes
    return max(1, min(default_bytes, int(gpu_budget.budget_bytes)))


def gpu_array_cache_key(namespace: str, batch_id: int, shape: tuple, dtype: str, **params: Any) -> tuple:
    """A cheap (no device-to-host transfer) cache key for one GPU batch's intermediate.

    Unlike :func:`featm.cache.array_cache_key`, this does NOT hash array
    content -- doing so would require a device-to-host copy of the very
    data the cache exists to keep on-device, defeating the point. Correctness
    instead comes from scoping (a fresh cache per batch, see
    :func:`scoped_gpu_cache`) plus ``batch_id`` (an caller-supplied
    monotonic counter unique within one run), not content hashing.
    """
    return (namespace, batch_id, shape, dtype, tuple(sorted(params.items())))
