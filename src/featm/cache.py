"""Shared intermediate computation cache.

Selected features frequently depend on the same expensive intermediate (one
FFT feeding several spectral descriptors, one Delaunay triangulation feeding
several simplex statistics, one graph Laplacian eigendecomposition feeding
several spectral graph features, and so on). This module gives feature
implementations a way to memoize such intermediates *within one execution
scope* (one batch of one :func:`featm.extract` / engine run) so that
selecting several related features does not recompute shared work.

Design:

- A cache is explicitly scoped via :func:`scoped_cache`, a context manager
  entered once per batch/call. Each scope gets its own cache instance, so
  data from different batches or different datasets can never leak into one
  another -- the cache simply does not outlive its scope.
- Within a scope, callers look up the *current* cache with
  :func:`current_cache` (backed by a ``contextvars.ContextVar`` so it is
  correct across threads) and call :meth:`SharedIntermediateCache.get_or_compute`.
- The cache obeys a byte budget with LRU eviction so a scope cannot grow
  without bound; evicted entries are simply recomputed on next use.
"""
from __future__ import annotations

import contextvars
import hashlib
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np

_DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024  # 256MB per execution scope by default

_CURRENT: contextvars.ContextVar["SharedIntermediateCache | None"] = contextvars.ContextVar(
    "featm_current_cache", default=None
)


def _default_size_estimate(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_default_size_estimate(v) for v in value) or 64
    if isinstance(value, dict):
        return sum(_default_size_estimate(v) for v in value.values()) or 64
    if isinstance(value, (bytes, bytearray, str)):
        return len(value)
    return 64  # crude fallback for scalars/small objects


class SharedIntermediateCache:
    """A bounded, LRU-evicting cache for one execution scope."""

    def __init__(self, max_bytes: int = _DEFAULT_BUDGET_BYTES) -> None:
        self.max_bytes = max_bytes
        self._store: "OrderedDict[tuple, tuple[Any, int]]" = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversized_skips = 0
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        key: tuple,
        compute_fn: Callable[[], Any],
        *,
        size_estimator: Callable[[Any], int] | None = None,
    ) -> Any:
        with self._lock:
            hit = self._store.get(key)
            if hit is not None:
                self.hits += 1
                self._store.move_to_end(key)
                return hit[0]

        # Compute outside the lock: recomputation for an already-cached key by a
        # concurrent caller is wasted work, not a correctness problem, and this
        # avoids holding the lock across a potentially expensive computation.
        value = compute_fn()
        estimator = size_estimator or _default_size_estimate
        nbytes = max(int(estimator(value)), 1)

        with self._lock:
            self.misses += 1
            if nbytes > self.max_bytes:
                # A single value larger than the whole budget must never be
                # retained -- caching it would permanently keep the cache over
                # budget (nothing left to evict it in favor of). Serve it to
                # the caller without storing it; it will simply be recomputed
                # next time. This is the correct trade for a hard budget.
                self.oversized_skips += 1
                return value
            self._store[key] = (value, nbytes)
            self._store.move_to_end(key)
            self._bytes += nbytes
            self._evict_if_needed()
        return value

    def _evict_if_needed(self) -> None:
        # No "leave at least one entry" exception: an entry only ever reaches
        # this point after already passing the oversized-value check above, so
        # evicting everything else is always sufficient to get back under
        # budget without having to evict the entry that was just inserted.
        while self._bytes > self.max_bytes and self._store:
            _, (_, nbytes) = self._store.popitem(last=False)
            self._bytes -= nbytes
            self.evictions += 1

    def clear(self) -> None:
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
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "oversized_skips": self.oversized_skips,
            "entries": self.entry_count,
            "bytes": self._bytes,
        }


def current_cache() -> SharedIntermediateCache | None:
    """Return the cache for the current execution scope, or ``None`` outside one."""
    return _CURRENT.get()


@contextmanager
def scoped_cache(
    cache: SharedIntermediateCache | None = None,
    *,
    max_bytes: int = _DEFAULT_BUDGET_BYTES,
) -> Iterator[SharedIntermediateCache]:
    """Open one fresh cache scope (one batch, or one whole-array ``extract`` call).

    A new :class:`SharedIntermediateCache` is created per scope unless one is
    passed explicitly, guaranteeing intermediates never leak across scopes.
    """
    c = cache if cache is not None else SharedIntermediateCache(max_bytes=max_bytes)
    token = _CURRENT.set(c)
    try:
        yield c
    finally:
        _CURRENT.reset(token)


def array_cache_key(namespace: str, arr: Any, **params: Any) -> tuple:
    """Build a content-addressed cache key for a numpy-convertible array.

    Content hashing (rather than object identity) is used here because arrays
    are frequently re-wrapped/copied between the caller and the feature
    implementation; hashing the actual bytes still gives a correct cache hit
    without relying on the same Python object surviving the whole call.

    ``np.ascontiguousarray(a).tobytes()`` always makes a full copy, which is
    wasteful for a large array whose only purpose here is to be hashed. The
    common case -- an already C-contiguous array -- can be hashed straight
    from its existing buffer via a ``memoryview`` with no copy at all; a copy
    is made only for the genuinely non-contiguous case, where one is
    unavoidable to get a well-defined byte order to hash.
    """
    a = np.asarray(arr)
    buf = a.data if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a).tobytes()
    digest = hashlib.blake2b(buf, digest_size=16).hexdigest()
    return (namespace, digest, a.shape, str(a.dtype), tuple(sorted(params.items())))


def identity_cache_key(namespace: str, obj: Any, **params: Any) -> tuple:
    """Build an object-identity cache key for objects too expensive to hash by content.

    Valid only *within one execution scope*: Python guarantees ``id()`` is
    unique among objects alive at the same time, and a scope's cache never
    outlives the batch whose objects it was built from, so this cannot
    collide across batches or datasets.
    """
    return (namespace, id(obj), tuple(sorted(params.items())))
