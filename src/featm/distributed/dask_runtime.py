from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from .contracts import DistributedRuntimeUnavailableError, DistributedTaskError, Partition
from .runtime import DistributedRuntime


class DaskDistributedRuntime(DistributedRuntime):
    """Lazy Dask Distributed adapter.

    FEATM owns partition semantics and state merging; Dask is only the task
    transport/scheduler. No Dask import occurs on ordinary ``import featm``.
    """

    name = "dask"

    def __init__(
        self,
        *,
        n_workers: int | None = None,
        threads_per_worker: int = 1,
        scheduler_address: str | None = None,
        processes: bool = True,
        client: Any | None = None,
    ) -> None:
        try:
            from distributed import Client, LocalCluster
        except ImportError as exc:
            raise DistributedRuntimeUnavailableError(
                "Dask distributed execution requires the optional dependency: pip install 'featm[distributed]'"
            ) from exc
        self._owns_client = client is None
        self._cluster = None
        if client is not None:
            self.client = client
        elif scheduler_address:
            self.client = Client(scheduler_address)
        else:
            self._cluster = LocalCluster(
                n_workers=n_workers or 2,
                threads_per_worker=threads_per_worker,
                processes=processes,
                dashboard_address=None,
            )
            self.client = Client(self._cluster)

    def map_stream(
        self,
        fn: Callable[[Partition], Any],
        partitions: Iterable[Partition],
        *,
        max_in_flight: int,
        retries: int,
        should_submit_more: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, Any]]:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        try:
            from distributed import as_completed
        except ImportError as exc:  # pragma: no cover - constructor already checks
            raise DistributedRuntimeUnavailableError("distributed is unavailable") from exc
        iterator = iter(partitions)
        ac = as_completed([], with_results=True, raise_errors=False)
        future_to_index: dict[Any, int] = {}
        exhausted = False

        def add_one(part: Partition) -> bool:
            fut = self.client.submit(fn, part, pure=False, retries=retries)
            future_to_index[fut] = part.index
            ac.add(fut)
            return True

        submitted = 0
        while submitted < max_in_flight:
            try:
                add_one(next(iterator)); submitted += 1
            except StopIteration:
                exhausted = True
                break

        outstanding = submitted
        while outstanding:
            fut, result = next(ac)
            outstanding -= 1
            idx = future_to_index.pop(fut)
            # A permanently-failed future (retries exhausted) surfaces here
            # instead of raising. IMPORTANT, version-specific, found by real
            # Dask validation (not assumable from the docs alone): on this
            # installed distributed version, `with_results=True,
            # raise_errors=False` does NOT hand back a bare exception
            # instance -- it hands back the classic `sys.exc_info()`-style
            # 3-tuple `(exc_type, exc_instance, traceback)`. An
            # `isinstance(result, BaseException)` check alone silently MISSES
            # this (the result is a tuple, not an exception), letting the
            # error tuple flow downstream as if it were real partition
            # output -- a silent-corruption bug, not a crash, so it would
            # never have been caught without deliberately testing a
            # permanent-failure case against the real installed version. Both
            # shapes are handled defensively since neither is documented as a
            # stable cross-version guarantee.
            exc = None
            if isinstance(result, BaseException):
                exc = result
            elif (
                isinstance(result, tuple) and len(result) == 3
                and isinstance(result[0], type) and issubclass(result[0], BaseException)
                and isinstance(result[1], BaseException)
            ):
                exc = result[1]
            elif fut.status == "error":
                # Last-resort fallback: ask the future itself rather than trust the
                # shape of what as_completed handed back.
                try:
                    fut.result()
                except BaseException as raised:
                    exc = raised
            if exc is not None:
                raise DistributedTaskError(idx, exc)
            yield idx, result
            # See LocalThreadRuntime.map_stream for why this is gated on the
            # caller's window rather than unconditional: submitting a
            # replacement here every time regardless of whether the caller
            # can use it is exactly what lets Dask's scheduler race ahead and
            # materialize the whole dataset while one slow early partition is
            # still outstanding.
            if exhausted or (should_submit_more is not None and not should_submit_more()):
                continue
            try:
                add_one(next(iterator)); outstanding += 1
            except StopIteration:
                exhausted = True

    def close(self) -> None:
        if self._owns_client:
            try:
                self.client.close()
            finally:
                if self._cluster is not None:
                    self._cluster.close()
