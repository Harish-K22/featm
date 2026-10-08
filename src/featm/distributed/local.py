from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, Iterator

from .contracts import DistributedTaskError, Partition
from .runtime import DistributedRuntime


class LocalThreadRuntime(DistributedRuntime):
    """Standard-library reference runtime used for semantic/unit validation.

    It is intentionally not marketed as the Step 4 performance backend. Its
    purpose is to exercise the exact same partition/state-merge coordinator
    without requiring Dask to be installed, so the distributed semantics can
    be tested independently of a third-party scheduler.
    """

    name = "local"

    def __init__(self, *, n_workers: int = 4) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.n_workers = int(n_workers)
        self._executor = ThreadPoolExecutor(max_workers=self.n_workers, thread_name_prefix="featm-dist")

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
        iterator = iter(partitions)
        pending: dict[Any, tuple[Partition, int]] = {}
        exhausted = False

        def submit_one(part: Partition, attempt: int = 0) -> None:
            pending[self._executor.submit(fn, part)] = (part, attempt)

        while len(pending) < max_in_flight:
            try:
                submit_one(next(iterator))
            except StopIteration:
                exhausted = True
                break

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                part, attempt = pending.pop(fut)
                try:
                    result = fut.result()
                except BaseException as exc:
                    if attempt < retries:
                        submit_one(part, attempt + 1)
                        continue
                    raise DistributedTaskError(part.index, exc) from exc
                yield part.index, result
                # Replenishment is gated on the caller's window, not just on
                # scheduler concurrency: retrieving one result does not by
                # itself mean it is safe to dispatch another, because the
                # caller may be holding a completed-but-unconsumable (out of
                # logical order) backlog it does not want to grow further.
                # Skipping a submission here just runs with fewer than
                # max_in_flight tasks outstanding for a while -- it never ends
                # the stream early, and the check is retried on every
                # subsequent pull.
                if exhausted or (should_submit_more is not None and not should_submit_more()):
                    continue
                try:
                    submit_one(next(iterator))
                except StopIteration:
                    exhausted = True

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
