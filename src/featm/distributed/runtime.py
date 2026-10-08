from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator

from .contracts import Partition


class DistributedRuntime(ABC):
    """Minimal runtime contract used by FEATM's distributed coordinator.

    The coordinator owns feature semantics, partition ordering and state
    merging. A runtime only executes pure partition tasks and yields results
    tagged by partition index. This intentionally prevents Dask/Ray-specific
    semantics from leaking into feature definitions.
    """

    name: str

    @abstractmethod
    def map_stream(
        self,
        fn: Callable[[Partition], Any],
        partitions: Iterable[Partition],
        *,
        max_in_flight: int,
        retries: int,
        should_submit_more: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, Any]]:
        """Execute partitions with bounded in-flight work; results may arrive out of order.

        ``max_in_flight`` bounds scheduler-level concurrency (how many tasks
        may be simultaneously submitted/running). It does NOT by itself bound
        how many completed-but-not-yet-consumed results accumulate on the
        caller's side when completions arrive out of logical order -- that is
        a property of how fast the caller drains this stream, not of the
        runtime alone.

        ``should_submit_more``, when given, is consulted immediately before
        each replacement task would be submitted (i.e. each time the caller
        pulls a result via ``next()``). Returning ``False`` skips that one
        submission (the runtime simply runs with fewer concurrently in-flight
        tasks until the caller is ready again) without ending the stream --
        the runtime keeps retrying the check on subsequent pulls. This is how
        a coordinator enforces a bounded logical reorder window: pass a
        callback that returns ``False`` once its own reassembly buffer has
        reached its window size, and it stops the runtime from racing ahead
        and materializing the rest of the dataset while an early partition is
        still outstanding.
        """

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
