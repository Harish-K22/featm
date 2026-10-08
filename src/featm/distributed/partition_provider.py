"""Partition providers: how the coordinator turns a Source into partition tasks.

The default (:class:`CoordinatorStreamPartitionProvider`) is what the Step 4
foundation shipped with: the coordinator itself scans the FEATM ``Source``
and submits materialized ``pandas.DataFrame`` partitions to workers. This is
semantically correct for every ``Source`` type, but the coordinator becomes
the I/O and network bottleneck -- it reads and re-serializes every row,
even ones a worker on a different machine could have read directly.

:class:`DaskParquetPartitionProvider` is the worker-native alternative for
Parquet: the coordinator only reads FOOTER METADATA (row-group row counts,
already cheap and used elsewhere in FEATM -- see
``ParquetSource.estimated_bytes_per_row``) to decide stable partition
boundaries, then hands each worker a small, picklable *descriptor*
(file path + row-group index range + column projection). The worker opens
the Parquet file itself and reads only its assigned row groups. The
coordinator never materializes the full dataset.

Report honestly which provider was used (``"coordinator_streamed"`` or
``"parquet_native"``) -- never claim distributed I/O scalability for the
generic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from ..execution import PreparedRun
from ..sources import ParquetSource
from .contracts import Partition


@dataclass(frozen=True)
class ParquetPartitionDescriptor:
    """A picklable, worker-resolvable description of one Parquet partition.

    Deliberately NOT a materialized DataFrame -- this is what crosses the
    wire to the worker; the worker reads the actual row data itself. Frozen
    and hashable-by-value so it can also serve as a stable partition
    identity contribution for checkpointing (path + row-group range +
    columns fully determines the row content, independent of which process
    reads it or when).
    """

    path: str
    row_groups: tuple[int, ...]
    columns: tuple[str, ...]
    expected_rows: int


class PartitionIntegrityError(RuntimeError):
    """Raised when a resolved worker partition does not match its expected row count.

    Finding 13.2 (independent audit): ``ParquetPartitionDescriptor.expected_rows``
    was carried across the wire but never actually checked against what the
    worker resolved -- decorative, not a real safety invariant. A worker now
    verifies ``len(frame) == expected_rows`` and raises rather than silently
    proceeding with a partition that does not match its own descriptor (a
    corrupted/concurrently-modified file, a metadata/data mismatch, or a
    provider bug would otherwise pass silently).
    """


def _read_parquet_descriptor(desc: ParquetPartitionDescriptor) -> pd.DataFrame:
    """Resolve one descriptor to an actual DataFrame -- runs ON THE WORKER, not the coordinator."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(desc.path)
    cols = list(desc.columns) if desc.columns else None
    tables = [pf.read_row_group(rg, columns=cols) for rg in desc.row_groups]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    frame = table.to_pandas()
    if len(frame) != desc.expected_rows:
        raise PartitionIntegrityError(
            f"Parquet partition (row groups {desc.row_groups}) resolved to {len(frame)} rows, expected "
            f"{desc.expected_rows} from footer metadata at partition-planning time. Refusing to proceed with a "
            f"partition that does not match its own descriptor: {desc.path}"
        )
    return frame


def resolve_partition_frame(frame_or_descriptor: Any) -> pd.DataFrame:
    """Worker-side resolution: a Partition's ``.frame`` may be a real DataFrame or a descriptor to read."""
    if isinstance(frame_or_descriptor, ParquetPartitionDescriptor):
        return _read_parquet_descriptor(frame_or_descriptor)
    return frame_or_descriptor


class CoordinatorStreamPartitionProvider:
    """The generic fallback: coordinator reads via ``Source.iter_batches()`` and ships DataFrames."""

    kind = "coordinator_streamed"

    def __init__(self, prepared: PreparedRun, partition_rows: int) -> None:
        self._prepared = prepared
        self._partition_rows = partition_rows

    def iter_partitions(self, columns: Sequence[Any], *, skip_indices: frozenset = frozenset()) -> Iterator[Partition]:
        for idx, frame in enumerate(self._prepared.src.iter_batches(columns=columns, batch_rows=self._partition_rows)):
            if idx in skip_indices:
                continue
            yield Partition(idx, frame)


class ParquetRowGroupTooLargeError(RuntimeError):
    """Raised (or caught internally by ``select_partition_provider``) when a single physical
    Parquet row group alone exceeds the resolved safe partition size.

    Finding 4 (independent audit): ``DaskParquetPartitionProvider`` groups
    only WHOLE row groups -- it cannot split one. If a single row group has
    (say) 5,000,000 rows but ``partition_rows`` resolved to 50,000, the
    resulting "partition" silently hands a worker 100x the memory Step 2's
    memory planner sized for. PyArrow has no cheap way to read a bounded
    row-slice from WITHIN one row group without decoding the whole group
    first (row groups are parquet's atomic decode unit), so there is no safe
    "preferred" fix available without a much larger reader rewrite. The
    accepted fallback (explicitly permitted by the audit): detect this
    BEFORE any task is submitted and refuse to silently call an oversized
    row group a bounded partition -- fall back to the coordinator-streamed
    provider, which reads and re-chunks rows at the requested granularity
    regardless of how the source file's row groups happen to be sized.
    """


def _max_row_group_rows(path: str) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return max((pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)), default=0)


class DaskParquetPartitionProvider:
    """Worker-native Parquet provider: partition boundaries come from footer metadata only.

    Partition boundaries (which row groups belong to partition ``k``) are
    computed ONCE, independent of which columns a given phase (fit vs.
    transform) needs -- row-group boundaries are a property of the file, not
    of a column projection, so fit and transform naturally agree on the same
    partition index -> row range mapping without extra coordination, exactly
    like the coordinator-streamed path already does today (both phases pass
    the same ``partition_rows`` to independent ``iter_batches()`` calls,
    which happens to align because both start counting from row 0).

    Raises :class:`ParquetRowGroupTooLargeError` at construction time if any
    single row group alone exceeds ``partition_rows`` -- callers should use
    :func:`select_partition_provider`, which catches this and falls back to
    :class:`CoordinatorStreamPartitionProvider` rather than propagate it, but
    the check lives here so direct construction is never silently unsafe.
    """

    kind = "parquet_native"

    def __init__(self, source: ParquetSource, partition_rows: int) -> None:
        self._path = str(source.path)
        self._partition_rows = int(partition_rows)
        max_rg_rows = _max_row_group_rows(self._path)
        if max_rg_rows > self._partition_rows:
            raise ParquetRowGroupTooLargeError(
                f"Parquet file {self._path!r} has a row group with {max_rg_rows:,} rows, larger than the resolved "
                f"safe partition size ({self._partition_rows:,} rows). A single row group cannot be split by this "
                "provider. Use select_partition_provider() to fall back to coordinator-streamed I/O automatically, "
                "or rewrite the file with a smaller row_group_size."
            )
        self._row_groups = self._compute_row_group_partitions()

    def _compute_row_group_partitions(self) -> list[tuple[int, ...]]:
        """Group whole row groups into partitions, never exceeding ``partition_rows``.

        Finding 1 (final closure audit, v0.4.1 -> v0.4.2): the v0.4.1 version
        flushed AFTER adding a row group, so two individually-safe row groups
        (each ``<= partition_rows`` -- guaranteed by ``__init__``'s
        ``ParquetRowGroupTooLargeError`` check) could still be COMBINED into
        one partition whose summed row count exceeds ``partition_rows`` --
        e.g. row groups of 30/30/30 rows with ``partition_rows=50`` produced
        ``[(0, 1), (2,)]``, a 60-row first partition against a 50-row bound.
        Flushing BEFORE adding a row group that would push the running total
        over the bound makes the invariant ``expected_rows <= partition_rows``
        actually hold for every descriptor this provider ever produces
        (except that a single row group is always added to an otherwise-empty
        partition even if, in principle, its own size were exactly at the
        bound -- there is nothing smaller to flush it against, and
        ``__init__`` already guarantees no individual row group exceeds the
        bound in the first place).
        """
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(self._path)
        groups: list[tuple[int, ...]] = []
        current: list[int] = []
        current_rows = 0
        for i in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(i).num_rows
            if current and current_rows + rg_rows > self._partition_rows:
                groups.append(tuple(current))
                current, current_rows = [], 0
            current.append(i)
            current_rows += rg_rows
        if current:
            groups.append(tuple(current))
        return groups

    @property
    def n_partitions(self) -> int:
        return len(self._row_groups)

    def iter_partitions(self, columns: Sequence[Any], *, skip_indices: frozenset = frozenset()) -> Iterator[Partition]:
        for idx, row_groups in enumerate(self._row_groups):
            if idx in skip_indices:
                continue
            expected_rows = sum(
                pf_row_group_rows for pf_row_group_rows in (self._row_group_row_count(rg) for rg in row_groups)
            )
            desc = ParquetPartitionDescriptor(
                path=self._path, row_groups=row_groups, columns=tuple(str(c) for c in columns), expected_rows=expected_rows,
            )
            yield Partition(idx, desc)

    def _row_group_row_count(self, index: int) -> int:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(self._path).metadata.row_group(index).num_rows)


def select_partition_provider(
    prepared: PreparedRun, partition_rows: int
) -> "CoordinatorStreamPartitionProvider | DaskParquetPartitionProvider":
    """Pick the best available provider for this source.

    Parquet gets the worker-native path UNLESS a single physical row group
    would exceed the resolved safe partition size (Finding 4, independent
    audit) -- in that case this falls back to the coordinator-streamed
    provider automatically rather than silently call an oversized row group
    a bounded partition. The fallback is reported honestly via
    ``partition_provider.kind`` (``"coordinator_streamed"``), never
    disguised as ``"parquet_native"``.
    """
    if isinstance(prepared.src, ParquetSource):
        try:
            return DaskParquetPartitionProvider(prepared.src, partition_rows)
        except ParquetRowGroupTooLargeError:
            return CoordinatorStreamPartitionProvider(prepared, partition_rows)
    return CoordinatorStreamPartitionProvider(prepared, partition_rows)
