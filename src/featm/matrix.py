"""FeatureMatrix: the result of an extraction, memory- or disk-backed.

Engineered output is not assumed to fit in RAM. ``FeatureMatrix`` is a
first-class result object that behaves the same way regardless of whether the
underlying data lives in one in-memory DataFrame or in a directory of
Parquet shards on disk: same shape/columns/dtypes introspection, same batch
iteration, same guarded materialization helpers.

Disk-backed storage uses Parquet shards plus one small atomically-written
JSON manifest recording column order, dtypes, row counts per shard, and
enough provenance to detect an incompatible reopen. Column order and row
order are always preserved exactly as written.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAX_MATERIALIZE_ROWS = 2_000_000


class MaterializationGuardError(RuntimeError):
    """Raised when .to_pandas()/.to_numpy() would materialize an unexpectedly large result."""


class DiskBudgetExceededError(RuntimeError):
    """Raised when writing another shard would exceed the configured disk budget."""


class CheckpointCorruptionError(RuntimeError):
    """Raised when a completed checkpoint's declared output shards are missing/corrupt."""


def _blake2b_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_disk_backend_integrity(directory: "str | Path", *, verify_checksums: bool = True) -> list[str]:
    """Check that every shard a manifest declares actually exists and matches what was written.

    Returns a list of problem descriptions (empty means valid). This exists so
    a "done" checkpoint's zero-read fast path can refuse to silently return a
    result if a shard file was deleted, truncated, or replaced after the run
    that wrote it -- ``FeatureMatrix.open()`` itself only checks the manifest
    exists and has a supported version, not that its declared shards are all
    present and unmodified.

    Three independent checks per shard, in increasing strength:

    1. existence and non-empty size (always);
    2. exact byte size against ``manifest["shard_byte_sizes"]`` when present
       (cheap -- one ``stat()``);
    3. content identity via a BLAKE2 checksum against
       ``manifest["shard_checksums"]`` when present and ``verify_checksums``
       is true (the strongest check: a same-size, same-row-count shard
       replaced with *different* valid Parquet content is still detected --
       row-count-only validation, this function's pre-fix behavior, cannot
       tell the difference). This reads every shard file in full, a cost
       proportional to the *output* size, not the input source -- the
       zero-source-read guarantee of the checkpoint fast path is unaffected.

    Checks 2 and 3 are skipped gracefully (not treated as a problem) for a
    manifest written before this fields existed, so an older on-disk result
    is still usable -- it just falls back to the weaker row-count check.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.exists():
        return [f"no manifest at {manifest_path}"]
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError) as exc:
        return [f"manifest at {manifest_path} is unreadable/corrupt: {exc}"]
    problems: list[str] = []
    row_counts = manifest.get("row_counts") or []
    shard_byte_sizes = manifest.get("shard_byte_sizes") or []
    shard_checksums = manifest.get("shard_checksums") or []
    for i, shard in enumerate(manifest.get("shards", [])):
        shard_path = directory / shard
        if not shard_path.exists():
            problems.append(f"missing shard file: {shard}")
            continue
        try:
            size = shard_path.stat().st_size
        except OSError as exc:
            problems.append(f"cannot stat shard {shard}: {exc}")
            continue
        if size == 0:
            problems.append(f"empty shard file: {shard}")
            continue
        expected_size = shard_byte_sizes[i] if i < len(shard_byte_sizes) else None
        if expected_size is not None and size != expected_size:
            problems.append(f"shard {shard} is {size:,} bytes on disk, manifest expects {expected_size:,} bytes")
            continue
        expected_checksum = shard_checksums[i] if i < len(shard_checksums) else None
        if expected_checksum is not None and verify_checksums:
            try:
                actual_checksum = _blake2b_file(shard_path)
                if actual_checksum != expected_checksum:
                    problems.append(f"shard {shard} content checksum does not match the manifest (replaced or corrupted)")
                    continue
            except OSError as exc:
                problems.append(f"cannot read shard {shard} to verify its checksum: {exc}")
                continue
        expected_rows = row_counts[i] if i < len(row_counts) else None
        if expected_rows:
            try:
                import pyarrow.parquet as pq

                actual_rows = pq.ParquetFile(shard_path).metadata.num_rows
                if actual_rows != expected_rows:
                    problems.append(f"shard {shard} has {actual_rows} rows, manifest expects {expected_rows}")
            except Exception as exc:
                problems.append(f"shard {shard} failed to open/read metadata: {exc}")
    return problems


DEFAULT_MAX_MATERIALIZE_BYTES = 2 * 1024 ** 3  # 2GB


def _dtype_itemsize(dtype_str: str) -> int:
    try:
        return int(np.dtype(dtype_str).itemsize)
    except (TypeError, ValueError):
        return 8  # conservative fallback for an unrecognized/object dtype string


def _estimate_bytes(n_rows: int, dtypes: dict) -> int:
    if not dtypes:
        return n_rows * 8  # no dtype info at all; conservative per-column-less fallback
    return n_rows * sum(_dtype_itemsize(d) for d in dtypes.values())


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on both POSIX and Windows (same filesystem)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Disk-backed FeatureMatrix output requires the optional 'io' extra: pip install 'featm[io]'"
        ) from exc


class _MemoryBackend:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.df.columns]

    @property
    def dtypes(self) -> dict[str, str]:
        return {str(c): str(self.df[c].dtype) for c in self.df.columns}

    def head(self, n: int) -> pd.DataFrame:
        return self.df.head(n)

    def select(self, columns: Sequence[str]) -> "_MemoryBackend":
        return _MemoryBackend(self.df[list(columns)])

    def iter_batches(self, batch_rows: int) -> Iterator[pd.DataFrame]:
        n = len(self.df)
        for start in range(0, max(n, 1), batch_rows):
            chunk = self.df.iloc[start : start + batch_rows]
            if len(chunk) or n == 0:
                yield chunk
            if start + batch_rows >= n:
                break

    def to_pandas(self) -> pd.DataFrame:
        return self.df

    def to_parquet(self, path: str | Path) -> None:
        _require_pyarrow()
        self.df.to_parquet(path, index=False)

    def to_csv(self, path: str | Path) -> None:
        self.df.to_csv(path, index=False)

    def to_polars(self, *, lazy: bool = False):
        import polars as pl

        pldf = pl.from_pandas(self.df)
        return pldf.lazy() if lazy else pldf


class _DiskBackend:
    def __init__(self, directory: Path, projected_columns: list[str] | None = None) -> None:
        _require_pyarrow()
        self.directory = Path(directory)
        manifest_path = self.directory / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}; not a valid FeatureMatrix directory")
        self.manifest = _read_json(manifest_path)
        if self.manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported FeatureMatrix manifest_version={self.manifest.get('manifest_version')!r}; "
                f"this featm build supports version {MANIFEST_VERSION}"
            )
        self._projected_columns = projected_columns

    @property
    def columns(self) -> list[str]:
        return list(self._projected_columns or self.manifest["columns"])

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.manifest["n_rows"]), len(self.columns))

    @property
    def dtypes(self) -> dict[str, str]:
        all_dtypes = dict(self.manifest.get("dtypes", {}))
        return {c: all_dtypes.get(c, "unknown") for c in self.columns}

    def head(self, n: int) -> pd.DataFrame:
        collected: list[pd.DataFrame] = []
        seen = 0
        for batch in self.iter_batches(max(n, 1)):
            collected.append(batch)
            seen += len(batch)
            if seen >= n:
                break
        if not collected:
            return pd.DataFrame(columns=self.columns)
        return pd.concat(collected, ignore_index=True).head(n)

    def select(self, columns: Sequence[str]) -> "_DiskBackend":
        unknown = set(columns) - set(self.manifest["columns"])
        if unknown:
            raise KeyError(f"Unknown column(s) for this FeatureMatrix: {sorted(unknown)}")
        return _DiskBackend(self.directory, projected_columns=list(columns))

    def iter_batches(self, batch_rows: int) -> Iterator[pd.DataFrame]:
        import pyarrow.parquet as pq

        cols = self.columns
        n_rows = int(self.manifest["n_rows"])
        if n_rows == 0:
            yield pd.DataFrame(columns=cols)
            return
        for shard in self.manifest["shards"]:
            pf = pq.ParquetFile(self.directory / shard)
            for record_batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
                yield record_batch.to_pandas()

    def to_pandas(self) -> pd.DataFrame:
        n_rows = int(self.manifest["n_rows"])
        if n_rows == 0:
            return pd.DataFrame(columns=self.columns)
        return pd.concat(list(self.iter_batches(batch_rows=1_000_000)), ignore_index=True)

    def to_parquet(self, path: str | Path) -> None:
        """Stream shard-by-shard into the output file -- never materializes the whole matrix."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        try:
            for batch in self.iter_batches(batch_rows=100_000):
                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(path), table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:  # empty matrix: write a schema-only file
            empty = pd.DataFrame(columns=self.columns)
            pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), str(path))

    def to_csv(self, path: str | Path) -> None:
        first = True
        for batch in self.iter_batches(batch_rows=100_000):
            batch.to_csv(path, mode="w" if first else "a", header=first, index=False)
            first = False
        if first:  # never wrote anything (empty matrix)
            pd.DataFrame(columns=self.columns).to_csv(path, index=False)

    def to_polars(self, *, lazy: bool = False):
        import polars as pl

        shard_paths = [str(self.directory / s) for s in self.manifest["shards"]]
        if lazy:
            # Genuinely lazy: no shard is read until the caller calls .collect()
            # (or otherwise triggers evaluation) on the returned LazyFrame.
            if not shard_paths:
                return pl.DataFrame().lazy()
            lf = pl.scan_parquet(shard_paths)
            return lf.select(self._projected_columns) if self._projected_columns else lf
        frames = [pl.read_parquet(p) for p in shard_paths]
        if self._projected_columns:
            frames = [f.select(self._projected_columns) for f in frames]
        return pl.concat(frames) if frames else pl.DataFrame()


class FeatureMatrix:
    """Uniform result type for extraction output, memory- or disk-backed."""

    def __init__(self, backend: "_MemoryBackend | _DiskBackend") -> None:
        self._backend = backend
        #: Set by the out-of-core engine after a run; ``None`` for a FeatureMatrix
        #: built directly (e.g. via from_pandas) without an ExecutionReport.
        self.report: Any | None = None

    # --- construction -----------------------------------------------------
    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "FeatureMatrix":
        return cls(_MemoryBackend(df))

    @classmethod
    def open(cls, directory: str | Path) -> "FeatureMatrix":
        """Reopen a disk-backed FeatureMatrix from a previously written manifest."""
        return cls(_DiskBackend(Path(directory)))

    # --- introspection ------------------------------------------------------
    @property
    def is_disk_backed(self) -> bool:
        return isinstance(self._backend, _DiskBackend)

    @property
    def shape(self) -> tuple[int, int]:
        return self._backend.shape

    @property
    def columns(self) -> list[str]:
        return self._backend.columns

    @property
    def dtypes(self) -> dict[str, str]:
        return self._backend.dtypes

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        kind = "disk" if self.is_disk_backed else "memory"
        n, c = self.shape
        return f"FeatureMatrix(backend={kind}, shape=({n}, {c}))"

    # --- data access --------------------------------------------------------
    def head(self, n: int = 5) -> pd.DataFrame:
        return self._backend.head(n)

    def select(self, columns: Sequence[str]) -> "FeatureMatrix":
        return FeatureMatrix(self._backend.select(columns))

    def iter_batches(self, batch_rows: int = 50_000) -> Iterator[pd.DataFrame]:
        yield from self._backend.iter_batches(batch_rows)

    def to_pandas(
        self,
        *,
        allow_large: bool = False,
        max_rows: int = DEFAULT_MAX_MATERIALIZE_ROWS,
        max_bytes: int = DEFAULT_MAX_MATERIALIZE_BYTES,
    ) -> pd.DataFrame:
        n = self.shape[0]
        if n > max_rows and not allow_large:
            raise MaterializationGuardError(
                f"Refusing to materialize {n} rows into a single pandas DataFrame "
                f"(guard max_rows={max_rows}). Pass allow_large=True or a higher max_rows "
                "to override, or use .iter_batches()/.select(...) to work in bounded pieces."
            )
        # Row count alone is not a reliable guard: a modest row count can still be
        # enormous if it is very wide (e.g. 1,000,000 rows x 10,000 float columns
        # is ~80GB but well under a typical row-count threshold). Estimate actual
        # bytes from dtypes when known and guard on that too.
        est_bytes = _estimate_bytes(n, self.dtypes)
        if est_bytes > max_bytes and not allow_large:
            raise MaterializationGuardError(
                f"Refusing to materialize an estimated {est_bytes:,} bytes ({n} rows x "
                f"{len(self.columns)} columns) into a single pandas DataFrame (guard "
                f"max_bytes={max_bytes:,}). Pass allow_large=True or a higher max_bytes to "
                "override, or use .iter_batches()/.select(...) to work in bounded pieces."
            )
        return self._backend.to_pandas()

    def to_numpy(
        self,
        *,
        allow_large: bool = False,
        max_rows: int = DEFAULT_MAX_MATERIALIZE_ROWS,
        max_bytes: int = DEFAULT_MAX_MATERIALIZE_BYTES,
    ):
        return self.to_pandas(allow_large=allow_large, max_rows=max_rows, max_bytes=max_bytes).to_numpy()

    def to_polars(self, *, lazy: bool = False):
        """Return a Polars DataFrame, or (``lazy=True``) a LazyFrame that reads nothing until collected.

        For a disk-backed matrix, ``lazy=True`` scans the underlying Parquet
        shards directly (``polars.scan_parquet``) and materializes nothing
        until the caller calls e.g. ``.collect()`` on the result -- the only
        non-materializing way to get a Polars view of a large disk-backed
        result. ``lazy=False`` (default) reads and concatenates every shard
        eagerly, same as before.
        """
        try:
            import polars  # noqa: F401
        except ImportError as exc:
            raise ImportError("to_polars() requires the optional 'polars' package") from exc
        return self._backend.to_polars(lazy=lazy)

    def to_parquet(self, path: str | Path) -> None:
        self._backend.to_parquet(path)

    def to_csv(self, path: str | Path) -> None:
        self._backend.to_csv(path)


class DiskFeatureMatrixWriter:
    """Streaming writer used by the execution engine to spill batches to disk.

    Column order is fixed at construction and enforced on every batch. Shards
    are written one Parquet file per batch (``part-00000.parquet``, ...), and
    the manifest is only written -- atomically -- once at :meth:`finalize`, so
    a crash mid-run leaves shard files but no manifest, and :func:`FeatureMatrix.open`
    correctly refuses to treat that directory as a complete result.
    """

    def __init__(
        self,
        directory: str | Path,
        columns: Sequence[str],
        *,
        start_shard_index: int = 0,
        existing_row_counts: Sequence[int] | None = None,
        existing_dtypes: dict[str, str] | None = None,
        disk_budget_bytes: int | None = None,
        existing_bytes_written: int = 0,
        existing_shard_byte_sizes: Sequence[int] | None = None,
        existing_shard_checksums: Sequence[str] | None = None,
    ) -> None:
        _require_pyarrow()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.columns = [str(c) for c in columns]
        self._shard_index = start_shard_index
        self._row_counts: list[int] = list(existing_row_counts or [])
        self._dtypes: dict[str, str] | None = dict(existing_dtypes) if existing_dtypes else None
        #: Finding 11 (third audit pass): per-shard size + content checksum,
        #: so a completed checkpoint's zero-read fast path can detect a shard
        #: replaced with different-but-valid-Parquet content, not just a
        #: missing/wrong-row-count one. See validate_disk_backend_integrity.
        self._shard_byte_sizes: list[int] = list(existing_shard_byte_sizes or [])
        self._shard_checksums: list[str] = list(existing_shard_checksums or [])
        #: Enforced against *actual* bytes written per shard (works for unknown
        #: row counts, compressed output, wide vector features, etc. -- not just
        #: a size estimated ahead of time).
        self.disk_budget_bytes = disk_budget_bytes
        self._bytes_written = existing_bytes_written

    @property
    def shard_count(self) -> int:
        return self._shard_index

    @property
    def row_counts(self) -> list[int]:
        return list(self._row_counts)

    @property
    def dtypes(self) -> dict[str, str] | None:
        return dict(self._dtypes) if self._dtypes else None

    @property
    def n_rows_written(self) -> int:
        return sum(self._row_counts)

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def shard_byte_sizes(self) -> list[int]:
        return list(self._shard_byte_sizes)

    @property
    def shard_checksums(self) -> list[str]:
        return list(self._shard_checksums)

    def write_batch(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Batch is missing declared column(s): {missing}")
        df = df[self.columns]
        if self._dtypes is None:
            self._dtypes = {c: str(df[c].dtype) for c in self.columns}
        shard_path = self.directory / f"part-{self._shard_index:05d}.parquet"
        df.to_parquet(shard_path, index=False)
        shard_bytes = shard_path.stat().st_size
        if self.disk_budget_bytes is not None and self._bytes_written + shard_bytes > self.disk_budget_bytes:
            # Actual bytes exceeded the budget: undo the write (don't leave a
            # shard on disk that violates the budget) and fail deterministically
            # rather than continuing to write indefinitely. Shards already
            # written (and any checkpoint reflecting them) are untouched --
            # only the file that pushed the run over budget is removed, and
            # the shard index is not advanced, so a caller retrying with a
            # larger budget (or resuming after fixing the budget) will simply
            # rewrite this same shard index next.
            shard_path.unlink(missing_ok=True)
            raise DiskBudgetExceededError(
                f"Writing shard {self._shard_index} ({shard_bytes:,} bytes) would bring total "
                f"disk usage to {self._bytes_written + shard_bytes:,} bytes, exceeding the "
                f"disk_budget of {self.disk_budget_bytes:,} bytes. {self._shard_index} shard(s) "
                f"already on disk ({self._bytes_written:,} bytes) are preserved."
            )
        self._bytes_written += shard_bytes
        self._row_counts.append(len(df))
        self._shard_byte_sizes.append(shard_bytes)
        self._shard_checksums.append(_blake2b_file(shard_path))
        self._shard_index += 1

    def finalize(self, *, extra_manifest: dict[str, Any] | None = None) -> FeatureMatrix:
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "columns": self.columns,
            "dtypes": self._dtypes or {},
            "n_rows": self.n_rows_written,
            "shards": [f"part-{i:05d}.parquet" for i in range(self._shard_index)],
            "row_counts": self._row_counts,
            "shard_byte_sizes": self._shard_byte_sizes,
            "shard_checksums": self._shard_checksums,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        _atomic_write_json(self.directory / MANIFEST_FILENAME, manifest)
        return FeatureMatrix.open(self.directory)
