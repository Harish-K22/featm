"""Step 4 completion pass: source-native distributed I/O (Parquet).

Uses the Dask-independent local reference runtime for the equivalence
checks (matching the rest of the foundation's testing style); real
multi-process worker-native reads are additionally covered on real Dask in
``tests/test_distributed_dask.py``-style manual validation (see
STEP4_DASK_HANDOFF.md section E) since ``pytest.importorskip`` keeps this
file runnable without Dask installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import featm
from featm.distributed.partition_provider import (
    CoordinatorStreamPartitionProvider,
    DaskParquetPartitionProvider,
    ParquetPartitionDescriptor,
    ParquetRowGroupTooLargeError,
    PartitionIntegrityError,
    resolve_partition_frame,
    select_partition_provider,
)
from featm.execution import _prepare_run
from featm.sources import ParquetSource


def _make_parquet(tmp_path, n_rows=530, row_group_size=57, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.standard_normal(n_rows), "y": rng.standard_normal(n_rows)})
    path = tmp_path / "data.parquet"
    df.to_parquet(path, row_group_size=row_group_size)
    return path, df


def _make_parquet_with_row_groups(tmp_path, rows_per_group, *, name="rg.parquet", seed=1):
    """Write a Parquet file with EXACT, individually-controlled row-group sizes
    (``to_parquet(row_group_size=...)`` only controls a uniform target, not
    arbitrary per-group sizes)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    total = sum(rows_per_group)
    x = rng.standard_normal(total)
    offset = 0
    tables = []
    for n in rows_per_group:
        tables.append(pa.Table.from_pandas(pd.DataFrame({"x": x[offset : offset + n]}), preserve_index=False))
        offset += n
    path = tmp_path / name
    writer = pq.ParquetWriter(path, tables[0].schema)
    for t in tables:
        writer.write_table(t)
    writer.close()
    return path, pd.DataFrame({"x": x})


def test_select_partition_provider_picks_parquet_native_for_parquet_source(tmp_path):
    path, _ = _make_parquet(tmp_path)
    prepared = _prepare_run(str(path), features=["tabular.numeric.abs"], modality=None, columns={"tabular.numeric.abs": "x"}, mode="exact", errors="raise", memory_budget=None, disk_budget=None, batch_rows=100, cache_budget=None, params=None, backend="cpu", gpu_fallback="fallback", device=None, gpu_memory_budget=None)
    provider = select_partition_provider(prepared, 100)
    assert isinstance(provider, DaskParquetPartitionProvider)
    assert provider.kind == "parquet_native"


def test_select_partition_provider_falls_back_for_pandas_source(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    prepared = _prepare_run(df, features=["tabular.numeric.abs"], modality=None, columns={"tabular.numeric.abs": "x"}, mode="exact", errors="raise", memory_budget=None, disk_budget=None, batch_rows=100, cache_budget=None, params=None, backend="cpu", gpu_fallback="fallback", device=None, gpu_memory_budget=None)
    provider = select_partition_provider(prepared, 100)
    assert isinstance(provider, CoordinatorStreamPartitionProvider)
    assert provider.kind == "coordinator_streamed"


def test_parquet_provider_partition_boundaries_deterministic_and_ordered(tmp_path):
    path, _ = _make_parquet(tmp_path, n_rows=500, row_group_size=50)
    provider = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=120)
    parts = list(provider.iter_partitions(["x", "y"]))
    indices = [p.index for p in parts]
    assert indices == sorted(indices) == list(range(len(parts)))
    # Re-instantiating against the same file must produce identical boundaries.
    provider2 = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=120)
    parts2 = list(provider2.iter_partitions(["x", "y"]))
    assert [p.frame.row_groups for p in parts] == [p.frame.row_groups for p in parts2]


def test_parquet_descriptor_resolves_to_correct_projected_rows(tmp_path):
    path, df = _make_parquet(tmp_path, n_rows=300, row_group_size=40)
    provider = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=100)
    parts = list(provider.iter_partitions(["x"]))
    resolved = [resolve_partition_frame(p.frame) for p in parts]
    total_rows = sum(len(f) for f in resolved)
    assert total_rows == len(df)
    for f in resolved:
        assert list(f.columns) == ["x"]  # column projection honored -- 'y' never read
    # Concatenated, in partition order, must equal the original column exactly.
    combined = pd.concat(resolved, ignore_index=True)
    assert np.allclose(combined["x"].to_numpy(), df["x"].to_numpy())


def test_resolve_partition_frame_passes_through_real_dataframe_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert resolve_partition_frame(df) is df


def test_row_independent_feature_over_parquet_matches_local_pandas(tmp_path):
    path, df = _make_parquet(tmp_path, n_rows=400, row_group_size=45, seed=5)
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    dist_r = featm.extract(
        str(path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=90,
        output="memory", errors="raise",
    )
    a = local_r.to_numpy()
    b = dist_r.to_pandas().to_numpy()
    assert np.allclose(a, b)
    assert dist_r.report.distributed_report["partition_provider"] == "parquet_native"


def test_stateful_feature_over_parquet_matches_local_pandas(tmp_path):
    path, df = _make_parquet(tmp_path, n_rows=350, row_group_size=40, seed=6)
    local_r = featm.extract(df, features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"}, execution="local", batch_rows=35, errors="raise")
    dist_r = featm.extract(
        str(path), features=["tabular.scaling.zscore"], columns={"tabular.scaling.zscore": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=35,
        output="memory", errors="raise",
    )
    a = local_r.to_pandas().to_numpy(dtype=float)
    b = dist_r.to_pandas().to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_parquet_native_provider_composes_with_checkpoint(tmp_path):
    path, df = _make_parquet(tmp_path, n_rows=260, row_group_size=30, seed=7)
    ckpt_dir = tmp_path / "ckpt"
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    dist_r = featm.extract(
        str(path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=60,
        output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    a = local_r.to_numpy()
    b = dist_r.to_pandas().to_numpy()
    assert np.allclose(a, b)
    resumed = featm.extract(
        str(path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=60,
        output="memory", errors="raise", checkpoint=str(ckpt_dir),
    )
    c = resumed.to_pandas().to_numpy()
    assert np.allclose(a, c)
    assert resumed.report.distributed_report["checkpoint_resumed"] is True
    assert resumed.report.distributed_report["partition_provider"] == "parquet_native"


# ==========================================================================
# Finding 4 (independent audit): a single physical row group larger than the
# resolved safe partition size must never be silently handed to a worker as
# a "bounded" partition -- PyArrow cannot split a row group, so the
# accepted fix is detect-and-fall-back-to-coordinator-streamed, reported
# honestly rather than disguised as parquet_native.
# ==========================================================================
def test_oversized_row_group_raises_from_direct_construction(tmp_path):
    # One row group covering ALL 500 rows -- requesting partition_rows=50
    # asks for a bound the file's own row-group layout cannot honor.
    path, _ = _make_parquet(tmp_path, n_rows=500, row_group_size=500)
    with pytest.raises(ParquetRowGroupTooLargeError, match="row group"):
        DaskParquetPartitionProvider(ParquetSource(path), partition_rows=50)


def test_select_partition_provider_falls_back_when_row_group_too_large(tmp_path):
    path, _ = _make_parquet(tmp_path, n_rows=500, row_group_size=500)
    prepared = _prepare_run(
        str(path), features=["tabular.numeric.abs"], modality=None, columns={"tabular.numeric.abs": "x"},
        mode="exact", errors="raise", memory_budget=None, disk_budget=None, batch_rows=50, cache_budget=None,
        params=None, backend="cpu", gpu_fallback="fallback", device=None, gpu_memory_budget=None,
    )
    provider = select_partition_provider(prepared, 50)
    assert isinstance(provider, CoordinatorStreamPartitionProvider)
    assert provider.kind == "coordinator_streamed", "must report the fallback honestly, never claim parquet_native"


def test_oversized_row_group_end_to_end_still_bounded_via_fallback(tmp_path):
    """The end-to-end distributed run over a file with one giant row group must
    still respect the requested partition size (via the coordinator-streamed
    fallback) and must still match the local reference -- never silently hand
    a worker the whole file as "one partition"."""
    path, df = _make_parquet(tmp_path, n_rows=500, row_group_size=500, seed=9)
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    dist_r = featm.extract(
        str(path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=3, distributed_partition_rows=50,
        output="memory", errors="raise",
    )
    a = local_r.to_numpy()
    b = dist_r.to_pandas().to_numpy()
    assert np.allclose(a, b)
    assert dist_r.report.distributed_report["partition_provider"] == "coordinator_streamed"
    assert dist_r.report.distributed_report["partitions_total"] == 10  # 500 rows / 50 per partition, actually bounded


# ==========================================================================
# Finding 13.2 (independent audit): expected_rows was carried across the
# wire but never actually checked -- decorative, not a real invariant.
# ==========================================================================
def test_expected_rows_mismatch_raises_partition_integrity_error(tmp_path):
    path, _ = _make_parquet(tmp_path, n_rows=200, row_group_size=40)
    provider = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=40)
    real_desc = next(iter(provider.iter_partitions(["x"]))).frame
    wrong_desc = ParquetPartitionDescriptor(
        path=real_desc.path, row_groups=real_desc.row_groups, columns=real_desc.columns,
        expected_rows=real_desc.expected_rows + 1,
    )
    with pytest.raises(PartitionIntegrityError, match="expected"):
        resolve_partition_frame(wrong_desc)


# ==========================================================================
# Finding 1 (final closure audit, v0.4.1 -> v0.4.2): grouping multiple
# INDIVIDUALLY-safe row groups could still sum past partition_rows -- v0.4.1
# only guarded against ONE oversized row group, not the SUM of several. The
# flush-before-add rewrite must guarantee expected_rows <= partition_rows for
# every descriptor this provider ever produces (the empty-source case is not
# reachable here since no descriptors are produced for an empty file).
# ==========================================================================
@pytest.mark.parametrize(
    "rows_per_group,bound,expected_partitions",
    [
        ([30, 30, 30], 50, [30, 30, 30]),  # v0.4.1 regression case: produced [60, 30] before this fix
        ([49, 49, 49], 50, [49, 49, 49]),
        ([25, 25, 25, 25], 50, [50, 50]),
        ([50, 50], 50, [50, 50]),
        ([1, 49, 1, 49], 50, [50, 50]),
    ],
)
def test_row_group_grouping_never_exceeds_partition_rows(tmp_path, rows_per_group, bound, expected_partitions):
    path, _ = _make_parquet_with_row_groups(tmp_path, rows_per_group, name=f"{'_'.join(map(str, rows_per_group))}.parquet")
    provider = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=bound)
    parts = list(provider.iter_partitions(["x"]))
    sizes = [p.frame.expected_rows for p in parts]
    assert sizes == expected_partitions
    assert all(1 <= s <= bound for s in sizes), f"partition sizes {sizes} violate the bound {bound}"


def test_row_group_grouping_end_to_end_real_pyarrow_matches_local(tmp_path):
    """End-to-end with real PyArrow reads (local reference runtime): the fix must not
    just report the right expected_rows, actual worker frames must match too, and the
    final result must match the local pandas reference exactly."""
    path, df = _make_parquet_with_row_groups(tmp_path, [30, 30, 30, 30, 30], name="e2e.parquet", seed=7)
    local_r = featm.extract(df, features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"}, execution="local")
    provider = DaskParquetPartitionProvider(ParquetSource(path), partition_rows=50)
    for part in provider.iter_partitions(["x"]):
        frame = resolve_partition_frame(part.frame)
        assert len(frame) <= 50
    dist_r = featm.extract(
        str(path), features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="distributed", distributed_backend="local", distributed_workers=2, distributed_partition_rows=50,
        output="memory", errors="raise",
    )
    a = local_r.to_numpy()
    b = dist_r.to_pandas().to_numpy()
    assert np.allclose(a, b)
    assert dist_r.report.distributed_report["partition_provider"] == "parquet_native"
    # 5 row groups of 30 rows each; 30+30=60 > the 50-row bound, so no two row
    # groups ever combine -- 5 partitions of 30 rows each, not fewer.
    assert dist_r.report.distributed_report["partitions_total"] == 5


def test_row_group_grouping_still_falls_back_for_single_oversized_row_group(tmp_path):
    """The v0.4.1 fallback for a single row group alone exceeding the bound must remain intact."""
    path, _ = _make_parquet_with_row_groups(tmp_path, [500], name="single_huge.parquet")
    with pytest.raises(ParquetRowGroupTooLargeError):
        DaskParquetPartitionProvider(ParquetSource(path), partition_rows=50)
