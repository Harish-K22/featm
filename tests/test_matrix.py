from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featm.matrix import DiskFeatureMatrixWriter, FeatureMatrix, MaterializationGuardError


@pytest.fixture
def df():
    return pd.DataFrame({"a": np.arange(1_000, dtype=float), "b": np.arange(1_000, dtype=float) * 2})


@pytest.fixture
def _pyarrow():
    """Request this fixture (unused parameter is fine) to skip a test cleanly when pyarrow is absent.

    Only disk-backed FeatureMatrix tests need this -- memory-backed tests must
    keep running in a base install with no optional dependencies, so this is
    applied per-test rather than at module level.
    """
    return pytest.importorskip("pyarrow")


def test_memory_backed_matrix_basic_api(df):
    fm = FeatureMatrix.from_pandas(df)
    assert fm.shape == (1_000, 2)
    assert fm.columns == ["a", "b"]
    assert not fm.is_disk_backed
    assert len(fm) == 1_000
    assert fm.head(3).shape == (3, 2)
    pdf = fm.to_pandas()
    assert pdf.equals(df)


def test_memory_backed_select_and_iter_batches(df):
    fm = FeatureMatrix.from_pandas(df)
    projected = fm.select(["a"])
    assert projected.columns == ["a"]
    batches = list(fm.iter_batches(batch_rows=300))
    assert sum(len(b) for b in batches) == 1_000


def test_disk_backed_matrix_write_and_reopen(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a", "b"])
    for start in range(0, len(df), 250):
        writer.write_batch(df.iloc[start : start + 250])
    fm = writer.finalize()
    assert fm.is_disk_backed
    assert fm.shape == (1_000, 2)

    reopened = FeatureMatrix.open(tmp_path / "fm")
    assert reopened.shape == (1_000, 2)
    assert reopened.columns == ["a", "b"]
    pdf = reopened.to_pandas()
    assert list(pdf["a"]) == list(df["a"])


def test_disk_backed_column_order_preserved(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["b", "a"])  # deliberately reversed
    writer.write_batch(df[["b", "a"]])
    fm = writer.finalize()
    assert fm.columns == ["b", "a"]
    assert list(fm.to_pandas().columns) == ["b", "a"]


def test_disk_backed_row_order_preserved_across_shards(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a"])
    for start in range(0, len(df), 111):
        writer.write_batch(df.iloc[start : start + 111][["a"]])
    fm = writer.finalize()
    pdf = fm.to_pandas()
    assert list(pdf["a"]) == list(df["a"])


def test_disk_backed_select_projects_without_loading_everything(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a", "b"])
    writer.write_batch(df)
    fm = writer.finalize()
    projected = fm.select(["a"])
    assert projected.columns == ["a"]
    pdf = projected.to_pandas()
    assert list(pdf.columns) == ["a"]


def test_disk_backed_select_rejects_unknown_column(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a", "b"])
    writer.write_batch(df)
    fm = writer.finalize()
    with pytest.raises(KeyError):
        fm.select(["nope"])


def test_open_missing_manifest_raises(_pyarrow, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        FeatureMatrix.open(tmp_path / "empty")


def test_crash_before_finalize_leaves_no_valid_manifest(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a"])
    writer.write_batch(df[["a"]])  # shard written, but finalize() never called (simulated crash)
    assert not (tmp_path / "fm" / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        FeatureMatrix.open(tmp_path / "fm")


def test_to_pandas_guard_blocks_huge_materialization(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a", "b"])
    writer.write_batch(df)
    fm = writer.finalize()
    with pytest.raises(MaterializationGuardError):
        fm.to_pandas(max_rows=10)
    # override works
    pdf = fm.to_pandas(max_rows=10, allow_large=True)
    assert len(pdf) == 1_000


def test_to_numpy_guard_blocks_huge_materialization(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a", "b"])
    writer.write_batch(df)
    fm = writer.finalize()
    with pytest.raises(MaterializationGuardError):
        fm.to_numpy(max_rows=10)
    arr = fm.to_numpy(max_rows=10, allow_large=True)
    assert arr.shape == (1_000, 2)


def test_memory_backed_guard_also_applies(df):
    fm = FeatureMatrix.from_pandas(df)
    with pytest.raises(MaterializationGuardError):
        fm.to_pandas(max_rows=5)


def test_to_csv_and_to_parquet_roundtrip(_pyarrow, tmp_path, df):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a", "b"])
    writer.write_batch(df)
    fm = writer.finalize()
    csv_out = tmp_path / "out.csv"
    fm.to_csv(csv_out)
    assert pd.read_csv(csv_out).shape == (1_000, 2)
    pq_out = tmp_path / "out.parquet"
    fm.to_parquet(pq_out)
    assert pd.read_parquet(pq_out).shape == (1_000, 2)


def test_empty_matrix_handles_gracefully(_pyarrow, tmp_path):
    writer = DiskFeatureMatrixWriter(tmp_path / "fm", columns=["a"])
    fm = writer.finalize()
    assert fm.shape == (0, 1)
    assert fm.to_pandas().shape == (0, 1)
