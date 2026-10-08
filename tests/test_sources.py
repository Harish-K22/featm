from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featm.sources import (
    ArrowSource,
    CSVSource,
    IteratorSource,
    JSONLSource,
    PandasSource,
    ParquetSource,
    SourceNotReplayableError,
    resolve_source,
)


def test_pandas_source_schema_and_batches(sample_tabular_df):
    src = PandasSource(sample_tabular_df)
    assert src.n_rows == len(sample_tabular_df)
    assert set(src.schema().names()) == set(sample_tabular_df.columns)
    batches = list(src.iter_batches(batch_rows=1_000))
    assert sum(len(b) for b in batches) == len(sample_tabular_df)
    # deterministic order: concatenated batches equal the original frame exactly
    combined = pd.concat(batches)
    assert list(combined["price"]) == list(sample_tabular_df["price"])


def test_pandas_source_column_projection(sample_tabular_df):
    src = PandasSource(sample_tabular_df)
    batches = list(src.iter_batches(columns=["price"], batch_rows=2_000))
    assert all(list(b.columns) == ["price"] for b in batches)


def test_csv_source_schema_discovery_without_full_scan(csv_path, sample_tabular_df):
    src = CSVSource(csv_path)
    schema = src.schema()
    assert set(schema.names()) == set(sample_tabular_df.columns)
    # row count is not cheaply knowable for CSV -- must stay unknown until scanned
    assert src.n_rows is None


def test_csv_source_bounded_batches_and_projection(csv_path, sample_tabular_df):
    src = CSVSource(csv_path)
    batches = list(src.iter_batches(columns=["price", "city"], batch_rows=777))
    assert len(batches) > 1
    assert all(len(b) <= 777 for b in batches)
    assert all(list(b.columns) == ["price", "city"] for b in batches)
    total = sum(len(b) for b in batches)
    assert total == len(sample_tabular_df)


def test_csv_source_projection_excludes_other_columns(csv_path):
    src = CSVSource(csv_path)
    batch = next(src.iter_batches(columns=["price"], batch_rows=100))
    assert "quantity" not in batch.columns
    assert "city" not in batch.columns


def test_csv_source_is_replayable(csv_path, sample_tabular_df):
    src = CSVSource(csv_path)
    first = sum(len(b) for b in src.iter_batches(batch_rows=1_000))
    second = sum(len(b) for b in src.iter_batches(batch_rows=500))
    assert first == second == len(sample_tabular_df)


def test_csv_deterministic_row_order(csv_path, sample_tabular_df):
    src = CSVSource(csv_path)
    combined = pd.concat(list(src.iter_batches(columns=["price"], batch_rows=333)), ignore_index=True)
    # CSV round-trips floats through text, so compare numerically rather than bit-exact.
    assert np.allclose(combined["price"].to_numpy(), sample_tabular_df["price"].to_numpy())


def test_jsonl_source(jsonl_path, sample_tabular_df):
    src = JSONLSource(jsonl_path)
    assert set(src.schema().names()) == set(sample_tabular_df.columns)
    batches = list(src.iter_batches(columns=["price"], batch_rows=1_000))
    assert sum(len(b) for b in batches) == len(sample_tabular_df)
    assert all(list(b.columns) == ["price"] for b in batches)


def test_parquet_source_cheap_row_count_and_schema(parquet_path, sample_tabular_df):
    src = ParquetSource(parquet_path)
    assert src.n_rows == len(sample_tabular_df)  # from footer metadata, no data read
    assert set(src.schema().names()) == set(sample_tabular_df.columns)


def test_parquet_source_row_group_batching_and_projection(parquet_path, sample_tabular_df):
    src = ParquetSource(parquet_path)
    batches = list(src.iter_batches(columns=["quantity"], batch_rows=900))
    assert all(list(b.columns) == ["quantity"] for b in batches)
    total = sum(len(b) for b in batches)
    assert total == len(sample_tabular_df)


def test_arrow_source_over_parquet_file(parquet_path, sample_tabular_df):
    pytest.importorskip("pyarrow")
    src = ArrowSource(parquet_path)
    assert src.n_rows == len(sample_tabular_df)
    batches = list(src.iter_batches(columns=["price"], batch_rows=1_500))
    assert sum(len(b) for b in batches) == len(sample_tabular_df)


def test_resolve_source_dispatch(csv_path, parquet_path, sample_tabular_df):
    assert isinstance(resolve_source(sample_tabular_df), PandasSource)
    assert isinstance(resolve_source(csv_path), CSVSource)
    assert isinstance(resolve_source(str(csv_path)), CSVSource)
    assert isinstance(resolve_source(parquet_path), ParquetSource)


def test_resolve_source_rejects_unknown_type():
    with pytest.raises(TypeError):
        resolve_source(12345)


def test_resolve_source_rejects_unknown_suffix(tmp_path):
    bogus = tmp_path / "data.xyz"
    bogus.write_text("nonsense")
    with pytest.raises(ValueError):
        resolve_source(bogus)


def test_iterator_source_single_pass_then_raises():
    df = pd.DataFrame({"a": np.arange(10)})

    def gen():
        yield df.iloc[:5]
        yield df.iloc[5:]

    schema = PandasSource(df).schema()
    src = IteratorSource(gen(), schema=schema, n_rows=10)
    first_pass_rows = sum(len(b) for b in src.iter_batches(batch_rows=10))
    assert first_pass_rows == 10
    with pytest.raises(SourceNotReplayableError):
        list(src.iter_batches(batch_rows=10))


def test_source_fingerprint_stable_for_same_file(csv_path):
    a = CSVSource(csv_path)
    b = CSVSource(csv_path)
    assert a.fingerprint() == b.fingerprint()


def test_source_fingerprint_changes_when_file_changes(tmp_path):
    path = tmp_path / "f.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    fp1 = CSVSource(path).fingerprint()
    pd.DataFrame({"a": [1, 2, 3, 4, 5]}).to_csv(path, index=False)
    fp2 = CSVSource(path).fingerprint()
    assert fp1 != fp2
