from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from featm.registry import FeatureSpec, REGISTRY


@pytest.fixture
def register_temp_feature():
    """Register a synthetic FeatureSpec for one test, guaranteed to be removed afterward.

    Step 3 GPU pass: promoted from several individual test files'
    locally-duplicated versions of this exact fixture (test_audit2.py,
    test_audit4.py, test_audit5.py, test_gpu_dispatch.py) into conftest.py
    so every test module gets automatic REGISTRY cleanup without having to
    remember to add it -- discovered because tests/test_execution.py's six
    `REGISTRY.register(FeatureSpec(name="test.exec...."))` calls had NO
    cleanup at all (a pre-existing gap, not introduced by Step 3), which
    this pass's own new `len(featm.REGISTRY.all()) == 552` GPU-registry
    test was the first assertion in the whole suite sensitive enough to
    actually catch (registry count was 558, not 552, whenever
    test_gpu_metadata.py ran after test_execution.py in the same session).
    """
    registered: list[str] = []

    def _register(spec: FeatureSpec) -> FeatureSpec:
        REGISTRY.register(spec)
        registered.append(spec.name)
        return spec

    yield _register
    for name in registered:
        REGISTRY._specs.pop(name, None)


@pytest.fixture
def sample_tabular_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 5_000
    return pd.DataFrame(
        {
            "price": rng.normal(100.0, 20.0, n),
            "quantity": rng.integers(1, 50, n).astype(float),
            "city": rng.choice(["NYC", "LA", "SF", "CHI"], n),
        }
    )


@pytest.fixture
def csv_path(tmp_path, sample_tabular_df):
    path = tmp_path / "data.csv"
    sample_tabular_df.to_csv(path, index=False)
    return path


@pytest.fixture
def parquet_path(tmp_path, sample_tabular_df):
    pytest.importorskip("pyarrow")
    path = tmp_path / "data.parquet"
    sample_tabular_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def jsonl_path(tmp_path, sample_tabular_df):
    path = tmp_path / "data.jsonl"
    sample_tabular_df.to_json(path, orient="records", lines=True)
    return path
