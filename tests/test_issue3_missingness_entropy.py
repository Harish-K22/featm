"""Issue #3: missingness entropy must be a whole-column statistic, not a batch statistic."""

import numpy as np
import pandas as pd
import pytest

import featm
from featm.registry import BatchSemantics
from featm.state import MissingnessEntropyEncoder, STATEFUL_ENCODERS

NAME = "tabular.missingness.entropy"


def _extract(values, *, batch_rows, distributed=False):
    frame = pd.DataFrame({"x": pd.Series(values, dtype="object")})
    kwargs = dict(features=[NAME], columns={NAME: "x"}, output="memory", errors="raise")
    if distributed:
        kwargs.update(execution="distributed", distributed_backend="local", distributed_workers=3,
                      distributed_partition_rows=batch_rows)
    else:
        kwargs["batch_rows"] = batch_rows
    return featm.extract(featm.PandasSource(frame), **kwargs).to_numpy().ravel()


def test_registered_as_mergeable_dataset_context():
    spec = featm.feature_info(NAME)
    assert spec.stateful is True
    assert spec.batchable is True
    assert spec.batch_semantics is BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED
    assert NAME in STATEFUL_ENCODERS
    assert STATEFUL_ENCODERS[NAME].distributed_merge_supported
    assert len(featm.REGISTRY.all()) == 552


@pytest.mark.parametrize("values,expected", [
    ([1.0, 2.0, np.nan, np.nan], np.log(2.0)),
    ([np.nan, 1.0, 2.0, 3.0], -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))),
    ([1.0, 2.0, 3.0, 4.0], 0.0),
    ([np.nan, None, pd.NA, np.nan], 0.0),
    ([1.0], 0.0),
    ([np.nan], 0.0),
])
@pytest.mark.parametrize("batch_rows", [1, 2, 3, 4, 7])
def test_identical_for_arbitrary_batch_boundaries(values, expected, batch_rows):
    actual = _extract(values, batch_rows=batch_rows)
    np.testing.assert_allclose(actual, [expected] * len(values), rtol=0, atol=1e-14)


@pytest.mark.parametrize("batch_rows", [1, 2, 3, 5])
def test_distributed_partition_invariance(batch_rows):
    values = [1.0, None, 2.0, 3.0, pd.NA, np.nan, 4.0]
    local = _extract(values, batch_rows=1)
    distributed = _extract(values, batch_rows=batch_rows, distributed=True)
    np.testing.assert_array_equal(distributed, local)


def test_raw_state_merge_is_associative_and_exact():
    enc = MissingnessEntropyEncoder()
    chunks = [pd.Series([1.0, np.nan]), pd.Series([np.nan]), pd.Series([2.0, pd.NA, 3.0])]
    whole = pd.concat(chunks, ignore_index=True)
    single = enc.update_state(enc.init_state(), whole)
    parts = [enc.update_state(enc.init_state(), c) for c in chunks]
    left = enc.merge_states(enc.merge_states(parts[0], parts[1]), parts[2])
    right = enc.merge_states(parts[0], enc.merge_states(parts[1], parts[2]))
    assert single == left == right == {"n": 6, "missing": 3}
    for state in (single, left, right):
        frozen = enc.finalize_state(state)
        assert frozen["entropy"] == pytest.approx(np.log(2))
        assert enc.transform_batch(whole, frozen).tolist() == pytest.approx([np.log(2)] * len(whole))


def test_train_test_frozen_entropy_and_state_roundtrip(tmp_path):
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, np.nan]})
    test = pd.DataFrame({"x": [3.0, 4.0, 5.0]})
    fitted = featm.fit(featm.PandasSource(train), features=[NAME], columns={NAME: "x"}, batch_rows=1)
    np.testing.assert_allclose(fitted.transform(featm.PandasSource(test), batch_rows=1, output="memory").to_numpy().ravel(),
                               [np.log(2)] * 3)
    artifact = tmp_path / "fitted"
    fitted.save(artifact)
    reloaded = featm.load_fitted(artifact)
    np.testing.assert_allclose(reloaded.transform(featm.PandasSource(test), batch_rows=2, output="memory").to_numpy().ravel(),
                               [np.log(2)] * 3)


def test_small_data_call_preserves_original_formula():
    values = pd.Series([1.0, 2.0, np.nan, np.nan])
    direct = np.asarray(featm.create_feature(NAME)(values), dtype=float)
    np.testing.assert_allclose(direct, np.full(4, np.log(2)))


def test_mixed_feature_does_not_change_other_outputs():
    frame = pd.DataFrame({"x": [1.0, np.nan, -2.0, np.nan]})
    names = [NAME, "tabular.numeric.abs"]
    columns = {n: "x" for n in names}
    results = [featm.extract(featm.PandasSource(frame), features=names, columns=columns,
                             batch_rows=b, output="memory", errors="raise").to_numpy() for b in (1, 2, 4)]
    for res in results[1:]:
        np.testing.assert_allclose(res, results[0], equal_nan=True)
    np.testing.assert_allclose(results[0][:, 0], [np.log(2)] * 4)
    np.testing.assert_allclose(results[0][:, 1], [1.0, np.nan, 2.0, np.nan], equal_nan=True)
