from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import featm
from featm.state import STATEFUL_ENCODERS, FrequencyEncoder, MinMaxEncoder, ZScoreEncoder


def test_zscore_registered_and_selectable():
    assert "tabular.scaling.zscore" in {s.name for s in featm.REGISTRY.all()}
    spec = featm.feature_info("tabular.scaling.zscore")
    assert spec.stateful is True
    assert spec.approximate_available is True


def test_zscore_single_shot_small_data():
    s = pd.Series(np.concatenate([np.zeros(100), np.ones(100) * 10.0]))
    f = featm.create_feature("tabular.scaling.zscore")
    out = pd.Series(f(s))
    assert abs(out.mean()) < 1e-9
    assert abs(out.std(ddof=0) - 1.0) < 1e-9


def test_zscore_multi_batch_merge_matches_single_shot():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(loc=5.0, scale=3.0, size=10_000))
    enc = ZScoreEncoder()

    state = enc.init_state()
    for start in range(0, len(x), 777):  # uneven batch boundaries on purpose
        state = enc.update_state(state, x.iloc[start : start + 777])
    frozen = enc.finalize_state(state)

    assert frozen["mean"] == pytest.approx(x.mean(), rel=1e-9)
    assert frozen["std"] == pytest.approx(x.std(ddof=0), rel=1e-9)


def test_minmax_multi_batch_merge_matches_single_shot():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=5_000))
    enc = MinMaxEncoder()
    state = enc.init_state()
    for start in range(0, len(x), 333):
        state = enc.update_state(state, x.iloc[start : start + 333])
    frozen = enc.finalize_state(state)
    assert frozen["min"] == pytest.approx(x.min())
    assert frozen["max"] == pytest.approx(x.max())
    out = enc.transform_batch(x, frozen)
    assert out.min() == pytest.approx(0.0, abs=1e-9)
    assert out.max() == pytest.approx(1.0, abs=1e-9)


def test_frequency_encoder_unseen_category_maps_to_zero():
    enc = FrequencyEncoder()
    state = enc.init_state()
    state = enc.update_state(state, pd.Series(["a", "a", "b"]))
    frozen = enc.finalize_state(state)
    out = enc.transform_batch(pd.Series(["a", "b", "totally_unseen"]), frozen)
    assert out.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])


def test_frequency_encoder_multi_batch_fit_then_transform_train_test_split():
    train = pd.Series(["x"] * 60 + ["y"] * 30 + ["z"] * 10)
    test = pd.Series(["x", "y", "z", "unseen1", "unseen2"])
    enc = FrequencyEncoder()
    state = enc.init_state()
    for chunk in [train.iloc[:40], train.iloc[40:70], train.iloc[70:]]:
        state = enc.update_state(state, chunk)
    frozen = enc.finalize_state(state)

    out_test = enc.transform_batch(test, frozen)
    assert out_test.iloc[0] == pytest.approx(0.6)
    assert out_test.iloc[1] == pytest.approx(0.3)
    assert out_test.iloc[2] == pytest.approx(0.1)
    assert out_test.iloc[3] == 0.0
    assert out_test.iloc[4] == 0.0

    # transforming train again with the SAME frozen state must not have drifted
    out_train_again = enc.transform_batch(train, frozen)
    assert out_train_again.iloc[0] == pytest.approx(0.6)


def test_fit_transform_uses_only_training_state_not_test_data():
    """Leakage check: transforming test data must never alter the fitted state."""
    enc = ZScoreEncoder()
    state = enc.init_state()
    state = enc.update_state(state, pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    frozen = enc.finalize_state(state)
    frozen_copy = dict(frozen)

    huge_outlier_test = pd.Series([1e9, -1e9, 1e9])
    enc.transform_batch(huge_outlier_test, frozen)  # must not mutate frozen state
    assert frozen == frozen_copy


def test_all_stateful_encoders_registered_consistently():
    # Fourth audit pass, Finding 1: STATEFUL_ENCODERS is no longer tabular-only
    # -- text.information.surprisal/text.bm25.bm25/text.tfidf.tfidf/
    # text.semantics.similarity are genuinely mergeable corpus-statistics
    # features now routed through the same stateful fit/transform protocol
    # (see tests/test_audit4.py for their dedicated coverage).
    for name, encoder in STATEFUL_ENCODERS.items():
        spec = featm.feature_info(name)
        assert spec.stateful
        assert spec.modality in ("tabular", "text")
        # sanity: single-shot fit_transform runs end to end
        if spec.modality == "text":
            s = pd.Series(["apple apple fruit", "banana fruit", "car road"])
        elif spec.input_types == ("categorical",):
            s = pd.Series(["a", "b", "a"])
        else:
            s = pd.Series([1.0, 2.0, 3.0, 4.0])
        params = {"query": "fruit"} if name == "text.bm25.bm25" else None
        out, _ = encoder.fit_transform_batch(s, params=params)
        from featm.registry import FeatureOutput

        n_out = out.as_array().shape[0] if isinstance(out, FeatureOutput) else len(out)
        assert n_out == len(s)
