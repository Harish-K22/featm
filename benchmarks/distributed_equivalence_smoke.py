"""Fast semantic smoke for the Step 4 local-reference distributed coordinator.

This is intentionally not a Dask validation. It proves that the coordinator's
partition/state-merge semantics match the existing local Step-2 engine for a
representative cross-section before a third-party scheduler is involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import featm


def _compare(df, *, features, columns, params=None, partition_rows=3):
    kwargs = dict(
        features=features, columns=columns, params=params, memory_budget="64MB",
        errors="raise", output="memory",
    )
    local = featm.extract(df, batch_rows=partition_rows, **kwargs)
    dist = featm.extract(
        df, execution="distributed", distributed_backend="local", distributed_workers=3,
        distributed_partition_rows=partition_rows, **kwargs,
    )
    if local.columns != dist.columns:
        return False, f"schema mismatch: local={local.columns}, distributed={dist.columns}"
    try:
        np.testing.assert_allclose(local.to_numpy(), dist.to_numpy(), rtol=1e-10, atol=1e-10, equal_nan=True)
    except AssertionError as exc:
        return False, str(exc)
    return True, ""


def main() -> int:
    rng = np.random.default_rng(42)
    cases = []
    tab = pd.DataFrame({"x": rng.normal(size=17), "cat": list("abacabadabacabadx")})
    cases.append(("tabular_abs", tab, ["tabular.numeric.abs"], {"tabular.numeric.abs": "x"}, None))
    cases.append(("zscore", tab, ["tabular.scaling.zscore"], {"tabular.scaling.zscore": "x"}, None))
    cases.append(("category_frequency", tab, ["tabular.encoding.category_frequency"], {"tabular.encoding.category_frequency": "cat"}, None))

    ts = pd.DataFrame(rng.normal(size=(19, 32)), columns=[f"t{i}" for i in range(32)])
    cases.append(("timeseries_mean", ts, ["timeseries.statistical.mean"], None, None))

    docs = pd.DataFrame({"doc": [
        "alpha beta", "alpha gamma", "delta beta", "epsilon alpha", "gamma delta",
        "beta epsilon", "alpha alpha", "zeta beta", "eta theta", "alpha theta",
    ]})
    cases.append(("tfidf", docs, ["text.tfidf.tfidf"], {"text.tfidf.tfidf": "doc"}, {"text.tfidf.tfidf": {"max_features": 16}}))
    cases.append(("surprisal", docs, ["text.information.surprisal"], {"text.information.surprisal": "doc"}, None))
    cases.append(("similarity", docs, ["text.semantics.similarity"], {"text.semantics.similarity": "doc"}, None))

    seq = pd.DataFrame({"s": ["aaaa", "aaab", "bbbb", "bbba", "cccc"]})
    cases.append(("sequence_explicit_reference", seq, ["sequence.distance.edit_distance"], {"sequence.distance.edit_distance": "s"}, {"sequence.distance.edit_distance": {"reference": "aaaa"}}))

    imgs = pd.DataFrame({"img": [rng.random((8, 8, 3)) for _ in range(7)]})
    cases.append(("image_object_column", imgs, ["image.color.red_mean"], {"image.color.red_mean": "img"}, None))

    failures = []
    for label, df, features, columns, params in cases:
        ok, err = _compare(df, features=features, columns=columns, params=params, partition_rows=3)
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append((label, err))

    # Unsafe contract spot-checks.
    try:
        featm.extract(
            docs, features=["text.semantics.topic"], columns={"text.semantics.topic": "doc"},
            memory_budget="64MB", execution="distributed", distributed_backend="local",
            distributed_partition_rows=3, errors="raise", output="memory",
        )
    except Exception:
        print("full_dataset_rejection: PASS")
    else:
        failures.append(("full_dataset_rejection", "unsafe feature was accepted"))

    print(f"cases={len(cases)} failures={len(failures)}")
    for label, err in failures:
        print(f"FAIL {label}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
