# Getting started

Begin with the [README quickstart](../README.md#quickstart) and
[base installation](INSTALLATION.md). All beginner snippets below use
`errors="raise"` and the base package.

## Input conventions

For conventional time-series features:

| Shape | Meaning | Output rows |
| --- | --- | ---: |
| `(n_steps,)` | One series | 1 |
| `(n_samples, n_steps)` | A batch; rows are samples, columns time steps | `n_samples` |
| `(8, 1)` | Eight separate one-step series | 8 |

```python
import numpy as np
import pandas as pd
import featm

observations = pd.DataFrame({"value": np.arange(1.0, 9.0)})
one_series = observations["value"].to_numpy()
one = featm.extract(
    one_series, features=["timeseries.statistical.mean"], errors="raise"
)
print(one)  # One row, mean 4.5.
assert one.shape == (1, 1)
np.testing.assert_allclose(one.iloc[0, 0], 4.5)

eight_samples = featm.extract(
    observations.to_numpy(),
    features=["timeseries.statistical.mean"], errors="raise",
)
assert eight_samples.shape == (8, 1)
```

Do not apply this reshape rule to other modalities. Text uses documents;
graphs use graph objects; images and audio use arrays with their modality's
sample/channel conventions. Structured pairs or dictionaries may be required
for cross-series and multimodal methods. Start with the feature's metadata and
the [catalogue](FEATURE_UNIVERSE.md), then its implementation docstring when
the metadata does not describe a parameter fully.

## Multiple families

```python
import numpy as np
import featm

X = np.stack([
    np.sin(np.linspace(0, 8 * np.pi, 256)),
    np.cos(np.linspace(0, 8 * np.pi, 256)),
])
names = [
    "timeseries.statistical.mean",
    "timeseries.spectral.spectral_entropy",
    "timeseries.complexity.permutation_entropy",
]
result = featm.extract(X, features=names, errors="raise")
assert result.shape == (2, 3)
print(result)
```

These families share a time-series input convention. Do not combine unrelated
modalities merely because they accept NumPy arrays. Vector features expand
into columns such as `feature.name__suffix`.

## Stateful workflows

Use the [README train/test example](../README.md#reuse-state-learned-from-training-data)
or run `python examples/stateful_categories.py`. `columns={name: "category"}`
binds a tabular feature to its input column. Fitted scaling/category statistics
and corpus statistics are learned from the training source only.

Fitted `.transform()` returns `FeatureMatrix`; `.to_numpy()` and `.to_pandas()`
materialize it. `.save(path)` and `featm.load_fitted(path)` persist the fitted
state. Only load fitted artifacts/checkpoints you trust: state persistence
uses Python serialization, not a safe parser for untrusted files.

## Error policy and reports

```python
import numpy as np
import warnings
import featm

bad = np.zeros((2, 3, 4))  # Not a conventional time-series batch.
name = "timeseries.statistical.mean"
skipped = featm.extract(bad, features=[name], errors="skip")
print(skipped.attrs.get("execution_report", {}))
assert skipped.empty
assert name in skipped.attrs["execution_report"]["failed_features"]

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warned = featm.extract(bad, features=[name], errors="warn")
assert caught
assert name in warned.attrs["execution_report"]["failed_features"]

try:
    featm.extract(bad, features=[name], errors="raise")
except ValueError as exc:
    print(type(exc).__name__, str(exc))
else:
    raise AssertionError("Expected an input-shape error")
```

For a `FeatureMatrix`, inspect a different report interface:

```python
import pandas as pd
import featm

name = "tabular.numeric.abs"
matrix = featm.extract(
    pd.DataFrame({"value": [-1.0, 2.0]}),
    features=[name], columns={name: "value"}, batch_rows=1, errors="raise",
)
print(matrix.report.to_dict())
print(matrix.to_pandas())
```

`warn` or `skip` can be useful when several optional methods are attempted
and successful features should survive another feature's failure. Always
inspect failures and the output schema. Depending on the execution path,
failed features may be absent or failed batches may contain NaNs.
Structural checkpoint corruption is not a per-feature failure: it raises
even under `warn` or `skip` and preserves the invalid manifest.
