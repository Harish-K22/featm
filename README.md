# FEATM

[Repository](https://github.com/Harish-K22/featm) · [Release notes](CHANGELOG.md)

**One searchable API for explicit, named feature extraction across data modalities.**

Use FEATM to discover an existing feature method by name, compute a small feature
set, and reuse fitted transformations on new data. You choose the methods;
FEATM does not select features automatically or promise better model accuracy.

## Quickstart

```python
import numpy as np
import featm

# Two samples, each with four time steps.
X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [10.0, 20.0, 30.0, 40.0],
])
result = featm.extract(
    X,
    features=["timeseries.statistical.mean", "timeseries.statistical.std"],
    errors="raise",
)
print(result)
```

The mean column contains `2.5` and `25.0`: one output row per input sample.
For time-series features, `(n_steps,)` means one series, and
`(n_samples, n_steps)` means a batch. An `(8, 1)` array means eight one-step
series, **not** one eight-step series. Other modalities use different inputs;
see [input conventions](docs/GETTING_STARTED.md#input-conventions).

## Installation

The planned PyPI installation command is:

```bash
python -m pip install featm
```

This preparation does not assert that v0.5.7 is already published. Use
[local distribution testing](docs/RELEASE_TESTING.md) before publication.
The quickstart needs only base dependencies: NumPy, pandas and psutil.
For checkpoints, Parquet, Arrow or disk-backed output, install the IO extra:

```bash
python -m pip install "featm[io]"
```

Quoted requirements work in PowerShell and POSIX shells. See the full
[installation table](docs/INSTALLATION.md) for minimal optional backends.

## Find a feature

```python
import numpy as np
import featm

print(featm.list_features("timeseries.fractal.*"))
print(featm.list_features("*entropy*"))
print(featm.list_features("*mean*"))

spec = featm.feature_info("timeseries.statistical.mean")
print(spec.input_types, spec.requires, spec.default_params)
mean = featm.create_feature("timeseries.statistical.mean")
print(mean(np.array([1.0, 2.0, 3.0, 4.0])))
```

Names follow `modality.family.feature_name`. Narrow a broad search such as
`timeseries.*` by family. `list_modalities()`, `list_families()` and `search()`
also help with discovery. Parameters and input types vary by feature;
[API guidance](docs/API.md) explains what metadata is available.
Some features, including MFCC, HOG and LBP, expand into multiple named columns.
There is intentionally no `features="all"` mode.

## Modalities and inputs

The registry covers time series, tabular, graph, image, audio, text, geometry,
point clouds, signal, video, information, sequence, matrix, topology, manifold,
spatial, event and multimodal data. Each family has its own input contract.
For example, text features typically accept an iterable of documents, while
graph features accept graph objects; do not reshape them using the time-series
convention.

The supporting inventory is 552 registered entries, 18 modalities and 182
modality/family groups. The [feature catalogue](docs/FEATURE_UNIVERSE.md) lists
names and dependencies. An entry's availability does not imply that its
optional backend or external model is installed.

## Reuse state learned from training data

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import featm

name = "tabular.encoding.category_frequency"
train = pd.DataFrame({"category": ["A", "A", "B", "B", "B", "C"]})
test = pd.DataFrame({"category": ["A", "B", "C", "D"]})
fitted = featm.fit(
    train, features=[name], columns={name: "category"}, errors="raise"
)
result = fitted.transform(test)
print(result.to_pandas())
np.testing.assert_allclose(result.to_numpy().ravel(), [2/6, 3/6, 1/6, 0])

with TemporaryDirectory() as directory:
    path = Path(directory) / "categories.pkl"
    fitted.save(path)
    restored = featm.load_fitted(path)
    np.testing.assert_array_equal(
        result.to_numpy(), restored.transform(test).to_numpy()
    )
```

Only `train` supplies fitted statistics. Unseen category `D` maps to zero;
transforming `test` does not refit the encoder. `extract()` fits and transforms
its own input, so use `fit()` followed by `transform()` for a train/test split.
There are seven stateful entries, including scaling and corpus-statistic text
features. See [stateful workflows](docs/GETTING_STARTED.md#stateful-workflows).

## Errors and return types

Start with `errors="raise"` so a shape, parameter or dependency error is visible.
The default `errors="skip"` isolates feature failures and can return an empty
result if none succeeds. `errors="warn"` also emits warnings. An empty result
is not proof that every requested feature ran.

For a simple in-memory `pandas.DataFrame` result, inspect
`result.attrs.get("execution_report", {})`. The report is present when this
path records failures. Out-of-core extraction and fitted transformations use
`FeatureMatrix`: inspect `result.report` and convert with `result.to_pandas()`.
For large results, iterate batches rather than materializing everything.
See [tested error examples](docs/GETTING_STARTED.md#error-policy-and-reports).

## Larger data, checkpointing and execution backends

File-backed extraction can read bounded batches, save disk-backed results and
resume a compatible checkpoint. IO is required for checkpoints even when the
input is already in memory. Existing incompatible or structurally invalid
checkpoints fail closed; choose a fresh directory when you want a new run.
Memory budgets guide execution; they are not a hard cap on total process RSS.

CPU is the default. Optional CuPy kernels cover 22 features; strict GPU mode
raises when a requested GPU implementation is unavailable, while fallback mode
reports a CPU fallback. Dask distributes eligible features according to their
batch semantics. Start with [scaling and checkpoints](docs/SCALING.md),
[GPU setup](docs/INSTALLATION.md#gpu) or [Dask](docs/DISTRIBUTED.md).

## Examples and documentation

- [Getting started](docs/GETTING_STARTED.md): shapes, multiple families,
  reports and fitted state.
- [Installation](docs/INSTALLATION.md) and [compatibility](docs/COMPATIBILITY.md).
- [Small executable examples](examples/README.md), including audio, images
  and dataset-loading recipes without model-performance claims.
- [Evidence and benchmark interpretation](docs/BENCHMARKS.md).
- [Feature authoring](docs/FEATURE_AUTHORING.md) and
  [architecture](docs/ARCHITECTURE.md) for contributors.

## Compatibility and known limits

Base certification covers Windows x86-64 and Ubuntu/WSL Linux x86-64 with
CPython 3.10–3.14. Optional extras have narrower scopes: node2vec is certified
on 3.10–3.12, and gensim-backed text/DeepWalk paths through 3.13.
`full` is not a universal recommendation across all base Python versions.
Real GPU evidence is limited to the tested Windows/CUDA-12/CuPy-14 configuration.
macOS, ARM and Python 3.15 are **not validated**, rather than proven incompatible.

Certain short harmonic/tonnetz inputs can yield non-finite values inside
librosa. This was reproduced upstream without importing FEATM. No minimum
safe duration, universal sample-rate claim or upstream fix is asserted.
See [compatibility details](docs/COMPATIBILITY.md).

FEATM is explicit extraction, not AutoML, relational DFS or automatic feature
selection. New windowing infrastructure and sklearn Pipeline integration are
outside v0.5.7. A legacy `AutoFeatures` planner remains for compatibility; it
is not the recommended discovery or extraction workflow.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and reproducible bug reports.
The owner-confirmed project address is [GitHub](https://github.com/Harish-K22/featm); report reproducible
problems using the [issue tracker](https://github.com/Harish-K22/featm/issues) once the repository is available.

FEATM's own code is distributed under the [MIT License](LICENSE),
Copyright (c) 2026 Harish K. Optional dependencies, datasets and external models
retain their own terms. See [third-party scope](THIRD_PARTY_NOTICES.md).
