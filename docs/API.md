# Public API workflows

## Discovery

`list_features(pattern=None, *, modality=None, family=None)` returns names.
`list_modalities()` and `list_families()` list groups; `search(query)` returns
matching feature specifications. `feature_info(name)` returns a `FeatureSpec`
with fields including `input_types`, `requires`, `default_params`, `arity`,
`stateful`, `output_kind` and `gpu_available`.

This metadata is useful but is not a complete, machine-readable validation
schema for every parameter. For a selected callable, `help()` and
`inspect.signature()` provide additional information; explicitly forwarded
parameters may not appear in a wrapper's generic signature. Do not infer that
`**kwargs` permits private/internal parameters or arbitrary names.

```python
import inspect
import featm

name = "timeseries.statistical.mean"
spec = featm.feature_info(name)
feature = featm.create_feature(name)
print(spec.description, spec.input_types, spec.requires)
print(inspect.signature(feature))
```

## Extraction and fitted state

`create_feature(name, **params)` creates the named callable; its return shape
depends on the method. `extract(X, features=[...], params={name: {...}},
errors="raise")` coordinates explicitly selected names. Do not use
`features="all"`. See [input examples](GETTING_STARTED.md).

`fit(train, features=[...], columns={name: column}, errors="raise")` returns
`FittedFeatureExtractor`. Its `.transform(test)` applies frozen training state;
`.save(path)` persists it and `load_fitted(path)` restores it. `FeatureExtractor`
also exposes a reusable fit/transform interface; it is not a promise of
sklearn Pipeline compatibility. The README shows the recommended public
functional workflow without internal lifecycle calls.

## Return types and materialization

A simple in-memory extraction returns `pandas.DataFrame`. Failures on that
path are accessible through `.attrs.get("execution_report", {})`; it need
not contain a success report when no feature failed. GPU backend information
uses `.attrs["backend_report"]` when requested.

File-backed/out-of-core execution and fitted transformations return
`FeatureMatrix`. Use `.report.to_dict()` for its execution report,
`.to_pandas()` or `.to_numpy()` to materialize, and
`.iter_batches(batch_rows=...)` for bounded conversion. Disk-backed output
requires the IO extra. Materialization guards may reject a result too large
for the configured working-memory budget; do not assume `.to_pandas()` is free.

## Execution choices

CPU is default. GPU options include `backend="gpu"`,
`gpu_fallback="strict"` or `"fallback"`, and `gpu_memory_budget`.
Dask uses `execution="distributed", distributed_backend="dask"` with
worker/partition settings. These choices do not change a feature's input
meaning. See [scaling](SCALING.md), [Dask](DISTRIBUTED.md),
[installation](INSTALLATION.md) and [compatibility](COMPATIBILITY.md).
