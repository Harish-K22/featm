# Release notes

## 0.5.7 — initial public release preparation

FEATM offers a searchable registry for explicit, named feature extraction
across time series, tabular data and other modalities. Discover names with
`list_features`, create a callable with `create_feature`, and extract a selected
set with `extract`. Reuse training-only fitted transformations with `fit`,
`transform`, `save` and `load_fitted`.

The package includes bounded source execution, disk-backed results, compatible
checkpoint resume, Dask partition execution and a scoped 22-method CuPy GPU
surface. Backends and external models are optional; the beginner example uses
only base dependencies. The supporting catalogue retains 552 entries across
18 modalities, not a claim of universal predictive improvement.

v0.5.7 rejects non-object checkpoint JSON roots without overwriting invalid
manifests, while preserving missing-file creation and valid resume behavior.
The staged documentation explains time-series shapes, error/report access,
training-only state, optional installation and execution limits.

Base certification is Windows/Ubuntu-WSL x86-64 CPython 3.10–3.14. Node2vec,
gensim/text and actual Windows CUDA support have narrower scopes; macOS, ARM
and Python 3.15 are unvalidated. Short harmonic/tonnetz inputs retain an
upstream librosa numerical limitation. See [compatibility](docs/COMPATIBILITY.md).

These notes are prepared locally for owner review. They do not announce PyPI
publication, a public tag, a GitHub release or a change in repository visibility.
