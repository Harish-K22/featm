# Release notes

## 0.5.8 — documentation and package metadata

Documentation and package metadata improvements for the public GitHub launch.

- Shared GitHub/PyPI README uses absolute links to real documentation paths.
- Removed obsolete private-repository and first-upload preparation notices.
- Clarified RGB/grayscale batch shapes, optional vision backends, error policy
  and trusted fitted-state loading, with executable examples.
- Updated release-only OIDC workflow for v0.5.8 from a public repository;
  retained owner/repository identity and approved-commit checks.
- No feature, runtime dependency, API or performance changes. Inventory remains
  552 features / 18 modalities / 182 groups / 7 stateful / 22 GPU-capable.

## 0.5.7 — initial public release

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
The documentation explains time-series shapes, error/report access,
training-only state, optional installation and execution limits.

Base certification is Windows/Ubuntu-WSL x86-64 CPython 3.10–3.14. Node2vec,
gensim/text and actual Windows CUDA support have narrower scopes; macOS, ARM
and Python 3.15 are unvalidated. Short harmonic/tonnetz inputs retain an
upstream librosa numerical limitation. See [compatibility](docs/COMPATIBILITY.md).

