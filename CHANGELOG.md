# Release notes

## 0.5.11 — documentation and feature descriptions

- Correct compatibility history without expanding certification and make feature-authoring examples self-contained or explicitly schematic.
- Replace 483 identifier-only descriptions and expand six dependency-only topology descriptions using implementation evidence across all 18 modalities. Preserve the prior 44 semantic descriptions and conditional dependency explanations.
- Preserve all feature identifiers, dependency declarations and numerical implementations; add description-quality and documentation regression checks.

## 0.5.10 — recurrence dependency metadata

- Declare SciPy for recurrence rate, determinism, laminarity and trapping time.
- Reuse existing optional-dependency hints so ordinary missing-SciPy failures
  identify `featm[scientific]`; SciPy remains optional for the base installation.
- Add focused published-v0.5.9 numerical parity and error-policy regressions.
  Algorithms, defaults, shapes and short-input early returns are unchanged.


## 0.5.9 — dependency metadata and feature descriptions

- Correct 52 optional-dependency declarations using implementation/helper evidence;
  document six conditional GUDHI/SciPy alternatives without eager enforcement.
- Add narrow installation hints for declared missing root modules in in-memory
  extraction and created callables, preserving error policies and exception details.
- Improve 50 reviewed descriptions, including six conditional-dependency notes;
  document parameter discovery and numerical conventions. Algorithms are unchanged.
- Add deterministic metadata/dependency regressions and numerical parity evidence;
  prepare the version-specific release guard for independent review. No publication.


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

