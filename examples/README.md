# Small examples

Run scripts from a source directory while using an installed FEATM environment.
They do not add `src/` to Python's import path and do not download data/models.

| Script | Dependencies | Command |
| --- | --- | --- |
| Time-series shape/means | Base | `python examples/time_series.py` |
| Training-only categories, unseen values, persistence | Base | `python examples/stateful_categories.py` |
| ECG-style rows or local label-first text data | Base | `python examples/ecg_features.py` |
| Synthetic audio / local ESC-10 subset | audio | `python examples/audio_features.py` |
| Synthetic grayscale image / local prepared array | vision | `python examples/image_features.py` |

Install extras with quoted requirements, for example
`python -m pip install "featm[audio]"`, or use
a local wheel or sdist as described in [release testing](../docs/RELEASE_TESTING.md). Read
[dataset recipes](../docs/PRACTICAL_EXAMPLES.md) before using real data.
Synthetic validations do not establish benchmark scores on those datasets.
