# Contributing

Use explicit named methods, and include a small reproducer when reporting a
problem. A report should contain the FEATM/Python versions, OS and architecture,
installed extras and dependency versions, input shape/dtypes, exact feature
names and parameters, error policy, actual exception/report and expected result.
For CUDA/Dask, include device/runtime or worker configuration. Remove credentials
and personal data; prefer synthetic inputs over private datasets.

The owner-confirmed repository is [https://github.com/Harish-K22/featm](https://github.com/Harish-K22/featm).
Use its [issue tracker](https://github.com/Harish-K22/featm/issues) when available; no private contact
address or repository visibility is assumed by these instructions.

## Development setup

Once repository access is available, obtain a checkout from `https://github.com/Harish-K22/featm`.
The package lives in `src/featm/`; method definitions live in `src/featm/features/`
and stateful encoders in `src/featm/state.py`. From a checkout, in a virtual environment:

```bash
python -m pip install -e ".[dev,io,distributed,scientific]"
python -m pytest -q -ra
```

These are actual extras. Tests for optional backends/models may skip when their
dependencies are absent; include skip reasons. CUDA tests need the certified
GPU setup, not merely the development extra. A minimal CI job can use the above
CPU setup on the tested Windows/Ubuntu x86-64 Python scope; do not add unvalidated
platform badges or infer optional-wide support from a green base job.

Feature authoring contracts are in [docs/FEATURE_AUTHORING.md](docs/FEATURE_AUTHORING.md).
Changes should explain inputs, named outputs, dependencies, batch/stateful
semantics and relevant numerical verification. Avoid committing environments,
datasets, model weights, checkpoints or private audit material. Product/API
changes require their own review; this staged release preserves the audited implementation.
