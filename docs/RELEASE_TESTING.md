# Local distribution testing

For the published package, use `pip install featm`. To check a local candidate
distribution, create a separate virtual environment
outside the source checkout and install the wheel from its directory:

```bash
python -m pip install "./featm-0.5.8-py3-none-any.whl"
python -m pip check
python -c "import featm; print(featm.__version__, featm.__file__)"
```

For IO/checkpoint examples:

```bash
python -m pip install "./featm-0.5.8-py3-none-any.whl[io]"
```

Independently test the source distribution in another environment:

```bash
python -m pip install "./featm-0.5.8.tar.gz"
```

Imports must resolve to the installed site-packages, never an editable source
checkout. Run the [quickstart](../README.md#quickstart) and
[example scripts](../examples/README.md) with the corresponding optional extras.
Preserve pip check, Python/platform/package versions and observed failures.
Builds containing changed README/LICENSE/metadata need new artifact hashes;
an older accepted wheel's digest does not identify the new distribution.

These are local review steps, not instructions to push, tag or upload a release.
The public project address is [the owner-confirmed repository](https://github.com/Harish-K22/featm).
