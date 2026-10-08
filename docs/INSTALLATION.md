# Installation

Use a virtual environment and install only the backends needed by your chosen
methods. Base dependencies are numpy>=1.24, pandas>=2.0 and psutil>=5.9.

## Installation and development

After PyPI publication:

```bash
python -m pip install "featm==0.5.7"
python -m pip install "featm[io]==0.5.7"
python -m pip install "featm[distributed,io]==0.5.7"
```

No publication is performed or asserted by this staging task. For testing
supplied distributions before publication, see [release testing](RELEASE_TESTING.md).
For a checkout, see [contributing](../CONTRIBUTING.md). Double-quoted requirements
work in PowerShell and POSIX shells; they keep extra brackets a single argument.

## Extras from the actual v0.5.7 metadata

| Extra | Purpose / boundary | Declared requirements |
| --- | --- | --- |
| `io` | Parquet/Arrow, Polars interop, checkpoints and disk-backed output | `pyarrow>=14`, `polars>=1.0` |
| `distributed` | Dask task execution; add io for checkpoints/Parquet | `dask[distributed]>=2024.1` |
| `scientific` | SciPy, scikit-learn and statsmodels methods | `scipy>=1.10`, `scikit-learn>=1.3`, `statsmodels>=0.14` |
| `geometry` | SciPy/scikit-learn geometry and manifold methods | `scipy>=1.10`, `scikit-learn>=1.3` |
| `graph` | Ordinary NetworkX/igraph/Leiden; gensim below Python 3.14 | `networkx>=3.0`, `scipy>=1.10`, `python-igraph>=0.11`, `leidenalg>=0.10`, `gensim>=4.4; python_version < '3.14'` |
| `graph-node2vec` | Separate node2vec stack, certified Python 3.10–3.12 / NumPy 1.x | `networkx>=3.0; python_version < '3.13'`, `gensim>=4.4; python_version < '3.13'`, `node2vec>=0.5; python_version < '3.13'` |
| `audio` | librosa-backed audio; retain the short harmonic/tonnetz limitation | `librosa>=0.11,<1.0; python_version < '3.12'`, `librosa>=1.0; python_version >= '3.12'`, `scipy>=1.10` |
| `vision` | OpenCV/scikit-image image methods | `opencv-python>=4.8`, `scikit-image>=0.22`, `scipy>=1.10` |
| `signal` | SciPy/statsmodels, EMD and VMD | `scipy>=1.10`, `statsmodels>=0.14`, `EMD-signal>=1.6`, `vmdpy>=0.2` |
| `topology` | GUDHI and SciPy | `gudhi>=3.9`, `scipy>=1.10` |
| `pointcloud` | Open3D/SciPy; native system prerequisites may apply | `open3d>=0.18`, `scipy>=1.10` |
| `text` | scikit-learn/gensim/spaCy; no certified gensim-backed 3.14 path | `scikit-learn>=1.3`, `gensim>=4.4`, `spacy>=3.7` |
| `embeddings` | Optional model backends; selected methods may need downloaded assets | `torch>=2.2`, `timm>=1.0`, `transformers>=4.40`, `sentence-transformers>=3.0`, `Pillow>=10` |
| `calendar` | Holiday features | `holidays>=0.50` |
| `gpu` | Tested CuPy CUDA-12 backend with toolkit components | `cupy-cuda12x[ctk]>=14.0; sys_platform != 'darwin'` |
| `gpu-cupy` | Explicit synonym of gpu | `cupy-cuda12x[ctk]>=14.0; sys_platform != 'darwin'` |
| `gpu-torch` | Torch convenience extra for future routing; no current Torch GPU kernels | `torch>=2.2` |
| `gpu-rapids` | Empty future placeholder; does not install RAPIDS |  |
| `full` | Broad feature-backend aggregate, not a universal base-version recommendation | `scipy>=1.10`, `scikit-learn>=1.3`, `statsmodels>=0.14`, `networkx>=3.0`, `python-igraph>=0.11`, `leidenalg>=0.10`, `librosa>=0.11,<1.0; python_version < '3.12'`, `librosa>=1.0; python_version >= '3.12'`, `opencv-python>=4.8`, `scikit-image>=0.22`, `EMD-signal>=1.6`, `vmdpy>=0.2`, `gudhi>=3.9`, `open3d>=0.18`, `gensim>=4.4`, `spacy>=3.7`, `holidays>=0.50`, `torch>=2.2`, `timm>=1.0`, `transformers>=4.40`, `sentence-transformers>=3.0`, `Pillow>=10` |
| `dev` | Test/build tools for contributors | `pytest>=8`, `pytest-cov>=5`, `build>=1.2` |

These are declarations, not a claim that every extra works on every certified
base interpreter. See [compatibility](COMPATIBILITY.md) for tested scopes.
`full` includes feature stacks but excludes IO, Dask, CUDA and node2vec. It
includes gensim-backed text, so do not assume universal Python 3.14 usability.

For node2vec use `python -m pip install "featm[graph,graph-node2vec]==0.5.7"`
on certified Python 3.10–3.12. Its NumPy-1 restriction conflicts with CuPy 14's
NumPy-2 stack; use separate environments for those workloads.

## GPU

`python -m pip install "featm[gpu]==0.5.7"` requests
`cupy-cuda12x[ctk]>=14.0`. Toolkit components are included; a compatible NVIDIA
driver and GPU are still required. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
Only the tested Windows/CUDA-12/CuPy-14 configuration is certified here.
Installing Torch or the empty RAPIDS placeholder does not enable FEATM's
CuPy kernels. Base CPU installation remains independent of these packages.

## Troubleshooting

Use `errors="raise"` to expose the first shape/parameter/dependency error.
Inspect `feature_info(name).requires` and choose the documented extra; import
names such as `skimage` and `sklearn` correspond to distributions
`scikit-image` and `scikit-learn`. Some diagnostics name only the missing module.
Checkpointing requires IO even for in-memory input; it fails before scanning
when that backend is absent. Real model-backed methods may also require an
explicit model/asset that no pip extra can supply automatically.
