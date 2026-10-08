# FEATM v0.5.7 compatibility

The dependency policy is unchanged from the accepted v0.5.6 Step 5.5C
certification. The v0.5.7 product delta is fail-closed checkpoint JSON-root
validation. The final v0.5.7 wheel was separately tested on Windows and
Ubuntu/WSL with CPython 3.11.14, real multiprocess Dask and the established
Windows CUDA configuration. This does not constitute a new broad minor-version
or optional-dependency matrix.

| Surface | Supported by current certification / limitation |
| --- | --- |
| Base | Windows x86-64 and Ubuntu/WSL Linux x86-64, CPython 3.10–3.14 |
| Ordinary graph | Tested NetworkX/igraph/Leiden through 3.14; DeepWalk/gensim through 3.13 |
| Node2vec | Windows/Linux 3.10–3.12, separate graph-node2vec extra, NumPy 1.x |
| Text/gensim | Tested through 3.13; usable 3.14 gensim wheel/build unavailable on audited hosts |
| Audio | Tested selected librosa paths/branches; short harmonic/tonnetz remains uncertified |
| GPU | Actual Windows CPython 3.11 / RTX3050 Laptop / CUDA-12 / CuPy-14 configuration only |
| Other optional stacks | Representative per-stack certification; no blanket every-feature/model/every-minor claim |
| macOS, ARM, Python 3.15 | Not validated; this is not evidence of incompatibility |

Metadata `requires-python >=3.10` is an installation constraint, not a claim
that every future Python version or platform was tested. Extra markers that
omit a dependency outside its supported range do not make its feature work.
The `full` aggregate includes text/gensim and is not universally certified
across the base Python range; IO/Dask/CUDA/node2vec remain separate.

Audio metadata chooses librosa >=0.11,<1.0 below Python 3.12 and >=1.0 from
3.12. Certain short harmonic/tonnetz inputs produced non-finite upstream
harmonic output, including direct librosa reproductions without importing
FEATM. The 512-sample audit case is a known limitation, not a minimum-duration
threshold. Longer passing inputs do not certify every length/sample rate.
No blanket upper pin or FEATM algorithm fix is claimed to resolve it.

The tested Linux Open3D setup needed Ubuntu libusb-1.0-0 (2:1.0.27-1) in addition
to Python packages. Check current system prerequisites for your installation;
do not infer another distro/architecture's support from the WSL evidence.

GPU-capable entries are limited to the 22 CuPy kernels in the registry.
The accepted and final checks used actual hardware, CPU/GPU comparisons and
declared numerical tolerances. These do not certify every driver/device/CUDA
combination. Node2vec's NumPy-1 stack cannot share the CuPy-14 NumPy-2 setup.

Independent release audits are retained by the owner; no public audit archive
URL is established here. Public release artifacts will be rebuilt and checked
separately after documentation approval, rather than reusing old hashes for
changed README/LICENSE/metadata contents.
