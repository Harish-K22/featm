# Feature metadata and optional dependencies

`feature_info(name)` returns the registered `FeatureSpec`, including the description,
input types, output kind, requirements and state/GPU capabilities. Requirements
describe optional distributions used by the CPU implementation; they are not an
eager import check, a dependency resolver, or a guarantee that every parameter path
will run. Base installation still requires only NumPy, pandas and psutil.

```python
import featm
spec = featm.feature_info("timeseries.statistical.skewness")
print(spec.requires)  # ('scipy',)
print(spec.description)
```

Install `featm[scientific]` for SciPy/scikit-learn/statsmodels, `featm[vision]`
for OpenCV/scikit-image, `featm[audio]` for librosa, `featm[graph]` for graph
backends, or `featm[calendar]` for holiday calendars. Python import names can
differ: sklearn is scikit-learn, skimage is scikit-image, PIL is Pillow,
cv2 is opencv-python, igraph is supplied by the python-igraph extra and PyEMD
by EMD-signal. Existing import-name aliases in older metadata remain supported;
new corrections use distribution names. Do not pass these names blindly to pip.

Missing exact, declared optional modules receive installation hints through
`create_feature(...)(X)` and in-memory `extract`; exception type/name and original
message remain available. `errors="raise"`, `"warn"`, and `"skip"` retain their
policies. No dependency check happens when a feature is discovered or created.
Existing wrapped ImportErrors retain their own messages. Source/batch/distributed
execution keeps its existing error handling; a uniform diagnostic system is deferred.

## Conditional paths

GUDHI persistence is attempted by four topology persistent-homology features and
the Betti/Euler summaries. If unavailable or unsuccessful, SciPy computes an
H0-only minimum-spanning-tree fallback (H1/H2 summaries are empty). A flat
`requires` tuple cannot express GUDHI OR SciPy, so these alternatives are described
in prose rather than enforced as two unconditional prerequisites. GUDHI failures
can also cause fallback; installing GUDHI alone does not guarantee every path.
Short-input returns in some dynamics/recurrence functions bypass SciPy. Backend
requirements also differ: GPU use needs a compatible CuPy/CUDA installation,
and the metadata for the CPU callable is not a GPU availability probe.

## Parameters and numerical conventions

`spec.default_params` contains registry defaults, which are often empty even when
the callable has defaults. Inspect `inspect.signature(spec.implementation)` and
its implementation/docstring. `inspect.signature(create_feature(...))` shows the
generic wrapper, not a full parameter schema. Some wrappers explicitly forward
helper parameters; `**kwargs` never means arbitrary names are accepted.

Examples of public parameters: `timeseries.basic.rolling(window=8)` yields four
summaries; `timeseries.autocorrelation.acf(max_lag=10)` yields ten coefficients;
`timeseries.spectral.psd(n_bins=16)` yields sixteen values; `audio.perceptual.mfcc`
uses `sr=22050, n_mfcc=13`. Use positive integer lengths, positive sample rates,
and sufficient input length. These are practical domain constraints, not a new
machine-readable range validator; existing validation and failures are unchanged.

Permutation entropy's order=3, delay=1 and normalization are fixed helper defaults
in the current registered wrapper; they are NOT accepted public parameters.
Sample/approximate entropy likewise use fixed m=2 and r=0.2*population standard
deviation. Private `_fn` parameters are not public configuration. Do not add helper
defaults to `default_params`: that could expose parameters the wrapper ignores.

Selected descriptions now state population standard deviation (ddof=0), NaN
handling, Fisher excess kurtosis, entropy logarithms/normalization, output widths,
and short/constant-input behavior. Time-series FFT frequencies are cycles/sample;
librosa audio frequency summaries use Hz at the selected sample rate. Spectral
entropy uses uncentered power, whereas centroid/dominant-frequency use a centered
spectrum. Sample entropy's zero-match conventions are implementation-specific.
Image statistics accept (N,H,W) or (N,H,W,C): add a batch axis to a single RGB image.

The v0.5.11 source review replaces all 483 identifier-only descriptions and
adds computation semantics to six topology entries that previously documented
only conditional dependencies. All 18 modalities were reviewed; the 44 earlier
reviewed semantic descriptions are retained. Descriptions explain the current
implementation, including proxies, user-model adapters and unavailable PCL
registrations; they do not certify every optional backend or parameter path. Accepted
ranges, exact vector dimensions and conditional requirements are not yet a complete
structured schema. Search still matches names/descriptions/tags without ranking
or typo correction. No algorithm, input contract, or execution routing was changed.
