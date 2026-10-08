from __future__ import annotations

import numpy as np

from ...registry import Cost, register_feature


def _series_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("time series must be shape (n_series, n_steps) or (n_steps,)")
    return a


#: Step 3: elementwise-reduction family transfer-overhead threshold for
#: backend="auto" -- these are O(t) per row, cheap enough per-element that
#: host<->device transfer dominates below a few thousand rows.
_GPU_MIN_ROWS_REDUCTION = 8_000
#: Auto-threshold fix pass (v0.3.2): real Kaggle P100 element-count threshold
#: for the "cheap single-pass reduction" shape (mean/std/min/max/range/rms/
#: energy + the matching signal.time_domain features) -- see
#: featm.gpu.memory.GPU_MIN_ROWS_CALIBRATION_SEQ_LEN's sibling docstring and
#: STEP3_GPU_AUTO_THRESHOLD_FIX_HANDOFF.md for the full methodology and raw
#: numbers. The v0.3.1 default (`_GPU_MIN_ROWS_REDUCTION * 64` = 512,000
#: elements) was contradicted by the v0.3.1 final benchmark: 10,000 rows x 64
#: (640,000 elements, ABOVE that threshold) measured 0.456x -- GPU SLOWER --
#: on real hardware. A dedicated recalibration run measured this exact
#: workload again independently and got a materially different result
#: (1.484x, a GPU win) -- real run-to-run variance in Kaggle's shared P100
#: allocation, not a bug in either measurement. Rather than trust either
#: single run at the disputed boundary, this threshold is set at 6,400,000
#: elements (100,000 rows x 64) -- the value BOTH independent real-hardware
#: datasets agree is a consistent, comfortable GPU win (1.677x and 2.505x
#: respectively), with the next larger measured point in both datasets also
#: comfortably clearing the bar (1.420x at 1,000,000 rows; 2.059x at
#: 250,000 rows). Conservative on purpose: correctness (never auto-selecting
#: GPU at a workload that can plausibly lose) matters more than capturing
#: every possible GPU win.
_GPU_MIN_ELEMENTS_CHEAP_REDUCTION = 6_400_000

@register_feature(
    name="timeseries.statistical.mean",
    family="statistical",
    modality="timeseries",
    input_types=("timeseries",),
    cost=Cost.TINY,
    streamable=True,
    description="Mean value of each time series.",
    tags=("statistical",),
    gpu_available=True, gpu_backend="cupy", gpu_min_rows=_GPU_MIN_ROWS_REDUCTION,
    gpu_min_elements=_GPU_MIN_ELEMENTS_CHEAP_REDUCTION,
)
def ts_mean(x, **kwargs):
    return np.nanmean(_series_matrix(x), axis=1)


@register_feature(
    name="timeseries.statistical.std",
    family="statistical",
    modality="timeseries",
    input_types=("timeseries",),
    cost=Cost.TINY,
    streamable=True,
    description="Standard deviation of each time series.",
    tags=("statistical",),
    gpu_available=True, gpu_backend="cupy", gpu_min_rows=_GPU_MIN_ROWS_REDUCTION,
    gpu_min_elements=_GPU_MIN_ELEMENTS_CHEAP_REDUCTION,
)
def ts_std(x, **kwargs):
    return np.nanstd(_series_matrix(x), axis=1)


@register_feature(
    name="timeseries.complexity.zero_crossing_rate",
    family="complexity",
    modality="timeseries",
    input_types=("timeseries",),
    cost=Cost.LOW,
    streamable=False,
    description="Fraction of adjacent samples that cross zero.",
    tags=("crossings", "complexity"),
)
def zero_crossing_rate(x, **kwargs):
    a = _series_matrix(x)
    return (np.signbit(a[:, 1:]) != np.signbit(a[:, :-1])).mean(axis=1)


@register_feature(
    name="timeseries.spectral.spectral_entropy",
    family="spectral",
    modality="timeseries",
    input_types=("timeseries",),
    cost=Cost.MEDIUM,
    streamable=False,
    description="Normalized Shannon entropy of the one-sided power spectrum.",
    tags=("fft", "entropy", "frequency"),
    gpu_available=True, gpu_backend="cupy", gpu_min_rows=1_500,
)
def spectral_entropy(x, **kwargs):
    a = _series_matrix(x)
    power = np.abs(np.fft.rfft(a, axis=1)) ** 2
    power_sum = power.sum(axis=1, keepdims=True)
    p = np.divide(power, power_sum, out=np.zeros_like(power), where=power_sum > 0)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    h = -(p * logp).sum(axis=1)
    denom = np.log(np.maximum(power.shape[1], 2))
    return h / denom
