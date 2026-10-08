"""GPU reduction kernels (Step 3): the CPU reference each mirrors is named in its docstring.

Every function here takes a device array ``a`` of shape ``(n_rows, n_steps)``
(already transferred to the device and reshaped by the dispatcher -- see
``featm.gpu.dispatch``) and an ``xp`` module (the ``cupy`` module for this
milestone; see ``featm.gpu.backend.xp_module``), and returns a device array
of shape ``(n_rows,)`` or ``(n_rows, k)``, matching its CPU counterpart's
*mathematical* output exactly (within the feature's declared
``gpu_numerical_tolerance``) -- these are direct translations, not
reimplementations, of the CPU formulas in
``featm.features.timeseries.basic``/``.full`` and
``featm.features.signal.full``, so a reviewer can diff the two side by side.

**Dtype policy (Step 3 spec §9):** every kernel computes in the array's own
dtype -- the dispatcher casts the input to ``float32`` by default (this
family's declared ``gpu_output_dtype``) before calling in, and back to the
caller's original dtype is not attempted (the CPU path already returns
float64; GPU float32 is a documented, tolerance-bounded approximation, never
silently claimed to be bit-identical). NaNs propagate through ``xp.nanmin``/
``xp.nanmean``/etc. exactly as they do through NumPy's -- no special-casing.
"""
from __future__ import annotations

from typing import Any


def gpu_mean(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.mean`` (``np.nanmean(a, axis=1)``)."""
    return xp.nanmean(a, axis=1)


def gpu_std(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.std`` (``np.nanstd(a, axis=1)``, ddof=0)."""
    return xp.nanstd(a, axis=1)


def gpu_min(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.min`` (``np.nanmin(r)`` per row)."""
    return xp.nanmin(a, axis=1)


def gpu_max(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.max`` (``np.nanmax(r)`` per row)."""
    return xp.nanmax(a, axis=1)


def gpu_median(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.median`` (``np.nanmedian(r)`` per row)."""
    return xp.nanmedian(a, axis=1)


def gpu_range(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.range`` (``np.nanmax(r) - np.nanmin(r)``)."""
    return xp.nanmax(a, axis=1) - xp.nanmin(a, axis=1)


def gpu_rms(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.rms`` / ``signal.time_domain.rms`` (``sqrt(nanmean(r*r))``)."""
    return xp.sqrt(xp.nanmean(a * a, axis=1))


def gpu_energy(a: Any, xp) -> Any:
    """Mirrors ``timeseries.statistical.energy`` (``np.nansum(r*r)``)."""
    return xp.nansum(a * a, axis=1)


def gpu_peak_to_peak(a: Any, xp) -> Any:
    """Mirrors ``signal.time_domain.peak_to_peak`` (``np.nanmax(r) - np.nanmin(r)``)."""
    return xp.nanmax(a, axis=1) - xp.nanmin(a, axis=1)


def gpu_line_length(a: Any, xp) -> Any:
    """Mirrors ``signal.time_domain.line_length`` (``np.nansum(np.abs(np.diff(r)))``)."""
    d = xp.diff(a, axis=1)
    return xp.nansum(xp.abs(d), axis=1)


def gpu_crest_factor(a: Any, xp) -> Any:
    """Mirrors ``signal.time_domain.crest_factor`` (``max(|r|) / max(rms(r), 1e-15)``)."""
    peak = xp.nanmax(xp.abs(a), axis=1)
    rms = xp.sqrt(xp.nanmean(a * a, axis=1))
    return peak / xp.maximum(rms, 1e-15)


def gpu_impulse_factor(a: Any, xp) -> Any:
    """Mirrors ``signal.time_domain.impulse_factor`` (``max(|r|) / max(mean(|r|), 1e-15)``)."""
    peak = xp.nanmax(xp.abs(a), axis=1)
    mean_abs = xp.nanmean(xp.abs(a), axis=1)
    return peak / xp.maximum(mean_abs, 1e-15)


#: name -> (gpu kernel, mirrored CPU feature names sharing this exact formula).
#: Both `timeseries.statistical.rms`/`range` and `signal.time_domain.rms`/
#: `peak_to_peak` map to the SAME kernel here because their CPU formulas are
#: byte-for-byte identical (`_rows(x, lambda r: ...)` with the same lambda) --
#: registering one kernel under two feature names is not a shortcut that
#: risks divergence, it is documenting a fact already true on the CPU side.
REDUCTION_KERNELS: dict[str, Any] = {
    "timeseries.statistical.mean": gpu_mean,
    "timeseries.statistical.std": gpu_std,
    "timeseries.statistical.min": gpu_min,
    "timeseries.statistical.max": gpu_max,
    "timeseries.statistical.median": gpu_median,
    "timeseries.statistical.range": gpu_range,
    "timeseries.statistical.rms": gpu_rms,
    "timeseries.statistical.energy": gpu_energy,
    "signal.time_domain.rms": gpu_rms,
    "signal.time_domain.peak_to_peak": gpu_peak_to_peak,
    "signal.time_domain.line_length": gpu_line_length,
    "signal.time_domain.crest_factor": gpu_crest_factor,
    "signal.time_domain.impulse_factor": gpu_impulse_factor,
}
