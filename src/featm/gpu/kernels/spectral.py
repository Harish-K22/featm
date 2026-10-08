"""GPU FFT/spectral kernels (Step 3): one shared FFT feeding many features.

Mirrors ``featm.features._utils.spectral_parts``'s exact math and its
shared-intermediate-cache pattern, but device-resident and backed by
``featm.gpu.cache`` instead of ``featm.cache``. This is the highest-leverage
shared primitive in the Step 3 GPU surface (Step 3 spec §7): selecting
several ``timeseries.spectral.*``/``signal.frequency.*`` features together
computes ONE GPU FFT per batch, not one per feature.

**Two distinct FFT computations exist on the CPU side and are preserved as
two distinct GPU computations here, not unified** -- unifying them would be
a real (if well-intentioned) mathematical behavior change relative to the
CPU reference, exactly the class of mistake Step 2's audits exist to catch:

- :func:`gpu_spectral_parts` -- **centered** (mean-subtracted) FFT, shared by
  ``timeseries.spectral.{centroid,dominant_frequency,flatness,periodogram,
  bandpower}`` and ``signal.frequency.{fft,psd}`` /
  ``signal.entropy.spectral_entropy`` (all of which call
  ``featm.features._utils.spectral_parts`` on the CPU side).
- :func:`gpu_spectral_entropy_uncentered` -- **uncentered** FFT, used only by
  ``timeseries.spectral.spectral_entropy`` (``featm.features.timeseries.basic
  .spectral_entropy``, which computes its own independent, non-mean-subtracted
  ``rfft`` and does NOT call ``spectral_parts`` at all -- an existing CPU-side
  inconsistency this pass preserves rather than "fixes", since a GPU pass's
  job is to accelerate the reference, not silently change it).

**Audit-fix pass (v0.3.1): ``timeseries.spectral.fft`` and ``timeseries.
spectral.cepstrum`` are deliberately NOT GPU-accelerated** (no
``gpu_fft_vector``/``gpu_cepstrum`` kernel here, and both are
``gpu_available=False`` in ``featm.features.timeseries.full``). Independent
audit found two real, distinct problems this pass could not responsibly
paper over:

- ``timeseries.spectral.fft`` mixes magnitude AND phase columns in one
  output vector. Its phase columns have a genuine cuFFT-vs-pocketfft
  DC-bin instability (~pi disagreement, not a precision bug -- unchanged by
  switching to float64), which had been accommodated with a blanket
  ``gpu_numerical_tolerance=(1e-2, 3.2)`` applied to the WHOLE vector --
  loose enough that a real ~1.0 magnitude-column corruption would pass
  validation undetected (confirmed by an adversarial test: see
  ``tests/test_gpu_dispatch.py``). ``signal.frequency.fft`` is unaffected and
  stays GPU-accelerated -- it is magnitude-only, no phase columns, no DC-bin
  instability to accommodate.
- ``timeseries.spectral.cepstrum``'s ``irfft(log(magnitude))`` chain showed
  real GPU-vs-CPU relative errors up to ~24% even at float64 (a genuine
  log-domain error-amplification property of the algorithm, not a coding
  bug) -- accepted at the time behind a ``gpu_numerical_tolerance=(0.3,
  0.05)`` that is too loose to call a "GPU accelerates the same feature"
  claim, without materially tighter per-coefficient evidence than a single
  aggregate pass counter provided.

This project has repeatedly prioritized correctness over GPU-coverage count
(see the Step 2 audit history) -- 22 genuinely tight-tolerance GPU features
is preferred here over 24 where two silently ship a validator that cannot
catch a real wrong answer.
"""
from __future__ import annotations

from typing import Any

from ..cache import current_gpu_cache, gpu_array_cache_key


def _spectral_parts_impl_gpu(a: Any, xp) -> tuple:
    centered = a - xp.nanmean(a, axis=1, keepdims=True)
    centered = xp.nan_to_num(centered)
    spec = xp.fft.rfft(centered, axis=1)
    spec[:, 0] = 0.0
    mag = xp.abs(spec)
    power = mag ** 2
    freqs = xp.fft.rfftfreq(a.shape[1], d=1.0)
    if xp.__name__ == "cupy":
        freqs = freqs.astype(a.dtype)
    return spec, mag, power, freqs


def gpu_spectral_parts(a: Any, xp, *, batch_id: int) -> tuple:
    """(spec, mag, power, freqs) for the CENTERED FFT, cached once per GPU batch scope.

    ``batch_id`` is a caller-supplied identity for "which batch is this" --
    see ``featm.gpu.cache.gpu_array_cache_key``'s docstring for why this
    (not content hashing) is the correctness anchor for a device-resident
    cache key.
    """
    cache = current_gpu_cache()
    if cache is None:
        return _spectral_parts_impl_gpu(a, xp)
    key = gpu_array_cache_key("gpu_spectral_parts", batch_id, a.shape, str(a.dtype))
    return cache.get_or_compute(key, lambda: _spectral_parts_impl_gpu(a, xp))


def gpu_centroid(a: Any, xp, *, batch_id: int) -> Any:
    """Mirrors ``timeseries.spectral.centroid``."""
    _, mag, _, f = gpu_spectral_parts(a, xp, batch_id=batch_id)
    den = mag.sum(axis=1)
    num = (mag * f).sum(axis=1)
    return xp.where(den > 0, num / xp.where(den > 0, den, 1), 0.0)


def gpu_dominant_frequency(a: Any, xp, *, batch_id: int) -> Any:
    """Mirrors ``timeseries.spectral.dominant_frequency``."""
    _, _, p, f = gpu_spectral_parts(a, xp, batch_id=batch_id)
    idx = xp.argmax(p, axis=1)
    return f[idx]


def gpu_flatness(a: Any, xp, *, batch_id: int) -> Any:
    """Mirrors ``timeseries.spectral.flatness`` (geometric mean / arithmetic mean of power)."""
    _, _, p, _ = gpu_spectral_parts(a, xp, batch_id=batch_id)
    g = xp.exp(xp.mean(xp.log(xp.maximum(p, 1e-15)), axis=1))
    arith = xp.mean(p, axis=1)
    return xp.where(arith > 0, g / xp.where(arith > 0, arith, 1), 0.0)


def gpu_periodogram(a: Any, xp, *, batch_id: int, n_bins: int = 16) -> Any:
    """Mirrors ``timeseries.spectral.periodogram`` / ``signal.frequency.psd`` (both default ``n_bins=16``, identical formula)."""
    _, _, p, _ = gpu_spectral_parts(a, xp, batch_id=batch_id)
    k = min(n_bins, p.shape[1])
    names = tuple(f"power_{i:02d}" for i in range(k))
    return p[:, :k], names


def gpu_fft_magnitude_only(a: Any, xp, *, batch_id: int, n_coeffs: int = 16) -> Any:
    """Mirrors ``signal.frequency.fft`` -- magnitude ONLY (no phase columns, unlike ``timeseries.spectral.fft``)."""
    spec, _, _, _ = gpu_spectral_parts(a, xp, batch_id=batch_id)
    k = min(n_coeffs, spec.shape[1])
    names = tuple(f"mag_{i:02d}" for i in range(k))
    return xp.abs(spec[:, :k]), names


def gpu_bandpower(a: Any, xp, *, batch_id: int, bands: int = 4) -> Any:
    """Mirrors ``timeseries.spectral.bandpower`` -- power summed over ``bands`` equal chunks."""
    import numpy as np

    _, _, p, _ = gpu_spectral_parts(a, xp, batch_id=batch_id)
    n_freqs = int(p.shape[1])
    # np.array_split's chunk boundaries are computed on the host (cheap,
    # shape-only) then applied as device-side slices -- this exactly
    # reproduces np.array_split's own (non-uniform-when-not-evenly-divisible)
    # chunking rule without needing a GPU-native array_split.
    chunks = np.array_split(np.arange(n_freqs), bands)
    cols = [p[:, c].sum(axis=1) if len(c) else xp.zeros(p.shape[0], dtype=p.dtype) for c in chunks]
    names = tuple(f"band_{i}" for i in range(bands))
    return xp.stack(cols, axis=1), names


def gpu_spectral_entropy_centered(a: Any, xp, *, batch_id: int) -> Any:
    """Mirrors ``signal.entropy.spectral_entropy`` (via the CENTERED ``spectral_parts``/``shannon_entropy_prob``)."""
    _, _, power, _ = gpu_spectral_parts(a, xp, batch_id=batch_id)
    p = xp.clip(power, 0, None)
    s = p.sum(axis=1, keepdims=True)
    p = xp.where(s > 1e-15, p / xp.where(s > 1e-15, s, 1), 0.0)
    logp = xp.where(p > 0, xp.log(xp.where(p > 0, p, 1)), 0.0)
    h = -(p * logp).sum(axis=1)
    k = p.shape[1]
    if k > 1:
        h = h / float(xp.log(xp.asarray(k, dtype=p.dtype)))
    return h


def gpu_spectral_entropy_uncentered(a: Any, xp) -> Any:
    """Mirrors ``timeseries.spectral.spectral_entropy`` -- its OWN, uncentered, uncached FFT.

    Deliberately not routed through :func:`gpu_spectral_parts`/the GPU cache:
    the CPU reference (``featm.features.timeseries.basic.spectral_entropy``)
    computes an independent, non-mean-subtracted FFT and was never wired
    into the shared ``spectral_parts`` cache on the CPU side either -- this
    mirrors that, not "improves" it.
    """
    power = xp.abs(xp.fft.rfft(a, axis=1)) ** 2
    power_sum = power.sum(axis=1, keepdims=True)
    p = xp.where(power_sum > 0, power / xp.where(power_sum > 0, power_sum, 1), 0.0)
    logp = xp.where(p > 0, xp.log(xp.where(p > 0, p, 1)), 0.0)
    h = -(p * logp).sum(axis=1)
    denom = float(xp.log(xp.asarray(max(power.shape[1], 2), dtype=p.dtype)))
    return h / denom


#: Feature name -> (kernel, needs_batch_id, extra static kwargs). Every one
#: of these shares gpu_spectral_parts's cache except spectral_entropy
#: (uncentered, standalone -- see module docstring).
SPECTRAL_KERNELS: dict[str, Any] = {
    "timeseries.spectral.centroid": gpu_centroid,
    "timeseries.spectral.dominant_frequency": gpu_dominant_frequency,
    "timeseries.spectral.flatness": gpu_flatness,
    "timeseries.spectral.periodogram": gpu_periodogram,
    "signal.frequency.psd": gpu_periodogram,
    "timeseries.spectral.bandpower": gpu_bandpower,
    "signal.frequency.fft": gpu_fft_magnitude_only,
    "signal.entropy.spectral_entropy": gpu_spectral_entropy_centered,
}

#: Standalone (uncentered, uncached) spectral kernels -- called without a
#: batch_id/shared cache.
STANDALONE_SPECTRAL_KERNELS: dict[str, Any] = {
    "timeseries.spectral.spectral_entropy": gpu_spectral_entropy_uncentered,
}

#: Features whose kernel returns ``(array, names)`` (a FeatureOutput-style
#: vector result) rather than a plain 1-D-per-row array -- dispatch.py uses
#: this to know which convention to expect back without inspecting the
#: return value's shape (an empty/degenerate batch could have an ambiguous
#: shape either way).
VECTOR_SPECTRAL_FEATURES: frozenset[str] = frozenset({
    "timeseries.spectral.periodogram", "signal.frequency.psd",
    "timeseries.spectral.bandpower", "signal.frequency.fft",
})
