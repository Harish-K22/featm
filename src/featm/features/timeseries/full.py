from __future__ import annotations

from .._utils import stable_phase, cepstrum_values

from collections import Counter
import math
import numpy as np

from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import (
    approximate_entropy_1d, as_2d_series, autocorr_row, compressed_ratio,
    dfa_1d, higuchi_fd_1d, hurst_1d, katz_fd_1d, lempel_ziv_complexity_1d,
    permutation_entropy_1d, petrosian_fd_1d, recurrence_matrix_1d, rqa_stats_1d,
    sample_entropy_1d, shannon_entropy_prob, spectral_entropy_rows, spectral_parts,
)


def _reg(name, family, *, cost=Cost.LOW, output_kind="scalar", tags=(), requires=(), description="", cache_key=None, structured=False, **gpu_extra):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:
            return fn
        # Fourth audit pass, Finding 4/13: timeseries.cross_series.* features
        # (structured=True below) expect a per-sample (2, t) pair via
        # _paired() -- a row-oriented Source batch cannot represent that;
        # a 2-row batch is silently (mis)interpreted as the pair itself
        # (the exact cross_correlation reproducer). Every other timeseries
        # feature (the overwhelming majority) genuinely treats one row as
        # one independent series and is unaffected.
        extra = {"batch_semantics": BatchSemantics.STRUCTURED_SAMPLE} if structured else {}
        # Step 3: gpu_extra carries gpu_available/gpu_backend/gpu_min_rows
        # for the curated subset of features with a real GPU kernel (see
        # featm.gpu.kernels) -- everything else in this file is unaffected
        # (gpu_available defaults to False).
        extra.update(gpu_extra)
        return register_feature(name=name,family=family,modality="timeseries",input_types=("timeseries",),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=requires,description=description or name,cache_key=cache_key,**extra)(fn)
    return deco


def _rows(x, fn):
    a=as_2d_series(x); return np.asarray([fn(r) for r in a],dtype=float)


def _paired(x):
    if isinstance(x,(tuple,list)) and len(x)==2:
        a,b=as_2d_series(x[0]),as_2d_series(x[1])
    else:
        z=np.asarray(x,dtype=float)
        if z.ndim==2 and z.shape[0]==2: a,b=z[0][None,:],z[1][None,:]
        elif z.ndim==3 and z.shape[1]==2: a,b=z[:,0,:],z[:,1,:]
        else: raise ValueError('cross-series features expect (A,B) or shape (n,2,t)')
    n=min(len(a),len(b));return a[:n],b[:n]


# Statistical family. Step 3: min/max/median/range/rms/energy have a real
# GPU kernel (featm.gpu.kernels.reductions) -- skewness/kurtosis are NOT
# GPU-accelerated this milestone (scipy's bias-corrected, nan_policy='omit'
# per-row moment formulas need more careful GPU numerical validation than
# this pass covers; see STEP3_GPU_HANDOFF.md's scope section).
_GPU_REDUCTION_NAMES = {'min', 'max', 'median', 'range', 'rms', 'energy'}
# Auto-threshold fix pass (v0.3.2): `median` is NOT the same GPU-vs-CPU
# workload shape as min/max/range/rms/energy -- np.nanmedian is a
# sort/partition per row, materially more CPU-expensive than a single-pass
# reduction, so its real measured GPU speedup was already large (21.9x) at
# the smallest calibrated size (1,000 rows x 64 = 64,000 elements), with the
# next larger point (5,000 rows, 61.4x) confirming it -- a wide safety
# margin above the required 1.10x bar, unlike the disputed
# 640,000-element region seen for the cheap reductions (see
# featm.features.timeseries.basic._GPU_MIN_ELEMENTS_CHEAP_REDUCTION's
# docstring for that comparison). Using the shared, much-higher, cheap-
# reduction threshold for median instead would just needlessly leave real
# GPU speedup on the table for a feature real measurement says wins early
# and by a wide margin. See STEP3_GPU_AUTO_THRESHOLD_FIX_HANDOFF.md for the
# full calibration table.
_GPU_MIN_ELEMENTS_MEDIAN = 64_000
from .basic import _GPU_MIN_ELEMENTS_CHEAP_REDUCTION as _GPU_MIN_ELEMENTS_CHEAP
for _name,_fn in {
    'min':lambda r:np.nanmin(r),'max':lambda r:np.nanmax(r),'median':lambda r:np.nanmedian(r),
    'range':lambda r:np.nanmax(r)-np.nanmin(r),'rms':lambda r:np.sqrt(np.nanmean(r*r)),
    'energy':lambda r:np.nansum(r*r),
    'skewness':lambda r:__import__('scipy').stats.skew(r,nan_policy='omit',bias=False),
    'kurtosis':lambda r:__import__('scipy').stats.kurtosis(r,nan_policy='omit',bias=False),
}.items():
    if _name in _GPU_REDUCTION_NAMES:
        _gpu_min_elements = _GPU_MIN_ELEMENTS_MEDIAN if _name == 'median' else _GPU_MIN_ELEMENTS_CHEAP
        _gpu_kw = dict(gpu_available=True, gpu_backend='cupy', gpu_min_rows=8_000, gpu_min_elements=_gpu_min_elements)
    else:
        _gpu_kw = {}
    @_reg(f'timeseries.statistical.{_name}','statistical',cost=Cost.TINY,tags=('statistical',),**_gpu_kw)
    def _stat(x,_fn=_fn,**kwargs): return _rows(x,_fn)

@_reg('timeseries.statistical.extrema','statistical',output_kind='vector',tags=('extrema',))
def ts_extrema(x,**kwargs):
    a=as_2d_series(x); arr=np.column_stack([np.nanmin(a,1),np.nanmax(a,1),np.nanargmin(np.nan_to_num(a,nan=np.inf),axis=1),np.nanargmax(np.nan_to_num(a,nan=-np.inf),axis=1)])
    return FeatureOutput(arr,('min','max','argmin','argmax'))

@_reg('timeseries.statistical.crossings','statistical',output_kind='vector',tags=('crossings',))
def ts_crossings(x,**kwargs):
    a=as_2d_series(x); med=np.nanmedian(a,axis=1,keepdims=True)
    z=(np.signbit(a[:,1:])!=np.signbit(a[:,:-1])).sum(1)
    m=((a[:,1:]-med)*(a[:,:-1]-med)<0).sum(1)
    return FeatureOutput(np.column_stack([z,m]),('zero_crossings','median_crossings'))

@_reg('timeseries.statistical.change_statistics','statistical',output_kind='vector',tags=('change',))
def ts_change_stats(x,**kwargs):
    a=as_2d_series(x); d=np.diff(a,axis=1)
    return FeatureOutput(np.column_stack([np.nanmean(d,1),np.nanstd(d,1),np.nanmean(np.abs(d),1),np.nanmax(np.abs(d),axis=1)]),('mean_diff','std_diff','mean_abs_diff','max_abs_diff'))


# Basic window transforms summarized to fixed-width outputs.
@_reg('timeseries.basic.differences','basic',output_kind='vector',cost=Cost.TINY,tags=('differences',))
def ts_differences(x,**kwargs):
    a=as_2d_series(x); d=np.diff(a,axis=1);return FeatureOutput(np.column_stack([np.nanmean(d,1),np.nanstd(d,1),np.nanmedian(d,axis=1)]),('mean','std','median'))

@_reg('timeseries.basic.derivatives','basic',output_kind='vector',cost=Cost.TINY,tags=('derivative',))
def ts_derivatives(x,**kwargs):
    a=as_2d_series(x); d1=np.gradient(a,axis=1);d2=np.gradient(d1,axis=1);return FeatureOutput(np.column_stack([np.nanmean(np.abs(d1),1),np.nanstd(d1,1),np.nanmean(np.abs(d2),1),np.nanstd(d2,1)]),('abs_d1_mean','d1_std','abs_d2_mean','d2_std'))

@_reg('timeseries.basic.lags','basic',output_kind='vector',cost=Cost.TINY,tags=('lags',))
def ts_lags(x,**kwargs):
    a=as_2d_series(x); lags=[1,2,4,8]; out=[]
    for lag in lags: out.append(np.nanmean(a[:,lag:]*a[:,:-lag],axis=1) if a.shape[1]>lag else np.full(len(a),np.nan))
    return FeatureOutput(np.column_stack(out),tuple(f'lag_{l}' for l in lags))

@_reg('timeseries.basic.rolling','basic',output_kind='vector',cost=Cost.LOW,tags=('rolling',))
def ts_rolling(x,window=8,**kwargs):
    a=as_2d_series(x);out=[]
    for r in a:
        if len(r)<window:out.append([np.nan]*4);continue
        w=np.lib.stride_tricks.sliding_window_view(r,window)
        means=np.nanmean(w,1);stds=np.nanstd(w,1);out.append([np.nanmean(means),np.nanstd(means),np.nanmean(stds),np.nanmax(stds)])
    return FeatureOutput(np.asarray(out),('rolling_mean_mean','rolling_mean_std','rolling_std_mean','rolling_std_max'))

@_reg('timeseries.basic.expanding','basic',output_kind='vector',cost=Cost.LOW,tags=('expanding',))
def ts_expanding(x,**kwargs):
    a=as_2d_series(x);out=[]
    for r in a:
        cs=np.nancumsum(np.nan_to_num(r)); n=np.arange(1,len(r)+1); em=cs/n
        out.append([em[-1],np.nanstd(em),np.nanmax(em)-np.nanmin(em)])
    return FeatureOutput(np.asarray(out),('final_expanding_mean','expanding_mean_std','expanding_mean_range'))


# Autocorrelation.
@_reg('timeseries.autocorrelation.acf','autocorrelation',output_kind='vector',cost=Cost.MEDIUM,tags=('acf',))
def ts_acf(x,max_lag=10,**kwargs):
    a=as_2d_series(x);arr=np.array([[autocorr_row(r,l) for l in range(1,max_lag+1)] for r in a]);return FeatureOutput(arr,tuple(f'lag_{l:02d}' for l in range(1,max_lag+1)))

@_reg('timeseries.autocorrelation.autocovariance','autocorrelation',output_kind='vector',cost=Cost.LOW,tags=('autocovariance',))
def ts_autocov(x,max_lag=8,**kwargs):
    a=as_2d_series(x);res=[]
    for r in a:
        rr=np.nan_to_num(r-np.nanmean(r));res.append([np.mean(rr[:-l]*rr[l:]) if len(rr)>l else np.nan for l in range(1,max_lag+1)])
    return FeatureOutput(np.asarray(res),tuple(f'lag_{l:02d}' for l in range(1,max_lag+1)))

@_reg('timeseries.autocorrelation.pacf','autocorrelation',output_kind='vector',cost=Cost.MEDIUM,tags=('pacf',),requires=('statsmodels',))
def ts_pacf(x,nlags=8,**kwargs):
    from statsmodels.tsa.stattools import pacf
    out=[]
    for r in as_2d_series(x):
        rr=r[np.isfinite(r)]; nl=min(nlags,max(1,len(rr)//3-1))
        try:v=pacf(rr,nlags=nl,method='ywm')[1:]
        except Exception:v=np.full(nl,np.nan)
        v=np.pad(v,(0,nlags-len(v)),constant_values=np.nan);out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'lag_{i:02d}' for i in range(1,nlags+1)))

@_reg('timeseries.autocorrelation.lag1','autocorrelation',cost=Cost.TINY,tags=('acf',))
def ts_lag1(x,**kwargs): return _rows(x,lambda r:autocorr_row(r,1))


# Spectrum. All of these share one FFT (see featm.cache / _utils.spectral_parts).
# Step 3: all 7 share featm.gpu.kernels.spectral's centered GPU FFT
# (gpu_spectral_parts), cached once per GPU batch exactly like the CPU side.
_GPU_SPECTRAL_MIN_ROWS = 1_500  # FFT does more compute/row than a plain reduction; pays off sooner.

@_reg('timeseries.spectral.centroid','spectral',tags=('fft',),cache_key='timeseries.spectral_parts',gpu_available=True,gpu_backend='cupy',gpu_min_rows=_GPU_SPECTRAL_MIN_ROWS)
def sp_centroid(x,**kwargs):
    _,mag,_,f=spectral_parts(x);den=mag.sum(1);return np.divide((mag*f).sum(1),den,out=np.zeros(len(mag)),where=den>0)

@_reg('timeseries.spectral.dominant_frequency','spectral',tags=('fft',),cache_key='timeseries.spectral_parts',gpu_available=True,gpu_backend='cupy',gpu_min_rows=_GPU_SPECTRAL_MIN_ROWS)
def sp_dom(x,**kwargs):
    _,_,p,f=spectral_parts(x);return f[np.argmax(p,axis=1)]

# gpu_output_dtype='float64': found by this milestone's real Kaggle
# validation (Tesla P100) -- flatness computes a geometric mean via
# exp(mean(log(power))), the same log-domain error-amplification pattern as
# cepstrum (float32 rounding in the power spectrum gets amplified by log()
# for low-power frequency bins). float64 substantially narrows this.
@_reg('timeseries.spectral.flatness','spectral',tags=('fft',),cache_key='timeseries.spectral_parts',gpu_available=True,gpu_backend='cupy',gpu_min_rows=_GPU_SPECTRAL_MIN_ROWS,gpu_output_dtype='float64')
def sp_flat(x,**kwargs):
    _,_,p,_=spectral_parts(x);g=np.exp(np.mean(np.log(np.maximum(p,1e-15)),1));a=np.mean(p,1);return np.divide(g,a,out=np.zeros_like(g),where=a>0)

# Audit-fix pass (v0.3.1): GPU acceleration REMOVED (was gpu_available=True
# with gpu_output_dtype='float64', gpu_numerical_tolerance=(1e-2, 3.2)).
# Root cause of the widened tolerance: the mean-centered signal's DC (0 Hz)
# bin has magnitude ~0 by construction (a mean-subtracted signal sums to
# ~0), so its phase (np.angle of a near-zero complex number) is
# fundamentally ill-conditioned -- cuFFT (GPU) and pocketfft/NumPy's FFT
# (CPU) are DIFFERENT numerical algorithms that do not guarantee
# bit-identical results even at identical (float64) precision, and near a
# magnitude-zero point that residual difference can round the DC bin's
# real/imaginary parts to opposite signs, flipping its reported phase by
# exactly pi (confirmed empirically: switching to float64 left the observed
# error unchanged at exactly pi -- not a precision issue). An independent
# audit found the tolerance that accommodated this (rtol=1e-2, atol=3.2,
# applied to the WHOLE mag+phase vector by featm.gpu.dispatch.
# _validate_against_cpu's single blanket np.allclose) is loose enough that a
# genuine ~1.0 magnitude-column corruption passes validation undetected
# (reproduced adversarially in tests/test_gpu_dispatch.py). Rather than
# build a component-aware (magnitude-vs-phase) validator whose own
# correctness would need auditing, and whose PRODUCTION GPU OUTPUT would
# still differ from CPU by up to pi on near-zero-magnitude bins, this
# feature is simply not GPU-accelerated -- CPU remains the sole, exact
# implementation. ``signal.frequency.fft`` (magnitude-only, no phase
# columns) is unaffected and stays GPU-accelerated.
@_reg('timeseries.spectral.fft','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('fft',),cache_key='timeseries.spectral_parts')
def sp_fft(x,n_coeffs=16,**kwargs):
    spec,_,_,_=spectral_parts(x);k=min(n_coeffs,spec.shape[1]);arr=np.column_stack([np.abs(spec[:,:k]),stable_phase(spec[:,:k])]);names=tuple([f'mag_{i:02d}' for i in range(k)]+[f'phase_{i:02d}' for i in range(k)]);return FeatureOutput(arr,names)

@_reg('timeseries.spectral.periodogram','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('periodogram',),cache_key='timeseries.spectral_parts',gpu_available=True,gpu_backend='cupy',gpu_min_rows=_GPU_SPECTRAL_MIN_ROWS)
def sp_periodogram(x,n_bins=16,**kwargs):
    _,_,p,_=spectral_parts(x);k=min(n_bins,p.shape[1]);return FeatureOutput(p[:,:k],tuple(f'power_{i:02d}' for i in range(k)))

@_reg('timeseries.spectral.psd','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('psd',),requires=('scipy',))
def sp_psd(x,n_bins=16,**kwargs):
    from scipy.signal import welch
    out=[]
    for r in as_2d_series(x):
        f,p=welch(np.nan_to_num(r),nperseg=min(256,len(r))); p=p[:n_bins];out.append(np.pad(p,(0,n_bins-len(p)),constant_values=np.nan))
    return FeatureOutput(np.asarray(out),tuple(f'welch_{i:02d}' for i in range(n_bins)))

@_reg('timeseries.spectral.bandpower','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('bandpower',),cache_key='timeseries.spectral_parts',gpu_available=True,gpu_backend='cupy',gpu_min_rows=_GPU_SPECTRAL_MIN_ROWS)
def sp_bandpower(x,bands=4,**kwargs):
    _,_,p,_=spectral_parts(x);chunks=np.array_split(np.arange(p.shape[1]),bands);arr=np.column_stack([p[:,c].sum(1) if len(c) else np.zeros(len(p)) for c in chunks]);return FeatureOutput(arr,tuple(f'band_{i}' for i in range(bands)))

# Audit-fix pass (v0.3.1): GPU acceleration REMOVED (was gpu_available=True
# with gpu_output_dtype='float64', gpu_numerical_tolerance=(0.3, 0.05)).
# Cepstrum's log(magnitude) -> inverse-FFT chain is a genuinely numerically
# sensitive (homomorphic) transform: at float32, the observed max relative
# error reached ~331x; float64 reduced it to ~24% relative / ~0.03 absolute
# -- a real improvement, but the residual is cuFFT-vs-pocketfft
# implementation-level rounding nonlinearly amplified through log() for
# low-power frequency bins (log's derivative is unbounded near zero), not a
# coding bug, and not fixable by precision alone. An independent audit
# concluded a 30%-relative-error tolerance is too loose to call this "GPU
# accelerates the same feature" without materially tighter per-coefficient
# evidence than a single aggregate pass counter provides -- rather than ship
# that, this feature is simply not GPU-accelerated; CPU remains the sole,
# exact implementation. Revisit if a normalization/implementation change is
# found that gives materially closer CPU<->GPU parity.
@_reg('timeseries.spectral.cepstrum','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('cepstrum',),cache_key='timeseries.spectral_parts')
def sp_cepstrum(x,n_coeffs=12,**kwargs):
    spec,mag,_,_=spectral_parts(x);cep=cepstrum_values(mag);k=min(n_coeffs,cep.shape[1]);return FeatureOutput(cep[:,:k],tuple(f'c_{i:02d}' for i in range(k)))

@_reg('timeseries.spectral.bispectrum','spectral',output_kind='vector',cost=Cost.HIGH,tags=('bispectrum','higher-order'))
def sp_bispec(x,n_freq=6,**kwargs):
    a=as_2d_series(x);outs=[]
    for r in a:
        X=np.fft.rfft(np.nan_to_num(r-np.nanmean(r)));k=min(n_freq,len(X)//2);vals=[]
        for i in range(1,k+1):
            vals.append(np.abs(X[i]*X[i]*np.conj(X[min(2*i,len(X)-1)]))/max(len(r),1))
        outs.append(np.pad(vals,(0,n_freq-len(vals)),constant_values=np.nan))
    return FeatureOutput(np.asarray(outs),tuple(f'b_{i:02d}' for i in range(1,n_freq+1)))

@_reg('timeseries.spectral.higher_order_spectra','spectral',output_kind='vector',cost=Cost.HIGH,tags=('higher-order',))
def sp_hos(x,**kwargs): return sp_bispec(x,n_freq=8)

# spectral_entropy may already be registered in basic.py.


# Complexity.
for _name,_fn in [
    ('approximate_entropy',approximate_entropy_1d),('sample_entropy',sample_entropy_1d),('permutation_entropy',permutation_entropy_1d),
    ('lempel_ziv',lempel_ziv_complexity_1d),('compressibility',compressed_ratio),
]:
    @_reg(f'timeseries.complexity.{_name}','complexity',cost=Cost.HIGH if 'entropy' in _name else Cost.MEDIUM,tags=('complexity',))
    def _complex(x,_fn=_fn,**kwargs): return _rows(x,_fn)

@_reg('timeseries.complexity.fuzzy_entropy','complexity',cost=Cost.HIGH,tags=('entropy',))
def fuzzy_entropy(x,m=2,r=None,n=2,**kwargs):
    def one(s):
        s=np.asarray(s,float);rr=0.2*np.nanstd(s) if r is None else r
        def phi(mm):
            w=np.array([s[i:i+mm] for i in range(len(s)-mm+1)]);vals=[]
            for i in range(len(w)):
                d=np.max(np.abs(w-w[i]),axis=1);d=np.delete(d,i);vals.extend(np.exp(-(d**n)/(max(rr,1e-12))))
            return np.mean(vals) if vals else np.nan
        a,b=phi(m),phi(m+1);return float(np.log(a/b)) if a>0 and b>0 else np.nan
    return _rows(x,one)

@_reg('timeseries.complexity.multiscale_entropy','complexity',output_kind='vector',cost=Cost.HIGH,tags=('entropy','multiscale'))
def multiscale_entropy(x,scales=(1,2,3,4,5),**kwargs):
    out=[]
    for r in as_2d_series(x):
        vals=[]
        for s in scales:
            n=len(r)//s
            coarse=r[:n*s].reshape(n,s).mean(1) if n else np.array([])
            vals.append(sample_entropy_1d(coarse) if len(coarse)>5 else np.nan)
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'scale_{s}' for s in scales))


# Fractals.
for _name,_fn in [('hurst',hurst_1d),('dfa',dfa_1d),('higuchi',higuchi_fd_1d),('katz',katz_fd_1d),('petrosian',petrosian_fd_1d)]:
    @_reg(f'timeseries.fractal.{_name}','fractal',cost=Cost.MEDIUM,tags=('fractal',))
    def _frac(x,_fn=_fn,**kwargs):return _rows(x,_fn)

@_reg('timeseries.fractal.multifractal','fractal',output_kind='vector',cost=Cost.HIGH,tags=('multifractal',))
def multifractal(x,qs=(-2,-1,0,1,2),**kwargs):
    out=[]
    for r in as_2d_series(x):
        d=np.abs(np.diff(r)); d=d[np.isfinite(d)&(d>0)]
        vals=[]
        for q in qs:
            if not len(d):vals.append(np.nan)
            elif q==0:vals.append(np.exp(np.mean(np.log(d))))
            else:vals.append(np.mean(d**q)**(1/q))
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'q_{q}' for q in qs))


# Trend and seasonality.
@_reg('timeseries.trend.linear','trend',output_kind='vector',cost=Cost.LOW,tags=('trend',))
def trend_linear(x,**kwargs):
    out=[]
    for r in as_2d_series(x):
        t=np.arange(len(r));mask=np.isfinite(r);p=np.polyfit(t[mask],r[mask],1) if mask.sum()>1 else [np.nan,np.nan];pred=np.polyval(p,t[mask]);ssr=np.sum((r[mask]-pred)**2);sst=np.sum((r[mask]-np.mean(r[mask]))**2);r2=1-ssr/sst if sst>0 else np.nan;out.append([p[0],p[1],r2])
    return FeatureOutput(np.asarray(out),('slope','intercept','r2'))

@_reg('timeseries.trend.polynomial','trend',output_kind='vector',cost=Cost.MEDIUM,tags=('trend','polynomial'))
def trend_poly(x,degree=2,**kwargs):
    out=[]
    for r in as_2d_series(x):
        t=np.linspace(-1,1,len(r));mask=np.isfinite(r);p=np.polyfit(t[mask],r[mask],degree) if mask.sum()>degree else np.full(degree+1,np.nan);out.append(p)
    return FeatureOutput(np.asarray(out),tuple(f'coef_{i}' for i in range(degree,-1,-1)))

@_reg('timeseries.trend.robust','trend',output_kind='vector',cost=Cost.MEDIUM,tags=('theil-sen','robust'))
def trend_robust(x,**kwargs):
    from scipy.stats import theilslopes
    out=[]
    for r in as_2d_series(x):
        t=np.arange(len(r));mask=np.isfinite(r)
        try:slope,inter,lo,hi=theilslopes(r[mask],t[mask]);out.append([slope,inter,lo,hi])
        except Exception:out.append([np.nan]*4)
    return FeatureOutput(np.asarray(out),('slope','intercept','slope_low','slope_high'))

@_reg('timeseries.trend.piecewise','trend',output_kind='vector',cost=Cost.LOW,tags=('piecewise',))
def trend_piecewise(x,segments=3,**kwargs):
    out=[]
    for r in as_2d_series(x):
        parts=np.array_split(r,segments);vals=[]
        for p in parts:
            t=np.arange(len(p));m=np.isfinite(p);vals.append(np.polyfit(t[m],p[m],1)[0] if m.sum()>1 else np.nan)
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'slope_segment_{i}' for i in range(segments)))

@_reg('timeseries.seasonality.periodicity','seasonality',output_kind='vector',cost=Cost.MEDIUM,tags=('periodicity',))
def season_periodicity(x,**kwargs):
    _,_,p,f=spectral_parts(x);p[:,0]=0;idx=np.argmax(p,1);freq=f[idx];period=np.divide(1.0,freq,out=np.full_like(freq,np.nan),where=freq>0);strength=np.divide(p[np.arange(len(p)),idx],p.sum(1),out=np.zeros(len(p)),where=p.sum(1)>0);return FeatureOutput(np.column_stack([period,strength]),('dominant_period','periodic_strength'))

@_reg('timeseries.seasonality.harmonic','seasonality',output_kind='vector',cost=Cost.MEDIUM,tags=('harmonic',))
def season_harmonic(x,n_harmonics=4,**kwargs):
    _,mag,_,_=spectral_parts(x);idx=np.argsort(mag[:,1:],axis=1)[:,-n_harmonics:]+1;vals=np.take_along_axis(mag,idx,axis=1);return FeatureOutput(vals,tuple(f'harmonic_amp_{i}' for i in range(n_harmonics)))

@_reg('timeseries.seasonality.seasonal_strength','seasonality',cost=Cost.MEDIUM,tags=('seasonality',))
def season_strength(x,**kwargs): return season_periodicity(x).values[:,1]


# Stationarity.
@_reg('timeseries.stationarity.adf','stationarity',output_kind='vector',cost=Cost.HIGH,tags=('adf',),requires=('statsmodels',))
def station_adf(x,**kwargs):
    from statsmodels.tsa.stattools import adfuller
    out=[]
    for r in as_2d_series(x):
        rr=r[np.isfinite(r)]
        try:res=adfuller(rr,autolag='AIC');out.append([res[0],res[1],res[2]])
        except Exception:out.append([np.nan]*3)
    return FeatureOutput(np.asarray(out),('statistic','pvalue','used_lag'))

@_reg('timeseries.stationarity.kpss','stationarity',output_kind='vector',cost=Cost.HIGH,tags=('kpss',),requires=('statsmodels',))
def station_kpss(x,**kwargs):
    from statsmodels.tsa.stattools import kpss
    out=[]
    for r in as_2d_series(x):
        rr=r[np.isfinite(r)]
        try:res=kpss(rr,regression='c',nlags='auto');out.append([res[0],res[1],res[2]])
        except Exception:out.append([np.nan]*3)
    return FeatureOutput(np.asarray(out),('statistic','pvalue','used_lags'))

@_reg('timeseries.stationarity.variance_ratio','stationarity',cost=Cost.LOW,tags=('variance-ratio',))
def station_vr(x,lag=2,**kwargs):
    def one(r):
        d=np.diff(r);dl=r[lag:]-r[:-lag];v1=np.nanvar(d);return np.nanvar(dl)/(lag*v1) if v1>0 else np.nan
    return _rows(x,one)


# Change points.
@_reg('timeseries.change_point.mean_shift','change_point',cost=Cost.LOW,tags=('change-point',))
def cp_mean(x,**kwargs):
    def one(r):
        m=len(r)//2;return abs(np.nanmean(r[:m])-np.nanmean(r[m:]))
    return _rows(x,one)

@_reg('timeseries.change_point.variance_shift','change_point',cost=Cost.LOW,tags=('change-point',))
def cp_var(x,**kwargs):
    def one(r):m=len(r)//2;return abs(np.nanvar(r[:m])-np.nanvar(r[m:]))
    return _rows(x,one)

@_reg('timeseries.change_point.regime_count','change_point',cost=Cost.MEDIUM,tags=('change-point',))
def cp_regimes(x,threshold=2.5,**kwargs):
    def one(r):
        w=max(4,len(r)//20); sm=np.convolve(np.nan_to_num(r,nan=np.nanmean(r)),np.ones(w)/w,mode='valid');d=np.abs(np.diff(sm));med=np.median(d);mad=np.median(np.abs(d-med));return float(np.sum(d>med+threshold*1.4826*mad)) if mad>0 else 0.0
    return _rows(x,one)


# Recurrence.
for _name,_key in [('recurrence_rate','rate'),('determinism','determinism'),('laminarity','laminarity'),('trapping_time','trapping')]:
    @_reg(f'timeseries.recurrence.{_name}','recurrence',cost=Cost.HIGH,tags=('recurrence','rqa'))
    def _rqa(x,_key=_key,**kwargs):return _rows(x,lambda r:rqa_stats_1d(r)[_key])

@_reg('timeseries.recurrence.recurrence_network','recurrence',output_kind='vector',cost=Cost.HIGH,tags=('recurrence','graph'))
def recurrence_network(x,**kwargs):
    out=[]
    for r in as_2d_series(x):
        R=recurrence_matrix_1d(r);np.fill_diagonal(R,False) if R.size else None
        if not R.size:out.append([np.nan]*3);continue
        deg=R.sum(1);density=R.mean();mean_deg=np.mean(deg);tri=np.trace(R.astype(int)@R.astype(int)@R.astype(int))/6;out.append([density,mean_deg,tri])
    return FeatureOutput(np.asarray(out),('density','mean_degree','triangle_count'))


# Dynamical systems.
def _delay(r,max_lag=50):
    for l in range(1,min(max_lag,len(r)//3)+1):
        c=autocorr_row(r,l)
        if np.isfinite(c) and c<1/math.e:return l
    return 1

@_reg('timeseries.dynamical.time_delay','dynamical',cost=Cost.MEDIUM,tags=('phase-space',))
def dyn_delay(x,**kwargs):return _rows(x,_delay)

@_reg('timeseries.dynamical.phase_space','dynamical',output_kind='vector',cost=Cost.MEDIUM,tags=('phase-space',))
def dyn_phase(x,dim=3,**kwargs):
    out=[]
    for r in as_2d_series(x):
        tau=_delay(r);n=len(r)-(dim-1)*tau
        if n<3:out.append([np.nan]*(dim+2));continue
        emb=np.column_stack([r[i*tau:i*tau+n] for i in range(dim)]);cov=np.cov(emb,rowvar=False);ev=np.linalg.eigvalsh(cov);out.append([tau,*ev.tolist(),np.linalg.det(cov)])
    return FeatureOutput(np.asarray(out),('delay',)+tuple(f'cov_eig_{i}' for i in range(dim))+('cov_volume',))

@_reg('timeseries.dynamical.embedding_dimension','dynamical',cost=Cost.HIGH,tags=('phase-space','false-nearest'))
def dyn_embdim(x,max_dim=6,**kwargs):
    def one(r):
        tau=_delay(r)
        from scipy.spatial import cKDTree
        for d in range(1,max_dim):
            n=len(r)-d*tau
            if n<20:return float(d)
            e=np.column_stack([r[i*tau:i*tau+n] for i in range(d)]);e2=np.column_stack([r[i*tau:i*tau+n] for i in range(d+1)])
            tree=cKDTree(e);dist,idx=tree.query(e,k=2);nn=idx[:,1];base=np.maximum(dist[:,1],1e-12);extra=np.abs(e2[:,-1]-e2[nn,-1]);fnn=np.mean(extra/base>10)
            if fnn<0.05:return float(d)
        return float(max_dim)
    return _rows(x,one)

@_reg('timeseries.dynamical.lyapunov','dynamical',cost=Cost.HIGH,tags=('chaos','lyapunov'))
def dyn_lyap(x,**kwargs):
    def one(r):
        tau=_delay(r);d=3;n=len(r)-(d-1)*tau
        if n<30:return np.nan
        emb=np.column_stack([r[i*tau:i*tau+n] for i in range(d)])
        from scipy.spatial import cKDTree
        tree=cKDTree(emb);k=min(10,len(emb));dist,idx=tree.query(emb,k=k);pairs=[]
        for i in range(len(emb)):
            cand=[j for j in idx[i,1:] if abs(int(j)-i)>tau*2]
            if cand:pairs.append((i,int(cand[0])))
        vals=[]
        for t in range(1,min(12,n//5)):
            ds=[np.linalg.norm(emb[i+t]-emb[j+t]) for i,j in pairs if i+t<n and j+t<n]
            ds=np.array(ds);ds=ds[ds>0]
            vals.append(np.mean(np.log(ds)) if len(ds) else np.nan)
        v=np.array(vals);m=np.isfinite(v)
        return float(np.polyfit(np.arange(len(v))[m],v[m],1)[0]) if m.sum()>2 else np.nan
    return _rows(x,one)

@_reg('timeseries.dynamical.correlation_dimension','dynamical',cost=Cost.HIGH,tags=('chaos','dimension'))
def dyn_corrdim(x,**kwargs):
    def one(r):
        tau=_delay(r);d=3;n=len(r)-(d-1)*tau
        if n<20:return np.nan
        emb=np.column_stack([r[i*tau:i*tau+n] for i in range(d)])
        if len(emb)>400:emb=emb[np.linspace(0,len(emb)-1,400).astype(int)]
        from scipy.spatial.distance import pdist
        D=pdist(emb);D=D[D>0]
        if len(D)<10:return np.nan
        radii=np.quantile(D,[.1,.2,.3,.4,.5]);C=np.array([(D<e).mean() for e in radii]);m=(C>0)&(radii>0)
        return float(np.polyfit(np.log(radii[m]),np.log(C[m]),1)[0]) if m.sum()>2 else np.nan
    return _rows(x,one)

@_reg('timeseries.dynamical.attractor_geometry','dynamical',output_kind='vector',cost=Cost.HIGH,tags=('attractor',))
def dyn_attractor(x,**kwargs):
    out=[]
    for r in as_2d_series(x):
        tau=_delay(r);n=len(r)-2*tau
        if n<5:out.append([np.nan]*4);continue
        e=np.column_stack([r[:n],r[tau:tau+n],r[2*tau:2*tau+n]]);cov=np.cov(e,rowvar=False);ev=np.sort(np.linalg.eigvalsh(cov))[::-1];out.append([*ev,np.sqrt(max(np.linalg.det(cov),0))])
    return FeatureOutput(np.asarray(out),('eig1','eig2','eig3','volume'))

@_reg('timeseries.dynamical.poincare','dynamical',output_kind='vector',cost=Cost.MEDIUM,tags=('poincare',))
def dyn_poincare(x,**kwargs):
    out=[]
    for r in as_2d_series(x):
        med=np.nanmedian(r);cross=np.where((r[:-1]-med)*(r[1:]-med)<0)[0];intervals=np.diff(cross);out.append([len(cross),np.mean(intervals) if len(intervals) else np.nan,np.std(intervals) if len(intervals) else np.nan])
    return FeatureOutput(np.asarray(out),('crossing_count','interval_mean','interval_std'))


# Cross-series. All expect a structured (2, t) per-sample pair (see
# _paired() above) that a row-oriented Source cannot represent -- Finding 4,
# fourth audit pass.
@_reg('timeseries.cross_series.cross_correlation','cross_series',output_kind='vector',cost=Cost.MEDIUM,tags=('cross-correlation','structured'),structured=True)
def cross_corr(x,max_lag=5,**kwargs):
    a,b=_paired(x);out=[]
    for r,s in zip(a,b):
        vals=[]
        for lag in range(-max_lag,max_lag+1):
            rr=r[max(0,lag):len(r)+min(0,lag)];ss=s[max(0,-lag):len(s)+min(0,-lag)];vals.append(np.corrcoef(rr,ss)[0,1] if len(rr)>2 and np.std(rr)>0 and np.std(ss)>0 else np.nan)
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'lag_{l:+d}' for l in range(-max_lag,max_lag+1)))

@_reg('timeseries.cross_series.lead_lag','cross_series',output_kind='vector',cost=Cost.MEDIUM,tags=('lead-lag','structured'),structured=True)
def cross_leadlag(x,max_lag=12,**kwargs):
    cc=cross_corr(x,max_lag=max_lag).values;idx=np.nanargmax(np.abs(cc),axis=1);lags=np.arange(-max_lag,max_lag+1);return FeatureOutput(np.column_stack([lags[idx],cc[np.arange(len(cc)),idx]]),('best_lag','correlation'))

@_reg('timeseries.cross_series.coherence','cross_series',output_kind='vector',cost=Cost.MEDIUM,tags=('coherence','structured'),structured=True)
def cross_coherence(x,bands=4,**kwargs):
    from scipy.signal import coherence
    a,b=_paired(x);out=[]
    for r,s in zip(a,b):
        f,c=coherence(np.nan_to_num(r),np.nan_to_num(s),nperseg=min(128,len(r)));parts=np.array_split(c,bands);out.append([np.mean(p) if len(p) else np.nan for p in parts])
    return FeatureOutput(np.asarray(out),tuple(f'band_{i}' for i in range(bands)))

@_reg('timeseries.cross_series.dynamic_time_warping','cross_series',cost=Cost.HIGH,tags=('dtw','structured'),structured=True)
def cross_dtw(x,**kwargs):
    a,b=_paired(x);out=[]
    for r,s in zip(a,b):
        n,m=len(r),len(s);prev=np.full(m+1,np.inf);prev[0]=0
        for i in range(1,n+1):
            cur=np.full(m+1,np.inf)
            for j in range(1,m+1):cur[j]=abs(r[i-1]-s[j-1])+min(cur[j-1],prev[j],prev[j-1])
            prev=cur
        out.append(prev[m]/max(n+m,1))
    return np.asarray(out)

@_reg('timeseries.cross_series.phase_synchrony','cross_series',cost=Cost.MEDIUM,tags=('phase','synchrony','structured'),structured=True)
def cross_phase(x,**kwargs):
    from scipy.signal import hilbert
    a,b=_paired(x);out=[]
    for r,s in zip(a,b):
        pa=np.angle(hilbert(np.nan_to_num(r)));pb=np.angle(hilbert(np.nan_to_num(s)));out.append(abs(np.mean(np.exp(1j*(pa-pb)))))
    return np.asarray(out)

@_reg('timeseries.cross_series.transfer_entropy','cross_series',cost=Cost.HIGH,tags=('information','transfer-entropy','structured'),structured=True)
def cross_te(x,bins=8,**kwargs):
    from .._utils import conditional_mi_discrete
    a,b=_paired(x);return np.asarray([conditional_mi_discrete(s[1:],r[:-1],s[:-1],bins=bins) for r,s in zip(a,b)])

@_reg('timeseries.cross_series.granger','cross_series',output_kind='vector',cost=Cost.HIGH,tags=('granger','structured'),requires=('statsmodels',),structured=True)
def cross_granger(x,maxlag=2,**kwargs):
    from statsmodels.tsa.stattools import grangercausalitytests
    a,b=_paired(x);out=[]
    for r,s in zip(a,b):
        data=np.column_stack([s,r]);
        try:
            res=grangercausalitytests(data,maxlag=maxlag,verbose=False);p=[res[l][0]['ssr_ftest'][1] for l in range(1,maxlag+1)];out.append(p)
        except Exception:out.append([np.nan]*maxlag)
    return FeatureOutput(np.asarray(out),tuple(f'p_lag_{i}' for i in range(1,maxlag+1)))


# Wavelets using built-in Haar and a Morlet FFT-convolution implementation.
def _haar_levels(r,levels=4):
    cur=np.asarray(r,float);vals=[]
    for _ in range(levels):
        n=(len(cur)//2)*2
        if n<2:vals.append((np.array([]),cur));break
        a=(cur[:n:2]+cur[1:n:2])/np.sqrt(2);d=(cur[:n:2]-cur[1:n:2])/np.sqrt(2);vals.append((d,a));cur=a
    return vals

@_reg('timeseries.wavelet.dwt','wavelet',output_kind='vector',cost=Cost.MEDIUM,tags=('wavelet','haar'))
def wave_dwt(x,levels=4,**kwargs):
    out=[]
    for r in as_2d_series(x):
        lev=_haar_levels(np.nan_to_num(r),levels);out.append([np.mean(d*d) if len(d) else np.nan for d,a in lev]+[np.mean(lev[-1][1]**2) if lev else np.nan])
    return FeatureOutput(np.asarray(out),tuple([f'detail_energy_{i+1}' for i in range(levels)]+['approx_energy']))

@_reg('timeseries.wavelet.cwt','wavelet',output_kind='vector',cost=Cost.HIGH,tags=('wavelet','morlet'))
def wave_cwt(x,scales=(2,4,8,16),**kwargs):
    from scipy.signal import fftconvolve
    out=[]
    for r in as_2d_series(x):
        vals=[]
        for s in scales:
            t=np.arange(-4*s,4*s+1);w=np.exp(1j*5*t/s)*np.exp(-(t*t)/(2*s*s));w-=w.mean();c=fftconvolve(np.nan_to_num(r),w,mode='same');vals.append(np.mean(np.abs(c)**2))
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'scale_{s}' for s in scales))

@_reg('timeseries.wavelet.wavelet_entropy','wavelet',cost=Cost.MEDIUM,tags=('wavelet','entropy'))
def wave_entropy(x,levels=4,**kwargs):
    e=wave_dwt(x,levels=levels).values;return shannon_entropy_prob(e,axis=1,normalize=True)

@_reg('timeseries.wavelet.wavelet_packet','wavelet',output_kind='vector',cost=Cost.HIGH,tags=('wavelet-packet',))
def wave_packet(x,depth=3,**kwargs):
    out=[]
    for r in as_2d_series(x):
        nodes=[np.nan_to_num(r)]
        for _ in range(depth):
            nxt=[]
            for cur in nodes:
                n=(len(cur)//2)*2
                if n<2: nxt.extend([cur,cur]);continue
                nxt.extend([(cur[:n:2]+cur[1:n:2])/np.sqrt(2),(cur[:n:2]-cur[1:n:2])/np.sqrt(2)])
            nodes=nxt
        out.append([np.mean(v*v) if len(v) else np.nan for v in nodes])
    return FeatureOutput(np.asarray(out),tuple(f'node_{i:02d}' for i in range(2**depth)))

@_reg('timeseries.wavelet.scattering','wavelet',output_kind='vector',cost=Cost.HIGH,tags=('scattering','wavelet'))
def wave_scattering(x,**kwargs):
    c=wave_cwt(x).values;return FeatureOutput(np.column_stack([c,np.sqrt(np.maximum(c,0))]),tuple([f'energy_{i}' for i in range(c.shape[1])]+[f'modulus_{i}' for i in range(c.shape[1])]))


# H0 persistent topology from an MST of phase-space points. No optional TDA
# package is required; higher-dimensional persistence can later use plugin packs.
def _h0_persistence(r,dim=2):
    tau=_delay(r);n=len(r)-(dim-1)*tau
    if n<3:return np.array([])
    pts=np.column_stack([r[i*tau:i*tau+n] for i in range(dim)]);pts=pts[np.all(np.isfinite(pts),axis=1)]
    if len(pts)>300:pts=pts[np.linspace(0,len(pts)-1,300).astype(int)]
    if len(pts)<2:return np.array([])
    from scipy.spatial.distance import pdist,squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    D=squareform(pdist(pts));mst=minimum_spanning_tree(D).toarray();deaths=mst[mst>0];return np.sort(deaths)

@_reg('timeseries.topology.persistence','topology',output_kind='vector',cost=Cost.HIGH,tags=('persistent-homology','H0'))
def topo_persistence(x,n=8,**kwargs):
    out=[]
    for r in as_2d_series(x):
        d=_h0_persistence(r);v=np.sort(d)[-n:] if len(d) else np.array([]);out.append(np.pad(v,(n-len(v),0),constant_values=0))
    return FeatureOutput(np.asarray(out),tuple(f'h0_death_{i:02d}' for i in range(n)))

@_reg('timeseries.topology.betti_curves','topology',output_kind='vector',cost=Cost.HIGH,tags=('betti','H0'))
def topo_betti(x,n_thresholds=8,**kwargs):
    out=[]
    for r in as_2d_series(x):
        d=_h0_persistence(r)
        if not len(d):out.append([np.nan]*n_thresholds);continue
        th=np.linspace(0,d.max(),n_thresholds);out.append([1+np.sum(d>t) for t in th])
    return FeatureOutput(np.asarray(out),tuple(f'beta0_{i:02d}' for i in range(n_thresholds)))

@_reg('timeseries.topology.persistence_landscape','topology',output_kind='vector',cost=Cost.HIGH,tags=('persistence-landscape','H0'))
def topo_landscape(x,n_grid=8,**kwargs):
    out=[]
    for r in as_2d_series(x):
        d=_h0_persistence(r)
        if not len(d):out.append([0]*n_grid);continue
        grid=np.linspace(0,d.max(),n_grid);# H0 bars [0,d]
        vals=[np.max(np.minimum(g,np.maximum(d-g,0))) for g in grid];out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'landscape_{i:02d}' for i in range(n_grid)))

@_reg('timeseries.topology.persistence_image','topology',output_kind='vector',cost=Cost.HIGH,tags=('persistence-image','H0'))
def topo_pimage(x,bins=8,**kwargs):
    out=[]
    for r in as_2d_series(x):
        d=_h0_persistence(r)
        if not len(d):out.append([0]*bins);continue
        h,_=np.histogram(d,bins=bins,range=(0,max(d.max(),1e-12)),weights=d);out.append(h)
    return FeatureOutput(np.asarray(out),tuple(f'bin_{i:02d}' for i in range(bins)))

@_reg('timeseries.trend.linear_slope','trend',cost=Cost.LOW,tags=('trend','slope'))
def trend_linear_slope(x,**kwargs):
    return trend_linear(x).values[:,0]
