from __future__ import annotations

from .._utils import stable_phase, cepstrum_values

import numpy as np

from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import (
    as_2d_series, content_seed, hurst_1d, sample_entropy_1d, spectral_entropy_rows, spectral_parts,
)


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=(),**extra):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        return register_feature(name=name,family=family,modality='signal',input_types=('signal',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=requires,description=name,**extra)(fn)
    return deco

def _rows(x,fn):return np.asarray([fn(r) for r in as_2d_series(x)],float)

# Step 3: all 5 time_domain reductions and the 3 spectral_parts-based
# features below have real GPU kernels (featm.gpu.kernels.reductions /
# .spectral) -- same formulas as timeseries.statistical.*/spectral.* (see
# those kernels' docstrings for the exact CPU feature each mirrors).
#
# Auto-threshold fix pass (v0.3.2): line_length (diff + abs + sum, a
# different computational shape than a plain single-pass reduction) was
# measured directly on real Kaggle P100 hardware and showed a large,
# consistent GPU win starting at the smallest calibrated size (1,000 rows x
# 64 = 64,000 elements, 12.9x speedup; next point 5,000 rows, 44.0x) -- a
# wide safety margin above the required 1.10x bar, so it gets its own,
# lower, measured threshold rather than the shared cheap-reduction one. The
# other 4 (rms/peak_to_peak/crest_factor/impulse_factor) are simple
# single-pass reductions structurally like timeseries.statistical.mean (not
# individually measured this pass) and use that same, more conservative,
# measured threshold as their safe default. See
# STEP3_GPU_AUTO_THRESHOLD_FIX_HANDOFF.md for the full calibration table.
from ..timeseries.basic import _GPU_MIN_ELEMENTS_CHEAP_REDUCTION as _GPU_MIN_ELEMENTS_CHEAP
_GPU_MIN_ELEMENTS_LINE_LENGTH = 64_000
for _name,_fn in [
    ('rms',lambda r:np.sqrt(np.nanmean(r*r))),('peak_to_peak',lambda r:np.nanmax(r)-np.nanmin(r)),
    ('line_length',lambda r:np.nansum(np.abs(np.diff(r)))),
    ('crest_factor',lambda r:np.nanmax(np.abs(r))/max(np.sqrt(np.nanmean(r*r)),1e-15)),
    ('impulse_factor',lambda r:np.nanmax(np.abs(r))/max(np.nanmean(np.abs(r)),1e-15)),
]:
    _gpu_min_elements = _GPU_MIN_ELEMENTS_LINE_LENGTH if _name == 'line_length' else _GPU_MIN_ELEMENTS_CHEAP
    @_reg(f'signal.time_domain.{_name}','time_domain',cost=Cost.TINY,tags=('time-domain',),gpu_available=True,gpu_backend='cupy',gpu_min_rows=8_000,gpu_min_elements=_gpu_min_elements)
    def _td(x,_fn=_fn,**kwargs):return _rows(x,_fn)

# gpu_output_dtype='float64': found by this milestone's real Kaggle
# validation (Tesla P100) -- magnitude-only, so no phase-instability
# concern (see timeseries.spectral.fft's comment for that separate issue),
# but float32 FFT magnitude was landing just outside the default tight
# atol=1e-6 (observed max_abs_error ~1e-5); float64 removes the gap cleanly
# rather than loosening the declared tolerance to paper over it.
@_reg('signal.frequency.fft','frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('fft',),gpu_available=True,gpu_backend='cupy',gpu_min_rows=1_500,gpu_output_dtype='float64')
def sig_fft(x,n_coeffs=16,**kwargs):
    s,_,_,_=spectral_parts(x);k=min(n_coeffs,s.shape[1]);return FeatureOutput(np.abs(s[:,:k]),tuple(f'mag_{i:02d}' for i in range(k)))

@_reg('signal.frequency.psd','frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('psd',),gpu_available=True,gpu_backend='cupy',gpu_min_rows=1_500)
def sig_psd(x,n_bins=16,**kwargs):
    _,_,p,_=spectral_parts(x);k=min(n_bins,p.shape[1]);return FeatureOutput(p[:,:k],tuple(f'power_{i:02d}' for i in range(k)))

@_reg('signal.entropy.spectral_entropy','entropy',cost=Cost.MEDIUM,tags=('entropy',),gpu_available=True,gpu_backend='cupy',gpu_min_rows=1_500)
def sig_sent(x,**kwargs):return spectral_entropy_rows(x)

@_reg('signal.entropy.sample_entropy','entropy',cost=Cost.HIGH,tags=('entropy',))
def sig_sampen(x,**kwargs):return _rows(x,sample_entropy_1d)

@_reg('signal.fractal.hurst','fractal',cost=Cost.MEDIUM,tags=('fractal',))
def sig_hurst(x,**kwargs):return _rows(x,hurst_1d)

@_reg('signal.envelope.hilbert','envelope',output_kind='vector',cost=Cost.MEDIUM,tags=('hilbert',))
def sig_hilbert(x,**kwargs):
    from scipy.signal import hilbert
    out=[]
    for r in as_2d_series(x):
        env=np.abs(hilbert(np.nan_to_num(r)));out.append([np.mean(env),np.std(env),np.max(env)])
    return FeatureOutput(np.asarray(out),('mean','std','max'))

@_reg('signal.phase.instantaneous_phase','phase',output_kind='vector',cost=Cost.MEDIUM,tags=('phase',))
def sig_phase(x,**kwargs):
    from scipy.signal import hilbert
    out=[]
    for r in as_2d_series(x):
        ph=np.unwrap(np.angle(hilbert(np.nan_to_num(r))));d=np.diff(ph);out.append([np.mean(d),np.std(d),np.mean(np.abs(d))])
    return FeatureOutput(np.asarray(out),('phase_velocity_mean','phase_velocity_std','phase_velocity_abs_mean'))

@_reg('signal.oscillation.frequency','oscillation',output_kind='vector',cost=Cost.MEDIUM,tags=('frequency',))
def sig_freq(x,**kwargs):
    _,_,p,f=spectral_parts(x);p[:,0]=0;idx=np.argmax(p,1);return FeatureOutput(np.column_stack([f[idx],p[np.arange(len(p)),idx]/np.maximum(p.sum(1),1e-15)]),('dominant_frequency','strength'))

@_reg('signal.peaks.peak_statistics','peaks',output_kind='vector',cost=Cost.MEDIUM,tags=('peaks',))
def sig_peaks(x,**kwargs):
    from scipy.signal import find_peaks
    out=[]
    for r in as_2d_series(x):
        p,prop=find_peaks(np.nan_to_num(r),prominence=max(np.nanstd(r)*.5,1e-12));prom=prop.get('prominences',[]);out.append([len(p),np.mean(prom) if len(prom) else 0,np.max(prom) if len(prom) else 0])
    return FeatureOutput(np.asarray(out),('count','prominence_mean','prominence_max'))

@_reg('signal.cepstral.cepstrum','cepstral',output_kind='vector',cost=Cost.MEDIUM,tags=('cepstrum',))
def sig_cep(x,n_coeffs=12,**kwargs):
    _,mag,_,_=spectral_parts(x);cep=cepstrum_values(mag);k=min(n_coeffs,cep.shape[1]);return FeatureOutput(cep[:,:k],tuple(f'c_{i:02d}' for i in range(k)))

@_reg('signal.time_frequency.stft','time_frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('stft',))
def sig_stft(x,bins=16,**kwargs):
    from scipy.signal import stft
    out=[]
    for r in as_2d_series(x):
        f,t,Z=stft(np.nan_to_num(r),nperseg=min(128,len(r)));v=np.mean(np.abs(Z)**2,axis=1);idx=np.linspace(0,max(len(v)-1,0),bins).astype(int);out.append(v[idx] if len(v) else np.zeros(bins))
    return FeatureOutput(np.asarray(out),tuple(f'band_{i:02d}' for i in range(bins)))

# Haar DWT and Morlet CWT summaries.
@_reg('signal.wavelet.dwt','wavelet',output_kind='vector',cost=Cost.MEDIUM,tags=('haar','wavelet'))
def sig_dwt(x,levels=4,**kwargs):
    out=[]
    for r in as_2d_series(x):
        cur=np.nan_to_num(r);v=[]
        for _ in range(levels):
            n=len(cur)//2*2
            if n<2:v.append(np.nan);continue
            a=(cur[:n:2]+cur[1:n:2])/np.sqrt(2);d=(cur[:n:2]-cur[1:n:2])/np.sqrt(2);v.append(np.mean(d*d));cur=a
        v.append(np.mean(cur*cur));out.append(v)
    return FeatureOutput(np.asarray(out),tuple([f'detail_{i+1}' for i in range(levels)]+['approx']))

@_reg('signal.wavelet.cwt','wavelet',output_kind='vector',cost=Cost.HIGH,tags=('morlet','wavelet'))
def sig_cwt(x,scales=(2,4,8,16),**kwargs):
    from scipy.signal import fftconvolve
    out=[]
    for r in as_2d_series(x):
        v=[]
        for s in scales:
            t=np.arange(-4*s,4*s+1);w=np.exp(1j*5*t/s)*np.exp(-(t*t)/(2*s*s));w-=w.mean();c=fftconvolve(np.nan_to_num(r),w,mode='same');v.append(np.mean(np.abs(c)**2))
        out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'scale_{s}' for s in scales))

@_reg('signal.higher_order_spectra.bispectrum','higher_order_spectra',output_kind='vector',cost=Cost.HIGH,tags=('bispectrum',))
def sig_bispectrum(x,n_freq=6,**kwargs):
    out=[]
    for r in as_2d_series(x):
        X=np.fft.rfft(np.nan_to_num(r-np.nanmean(r)));k=min(n_freq,len(X)//2);v=[np.abs(X[i]*X[i]*np.conj(X[min(2*i,len(X)-1)]))/max(len(r),1) for i in range(1,k+1)];out.append(np.pad(v,(0,n_freq-len(v)),constant_values=np.nan))
    return FeatureOutput(np.asarray(out),tuple(f'b_{i:02d}' for i in range(1,n_freq+1)))

@_reg('signal.modulation.amplitude_modulation','modulation',output_kind='vector',cost=Cost.MEDIUM,tags=('modulation',))
def sig_am(x,**kwargs):
    from scipy.signal import hilbert
    out=[]
    for r in as_2d_series(x):
        env=np.abs(hilbert(np.nan_to_num(r)));F=np.abs(np.fft.rfft(env-env.mean()));freq=np.fft.rfftfreq(len(env));i=np.argmax(F[1:])+1 if len(F)>1 else 0;out.append([freq[i],F[i]/max(F.sum(),1e-15)])
    return FeatureOutput(np.asarray(out),('modulation_frequency','strength'))

@_reg('signal.synchronization.phase_locking','synchronization',cost=Cost.MEDIUM,tags=('phase-locking','structured'),batch_semantics=BatchSemantics.STRUCTURED_SAMPLE)
def sig_plv(x,**kwargs):
    # Expects a structured (2, t) per-sample pair (raises ValueError below if
    # not) -- discovered by the fourth audit pass's automated
    # batch-equivalence harness; the same STRUCTURED_SAMPLE class as
    # timeseries.cross_series.*/information.dependence.*/multimodal.*.
    from scipy.signal import hilbert
    z=np.asarray(x,float)
    if z.ndim==2 and z.shape[0]==2:z=z[None,:,:]
    if z.ndim!=3 or z.shape[1]!=2:raise ValueError('phase locking expects shape (n,2,t)')
    out=[]
    for p in z:
        a=np.angle(hilbert(p[0]));b=np.angle(hilbert(p[1]));out.append(abs(np.mean(np.exp(1j*(a-b)))))
    return np.asarray(out)

@_reg('signal.nonlinear.lyapunov','nonlinear',cost=Cost.HIGH,tags=('lyapunov',))
def sig_lyap(x,**kwargs):
    # Reuse the timeseries implementation lazily to avoid circular import during registration.
    from ..timeseries.full import dyn_lyap
    return dyn_lyap(x)

# Decompositions.
@_reg('signal.decomposition.ssa','decomposition',output_kind='vector',cost=Cost.HIGH,tags=('ssa',))
def sig_ssa(x,components=8,window=None,**kwargs):
    out=[]
    for r in as_2d_series(x):
        L=window or min(max(4,len(r)//3),64);K=len(r)-L+1
        if K<=1:out.append([np.nan]*components);continue
        H=np.column_stack([r[i:i+L] for i in range(K)]);s=np.linalg.svd(np.nan_to_num(H),compute_uv=False);e=s*s;e=e/e.sum() if e.sum() else e;out.append(np.pad(e[:components],(0,max(0,components-len(e))),constant_values=0))
    return FeatureOutput(np.asarray(out),tuple(f'component_energy_{i:02d}' for i in range(components)))

@_reg(
    'signal.decomposition.ica','decomposition',output_kind='vector',cost=Cost.HIGH,tags=('ica','full-materialization'),
    requires=('scikit-learn',),requires_full_materialization=True,
)
def sig_ica(x,n_components=4,**kwargs):
    # FastICA.fit_transform(a) treats every ROW of the batch as one jointly-
    # analyzed observation -- the estimated unmixing matrix (and therefore
    # every row's independent-component values) depends on which OTHER rows
    # happen to be in the same batch, not just that row's own signal. Not
    # row-independent at all -- discovered by the fourth audit pass's
    # automated batch-equivalence harness. Marked requires_full_materialization
    # (same treatment as text.semantics.topic's NMF, Finding 1E): the whole
    # source is read as one grown batch (or the run is rejected if it cannot
    # fit the declared memory_budget), rather than silently estimating a
    # different, batch-local decomposition. Not integrated with the public
    # featm.fit()/.transform() train/test API in this pass -- FastICA.transform()
    # on frozen components would make that possible as a follow-up, matching
    # text.semantics.topic's stated limitation.
    from sklearn.decomposition import FastICA
    a=as_2d_series(x);n=min(n_components,a.shape[0],a.shape[1]);model=FastICA(n_components=n,random_state=0,whiten='unit-variance');Z=model.fit_transform(np.nan_to_num(a));return FeatureOutput(Z,tuple(f'ic_{i:02d}' for i in range(n)))

@_reg('signal.decomposition.emd','decomposition',output_kind='vector',cost=Cost.EXTREME,tags=('emd',),requires=('PyEMD',))
def sig_emd(x,n_imfs=6,**kwargs):
    try:from PyEMD import EMD
    except ImportError as exc:raise ImportError("Install EMD-signal (PyEMD) for signal.decomposition.emd") from exc
    out=[]
    for r in as_2d_series(x):
        imfs=EMD().emd(r,max_imf=n_imfs);e=np.array([np.mean(v*v) for v in imfs[:n_imfs]]);out.append(np.pad(e,(0,n_imfs-len(e)),constant_values=0))
    return FeatureOutput(np.asarray(out),tuple(f'imf_energy_{i:02d}' for i in range(n_imfs)))

@_reg('signal.decomposition.eemd','decomposition',output_kind='vector',cost=Cost.EXTREME,tags=('eemd',),requires=('PyEMD',))
def sig_eemd(x,n_imfs=6,**kwargs):
    try:from PyEMD import EEMD
    except ImportError as exc:raise ImportError("Install EMD-signal (PyEMD) for signal.decomposition.eemd") from exc
    out=[]
    for r in as_2d_series(x):
        # parallel=False: PyEMD's EEMD spawns a multiprocessing.Pool by default. A
        # library feature must not silently start background processes -- on
        # Windows in particular, calling this from a script without an
        # `if __name__ == "__main__":` guard makes multiprocessing's "spawn" start
        # method re-import and re-execute the caller's whole module in every worker,
        # which can recursively spawn further pools. Running sequentially is slower
        # but produces the identical EEMD result and never surprises the caller with
        # background processes.
        #
        # Fifth audit pass, Finding 2: EEMD's ensemble noise is internally
        # randomized (previously left unseeded, so this feature was
        # genuinely non-reproducible run to run -- excluded from the
        # batch-equivalence harness as merely "known stochastic" rather than
        # fixed). PyEMD's EEMD.noise_seed(seed) makes it fully reproducible
        # given the same seed; seeding it from THIS row's own signal content
        # (not row position) makes the result both reproducible AND
        # batch-partition-invariant, the same content-seeding pattern used
        # for graph.embeddings.deepwalk.
        eemd=EEMD(trials=20,parallel=False)
        eemd.noise_seed(content_seed(np.ascontiguousarray(r,dtype=float).tobytes())%(2**32))
        imfs=eemd.eemd(r,max_imf=n_imfs);e=np.array([np.mean(v*v) for v in imfs[:n_imfs]]);out.append(np.pad(e,(0,n_imfs-len(e)),constant_values=0))
    return FeatureOutput(np.asarray(out),tuple(f'imf_energy_{i:02d}' for i in range(n_imfs)))

@_reg('signal.decomposition.vmd','decomposition',output_kind='vector',cost=Cost.EXTREME,tags=('vmd',),requires=('vmdpy',))
def sig_vmd(x,modes=5,**kwargs):
    try:from vmdpy import VMD
    except ImportError as exc:raise ImportError("Install vmdpy for signal.decomposition.vmd") from exc
    out=[]
    for r in as_2d_series(x):
        u,_,_=VMD(r,2000,0, modes,0,1,1e-7);out.append([np.mean(v*v) for v in u])
    return FeatureOutput(np.asarray(out),tuple(f'mode_energy_{i:02d}' for i in range(modes)))
