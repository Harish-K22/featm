from __future__ import annotations

from collections import Counter
import hashlib
from math import factorial
from typing import Iterable, Sequence
import zlib

import numpy as np


def content_seed(data: bytes) -> int:
    """Deterministic, non-salted integer seed derived from arbitrary content bytes.

    Used to seed a fresh per-sample RNG for a feature whose underlying
    algorithm is internally randomized (e.g. DeepWalk's random-walk
    generation, EEMD's ensemble noise) so the result depends only on that
    one logical sample's own content -- never on Python's per-process hash
    salt, nor on which batch/call position the sample happened to land in
    (fifth audit pass, Finding 2: a shared RNG created once outside a
    per-sample loop advances across samples within one call, so the same
    sample's result silently differed depending on which other samples were
    computed alongside it in the same batch). Deliberately not the builtin
    ``hash()``: str/bytes hashing is salted per-process by default
    (``PYTHONHASHSEED``) and is not reproducible across runs/processes, so
    it cannot be used for persistent, checkpoint-safe seeding.
    """
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def as_2d_series(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("expected shape (n_samples, n_steps) or (n_steps,)")
    return a


def as_images(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.ndim not in (3, 4):
        raise ValueError("images must be (n,h,w) or (n,h,w,c)")
    return a.astype(np.float64, copy=False)


def gray_images(x) -> np.ndarray:
    a = as_images(x)
    if a.ndim == 4:
        if a.shape[-1] >= 3:
            a = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        else:
            a = a[..., 0]
    return a


def as_videos(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 3:  # t,h,w
        a = a[None, :, :, :]
    if a.ndim not in (4, 5):
        raise ValueError("videos must be (n,t,h,w) or (n,t,h,w,c)")
    return a


def gray_videos(x) -> np.ndarray:
    a = as_videos(x)
    if a.ndim == 5:
        if a.shape[-1] >= 3:
            a = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        else:
            a = a[..., 0]
    return a


def normalize_prob(p, axis=-1, eps=1e-15):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0, None)
    s = p.sum(axis=axis, keepdims=True)
    return np.divide(p, s, out=np.zeros_like(p), where=s > eps)


def shannon_entropy_prob(p, axis=-1, normalize=False):
    p = normalize_prob(p, axis=axis)
    lp = np.zeros_like(p)
    np.log(p, out=lp, where=p > 0)
    h = -(p * lp).sum(axis=axis)
    if normalize:
        k = p.shape[axis]
        if k > 1:
            h = h / np.log(k)
    return h


def discrete_entropy(values) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    counts = np.asarray(list(Counter(vals).values()), dtype=float)
    return float(shannon_entropy_prob(counts))


def discretize_1d(x, bins=16):
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(len(x), dtype=int)
    if np.nanmax(finite) == np.nanmin(finite):
        return np.zeros(len(x), dtype=int)
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(finite, qs))
    return np.digitize(np.nan_to_num(x, nan=np.nanmedian(finite)), edges)


def mutual_information_discrete(x, y, bins=16) -> float:
    xi = discretize_1d(x, bins)
    yi = discretize_1d(y, bins)
    n = min(len(xi), len(yi))
    if n == 0:
        return 0.0
    xi, yi = xi[:n], yi[:n]
    joint = Counter(zip(xi.tolist(), yi.tolist()))
    cx, cy = Counter(xi.tolist()), Counter(yi.tolist())
    out = 0.0
    for (a, b), c in joint.items():
        pxy = c / n
        px = cx[a] / n
        py = cy[b] / n
        out += pxy * np.log(max(pxy / max(px * py, 1e-15), 1e-15))
    return float(out)


def conditional_mi_discrete(x, y, z, bins=8) -> float:
    xi, yi, zi = discretize_1d(x, bins), discretize_1d(y, bins), discretize_1d(z, bins)
    n = min(len(xi), len(yi), len(zi))
    if n == 0:
        return 0.0
    xi, yi, zi = xi[:n], yi[:n], zi[:n]
    cz = Counter(zi.tolist())
    out = 0.0
    for zv, nz in cz.items():
        mask = zi == zv
        if mask.sum() < 2:
            continue
        out += (nz / n) * mutual_information_discrete(xi[mask], yi[mask], bins=bins)
    return float(out)


def _spectral_parts_impl(a: np.ndarray):
    a = np.ascontiguousarray(a)
    centered = a - np.nanmean(a, axis=1, keepdims=True)
    centered = np.nan_to_num(centered)
    spec = np.fft.rfft(centered, axis=1)
    spec[:, 0] = 0.0  # Mean-centered DC is mathematically zero.
    mag = np.abs(spec)
    power = mag ** 2
    freqs = np.fft.rfftfreq(a.shape[1], d=1.0)
    return spec, mag, power, freqs


def spectral_parts(a: np.ndarray):
    """FFT/magnitude/power/freqs shared by every ``timeseries.spectral.*`` feature.

    Memoized per execution scope (see :mod:`featm.cache`) so selecting several
    spectral features together computes the FFT once instead of once per
    feature.
    """
    a = as_2d_series(a)
    from ..cache import array_cache_key, current_cache

    cache = current_cache()
    if cache is None:
        return _spectral_parts_impl(a)
    key = array_cache_key("timeseries.spectral_parts", a)
    return cache.get_or_compute(key, lambda: _spectral_parts_impl(a))


def stable_phase(spec):
    threshold = np.finfo(float).eps * 8 * np.max(np.abs(spec), axis=1, keepdims=True)
    return np.where(np.abs(spec) <= threshold, 0.0, np.angle(spec))


def cepstrum_values(mag):
    floor = np.maximum(1e-15, np.finfo(float).eps * 8 * np.max(mag, axis=1, keepdims=True))
    return np.fft.irfft(np.log(np.maximum(mag, floor)), axis=1)


def spectral_entropy_rows(a: np.ndarray):
    _, _, power, _ = spectral_parts(a)
    return shannon_entropy_prob(power, axis=1, normalize=True)


def autocorr_row(x: np.ndarray, lag: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag or np.nanstd(x) == 0:
        return np.nan
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def sample_entropy_1d(x, m=2, r=None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0:
        return 0.0

    def count(mm):
        seq = np.asarray([x[i:i+mm] for i in range(n-mm+1)])
        c = 0
        for i in range(len(seq)-1):
            c += int(np.sum(np.max(np.abs(seq[i+1:] - seq[i]), axis=1) <= r))
        return c
    b = count(m)
    a = count(m+1)
    if b == 0:
        return np.inf
    if a == 0:
        return float(np.log(b + 1.0))
    return float(-np.log(a / b))


def approximate_entropy_1d(x, m=2, r=None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x)
    if r <= 0:
        return 0.0

    def phi(mm):
        seq = np.asarray([x[i:i+mm] for i in range(n-mm+1)])
        vals=[]
        for i in range(len(seq)):
            c=np.mean(np.max(np.abs(seq-seq[i]), axis=1) <= r)
            vals.append(np.log(max(c,1e-15)))
        return np.mean(vals)
    return float(phi(m)-phi(m+1))


def permutation_entropy_1d(x, order=3, delay=1, normalize=True) -> float:
    x=np.asarray(x,dtype=float)
    n=len(x)-(order-1)*delay
    if n <= 1:
        return np.nan
    counts=Counter()
    for i in range(n):
        w=x[i:i+order*delay:delay]
        if np.any(~np.isfinite(w)):
            continue
        counts[tuple(np.argsort(w, kind='mergesort'))]+=1
    if not counts:
        return np.nan
    h=discrete_entropy(list(k for k,c in counts.items() for _ in range(c)))
    if normalize:
        h/=np.log(factorial(order))
    return float(h)


def lempel_ziv_complexity_1d(x, bins=2) -> float:
    if not isinstance(x, (str, bytes)):
        arr=np.asarray(x)
        if arr.dtype.kind in 'fiu':
            if bins == 2:
                med=np.nanmedian(arr.astype(float))
                s=''.join('1' if v>med else '0' for v in np.nan_to_num(arr, nan=med))
            else:
                d=discretize_1d(arr.astype(float), bins)
                s=''.join(chr(65+int(v)%26) for v in d)
        else:
            s='|'.join(map(str,arr.tolist()))
    else:
        s=x.decode(errors='ignore') if isinstance(x,bytes) else x
    n=len(s)
    if n==0:
        return 0.0
    i,k,l,c=0,1,1,1
    while True:
        if i+k>n or l+k>n:
            c+=1
            break
        if s[i:i+k]==s[l:l+k]:
            k+=1
            if l+k>n:
                c+=1
                break
        else:
            i+=1
            if i==l:
                c+=1
                l+=k
                if l>=n:
                    break
                i=0;k=1
    return float(c)


def compressed_ratio(obj) -> float:
    if isinstance(obj, bytes):
        b=obj
    elif isinstance(obj,str):
        b=obj.encode('utf-8',errors='ignore')
    else:
        b=np.asarray(obj).tobytes()
    if not b:
        return 0.0
    return len(zlib.compress(b)) / len(b)


def hurst_1d(x) -> float:
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    n=len(x)
    if n<20:
        return np.nan
    lags=np.unique(np.logspace(np.log10(2), np.log10(max(3,n//4)), 12).astype(int))
    tau=[]; valid=[]
    for lag in lags:
        d=x[lag:]-x[:-lag]
        sd=np.std(d)
        if sd>0:
            tau.append(sd); valid.append(lag)
    if len(tau)<2:
        return np.nan
    slope=np.polyfit(np.log(valid), np.log(tau),1)[0]
    return float(slope)


def katz_fd_1d(x) -> float:
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    n=len(x)
    if n<2:return np.nan
    L=np.sum(np.sqrt(1+np.diff(x)**2))
    d=np.max(np.sqrt(np.arange(n)**2+(x-x[0])**2))
    if L<=0 or d<=0:return 1.0
    return float(np.log10(n)/(np.log10(n)+np.log10(d/L)))


def petrosian_fd_1d(x) -> float:
    x=np.asarray(x,dtype=float)
    n=len(x)
    if n<3:return np.nan
    d=np.diff(x)
    n_delta=np.sum(d[1:]*d[:-1]<0)
    return float(np.log10(n)/(np.log10(n)+np.log10(n/(n+0.4*n_delta))))


def higuchi_fd_1d(x, kmax=8) -> float:
    x=np.asarray(x,dtype=float)
    x=np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0)
    n=len(x)
    if n<kmax+2:return np.nan
    L=[]; K=[]
    for k in range(1,kmax+1):
        lm=[]
        for m in range(k):
            idx=np.arange(m,n,k)
            if len(idx)<2: continue
            length=np.sum(np.abs(np.diff(x[idx])))
            norm=(n-1)/((len(idx)-1)*k)
            lm.append(length*norm/k)
        if lm and np.mean(lm)>0:
            L.append(np.mean(lm)); K.append(k)
    if len(L)<2:return np.nan
    return float(np.polyfit(np.log(1/np.asarray(K)), np.log(L),1)[0])


def dfa_1d(x) -> float:
    x=np.asarray(x,dtype=float)
    x=np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(x).any() else 0)
    n=len(x)
    if n<16:return np.nan
    y=np.cumsum(x-np.mean(x))
    sizes=np.unique(np.logspace(np.log10(4), np.log10(max(5,n//4)), 10).astype(int))
    F=[]; S=[]
    for s in sizes:
        vals=[]
        for start in range(0,n-s+1,s):
            seg=y[start:start+s]
            t=np.arange(s)
            p=np.polyfit(t,seg,1)
            vals.append(np.sqrt(np.mean((seg-np.polyval(p,t))**2)))
        if vals and np.mean(vals)>0:
            F.append(np.mean(vals));S.append(s)
    if len(F)<2:return np.nan
    return float(np.polyfit(np.log(S),np.log(F),1)[0])


def rolling_windows(x: np.ndarray, window: int):
    x=np.asarray(x)
    if len(x)<window:return np.empty((0,window))
    return np.lib.stride_tricks.sliding_window_view(x,window)


def recurrence_matrix_1d(x, eps=None, embedding_dim=2, delay=1, max_points=600):
    x=np.asarray(x,dtype=float)
    n=len(x)-(embedding_dim-1)*delay
    if n<=2:return np.zeros((0,0),dtype=bool)
    emb=np.column_stack([x[i*delay:i*delay+n] for i in range(embedding_dim)])
    emb=emb[np.all(np.isfinite(emb),axis=1)]
    if len(emb)>max_points:
        idx=np.linspace(0,len(emb)-1,max_points).astype(int);emb=emb[idx]
    if len(emb)<2:return np.zeros((0,0),dtype=bool)
    from scipy.spatial.distance import pdist,squareform
    d=squareform(pdist(emb))
    if eps is None:
        nz=d[d>0]
        eps=np.quantile(nz,0.1) if nz.size else 0
    return d<=eps


def rqa_stats_1d(x):
    R=recurrence_matrix_1d(x)
    if R.size==0:return dict(rate=np.nan,determinism=np.nan,laminarity=np.nan,trapping=np.nan)
    np.fill_diagonal(R,False)
    rate=R.mean()
    diag_lengths=[]
    for k in range(-len(R)+1,len(R)):
        arr=np.diag(R,k)
        runs=_true_runs(arr)
        diag_lengths.extend([r for r in runs if r>=2])
    vert_lengths=[]
    for j in range(R.shape[1]):
        vert_lengths.extend([r for r in _true_runs(R[:,j]) if r>=2])
    rec=max(R.sum(),1)
    det=sum(diag_lengths)/rec if diag_lengths else 0.0
    lam=sum(vert_lengths)/rec if vert_lengths else 0.0
    trap=float(np.mean(vert_lengths)) if vert_lengths else 0.0
    return dict(rate=float(rate),determinism=float(det),laminarity=float(lam),trapping=trap)


def _true_runs(a):
    a=np.asarray(a,dtype=bool)
    if len(a)==0:return []
    z=np.r_[False,a,False].astype(int)
    d=np.diff(z)
    starts=np.where(d==1)[0];ends=np.where(d==-1)[0]
    return (ends-starts).tolist()


def local_cov_eigen_features(points, k=16):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or len(points)<3:
        raise ValueError('points must be (n,d) with n>=3')
    from scipy.spatial import cKDTree
    kk=min(k+1,len(points))
    _, idx=cKDTree(points).query(points,k=kk)
    if idx.ndim==1:idx=idx[:,None]
    vals=np.zeros((len(points),points.shape[1]))
    normals=np.zeros_like(points)
    for i,ids in enumerate(idx):
        ids=np.atleast_1d(ids)
        ids=ids[ids!=i]
        q=points[ids]-points[i]
        if len(q)<2:continue
        C=np.cov(q,rowvar=False)
        ev,vec=np.linalg.eigh(np.atleast_2d(C))
        ev=np.clip(ev,0,None)
        order=np.argsort(ev)[::-1]
        ev=ev[order]
        vals[i,:len(ev)]=ev
        normals[i]=vec[:,order[-1]] if vec.shape[0]==points.shape[1] else 0
    return vals,normals,idx


def pointcloud_batch(x):
    if isinstance(x,np.ndarray) and x.ndim==3:
        return [x[i] for i in range(len(x))]
    if isinstance(x,np.ndarray) and x.ndim==2:
        return [x]
    return [np.asarray(v,dtype=float) for v in x]


def matrix_batch(x):
    a=np.asarray(x,dtype=float)
    if a.ndim==2:a=a[None,:,:]
    if a.ndim!=3:raise ValueError('expected (n,m,k) or (m,k)')
    return a


def sequence_batch(x):
    if isinstance(x,(str,bytes)):
        return [x]
    if isinstance(x,np.ndarray) and x.ndim==1:
        # Ambiguous; treat one numeric vector as one sequence.
        return [x]
    return list(x)
