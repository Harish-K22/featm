from __future__ import annotations

import numpy as np
from ...registry import Complexity, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import discretize_1d, shannon_entropy_prob


def _dense_pairwise_distance_bytes(n_rows: int) -> int:
    """Curated estimator for features that build a dense (n, n) float64 distance matrix (scipy.spatial.distance.cdist(p, p))."""
    return int(n_rows) * int(n_rows) * 8


def _reg(name,family,*,cost=Cost.HIGH,output_kind='scalar',tags=(),complexity=None,memory_estimator=None):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        kwargs = {}
        if complexity is not None:
            kwargs['complexity'] = complexity
        if memory_estimator is not None:
            kwargs['memory_estimator'] = memory_estimator
        return register_feature(name=name,family=family,modality='spatial',input_types=('spatial',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=('scipy',),description=name,requires_full_materialization=True,**kwargs)(fn)
    return deco

def _parse(x):
    if isinstance(x,dict):p=np.asarray(x.get('coordinates'),float);v=x.get('values');v=None if v is None else np.asarray(v,float)
    else:p=np.asarray(x,float);v=None
    if p.ndim!=2 or p.shape[1]<2:raise ValueError('spatial features expect coordinates (n,d) or {coordinates, values}')
    if v is not None and len(v)!=len(p):raise ValueError('values length must match coordinates')
    return p,v

def _knn(p,k=8):
    from scipy.spatial import cKDTree
    d,idx=cKDTree(p).query(p,k=min(k+1,len(p)));return np.atleast_2d(d)[:,1:],np.atleast_2d(idx)[:,1:]

@_reg('spatial.neighbors.nearest_neighbor','neighbors',output_kind='vector',cost=Cost.LOW,tags=('nearest-neighbor',))
def nearest(x,k=8,**kwargs):
    p,_=_parse(x);d,idx=_knn(p,k);return FeatureOutput(np.column_stack([d[:,0],d.mean(1),d[:,-1]]),('nearest','k_mean','k_radius'))
@_reg('spatial.density.kernel_density','density',cost=Cost.MEDIUM,tags=('kde',))
def kde(x,bandwidth=None,**kwargs):
    from sklearn.neighbors import KernelDensity
    p,_=_parse(x)
    if bandwidth is None:
        d,_=_knn(p,min(5,len(p)-1));bandwidth=max(np.median(d[:,-1]),1e-6)
    model=KernelDensity(bandwidth=bandwidth).fit(p);return np.exp(model.score_samples(p))
@_reg('spatial.point_patterns.ripley_k','point_patterns',output_kind='vector',cost=Cost.HIGH,tags=('ripley-k',),complexity=Complexity.QUADRATIC,memory_estimator=_dense_pairwise_distance_bytes)
def ripley(x,radii=None,**kwargs):
    from scipy.spatial.distance import cdist
    p,_=_parse(x);D=cdist(p,p);np.fill_diagonal(D,np.inf)
    if radii is None:
        finite=D[np.isfinite(D)];radii=np.quantile(finite,[.1,.25,.5,.75])
    area=np.prod(np.ptp(p,axis=0)[:2]) if p.shape[1]>=2 else np.ptp(p[:,0]);density=len(p)/max(area,1e-15)
    arr=np.column_stack([(D<=r).sum(1)/max(density,1e-15) for r in radii]);return FeatureOutput(arr,tuple(f'r_{i}' for i in range(len(radii))))
@_reg('spatial.point_patterns.pair_correlation','point_patterns',output_kind='vector',cost=Cost.HIGH,tags=('pair-correlation',),complexity=Complexity.QUADRATIC,memory_estimator=_dense_pairwise_distance_bytes)
def pair_correlation(x,bins=4,**kwargs):
    from scipy.spatial.distance import cdist
    p,_=_parse(x);D=cdist(p,p);np.fill_diagonal(D,np.nan);edges=np.quantile(D[np.isfinite(D)],np.linspace(0,1,bins+1));arr=[]
    for i in range(len(p)):arr.append([np.mean((D[i]>=edges[j])&(D[i]<edges[j+1])) for j in range(bins)])
    return FeatureOutput(np.asarray(arr),tuple(f'shell_{i}' for i in range(bins)))

# Local spatial autocorrelation with binary kNN weights.
def _weights(p,k=8):
    d,idx=_knn(p,k);W=np.zeros((len(p),len(p)),float)
    for i,ids in enumerate(idx):W[i,np.atleast_1d(ids)]=1
    W=np.maximum(W,W.T);return W

def _need_values(x):
    p,v=_parse(x)
    if v is None:raise ValueError('this spatial statistic requires input {"coordinates": ..., "values": ...}')
    return p,v
@_reg('spatial.autocorrelation.moran_i','autocorrelation',cost=Cost.MEDIUM,tags=('moran-i',))
def moran(x,k=8,**kwargs):
    p,v=_need_values(x);W=_weights(p,k);z=v-np.mean(v);den=np.mean(z*z);return z*(W@z)/np.maximum(W.sum(1)*den,1e-15)
@_reg('spatial.autocorrelation.geary_c','autocorrelation',cost=Cost.MEDIUM,tags=('geary-c',))
def geary(x,k=8,**kwargs):
    p,v=_need_values(x);W=_weights(p,k);den=np.var(v);out=[]
    for i in range(len(v)):ids=np.where(W[i]>0)[0];out.append(np.mean((v[i]-v[ids])**2)/(2*den+1e-15) if len(ids) else np.nan)
    return np.asarray(out)
@_reg('spatial.hotspots.getis_ord','hotspots',cost=Cost.MEDIUM,tags=('getis-ord',))
def getis(x,k=8,**kwargs):
    p,v=_need_values(x);W=_weights(p,k);return (W@v)/np.maximum(W.sum(1),1)
@_reg('spatial.entropy.spatial_entropy','entropy',cost=Cost.MEDIUM,tags=('entropy',))
def spatial_entropy(x,k=8,bins=8,**kwargs):
    p,v=_need_values(x);_,idx=_knn(p,k);disc=discretize_1d(v,bins);out=[]
    for ids in idx:
        c=np.bincount(disc[np.atleast_1d(ids)],minlength=bins);out.append(float(shannon_entropy_prob(c)))
    return np.asarray(out)

@_reg('spatial.clustering.spatial_clustering','clustering',output_kind='vector',cost=Cost.MEDIUM,tags=('dbscan',))
def spatial_clustering(x,eps=None,min_samples=5,**kwargs):
    from sklearn.cluster import DBSCAN
    p,_=_parse(x)
    if eps is None:d,_=_knn(p,min(min_samples,len(p)-1));eps=max(np.median(d[:,-1]),1e-12)
    lab=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(p);cnt={l:int(np.sum(lab==l)) for l in set(lab)};return FeatureOutput(np.column_stack([lab,[cnt[l] for l in lab]]),('cluster_label','cluster_size'))
@_reg('spatial.graph.spatial_graph','graph',output_kind='vector',cost=Cost.MEDIUM,tags=('knn-graph',))
def spatial_graph(x,k=8,**kwargs):
    p,_=_parse(x);d,idx=_knn(p,k);return FeatureOutput(np.column_stack([np.full(len(p),idx.shape[1]),d.mean(1),d.max(1)]),('degree','edge_mean','edge_max'))
@_reg('spatial.triangulation.delaunay','triangulation',output_kind='vector',cost=Cost.HIGH,tags=('delaunay',),complexity=Complexity.LINEARITHMIC)
def spatial_delaunay(x,**kwargs):
    # complexity is a curated correction (see geometry/triangulation.py's _reg
    # for the same note): delegates to Qhull-based scipy.spatial.Delaunay,
    # O(n log n) for 2D/3D, not the cost=HIGH-derived quadratic default.
    from ..geometry.triangulation import delaunay_degree, mean_edge_length
    p,_=_parse(x);return FeatureOutput(np.column_stack([delaunay_degree(p),mean_edge_length(p)]),('degree','mean_edge_length'))
@_reg('spatial.tessellation.voronoi','tessellation',output_kind='vector',cost=Cost.HIGH,tags=('voronoi',),complexity=Complexity.LINEARITHMIC)
def spatial_voronoi(x,**kwargs):
    # complexity is a curated correction: delegates to Qhull-based
    # scipy.spatial.Voronoi, O(n log n) for 2D/3D, not the quadratic default.
    from ..geometry.triangulation import vor_neighbors,vor_area
    p,_=_parse(x);return FeatureOutput(np.column_stack([vor_neighbors(p),vor_area(p)]),('neighbor_count','cell_area'))
