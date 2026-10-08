from __future__ import annotations

import numpy as np
from ...registry import Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import pointcloud_batch, shannon_entropy_prob


def _reg(name,family,*,cost=Cost.HIGH,output_kind='vector',tags=(),requires=()):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        return register_feature(name=name,family=family,modality='topology',input_types=('pointcloud',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=requires,description=name,requires_full_materialization=True)(fn)
    return deco


def _h0_intervals(p,max_points=400):
    from scipy.spatial.distance import pdist,squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    p=np.asarray(p,float)
    if p.ndim!=2 or len(p)<2:return np.empty((0,2))
    if len(p)>max_points:p=p[np.linspace(0,len(p)-1,max_points).astype(int)]
    D=squareform(pdist(p));m=minimum_spanning_tree(D).toarray();d=np.sort(m[m>0]);return np.column_stack([np.zeros(len(d)),d])

def _gudhi_intervals(p,kind='rips',max_dim=2):
    try:import gudhi as gd
    except ImportError:return None
    p=np.asarray(p,float)
    try:
        if kind=='rips':st=gd.RipsComplex(points=p).create_simplex_tree(max_dimension=max_dim+1)
        elif kind=='alpha':st=gd.AlphaComplex(points=p).create_simplex_tree()
        elif kind=='cech':st=gd.CechComplex(points=p,max_radius=float(np.linalg.norm(np.ptp(p,axis=0)))).create_simplex_tree(max_dimension=max_dim+1)
        elif kind=='witness':
            landmarks=p[np.linspace(0,len(p)-1,min(50,len(p))).astype(int)];wc=gd.EuclideanWitnessComplex(witnesses=p,landmarks=landmarks);st=wc.create_simplex_tree(max_alpha_square=float(np.linalg.norm(np.ptp(p,axis=0))**2),limit_dimension=max_dim+1)
        else:return None
        st.persistence();return [np.asarray(st.persistence_intervals_in_dimension(d)) for d in range(max_dim+1)]
    except Exception:return None

def _summary(intervals,max_dim=2):
    vals=[]
    if isinstance(intervals,np.ndarray):intervals=[intervals]
    for d in range(max_dim+1):
        a=intervals[d] if d<len(intervals) else np.empty((0,2));a=np.asarray(a,float)
        finite=a[np.isfinite(a[:,1])] if a.size else np.empty((0,2));life=finite[:,1]-finite[:,0] if len(finite) else np.array([]);vals.extend([len(a),np.max(life) if len(life) else 0,np.sum(life) if len(life) else 0])
    return vals

for _nm,_kind in [('vietoris_rips','rips'),('alpha_complex','alpha'),('cech','cech'),('witness_complex','witness')]:
    @_reg(f'topology.persistent_homology.{_nm}','persistent_homology',tags=('persistent-homology',_kind),requires=('gudhi (optional for H1+)',))
    def _ph(x,_kind=_kind,**kwargs):
        out=[]
        for p in pointcloud_batch(x):
            ints=_gudhi_intervals(p,_kind)
            if ints is None:ints=[_h0_intervals(p),np.empty((0,2)),np.empty((0,2))]
            out.append(_summary(ints,2))
        return FeatureOutput(np.asarray(out),tuple(f'H{d}_{m}' for d in range(3) for m in ('count','max_persistence','total_persistence')))

@_reg('topology.persistent_homology.cubical_complex','persistent_homology',tags=('cubical-complex',),requires=('gudhi',))
def cubical(x,**kwargs):
    try:import gudhi as gd
    except ImportError as exc:raise ImportError("Install 'gudhi' for cubical complex persistence") from exc
    arr=np.asarray(x,float)
    if arr.ndim==2:arr=arr[None,:,:]
    out=[]
    for im in arr:
        cc=gd.CubicalComplex(top_dimensional_cells=im);cc.persistence();ints=[np.asarray(cc.persistence_intervals_in_dimension(d)) for d in range(3)];out.append(_summary(ints,2))
    return FeatureOutput(np.asarray(out),tuple(f'H{d}_{m}' for d in range(3) for m in ('count','max_persistence','total_persistence')))

@_reg('topology.betti.numbers','betti',tags=('betti',))
def betti(x,**kwargs):
    r=_ph_dispatch(x);return FeatureOutput(r[:,[0,3,6]],('beta0','beta1','beta2'))
def _ph_dispatch(x):return globals()['_ph_rips_alias'](x).values if '_ph_rips_alias' in globals() else _rips_direct(x)
def _rips_direct(x):
    out=[]
    for p in pointcloud_batch(x):
        ints=_gudhi_intervals(p,'rips') or [_h0_intervals(p),np.empty((0,2)),np.empty((0,2))];out.append(_summary(ints,2))
    return np.asarray(out)
# keep an internal alias independent of decorator-generated names
_ph_rips_alias=lambda x: FeatureOutput(_rips_direct(x),tuple(f'H{d}_{m}' for d in range(3) for m in ('count','max_persistence','total_persistence')))

@_reg('topology.euler.characteristic','euler',cost=Cost.MEDIUM,tags=('euler',))
def euler(x,**kwargs):
    r=_rips_direct(x);return r[:,0]-r[:,3]+r[:,6]

@_reg('topology.persistence.entropy','persistence',cost=Cost.HIGH,tags=('persistence-entropy',))
def pers_entropy(x,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        d=_h0_intervals(p);life=d[:,1]-d[:,0] if len(d) else np.array([]);out.append(float(shannon_entropy_prob(life)) if len(life) else 0)
    return np.asarray(out)
@_reg('topology.persistence.landscape','persistence',output_kind='vector',cost=Cost.HIGH,tags=('landscape',))
def landscape(x,grid=16,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        d=_h0_intervals(p)[:,1];mx=d.max() if len(d) else 1;g=np.linspace(0,mx,grid);out.append([np.max(np.minimum(t,np.maximum(d-t,0))) if len(d) else 0 for t in g])
    return FeatureOutput(np.asarray(out),tuple(f'x_{i:02d}' for i in range(grid)))
@_reg('topology.persistence.image','persistence',output_kind='vector',cost=Cost.HIGH,tags=('persistence-image',))
def pimage(x,bins=16,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        d=_h0_intervals(p)[:,1];h,_=np.histogram(d,bins=bins,range=(0,max(d.max(),1e-12)) if len(d) else (0,1),weights=d if len(d) else None);out.append(h)
    return FeatureOutput(np.asarray(out),tuple(f'bin_{i:02d}' for i in range(bins)))
@_reg('topology.local_topology.local_persistence','local_topology',output_kind='vector',cost=Cost.EXTREME,tags=('local-persistence',))
def local_persistence(x,k=20,max_centers=50,**kwargs):
    from scipy.spatial import cKDTree
    out=[]
    for p in pointcloud_batch(x):
        p=np.asarray(p,float);_,idx=cKDTree(p).query(p,k=min(k,len(p)));centers=np.linspace(0,len(p)-1,min(max_centers,len(p))).astype(int);vals=[]
        for i in centers:
            d=_h0_intervals(p[np.atleast_1d(idx[i])]);life=d[:,1] if len(d) else np.array([]);vals.append(np.sum(life))
        out.append([np.mean(vals),np.std(vals),np.max(vals) if vals else 0])
    return FeatureOutput(np.asarray(out),('mean_total_persistence','std_total_persistence','max_total_persistence'))

@_reg('topology.mapper.mapper','mapper',output_kind='vector',cost=Cost.EXTREME,tags=('mapper',),requires=('scikit-learn',))
def mapper(x,n_intervals=8,overlap=.3,**kwargs):
    # Compact Mapper graph summary: PCA lens -> overlapping intervals -> DBSCAN clusters -> nerve graph.
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    out=[]
    for p in pointcloud_batch(x):
        p=np.asarray(p,float);lens=PCA(n_components=1).fit_transform(p).ravel();lo,hi=lens.min(),lens.max();width=(hi-lo)/(n_intervals-(n_intervals-1)*overlap+1e-12);step=width*(1-overlap);clusters=[]
        for i in range(n_intervals):
            a=lo+i*step;b=a+width;ids=np.where((lens>=a)&(lens<=b))[0]
            if len(ids)<2:continue
            eps=max(np.median(np.linalg.norm(p[ids]-p[ids].mean(0),axis=1))*.5,1e-12);lab=DBSCAN(eps=eps,min_samples=2).fit_predict(p[ids])
            for l in set(lab)-{-1}:clusters.append(set(ids[lab==l]))
        edges=0
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                if clusters[i]&clusters[j]:edges+=1
        comps=max(len(clusters)-edges,1) if clusters else 0;out.append([len(clusters),edges,comps])
    return FeatureOutput(np.asarray(out),('nodes','edges','component_proxy'))
