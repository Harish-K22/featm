from __future__ import annotations

from ...contracts import forward_parameters

import numpy as np
from ...registry import Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import local_cov_eigen_features


def _reg(name,family,*,cost=Cost.HIGH,output_kind='scalar',tags=(),requires=('scipy','scikit-learn')):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        return register_feature(name=name,family=family,modality='manifold',input_types=('point_coordinates',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=requires,description=name,requires_full_materialization=True)(fn)
    return deco

def _p(x):
    a=np.asarray(x,float)
    if a.ndim!=2 or len(a)<3:raise ValueError('manifold features expect (n_samples,n_dimensions)')
    return a

@_reg('manifold.dimension.local_dimension','dimension',cost=Cost.MEDIUM,tags=('intrinsic-dimension','knn'))
def local_dim(x,k=10,**kwargs):
    from scipy.spatial import cKDTree
    p=_p(x);d,_=cKDTree(p).query(p,k=min(k+1,len(p)));d=np.atleast_2d(d)[:,1:];rk=np.maximum(d[:,-1],1e-12);logs=np.log(rk[:,None]/np.maximum(d[:,:-1],1e-12));return (d.shape[1]-1)/np.maximum(logs.sum(1),1e-12)
@_reg('manifold.dimension.intrinsic_dimension','dimension',cost=Cost.MEDIUM,tags=('intrinsic-dimension',))
def intrinsic_dim(x,**kwargs):
    d=local_dim(x,**kwargs);return np.full(len(_p(x)),np.nanmedian(d))
intrinsic_dim.__featm_parameters__ = forward_parameters(local_dim, positional=1, bound=[])
@_reg('manifold.curvature.curvature','curvature',cost=Cost.MEDIUM,tags=('curvature','local-pca'))
def curvature(x,k=12,**kwargs):
    p=_p(x);vals,_,_=local_cov_eigen_features(p,k=k);vals=np.pad(vals,((0,0),(0,max(0,3-vals.shape[1]))));return vals[:,-1]/np.maximum(vals[:,:3].sum(1),1e-15)
@_reg('manifold.tangent.local_tangent_space','tangent',output_kind='vector',cost=Cost.MEDIUM,tags=('tangent','local-pca'))
def tangent(x,k=12,**kwargs):
    from scipy.spatial import cKDTree
    p=_p(x);_,idx=cKDTree(p).query(p,k=min(k+1,len(p)));out=[]
    for i,ids in enumerate(np.atleast_2d(idx)):
        q=p[np.atleast_1d(ids)]-p[i];_,_,vt=np.linalg.svd(q,full_matrices=False);out.append(vt[0])
    return FeatureOutput(np.asarray(out),tuple(f'axis_{i}' for i in range(p.shape[1])))
@_reg('manifold.residual.local_manifold_residual','residual',cost=Cost.MEDIUM,tags=('residual','local-pca'))
def manifold_residual(x,k=12,components=2,**kwargs):
    from scipy.spatial import cKDTree
    p=_p(x);_,idx=cKDTree(p).query(p,k=min(k+1,len(p)));out=[]
    for i,ids in enumerate(np.atleast_2d(idx)):
        q=p[np.atleast_1d(ids)]-p[i];_,_,vt=np.linalg.svd(q,full_matrices=False);B=vt[:min(components,len(vt))];recon=(q@B.T)@B;out.append(np.mean(np.linalg.norm(q-recon,axis=1)))
    return np.asarray(out)
@_reg('manifold.distance.geodesic','distance',cost=Cost.HIGH,tags=('geodesic','knn-graph'))
def geodesic(x,k=8,**kwargs):
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra
    p=_p(x);G=kneighbors_graph(p,n_neighbors=min(k,len(p)-1),mode='distance',include_self=False);G=G.maximum(G.T);center=np.argmin(np.linalg.norm(p-p.mean(0),axis=1));return dijkstra(G,directed=False,indices=center)

@_reg('manifold.embedding.isomap','embedding',output_kind='vector',cost=Cost.HIGH,tags=('isomap',))
def isomap(x,n_components=2,n_neighbors=8,**kwargs):
    from sklearn.manifold import Isomap
    p=_p(x);k=min(n_components,p.shape[1],len(p)-1);z=Isomap(n_neighbors=min(n_neighbors,len(p)-1),n_components=k).fit_transform(p);return FeatureOutput(z,tuple(f'dim_{i}' for i in range(k)))
@_reg('manifold.embedding.laplacian_eigenmaps','embedding',output_kind='vector',cost=Cost.HIGH,tags=('spectral-embedding',))
def lap_eigen(x,n_components=2,n_neighbors=8,**kwargs):
    from sklearn.manifold import SpectralEmbedding
    p=_p(x);k=min(n_components,len(p)-1);z=SpectralEmbedding(n_components=k,n_neighbors=min(n_neighbors,len(p)-1),random_state=0).fit_transform(p);return FeatureOutput(z,tuple(f'dim_{i}' for i in range(k)))
@_reg('manifold.embedding.diffusion_maps','embedding',output_kind='vector',cost=Cost.HIGH,tags=('diffusion-map',))
def diffusion_maps(x,n_components=2,gamma=None,**kwargs):
    from sklearn.metrics.pairwise import rbf_kernel
    p=_p(x);K=rbf_kernel(p,gamma=gamma);D=K.sum(1);P=K/np.maximum(D[:,None],1e-15);e,v=np.linalg.eig(P);idx=np.argsort(np.real(e))[::-1][1:n_components+1];z=np.real(v[:,idx])*np.real(e[idx]);return FeatureOutput(z,tuple(f'dim_{i}' for i in range(z.shape[1])))
