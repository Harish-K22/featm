from __future__ import annotations

import numpy as np

from ...registry import Complexity, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import local_cov_eigen_features, pointcloud_batch, shannon_entropy_prob


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=(),complexity=None,memory_estimator=None):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        kwargs = {}
        if complexity is not None:
            kwargs['complexity'] = complexity
        if memory_estimator is not None:
            kwargs['memory_estimator'] = memory_estimator
        return register_feature(name=name,family=family,modality='pointcloud',input_types=('pointcloud',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=requires,description=name,requires_full_materialization=True,**kwargs)(fn)
    return deco


def _pc3(p):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]<2:raise ValueError('point cloud must be (n_points, dimensions>=2)')
    return p


def _local_metrics(p,k=16):
    p=_pc3(p);vals,normals,idx=local_cov_eigen_features(p,k=k)
    if vals.shape[1]<3:vals=np.pad(vals,((0,0),(0,3-vals.shape[1])))
    l1,l2,l3=vals[:,0],vals[:,1],vals[:,2];den=np.maximum(l1,1e-15);s=np.maximum(vals[:,:3].sum(1),1e-15)
    feats={
        'linearity':(l1-l2)/den,
        'planarity':(l2-l3)/den,
        'scattering':l3/den,
        'sphericity':l3/den,
        'anisotropy':(l1-l3)/den,
        'omnivariance':np.cbrt(np.maximum(l1*l2*l3,0)),
        'eigenentropy':shannon_entropy_prob(vals[:,:3],axis=1),
        'surface_variation':l3/s,
    }
    return feats,normals,idx

@_reg('pointcloud.size.n_points','size',cost=Cost.TINY,tags=('size',))
def n_points(x,**kwargs):return np.asarray([len(_pc3(p)) for p in pointcloud_batch(x)],float)
@_reg('pointcloud.geometry.bbox_volume','geometry',cost=Cost.TINY,tags=('bbox',))
def bbox_volume(x,**kwargs):return np.asarray([np.prod(np.ptp(_pc3(p),axis=0)) for p in pointcloud_batch(x)],float)
@_reg('pointcloud.geometry.centroid_radius_mean','geometry',cost=Cost.TINY,tags=('centroid',))
def centroid_mean(x,**kwargs):
    return np.asarray([np.mean(np.linalg.norm(_pc3(p)-_pc3(p).mean(0),axis=1)) for p in pointcloud_batch(x)],float)
@_reg('pointcloud.geometry.centroid_radius_std','geometry',cost=Cost.TINY,tags=('centroid',))
def centroid_std(x,**kwargs):
    return np.asarray([np.std(np.linalg.norm(_pc3(p)-_pc3(p).mean(0),axis=1)) for p in pointcloud_batch(x)],float)

for _catalog,_key in [
    ('pca_linearity','linearity'),('linearity','linearity'),('pca_planarity','planarity'),('planarity','planarity'),
    ('pca_sphericity','sphericity'),('anisotropy','anisotropy'),('eigenentropy','eigenentropy'),('omnivariance','omnivariance'),('scattering','scattering'),
]:
    @_reg(f'pointcloud.local_geometry.{_catalog}','local_geometry',output_kind='vector',cost=Cost.MEDIUM,tags=('local-pca',_key))
    def _metric(x,_key=_key,k=16,**kwargs):
        rows=[]
        for p in pointcloud_batch(x):
            v=_local_metrics(p,k=k)[0][_key];rows.append([np.nanmean(v),np.nanstd(v),np.nanquantile(v,.9)])
        return FeatureOutput(np.asarray(rows),('mean','std','q90'))

@_reg('pointcloud.surface.surface_variation','surface',output_kind='vector',cost=Cost.MEDIUM,tags=('surface-variation',))
def surface_variation(x,k=16,**kwargs):
    rows=[]
    for p in pointcloud_batch(x):v=_local_metrics(p,k=k)[0]['surface_variation'];rows.append([np.mean(v),np.std(v),np.quantile(v,.9)])
    return FeatureOutput(np.asarray(rows),('mean','std','q90'))

@_reg('pointcloud.geometry.curvature','geometry',output_kind='vector',cost=Cost.MEDIUM,tags=('curvature',))
def curvature(x,k=16,**kwargs):return surface_variation(x,k=k)

@_reg('pointcloud.geometry.normals','geometry',output_kind='vector',cost=Cost.MEDIUM,tags=('normals',))
def normals(x,k=16,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        _,n,_=_local_metrics(p,k=k);m=np.mean(np.abs(n),axis=0);s=np.std(n,axis=0);out.append(np.r_[m,s])
    width=len(out[0]) if out else 0;d=width//2;return FeatureOutput(np.asarray(out),tuple([f'abs_mean_{i}' for i in range(d)]+[f'std_{i}' for i in range(d)]))

@_reg('pointcloud.geometry.density','geometry',output_kind='vector',cost=Cost.MEDIUM,tags=('density',))
def density(x,k=8,**kwargs):
    from scipy.spatial import cKDTree
    out=[]
    for p in pointcloud_batch(x):
        p=_pc3(p);d,_=cKDTree(p).query(p,k=min(k+1,len(p)));d=np.atleast_2d(d);rk=d[:,-1];dim=p.shape[1];den=k/(np.maximum(rk,1e-12)**dim);out.append([np.mean(den),np.std(den),np.quantile(den,.9)])
    return FeatureOutput(np.asarray(out),('mean','std','q90'))

@_reg('pointcloud.geometry.roughness','geometry',output_kind='vector',cost=Cost.MEDIUM,tags=('roughness',))
def roughness(x,k=16,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        p=_pc3(p);_,norm,idx=_local_metrics(p,k=k);vals=[]
        for i,ids in enumerate(np.atleast_2d(idx)):
            ids=np.atleast_1d(ids);q=p[ids]-p[i];vals.append(np.mean(np.abs(q@norm[i])) if len(q) else 0)
        out.append([np.mean(vals),np.std(vals),np.max(vals)])
    return FeatureOutput(np.asarray(out),('mean','std','max'))

@_reg('pointcloud.voxel.voxel_density','voxel',output_kind='vector',cost=Cost.LOW,tags=('voxel',))
def voxel_density(x,resolution=16,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        p=_pc3(p);lo=p.min(0);span=np.maximum(np.ptp(p,axis=0),1e-12);idx=np.floor((p-lo)/span*(resolution-1)).astype(int);unique,c=np.unique(idx,axis=0,return_counts=True);out.append([len(unique)/(resolution**p.shape[1]),np.mean(c),np.max(c)])
    return FeatureOutput(np.asarray(out),('occupancy_fraction','points_per_occupied_mean','points_per_occupied_max'))

# Shape context: global aggregation of point-centered log-radial/angular histograms.
@_reg('pointcloud.descriptors.shape_context','descriptors',output_kind='vector',cost=Cost.HIGH,tags=('shape-context',))
def shape_context(x,radial_bins=5,angular_bins=12,**kwargs):
    out=[]
    for p in pointcloud_batch(x):
        p=_pc3(p)
        if p.shape[1]>2:p2=p[:,:2]
        else:p2=p
        c=p2.mean(0);v=p2-c;r=np.linalg.norm(v,axis=1);theta=(np.arctan2(v[:,1],v[:,0])+2*np.pi)%(2*np.pi);positive=r[r>0];rmin=max(np.min(positive) if len(positive) else 1e-6,1e-6);redges=np.geomspace(rmin,max(r.max(),rmin*1.01),radial_bins+1);aedges=np.linspace(0,2*np.pi,angular_bins+1);H,_,_=np.histogram2d(r,theta,bins=(redges,aedges));H=H/H.sum() if H.sum() else H;out.append(H.ravel())
    return FeatureOutput(np.asarray(out),tuple(f'r{r}_a{a}' for r in range(radial_bins) for a in range(angular_bins)))

# FPFH through Open3D if selected.
@_reg('pointcloud.descriptors.fpfh','descriptors',output_kind='vector',cost=Cost.HIGH,tags=('fpfh',),requires=('open3d',))
def fpfh(x,radius=None,max_nn=100,**kwargs):
    try:import open3d as o3d
    except ImportError as exc:raise ImportError("Install 'open3d' for pointcloud.descriptors.fpfh") from exc
    out=[]
    for p in pointcloud_batch(x):
        p=_pc3(p)
        if p.shape[1]!=3:raise ValueError('FPFH requires 3D points')
        pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p));scale=np.linalg.norm(np.ptp(p,axis=0));rad=radius or max(scale*.05,1e-6);pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad,max_nn=30));f=o3d.pipelines.registration.compute_fpfh_feature(pc,o3d.geometry.KDTreeSearchParamHybrid(radius=rad*2,max_nn=max_nn));out.append(np.asarray(f.data).mean(axis=1))
    return FeatureOutput(np.asarray(out),tuple(f'dim_{i:02d}' for i in range(33)))

# PCL-backed exact SHOT / Spin Image adapters.
for _nm,_cls in [('shot','SHOTEstimation'),('spin_images','SpinImageEstimation')]:
    @_reg(f'pointcloud.descriptors.{_nm}','descriptors',output_kind='vector',cost=Cost.EXTREME,tags=(_nm,),requires=('pclpy',))
    def _pcl_descriptor(x,_nm=_nm,**kwargs):
        raise ImportError(f"pointcloud.descriptors.{_nm} requires a PCL/pclpy descriptor backend; install pclpy and use the featm pointcloud plugin pack")

@_reg('pointcloud.triangulation.delaunay','triangulation',output_kind='vector',cost=Cost.HIGH,tags=('delaunay',),complexity=Complexity.LINEARITHMIC)
def pc_delaunay(x,**kwargs):
    # complexity is a curated correction (same Qhull-family issue already
    # fixed for geometry.triangulation.*/spatial.triangulation.delaunay/
    # spatial.tessellation.voronoi in prior audit passes -- this registration
    # was missed then): delegates to Qhull-based scipy.spatial.Delaunay via
    # geometry.triangulation.delaunay_degree, O(n log n) for 2D/3D, not the
    # cost=HIGH-derived quadratic default the planner would otherwise assume.
    from ..geometry.triangulation import delaunay_degree, mean_edge_length
    out=[]
    for p in pointcloud_batch(x):
        deg=delaunay_degree(p);edge=mean_edge_length(p);out.append([np.nanmean(deg),np.nanstd(deg),np.nanmean(edge),np.nanstd(edge)])
    return FeatureOutput(np.asarray(out),('degree_mean','degree_std','edge_mean','edge_std'))

@_reg('pointcloud.topology.persistence','topology',output_kind='vector',cost=Cost.HIGH,tags=('persistence','H0'))
def pc_persistence(x,n=8,**kwargs):
    from scipy.spatial.distance import pdist,squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    out=[]
    for p in pointcloud_batch(x):
        p=_pc3(p)
        if len(p)>400:p=p[np.linspace(0,len(p)-1,400).astype(int)]
        D=squareform(pdist(p));d=minimum_spanning_tree(D).toarray();v=np.sort(d[d>0])[-n:];out.append(np.pad(v,(n-len(v),0)))
    return FeatureOutput(np.asarray(out),tuple(f'h0_death_{i:02d}' for i in range(n)))

# Learned point-cloud embeddings require user-supplied encoders. This keeps the
# registry framework-agnostic while making the named features executable.
for _nm in ('pointnet','point_transformer'):
    @_reg(f'pointcloud.embeddings.{_nm}','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=(_nm,'embedding'),requires=('user-model',))
    def _pc_emb(x,model=None,_nm=_nm,**kwargs):
        if model is None:raise ValueError(f'pointcloud.embeddings.{_nm} requires callable model(points)->fixed vector')
        rows=[np.asarray(model(_pc3(p)),float).ravel() for p in pointcloud_batch(x)];w=len(rows[0]);
        if any(len(v)!=w for v in rows):raise ValueError('model outputs must have fixed width')
        return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:03d}' for i in range(w)))
