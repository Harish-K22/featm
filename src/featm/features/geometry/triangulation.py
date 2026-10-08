from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import itertools
import math
import numpy as np

from ...registry import Complexity, Cost, FeatureOutput, REGISTRY, register_feature


def _points(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or a.shape[1]<2:raise ValueError('points must be (n_samples,n_dimensions>=2)')
    if len(a)<2:raise ValueError('at least two points required')
    return a


def _reg(name,family,*,cost=Cost.HIGH,output_kind='scalar',tags=(),cache_key='geometry.delaunay',min_rows=3):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        # min_rows=3 is the true minimum for the common 2D case (Delaunay needs
        # dim+1 points); higher-dimensional inputs need more, but the static
        # registry field cannot know the runtime dimensionality, so this is a
        # conservative floor, not an exact bound -- _tri_data_impl still raises
        # its own precise error for higher-dimensional under-provisioned input.
        #
        # complexity=LINEARITHMIC is a curated correction, not the cost-derived
        # default: cost=HIGH would otherwise default to Complexity.QUADRATIC
        # (registry.py's coarse cost->complexity mapping), which the Step 2
        # resource planner treats as "may need an O(n^2) working structure"
        # for full-materialization safety checks (see
        # resources.estimate_full_materialization_bytes). scipy.spatial.Delaunay
        # (Qhull) is O(n log n) time with O(n) simplices for the 2D/3D case
        # these features are used for; leaving the quadratic default in place
        # would make the planner wrongly refuse a perfectly safe large Delaunay
        # run (e.g. 100k 2D points is tens of MB in practice, not the ~80GB an
        # O(n^2) assumption would estimate).
        return register_feature(name=name,family=family,modality='geometry',input_types=('point_coordinates',),cost=cost,complexity=Complexity.LINEARITHMIC,streamable=False,output_kind=output_kind,tags=tags,requires=('scipy',),description=name,cache_key=cache_key,requires_full_materialization=True,min_rows=min_rows)(fn)
    return deco


def _simplex_metrics(pts, simplex):
    q=pts[np.asarray(simplex)]
    d=q.shape[1]
    edges=[np.linalg.norm(q[i]-q[j]) for i,j in itertools.combinations(range(len(q)),2)]
    # d-simplex volume = |det(q1-q0,...,qd-q0)|/d!
    if len(q)==d+1:
        M=(q[1:]-q[0]).T
        vol=abs(np.linalg.det(M))/math.factorial(d) if d>0 else 0
    else:vol=np.nan
    # Circumradius using linear solve relative to first vertex.
    try:
        A=2*(q[1:]-q[0]);b=np.sum(q[1:]**2,axis=1)-np.sum(q[0]**2);center=np.linalg.solve(A,b);circ=np.linalg.norm(center-q[0])
    except Exception:circ=np.nan
    # Inradius = d*V/surface area for simplex. Compute (d-1)-face volumes.
    faces=[]
    if d>=2 and len(q)==d+1:
        for omit in range(len(q)):
            f=np.delete(q,omit,axis=0);base=f[1:]-f[0];gram=base@base.T
            fv=np.sqrt(max(np.linalg.det(gram),0))/math.factorial(d-1);faces.append(fv)
    surf=sum(faces);inr=d*vol/surf if surf>0 else np.nan
    aspect=max(edges)/(min(edges)+1e-15) if edges else np.nan
    quality=(d*(vol**(1/d))/(max(edges)+1e-15)) if d>0 and vol>0 and edges else 0
    angles=[]
    if d==2 and len(q)==3:
        for i in range(3):
            a=q[(i+1)%3]-q[i];b2=q[(i+2)%3]-q[i];co=np.dot(a,b2)/(np.linalg.norm(a)*np.linalg.norm(b2)+1e-15);angles.append(np.arccos(np.clip(co,-1,1)))
    return dict(edges=np.asarray(edges),volume=vol,circumradius=circ,inradius=inr,aspect=aspect,quality=quality,angles=np.asarray(angles))


def _tri_data_impl(pts):
    from scipy.spatial import Delaunay
    if len(pts)<pts.shape[1]+1:raise ValueError('not enough points for Delaunay triangulation')
    tri=Delaunay(pts);neigh=[set() for _ in pts];incident=[[] for _ in pts]
    metrics=[]
    for si,simplex in enumerate(tri.simplices):
        m=_simplex_metrics(pts,simplex);metrics.append(m)
        for i in simplex:incident[int(i)].append(si)
        for a,b in itertools.combinations(simplex,2):neigh[int(a)].add(int(b));neigh[int(b)].add(int(a))
    return pts,tri,neigh,incident,metrics


def _tri_data(points):
    """Delaunay triangulation shared by every ``geometry.triangulation.*`` feature.

    Memoized per execution scope so selecting several triangulation-derived
    features (degree, edge lengths, circumradius, simplex quality, ...)
    triangulates the point set once instead of once per feature.
    """
    pts=_points(points)
    from ...cache import array_cache_key, current_cache
    cache=current_cache()
    if cache is None:
        return _tri_data_impl(pts)
    key=array_cache_key('geometry.delaunay', pts)
    return cache.get_or_compute(key, lambda: _tri_data_impl(pts))


def _edge_values(pts,neigh,i):
    ns=list(neigh[i]);return np.linalg.norm(pts[ns]-pts[i],axis=1) if ns else np.array([])

for _name,_fn in [
    ('delaunay_degree',lambda pts,neigh,i:len(neigh[i])),
    ('mean_edge_length',lambda pts,neigh,i:np.mean(_edge_values(pts,neigh,i)) if len(neigh[i]) else np.nan),
    ('min_edge_length',lambda pts,neigh,i:np.min(_edge_values(pts,neigh,i)) if len(neigh[i]) else np.nan),
    ('max_edge_length',lambda pts,neigh,i:np.max(_edge_values(pts,neigh,i)) if len(neigh[i]) else np.nan),
    ('std_edge_length',lambda pts,neigh,i:np.std(_edge_values(pts,neigh,i)) if len(neigh[i]) else np.nan),
    ('edge_length',lambda pts,neigh,i:np.mean(_edge_values(pts,neigh,i)) if len(neigh[i]) else np.nan),
]:
    @_reg(f'geometry.triangulation.{_name}','triangulation',tags=('delaunay','local-geometry'))
    def _edge_feat(points,_fn=_fn,**kwargs):
        pts,tri,neigh,incident,metrics=_tri_data(points);return np.asarray([_fn(pts,neigh,i) for i in range(len(pts))],float)

for _name,_key in [('simplex_volume','volume'),('mean_simplex_volume','volume'),('circumradius','circumradius'),('inradius','inradius'),('aspect_ratio','aspect'),('simplex_quality','quality')]:
    @_reg(f'geometry.triangulation.{_name}','triangulation',tags=('simplex',_key))
    def _simp(points,_key=_key,**kwargs):
        pts,tri,neigh,incident,metrics=_tri_data(points);out=[]
        for ids in incident:
            v=[metrics[j][_key] for j in ids];out.append(np.nanmean(v) if v else np.nan)
        return np.asarray(out,float)

@_reg('geometry.triangulation.triangle_area','triangulation',tags=('triangle','area'))
def triangle_area(points,**kwargs):
    pts,tri,neigh,incident,metrics=_tri_data(points)
    if pts.shape[1]!=2:raise ValueError('triangle_area is defined for 2D Delaunay triangulations')
    return np.asarray([np.nanmean([metrics[j]['volume'] for j in ids]) if ids else np.nan for ids in incident])

@_reg('geometry.triangulation.triangle_angles','triangulation',output_kind='vector',tags=('triangle','angles'))
def triangle_angles(points,**kwargs):
    pts,tri,neigh,incident,metrics=_tri_data(points)
    if pts.shape[1]!=2:raise ValueError('triangle_angles is defined for 2D Delaunay triangulations')
    out=[]
    for ids in incident:
        a=np.concatenate([metrics[j]['angles'] for j in ids]) if ids else np.array([]);out.append([np.mean(a) if len(a) else np.nan,np.std(a) if len(a) else np.nan,np.min(a) if len(a) else np.nan,np.max(a) if len(a) else np.nan])
    return FeatureOutput(np.asarray(out),('mean','std','min','max'))

# Distances and local angles.
@_reg('geometry.distances.euclidean','distances',cost=Cost.TINY,tags=('distance',))
def euclidean(points,**kwargs):
    p=_points(points);c=np.mean(p,axis=0);return np.linalg.norm(p-c,axis=1)
@_reg('geometry.distances.mahalanobis','distances',cost=Cost.LOW,tags=('mahalanobis',))
def mahalanobis(points,**kwargs):
    p=_points(points);c=np.mean(p,axis=0);cov=np.cov(p,rowvar=False);inv=np.linalg.pinv(np.atleast_2d(cov));d=p-c;return np.sqrt(np.einsum('ij,jk,ik->i',d,inv,d))
@_reg('geometry.angles.local_angles','angles',output_kind='vector',cost=Cost.MEDIUM,tags=('angles','knn'))
def local_angles(points,k=6,**kwargs):
    from scipy.spatial import cKDTree
    p=_points(points);_,idx=cKDTree(p).query(p,k=min(k+1,len(p)));out=[]
    for i,ids in enumerate(np.atleast_2d(idx)):
        ids=[j for j in np.atleast_1d(ids) if j!=i][:k];vec=p[ids]-p[i];ang=[]
        for a,b in itertools.combinations(vec,2):ang.append(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-15),-1,1)))
        out.append([np.mean(ang) if ang else np.nan,np.std(ang) if ang else np.nan])
    return FeatureOutput(np.asarray(out),('mean','std'))
@_reg('geometry.projections.projection','projections',output_kind='vector',cost=Cost.MEDIUM,tags=('pca','projection'))
def projection(points,n_components=3,**kwargs):
    p=_points(points);q=p-p.mean(0);_,_,vt=np.linalg.svd(q,full_matrices=False);k=min(n_components,vt.shape[0]);z=q@vt[:k].T;return FeatureOutput(z,tuple(f'pc_{i:02d}' for i in range(k)))

# Hull.
def _hull(points):
    from scipy.spatial import ConvexHull
    p=_points(points);return p,ConvexHull(p)
@_reg('geometry.hull.convex_hull','hull',output_kind='vector',tags=('convex-hull',))
def hull_bundle(points,**kwargs):
    p,h=_hull(points);mask=np.zeros(len(p));mask[h.vertices]=1;dist=hull_distance(points);return FeatureOutput(np.column_stack([mask,dist]),('is_vertex','distance_to_boundary'))
@_reg('geometry.hull.hull_area','hull',tags=('convex-hull','area'))
def hull_area(points,**kwargs):
    p,h=_hull(points);# scipy .area is perimeter in 2D, surface area in d>2
    val=h.volume if p.shape[1]==2 else h.area;return np.full(len(p),val,float)
@_reg('geometry.hull.hull_volume','hull',tags=('convex-hull','volume'))
def hull_volume(points,**kwargs):
    p,h=_hull(points);return np.full(len(p),h.volume,float)
@_reg('geometry.hull.solidity','hull',tags=('solidity',))
def hull_solidity(points,**kwargs):
    p,h=_hull(points);bbox=np.prod(np.ptp(p,axis=0));return np.full(len(p),h.volume/(bbox+1e-15),float)
@_reg('geometry.hull.hull_distance','hull',cost=Cost.MEDIUM,tags=('hull-distance',))
def hull_distance(points,**kwargs):
    p,h=_hull(points);# equations: normal*x + offset <= 0, normalized normals
    vals=h.equations[:,:-1]@p.T+h.equations[:,-1,None];return -np.max(vals,axis=0)

# Voronoi.
def _vor(points):
    from scipy.spatial import Voronoi
    p=_points(points);return p,Voronoi(p)
@_reg('geometry.voronoi.neighbor_count','voronoi',cost=Cost.HIGH,tags=('voronoi',))
def vor_neighbors(points,**kwargs):
    p,v=_vor(points);sets=[set() for _ in p]
    for a,b in v.ridge_points:sets[a].add(b);sets[b].add(a)
    return np.asarray([len(s) for s in sets],float)

def _polygon_area(poly):
    x=poly[:,0];y=poly[:,1];return .5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def _polygon_perimeter(poly):return np.sum(np.linalg.norm(poly-np.roll(poly,1,axis=0),axis=1))
@_reg('geometry.voronoi.cell_area','voronoi',cost=Cost.HIGH,tags=('voronoi','area'))
def vor_area(points,**kwargs):
    p,v=_vor(points)
    if p.shape[1]!=2:raise ValueError('cell_area is for 2D Voronoi cells')
    out=[]
    for i in range(len(p)):
        region=v.regions[v.point_region[i]];out.append(_polygon_area(v.vertices[region]) if region and -1 not in region else np.nan)
    return np.asarray(out)
@_reg('geometry.voronoi.boundary_length','voronoi',cost=Cost.HIGH,tags=('voronoi','perimeter'))
def vor_boundary(points,**kwargs):
    p,v=_vor(points)
    if p.shape[1]!=2:raise ValueError('boundary_length is for 2D Voronoi cells')
    out=[]
    for i in range(len(p)):
        region=v.regions[v.point_region[i]];out.append(_polygon_perimeter(v.vertices[region]) if region and -1 not in region else np.nan)
    return np.asarray(out)
@_reg('geometry.voronoi.cell_volume','voronoi',cost=Cost.EXTREME,tags=('voronoi','volume'))
def vor_volume(points,**kwargs):
    from scipy.spatial import ConvexHull
    p,v=_vor(points);out=[]
    for i in range(len(p)):
        region=v.regions[v.point_region[i]]
        if not region or -1 in region:out.append(np.nan);continue
        try:out.append(ConvexHull(v.vertices[region]).volume)
        except Exception:out.append(np.nan)
    return np.asarray(out)

# Proximity graphs.
def _pairwise_graph(points,kind):
    from scipy.spatial.distance import cdist
    from scipy.sparse.csgraph import minimum_spanning_tree
    p=_points(points);D=cdist(p,p);n=len(p);adj=np.zeros((n,n),bool)
    if kind=='mst':
        M=minimum_spanning_tree(D).toarray();adj=(M+M.T)>0
    else:
        for i in range(n):
            for j in range(i+1,n):
                dij=D[i,j]
                if kind=='gabriel':
                    mid=(p[i]+p[j])/2;rad=dij/2;ok=np.all(np.linalg.norm(p-mid,axis=1)[[k for k in range(n) if k not in (i,j)]]>=rad-1e-12)
                else: # relative neighborhood graph
                    others=[k for k in range(n) if k not in (i,j)];ok=all(max(D[i,k],D[j,k])>=dij-1e-12 for k in others)
                if ok:adj[i,j]=adj[j,i]=True
    return p,D,adj
for _name,_kind in [('gabriel_graph','gabriel'),('relative_neighborhood_graph','rng'),('minimum_spanning_tree','mst')]:
    @_reg(f'geometry.proximity.{_name}','proximity',output_kind='vector',cost=Cost.EXTREME,tags=('proximity-graph',_kind))
    def _prox(points,_kind=_kind,**kwargs):
        p,D,A=_pairwise_graph(points,_kind);out=[]
        for i in range(len(p)):
            ds=D[i,A[i]];out.append([A[i].sum(),ds.mean() if len(ds) else np.nan,ds.max() if len(ds) else np.nan])
        return FeatureOutput(np.asarray(out),('degree','edge_length_mean','edge_length_max'))

@_reg('geometry.alpha_shapes.alpha_shape','alpha_shapes',output_kind='vector',cost=Cost.HIGH,tags=('alpha-shape',))
def alpha_shape(points,alpha=None,**kwargs):
    p,tri,neigh,incident,metrics=_tri_data(points)
    if p.shape[1]!=2:raise ValueError('native alpha-shape implementation currently supports 2D')
    radii=np.asarray([m['circumradius'] for m in metrics]);thr=np.nanmedian(radii) if alpha is None else 1/max(alpha,1e-12);keep=radii<=thr;edge_count=Counter()
    for simplex,k in zip(tri.simplices,keep):
        if not k:continue
        for e in itertools.combinations(sorted(map(int,simplex)),2):edge_count[e]+=1
    boundary=np.zeros(len(p));degree=np.zeros(len(p))
    for (a,b),c in edge_count.items():
        degree[a]+=1;degree[b]+=1
        if c==1:boundary[a]=boundary[b]=1
    return FeatureOutput(np.column_stack([boundary,degree]),('is_boundary','alpha_degree'))

@_reg('geometry.stochastic_geometry.point_process','stochastic_geometry',output_kind='vector',cost=Cost.MEDIUM,tags=('point-process','density'))
def point_process(points,k=6,**kwargs):
    from scipy.spatial import cKDTree
    p=_points(points);d,idx=cKDTree(p).query(p,k=min(k+1,len(p)));d=np.atleast_2d(d);nn=d[:,1] if d.shape[1]>1 else np.nan;rk=d[:,-1];dim=p.shape[1];density=(min(k,len(p)-1))/(np.maximum(rk,1e-12)**dim);return FeatureOutput(np.column_stack([nn,rk,density]),('nearest_neighbor','k_radius','local_density'))


def delaunay_degree(points):
    pts,tri,neigh,incident,metrics=_tri_data(points)
    return np.asarray([len(n) for n in neigh],dtype=float)

def mean_edge_length(points):
    pts,tri,neigh,incident,metrics=_tri_data(points)
    return np.asarray([np.mean(_edge_values(pts,neigh,i)) if len(neigh[i]) else np.nan for i in range(len(pts))],dtype=float)

# Backwards-compatible grouped helper retained for users of the prototype API.
@dataclass(frozen=True)
class DelaunaySummary:
    degree: np.ndarray
    mean_edge_length: np.ndarray

def delaunay_summary(points: np.ndarray) -> DelaunaySummary:
    return DelaunaySummary(
        degree=np.asarray(delaunay_degree(points),dtype=float),
        mean_edge_length=np.asarray(mean_edge_length(points),dtype=float),
    )
