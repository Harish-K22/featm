from __future__ import annotations

from ...contracts import forward_parameters

from collections import Counter
import numpy as np

from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import content_seed, shannon_entropy_prob


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=('networkx',),cache_key=None):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        # Fifth audit pass, Finding 1: every graph.* feature's per-row
        # sample is a real pre-loaded networkx.Graph object, only
        # representable by an in-memory PandasSource -- see
        # BatchSemantics.OPAQUE_OBJECT_SAMPLE.
        return register_feature(name=name,family=family,modality='graph',input_types=('graph',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,requires=requires,description=name,cache_key=cache_key,batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE)(fn)
    return deco


def _nx():
    try: import networkx as nx
    except ImportError as exc: raise ImportError('Install featm[graph] (networkx)') from exc
    return nx


def _graphs(x):
    nx=_nx()
    if isinstance(x,(nx.Graph,nx.DiGraph,nx.MultiGraph,nx.MultiDiGraph)):return [x]
    return list(x)


def _summary_dict(d):
    v=np.asarray(list(d.values()),float)
    return [np.mean(v) if len(v) else 0,np.std(v) if len(v) else 0,np.max(v) if len(v) else 0,np.min(v) if len(v) else 0]


def _dict_feature(x,fn):return FeatureOutput(np.asarray([_summary_dict(fn(g)) for g in _graphs(x)]),('mean','std','max','min'))

@_reg('graph.size.n_nodes','size',cost=Cost.TINY,tags=('size',))
def n_nodes(x,**kwargs):return np.asarray([g.number_of_nodes() for g in _graphs(x)],float)
@_reg('graph.size.n_edges','size',cost=Cost.TINY,tags=('size',))
def n_edges(x,**kwargs):return np.asarray([g.number_of_edges() for g in _graphs(x)],float)
@_reg('graph.degree.average_degree','degree',cost=Cost.TINY,tags=('degree',))
def avg_degree(x,**kwargs):return np.asarray([np.mean([d for _,d in g.degree()]) if g.number_of_nodes() else 0 for g in _graphs(x)],float)
@_reg('graph.connectivity.density','connectivity',cost=Cost.TINY,tags=('density',))
def density(x,**kwargs):
    nx=_nx();return np.asarray([nx.density(g) for g in _graphs(x)],float)
@_reg('graph.connectivity.components','connectivity',cost=Cost.LOW,tags=('components',))
def components(x,**kwargs):
    nx=_nx();return np.asarray([nx.number_weakly_connected_components(g) if g.is_directed() else nx.number_connected_components(g) for g in _graphs(x)],float)

# Centralities as distribution summaries.
CENTRAL = {
'degree':lambda nx,g:nx.degree_centrality(g),
'betweenness':lambda nx,g:nx.betweenness_centrality(g),
'closeness':lambda nx,g:nx.closeness_centrality(g),
'harmonic':lambda nx,g:nx.harmonic_centrality(g),
'pagerank':lambda nx,g:nx.pagerank(g),
'eigenvector':lambda nx,g:nx.eigenvector_centrality_numpy(g) if g.number_of_nodes()>1 else {n:0 for n in g},
'katz':lambda nx,g:nx.katz_centrality_numpy(g,alpha=0.01,beta=1.0) if g.number_of_nodes()>1 else {n:0 for n in g},
}
for _nm,_fn in CENTRAL.items():
    @_reg(f'graph.centrality.{_nm}','centrality',output_kind='vector',cost=Cost.HIGH if _nm in ('betweenness','eigenvector','katz') else Cost.MEDIUM,tags=('centrality',_nm))
    def _cent(x,_fn=_fn,**kwargs):
        nx=_nx();return FeatureOutput(np.asarray([_summary_dict(_fn(nx,g)) for g in _graphs(x)]),('mean','std','max','min'))

@_reg('graph.centrality.current_flow','centrality',output_kind='vector',cost=Cost.EXTREME,tags=('current-flow',))
def current_flow(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=g.to_undirected()
        if not nx.is_connected(h) or len(h)<3:out.append([np.nan]*4);continue
        try:d=nx.current_flow_betweenness_centrality(h);out.append(_summary_dict(d))
        except Exception:out.append([np.nan]*4)
    return FeatureOutput(np.asarray(out),('mean','std','max','min'))

# Clustering.
@_reg('graph.clustering.average_clustering','clustering',cost=Cost.MEDIUM,tags=('clustering',))
def avg_clust(x,**kwargs):
    nx=_nx();return np.asarray([nx.average_clustering(g.to_undirected()) if len(g) else 0 for g in _graphs(x)],float)
@_reg('graph.clustering.transitivity','clustering',cost=Cost.MEDIUM,tags=('transitivity',))
def transitivity(x,**kwargs):
    nx=_nx();return np.asarray([nx.transitivity(g.to_undirected()) for g in _graphs(x)],float)
@_reg('graph.clustering.coefficient','clustering',output_kind='vector',cost=Cost.MEDIUM,tags=('clustering',))
def coefficient(x,**kwargs):
    nx=_nx();return FeatureOutput(np.asarray([_summary_dict(nx.clustering(g.to_undirected())) for g in _graphs(x)]),('mean','std','max','min'))
@_reg('graph.clustering.triangles','clustering',output_kind='vector',cost=Cost.MEDIUM,tags=('triangles',))
def triangles(x,**kwargs):
    nx=_nx();return FeatureOutput(np.asarray([_summary_dict(nx.triangles(g.to_undirected())) for g in _graphs(x)]),('node_mean','node_std','node_max','node_min'))

# Connectivity / cuts.
@_reg('graph.connectivity.bridges','connectivity',cost=Cost.MEDIUM,tags=('bridges',))
def bridges(x,**kwargs):
    nx=_nx();return np.asarray([len(list(nx.bridges(g.to_undirected()))) for g in _graphs(x)],float)
@_reg('graph.connectivity.articulation','connectivity',cost=Cost.MEDIUM,tags=('articulation',))
def articulation(x,**kwargs):
    nx=_nx();return np.asarray([len(list(nx.articulation_points(g.to_undirected()))) for g in _graphs(x)],float)

# Paths.
def _connected_undirected(g):
    nx=_nx();h=g.to_undirected()
    if not len(h):return h
    if nx.is_connected(h):return h
    nodes=max(nx.connected_components(h),key=len);return h.subgraph(nodes).copy()

for _nm,_fn in [
    ('diameter',lambda nx,h:nx.diameter(h) if len(h)>1 else 0),('radius',lambda nx,h:nx.radius(h) if len(h)>1 else 0),
    ('efficiency',lambda nx,h:nx.global_efficiency(h) if len(h)>1 else 0),
]:
    @_reg(f'graph.paths.{_nm}','paths',cost=Cost.HIGH,tags=('paths',_nm))
    def _path(x,_fn=_fn,**kwargs):
        nx=_nx();return np.asarray([_fn(nx,_connected_undirected(g)) for g in _graphs(x)],float)

@_reg('graph.paths.eccentricity','paths',output_kind='vector',cost=Cost.HIGH,tags=('eccentricity',))
def eccentricity(x,**kwargs):
    nx=_nx();return FeatureOutput(np.asarray([_summary_dict(nx.eccentricity(_connected_undirected(g))) if len(g) else [0]*4 for g in _graphs(x)]),('mean','std','max','min'))
@_reg('graph.paths.shortest_paths','paths',output_kind='vector',cost=Cost.HIGH,tags=('shortest-path',))
def shortest_paths(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=_connected_undirected(g);vals=[]
        for _,d in nx.all_pairs_shortest_path_length(h):vals.extend(v for k,v in d.items() if v>0)
        v=np.asarray(vals,float);out.append([v.mean() if len(v) else 0,v.std() if len(v) else 0,v.max() if len(v) else 0])
    return FeatureOutput(np.asarray(out),('mean','std','max'))

# Cores.
@_reg('graph.cores.k_core','cores',output_kind='vector',cost=Cost.MEDIUM,tags=('core',))
def kcore(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=g.to_undirected();d=nx.core_number(h) if len(h) and h.number_of_edges() else {n:0 for n in h};out.append(_summary_dict(d))
    return FeatureOutput(np.asarray(out),('mean_core','std_core','max_core','min_core'))
@_reg('graph.cores.k_shell','cores',output_kind='vector',cost=Cost.MEDIUM,tags=('shell',))
def kshell(x,**kwargs):return kcore(x)
@_reg('graph.cores.onion_layers','cores',output_kind='vector',cost=Cost.MEDIUM,tags=('onion',))
def onion(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        try:d=nx.onion_layers(g.to_undirected());out.append(_summary_dict(d))
        except Exception:out.append([np.nan]*4)
    return FeatureOutput(np.asarray(out),('mean_layer','std_layer','max_layer','min_layer'))
@_reg('graph.cores.k_truss','cores',output_kind='vector',cost=Cost.HIGH,tags=('truss',))
def ktruss(x,max_k=8,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=g.to_undirected();max_found=2;size=0
        for k in range(3,max_k+1):
            try:t=nx.k_truss(h,k);sz=t.number_of_edges()
            except Exception:break
            if sz: max_found=k;size=sz
            else:break
        out.append([max_found,size])
    return FeatureOutput(np.asarray(out),('max_k','edges_at_max_k'))

# Cycles / cliques / graphlets / motifs.
@_reg('graph.cycles.cycle_count','cycles',cost=Cost.HIGH,tags=('cycles',))
def cycle_count(x,**kwargs):
    nx=_nx();return np.asarray([len(nx.cycle_basis(g.to_undirected())) for g in _graphs(x)],float)
@_reg('graph.cycles.girth','cycles',cost=Cost.HIGH,tags=('girth',))
def girth(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        try:out.append(nx.girth(g.to_undirected()))
        except Exception:out.append(np.nan)
    return np.asarray(out,float)
@_reg('graph.cliques.cliques','cliques',output_kind='vector',cost=Cost.HIGH,tags=('clique',))
def cliques(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        cs=list(nx.find_cliques(g.to_undirected()));sizes=[len(c) for c in cs];out.append([len(cs),max(sizes) if sizes else 0,np.mean(sizes) if sizes else 0])
    return FeatureOutput(np.asarray(out),('maximal_clique_count','clique_number','clique_size_mean'))
@_reg('graph.graphlets.graphlets','graphlets',output_kind='vector',cost=Cost.HIGH,tags=('graphlets',))
def graphlets(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=g.to_undirected();tri=sum(nx.triangles(h).values())/3;wedges=sum(d*(d-1)/2 for _,d in h.degree())-3*tri;q4=sum(1 for c in nx.enumerate_all_cliques(h) if len(c)==4);cycles4=0
        # Count distinct chord-agnostic 4-cycles approximately from common neighbors.
        nodes=list(h)
        for i,u in enumerate(nodes):
            Nu=set(h[u])
            for v in nodes[i+1:]:
                if u==v:continue
                c=len(Nu & set(h[v]));cycles4+=c*(c-1)/2
        cycles4/=2
        out.append([h.number_of_edges(),wedges,tri,q4,cycles4])
    return FeatureOutput(np.asarray(out),('edges','wedges','triangles','four_cliques','four_cycles'))
@_reg('graph.motifs.motifs','motifs',output_kind='vector',cost=Cost.HIGH,tags=('motifs',))
def motifs(x,**kwargs):
    nx=_nx();names=['003','012','102','021D','021U','021C','111D','111U','030T','030C','201','120D','120U','120C','210','300'];out=[]
    for g in _graphs(x):
        if g.is_directed():d=nx.triadic_census(g);out.append([d[k] for k in names])
        else:
            # Undirected 3-node induced patterns: empty, one-edge, wedge, triangle.
            nodes=list(g);cnt=[0,0,0,0]
            import itertools
            for c in itertools.combinations(nodes,3):
                e=g.subgraph(c).number_of_edges();cnt[e if e<3 else 3]+=1
            out.append(cnt+[0]*12)
    return FeatureOutput(np.asarray(out),tuple(names))

# Community.
@_reg('graph.community.louvain','community',output_kind='vector',cost=Cost.HIGH,tags=('community','louvain'))
def louvain(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        cs=nx.community.louvain_communities(g.to_undirected(),seed=0) if len(g) else [];sizes=[len(c) for c in cs];out.append([len(cs),np.mean(sizes) if sizes else 0,max(sizes) if sizes else 0])
    return FeatureOutput(np.asarray(out),('community_count','size_mean','size_max'))
@_reg('graph.community.modularity','community',cost=Cost.HIGH,tags=('modularity',))
def modularity(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        cs=nx.community.louvain_communities(g.to_undirected(),seed=0) if len(g) else [];out.append(nx.community.modularity(g.to_undirected(),cs) if cs and g.number_of_edges() else 0)
    return np.asarray(out,float)
@_reg('graph.community.leiden','community',output_kind='vector',cost=Cost.HIGH,tags=('community','leiden'),requires=('igraph','leidenalg'))
def leiden(x,**kwargs):
    try:import igraph as ig;import leidenalg
    except ImportError as exc:raise ImportError('Install igraph and leidenalg for graph.community.leiden') from exc
    out=[]
    for g in _graphs(x):
        nodes=list(g);idx={n:i for i,n in enumerate(nodes)};ig_g=ig.Graph(n=len(nodes),edges=[(idx[u],idx[v]) for u,v in g.edges()],directed=g.is_directed());p=leidenalg.find_partition(ig_g,leidenalg.ModularityVertexPartition);sizes=p.sizes();out.append([len(sizes),np.mean(sizes) if sizes else 0,max(sizes) if sizes else 0,p.quality()])
    return FeatureOutput(np.asarray(out),('community_count','size_mean','size_max','quality'))

# Assortativity/mixing/homophily/reciprocity.
@_reg('graph.assortativity.degree','assortativity',cost=Cost.MEDIUM,tags=('assortativity',))
def assort(x,**kwargs):
    nx=_nx();return np.asarray([nx.degree_assortativity_coefficient(g) if g.number_of_edges() else np.nan for g in _graphs(x)],float)
@_reg('graph.reciprocity.reciprocity','reciprocity',cost=Cost.LOW,tags=('reciprocity',))
def reciprocity(x,**kwargs):
    nx=_nx();return np.asarray([nx.reciprocity(g) if g.is_directed() and g.number_of_edges() else (1.0 if not g.is_directed() and g.number_of_edges() else 0.0) for g in _graphs(x)],float)
@_reg('graph.homophily.homophily','homophily',cost=Cost.LOW,tags=('homophily',))
def homophily(x,label_attr='label',**kwargs):
    out=[]
    for g in _graphs(x):
        vals=[float(g.nodes[u].get(label_attr)==g.nodes[v].get(label_attr)) for u,v in g.edges() if label_attr in g.nodes[u] and label_attr in g.nodes[v]];out.append(np.mean(vals) if vals else np.nan)
    return np.asarray(out,float)
@_reg('graph.mixing.mixing','mixing',output_kind='vector',cost=Cost.MEDIUM,tags=('mixing',))
def mixing(x,attribute='label',**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        try:M=nx.attribute_mixing_matrix(g,attribute,normalized=True);ent=float(shannon_entropy_prob(M.ravel()));diag=float(np.trace(M));out.append([diag,ent,M.shape[0]])
        except Exception:out.append([np.nan]*3)
    return FeatureOutput(np.asarray(out),('diagonal_mass','entropy','category_count'))

# Link prediction aggregate over nonedges (sampled on large graphs).
@_reg('graph.link_prediction.common_neighbors','link_prediction',output_kind='vector',cost=Cost.HIGH,tags=('link-prediction',))
def common_neighbors(x,max_pairs=5000,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=g.to_undirected();pairs=list(nx.non_edges(h));
        if len(pairs)>max_pairs:pairs=pairs[:max_pairs]
        vals=[len(list(nx.common_neighbors(h,u,v))) for u,v in pairs];v=np.asarray(vals,float);out.append([v.mean() if len(v) else 0,v.std() if len(v) else 0,v.max() if len(v) else 0])
    return FeatureOutput(np.asarray(out),('mean','std','max'))

# Spectral.
def _lap_eigs_full_impl(g):
    nx=_nx();h=g.to_undirected();n=len(h)
    if n==0:return np.array([])
    L=nx.laplacian_matrix(h).astype(float).toarray();return np.sort(np.linalg.eigvalsh(L))

def _lap_eigs_full(g):
    """Full Laplacian eigendecomposition shared by every ``graph.spectral.*`` feature.

    Memoized per execution scope, keyed by graph object identity (safe: a
    scope never outlives the batch it was built from). Different callers
    requesting different ``k`` truncations of the spectrum reuse the same
    decomposition instead of recomputing it.
    """
    from ...cache import identity_cache_key, current_cache
    cache=current_cache()
    if cache is None:
        return _lap_eigs_full_impl(g)
    key=identity_cache_key('graph.laplacian_eigs', g)
    return cache.get_or_compute(key, lambda: _lap_eigs_full_impl(g))

def _lap_eigs(g,k=8):
    e=_lap_eigs_full(g);return e[:min(k,len(e))]
@_reg('graph.spectral.eigenvalues','spectral',output_kind='vector',cost=Cost.HIGH,tags=('spectral',),cache_key='graph.laplacian_eigs')
def eigvals(x,k=8,**kwargs):
    out=[]
    for g in _graphs(x):
        e=_lap_eigs(g,k);out.append(np.pad(e,(0,k-len(e)),constant_values=np.nan))
    return FeatureOutput(np.asarray(out),tuple(f'lambda_{i:02d}' for i in range(k)))
@_reg('graph.spectral.laplacian','spectral',output_kind='vector',cost=Cost.HIGH,tags=('laplacian',),cache_key='graph.laplacian_eigs')
def laplacian(x,**kwargs):return eigvals(x,**kwargs)
laplacian.__featm_parameters__ = forward_parameters(eigvals, positional=1, bound=[])
@_reg('graph.spectral.algebraic_connectivity','spectral',cost=Cost.HIGH,tags=('algebraic-connectivity',),cache_key='graph.laplacian_eigs')
def alg_conn(x,**kwargs):
    out=[]
    for g in _graphs(x):e=_lap_eigs(g,3);out.append(e[1] if len(e)>1 else 0)
    return np.asarray(out,float)
@_reg('graph.spectral.spectral_gap','spectral',cost=Cost.HIGH,tags=('spectral-gap',))
def spectral_gap(x,**kwargs):
    out=[]
    for g in _graphs(x):
        A=_nx().to_numpy_array(g.to_undirected(),dtype=float);e=np.sort(np.abs(np.linalg.eigvalsh(A)))[::-1] if len(A) else np.array([]);out.append(e[0]-e[1] if len(e)>1 else 0)
    return np.asarray(out,float)

# Topology: clique-complex simplex counts and graph cycle-rank Betti summaries.
@_reg('graph.topology.clique_complex','topology',output_kind='vector',cost=Cost.HIGH,tags=('clique-complex',))
def clique_complex(x,max_dim=3,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        counts=np.zeros(max_dim+1,float)
        for c in nx.enumerate_all_cliques(g.to_undirected()):
            d=len(c)-1
            if d<=max_dim:counts[d]+=1
            else:break
        out.append(counts)
    return FeatureOutput(np.asarray(out),tuple(f'simplices_dim_{d}' for d in range(max_dim+1)))
@_reg('graph.topology.persistent_homology','topology',output_kind='vector',cost=Cost.HIGH,tags=('betti','topology'))
def graph_ph(x,**kwargs):
    nx=_nx();out=[]
    for g in _graphs(x):
        h=g.to_undirected();c=nx.number_connected_components(h) if len(h) else 0;beta1=h.number_of_edges()-h.number_of_nodes()+c;tri=sum(nx.triangles(h).values())/3;out.append([c,beta1,tri])
    return FeatureOutput(np.asarray(out),('beta0','cycle_rank_beta1','filled_triangles'))

# Embeddings. Node2Vec/DeepWalk require gensim or node2vec; aggregate node vectors
# into one graph vector so variable graph sizes still produce a fixed dataset.
@_reg('graph.embeddings.node2vec','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('node2vec','embedding'),requires=('node2vec',))
def node2vec(x,dimensions=32,**kwargs):
    try:from node2vec import Node2Vec
    except ImportError as exc:raise ImportError("Install 'node2vec' for graph.embeddings.node2vec") from exc
    out=[]
    for g in _graphs(x):
        model=Node2Vec(g,dimensions=dimensions,walk_length=20,num_walks=20,workers=1,seed=0,quiet=True).fit(window=5,min_count=1,batch_words=64);vs=np.array([model.wv[str(n)] for n in g.nodes()]);out.append(vs.mean(0) if len(vs) else np.zeros(dimensions))
    return FeatureOutput(np.asarray(out),tuple(f'dim_{i:03d}' for i in range(dimensions)))
def _graph_content_seed(g) -> int:
    """Deterministic seed derived only from a graph's own topology (node labels + edge set).

    Order-independent (sorted node labels; edge pairs sorted, and each pair
    itself sorted when undirected so (u, v) and (v, u) are identical) and
    direction-aware. See :func:`content_seed`'s docstring for why this
    matters: a stochastic per-graph feature (DeepWalk's random-walk
    generation) must seed its own RNG from something that depends only on
    "this logical sample's content", not on where it appears in a call or
    batch -- fifth audit pass, Finding 2.
    """
    node_part=sorted(str(n) for n in g.nodes())
    if g.is_directed():
        edge_part=sorted((str(u),str(v)) for u,v in g.edges())
    else:
        edge_part=sorted(tuple(sorted((str(u),str(v)))) for u,v in g.edges())
    canonical="N:"+"\x1f".join(node_part)+"|E:"+"\x1f".join(f"{u}\x1e{v}" for u,v in edge_part)
    return content_seed(canonical.encode("utf-8",errors="surrogatepass"))

@_reg('graph.embeddings.deepwalk','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('deepwalk','embedding'),requires=('gensim',))
def deepwalk(x,dimensions=32,walk_length=20,num_walks=20,**kwargs):
    try:from gensim.models import Word2Vec
    except ImportError as exc:raise ImportError("Install 'gensim' for graph.embeddings.deepwalk") from exc
    out=[]
    for g in _graphs(x):
        # Fifth audit pass, Finding 2: a fresh RNG seeded from THIS graph's
        # own content (not one shared `rng` created outside the loop, the
        # pre-fix bug -- its position in the draw sequence depended on how
        # many OTHER graphs preceded it in the same call, so the same graph
        # silently produced a different embedding depending on which batch
        # it landed in). Same content -> same seed -> same walks -> same
        # embedding, regardless of batch partitioning or call position.
        rng=np.random.default_rng(_graph_content_seed(g))
        walks=[];nodes=list(g)
        for _ in range(num_walks):
            for s in nodes:
                cur=s;w=[str(cur)]
                for _ in range(walk_length-1):
                    ns=list(g.neighbors(cur))
                    if not ns:break
                    cur=ns[int(rng.integers(len(ns)))];w.append(str(cur))
                walks.append(w)
        m=Word2Vec(walks,vector_size=dimensions,window=5,min_count=1,workers=1,sg=1,seed=0);vs=np.array([m.wv[str(n)] for n in nodes]);out.append(vs.mean(0) if len(vs) else np.zeros(dimensions))
    return FeatureOutput(np.asarray(out),tuple(f'dim_{i:03d}' for i in range(dimensions)))
@_reg('graph.embeddings.gnn','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('gnn','embedding'),requires=('user-model',))
def gnn_embedding(x,model=None,**kwargs):
    if model is None:raise ValueError('graph.embeddings.gnn requires an explicit user-supplied callable model(graph)->1D embedding')
    rows=[np.asarray(model(g),float).ravel() for g in _graphs(x)];width=len(rows[0]) if rows else 0
    if any(len(r)!=width for r in rows):raise ValueError('GNN model must return a fixed-width vector')
    return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:03d}' for i in range(width)))
