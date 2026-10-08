from __future__ import annotations

import numpy as np
from ...registry import Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import matrix_batch, shannon_entropy_prob


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=()):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        # matrix_batch() treats a plain 2D array as ONE matrix (not many independent row-samples);
        # the out-of-core engine can only ever hand this a 2D batch, so row-chunking would silently
        # compute properties (determinant, rank, eigenvalues, ...) of a different, wrong sub-matrix
        # per batch. Correct only when the whole matrix is available at once.
        return register_feature(name=name,family=family,modality='matrix',input_types=('matrix',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,description=name,requires_full_materialization=True)(fn)
    return deco

for _name,_fn in [
('determinant',lambda m:np.linalg.det(m) if m.shape[0]==m.shape[1] else np.nan),('trace',lambda m:np.trace(m)),('rank',lambda m:np.linalg.matrix_rank(m)),
('condition_number',lambda m:np.linalg.cond(m)),('frobenius_norm',lambda m:np.linalg.norm(m,'fro')),('spectral_norm',lambda m:np.linalg.norm(m,2)),('sparsity',lambda m:np.mean(np.isclose(m,0))),
]:
    @_reg(f'matrix.algebra.{_name}','algebra',cost=Cost.MEDIUM if _name in ('condition_number','rank') else Cost.TINY,tags=('linear-algebra',))
    def _alg(x,_fn=_fn,**kwargs):return np.asarray([_fn(m) for m in matrix_batch(x)],float)

@_reg('matrix.algebra.eigenvalues','algebra',output_kind='vector',cost=Cost.MEDIUM,tags=('eigenvalues',))
def eig(x,k=8,**kwargs):
    out=[]
    for m in matrix_batch(x):
        if m.shape[0]!=m.shape[1]:v=np.linalg.eigvalsh(m.T@m)
        else:
            try:v=np.linalg.eigvals(m)
            except Exception:v=np.array([])
        v=np.sort(np.abs(v))[::-1][:k];out.append(np.pad(v,(0,k-len(v)),constant_values=np.nan))
    return FeatureOutput(np.asarray(out),tuple(f'eig_{i:02d}' for i in range(k)))
@_reg('matrix.algebra.singular_values','algebra',output_kind='vector',cost=Cost.MEDIUM,tags=('svd',))
def singular(x,k=8,**kwargs):
    out=[]
    for m in matrix_batch(x):v=np.linalg.svd(m,compute_uv=False)[:k];out.append(np.pad(v,(0,k-len(v)),constant_values=np.nan))
    return FeatureOutput(np.asarray(out),tuple(f'sv_{i:02d}' for i in range(k)))
@_reg('matrix.algebra.norms','algebra',output_kind='vector',cost=Cost.LOW,tags=('norms',))
def norms(x,**kwargs):
    out=[]
    for m in matrix_batch(x):out.append([np.linalg.norm(m,'fro'),np.linalg.norm(m,1),np.linalg.norm(m,np.inf),np.linalg.norm(m,2)])
    return FeatureOutput(np.asarray(out),('frobenius','one','infinity','spectral'))
@_reg('matrix.information.matrix_entropy','information',cost=Cost.MEDIUM,tags=('von-neumann-entropy',))
def matrix_entropy(x,**kwargs):
    out=[]
    for m in matrix_batch(x):
        A=m@m.T;s=np.linalg.eigvalsh(A);out.append(float(shannon_entropy_prob(np.clip(s,0,None))))
    return np.asarray(out)
@_reg('matrix.spectral.spectral_statistics','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('spectrum',))
def spectral_stats(x,**kwargs):
    out=[]
    for m in matrix_batch(x):
        s=np.linalg.svd(m,compute_uv=False);p=s*s;out.append([s[0] if len(s) else 0,np.mean(s),np.std(s),float(shannon_entropy_prob(p)),np.sum(s>1e-12)])
    return FeatureOutput(np.asarray(out),('max_sv','mean_sv','std_sv','spectral_entropy','effective_rank_count'))
