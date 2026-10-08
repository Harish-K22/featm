from __future__ import annotations

import numpy as np
from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature


def _reg(name,family,*,output_kind='scalar',tags=()):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        # Fourth audit pass, Finding 4/13: every multimodal.* feature's
        # convention is "one call = a dict/tuple/3D-array pair of embedding
        # matrices" (see _pair() below) -- a structured per-sample input a
        # row-oriented Source batch cannot represent (it would misinterpret
        # a 2-row batch as the pair itself, exactly the cross-series bug).
        # All multimodal features are STRUCTURED_SAMPLE; none of them are
        # currently row-independent from a row-oriented Source.
        return register_feature(name=name,family=family,modality='multimodal',input_types=('embeddings',),cost=Cost.LOW,streamable=True,output_kind=output_kind,tags=tags,description=name,batch_semantics=BatchSemantics.STRUCTURED_SAMPLE)(fn)
    return deco

def _pair(x,keys=None):
    if isinstance(x,dict):
        if keys and all(k in x for k in keys):a,b=x[keys[0]],x[keys[1]]
        else:
            vals=list(x.values());a,b=vals[0],vals[1]
    elif isinstance(x,(tuple,list)) and len(x)==2:a,b=x
    else:
        z=np.asarray(x,float)
        if z.ndim==3 and z.shape[1]==2:a,b=z[:,0],z[:,1]
        else:raise ValueError('multimodal features expect two embedding matrices')
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.ndim==1:a=a[None,:]
    if b.ndim==1:b=b[None,:]
    if len(a)!=len(b):raise ValueError('modalities must have same sample count')
    return a,b

def _align_dims(a,b):
    k=min(a.shape[1],b.shape[1]);return a[:,:k],b[:,:k]

def _cos(a,b):
    a,b=_align_dims(a,b);return np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-15)

@_reg('multimodal.similarity.cross_modal','similarity',tags=('cosine',))
def similarity(x,**kwargs):a,b=_pair(x);return _cos(a,b)
@_reg('multimodal.embedding_distances.cross_modal','embedding_distances',output_kind='vector',tags=('distance',))
def distances(x,**kwargs):
    a,b=_pair(x);a,b=_align_dims(a,b);d=a-b;return FeatureOutput(np.column_stack([np.linalg.norm(d,axis=1),np.sum(np.abs(d),1),1-_cos(a,b)]),('euclidean','manhattan','cosine_distance'))
for _nm,_keys in [('image_text',('image','text')),('audio_text',('audio','text')),('video_text',('video','text'))]:
    @_reg(f'multimodal.alignment.{_nm}','alignment',output_kind='vector',tags=('alignment',_nm))
    def _alignment(x,_keys=_keys,**kwargs):
        a,b=_pair(x,_keys);a,b=_align_dims(a,b);return FeatureOutput(np.column_stack([_cos(a,b),np.linalg.norm(a-b,axis=1)]),('cosine','euclidean'))

@_reg('multimodal.fusion.image_tabular','fusion',output_kind='vector',tags=('fusion','image-tabular'))
def image_tabular(x,**kwargs):
    a,b=_pair(x,('image','tabular'));z=np.column_stack([a,b]);return FeatureOutput(z,tuple(f'dim_{i:04d}' for i in range(z.shape[1])))
@_reg('multimodal.fusion.temporal_tabular','fusion',output_kind='vector',tags=('fusion','temporal-tabular'))
def temporal_tabular(x,**kwargs):
    a,b=_pair(x,('temporal','tabular'));z=np.column_stack([a,b]);return FeatureOutput(z,tuple(f'dim_{i:04d}' for i in range(z.shape[1])))
@_reg('multimodal.fusion.late_fusion','fusion',output_kind='vector',tags=('late-fusion',))
def late_fusion(x,weights=None,**kwargs):
    a,b=_pair(x);k=min(a.shape[1],b.shape[1]);w=(.5,.5) if weights is None else weights;z=w[0]*a[:,:k]+w[1]*b[:,:k];return FeatureOutput(z,tuple(f'dim_{i:04d}' for i in range(k)))
