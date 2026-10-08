"""Exercise every registered feature that can run with installed dependencies.

This is intentionally not a speed benchmark. It catches registry/API/shape errors
across the complete feature universe using small synthetic inputs.
"""
from __future__ import annotations

import inspect
import math
import numpy as np
import pandas as pd
import networkx as nx

import featm
from featm.registry import FeatureOutput

rng=np.random.default_rng(42)
series=np.vstack([np.sin(np.linspace(0,8*np.pi,128))+0.05*rng.normal(size=128) for _ in range(4)])
audio=np.vstack([np.sin(2*np.pi*220*np.arange(4096)/22050)+.01*rng.normal(size=4096) for _ in range(3)])
stereo=np.stack([audio[:2],np.roll(audio[:2],2,axis=1)],axis=-1)
images=(rng.random((3,32,32,3))*255).astype(np.uint8)
video=(rng.random((2,6,24,24,3))*255).astype(np.uint8)
points=rng.normal(size=(32,2))
clouds=[rng.normal(size=(40,3)),rng.normal(size=(35,3))+1]
matrices=rng.normal(size=(3,5,5))
sequences=['ababcaaab','xyzxyzxy','mississippi']
docs=['Feature engineering discovers useful mathematical structure.','Graphs, topology and signals can expose different patterns.','A small text document for testing feature extraction.']
probs=np.abs(rng.normal(size=(3,12)));probs/=probs.sum(1,keepdims=True)
paired_probs=np.stack([probs,np.roll(probs,1,axis=1)],axis=1)
pair_ts=np.stack([series[:3],np.roll(series[:3],3,axis=1)],axis=1)
triple_ts=np.stack([series[:3],np.roll(series[:3],2,axis=1),np.roll(series[:3],5,axis=1)],axis=1)
info_multi=rng.normal(size=(2,80,4))
spatial={'coordinates':points,'values':np.sin(points[:,0])+rng.normal(scale=.1,size=len(points))}
events=[np.cumsum(rng.exponential(1,size=30)),np.cumsum(rng.exponential(.6,size=35))]
survival={'durations':np.array([[1,2,3,5,8],[1,4,4,7,9]],float),'observed':np.array([[1,1,0,1,0],[1,0,1,1,1]],bool)}
emb_a=rng.normal(size=(4,16));emb_b=rng.normal(size=(4,16));multi={'image':emb_a,'text':emb_b,'audio':emb_a*.8,'video':emb_a*1.1,'tabular':rng.normal(size=(4,5)),'temporal':rng.normal(size=(4,7))}
graphs=[]
for i in range(3):
    g=nx.watts_strogatz_graph(18,4,.2,seed=i)
    for n in g:nx.set_node_attributes(g,{n:str(n%3)},'label')
    graphs.append(g)

class DummyVectors:
    vector_size=6
    def __getitem__(self,key):
        rr=np.random.default_rng(abs(hash(key))%(2**32));return rr.normal(size=6)

def dummy_model(x):
    a=np.asarray(x,float);return np.array([a.mean(),a.std(),a.min(),a.max()])


def tabular_args(spec):
    numeric=pd.Series(np.linspace(-2,3,64)+rng.normal(scale=.1,size=64))
    categorical=pd.Series(['a','b','a','c']*16)
    dt=pd.Series(pd.date_range('2026-01-01',periods=64,freq='h'))
    args=[]
    for typ in spec.input_types:
        if typ=='numeric':args.append(numeric)
        elif typ=='categorical':args.append(categorical)
        elif typ=='datetime':args.append(dt)
        elif typ=='geographic':args.append(points)
        else:args.append(numeric)
    return args


def call(spec):
    fn=spec.implementation
    kw={}
    if spec.modality=='tabular':
        if spec.family=='geographic':return fn(points,fit=False,state=None)
        args=tabular_args(spec);return fn(*args,fit=True,state=None)
    if spec.modality=='timeseries':
        x=triple_ts if ('conditional' in spec.name or 'interaction' in spec.name) else (pair_ts if spec.family=='cross_series' else series)
        return fn(x)
    if spec.modality=='signal':
        x=pair_ts if 'phase_locking' in spec.name else series
        return fn(x)
    if spec.modality=='audio':
        x=stereo if spec.family=='spatial' else audio
        return fn(x)
    if spec.modality=='image':return fn(images)
    if spec.modality=='video':
        if 'user-model' in spec.requires:kw['model']=dummy_model
        return fn(video,**kw)
    if spec.modality=='graph':
        if spec.name=='graph.embeddings.gnn':kw['model']=lambda g:np.array([len(g),g.number_of_edges(),nx.density(g)])
        return fn(graphs,**kw)
    if spec.modality=='geometry':return fn(points)
    if spec.modality=='pointcloud':
        if 'user-model' in spec.requires:kw['model']=dummy_model
        return fn(clouds,**kw)
    if spec.modality=='manifold':return fn(points)
    if spec.modality=='spatial':return fn(spatial)
    if spec.modality=='topology':return fn(clouds)
    if spec.modality=='matrix':return fn(matrices)
    if spec.modality=='sequence':return fn(sequences)
    if spec.modality=='text':
        if spec.name=='text.bm25.bm25':kw['query']='feature graphs'
        if spec.name=='text.embeddings.word':kw['model']=DummyVectors()
        return fn(docs,**kw)
    if spec.modality=='event':return fn(survival if spec.family=='survival' else events)
    if spec.modality=='multimodal':return fn(multi)
    if spec.modality=='information':
        if spec.family=='entropy':
            if spec.name.endswith('.joint') or spec.name.endswith('.conditional'):return fn(pair_ts)
            return fn(probs)
        if spec.family=='dependence':
            return fn(triple_ts if ('conditional' in spec.name or 'interaction' in spec.name) else pair_ts)
        if spec.family=='divergence':
            return fn(pair_ts if spec.name.endswith('variation_of_information') else paired_probs)
        if spec.family=='multivariate':return fn(info_multi)
        if spec.family=='pid':return fn(triple_ts)
        return fn(probs)
    raise RuntimeError(spec.modality)


def validate(spec,result):
    if spec.modality=='tabular' and isinstance(result,tuple):result=result[0]
    if isinstance(result,FeatureOutput):a=result.as_array()
    else:a=np.asarray(result)
    if a.ndim not in (1,2):raise AssertionError(f'bad ndim {a.ndim}')
    if a.shape[0]==0:raise AssertionError('empty output')
    return a.shape

def main() -> int:
    ok=[];optional=[];failed=[]
    for spec in featm.REGISTRY.all():
        try:
            result=call(spec);shape=validate(spec,result);ok.append((spec.name,shape))
        except (ImportError,ModuleNotFoundError) as exc:
            optional.append((spec.name,str(exc)))
        except Exception as exc:
            failed.append((spec.name,f'{type(exc).__name__}: {exc}'))

    print(f'total={len(featm.REGISTRY.all())} ok={len(ok)} optional_unavailable={len(optional)} failed={len(failed)}')
    if optional:
        print('\nOPTIONAL BACKENDS NOT INSTALLED:')
        for n,e in optional:print(' ',n,'->',e[:140])
    if failed:
        print('\nUNEXPECTED FAILURES:')
        for n,e in failed:print(' ',n,'->',e)
        return 1
    return 0


if __name__ == '__main__':
    # Required on Windows (and safe everywhere else): at least one registered
    # feature (signal.decomposition.eemd, via PyEMD's EEMD) spawns a
    # multiprocessing.Pool internally. multiprocessing's "spawn" start method
    # (Windows' default, and available elsewhere) re-imports this file as
    # __main__ in each worker process; without this guard the whole registry
    # loop above would re-run inside every worker, which would reach EEMD
    # again and spawn another generation of workers recursively.
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
