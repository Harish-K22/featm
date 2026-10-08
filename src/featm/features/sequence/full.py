from __future__ import annotations

from ...contracts import forward_parameters

from collections import Counter
import numpy as np
from ...registry import Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import compressed_ratio, content_seed, discrete_entropy, sequence_batch


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),**extra):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        return register_feature(name=name,family=family,modality='sequence',input_types=('sequence',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,description=name,**extra)(fn)
    return deco

def _tokens(s):return list(s) if not isinstance(s,str) else list(s)
def _runs(s):
    t=_tokens(s)
    if not t:return []
    out=[];n=1
    for a,b in zip(t,t[1:]):
        if a==b:n+=1
        else:out.append(n);n=1
    out.append(n);return out

@_reg('sequence.basic.length','basic',cost=Cost.TINY,tags=('length',))
def length(x,**kwargs):return np.asarray([len(s) for s in sequence_batch(x)],float)
@_reg('sequence.basic.unique_ratio','basic',cost=Cost.TINY,tags=('unique',))
def unique_ratio(x,**kwargs):return np.asarray([len(set(_tokens(s)))/max(len(_tokens(s)),1) for s in sequence_batch(x)],float)
@_reg('sequence.entropy.shannon','entropy',cost=Cost.TINY,tags=('entropy',))
def entropy(x,**kwargs):return np.asarray([discrete_entropy(_tokens(s)) for s in sequence_batch(x)],float)
@_reg('sequence.entropy.shannon_entropy','entropy',cost=Cost.TINY,tags=('entropy',))
def entropy_alias(x,**kwargs):return entropy(x)
@_reg('sequence.compression.compression','compression',cost=Cost.LOW,tags=('compression',))
def compression(x,**kwargs):return np.asarray([compressed_ratio(s) for s in sequence_batch(x)],float)
@_reg('sequence.run_length.run_count','run_length',cost=Cost.TINY,tags=('runs',))
def run_count(x,**kwargs):return np.asarray([len(_runs(s)) for s in sequence_batch(x)],float)
@_reg('sequence.run_length.runs','run_length',output_kind='vector',cost=Cost.TINY,tags=('runs',))
def runs(x,**kwargs):
    out=[]
    for s in sequence_batch(x):r=np.asarray(_runs(s),float);out.append([len(r),r.mean() if len(r) else 0,r.std() if len(r) else 0,r.max() if len(r) else 0])
    return FeatureOutput(np.asarray(out),('count','length_mean','length_std','length_max'))
@_reg('sequence.transitions.transition_rate','transitions',cost=Cost.TINY,tags=('transition',))
def transition_rate(x,**kwargs):
    out=[]
    for s in sequence_batch(x):t=_tokens(s);out.append(np.mean([a!=b for a,b in zip(t,t[1:])]) if len(t)>1 else 0)
    return np.asarray(out)
@_reg('sequence.transitions.transition_matrix','transitions',output_kind='vector',cost=Cost.LOW,tags=('transition-matrix',))
def transition_matrix(x,buckets=8,**kwargs):
    out=[]
    for s in sequence_batch(x):
        t=_tokens(s);keys={v:i for i,v in enumerate(sorted(set(t),key=str)[:buckets])};M=np.zeros((buckets,buckets))
        for a,b in zip(t,t[1:]):
            if a in keys and b in keys:M[keys[a],keys[b]]+=1
        if M.sum():M/=M.sum()
        out.append(M.ravel())
    return FeatureOutput(np.asarray(out),tuple(f'{i}_{j}' for i in range(buckets) for j in range(buckets)))
def _ngram_bucket(tokens,buckets):
    # Step 4 completion pass: real multi-process Dask validation found the
    # builtin hash() here bucketed the SAME n-gram differently depending on
    # which worker PROCESS computed it (str hashing is randomized per
    # process by default) -- see featm.features.text.full._stable_bucket's
    # docstring for the full root-cause writeup; content_seed is the
    # established, process-independent replacement (already used for
    # DeepWalk/EEMD's deterministic seeding).
    return content_seed('\x1f'.join(str(tok) for tok in tokens).encode('utf-8'))%buckets

@_reg('sequence.ngrams.ngrams','ngrams',output_kind='vector',cost=Cost.LOW,tags=('ngrams',))
def ngrams(x,n=2,buckets=32,**kwargs):
    out=[]
    for s in sequence_batch(x):
        t=_tokens(s);v=np.zeros(buckets)
        for i in range(max(0,len(t)-n+1)):v[_ngram_bucket(t[i:i+n],buckets)]+=1
        if v.sum():v/=v.sum()
        out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'bucket_{i:02d}' for i in range(buckets)))
@_reg('sequence.markov.markov','markov',output_kind='vector',cost=Cost.LOW,tags=('markov',))
def markov(x,**kwargs):return transition_matrix(x,**kwargs)
markov.__featm_parameters__ = forward_parameters(transition_matrix, positional=1, bound=[])
@_reg('sequence.motifs.motifs','motifs',output_kind='vector',cost=Cost.MEDIUM,tags=('motifs',))
def motifs(x,max_len=4,**kwargs):
    out=[]
    for s in sequence_batch(x):
        t=_tokens(s);vals=[]
        for n in range(2,max_len+1):
            c=Counter(tuple(t[i:i+n]) for i in range(max(0,len(t)-n+1)));vals.extend([len(c),max(c.values()) if c else 0])
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(v for n in range(2,max_len+1) for v in (f'unique_{n}',f'max_count_{n}')))
@_reg('sequence.subsequences.subsequences','subsequences',output_kind='vector',cost=Cost.MEDIUM,tags=('subsequence',))
def subsequences(x,window=3,**kwargs):
    out=[]
    for s in sequence_batch(x):
        t=_tokens(s);subs=[tuple(t[i:i+window]) for i in range(max(0,len(t)-window+1))];c=Counter(subs);out.append([len(subs),len(c),max(c.values()) if c else 0])
    return FeatureOutput(np.asarray(out),('count','unique','max_frequency'))
@_reg('sequence.symbolic_dynamics.symbolic_dynamics','symbolic_dynamics',output_kind='vector',cost=Cost.MEDIUM,tags=('symbolic-dynamics',))
def symbolic(x,**kwargs):return FeatureOutput(np.column_stack([entropy(x),transition_rate(x),compression(x)]),('entropy','transition_rate','compression_ratio'))
@_reg('sequence.distance.edit_distance','distance',cost=Cost.HIGH,tags=('edit-distance',),required_params_for_batching=('reference',))
def edit_distance(x,reference=None,**kwargs):
    # Fourth audit pass, Finding 2: reference=None picks the first sequence
    # of *whatever collection this call receives* as the implicit anchor --
    # correct and convenient for a single small-data call, but silently
    # batch-local (a different "first sequence" per out-of-core batch) if
    # left implicit across multiple batches. required_params_for_batching
    # makes the out-of-core engine reject a multi-batch run that omits an
    # explicit reference=, rather than silently using a different reference
    # per batch.
    seqs=sequence_batch(x);ref=_tokens(reference) if reference is not None else (_tokens(seqs[0]) if seqs else [])
    def lev(a,b):
        prev=list(range(len(b)+1))
        for i,ca in enumerate(a,1):
            cur=[i]
            for j,cb in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
            prev=cur
        return prev[-1]
    return np.asarray([lev(_tokens(s),ref) for s in seqs],float)
@_reg('sequence.alignment.alignment','alignment',output_kind='vector',cost=Cost.HIGH,tags=('alignment',),required_params_for_batching=('reference',))
def alignment(x,reference=None,match=1,mismatch=-1,gap=-1,**kwargs):
    # Same implicit-first-sequence-reference issue as edit_distance above --
    # see Finding 2, fourth audit pass.
    seqs=sequence_batch(x);ref=_tokens(reference) if reference is not None else (_tokens(seqs[0]) if seqs else [])
    out=[]
    for s in seqs:
        a=_tokens(s);dp=np.zeros((len(a)+1,len(ref)+1),int);dp[:,0]=np.arange(len(a)+1)*gap;dp[0,:]=np.arange(len(ref)+1)*gap
        for i in range(1,len(a)+1):
            for j in range(1,len(ref)+1):dp[i,j]=max(dp[i-1,j-1]+(match if a[i-1]==ref[j-1] else mismatch),dp[i-1,j]+gap,dp[i,j-1]+gap)
        out.append([dp[-1,-1],dp[-1,-1]/max(len(a),len(ref),1)])
    return FeatureOutput(np.asarray(out),('score','normalized_score'))
