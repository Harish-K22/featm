from __future__ import annotations

import numpy as np
from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=(),**extra):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        return register_feature(name=name,family=family,modality='event',input_types=('event_times',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,requires=requires,description=name,**extra)(fn)
    return deco

def _events(x):
    if isinstance(x,np.ndarray) and x.ndim==1:return [np.asarray(x,float)]
    return [np.sort(np.asarray(v,float)) for v in x]
def _intervals(e):return np.diff(e[np.isfinite(e)])

@_reg('event.timing.interarrival_time','timing',output_kind='vector',cost=Cost.TINY,tags=('interarrival',))
def interarrival(x,**kwargs):
    out=[]
    for e in _events(x):d=_intervals(e);out.append([np.mean(d) if len(d) else np.nan,np.std(d) if len(d) else np.nan,np.min(d) if len(d) else np.nan,np.max(d) if len(d) else np.nan])
    return FeatureOutput(np.asarray(out),('mean','std','min','max'))
@_reg('event.timing.event_rate','timing',cost=Cost.TINY,tags=('rate',))
def rate(x,**kwargs):
    out=[]
    for e in _events(x):out.append((len(e)-1)/(e[-1]-e[0]) if len(e)>1 and e[-1]>e[0] else 0)
    return np.asarray(out)
@_reg('event.timing.burstiness','timing',cost=Cost.TINY,tags=('burstiness',))
def burstiness(x,**kwargs):
    out=[]
    for e in _events(x):d=_intervals(e);mu=np.mean(d) if len(d) else 0;sd=np.std(d) if len(d) else 0;out.append((sd-mu)/(sd+mu+1e-15))
    return np.asarray(out)
@_reg('event.timing.recency','timing',cost=Cost.TINY,tags=('recency',),required_params_for_batching=('reference_time',))
def recency(x,reference_time=None,**kwargs):
    # Fourth audit pass, Finding 3: reference_time=None uses the maximum
    # last-event time across *whatever collection this call receives* --
    # correct for one small-data call, silently batch-local across multiple
    # out-of-core batches (each batch gets its own max). See sequence.py's
    # edit_distance/alignment for the same pattern.
    ev=_events(x);ref=max((e[-1] for e in ev if len(e)),default=0) if reference_time is None else reference_time;return np.asarray([ref-e[-1] if len(e) else np.nan for e in ev],float)
@_reg('event.sessions.session','sessions',output_kind='vector',cost=Cost.LOW,tags=('sessionization',))
def sessions(x,gap=None,**kwargs):
    out=[]
    for e in _events(x):
        d=_intervals(e);g=gap if gap is not None else (np.median(d)*5 if len(d) else 1);cuts=np.where(d>g)[0];bounds=np.r_[0,cuts+1,len(e)];sizes=np.diff(bounds);out.append([len(sizes),np.mean(sizes) if len(sizes) else 0,np.max(sizes) if len(sizes) else 0])
    return FeatureOutput(np.asarray(out),('session_count','events_per_session_mean','events_per_session_max'))
@_reg('event.point_process.temporal_point_process','point_process',output_kind='vector',cost=Cost.LOW,tags=('point-process',))
def point_process(x,**kwargs):
    ia=interarrival(x).values;return FeatureOutput(np.column_stack([rate(x),burstiness(x),ia[:,0],ia[:,1]]),('rate','burstiness','interarrival_mean','interarrival_std'))

# Univariate exponential Hawkes MLE: lambda(t)=mu + alpha*sum exp(-beta(t-ti)).
@_reg('event.process.hawkes','process',output_kind='vector',cost=Cost.HIGH,tags=('hawkes',),requires=('scipy',))
def hawkes(x,**kwargs):
    from scipy.optimize import minimize
    out=[]
    for e in _events(x):
        e=e[np.isfinite(e)]
        if len(e)<3:out.append([np.nan]*3);continue
        e=e-e[0];T=max(e[-1],1e-9)
        def nll(theta):
            mu,alpha,beta=np.exp(theta);past=0.;last=0.;ll=0.
            for i,t in enumerate(e):
                if i>0:past=np.exp(-beta*(t-last))*(past+1)
                lam=mu+alpha*past
                if lam<=0:return 1e12
                ll+=np.log(lam);last=t
            compens=mu*T+(alpha/beta)*np.sum(1-np.exp(-beta*(T-e)))
            return -(ll-compens)
        init=np.log([max(len(e)/T*.5,1e-3),.5,1/max(np.mean(np.diff(e)),1e-3)]);res=minimize(nll,init,method='L-BFGS-B');out.append(np.exp(res.x) if res.success else [np.nan]*3)
    return FeatureOutput(np.asarray(out),('mu','alpha','beta'))

@_reg(
    'event.survival.survival','survival',output_kind='vector',cost=Cost.LOW,tags=('kaplan-meier','structured'),
    batch_semantics=BatchSemantics.STRUCTURED_SAMPLE,
)
def survival(x,**kwargs):
    # This feature's whole-call convention is {"durations": 2D_array,
    # "observed": 2D_array} -- one call, N cohorts -- which a row-oriented
    # Source (rows of scalar/array columns) cannot represent at all; the
    # out-of-core whole-batch path only ever passes a numpy array or Series,
    # never a dict, so this already fails immediately and clearly (the
    # isinstance check below) rather than computing a wrong answer. Marked
    # STRUCTURED_SAMPLE explicitly (fourth audit pass, Finding 4/13) so this
    # is documented/auditable rather than "happens to fail safely by luck".
    # Accept {durations, observed}; each row in x is treated as one cohort when list-like.
    if not isinstance(x,dict):raise ValueError('event.survival.survival expects {"durations": ..., "observed": ...}')
    d=np.asarray(x['durations'],float);o=np.asarray(x.get('observed',np.ones_like(d)),bool)
    if d.ndim==1:d=d[None,:];o=o[None,:]
    out=[]
    for dur,obs in zip(d,o):
        order=np.argsort(dur);dur,obs=dur[order],obs[order];S=1.;median=np.nan;area=0.;prev=0.;n=len(dur)
        for t in np.unique(dur):
            area+=S*(t-prev);at=np.sum(dur>=t);events=np.sum((dur==t)&obs);S*=1-events/max(at,1);prev=t
            if np.isnan(median) and S<=.5:median=t
        out.append([S,median,area])
    return FeatureOutput(np.asarray(out),('survival_end','median_survival','restricted_mean'))
