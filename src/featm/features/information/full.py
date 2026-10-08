from __future__ import annotations

from collections import Counter
import numpy as np

from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import conditional_mi_discrete, discretize_1d, mutual_information_discrete, normalize_prob, shannon_entropy_prob


def _reg(name,family,*,cost=Cost.MEDIUM,output_kind='scalar',tags=(),structured=False):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        extra = {'batch_semantics': BatchSemantics.STRUCTURED_SAMPLE} if structured else {}
        return register_feature(name=name,family=family,modality='information',input_types=('distribution_or_samples',),cost=cost,streamable=False,output_kind=output_kind,tags=tags,description=name,**extra)(fn)
    return deco


def _prob_rows(x):
    a=np.asarray(x,float)
    if a.ndim==1:a=a[None,:]
    if a.ndim!=2:raise ValueError('expected probability/count rows shape (n,k)')
    return normalize_prob(a,axis=1)

def _pair_series(x):
    if isinstance(x,(tuple,list)) and len(x)==2:
        a=np.asarray(x[0]);b=np.asarray(x[1]);
        if a.ndim==1:a=a[None,:]
        if b.ndim==1:b=b[None,:]
    else:
        z=np.asarray(x)
        if z.ndim==2 and z.shape[0]==2:a,b=z[0][None,:],z[1][None,:]
        elif z.ndim==3 and z.shape[1]==2:a,b=z[:,0,:],z[:,1,:]
        else:raise ValueError('expected (A,B) or shape (n,2,t)')
    return a,b

def _triple_series(x):
    if isinstance(x,(tuple,list)) and len(x)==3:
        arr=[]
        for v in x:
            a=np.asarray(v);arr.append(a[None,:] if a.ndim==1 else a)
        return arr
    z=np.asarray(x)
    if z.ndim==2 and z.shape[0]==3:return [z[i][None,:] for i in range(3)]
    if z.ndim==3 and z.shape[1]==3:return [z[:,i,:] for i in range(3)]
    raise ValueError('expected (X,Y,Z) or shape (n,3,t)')

@_reg('information.entropy.shannon','entropy',cost=Cost.TINY,tags=('shannon',))
def ent_shannon(x,**kwargs):return shannon_entropy_prob(_prob_rows(x),axis=1)
@_reg('information.entropy.renyi','entropy',cost=Cost.TINY,tags=('renyi',))
def ent_renyi(x,alpha=2.0,**kwargs):
    p=_prob_rows(x)
    if alpha==1:return ent_shannon(p)
    return np.log(np.maximum(np.sum(p**alpha,axis=1),1e-15))/(1-alpha)
@_reg('information.entropy.tsallis','entropy',cost=Cost.TINY,tags=('tsallis',))
def ent_tsallis(x,q=2.0,**kwargs):
    p=_prob_rows(x)
    if q==1:return ent_shannon(p)
    return (1-np.sum(p**q,axis=1))/(q-1)

@_reg('information.entropy.joint','entropy',cost=Cost.MEDIUM,tags=('joint-entropy','structured'),structured=True)
def ent_joint(x,bins=16,**kwargs):
    a,b=_pair_series(x);out=[]
    for r,s in zip(a,b):
        dr,ds=discretize_1d(r,bins),discretize_1d(s,bins);c=Counter(zip(dr.tolist(),ds.tolist()));out.append(float(shannon_entropy_prob(np.array(list(c.values()),float))))
    return np.asarray(out)
@_reg('information.entropy.conditional','entropy',cost=Cost.MEDIUM,tags=('conditional-entropy','structured'),structured=True)
def ent_cond(x,bins=16,**kwargs):
    a,b=_pair_series(x);out=[]
    for r,s in zip(a,b):
        dr,ds=discretize_1d(r,bins),discretize_1d(s,bins);hy=shannon_entropy_prob(np.bincount(ds));hxy=ent_joint((r,s),bins=bins)[0];out.append(hxy-hy)
    return np.asarray(out)

@_reg('information.dependence.mutual_information','dependence',cost=Cost.MEDIUM,tags=('mutual-information','structured'),structured=True)
def dep_mi(x,bins=16,**kwargs):
    a,b=_pair_series(x);return np.asarray([mutual_information_discrete(r,s,bins) for r,s in zip(a,b)])
@_reg('information.dependence.conditional_mi','dependence',cost=Cost.HIGH,tags=('conditional-mi','structured'),structured=True)
def dep_cmi(x,bins=8,**kwargs):
    a,b,c=_triple_series(x);return np.asarray([conditional_mi_discrete(r,s,t,bins) for r,s,t in zip(a,b,c)])
@_reg('information.dependence.interaction_information','dependence',cost=Cost.HIGH,tags=('interaction-information','structured'),structured=True)
def interaction_information(x,bins=8,**kwargs):
    a,b,c=_triple_series(x);return np.asarray([mutual_information_discrete(r,s,bins)-conditional_mi_discrete(r,s,t,bins) for r,s,t in zip(a,b,c)])
@_reg('information.dependence.transfer_entropy','dependence',cost=Cost.HIGH,tags=('transfer-entropy','structured'),structured=True)
def transfer_entropy(x,bins=8,**kwargs):
    a,b=_pair_series(x);return np.asarray([conditional_mi_discrete(s[1:],r[:-1],s[:-1],bins) for r,s in zip(a,b)])


def _pair_probs(x):
    if isinstance(x,(tuple,list)) and len(x)==2:return _prob_rows(x[0]),_prob_rows(x[1])
    z=np.asarray(x,float)
    if z.ndim==2 and z.shape[0]==2:return _prob_rows(z[0]),_prob_rows(z[1])
    if z.ndim==3 and z.shape[1]==2:return normalize_prob(z[:,0,:],1),normalize_prob(z[:,1,:],1)
    raise ValueError('expected (P,Q) or shape (n,2,k)')
@_reg('information.divergence.kl','divergence',cost=Cost.TINY,tags=('kl','structured'),structured=True)
def kl(x,**kwargs):
    p,q=_pair_probs(x);return np.sum(p*np.log(np.maximum(p,1e-15)/np.maximum(q,1e-15)),axis=1)
@_reg('information.divergence.jensen_shannon','divergence',cost=Cost.TINY,tags=('js','structured'),structured=True)
def js(x,**kwargs):
    p,q=_pair_probs(x);m=.5*(p+q);return .5*np.sum(p*np.log(np.maximum(p,1e-15)/np.maximum(m,1e-15)),1)+.5*np.sum(q*np.log(np.maximum(q,1e-15)/np.maximum(m,1e-15)),1)
@_reg('information.divergence.variation_of_information','divergence',cost=Cost.MEDIUM,tags=('variation-information','structured'),structured=True)
def vi(x,bins=16,**kwargs):
    a,b=_pair_series(x);out=[]
    for r,s in zip(a,b):
        dr,ds=discretize_1d(r,bins),discretize_1d(s,bins);hx=shannon_entropy_prob(np.bincount(dr));hy=shannon_entropy_prob(np.bincount(ds));mi=mutual_information_discrete(dr,ds,bins);out.append(hx+hy-2*mi)
    return np.asarray(out)

# Multivariate entropy measures for a batch of sample matrices (obs, variables).
def _datasets(x):
    a=np.asarray(x,float)
    if a.ndim==2:a=a[None,:,:]
    if a.ndim!=3:raise ValueError('expected (n_datasets,n_observations,n_variables) or one 2D dataset')
    return a

def _joint_entropy_matrix(M,bins=8):
    D=np.column_stack([discretize_1d(M[:,j],bins) for j in range(M.shape[1])]);c=Counter(map(tuple,D.tolist()));return float(shannon_entropy_prob(np.array(list(c.values()),float)))
@_reg('information.multivariate.total_correlation','multivariate',cost=Cost.HIGH,tags=('total-correlation','structured'),structured=True)
def total_corr(x,bins=8,**kwargs):
    out=[]
    for M in _datasets(x):out.append(sum(shannon_entropy_prob(np.bincount(discretize_1d(M[:,j],bins))) for j in range(M.shape[1]))-_joint_entropy_matrix(M,bins))
    return np.asarray(out)
@_reg('information.multivariate.dual_total_correlation','multivariate',cost=Cost.HIGH,tags=('dual-total-correlation','structured'),structured=True)
def dual_total(x,bins=8,**kwargs):
    out=[]
    for M in _datasets(x):
        h_all=_joint_entropy_matrix(M,bins);s=0
        for j in range(M.shape[1]):s+=_joint_entropy_matrix(np.delete(M,j,axis=1),bins)
        out.append(s-(M.shape[1]-1)*h_all)
    return np.asarray(out)

# Williams-Beer I_min PID for two predictors X1/X2 and target Y.
def _specific_information(x,y,bins=8):
    dx,dy=discretize_1d(x,bins),discretize_1d(y,bins);n=len(dx);cx,cy=Counter(dx.tolist()),Counter(dy.tolist());cxy=Counter(zip(dx.tolist(),dy.tolist()));vals={}
    for yy,ny in cy.items():
        p_y=ny/n;info=0
        for xx,nxy in [(xx,cxy.get((xx,yy),0)) for xx in cx]:
            if nxy==0:continue
            p_x_given_y=nxy/ny;p_y_given_x=nxy/cx[xx];info+=p_x_given_y*np.log(max(p_y_given_x/p_y,1e-15))
        vals[yy]=info
    return vals,cy,n

def _pid_one(x1,x2,y,bins=8):
    i1=mutual_information_discrete(x1,y,bins);i2=mutual_information_discrete(x2,y,bins);i12=mutual_information_discrete(np.array([a*bins+b for a,b in zip(discretize_1d(x1,bins),discretize_1d(x2,bins))]),y,bins*bins)
    s1,cy,n=_specific_information(x1,y,bins);s2,_,_=_specific_information(x2,y,bins);red=sum((ny/n)*min(s1.get(yy,0),s2.get(yy,0)) for yy,ny in cy.items());u1=i1-red;u2=i2-red;syn=i12-red-u1-u2;return red,u1,u2,syn
for _name,_idx in [('redundancy',0),('unique_information',1),('synergy',3)]:
    @_reg(f'information.pid.{_name}','pid',cost=Cost.EXTREME,tags=('pid',_name,'structured'),structured=True)
    def _pid(x,bins=8,_idx=_idx,**kwargs):
        a,b,y=_triple_series(x);return np.asarray([_pid_one(r,s,t,bins)[_idx] for r,s,t in zip(a,b,y)])

@_reg('information.geometry.information_geometry','geometry',output_kind='vector',cost=Cost.LOW,tags=('fisher-rao','hellinger'))
def info_geometry(x,**kwargs):
    p=_prob_rows(x);u=np.full_like(p,1/p.shape[1]);bc=np.sum(np.sqrt(p*u),axis=1);fisher=2*np.arccos(np.clip(bc,0,1));hell=np.sqrt(np.sum((np.sqrt(p)-np.sqrt(u))**2,axis=1))/np.sqrt(2);return FeatureOutput(np.column_stack([fisher,hell,ent_shannon(p)]),('fisher_rao_to_uniform','hellinger_to_uniform','entropy'))
