from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np
import pandas as pd

from ...registry import Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import discrete_entropy


def _reg(name, *, family, input_types=("numeric",), arity=1, cost=Cost.TINY, tags=(), description="", output_kind="scalar", incremental=False, stateful=False):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:
            return fn
        return register_feature(
            name=name, family=family, modality="tabular", input_types=input_types,
            arity=arity, cost=cost, streamable=not incremental, incremental=incremental,
            stateful=stateful,
            description=description or name, tags=tags, output_kind=output_kind,
        )(fn)
    return deco


def _num(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").astype(float)


def _cat(s):
    return pd.Series(s).astype("string").fillna("<NA>")


def _ret(v, state=None):
    return v, state


# Concrete unary numeric transforms. Genuinely row-independent -- each
# output value is a pure function of that row's own input value alone.
_NUMERIC = {
    "abs": lambda x: np.abs(x),
    "cube_signed": lambda x: x ** 3,
    "exponential": lambda x: np.exp(np.clip(x, -50, 50)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
    "quantization": lambda x: pd.Series(np.round(x), index=x.index),
    "piecewise": lambda x: pd.Series(np.where(x < -1, -1.0, np.where(x > 1, 1.0, x)), index=x.index),
}
for _short, _fn in _NUMERIC.items():
    @_reg(f"tabular.numeric.{_short}", family="numeric", tags=("numeric", _short))
    def _generated(s, *, fit=False, state=None, _fn=_fn):
        x=_num(s); return _ret(pd.Series(_fn(x), index=x.index))

# rank/percentile/ecdf are NOT row-independent: pandas' .rank() ranks each
# value relative to every OTHER value in the same column -- a genuinely
# whole-column-dependent statistic, not a pure per-row function. Discovered
# by the fourth audit pass's automated batch-equivalence harness
# (Finding 7/8: at batch_rows=1, each row would see only itself as "the
# whole column," trivially giving rank=1/percentile=1.0 for every row,
# silently different from the true whole-column rank). Marked
# incremental=True, matching the same legacy-single-shot-recompute contract
# (and the same engine-enforced multi-batch rejection) already established
# for tabular.numeric.zscore/tabular.statistical.moments/etc. -- correct for
# a single in-memory call or a source that fits in one out-of-core batch,
# rejected (not silently wrong) across multiple out-of-core batches.
_NUMERIC_WHOLE_COLUMN = {
    "rank": lambda x: x.rank(pct=False, method="average"),
    "percentile": lambda x: x.rank(pct=True, method="average"),
    "ecdf": lambda x: x.rank(pct=True, method="max"),
}
for _short, _fn in _NUMERIC_WHOLE_COLUMN.items():
    @_reg(f"tabular.numeric.{_short}", family="numeric", tags=("numeric", _short), incremental=True)
    def _generated_wc(s, *, fit=False, state=None, _fn=_fn):
        x=_num(s); return _ret(pd.Series(_fn(x), index=x.index))


@_reg("tabular.numeric.logarithmic", family="numeric", output_kind="vector", tags=("log",))
def numeric_logarithmic(s, *, fit=False, state=None):
    x=_num(s); a=np.column_stack([np.sign(x)*np.log1p(np.abs(x)), np.log1p(np.clip(x,0,None)), np.log10(1+np.abs(x))])
    return FeatureOutput(a,("signed_log1p","log1p_positive","log10_abs")), None

@_reg("tabular.numeric.roots", family="numeric", output_kind="vector", tags=("root",))
def numeric_roots(s, *, fit=False, state=None):
    x=_num(s).to_numpy(); a=np.column_stack([np.sqrt(np.abs(x)), np.cbrt(x), np.abs(x)**0.25])
    return FeatureOutput(a,("sqrt_abs","cbrt","fourth_root_abs")), None

@_reg("tabular.numeric.powers", family="numeric", output_kind="vector", tags=("polynomial",))
def numeric_powers(s, *, fit=False, state=None):
    x=_num(s).to_numpy(); return FeatureOutput(np.column_stack([x*x,x**3,x**4]),("square","cube","fourth")), None

@_reg("tabular.numeric.trigonometric", family="numeric", output_kind="vector", tags=("trigonometric",))
def numeric_trig(s, *, fit=False, state=None):
    x=_num(s).to_numpy(); return FeatureOutput(np.column_stack([np.sin(x),np.cos(x),np.tan(np.clip(x,-1.4,1.4))]),("sin","cos","tan_clipped")),None

@_reg("tabular.numeric.hyperbolic", family="numeric", output_kind="vector", tags=("hyperbolic",))
def numeric_hyperbolic(s, *, fit=False, state=None):
    x=np.clip(_num(s).to_numpy(),-20,20); return FeatureOutput(np.column_stack([np.sinh(x),np.cosh(x),np.tanh(x)]),("sinh","cosh","tanh")),None

@_reg("tabular.numeric.arithmetic", family="numeric", output_kind="vector", tags=("arithmetic",))
def numeric_arithmetic(s, *, fit=False, state=None):
    x=_num(s).to_numpy(); return FeatureOutput(np.column_stack([x, -x, np.abs(x)]),("identity","negate","abs")),None

@_reg("tabular.numeric.ratios", family="numeric", output_kind="vector", tags=("ratio",))
def numeric_ratios(s, *, fit=False, state=None):
    x=_num(s).to_numpy(); den=np.where(x==0,np.nan,x); return FeatureOutput(np.column_stack([1/den, x/(1+np.abs(x))]),("reciprocal","soft_ratio")),None

@_reg("tabular.numeric.binning", family="numeric", incremental=True, tags=("binning",))
def numeric_binning(s, *, fit=False, state=None, bins=10):
    x=_num(s)
    if fit or state is None:
        finite=x[np.isfinite(x)]
        edges=np.unique(np.quantile(finite,np.linspace(0,1,bins+1)[1:-1])) if len(finite) else np.array([])
        state=edges
    return pd.Series(np.digitize(x.to_numpy(),state),index=x.index,dtype=float), state


# Categorical features.
@_reg("tabular.categorical.count", family="categorical", input_types=("categorical",), incremental=True, tags=("count",))
def cat_count(s, *, fit=False, state=None):
    x=_cat(s)
    mapping=x.value_counts().to_dict() if fit or state is None else state
    return x.map(mapping).fillna(0).astype(float),mapping

@_reg("tabular.categorical.rarity", family="categorical", input_types=("categorical",), incremental=True, tags=("rarity",))
def cat_rarity(s, *, fit=False, state=None):
    x=_cat(s)
    if fit or state is None:
        counts=x.value_counts(); state=(counts/max(len(x),1)).to_dict()
    f=x.map(state).fillna(0).astype(float)
    return -np.log(np.maximum(f,1/max(len(x),1))),state

@_reg("tabular.categorical.is_rare", family="categorical", input_types=("categorical",), incremental=True, tags=("rarity",))
def cat_is_rare(s, *, fit=False, state=None, threshold=0.01):
    x=_cat(s)
    if fit or state is None:
        counts=x.value_counts(); state=(counts/max(len(x),1)).to_dict()
    return (x.map(state).fillna(0)<threshold).astype(float),state

@_reg("tabular.categorical.string_length", family="categorical", input_types=("categorical",), tags=("string",))
def cat_strlen(s, *, fit=False, state=None):
    return _cat(s).str.len().astype(float),None

@_reg("tabular.categorical.entropy", family="categorical", input_types=("categorical",), tags=("entropy",), incremental=True)
def cat_entropy(s, *, fit=False, state=None):
    # Whole-column entropy broadcast to every row -- not row-independent
    # (fourth audit pass, Finding 7/8, same class as numeric.rank above).
    x=_cat(s); h=discrete_entropy(x.tolist()); return pd.Series(h,index=x.index),None

@_reg("tabular.categorical.hashing", family="categorical", input_types=("categorical",), tags=("hash",))
def cat_hashing(s, *, fit=False, state=None, buckets=1024):
    x=_cat(s)
    out=x.map(lambda v:int.from_bytes(hashlib.blake2b(str(v).encode(),digest_size=8).digest(),'little')%buckets).astype(float)
    return out,None

@_reg("tabular.categorical.binary_encoding", family="categorical", input_types=("categorical",), output_kind="vector", incremental=True, tags=("encoding",))
def cat_binary(s, *, fit=False, state=None, bits=16):
    x=_cat(s)
    if fit or state is None:
        state={v:i for i,v in enumerate(sorted(x.unique().tolist()))}
    codes=x.map(state).fillna(-1).astype(int).to_numpy()
    arr=np.column_stack([((codes>>i)&1).astype(float) for i in range(bits)])
    return FeatureOutput(arr,tuple(f"bit_{i:02d}" for i in range(bits))),state

@_reg("tabular.categorical.embeddings", family="categorical", input_types=("categorical",), output_kind="vector", tags=("hash-embedding",))
def cat_embedding(s, *, fit=False, state=None, dim=8):
    x=_cat(s)
    arr=np.empty((len(x),dim),float)
    for i,v in enumerate(x):
        seed=int.from_bytes(hashlib.blake2b(str(v).encode(),digest_size=8).digest(),'little')
        rng=np.random.default_rng(seed); arr[i]=rng.standard_normal(dim)
    return FeatureOutput(arr,tuple(f"dim_{i:02d}" for i in range(dim))),None

@_reg("tabular.categorical.cooccurrence", family="categorical", input_types=("categorical","categorical"), arity=2, incremental=True, tags=("pairwise",))
def cat_cooccurrence(a,b,*,fit=False,state=None):
    a,b=_cat(a),_cat(b); key=a.str.cat(b,sep="|")
    if fit or state is None: state=key.value_counts().to_dict()
    return key.map(state).fillna(0).astype(float),state

@_reg("tabular.categorical.transition", family="categorical", input_types=("categorical",), tags=("sequence",), incremental=True)
def cat_transition(s, *, fit=False, state=None):
    # Depends on the PREVIOUS row (x.shift(1)) -- a batch boundary would
    # silently see NaN instead of the true previous value from the prior
    # batch, changing row 0 of every batch after the first (fourth audit
    # pass, Finding 7/8). Marked incremental=True for the same reason as
    # the whole-column-aggregate features above: the existing engine guard
    # rejects unsafe multi-batch execution rather than silently computing a
    # per-batch-local (wrong) answer at the first row of every batch.
    x=_cat(s); prev=x.shift(1); return (x!=prev).astype(float),None


# Datetime feature groups and missing calendar features.
def _dt(s): return pd.to_datetime(pd.Series(s),errors='coerce')

for _nm,_period,_attr in [("month_sin",12,"month"),("month_cos",12,"month"),("hour_sin",24,"hour"),("hour_cos",24,"hour"),("dayofyear_sin",365.2425,"dayofyear"),("dayofyear_cos",365.2425,"dayofyear")]:
    @_reg(f"tabular.datetime.{_nm}",family="datetime",input_types=("datetime",),tags=("cyclical",))
    def _cyc(s,*,fit=False,state=None,_nm=_nm,_period=_period,_attr=_attr):
        v=getattr(_dt(s).dt,_attr).astype(float); fun=np.sin if _nm.endswith('sin') else np.cos
        return pd.Series(fun(2*np.pi*v/_period),index=v.index),None

@_reg("tabular.datetime.hour",family="datetime",input_types=("datetime",),tags=("calendar",))
def dt_hour(s,*,fit=False,state=None): return _dt(s).dt.hour.astype(float),None

@_reg("tabular.datetime.elapsed_time",family="datetime",input_types=("datetime",),incremental=True,tags=("elapsed",))
def dt_elapsed(s,*,fit=False,state=None):
    x=_dt(s)
    origin=x.min() if fit or state is None else state
    return (x-origin).dt.total_seconds().astype(float),origin

@_reg("tabular.datetime.business_calendar",family="datetime",input_types=("datetime",),tags=("business-day",))
def dt_business(s,*,fit=False,state=None):
    x=_dt(s); return (~x.dt.dayofweek.isin([5,6])).astype(float),None

@_reg("tabular.datetime.holidays",family="datetime",input_types=("datetime",),tags=("holiday",),cost=Cost.LOW)
def dt_holiday(s,*,fit=False,state=None,country="US"):
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("Install the optional 'holidays' package to use tabular.datetime.holidays") from exc
    x=_dt(s); years=sorted(set(x.dropna().dt.year.tolist())); cal=holidays.country_holidays(country,years=years)
    return x.dt.date.map(lambda d:float(d in cal) if pd.notna(d) else np.nan),None

@_reg("tabular.datetime.astronomical",family="datetime",input_types=("datetime",),output_kind="vector",tags=("astronomy",))
def dt_astronomical(s,*,fit=False,state=None):
    x=_dt(s); n=x.dt.dayofyear.astype(float).to_numpy(); gamma=2*np.pi/365.2425*(n-1)
    decl=(0.006918-0.399912*np.cos(gamma)+0.070257*np.sin(gamma)-0.006758*np.cos(2*gamma)+0.000907*np.sin(2*gamma)-0.002697*np.cos(3*gamma)+0.00148*np.sin(3*gamma))
    eq=229.18*(0.000075+0.001868*np.cos(gamma)-0.032077*np.sin(gamma)-0.014615*np.cos(2*gamma)-0.040849*np.sin(2*gamma))
    return FeatureOutput(np.column_stack([decl,eq]),("solar_declination_rad","equation_of_time_min")),None

@_reg("tabular.datetime.calendar",family="datetime",input_types=("datetime",),output_kind="vector",tags=("calendar",))
def dt_calendar(s,*,fit=False,state=None):
    x=_dt(s); arr=np.column_stack([x.dt.year,x.dt.month,x.dt.day,x.dt.dayofweek,x.dt.dayofyear,x.dt.quarter,x.dt.hour])
    return FeatureOutput(arr,("year","month","day","dayofweek","dayofyear","quarter","hour")),None

@_reg("tabular.datetime.cyclical",family="datetime",input_types=("datetime",),output_kind="vector",tags=("cyclical",))
def dt_cyclical(s,*,fit=False,state=None):
    x=_dt(s); mo=x.dt.month.to_numpy(float); dw=x.dt.dayofweek.to_numpy(float); hr=x.dt.hour.to_numpy(float)
    arr=np.column_stack([np.sin(2*np.pi*mo/12),np.cos(2*np.pi*mo/12),np.sin(2*np.pi*dw/7),np.cos(2*np.pi*dw/7),np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24)])
    return FeatureOutput(arr,("month_sin","month_cos","dow_sin","dow_cos","hour_sin","hour_cos")),None


# Missingness.
for _typ in ("numeric","categorical"):
    @_reg(f"tabular.missingness.indicator_{_typ}",family="missingness",input_types=(_typ,),tags=("missingness",))
    def _miss_ind(s,*,fit=False,state=None): return pd.Series(s).isna().astype(float),None

@_reg("tabular.missingness.co_missingness",family="missingness",input_types=("numeric","numeric"),arity=2,tags=("missingness","pairwise"))
def co_missing(a,b,*,fit=False,state=None): return (pd.Series(a).isna() & pd.Series(b).isna()).astype(float),None

@_reg("tabular.missingness.entropy",family="missingness",input_types=("numeric",),tags=("missingness","entropy"),stateful=True)
def miss_entropy(s,*,fit=False,state=None):
    m=pd.Series(s).isna().to_numpy(); p=m.mean(); h=0.0 if p in (0,1) else -(p*np.log(p)+(1-p)*np.log(1-p)); return pd.Series(h,index=pd.Series(s).index),None

@_reg("tabular.missingness.patterns",family="missingness",input_types=("numeric",),tags=("missingness",))
def miss_patterns(s,*,fit=False,state=None): return pd.Series(s).isna().astype(float),None


# Interactions.
for _nm,_op in {
    "sum":lambda a,b:a+b,
    "difference":lambda a,b:a-b,
    "abs_difference":lambda a,b:np.abs(a-b),
}.items():
    @_reg(f"tabular.interactions.{_nm}",family="interactions",input_types=("numeric","numeric"),arity=2,tags=("pairwise",))
    def _pair(a,b,*,fit=False,state=None,_op=_op):
        x,y=_num(a),_num(b); return pd.Series(_op(x,y),index=x.index),None

@_reg("tabular.interactions.pairwise",family="interactions",input_types=("numeric","numeric"),arity=2,output_kind="vector",tags=("pairwise",))
def pairwise_bundle(a,b,*,fit=False,state=None):
    x,y=_num(a).to_numpy(),_num(b).to_numpy(); den=np.where(y==0,np.nan,y)
    return FeatureOutput(np.column_stack([x+y,x-y,x*y,x/den,np.abs(x-y)]),("sum","difference","product","ratio","abs_difference")),None

@_reg("tabular.interactions.polynomial",family="interactions",input_types=("numeric","numeric"),arity=2,output_kind="vector",tags=("polynomial",))
def pair_poly(a,b,*,fit=False,state=None):
    x,y=_num(a).to_numpy(),_num(b).to_numpy(); return FeatureOutput(np.column_stack([x*y,x*x,y*y,x*x*y,x*y*y]),("xy","x2","y2","x2y","xy2")),None

@_reg("tabular.interactions.logical",family="interactions",input_types=("numeric","numeric"),arity=2,output_kind="vector",tags=("logical",))
def pair_logical(a,b,*,fit=False,state=None):
    x=_num(a).to_numpy()!=0;y=_num(b).to_numpy()!=0;return FeatureOutput(np.column_stack([x&y,x|y,x^y]).astype(float),("and","or","xor")),None

@_reg("tabular.interactions.three_way",family="interactions",input_types=("numeric","numeric","numeric"),arity=3,tags=("three-way",))
def three_way(a,b,c,*,fit=False,state=None): return _num(a)*_num(b)*_num(c),None


# Ordinal.
@_reg("tabular.ordinal.cumulative",family="ordinal",input_types=("categorical",),incremental=True,tags=("ordinal",))
def ordinal_cumulative(s,*,fit=False,state=None):
    x=pd.Series(s)
    if fit or state is None:
        vals=sorted(x.dropna().unique().tolist());state={v:i for i,v in enumerate(vals)}
    code=x.map(state).astype(float); denom=max(len(state)-1,1); return code/denom,state

@_reg("tabular.ordinal.threshold",family="ordinal",input_types=("categorical",),output_kind="vector",incremental=True,tags=("ordinal",))
def ordinal_threshold(s,*,fit=False,state=None,max_levels=8):
    x=pd.Series(s)
    if fit or state is None:
        vals=sorted(x.dropna().unique().tolist())[:max_levels];state={v:i for i,v in enumerate(vals)}
    code=x.map(state).fillna(-1).astype(int).to_numpy(); k=min(len(state),max_levels)
    arr=np.column_stack([(code>=i).astype(float) for i in range(1,k)]) if k>1 else np.zeros((len(x),1))
    return FeatureOutput(arr,tuple(f"ge_{i}" for i in range(1,k)) if k>1 else ("ge_1",)),state


# Relational proxies operate on an entity/category series (or pair for groupby).
@_reg("tabular.relational.entity_counts",family="relational",input_types=("categorical",),incremental=True,tags=("entity",))
def rel_entity_counts(s,*,fit=False,state=None): return cat_count(s,fit=fit,state=state)

@_reg("tabular.relational.recency",family="relational",input_types=("datetime",),incremental=True,tags=("recency",))
def rel_recency(s,*,fit=False,state=None):
    x=_dt(s); ref=x.max() if fit or state is None else state; return (ref-x).dt.total_seconds().astype(float),ref

@_reg("tabular.relational.groupby",family="relational",input_types=("categorical","numeric"),arity=2,output_kind="vector",incremental=True,tags=("groupby",))
def rel_groupby(g,v,*,fit=False,state=None):
    g=_cat(g);v=_num(v);df=pd.DataFrame({'g':g,'v':v});stats=df.groupby('g')['v'].agg(['mean','std','median','count'])
    arr=np.column_stack([g.map(stats[c]).to_numpy(float) for c in stats.columns])
    return FeatureOutput(arr,tuple(stats.columns)),stats

@_reg("tabular.relational.entity_graph",family="relational",input_types=("categorical","categorical"),arity=2,output_kind="vector",tags=("graph",),incremental=True)
def rel_entity_graph(a,b,*,fit=False,state=None):
    # value_counts() over the whole column -- degree/multiplicity are
    # relation-wide statistics, not row-independent (fourth audit pass,
    # Finding 7/8, same class as tabular.relational.groupby above, which was
    # already correctly marked incremental=True).
    a,b=_cat(a),_cat(b); ca=a.value_counts();cb=b.value_counts(); pair=a.str.cat(b,sep='|');cp=pair.value_counts()
    arr=np.column_stack([a.map(ca),b.map(cb),pair.map(cp)]).astype(float)
    return FeatureOutput(arr,("left_degree","right_degree","edge_multiplicity")),None


# Statistical summaries encoded as rowwise standardized/outlier transforms. These
# names are intentionally bundle features because the old catalog entries were
# family concepts, not single formulas.
@_reg("tabular.statistical.moments",family="statistical",input_types=("numeric",),output_kind="vector",incremental=True,tags=("moments",))
def stat_moments(s,*,fit=False,state=None):
    x=_num(s)
    if fit or state is None:
        state=(x.mean(),x.std(ddof=0),x.skew(),x.kurt())
    mu,sd,sk,ku=state; z=(x-mu)/(sd if sd else np.nan)
    arr=np.column_stack([z,z**2,z**3,z**4])
    return FeatureOutput(arr,("z1","z2","z3","z4")),state

@_reg("tabular.statistical.quantiles",family="statistical",input_types=("numeric",),output_kind="vector",incremental=True,tags=("quantiles",))
def stat_quantiles(s,*,fit=False,state=None):
    x=_num(s)
    if fit or state is None: state=x.quantile([.1,.25,.5,.75,.9]).to_numpy()
    arr=np.column_stack([(x.to_numpy()<=q).astype(float) for q in state])
    return FeatureOutput(arr,("le_q10","le_q25","le_q50","le_q75","le_q90")),state

@_reg("tabular.statistical.order_statistics",family="statistical",input_types=("numeric",),output_kind="vector",tags=("order",),incremental=True)
def stat_order(s,*,fit=False,state=None):
    # x.rank() -- whole-column rank, same class as tabular.numeric.rank
    # above (fourth audit pass, Finding 7/8).
    x=_num(s); r=x.rank(method='average');return FeatureOutput(np.column_stack([r,r/max(len(x),1)]),("rank","rank_fraction")),None

@_reg("tabular.statistical.robust",family="statistical",input_types=("numeric",),output_kind="vector",incremental=True,tags=("robust",))
def stat_robust(s,*,fit=False,state=None):
    x=_num(s)
    if fit or state is None:
        med=x.median();mad=np.nanmedian(np.abs(x-med));q1,q3=x.quantile([.25,.75]);state=(med,mad,q1,q3)
    med,mad,q1,q3=state; rz=(x-med)/(1.4826*mad if mad else np.nan);iqr=q3-q1;ri=(x-med)/(iqr if iqr else np.nan)
    return FeatureOutput(np.column_stack([rz,ri]),("robust_z_mad","robust_iqr")),state

@_reg("tabular.statistical.tail",family="statistical",input_types=("numeric",),output_kind="vector",incremental=True,tags=("tail",))
def stat_tail(s,*,fit=False,state=None):
    x=_num(s)
    if fit or state is None: state=tuple(x.quantile([.01,.05,.95,.99]))
    q01,q05,q95,q99=state; arr=np.column_stack([x<q01,x<q05,x>q95,x>q99]).astype(float)
    return FeatureOutput(arr,("below_q01","below_q05","above_q95","above_q99")),state

@_reg("tabular.statistical.outlier_scores",family="statistical",input_types=("numeric",),output_kind="vector",incremental=True,tags=("outlier",))
def stat_outlier(s,*,fit=False,state=None):
    x=_num(s)
    if fit or state is None:
        mu=x.mean();sd=x.std(ddof=0);med=x.median();mad=np.nanmedian(np.abs(x-med));state=(mu,sd,med,mad)
    mu,sd,med,mad=state
    return FeatureOutput(np.column_stack([np.abs((x-mu)/(sd if sd else np.nan)),np.abs((x-med)/(1.4826*mad if mad else np.nan))]),("abs_z","abs_robust_z")),state


# Geographic unit sphere coordinates. Input is Nx2 for direct usage; these are
# not auto-scheduled by the legacy tabular planner.
def _latlon(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or a.shape[1]<2: raise ValueError('expected (n,2+) latitude, longitude degrees')
    lat=np.deg2rad(a[:,0]);lon=np.deg2rad(a[:,1]);return lat,lon
for _name,_fn in [("unit_x",lambda lat,lon:np.cos(lat)*np.cos(lon)),("unit_y",lambda lat,lon:np.cos(lat)*np.sin(lon)),("unit_z",lambda lat,lon:np.sin(lat))]:
    @_reg(f"tabular.geographic.{_name}",family="geographic",input_types=("geographic",),tags=("geographic",))
    def _geo(x,*,fit=False,state=None,_fn=_fn):
        lat,lon=_latlon(x);return pd.Series(_fn(lat,lon)),None

@_reg("tabular.numeric.minmax", family="numeric", incremental=True, tags=("scaling",))
def numeric_minmax(s, *, fit=False, state=None):
    x=_num(s)
    if fit or state is None: state=(x.min(),x.max())
    lo,hi=state;return (x-lo)/(hi-lo if hi>lo else np.nan),state

@_reg("tabular.numeric.zscore", family="numeric", incremental=True, tags=("scaling",))
def numeric_zscore(s, *, fit=False, state=None):
    x=_num(s)
    if fit or state is None: state=(x.mean(),x.std(ddof=0))
    mu,sd=state;return (x-mu)/(sd if sd else np.nan),state

for _nm,_fn in [
    ('char_count',lambda s:s.astype('string').fillna('').str.len().astype(float)),
    ('word_count',lambda s:s.astype('string').fillna('').str.split().str.len().astype(float)),
    ('digit_ratio',lambda s:s.astype('string').fillna('').map(lambda v:sum(c.isdigit() for c in v)/max(len(v),1)).astype(float)),
]:
    @_reg(f'tabular.text.{_nm}',family='text',input_types=('categorical',),tags=('text',_nm))
    def _tab_text(s,*,fit=False,state=None,_fn=_fn):return _fn(pd.Series(s)),None
