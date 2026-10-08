from __future__ import annotations

"""Controlled predictive-utility probes for Step 5.

This is intentionally a FOUNDATION, not the final public-dataset leaderboard.
It proves the evaluation machinery and leakage rules. Claude's completion pass
will add/download larger real datasets per modality and save the dataset
identity/license/version in the final evidence artifacts.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import featm


@dataclass(frozen=True)
class Probe:
    name: str
    task: str
    frame: pd.DataFrame
    y: np.ndarray
    features: tuple[str, ...]
    columns: dict


def _tabular_probe(n=800, seed=41) -> Probe:
    rng=np.random.default_rng(seed)
    x=rng.normal(size=n)
    z=rng.normal(size=n)
    cat=np.where(x+0.25*rng.normal(size=n)>0,"high","low")
    y=(1.4*x + .5*z + rng.normal(scale=.7,size=n)>0).astype(int)
    df=pd.DataFrame({"x":x,"z":z,"cat":cat})
    features=("tabular.numeric.abs","tabular.scaling.zscore","tabular.encoding.category_frequency")
    columns={"tabular.numeric.abs":"x","tabular.scaling.zscore":"x","tabular.encoding.category_frequency":"cat"}
    return Probe("tabular_controlled","classification",df,y,features,columns)


def _timeseries_probe(n=500, width=64, seed=42) -> Probe:
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,size=n)
    t=np.linspace(0,1,width,endpoint=False)
    arr=np.vstack([np.sin(2*np.pi*(3 if c==0 else 8)*t)+.25*rng.normal(size=width) for c in y])
    df=pd.DataFrame(arr,columns=[f"t{i}" for i in range(width)])
    features=("timeseries.statistical.mean","timeseries.statistical.std","timeseries.spectral.centroid","timeseries.spectral.spectral_entropy")
    return Probe("timeseries_frequency_controlled","classification",df,y,features,{})


def _text_lexical_probe(n=600, seed=43) -> Probe:
    """Falsifiable question: does the lexical/TF-IDF family recover a
    word-DISTRIBUTION signal? Two document classes drawn from disjoint core
    vocabularies (plus shared noise words) -- tfidf (stateful, vocabulary-
    driven) should separate them well; plain scalar lexical stats
    (word_count/unique_word_ratio), which do not look at WHICH words appear,
    are included as a low-signal contrast, not because they are expected to
    win."""
    rng=np.random.default_rng(seed)
    # Equal-length prefixes ("worda"/"wordb", both 5 chars) so every vocab_a/vocab_b
    # word pair at the same index has IDENTICAL character length -- otherwise a
    # length-based lexical stat (avg_word_length/char_count) can trivially
    # discriminate the classes without looking at word IDENTITY at all, which
    # would make "does the lexical family recover word-distribution signal"
    # unfalsifiable (a length confound, not real gap between word_count/
    # unique_word_ratio-style lexical stats and identity-aware ones like tfidf).
    vocab_a=[f"worda{i}" for i in range(12)]
    vocab_b=[f"wordb{i}" for i in range(12)]
    noise=["the","and","of","a","in","to"]
    y=rng.integers(0,2,size=n)
    docs=[]
    for c in y:
        core=vocab_a if c==0 else vocab_b
        words=list(rng.choice(core,size=8))+list(rng.choice(noise,size=4))
        rng.shuffle(words)
        docs.append(" ".join(words))
    df=pd.DataFrame({"text":docs})
    features=("text.tfidf.tfidf","text.lexical.word_count","text.lexical.unique_word_ratio")
    return Probe("text_lexical_word_distribution_controlled","classification",df,y,features,{})


def _graph_topology_probe(n=300, seed=44) -> Probe:
    """Falsifiable question: does graph topology/centrality recover a
    generated graph-CLASS difference? Class 0: sparse Erdos-Renyi graphs;
    class 1: denser Watts-Strogatz small-world graphs with high rewiring --
    density/clustering/degree should differ systematically. Uses graph-level
    SCALAR summary features only (one value per graph), never a per-node
    vector feature, so every row stays a fixed-width, row-aligned vector
    regardless of each graph's size."""
    import networkx as nx
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,size=n)
    graphs=[]
    for i,c in enumerate(y):
        nodes=14+int(rng.integers(0,6))
        if c==0:
            g=nx.erdos_renyi_graph(nodes,0.12,seed=seed*1000+i)
        else:
            g=nx.watts_strogatz_graph(nodes,4,0.6,seed=seed*1000+i)
        graphs.append(g)
    df=pd.DataFrame({"g":graphs})
    features=("graph.connectivity.density","graph.degree.average_degree",
              "graph.clustering.average_clustering","graph.clustering.transitivity","graph.assortativity.degree")
    return Probe("graph_topology_class_controlled","classification",df,y,features,{})


def _score_matrix(Xtr,Xte,ytr,yte):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1000))
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xte)[:,1]
    return float(roc_auc_score(yte,p))


def _feature_matrix(feature: str, probe: "Probe", train: pd.DataFrame, test: pd.DataFrame):
    """Mandatory Step-5 leakage rule, in one place: a stateful feature is fit
    ONLY on the training fold and then applied (frozen) to both folds. It
    must never be independently fit on the validation/test fold. Both
    ``evaluate_probe`` (single feature) and ``evaluate_probe_bundle`` (whole
    family together) route through this one function so the rule cannot
    silently diverge between the two call sites."""
    spec=featm.REGISTRY.get(feature)
    if spec.stateful:
        fitted=featm.fit(train,features=[feature],columns=probe.columns or None,errors="raise")
        Xtr=fitted.transform(train,output="memory").to_pandas().to_numpy()
        Xte=fitted.transform(test,output="memory").to_pandas().to_numpy()
    else:
        Xtr=featm.extract(train,features=[feature],columns=probe.columns or None,errors="raise",output="memory").to_pandas().to_numpy()
        Xte=featm.extract(test,features=[feature],columns=probe.columns or None,errors="raise",output="memory").to_pandas().to_numpy()
    return Xtr,Xte,spec


def evaluate_probe(probe:Probe, folds=5):
    from sklearn.model_selection import StratifiedKFold
    skf=StratifiedKFold(folds,shuffle=True,random_state=123)
    rows=[]
    for feature in probe.features:
        spec=featm.REGISTRY.get(feature)
        try:
            scores=[]
            for tr,te in skf.split(probe.frame,probe.y):
                train=probe.frame.iloc[tr].reset_index(drop=True)
                test=probe.frame.iloc[te].reset_index(drop=True)
                Xtr,Xte,spec=_feature_matrix(feature,probe,train,test)
                scores.append(_score_matrix(Xtr,Xte,probe.y[tr],probe.y[te]))
            rows.append({"probe":probe.name,"task":probe.task,"feature":feature,"modality":spec.modality,"family":spec.family,"folds":folds,
                         "mean_score":float(np.mean(scores)),"std_score":float(np.std(scores)),"metric":"roc_auc","status":"ok"})
        except Exception as exc:
            # Not utility-eligible with a reason, rather than crashing the whole
            # probe run or fabricating a score (e.g. a disconnected-graph edge
            # case an assortativity/degree computation cannot handle).
            rows.append({"probe":probe.name,"task":probe.task,"feature":feature,"modality":spec.modality,"family":spec.family,"folds":folds,
                         "mean_score":None,"std_score":None,"metric":"roc_auc","status":"not_utility_eligible",
                         "reason":f"{type(exc).__name__}: {str(exc)[:200]}"})
    return rows


def evaluate_probe_bundle(probe:Probe, folds=5):
    """Family-bundle utility: all of ``probe.features`` concatenated into one
    matrix per fold, scored together. This is a DIFFERENT, complementary
    number to each feature's individual score in ``evaluate_probe`` -- a
    family can be individually weak but collectively strong (or vice versa),
    and the two must never be conflated into a single reported figure."""
    from sklearn.model_selection import StratifiedKFold
    skf=StratifiedKFold(folds,shuffle=True,random_state=123)
    modalities={featm.REGISTRY.get(f).modality for f in probe.features}
    families={featm.REGISTRY.get(f).family for f in probe.features}
    base={"probe":probe.name,"task":probe.task,"bundle":"+".join(probe.features),
          "modality":next(iter(modalities)) if len(modalities)==1 else ";".join(sorted(modalities)),
          "family":next(iter(families)) if len(families)==1 else ";".join(sorted(families)),
          "n_features":len(probe.features),"folds":folds,"metric":"roc_auc"}
    try:
        scores=[]
        for tr,te in skf.split(probe.frame,probe.y):
            train=probe.frame.iloc[tr].reset_index(drop=True)
            test=probe.frame.iloc[te].reset_index(drop=True)
            Xtr_parts=[];Xte_parts=[]
            for feature in probe.features:
                Xtr,Xte,_=_feature_matrix(feature,probe,train,test)
                Xtr_parts.append(Xtr);Xte_parts.append(Xte)
            scores.append(_score_matrix(np.concatenate(Xtr_parts,axis=1),np.concatenate(Xte_parts,axis=1),probe.y[tr],probe.y[te]))
        return [{**base,"mean_score":float(np.mean(scores)),"std_score":float(np.std(scores)),"status":"ok"}]
    except Exception as exc:
        return [{**base,"mean_score":None,"std_score":None,"status":"not_utility_eligible",
                 "reason":f"{type(exc).__name__}: {str(exc)[:200]}"}]


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="step5_results/step5_controlled_utility_results.csv")
    ap.add_argument("--bundle-out",default="step5_results/step5_family_bundle_utility_results.csv")
    args=ap.parse_args()
    try:
        import sklearn  # noqa:F401
    except Exception as exc:
        print(f"scikit-learn unavailable: {exc}");return 2
    probes=(_tabular_probe(),_timeseries_probe(),_text_lexical_probe(),_graph_topology_probe())
    rows=[]
    bundle_rows=[]
    for p in probes:
        print(f"probe={p.name} ...")
        rows.extend(evaluate_probe(p))
        bundle_rows.extend(evaluate_probe_bundle(p))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    row_fields=sorted({k for r in rows for k in r})
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=row_fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    bout=Path(args.bundle_out);bout.parent.mkdir(parents=True,exist_ok=True)
    bundle_fields=sorted({k for r in bundle_rows for k in r})
    with bout.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=bundle_fields,extrasaction="ignore");w.writeheader();w.writerows(bundle_rows)
    print(f"utility_rows={len(rows)} bundle_rows={len(bundle_rows)}")
    return 0

if __name__=="__main__":raise SystemExit(main())
