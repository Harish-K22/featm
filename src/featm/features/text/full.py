from __future__ import annotations

from ...contracts import forward_parameters

from collections import Counter
import math
import re
import zlib
import numpy as np

from ...contracts import normalize_text, word_tokens
from ...registry import Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import content_seed, discrete_entropy


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=(),**extra):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        return register_feature(name=name,family=family,modality='text',input_types=('text',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,requires=requires,description=name,**extra)(fn)
    return deco

def _docs(x):
    if isinstance(x,str):return [x]
    return [normalize_text(v) for v in x]
def _words(s):return word_tokens(s)
def _sentences(s):return [p.strip() for p in re.split(r'[.!?]+',s) if p.strip()]

def _syllables(word):
    w=re.sub(r'[^a-z]','',word.lower())
    if not w:return 0
    groups=re.findall(r'[aeiouy]+',w);n=len(groups)
    if w.endswith('e') and n>1:n-=1
    return max(1,n)

for _name,_fn in [
('word_count',lambda s:len(_words(s))),('char_count',len),('avg_word_length',lambda s:np.mean([len(w) for w in _words(s)]) if _words(s) else 0),
('digit_ratio',lambda s:sum(c.isdigit() for c in s)/max(len(s),1)),('punctuation_ratio',lambda s:sum((not c.isalnum()) and (not c.isspace()) for c in s)/max(len(s),1)),
('unique_word_ratio',lambda s:len(set(_words(s)))/max(len(_words(s)),1)),('lexical_diversity',lambda s:len(set(_words(s)))/max(len(_words(s)),1)),
('vocabulary',lambda s:len(set(_words(s)))),
]:
    @_reg(f'text.lexical.{_name}','lexical',cost=Cost.TINY,tags=('lexical',))
    def _lex(x,_fn=_fn,**kwargs):return np.asarray([_fn(s) for s in _docs(x)],float)

def _stable_bucket(s: str, buckets: int) -> int:
    """A deterministic hash bucket, stable ACROSS separate Python processes.

    Step 4 completion pass: found via real multi-process Dask validation
    (word_ngrams/char_ngrams previously used the builtin ``hash()``, whose
    string hashing is randomized PER PROCESS by default -- see
    ``featm.features._utils.content_seed``'s docstring, which documents the
    exact same bug class already fixed once for DeepWalk/EEMD's internal RNG
    seeding in a prior Step 2 audit pass). Invisible in every prior
    single-process CPU run (Steps 1-3): one process means one consistent
    hash seed for that whole run, so results looked internally reproducible
    even though a NEW process (a restarted script, or -- the case that
    actually caught it -- a genuinely SEPARATE Dask worker PROCESS computing
    a different partition of the SAME feature selection) would silently
    bucket identical n-grams differently. Reuses ``content_seed`` rather than
    introducing a second, parallel deterministic-hash helper.
    """
    return content_seed(s.encode("utf-8")) % buckets


@_reg('text.lexical.ngrams','lexical',output_kind='vector',cost=Cost.LOW,tags=('ngrams',))
def word_ngrams(x,n=2,buckets=64,**kwargs):
    out=[]
    for s in _docs(x):
        w=_words(s);v=np.zeros(buckets)
        for i in range(max(0,len(w)-n+1)):v[_stable_bucket('\x1f'.join(w[i:i+n]),buckets)]+=1
        if v.sum():v/=v.sum()
        out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'bucket_{i:02d}' for i in range(buckets)))

@_reg('text.character.character_ngrams','character',output_kind='vector',cost=Cost.LOW,tags=('char-ngram',))
def char_ngrams(x,n=3,buckets=64,**kwargs):
    out=[]
    for s in _docs(x):
        v=np.zeros(buckets)
        for i in range(max(0,len(s)-n+1)):v[_stable_bucket(s[i:i+n],buckets)]+=1
        if v.sum():v/=v.sum()
        out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'bucket_{i:02d}' for i in range(buckets)))
@_reg('text.character.punctuation','character',output_kind='vector',cost=Cost.TINY,tags=('punctuation',))
def punctuation(x,**kwargs):
    chars='.,;:!?()[]{}"\'';out=[]
    for s in _docs(x):out.append([s.count(c)/max(len(s),1) for c in chars])
    return FeatureOutput(np.asarray(out),tuple(f'u{ord(c):04x}' for c in chars))
@_reg('text.character.unicode','character',output_kind='vector',cost=Cost.TINY,tags=('unicode',))
def unicode_stats(x,**kwargs):
    import unicodedata
    cats=['L','N','P','S','Z','M'];out=[]
    for s in _docs(x):
        c=Counter(unicodedata.category(ch)[0] for ch in s);out.append([c[k]/max(len(s),1) for k in cats]+[sum(ord(ch)>127 for ch in s)/max(len(s),1)])
    return FeatureOutput(np.asarray(out),tuple(cats+['non_ascii_ratio']))

@_reg('text.information.character_entropy','information',cost=Cost.TINY,tags=('entropy',))
def char_entropy(x,**kwargs):return np.asarray([discrete_entropy(s) for s in _docs(x)],float)
@_reg('text.information.entropy','information',output_kind='vector',cost=Cost.TINY,tags=('entropy',))
def text_entropy(x,**kwargs):
    return FeatureOutput(np.column_stack([char_entropy(x),np.asarray([discrete_entropy(_words(s)) for s in _docs(x)])]),('character','word'))
@_reg('text.information.compression','information',cost=Cost.LOW,tags=('compression',))
def text_compression(x,**kwargs):
    out=[]
    for s in _docs(x):b=s.encode('utf-8');out.append(len(zlib.compress(b))/max(len(b),1))
    return np.asarray(out)
@_reg('text.information.surprisal','information',cost=Cost.LOW,tags=('surprisal','stateful'),stateful=True,incremental=True)
def text_surprisal(x,alpha=1.0,**kwargs):
    # This is the small-data single-shot convenience (one call = fit the
    # global word distribution over exactly the documents given, then score
    # them against it -- correct for a single in-memory call). Out-of-core
    # multi-batch execution routes through featm.state.SurprisalEncoder
    # instead (registered into STATEFUL_ENCODERS below), which fits the
    # global distribution across ALL fit batches before scoring any of them
    # -- fourth audit pass, Finding 1A: the pre-fix out-of-core path called
    # this exact function per batch, recomputing an independent (and wrong)
    # global distribution from each batch alone.
    docs=_docs(x);global_c=Counter();total=0
    for s in docs:global_c.update(_words(s));total+=len(_words(s))
    V=max(len(global_c),1);out=[]
    for s in docs:
        w=_words(s);out.append(np.mean([-math.log((global_c[t]+alpha)/(total+alpha*V)) for t in w]) if w else 0)
    return np.asarray(out)

@_reg('text.readability.flesch','readability',output_kind='vector',cost=Cost.LOW,tags=('flesch',))
def flesch(x,**kwargs):
    out=[]
    for s in _docs(x):
        w=_words(s);sent=_sentences(s);sy=sum(_syllables(t) for t in w);wc=max(len(w),1);sc=max(len(sent),1);reading=206.835-1.015*(wc/sc)-84.6*(sy/wc);grade=.39*(wc/sc)+11.8*(sy/wc)-15.59;out.append([reading,grade,wc/sc,sy/wc])
    return FeatureOutput(np.asarray(out),('reading_ease','grade','words_per_sentence','syllables_per_word'))

@_reg('text.stylometry.style','stylometry',output_kind='vector',cost=Cost.LOW,tags=('stylometry',))
def style(x,**kwargs):
    out=[]
    for s in _docs(x):
        w=_words(s);sent=_sentences(s);out.append([len(w),len(sent),np.mean([len(t) for t in w]) if w else 0,len(set(w))/max(len(w),1),sum(c.isupper() for c in s)/max(len(s),1),sum(c.isdigit() for c in s)/max(len(s),1),sum(c in ',.;:!?' for c in s)/max(len(s),1)])
    return FeatureOutput(np.asarray(out),('words','sentences','avg_word_length','type_token_ratio','uppercase_ratio','digit_ratio','punctuation_ratio'))

# Morphology/syntax via spaCy only when requested.
def _spacy_docs(x,model='en_core_web_sm'):
    try:import spacy
    except ImportError as exc:raise ImportError("Install spaCy and a language model for text syntax/morphology features") from exc
    try:nlp=spacy.load(model)
    except Exception as exc:raise ImportError(f"Could not load spaCy model {model!r}; install/download it first") from exc
    return list(nlp.pipe(_docs(x)))
@_reg('text.morphology.morphemes','morphology',output_kind='vector',cost=Cost.HIGH,tags=('morphology',),requires=('spacy',))
def morphemes(x,model='en_core_web_sm',**kwargs):
    out=[]
    for d in _spacy_docs(x,model):
        morph=[len(t.morph) for t in d if not t.is_space];out.append([np.mean(morph) if morph else 0,np.std(morph) if morph else 0,np.max(morph) if morph else 0])
    return FeatureOutput(np.asarray(out),('features_per_token_mean','std','max'))
@_reg('text.syntax.pos','syntax',output_kind='vector',cost=Cost.HIGH,tags=('pos',),requires=('spacy',))
def pos(x,model='en_core_web_sm',**kwargs):
    tags=['NOUN','VERB','ADJ','ADV','PRON','PROPN','ADP','DET','NUM','PUNCT','AUX','OTHER'];out=[]
    for d in _spacy_docs(x,model):
        c=Counter(t.pos_ if t.pos_ in tags[:-1] else 'OTHER' for t in d if not t.is_space);n=max(sum(c.values()),1);out.append([c[t]/n for t in tags])
    return FeatureOutput(np.asarray(out),tuple(tags))
@_reg('text.syntax.dependencies','syntax',output_kind='vector',cost=Cost.HIGH,tags=('dependency',),requires=('spacy',))
def dependencies(x,model='en_core_web_sm',buckets=32,**kwargs):
    out=[]
    for d in _spacy_docs(x,model):
        v=np.zeros(buckets)
        for t in d:v[_stable_bucket(t.dep_,buckets)]+=1
        if v.sum():v/=v.sum()
        out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'bucket_{i:02d}' for i in range(buckets)))
@_reg('text.syntax.parse_depth','syntax',output_kind='vector',cost=Cost.HIGH,tags=('parse-depth',),requires=('spacy',))
def parse_depth(x,model='en_core_web_sm',**kwargs):
    out=[]
    for d in _spacy_docs(x,model):
        depths=[]
        for t in d:
            cur=t;dep=0;seen=set()
            while cur.head!=cur and cur.i not in seen:seen.add(cur.i);cur=cur.head;dep+=1
            depths.append(dep)
        out.append([np.mean(depths) if depths else 0,np.max(depths) if depths else 0])
    return FeatureOutput(np.asarray(out),('mean','max'))

# Corpus vectorizers.
@_reg('text.tfidf.tfidf','tfidf',output_kind='vector',cost=Cost.MEDIUM,tags=('tfidf','stateful'),requires=('scikit-learn',),stateful=True,incremental=True)
def tfidf(x,max_features=128,**kwargs):
    from ...state import TfidfEncoder
    import pandas as pd
    encoder=TfidfEncoder(); docs=pd.Series(_docs(x)); params={"max_features":max_features}
    state=encoder.finalize_state(encoder.update_state(encoder.init_state(params=params),docs,params=params),params=params)
    return encoder.transform_batch(docs,state,params=params)
@_reg('text.bm25.bm25','bm25',cost=Cost.MEDIUM,tags=('bm25','stateful'),stateful=True,incremental=True)
def bm25(x,query=None,k1=1.5,b=.75,**kwargs):
    # Small-data single-shot convenience; out-of-core multi-batch execution
    # routes through featm.state.Bm25Encoder instead (registered below),
    # which fits document-frequency/corpus-size/average-document-length
    # across ALL fit batches -- fourth audit pass, Finding 1B: these are
    # corpus statistics, not something a per-batch recompute can get right.
    docs=_docs(x)
    if query is None:raise ValueError('text.bm25.bm25 requires query="..."')
    token=[_words(s) for s in docs];q=_words(query);N=len(docs);avg=np.mean([len(d) for d in token]) or 1;df=Counter()
    for d in token:
        for t in set(d):df[t]+=1
    out=[]
    for d in token:
        tf=Counter(d);score=0
        for t in q:
            idf=np.log(1+(N-df[t]+.5)/(df[t]+.5));f=tf[t];score+=idf*f*(k1+1)/(f+k1*(1-b+b*len(d)/avg)+1e-15)
        out.append(score)
    return np.asarray(out)

# Semantics with lightweight corpus methods.
@_reg('text.semantics.semantic_density','semantics',cost=Cost.MEDIUM,tags=('semantic-density',),requires=('scikit-learn',))
def semantic_density(x,**kwargs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs=_docs(x);out=[]
    for s in docs:
        sent=_sentences(s)
        if len(sent)<2:out.append(1.0);continue
        Z=TfidfVectorizer().fit_transform(sent);S=(Z@Z.T).toarray();out.append((S.sum()-np.trace(S))/(len(sent)*(len(sent)-1)))
    return np.asarray(out)
@_reg('text.semantics.similarity','semantics',cost=Cost.MEDIUM,tags=('similarity','stateful'),requires=('scikit-learn',),stateful=True,incremental=True)
def similarity(x,reference=None,**kwargs):
    # Small-data single-shot convenience: fits vocabulary/IDF over docs+[ref]
    # in one call (matches v0.1 behavior exactly). Out-of-core multi-batch
    # execution routes through featm.state.SimilarityEncoder instead
    # (registered below), which freezes vocabulary/IDF from the TRAINING
    # corpus only (never including a transform-time reference -- that would
    # be leakage for a genuine train/test split) and freezes the *first
    # training document seen during fit* as the default reference when none
    # is given -- fourth audit pass, Finding 1D: the pre-fix out-of-core path
    # both refit TF-IDF per batch AND picked a new "first document" default
    # reference per batch, so rows 0/2/4 of a batch_rows=2 run each scored
    # 1.0 against themselves instead of only the true first row doing so.
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs=_docs(x);ref=reference if reference is not None else (docs[0] if docs else '');Z=TfidfVectorizer().fit_transform(docs+[ref]);r=Z[-1];return (Z[:-1]@r.T).toarray().ravel()
@_reg(
    'text.semantics.topic','semantics',output_kind='vector',cost=Cost.HIGH,tags=('topic','nmf','full-materialization'),
    requires=('scikit-learn',),requires_full_materialization=True,
)
def topic(x,n_topics=8,**kwargs):
    # NMF topic factorization has no mergeable/streaming fit (scikit-learn's
    # NMF has no partial_fit, and the decomposition itself is an iterative
    # solve over the whole term-document matrix at once, not a per-batch
    # accumulation) -- unlike surprisal/BM25/TF-IDF/similarity above, this
    # genuinely cannot be made a mergeable stateful fit without an unverified,
    # possibly-fragile custom streaming NMF implementation. Per the fourth
    # audit pass's explicit guidance for exactly this situation ("if exact
    # streaming fitting cannot be implemented safely now, mark the operation
    # non-batchable/full-materialization and fail safely on large sources"),
    # this is marked requires_full_materialization=True: the whole corpus is
    # read as one grown batch (or the run is rejected if it cannot fit the
    # declared memory_budget, via the existing full-materialization resource
    # checks), giving the mathematically correct whole-corpus topic model
    # rather than per-batch-incomparable topic coordinates. Not currently
    # integrated with the public featm.fit()/.transform() train/test API --
    # a stated limitation, not a silent gap; see STEP2_AUDIT4_FIX_HANDOFF.md.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    docs=_docs(x);Z=TfidfVectorizer(max_features=2000).fit_transform(docs);k=min(n_topics,max(1,min(Z.shape)-1));W=NMF(n_components=k,random_state=0,init='nndsvda',max_iter=300).fit_transform(Z);return FeatureOutput(W,tuple(f'topic_{i:02d}' for i in range(k)))

# Sentiment/emotion adapters.
def _hf_classification(x,model_name,labels=None):
    try:from transformers import pipeline
    except ImportError as exc:raise ImportError("Install 'transformers' for sentiment/emotion features") from exc
    pipe=pipeline('text-classification',model=model_name,top_k=None);rows=[];names=None
    for s in _docs(x):
        res=pipe(s[:5000]);res=res[0] if res and isinstance(res[0],list) else res;d={r['label']:r['score'] for r in res};
        if names is None:names=sorted(d)
        rows.append([d.get(n,0) for n in names])
    return FeatureOutput(np.asarray(rows),tuple(names or []))
@_reg('text.sentiment.sentiment','sentiment',output_kind='vector',cost=Cost.EXTREME,tags=('sentiment',),requires=('transformers',))
def sentiment(x,model_name='distilbert-base-uncased-finetuned-sst-2-english',**kwargs):return _hf_classification(x,model_name)
@_reg('text.emotion.emotion','emotion',output_kind='vector',cost=Cost.EXTREME,tags=('emotion',),requires=('transformers',))
def emotion(x,model_name='j-hartmann/emotion-english-distilroberta-base',**kwargs):return _hf_classification(x,model_name)

# Embeddings.
@_reg('text.embeddings.sentence','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('sentence-embedding',),requires=('sentence-transformers',))
def sentence_embedding(x,model_name='sentence-transformers/all-MiniLM-L6-v2',**kwargs):
    try:from sentence_transformers import SentenceTransformer
    except ImportError as exc:raise ImportError("Install sentence-transformers for sentence embeddings") from exc
    Z=np.asarray(SentenceTransformer(model_name).encode(_docs(x),show_progress_bar=False));return FeatureOutput(Z,tuple(f'dim_{i:03d}' for i in range(Z.shape[1])))
@_reg('text.embeddings.document','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('document-embedding',),requires=('sentence-transformers',))
def document_embedding(x,**kwargs):return sentence_embedding(x,**kwargs)
document_embedding.__featm_parameters__ = forward_parameters(sentence_embedding, positional=1, bound=[])
@_reg('text.embeddings.transformer','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('transformer',),requires=('transformers','torch'))
def transformer_embedding(x,model_name='distilbert-base-uncased',**kwargs):
    try:import torch;from transformers import AutoTokenizer,AutoModel
    except ImportError as exc:raise ImportError('Install transformers and torch') from exc
    tok=AutoTokenizer.from_pretrained(model_name);model=AutoModel.from_pretrained(model_name);model.eval();rows=[]
    for s in _docs(x):
        inp=tok(s,return_tensors='pt',truncation=True,max_length=512)
        with torch.no_grad():h=model(**inp).last_hidden_state.mean(1).squeeze(0).cpu().numpy()
        rows.append(h)
    return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:03d}' for i in range(len(rows[0]))))
@_reg('text.embeddings.word','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('word-embedding',),requires=('user-model',))
def word_embedding(x,model=None,dim=None,**kwargs):
    if model is None:raise ValueError('text.embeddings.word requires a keyed-vector/model object supporting model[word]')
    docs=_docs(x);rows=[]
    for s in docs:
        vs=[]
        for w in _words(s):
            try:vs.append(np.asarray(model[w],float))
            except Exception:pass
        if vs:rows.append(np.mean(vs,axis=0))
        else:
            d=dim or getattr(model,'vector_size',0);rows.append(np.zeros(d))
    width=len(rows[0]);return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:03d}' for i in range(width)))

# Fourth audit pass, Finding 1: wire the genuinely mergeable out-of-core
# stateful encoders (featm.state) into the same STATEFUL_ENCODERS dict the
# execution engine already consults for the three Step-2 tabular encoders.
# The implementation functions above remain the small-data single-shot
# convenience (correct for one in-memory call); the out-of-core engine
# checks `spec.stateful and spec.name in STATEFUL_ENCODERS` and, when true,
# uses these encoders' update_state/finalize_state/transform_batch directly
# instead, bypassing the above functions entirely for multi-batch execution.
from ...state import STATEFUL_ENCODERS, Bm25Encoder, SimilarityEncoder, SurprisalEncoder, TfidfEncoder

STATEFUL_ENCODERS["text.information.surprisal"] = SurprisalEncoder()
STATEFUL_ENCODERS["text.bm25.bm25"] = Bm25Encoder()
STATEFUL_ENCODERS["text.tfidf.tfidf"] = TfidfEncoder()
STATEFUL_ENCODERS["text.semantics.similarity"] = SimilarityEncoder()
