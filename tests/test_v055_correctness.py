"""Canonical v0.5.5 defects: original inputs plus boundary/path/policy controls."""
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import featm
from featm.sources import PandasSource, CSVSource
from featm.state import STATEFUL_ENCODERS

Z='tabular.scaling.zscore'; MM='tabular.scaling.minmax'; CF='tabular.encoding.category_frequency'
ABS='tabular.numeric.abs'; TF='text.tfidf.tfidf'; IMAGE='image.intensity.mean'

def extract(data,names,**kw): return featm.extract(data,features=names,errors=kw.pop('errors','raise'),**kw)
def frame(out): return out if isinstance(out,pd.DataFrame) else out.to_pandas()

@pytest.mark.parametrize('runtime',['local','dask'])
@pytest.mark.parametrize('policy',['skip','warn'])
def test_v055_c01_prepared_active_set(runtime,policy):
    if runtime=='dask': pytest.importorskip('distributed')
    name='sequence.distance.edit_distance'; df=pd.DataFrame({'text':['aaa','bbb','abc','abcd']})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out=extract(df,[name],errors=policy,execution='distributed',distributed_backend=runtime,distributed_partition_rows=1)
    assert name in out.report.skipped_features and name not in out.report.completed_features and out.shape==(0,0)
    if policy=='warn': assert caught
    df=pd.DataFrame({'x':[1.,2.]})
    out=extract(df,[ABS,Z],columns={ABS:'x',Z:'missing'},errors=policy,execution='distributed',distributed_backend=runtime,distributed_partition_rows=1)
    assert out.columns==[ABS] and Z not in out.report.completed_features
    np.testing.assert_array_equal(frame(out).values,[[1],[2]])

def test_v055_c02_categorical_dtype_and_missing(tmp_path):
    f=featm.fit(pd.DataFrame({'c':[1,2,2,3]}),features=[CF],columns={CF:'c'},errors='raise')
    np.testing.assert_allclose(f.transform(pd.DataFrame({'c':[1.,2.,np.nan]})).to_numpy().ravel(),[.25,.5,0])
    # Typed strings, bools and missing must not collapse into numeric or literal sentinel categories.
    data=pd.DataFrame({'c':pd.Series([1,1.,'1',True,None,'<NA>'],dtype=object)})
    out=extract(data,[CF],columns={CF:'c'},batch_rows=1)
    np.testing.assert_allclose(out.to_numpy().ravel(),[2/6,2/6,1/6,1/6,1/6,1/6])
    path=tmp_path/'cats.csv'; path.write_text('c,z\n1,0\n2,0\n2,0\n1,0\n3,0\n,0\n1,0\n2,0\n')
    a=extract(path,[CF],columns={CF:'c'},batch_rows=1); b=extract(path,[CF],columns={CF:'c'},batch_rows=4)
    pd.testing.assert_frame_equal(frame(a),frame(b))

@pytest.mark.parametrize('kind',['jsonl','parquet'])
def test_v055_c02_parser_dtype_variants(tmp_path,kind):
    values=[1,2,2,1,3,None,1,2]; path=tmp_path/f'input.{kind}'
    if kind=='parquet':
        pa=pytest.importorskip('pyarrow'); import pyarrow.parquet as pq
        pq.write_table(pa.table({'c':pa.array(values,type=pa.int64())}),path,row_group_size=4)
    else:
        path.write_text('\n'.join(json.dumps({'c':v}) for v in values)+'\n')
    opts=dict(columns={CF:'c'})
    small=extract(path,[CF],batch_rows=8,**opts)
    for count in [1,4]:
        batched=extract(path,[CF],batch_rows=count,**opts)
        distributed=extract(path,[CF],execution='distributed',distributed_backend='local',distributed_partition_rows=count,**opts)
        pd.testing.assert_frame_equal(frame(small),frame(batched)); pd.testing.assert_frame_equal(frame(small),frame(distributed))

def test_v055_c04_legacy_fitted_artifact_guard(tmp_path):
    import pickle
    fitted=featm.fit(pd.DataFrame({'x':[1.,2.]}),features=[Z],columns={Z:'x'},errors='raise')
    path=tmp_path/'artifact'; fitted.save(path)
    payload=pickle.loads(path.read_bytes()); payload['artifact_format_version']=1
    path.write_bytes(pickle.dumps(payload))
    assert featm.load_fitted(path).transform(pd.DataFrame({'x':[3.]})).shape==(1,1)
    payload['columns']=None; path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='refit'): featm.load_fitted(path)

@pytest.mark.parametrize('policy',['raise','skip','warn'])
def test_v055_c04_bound_fitted_column(tmp_path,policy):
    f=featm.fit(pd.DataFrame({'income':[0.,2.,4.],'city':['a','b','a']}),features=[Z],errors=policy)
    f.save(tmp_path/'fit'); f=featm.load_fitted(tmp_path/'fit')
    if policy=='raise':
        with pytest.raises(ValueError,match='income'): f.transform(pd.DataFrame({'age':[20.,30.]}))
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); out=f.transform(pd.DataFrame({'age':[20.,30.]}))
        assert Z not in out.columns and Z in out.report.skipped_features
        if policy=='warn': assert caught
    out=f.transform(pd.DataFrame({'new':[99.],'city':['z'],'income':[4.]}))
    np.testing.assert_allclose(out.to_numpy(),[[np.sqrt(1.5)]])

def test_v055_c05_output_order(tmp_path):
    names=[Z,'tabular.numeric.powers',ABS]; df=pd.DataFrame({'x':np.arange(1.,6.)}); cols={n:'x' for n in names}
    small=extract(df.x,names); local=extract(df,names,columns=cols,batch_rows=2,checkpoint=tmp_path/'cp')
    dist=extract(df,names,columns=cols,execution='distributed',distributed_backend='local',distributed_partition_rows=2)
    replay=extract(df,names,columns=cols,batch_rows=2,checkpoint=tmp_path/'cp')
    for out in [local,dist,replay]: pd.testing.assert_frame_equal(small,frame(out),check_exact=False,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('policy',['raise','skip','warn'])
@pytest.mark.parametrize('name',[Z,MM])
def test_v055_c06_c07_invalid_scaler_state(name,policy):
    for values in [['bad','worse'],[1.,np.inf],[np.nan,np.nan],[]]:
        df=pd.DataFrame({'x':pd.Series(values)})
        if policy=='raise':
            with pytest.raises(ValueError): extract(df,[name],columns={name:'x'},errors=policy,batch_rows=1)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); out=extract(df,[name],columns={name:'x'},errors=policy,batch_rows=1)
            assert name not in out.columns
            if policy=='warn': assert caught
    fitted=featm.fit(pd.DataFrame({'x':[2.,np.nan,2.]}),features=[name],columns={name:'x'},errors='raise')
    np.testing.assert_array_equal(fitted.transform(pd.DataFrame({'x':[2.,3.]})).to_numpy(),[[0],[1]])

def test_v055_c08_c09_c15_c24_tfidf_contract(tmp_path):
    text=pd.Series(["foo-bar","foo'bar","foo_bar","a CAT",None,np.nan,pd.NA,'nan','none','', ' '*3])
    df=pd.DataFrame({'t':text}); kwargs=dict(columns={TF:'t'},batch_rows=2)
    small=extract(text,[TF]); local=extract(df,[TF],**kwargs,checkpoint=tmp_path/'cp')
    resumed=extract(df,[TF],**kwargs,checkpoint=tmp_path/'cp')
    dist=extract(df,[TF],columns={TF:'t'},execution='distributed',distributed_backend='local',distributed_partition_rows=2)
    for result in [local,resumed,dist]: pd.testing.assert_frame_equal(small,frame(result))
    fitted=featm.fit(df,features=[TF],columns={TF:'t'},errors='raise')
    vocab=fitted.fitted_states[TF]['vocab']; assert len(vocab)==len(small.columns) and len(set(small.columns))==len(vocab)
    assert not small.iloc[4:7].to_numpy().any() and small.iloc[7:9].to_numpy().any()
    empty=fitted.transform(pd.DataFrame({'t':pd.Series([],dtype=str)})); assert empty.shape==(0,len(vocab)) and empty.columns==list(small.columns)
    fitted.save(tmp_path/'state'); again=featm.load_fitted(tmp_path/'state')
    assert again.transform(pd.DataFrame({'t':[]})).columns==empty.columns
    missing=extract(pd.Series([None,np.nan,pd.NA]),[TF]); assert 'nan' not in ' '.join(missing.columns) and missing.shape[0]==3
    for name in ['text.lexical.word_count','text.information.surprisal']:
        out=extract(text,[name]); assert out.iloc[4:7].to_numpy().sum()==0

class InterruptFit(PandasSource):
    def iter_batches(self,**kwargs):
        for i,b in enumerate(super().iter_batches(**kwargs)):
            if i==1: raise RuntimeError('intentional fit interruption')
            yield b

@pytest.mark.parametrize('damage',['delete','truncate'])
def test_v055_c10_missing_fit_snapshot(tmp_path,damage):
    df=pd.DataFrame({'x':[0.,2.,10.,12.]}); kw=dict(columns={Z:'x'},batch_rows=2,memory_budget='16MB',checkpoint=tmp_path/'cp')
    with pytest.raises(RuntimeError,match='intentional'): extract(InterruptFit(df),[Z],**kw)
    meta=json.loads((tmp_path/'cp/checkpoint.json').read_text()); state=tmp_path/'cp/state'/meta['fit_accumulator_file']
    if damage=='delete': state.unlink()
    else: state.write_bytes(b'')
    with pytest.raises(featm.matrix.CheckpointCorruptionError,match='snapshot'): extract(df,[Z],**kw)

@pytest.mark.parametrize('name,good,bad',[
    (TF,{'max_features':3},{'max_feature':3}),
    ('event.timing.recency',{'reference_time':9.},{'reference_tme':9.}),
    ('timeseries.complexity.permutation_entropy',{}, {'_fn':lambda x:42}),
    ('timeseries.spectral.periodogram',{'n_bins':3},{'n_bin':3}),
    (Z,{}, {'bogus':1}),
])
def test_v055_c11_parameters(name,good,bad):
    featm.create_feature(name,**good)
    with pytest.raises(ValueError,match='unknown or internal'): featm.create_feature(name,**bad)
    with pytest.raises(ValueError,match='unknown or internal'): extract(pd.Series([1.,2.]),[name],params={name:bad})

@pytest.mark.parametrize('fail_calls',[[3],[3,6]])
def test_v055_c13_transactional_memory_retry(monkeypatch,fail_calls):
    import featm.state as state
    original=state._real_item_bytes; calls=0
    def flaky(value,**kw):
        nonlocal calls
        calls+=1
        if calls in fail_calls: raise MemoryError('injected partial mutation')
        return original(value,**kw)
    df=pd.DataFrame({'c':list('abcdabcd')}); clean=extract(df,[CF],columns={CF:'c'},memory_budget='64MB')
    monkeypatch.setattr(state,'_real_item_bytes',flaky)
    retried=extract(df,[CF],columns={CF:'c'},memory_budget='64MB')
    pd.testing.assert_frame_equal(frame(clean),frame(retried)); assert retried.report.retries[CF]>=len(fail_calls)

@pytest.mark.parametrize('name',['timeseries.spectral.fft','timeseries.spectral.cepstrum','signal.cepstral.cepstrum'])
def test_v055_c14_layout_numerics(name):
    x=np.random.default_rng(0).normal(size=(6,64)); reference=extract(x,[name])
    for arr in [np.asfortranarray(x),x.copy(),x[:,::-1][:,::-1]]: pd.testing.assert_frame_equal(reference,extract(arr,[name]))
    for batch in [1,3,6]: pd.testing.assert_frame_equal(reference,frame(extract(pd.DataFrame(x),[name],batch_rows=batch)))
    if name.endswith('.fft'): assert np.all(reference.filter(like='phase_00').values==0)

@pytest.mark.parametrize('execution',['local','distributed'])
@pytest.mark.parametrize('position',[0,1,2])
@pytest.mark.parametrize('policy',['raise','skip','warn'])
def test_v055_c17_error_policy(execution,position,policy):
    values=[np.ones((2,2))]*3; values[position]='bad'
    df=pd.DataFrame({'im':values}); opts=dict(columns={IMAGE:'im'},errors=policy)
    opts.update(dict(batch_rows=1) if execution=='local' else dict(execution='distributed',distributed_backend='local',distributed_partition_rows=1))
    if policy=='raise':
        with pytest.raises((ValueError,featm.distributed.contracts.DistributedFeatureError)): extract(df,[IMAGE],**opts)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); out=extract(df,[IMAGE],**opts)
        if position: assert np.isnan(frame(out).iloc[position,0])
        else: assert IMAGE not in out.columns
        if policy=='warn': assert caught

@pytest.mark.parametrize('execution',['small','local','distributed'])
def test_v055_c18_strict_gpu(execution):
    name='timeseries.complexity.permutation_entropy'; x=np.arange(24.).reshape(3,8); kw={}
    data=x
    if execution!='small': data=pd.DataFrame(x)
    if execution=='local': kw={'batch_rows':1}
    if execution=='distributed': kw=dict(execution='distributed',distributed_backend='local',distributed_partition_rows=1)
    with pytest.raises((featm.gpu.GPUImplementationMissingError,featm.distributed.contracts.DistributedFeatureError)):
        extract(data,[name],backend='gpu',gpu_fallback='strict',**kw)
    out=extract(data,[name],backend='gpu',gpu_fallback='fallback',**kw)
    report=out.attrs['backend_report'] if isinstance(out,pd.DataFrame) else out.report.backend_report
    assert report[name]['actual_backend']=='cpu' and report[name]['fallback_reason']=='feature_not_gpu_capable'
    out=extract(data,[name],backend='auto',gpu_fallback='strict',**kw)
    report=out.attrs['backend_report'] if isinstance(out,pd.DataFrame) else out.report.backend_report
    assert report[name]['actual_backend']=='cpu' and report[name]['fallback_reason']=='feature_not_gpu_capable'

@pytest.mark.parametrize('name',[ABS,Z])
@pytest.mark.parametrize('execution',['small','local','distributed'])
def test_v055_c18_tabular_backend_contract(name,execution):
    df=pd.DataFrame({'x':[1.,2.,3.]}); kw=dict(columns={name:'x'},batch_rows=1)
    if execution=='small': kw={}
    if execution=='distributed': kw.update(execution='distributed',distributed_backend='local')
    with pytest.raises((featm.gpu.GPUImplementationMissingError,featm.distributed.contracts.DistributedFeatureError)):
        extract(df,[name],backend='gpu',gpu_fallback='strict',**kw)
    out=extract(df,[name],backend='auto',**kw)
    report=out.attrs['backend_report'] if isinstance(out,pd.DataFrame) else out.report.backend_report
    assert report[name]['actual_backend']=='cpu'

@pytest.mark.parametrize('cpu,gpu,passed',[(1,np.inf,False),(np.inf,1,False),(np.inf,-np.inf,False),(-np.inf,-np.inf,True),(np.nan,np.nan,True),(np.nan,1,False)])
def test_v055_c19_validator(cpu,gpu,passed):
    from featm.gpu.dispatch import _validate_against_cpu
    result=_validate_against_cpu(featm.feature_info('timeseries.statistical.mean'),np.array([cpu]),np.array([gpu]),(1e-4,1e-6))
    assert result['passed']==passed

def test_v055_c20_arrow_table():
    pa=pytest.importorskip('pyarrow'); import pyarrow.dataset as ds
    df=pd.DataFrame({'x':[-1.,2.,3.],'other':[1,2,3]}); table=pa.Table.from_pandas(df)
    for obj in [table,ds.dataset(table),pa.table({'x':pa.array([],type=pa.float64())})]:
        out=extract(featm.ArrowSource(obj),[ABS],columns={ABS:'x'},batch_rows=1)
        assert out.columns==[ABS]
        np.testing.assert_array_equal(out.to_numpy().ravel(),np.abs(obj.to_table()['x'].to_numpy() if hasattr(obj,'to_table') else obj['x'].to_numpy()))

def test_v055_c23_replay_history(tmp_path):
    df=pd.DataFrame({'im':[np.ones((2,2)),'bad',np.ones((2,2))*3]})
    kw=dict(columns={IMAGE:'im'},errors='skip',execution='distributed',distributed_backend='local',distributed_partition_rows=1,checkpoint=tmp_path/'cp')
    a=extract(df,[IMAGE],**kw); b=extract(df,[IMAGE],**kw)
    pd.testing.assert_frame_equal(frame(a),frame(b))
    assert a.report.distributed_report['degraded_partitions_by_feature']==b.report.distributed_report['degraded_partitions_by_feature']=={IMAGE:1}
    assert a.report.warnings==b.report.warnings

def test_v055_c23_partial_priming_failure_replay(tmp_path):
    first='signal.time_domain.rms'; missing='signal.decomposition.emd'
    df=pd.DataFrame(np.arange(24.).reshape(3,8)); kw=dict(errors='skip',execution='distributed',distributed_backend='local',distributed_partition_rows=1,checkpoint=tmp_path/'cp')
    a=extract(df,[first,missing],**kw); b=extract(df,[first,missing],**kw)
    assert a.report.failed_features==b.report.failed_features and missing in b.report.failed_features
    pd.testing.assert_frame_equal(frame(a),frame(b))

def test_v055_c11_forwarded_parameters():
    featm.create_feature('sequence.markov.markov',buckets=4)
    featm.create_feature('graph.spectral.laplacian',k=3)
    featm.create_feature('text.embeddings.document',model_name='unused-test-model')

def test_v055_c06_numeric_object_inputs():
    df=pd.DataFrame({'x':pd.Series(['1','2','3'],dtype=object)})
    small=extract(df.x,[Z,ABS]); mapped=extract(df,[Z,ABS],columns={Z:'x',ABS:'x'},batch_rows=1)
    pd.testing.assert_frame_equal(small,frame(mapped))

@pytest.mark.parametrize('name',[Z,MM])
def test_v055_c07_overflow_fails_closed(name):
    with pytest.raises(ValueError,match='Nonfinite'):
        featm.fit(pd.DataFrame({'x':[-1e308,1e308]}),features=[name],columns={name:'x'},errors='raise')

def test_v055_c15_missing_scalar_values():
    from featm.contracts import normalize_text
    for value in [None,np.nan,pd.NA,pd.NaT]: assert normalize_text(value)==''
    for value in ['nan','none','NaT']: assert normalize_text(value)==value

@pytest.mark.parametrize('policy',['raise','skip','warn'])
@pytest.mark.parametrize('execution',['local','distributed'])
def test_v055_c25_finalization_policy(policy,execution,monkeypatch):
    original=STATEFUL_ENCODERS[TF].finalize_state
    def fail(*args,**kwargs): raise RuntimeError('finalizer failure')
    monkeypatch.setattr(STATEFUL_ENCODERS[TF],'finalize_state',fail)
    df=pd.DataFrame({'t':['cat dog','mouse']}); other='text.information.surprisal'; kw=dict(columns={TF:'t',other:'t'},errors=policy,batch_rows=1)
    if execution=='distributed': kw.update(execution='distributed',distributed_backend='local')
    if policy=='raise':
        with pytest.raises(RuntimeError,match='finalizer'): extract(df,[TF,other],**kw)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); out=extract(df,[TF,other],**kw)
        assert out.columns==[other] and TF in out.report.failed_features
        if policy=='warn': assert caught

def test_v055_c26_csv_parser_identity(tmp_path):
    p=tmp_path/'input.csv'; p.write_text('x\n1\n2\n3\n'); cp=tmp_path/'cp'
    kw=dict(columns={ABS:'x'},batch_rows=1,checkpoint=cp)
    first=extract(CSVSource(p,skiprows=[1]),[ABS],**kw)
    replay=extract(CSVSource(p,skiprows=[1]),[ABS],**kw); pd.testing.assert_frame_equal(frame(first),frame(replay))
    other_io=extract(CSVSource(p,skiprows=[1],memory_map=True,schema_sample_rows=1),[ABS],**kw)
    pd.testing.assert_frame_equal(frame(first),frame(other_io))
    with pytest.raises(featm.checkpoint.CheckpointIncompatibleError): extract(CSVSource(p,skiprows=[2]),[ABS],**kw)
    with pytest.raises(featm.sources.SourceNotCheckpointableError): extract(CSVSource(p,skiprows=lambda x:False),[ABS],**kw)

def test_v055_c27_graph_identity(tmp_path):
    nx=pytest.importorskip('networkx'); g=nx.path_graph(3); nx.set_edge_attributes(g,1.,'weight')
    name='graph.spectral.eigenvalues'; df=pd.DataFrame({'g':[g]}); kw=dict(columns={name:'g'},checkpoint=tmp_path/'cp',batch_rows=1)
    first=extract(df,[name],**kw); replay=extract(pd.DataFrame({'g':[g.copy()]}),[name],**kw)
    pd.testing.assert_frame_equal(frame(first),frame(replay))
    g[0][1]['weight']=10
    with pytest.raises(featm.checkpoint.CheckpointIncompatibleError): extract(df,[name],**kw)

@pytest.mark.parametrize('policy',['raise','skip','warn'])
def test_v055_c28_tabular_dataframe(policy):
    df=pd.DataFrame({'x':[1.,2.,3.]})
    series=extract(df.x,[ABS,Z],errors=policy); small=extract(df,[ABS,Z],errors=policy)
    pd.testing.assert_frame_equal(series,small)
    mapped=extract(df.assign(other=['a','b','c']),[ABS,Z],columns={ABS:'x',Z:'x'},errors=policy)
    pd.testing.assert_frame_equal(series,frame(mapped))

def test_v055_c28_small_failure_is_reported():
    out=extract(pd.DataFrame({'x':['bad','input']}),[ABS],errors='skip')
    assert ABS in out.attrs['execution_report']['failed_features']
    out=extract(pd.DataFrame({'x':[1.,2.],'y':[3.,4.]}),[ABS],errors='skip')
    assert ABS in out.attrs['execution_report']['failed_features']

@pytest.mark.parametrize('backend',['gpu','auto'])
@pytest.mark.parametrize('offset',[1e6,1e8])
def test_v055_c03_real_cuda_large_offset(backend,offset):
    from featm.gpu.backend import cupy_available
    if not cupy_available(): pytest.skip('NOT EXECUTED — CuPy/CUDA unavailable')
    names=['timeseries.statistical.std','timeseries.spectral.centroid']
    x=offset+np.random.default_rng(0).normal(size=(4096,256))
    cpu=extract(x,names,backend='cpu'); device=extract(x,names,backend=backend,gpu_fallback='strict')
    np.testing.assert_allclose(cpu.values,device.values,rtol=1e-4,atol=1e-6)
    assert device.attrs['backend_report'][names[1]]['actual_backend']=='gpu'
    assert device.attrs['backend_report'][names[1]]['input_dtype']=='float64'
