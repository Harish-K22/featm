"""Targeted recurrence metadata, execution timing and published numerical parity."""
from pathlib import Path
import builtins,json,subprocess,sys,warnings
import numpy as np
import pytest
import featm

NAMES=['timeseries.recurrence.'+s for s in ['recurrence_rate','determinism','laminarity','trapping_time']]
FIXTURE=json.loads((Path(__file__).parent/'fixtures/v0510_recurrence_v059.json').read_text())

def block(monkeypatch,error=None):
    old=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='scipy' or name.startswith('scipy.'):
            if error is not None:raise error
            raise ModuleNotFoundError("No module named 'scipy'",name='scipy')
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)

@pytest.mark.parametrize('name',NAMES)
def test_public_requirement(name):
    assert featm.feature_info(name).requires==('scipy',)

@pytest.mark.parametrize('name',NAMES)
def test_missing_scipy_creation_and_execution(monkeypatch,name):
    block(monkeypatch)
    feature=featm.create_feature(name)  # metadata never enforces an eager import
    with pytest.raises(ModuleNotFoundError) as caught:
        feature(np.asarray(FIXTURE['inputs']['multiple']))
    error=caught.value
    assert error.name=='scipy'
    assert "No module named 'scipy'" in str(error)
    assert 'featm[scientific]' in str(error) and name in str(error)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('case',['short0','short1','short2','short3','all_nonfinite'])
def test_early_returns_without_scipy(monkeypatch,name,case):
    block(monkeypatch)
    result=featm.create_feature(name)(np.asarray(FIXTURE['inputs'][case]))
    assert np.asarray(result).shape==(2,)
    assert np.isnan(result).all()

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('policy',['raise','warn','skip'])
def test_error_policy_and_base_feature(monkeypatch,name,policy):
    block(monkeypatch);x=np.asarray(FIXTURE['inputs']['multiple'])
    base='timeseries.statistical.mean'
    if policy=='raise':
        with pytest.raises(ModuleNotFoundError) as caught:
            featm.extract(x,features=[base,name],errors=policy)
        assert caught.value.name=='scipy' and 'featm[scientific]' in str(caught.value)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out=featm.extract(x,features=[base,name],errors=policy)
        assert list(out.columns)==[base] and out.shape==(2,1)
        np.testing.assert_allclose(out[base].to_numpy(),x.mean(1),rtol=0,atol=0)
        detail=out.attrs['execution_report']['failed_features'][name]
        assert 'ModuleNotFoundError' in detail and 'featm[scientific]' in detail
        if policy=='warn':assert any('featm[scientific]' in str(w.message) for w in caught)
        else:assert not caught
    np.testing.assert_allclose(featm.extract(x,features=[base],errors='raise').to_numpy().ravel(),x.mean(1),rtol=0,atol=0)

@pytest.mark.parametrize('name',NAMES)
def test_unrelated_importerror_not_relabeled(monkeypatch,name):
    original=ImportError('binary loader problem in scipy')
    block(monkeypatch,original)
    with pytest.raises(ImportError) as caught:
        featm.create_feature(name)(np.asarray(FIXTURE['inputs']['normal']))
    assert caught.value is original
    assert str(caught.value)=='binary loader problem in scipy'
    assert 'featm[scientific]' not in str(caught.value)

@pytest.mark.parametrize('name',list(FIXTURE['expected']))
@pytest.mark.parametrize('case',list(FIXTURE['inputs']))
def test_published_v059_numerical_parity(name,case):
    pytest.importorskip('scipy')
    result=featm.create_feature(name)(np.asarray(FIXTURE['inputs'][case]))
    values=np.asarray(result.values if hasattr(result,'values') else result)
    expected=FIXTURE['expected'][name][case]
    assert list(values.shape)==expected['shape']
    np.testing.assert_allclose(values,expected['values'],rtol=0,atol=0,equal_nan=True)

def test_related_recurrence_network_requirement():
    assert featm.feature_info('timeseries.recurrence.recurrence_network').requires==('scipy',)

def test_import_does_not_load_scipy():
    code="import sys;sys.path.insert(0,sys.argv[1]);import featm;assert 'scipy' not in sys.modules;print('lazy import PASS')"
    subprocess.run([sys.executable,'-c',code,str(Path(featm.__file__).resolve().parent.parent)],check=True,capture_output=True,text=True)
