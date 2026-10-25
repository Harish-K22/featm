"""Metadata contracts and deterministic optional-module failure checks."""
import builtins
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import featm
from featm._metadata import builtin_metadata
from featm._dependency_errors import annotate_missing_dependency

EXPECTED=json.loads((Path(__file__).parent/'fixtures/v059_metadata_expectations.json').read_text())
REQUIREMENTS=EXPECTED['requirements']
DESCRIPTIONS=EXPECTED['descriptions']


@pytest.mark.parametrize('name,requirements', list(REQUIREMENTS.items()))
def test_curated_requirements(name, requirements):
    assert featm.feature_info(name).requires == tuple(requirements)


@pytest.mark.parametrize('name,description', list(DESCRIPTIONS.items()))
def test_curated_descriptions(name, description):
    description=featm.feature_info(name).description
    import re
    assert re.sub(r"[^a-z0-9]", "", description.lower()) != re.sub(r"[^a-z0-9]", "", name.lower())


def block(monkeypatch, root):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == root or name.startswith(root + '.'):
            raise ModuleNotFoundError(f"No module named '{root}'", name=root)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def data(name):
    a = np.random.default_rng(59).normal(size=(2,128))
    if name.startswith(('pointcloud.', 'topology.')):
        return np.random.default_rng(59).normal(size=(2,30,3))
    if '.cross_series.' in name or name=='signal.synchronization.phase_locking':return np.stack([a,a[:,::-1]],axis=1)
    return a


SCIPY = [name for name, req in REQUIREMENTS.items() if req == ['scipy']]
@pytest.mark.parametrize('name', SCIPY)
def test_corrected_scipy_execution_missing(monkeypatch, name):
    # Block the import even if SciPy is already installed/imported elsewhere.
    block(monkeypatch,'scipy')
    feature = featm.create_feature(name)  # no eager dependency check
    with pytest.raises(ModuleNotFoundError) as error:feature(data(name))
    assert error.value.name == 'scipy'
    assert 'featm[scientific]' in str(error.value)
    assert name in str(error.value)


@pytest.mark.parametrize('name', SCIPY)
def test_corrected_scipy_execution_available(name):
    pytest.importorskip('scipy')
    result=featm.create_feature(name)(data(name))
    values=result.values if hasattr(result,'values') else result
    assert np.asarray(values).shape[0]==2


@pytest.mark.parametrize('policy',['raise','warn','skip'])
def test_error_policies(monkeypatch,policy):
    block(monkeypatch,'scipy');name='timeseries.statistical.skewness'
    if policy=='raise':
        with pytest.raises(ModuleNotFoundError,match=r'featm\[scientific\]'):
            featm.extract(data(name),features=[name],errors=policy)
    else:
        if policy=='warn':
            with pytest.warns(RuntimeWarning,match=r'featm\[scientific\]'):
                result=featm.extract(data(name),features=[name],errors=policy)
        else:result=featm.extract(data(name),features=[name],errors=policy)
        assert name in result.attrs['execution_report']['failed_features']


def test_narrow_exception_annotation():
    spec=featm.feature_info('timeseries.statistical.skewness')
    original=ModuleNotFoundError('original detail',name='scipy',path='test-path')
    annotate_missing_dependency(spec,original)
    assert original.name=='scipy' and original.path=='test-path'
    assert str(original).startswith('original detail')
    first=str(original);annotate_missing_dependency(spec,original);assert str(original)==first
    for error in [ModuleNotFoundError('submodule',name='scipy.defect'),
                  ModuleNotFoundError('typo',name='unrelated'),ImportError('DLL failure')]:
        before=str(error);annotate_missing_dependency(spec,error);assert str(error)==before


def test_inventory_and_base_conventions():
    specs=featm.REGISTRY.all()
    assert sorted(s.name for s in specs)==EXPECTED['feature_identifiers']
    assert [len(specs),len({s.modality for s in specs}),len({(s.modality,s.family) for s in specs}),sum(s.stateful for s in specs),sum(s.gpu_available for s in specs)]==[552,18,182,8,22]
    a=np.array([[1.,2.,np.nan,3.]])
    assert featm.create_feature('timeseries.statistical.mean')(a)[0]==2
    np.testing.assert_allclose(featm.create_feature('timeseries.statistical.std')(a),np.nanstd(a,axis=1))
    assert featm.create_feature('timeseries.complexity.permutation_entropy')(np.ones((1,20)))[0]==0
    assert np.isnan(featm.create_feature('timeseries.fractal.higuchi')(np.ones((1,5)))[0])
    with pytest.raises((TypeError,ValueError)):
        featm.create_feature('timeseries.complexity.permutation_entropy',order=4)
    assert 'timeseries.statistical.std' in featm.list_features('population')


def test_plugins_not_overridden():
    def plugin(x):return x
    metadata={'name':'timeseries.statistical.skewness','requires':()}
    assert builtin_metadata(plugin,metadata) is metadata
    class Callable:
        __module__=None
        def __call__(self,x):return x
    assert builtin_metadata(Callable(),metadata) is metadata


def test_topology_fallback_without_gudhi(monkeypatch):
    pytest.importorskip('scipy')
    block(monkeypatch,'gudhi')
    out=featm.create_feature('topology.persistent_homology.vietoris_rips')(data('topology.x'))
    assert out.values.shape==(2,9)
    assert np.all(out.values[:,3:]==0)


def test_short_dynamics_preserves_no_scipy_path(monkeypatch):
    block(monkeypatch,'scipy')
    value=featm.create_feature('timeseries.dynamical.lyapunov')(np.ones((1,3)))
    assert np.isnan(value[0])


@pytest.mark.parametrize('name', ['spatial.clustering.spatial_clustering','spatial.density.kernel_density'])
def test_corrected_sklearn_missing(monkeypatch,name):
    block(monkeypatch,'sklearn')
    with pytest.raises(ModuleNotFoundError,match=r'featm\[scientific\]'):
        featm.create_feature(name)(np.ones((10,2)))


def test_corrected_calendar_missing(monkeypatch):
    import pandas as pd
    block(monkeypatch,'holidays')
    with pytest.raises(ImportError,match="holidays") as error:
        featm.create_feature('tabular.datetime.holidays')(pd.Series(['2026-01-01']))
    assert isinstance(error.value.__cause__,ModuleNotFoundError)


def test_corrected_graph_model_path_missing(monkeypatch):
    block(monkeypatch,'networkx')
    with pytest.raises(ImportError,match=r'featm\[graph\]') as error:
        featm.create_feature('graph.embeddings.gnn',model=lambda g:np.ones(2))([np.eye(3)])
    assert error.value.__cause__.name=='networkx'


def test_clip_pillow_import_without_models(monkeypatch):
    # Satisfy preceding import statements only, then block PIL before loading a model.
    import sys,types
    torch=types.ModuleType('torch');transformers=types.ModuleType('transformers')
    transformers.CLIPModel=object;transformers.CLIPProcessor=object
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    block(monkeypatch,'PIL')
    with pytest.raises(ImportError,match='pillow') as error:
        featm.create_feature('image.embeddings.clip')(np.zeros((1,8,8,3)))
    assert error.value.__cause__.name=='PIL'


def test_exception_traceback_and_identity(monkeypatch):
    original=ModuleNotFoundError('precise original',name='scipy')
    old=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='scipy':raise original
        return old(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ModuleNotFoundError) as caught:
        featm.create_feature('timeseries.statistical.skewness')(np.ones((1,8)))
    assert caught.value is original
    import traceback
    assert any(frame.name=='guarded' for frame in traceback.extract_tb(original.__traceback__))

