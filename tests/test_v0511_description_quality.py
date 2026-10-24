"""Catalogue quality invariants and independent representative semantic checks."""
import ast
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import featm
from featm import _metadata

FIXTURES = Path(__file__).parent / 'fixtures'
REPRESENTATIVE = json.loads((FIXTURES/'v0511_description_semantics.json').read_text())


def test_catalogue_description_quality():
    specs=featm.REGISTRY.all()
    assert [len(specs),len({s.modality for s in specs}),len({(s.modality,s.family) for s in specs}),sum(s.stateful for s in specs),sum(s.gpu_available for s in specs)]==[552,18,182,7,22]
    assert set(_metadata.DESCRIPTIONS)<={s.name for s in specs}
    normalize=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
    for s in specs:
        assert s.description.strip(),s.name
        assert normalize(s.description)!=normalize(s.name),s.name
        assert not any(p in s.description.lower() for p in ('todo','tbd','powerful feature','valuable insights','useful for machine learning','captures important patterns')),s.name


def test_no_duplicate_curated_keys():
    tree=ast.parse(inspect.getsource(_metadata))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'DESCRIPTIONS','REQUIREMENTS'} for t in n.targets):
            assert isinstance(n.value,ast.Dict)
            keys=[ast.literal_eval(k) for k in n.value.keys]
            assert len(keys)==len(set(keys))


def test_prior_reviewed_semantics_preserved():
    old=json.loads((FIXTURES/'v059_metadata_expectations.json').read_text())['descriptions']
    for name,description in old.items():
        if name.startswith('topology.'):
            assert 'GUDHI' in featm.feature_info(name).description
            assert all(term in featm.feature_info(name).description for term in ('SciPy','H0','MST','fall'))
        else:
            assert featm.feature_info(name).description==description


def test_representative_reviewed_words_and_coverage():
    assert {featm.feature_info(n).modality for n in REPRESENTATIVE}==set(featm.list_modalities())
    for name,description in REPRESENTATIVE.items():
        assert featm.feature_info(name).description==description


def test_reinitialization_is_stable():
    code='import featm,json; print(json.dumps({s.name:s.description for s in featm.REGISTRY.all()},sort_keys=True))'
    first=subprocess.check_output([sys.executable,'-c',code],text=True)
    second=subprocess.check_output([sys.executable,'-c',code],text=True)
    assert first==second


def test_third_party_registration_unaffected(monkeypatch):
    from featm.registry import FeatureRegistry
    import featm.registry as registry
    isolated=FeatureRegistry();monkeypatch.setattr(registry,'REGISTRY',isolated)
    @registry.register_feature(name='timeseries.statistical.mean',modality='timeseries',family='statistical',description='Plugin-defined meaning.',requires=('plugin-backend',))
    def plugin(x):return np.asarray(x)+2
    spec=isolated.get('timeseries.statistical.mean')
    assert spec.description=='Plugin-defined meaning.' and spec.requires==('plugin-backend',)
    np.testing.assert_array_equal(plugin([1,2]),[3,4])


def values(name,x,**params):
    result=featm.create_feature(name,**params)(x)
    return result.values if isinstance(result,featm.FeatureOutput) else result


def test_representative_actual_semantics():
    # These constants distinguish power from density, proxy from named theory,
    # normalization conventions and wrapper/adaptor behavior across modalities.
    np.testing.assert_allclose(values('audio.waveform.dynamic_range',np.ones(20)),[0])
    np.testing.assert_allclose(values('event.timing.event_rate',[np.array([0.,2.,4.])]),[.5])
    np.testing.assert_allclose(values('geometry.distances.euclidean',[[0.,0.],[2.,0.]]),[1,1])
    nx=pytest.importorskip('networkx')
    assert values('graph.cycles.cycle_count',nx.complete_graph(4))[0]==3  # cycle basis, not all cycles
    np.testing.assert_allclose(values('image.color.red_mean',np.array([[[[2.,10.,20.]]]])),[2])
    np.testing.assert_allclose(values('information.entropy.shannon',[[1.,1.]]),[np.log(2)])
    np.testing.assert_allclose(values('matrix.algebra.sparsity',[[0.,1e-10],[1.,2.]]),[.5])
    np.testing.assert_allclose(values('multimodal.fusion.late_fusion',([[2.,4.,9.]],[[4.,8.]])),[[3.,6.]])
    np.testing.assert_allclose(values('pointcloud.geometry.bbox_volume',np.array([[0.,0.,0.],[2.,3.,4.]])),[24])
    np.testing.assert_allclose(values('sequence.entropy.shannon',['aabb']),[np.log(2)])
    np.testing.assert_allclose(values('signal.frequency.psd',[[1.,-1.,1.,-1.]],n_bins=3),[[0,0,16]])
    # NaN != 0 is true in the tabular logical implementation.
    import pandas as pd
    actual=featm.feature_info('tabular.interactions.logical').implementation(pd.Series([np.nan]),pd.Series([1.]))[0].values
    np.testing.assert_array_equal(actual,[[1,1,0]])
    assert values('text.lexical.word_count',["Don't re-test."])[0]==2
    np.testing.assert_allclose(values('video.motion.frame_difference_energy',np.array([[[[0.]],[[2.]]]])),[4])


def test_optional_representative_semantics():
    pytest.importorskip('scipy')
    np.testing.assert_allclose(values('timeseries.recurrence.recurrence_rate',np.ones((1,5))),[.75])
    # The so-called persistence image is an unblurred death-weighted histogram.
    np.testing.assert_allclose(values('topology.persistence.image',np.array([[0.,0.],[1.,0.],[3.,0.]]),bins=2),[[0.,3.]])
    x={'coordinates':np.array([[0.,0.],[1.,0.],[2.,0.]]),'values':np.array([1.,2.,9.])}
    np.testing.assert_allclose(values('spatial.hotspots.getis_ord',x,k=2),[5.5,5.,1.5])
    pytest.importorskip('sklearn')
    p=np.array([[0.,0.],[1.,0.],[0.,1.],[2.,2.]])
    d=values('manifold.dimension.local_dimension',p,k=2)
    np.testing.assert_allclose(values('manifold.dimension.intrinsic_dimension',p,k=2),np.repeat(np.nanmedian(d),4))


def test_unavailable_adapters_described_honestly():
    for name in ('shot','spin_images'):
        feature='pointcloud.descriptors.'+name
        assert 'always raises ImportError' in featm.feature_info(feature).description
        with pytest.raises(ImportError,match='plugin pack'):
            featm.create_feature(feature)(np.zeros((3,3)))
