"""Automatic RAM observations must not reinterpret committed state or row boundaries."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError
from featm.matrix import CheckpointCorruptionError
from featm.resources import ResourceBudgetTooSmallError
from featm.sources import PandasSource

Z = 'tabular.scaling.zscore'
CF = 'tabular.encoding.category_frequency'


@pytest.fixture(autouse=True)
def io():
    pytest.importorskip('pyarrow')


def ram(monkeypatch, mib):
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=int(mib * 1024**2)))


def options(tmp_path, engine):
    kw = dict(features=[Z], columns={Z: 'x'}, errors='raise', checkpoint=tmp_path / 'cp')
    if engine == 'local':
        kw.update(batch_rows=5)
    else:
        kw.update(execution='distributed', distributed_backend=engine,
                  distributed_partition_rows=5, distributed_workers=2,
                  distributed_processes=False)
        if engine == 'dask':
            pytest.importorskip('distributed')
    return kw


def data():
    return pd.DataFrame({'x': np.arange(20, dtype=float)})


def manifest(tmp_path, engine):
    return tmp_path / 'cp' / ('checkpoint.json' if engine == 'local' else 'distributed_checkpoint.json')


@pytest.mark.parametrize('engine', ['local', 'local_distributed', 'dask'])
def test_auto_completed_ram_changes(tmp_path, monkeypatch, engine):
    kw = options(tmp_path, 'local' if engine == 'local_distributed' else engine)
    if engine == 'local_distributed':
        kw.update(execution='distributed', distributed_backend='local', distributed_partition_rows=5)
        kw.pop('batch_rows')
    ram(monkeypatch, 900)
    first = featm.extract(data(), **kw)
    path = manifest(tmp_path, 'distributed' if engine == 'local_distributed' else engine)
    saved = json.loads(path.read_text())
    ram(monkeypatch, 800)
    # No data batches may be read by a fully committed checkpoint resume.
    def forbidden(*args, **kwargs):
        raise AssertionError('completed checkpoint read input batches')
    monkeypatch.setattr(PandasSource, 'iter_batches', forbidden)
    second = featm.extract(data(), **kw)
    pd.testing.assert_frame_equal(first.to_pandas(), second.to_pandas())
    assert json.loads(path.read_text()) == saved


class Interrupted(PandasSource):
    def iter_batches(self, **kw):
        for i, batch in enumerate(super().iter_batches(**kw)):
            if i == 1:
                raise RuntimeError('intentional interrupted fit')
            yield batch


@pytest.mark.parametrize('mode', ['exact', 'approximate'])
def test_local_partial_auto_preserves_state_and_sampling(tmp_path, monkeypatch, mode):
    df = pd.DataFrame({'c': [f'category-{i}' for i in range(30)]})
    kw = dict(features=[CF], columns={CF: 'c'}, batch_rows=5, mode=mode,
              approximate_sample_size=8, errors='raise')
    ram(monkeypatch, 900)
    clean = featm.extract(df, **kw)
    with pytest.raises(RuntimeError, match='intentional'):
        featm.extract(Interrupted(df), checkpoint=tmp_path / 'cp', **kw)
    before = json.loads(manifest(tmp_path, 'local').read_text())
    ram(monkeypatch, 800)
    resumed = featm.extract(df, checkpoint=tmp_path / 'cp', **kw)
    pd.testing.assert_frame_equal(clean.to_pandas(), resumed.to_pandas())
    assert json.loads(manifest(tmp_path, 'local').read_text())['settings'] == before['settings']


@pytest.mark.parametrize('engine', ['local', 'dask'])
def test_partial_distributed_auto_state_equivalent(tmp_path, monkeypatch, engine):
    from featm.distributed.checkpoint import DistributedCheckpointManager
    kw = options(tmp_path, engine)
    kw.update(execution='distributed', distributed_backend=engine, distributed_partition_rows=5)
    kw.pop('batch_rows', None)
    ram(monkeypatch, 900)
    clean = featm.extract(data(), **{k: v for k, v in kw.items() if k != 'checkpoint'})
    original = DistributedCheckpointManager.commit_fit_partition
    calls = []
    def interrupted(self, index, states, **kwargs):
        original(self, index, states, **kwargs)
        calls.append(index)
        raise RuntimeError('intentional interrupted fit')
    with monkeypatch.context() as m:
        m.setattr(DistributedCheckpointManager, 'commit_fit_partition', interrupted)
        with pytest.raises(RuntimeError, match='intentional'):
            featm.extract(data(), **kw)
    assert calls
    before = json.loads(manifest(tmp_path, 'distributed').read_text())
    ram(monkeypatch, 800)
    resumed = featm.extract(data(), **kw)
    pd.testing.assert_frame_equal(clean.to_pandas(), resumed.to_pandas())
    after = json.loads(manifest(tmp_path, 'distributed').read_text())
    assert before['state_budget_bytes'] == after['state_budget_bytes']
    assert resumed.report.distributed_report['fit_partitions_reused_from_checkpoint'] >= 1


@pytest.mark.parametrize('engine', ['local', 'distributed'])
def test_unsafe_partial_capacity_is_resource_error(tmp_path, monkeypatch, engine):
    kw = options(tmp_path, 'local')
    if engine == 'distributed':
        kw.update(execution='distributed', distributed_backend='local', distributed_partition_rows=5)
        kw.pop('batch_rows')
    ram(monkeypatch, 900)
    with pytest.raises(RuntimeError, match='intentional'):
        featm.extract(Interrupted(data()), **kw)
    ram(monkeypatch, 100)
    with pytest.raises(ResourceBudgetTooSmallError, match='Free memory or resume on a larger machine'):
        featm.extract(data(), **kw)


def test_auto_local_unpinned_batches_stable(tmp_path, monkeypatch):
    kw = options(tmp_path, 'local')
    kw.pop('batch_rows')
    ram(monkeypatch, 900)
    first = featm.extract(data(), **kw)
    ram(monkeypatch, 800)
    second = featm.extract(data(), **kw)
    assert second.report.batch_rows == first.report.batch_rows
    pd.testing.assert_frame_equal(first.to_pandas(), second.to_pandas())


def test_partial_auto_retains_unpinned_row_boundaries(tmp_path, monkeypatch):
    df = pd.DataFrame({'x': np.arange(30_000, dtype=float)})
    kw = options(tmp_path, 'local')
    kw.pop('batch_rows')
    ram(monkeypatch, 1)
    clean = featm.extract(df, **{k: v for k, v in kw.items() if k != 'checkpoint'})
    with pytest.raises(RuntimeError, match='intentional'):
        featm.extract(Interrupted(df), **kw)
    saved = json.loads(manifest(tmp_path, 'local').read_text())['settings']
    ram(monkeypatch, .9)
    resumed = featm.extract(df, **kw)
    pd.testing.assert_frame_equal(clean.to_pandas(), resumed.to_pandas())
    assert resumed.report.batch_rows == saved['batch_rows']


def test_partial_auto_retains_budget_limited_reservoir(tmp_path, monkeypatch):
    df = pd.DataFrame({'c': [f'category-{i % 100}' for i in range(2_000)]})
    kw = dict(features=[CF], columns={CF: 'c'}, batch_rows=100, mode='approximate',
              approximate_sample_size=10_000, errors='raise')
    ram(monkeypatch, 1)
    clean = featm.extract(df, **kw)
    with pytest.raises(RuntimeError, match='intentional'):
        featm.extract(Interrupted(df), checkpoint=tmp_path / 'cp', **kw)
    ram(monkeypatch, .9)
    resumed = featm.extract(df, checkpoint=tmp_path / 'cp', **kw)
    pd.testing.assert_frame_equal(clean.to_pandas(), resumed.to_pandas())
    assert clean.report.approximations_used[CF] == resumed.report.approximations_used[CF]


@pytest.mark.parametrize('engine', ['local', 'distributed'])
@pytest.mark.parametrize('damage', ['version', 'budget', 'snapshot'])
def test_auto_checkpoint_damage_fails_closed(tmp_path, monkeypatch, engine, damage):
    kw = options(tmp_path, 'local')
    if engine == 'distributed':
        kw.update(execution='distributed', distributed_backend='local', distributed_partition_rows=5)
        kw.pop('batch_rows')
    ram(monkeypatch, 900)
    if engine == 'distributed':
        from featm.distributed.checkpoint import DistributedCheckpointManager
        original = DistributedCheckpointManager.commit_fit_partition
        def interrupted(self, index, states, **kwargs):
            original(self, index, states, **kwargs)
            raise RuntimeError('intentional interrupted fit')
        with monkeypatch.context() as m:
            m.setattr(DistributedCheckpointManager, 'commit_fit_partition', interrupted)
            with pytest.raises(RuntimeError, match='intentional'):
                featm.extract(data(), **kw)
    else:
        with pytest.raises(RuntimeError, match='intentional'):
            featm.extract(Interrupted(data()), **kw)
    path = manifest(tmp_path, engine)
    meta = json.loads(path.read_text())
    if damage == 'version':
        meta['manifest_version'] = -1
        expected = CheckpointIncompatibleError
    elif damage == 'budget':
        (meta['settings'] if engine == 'local' else meta)['state_budget_bytes'] = -1
        expected = CheckpointCorruptionError
    else:
        if engine == 'local':
            (tmp_path / 'cp' / 'state' / meta['fit_accumulator_file']).write_bytes(b'broken')
            expected = CheckpointCorruptionError
        else:
            committed_index = meta['fit_committed_partitions'][0]
            (tmp_path / 'cp' / 'fit_state' / f'part-{committed_index:06d}.pkl').write_bytes(b'broken')
            expected = RuntimeError
    path.write_text(json.dumps(meta))
    ram(monkeypatch, 800)
    with pytest.raises(expected):
        featm.extract(data(), **kw)
