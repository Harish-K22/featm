"""Existing checkpoint roots must fail closed before per-feature error policy."""
import json

import numpy as np
import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError, CheckpointManager
from featm.distributed.checkpoint import DistributedCheckpointManager


@pytest.mark.parametrize('manager_type', [CheckpointManager, DistributedCheckpointManager])
@pytest.mark.parametrize('payload', [b'null', b'[]', b'"checkpoint"', b'0', b'1.5', b'true', b'false'])
def test_manager_non_object_root_preserved(tmp_path, manager_type, payload):
    manager = manager_type(tmp_path / 'cp')
    manager.manifest_path.write_bytes(payload)
    with pytest.raises(CheckpointIncompatibleError, match='JSON root must be an object'):
        manager.load()
    assert manager.manifest_path.read_bytes() == payload


@pytest.mark.parametrize('manager_type', [CheckpointManager, DistributedCheckpointManager])
def test_manager_missing_and_object_roots(tmp_path, manager_type):
    manager = manager_type(tmp_path / 'cp')
    assert manager.load() is None
    manager.manifest_path.write_bytes(b'{}')
    assert manager.load() == {}


@pytest.fixture(params=['local', 'distributed'])
def public_checkpoint(request, tmp_path):
    pytest.importorskip('pyarrow')
    cp = tmp_path / 'cp'
    options = dict(features=['tabular.scaling.zscore'], columns={'tabular.scaling.zscore': 'x'},
                   checkpoint=cp, memory_budget='128MB', errors='raise')
    if request.param == 'local':
        options['batch_rows'] = 5
        manifest = cp / 'checkpoint.json'
    else:
        options.update(execution='distributed', distributed_backend='local', distributed_partition_rows=5)
        manifest = cp / 'distributed_checkpoint.json'
    data = pd.DataFrame({'x': np.arange(20, dtype=float)})
    return data, options, manifest


@pytest.mark.parametrize('policy', ['raise', 'warn', 'skip'])
@pytest.mark.parametrize('payload', [b'null', b'[]', b'"checkpoint"', b'0', b'1.5', b'true', b'false', b'{}', b'{broken'])
def test_public_corrupt_root_fails_closed(public_checkpoint, policy, payload):
    data, options, manifest = public_checkpoint
    options['errors'] = policy
    featm.extract(data, **options)
    manifest.write_bytes(payload)
    expected = json.JSONDecodeError if payload == b'{broken' else CheckpointIncompatibleError
    with pytest.raises(expected):
        featm.extract(data, **options)
    assert manifest.read_bytes() == payload


def test_public_missing_and_compatible_resume(public_checkpoint):
    data, options, manifest = public_checkpoint
    assert not manifest.exists()
    first = featm.extract(data, **options).to_pandas()
    committed = manifest.read_bytes()
    second = featm.extract(data, **options).to_pandas()
    pd.testing.assert_frame_equal(first, second)
    assert manifest.read_bytes() == committed


@pytest.mark.parametrize('field', ['manifest_version', 'featm_version'])
def test_public_incompatible_object_preserved(public_checkpoint, field):
    data, options, manifest = public_checkpoint
    featm.extract(data, **options)
    body = json.loads(manifest.read_text())
    body[field] = -1 if field == 'manifest_version' else '0.0.0'
    payload = json.dumps(body).encode()
    manifest.write_bytes(payload)
    with pytest.raises(CheckpointIncompatibleError):
        featm.extract(data, **options)
    assert manifest.read_bytes() == payload
