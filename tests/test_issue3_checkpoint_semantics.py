"""Issue #3: pre-fix committed entropy shards must never survive a resume.

Before the two-pass fix, tabular.missingness.entropy was row-independent and
old checkpointed output could contain batch-local zero entropies.  A checkpoint
with ANOTHER fitted feature is especially dangerous because it already has a
fit phase and matching state-budget identity.  These tests deliberately keep
all other identity fields unchanged, including the package version.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from featm.checkpoint import (
    CHECKPOINT_MANIFEST_VERSION,
    CheckpointIdentity,
    CheckpointIncompatibleError,
    CheckpointManager,
    checkpoint_feature_semantic_revisions,
)
from featm.distributed.checkpoint import (
    DistributedCheckpointIdentity,
    DistributedCheckpointManager,
)

ENTROPY = "tabular.missingness.entropy"
OTHER = "tabular.scaling.zscore"


def _identity(features, *, distributed):
    shared = dict(
        featm_version="0.5.11",  # deliberate same-version pre-release checkpoint
        source_fingerprint="unchanged-source",
        schema_fingerprint="unchanged-schema",
        features=list(features),
        params={},
    )
    if distributed:
        return DistributedCheckpointIdentity(
            **shared,
            partition_rows=2,
            mode="exact",
            errors="raise",
            columns={name: ["str:x"] for name in features},
            state_budget_bytes=64_000,
            state_budget_policy="explicit",
        )
    return CheckpointIdentity(
        **shared,
        settings={
            "batch_rows": 2,
            "mode": "exact",
            "errors": "raise",
            "columns": {name: ["str:x"] for name in features},
            "state_budget_bytes": 64_000,
            "state_budget_policy": "explicit",
        },
    )


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("features", [[ENTROPY], [ENTROPY, OTHER], [OTHER, ENTROPY]])
def test_old_entropy_checkpoint_fails_closed_before_reusing_commits(tmp_path, distributed, features):
    identity = _identity(features, distributed=distributed)
    mgr = DistributedCheckpointManager(tmp_path) if distributed else CheckpointManager(tmp_path)
    if distributed:
        mgr.initialize_or_resume(identity)
    else:
        mgr.initialize_or_resume(identity, has_fit_phase=True)
    manifest = mgr.load()
    assert manifest["feature_semantic_revisions"] == {ENTROPY: "whole-column-stateful-v1"}
    manifest.pop("feature_semantic_revisions")
    # Model a committed, pre-fix result.  A real resume must reject BEFORE
    # reading fit_state, transform output or the zero-read fast path.
    if distributed:
        manifest["fit_done"] = True
        manifest["transform_done"] = True
        manifest["fit_committed_partitions"] = [0, 1]
        manifest["transform_committed_partitions"] = [0, 1]
    else:
        manifest["manifest_version"] = CHECKPOINT_MANIFEST_VERSION
        manifest["phase"] = "transform"
        manifest["fit_batches_completed"] = 2
        manifest["transform_batches_completed"] = 2
        manifest["done"] = True
    from featm.checkpoint import _atomic_write_json
    _atomic_write_json(mgr.manifest_path, manifest)
    before = deepcopy(mgr.load())
    with pytest.raises(CheckpointIncompatibleError, match="feature_semantic_revisions"):
        if distributed:
            mgr.initialize_or_resume(identity)
        else:
            mgr.initialize_or_resume(identity, has_fit_phase=True)
    assert mgr.load() == before, "A rejected checkpoint must not be altered or migrated silently"


@pytest.mark.parametrize("distributed", [False, True])
def test_mismatched_entropy_revision_is_incompatible(tmp_path, distributed):
    identity = _identity([OTHER, ENTROPY], distributed=distributed)
    mgr = DistributedCheckpointManager(tmp_path) if distributed else CheckpointManager(tmp_path)
    if distributed:
        mgr.initialize_or_resume(identity)
    else:
        mgr.initialize_or_resume(identity, has_fit_phase=True)
    from featm.checkpoint import _atomic_write_json
    manifest = mgr.load()
    manifest["feature_semantic_revisions"] = {ENTROPY: "old-batch-local-v0"}
    _atomic_write_json(mgr.manifest_path, manifest)
    with pytest.raises(CheckpointIncompatibleError, match="feature_semantic_revisions"):
        if distributed:
            mgr.initialize_or_resume(identity)
        else:
            mgr.initialize_or_resume(identity, has_fit_phase=True)


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("features", [[ENTROPY], [ENTROPY, OTHER]])
def test_current_entropy_checkpoint_can_resume(tmp_path, distributed, features):
    identity = _identity(features, distributed=distributed)
    mgr = DistributedCheckpointManager(tmp_path) if distributed else CheckpointManager(tmp_path)
    first = mgr.initialize_or_resume(identity) if distributed else mgr.initialize_or_resume(identity, has_fit_phase=True)
    second = mgr.initialize_or_resume(identity) if distributed else mgr.initialize_or_resume(identity, has_fit_phase=True)
    assert first == second
    assert second["feature_semantic_revisions"] == {ENTROPY: "whole-column-stateful-v1"}


@pytest.mark.parametrize("distributed", [False, True])
def test_unrelated_legacy_checkpoints_remain_compatible(tmp_path, distributed):
    # There is no global manifest-version bump, so existing unaffected
    # checkpoints retain their previous identity and resume normally.
    features = [OTHER]
    identity = _identity(features, distributed=distributed)
    assert "feature_semantic_revisions" not in identity.to_dict()
    assert checkpoint_feature_semantic_revisions(features) == {}
    mgr = DistributedCheckpointManager(tmp_path) if distributed else CheckpointManager(tmp_path)
    first = mgr.initialize_or_resume(identity) if distributed else mgr.initialize_or_resume(identity, has_fit_phase=True)
    assert "feature_semantic_revisions" not in first
    second = mgr.initialize_or_resume(identity) if distributed else mgr.initialize_or_resume(identity, has_fit_phase=True)
    assert second == first


def test_revision_is_targeted_and_order_independent():
    assert checkpoint_feature_semantic_revisions([OTHER, ENTROPY, ENTROPY]) == {
        ENTROPY: "whole-column-stateful-v1"
    }
    assert checkpoint_feature_semantic_revisions([OTHER]) == {}

@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("features", [[ENTROPY, OTHER], [ENTROPY]])
def test_real_completed_checkpoint_rejects_legacy_identity(tmp_path, distributed, features):
    """Integration gate: even a fully committed shard may not be reused.

    Requires PyArrow for the on-disk shards; identity-only tests above run
    without it and cover the same rejection, including pre-fix zero shards.
    """
    pytest.importorskip("pyarrow")
    import numpy as np
    import pandas as pd
    import featm
    from featm.checkpoint import _atomic_write_json

    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, np.nan]})
    source = featm.PandasSource(frame)
    args = dict(features=features, columns={name: "x" for name in features},
                checkpoint=tmp_path / "checkpoint", output="memory", errors="raise")
    if distributed:
        args.update(execution="distributed", distributed_backend="local", distributed_workers=2,
                    distributed_partition_rows=2)
    else:
        args["batch_rows"] = 2

    fresh = featm.extract(source, **args).to_numpy()
    # The current implementation must be safe to resume from its own shards.
    resumed = featm.extract(featm.PandasSource(frame), **args).to_numpy()
    np.testing.assert_allclose(resumed, fresh, equal_nan=True)
    cp = tmp_path / "checkpoint"
    manifest_path = cp / ("distributed_checkpoint.json" if distributed else "checkpoint.json")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.pop("feature_semantic_revisions") == {ENTROPY: "whole-column-stateful-v1"}
    # Identity now looks like one written by the pre-fix implementation.
    _atomic_write_json(manifest_path, manifest)
    before = manifest_path.read_bytes()
    with pytest.raises(CheckpointIncompatibleError, match="feature_semantic_revisions"):
        featm.extract(featm.PandasSource(frame), **args)
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("distributed", [False, True])
def test_real_unaffected_checkpoint_still_resumes(tmp_path, distributed):
    pytest.importorskip("pyarrow")
    import numpy as np
    import pandas as pd
    import featm

    source = featm.PandasSource(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
    args = dict(features=[OTHER], columns={OTHER: "x"}, checkpoint=tmp_path / "checkpoint",
                output="memory", errors="raise")
    if distributed:
        args.update(execution="distributed", distributed_backend="local", distributed_workers=2,
                    distributed_partition_rows=2)
    else:
        args["batch_rows"] = 2
    first = featm.extract(source, **args).to_numpy()
    second = featm.extract(featm.PandasSource(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})), **args).to_numpy()
    np.testing.assert_allclose(first, second, equal_nan=True)
