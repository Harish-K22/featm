"""Checkpoint budget identity closure pass (v0.4.2 -> v0.4.3), local Step-2 engine.

The distributed checkpoint identity closure fix (same version bump) added
``state_budget_bytes`` to ``DistributedCheckpointIdentity`` because
``memory_budget`` is not merely a performance/resource hint for an exact
stateful feature -- the derived state budget directly decides whether
fitting succeeds or raises ``StateBudgetExceededError``, which under
``errors="skip"``/``"warn"`` changes the actual selected feature set and
result shape.

The independent audit asked whether the CANONICAL local/out-of-core
``featm.checkpoint.CheckpointIdentity`` has the same latent bug, since its
fit path also computes and uses
``derive_state_budget_bytes(prepared.budget.memory_bytes)``. Reproduced here
with ``batch_rows=1`` pinned explicitly (so changing ``memory_budget`` cannot
also change batch boundaries and thereby mask the identity issue via an
unrelated mechanism) -- confirmed vulnerable, and in a WORSE way than the
distributed case predicted: a stale small-budget-then-large-budget-resume
did not merely reuse stale "failed" semantics, it produced an internally
INCOHERENT state (empty output, but an EMPTY ``failed_features`` list too --
matching neither the original small-budget run nor a clean large-budget
run). Root cause: ``_run_fit_phase``'s resume branch
(``elif ... manifest.get("fit_state_files")``) only ever recorded files for
stateful features that succeeded; a checkpoint whose fit failed for the only
selected feature had an EMPTY ``fit_state_files``, so that whole branch's
guard (a falsy empty dict) never even ran, leaving ``fitted_states`` empty
and ``failed_stateful`` empty on resume alike.

Fixed by adding ``state_budget_bytes`` to
``CheckpointIdentity.settings`` -- ``None`` when the run has no fit phase at
all (matches the distributed policy exactly, same reasoning: an unrelated
memory_budget change must not invalidate a checkpoint whose result never
depended on the state budget).
"""
from __future__ import annotations

import pandas as pd
import pytest

import featm
from featm.checkpoint import CheckpointIncompatibleError


@pytest.fixture
def _pyarrow():
    return pytest.importorskip("pyarrow")


def _budget_reproducer_df():
    long_value = "X" * 20_000
    return pd.DataFrame({"cat": ["a", long_value, "b", "c"]})


_LOCAL_BUDGET_KWARGS = dict(
    features=["tabular.encoding.category_frequency"], columns={"tabular.encoding.category_frequency": "cat"},
    execution="local", batch_rows=1, errors="skip",
)


def test_budget_changes_local_result_sanity_check():
    df = _budget_reproducer_df()
    small = featm.extract(df, memory_budget=65_536, **_LOCAL_BUDGET_KWARGS)
    large = featm.extract(df, memory_budget=1_048_576, **_LOCAL_BUDGET_KWARGS)
    assert small.shape == (0, 0)
    assert small.report.failed_features == ["tabular.encoding.category_frequency"]
    assert large.shape == (4, 1)
    assert large.report.failed_features == []


def test_local_resume_rejects_incompatible_state_budget_small_to_large(_pyarrow, tmp_path):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_LOCAL_BUDGET_KWARGS)
    assert first.shape == (0, 0)
    with pytest.raises(CheckpointIncompatibleError, match="settings"):
        featm.extract(df, memory_budget=1_048_576, checkpoint=str(ckpt), **_LOCAL_BUDGET_KWARGS)


def test_local_resume_rejects_incompatible_state_budget_large_to_small(_pyarrow, tmp_path):
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=1_048_576, checkpoint=str(ckpt), **_LOCAL_BUDGET_KWARGS)
    assert first.shape == (4, 1)
    with pytest.raises(CheckpointIncompatibleError, match="settings"):
        featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_LOCAL_BUDGET_KWARGS)


def test_local_resume_with_equivalent_derived_state_budget_remains_compatible(_pyarrow, tmp_path):
    from featm.resources import derive_state_budget_bytes, parse_size

    assert derive_state_budget_bytes(parse_size(65_536)) == derive_state_budget_bytes(parse_size(65_537))
    df = _budget_reproducer_df()
    ckpt = tmp_path / "ckpt"
    first = featm.extract(df, memory_budget=65_536, checkpoint=str(ckpt), **_LOCAL_BUDGET_KWARGS)
    second = featm.extract(df, memory_budget=65_537, checkpoint=str(ckpt), **_LOCAL_BUDGET_KWARGS)
    assert first.shape == second.shape == (0, 0)
    assert second.report.resumed is True


def test_local_stateless_only_checkpoint_not_invalidated_by_memory_budget_change(_pyarrow, tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    ckpt = tmp_path / "ckpt"
    kwargs = dict(
        features=["tabular.numeric.abs"], columns={"tabular.numeric.abs": "x"},
        execution="local", batch_rows=1, errors="raise", checkpoint=str(ckpt),
    )
    first = featm.extract(df, memory_budget=65_536, **kwargs)
    second = featm.extract(df, memory_budget=999_999_999, **kwargs)
    assert first.shape == second.shape == (4, 1)
    assert second.report.resumed is True
