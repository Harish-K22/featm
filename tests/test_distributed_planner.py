"""Step 4 completion pass: build_distributed_plan() unit tests.

tests/test_distributed_foundation.py already exercises the planner's
decisions END TO END through featm.extract() for every rejection class
(FULL_DATASET, STRUCTURED_SAMPLE, required-param, approximate, legacy
incremental). This file unit-tests build_distributed_plan() directly against
a real _prepare_run()-produced PreparedRun, checking the DECISION OBJECTS
themselves (action/reason), not just the end-to-end extract() outcome.
"""
from __future__ import annotations

import pandas as pd
import pytest

from featm.distributed.contracts import DistributedAction
from featm.distributed.planner import build_distributed_plan
from featm.execution import _prepare_run
from featm.registry import REGISTRY


def _plan_for(df, features, columns, **kwargs):
    prepared = _prepare_run(
        df, features=features, modality=None, columns=columns, mode=kwargs.pop("mode", "exact"), errors="raise",
        memory_budget=kwargs.pop("memory_budget", None), disk_budget=None, batch_rows=kwargs.pop("batch_rows", 10),
        cache_budget=None, params=kwargs.pop("params", None), backend="cpu", gpu_fallback="fallback", device=None,
        gpu_memory_budget=None,
    )
    return build_distributed_plan(prepared, runtime="local", partition_rows=10, max_in_flight=4)


def test_row_independent_feature_maps():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    plan = _plan_for(df, ["tabular.numeric.abs"], {"tabular.numeric.abs": "x"})
    assert plan.map_features == ["tabular.numeric.abs"]
    assert plan.decisions[0].action == DistributedAction.MAP
    assert plan.decisions[0].reason == ""


def test_stateful_exact_feature_fit_then_maps():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    plan = _plan_for(df, ["tabular.scaling.zscore"], {"tabular.scaling.zscore": "x"})
    assert plan.stateful_features == ["tabular.scaling.zscore"]
    assert plan.decisions[0].action == DistributedAction.FIT_THEN_MAP
    assert plan.has_fit_phase is True


def test_mixed_map_and_stateful_selection_classified_independently():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    features = ["tabular.numeric.abs", "tabular.scaling.zscore"]
    plan = _plan_for(df, features, {"tabular.numeric.abs": "x", "tabular.scaling.zscore": "x"})
    assert plan.map_features == ["tabular.numeric.abs"]
    assert plan.stateful_features == ["tabular.scaling.zscore"]
    assert plan.rejected_features == {}


def _fake_prepared(specs):
    """A minimal stand-in for PreparedRun -- build_distributed_plan only reads
    .active_specs and .effective_mode_by_spec, so a full _prepare_run() call is
    unnecessary (and, for STRUCTURED_SAMPLE/FULL_DATASET features, Step 2's OWN
    out-of-core safety gate already rejects them EARLIER than the planner would
    ever see them for a real Source -- this directly unit-tests the planner's
    OWN redundant defense-in-depth check, decoupled from that earlier gate)."""
    import types

    return types.SimpleNamespace(active_specs=specs, effective_mode_by_spec={s.name: "exact" for s in specs})


def test_structured_sample_feature_rejected_with_reason():
    spec = REGISTRY.get("information.dependence.mutual_information")
    plan = build_distributed_plan(_fake_prepared([spec]), runtime="local", partition_rows=10, max_in_flight=4)
    assert plan.decisions[0].action == DistributedAction.REJECT
    assert "information.dependence.mutual_information" in plan.rejected_features
    assert plan.map_features == [] and plan.stateful_features == []


def test_full_dataset_feature_rejected_with_reason():
    full_dataset_names = [s.name for s in REGISTRY.all() if s.batch_semantics.value == "full_dataset"]
    spec = REGISTRY.get(full_dataset_names[0])
    plan = build_distributed_plan(_fake_prepared([spec]), runtime="local", partition_rows=10, max_in_flight=4)
    assert plan.decisions[0].action == DistributedAction.REJECT
    assert spec.name in plan.rejected_features


def test_all_features_rejected_gives_empty_map_and_stateful_lists():
    spec = REGISTRY.get("information.dependence.mutual_information")
    plan = build_distributed_plan(_fake_prepared([spec]), runtime="local", partition_rows=10, max_in_flight=4)
    assert plan.map_features == []
    assert plan.stateful_features == []
    assert len(plan.rejected_features) == 1


def test_plan_runtime_and_partition_rows_recorded_verbatim():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    prepared = _prepare_run(
        df, features=["tabular.numeric.abs"], modality=None, columns={"tabular.numeric.abs": "x"}, mode="exact",
        errors="raise", memory_budget=None, disk_budget=None, batch_rows=10, cache_budget=None, params=None,
        backend="cpu", gpu_fallback="fallback", device=None, gpu_memory_budget=None,
    )
    plan = build_distributed_plan(prepared, runtime="dask", partition_rows=37, max_in_flight=9)
    assert plan.runtime == "dask"
    assert plan.partition_rows == 37
    assert plan.max_in_flight == 9
