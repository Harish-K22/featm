from __future__ import annotations

from typing import Any, Mapping

from ..execution import PreparedRun
from ..registry import BatchSemantics
from ..state import STATEFUL_ENCODERS
from .contracts import DistributedAction, DistributedFeatureDecision, DistributedPlan


def build_distributed_plan(
    prepared: PreparedRun,
    *,
    runtime: str,
    partition_rows: int,
    max_in_flight: int,
    params: Mapping[str, Mapping[str, Any]] | None = None,
) -> DistributedPlan:
    """Classify every already-Step2-safe spec for distributed execution.

    Important: Step 2's ``_prepare_run`` remains the first safety gate. This
    planner adds the stronger question "can *different workers* compute this
    feature independently or merge its fit state?". It never weakens a Step 2
    rejection.
    """
    decisions: list[DistributedFeatureDecision] = []
    map_features: list[str] = []
    stateful_features: list[str] = []
    rejected: dict[str, str] = {}

    for spec in prepared.active_specs:
        reason = ""
        action = DistributedAction.REJECT
        if spec.batch_semantics in (BatchSemantics.FULL_DATASET, BatchSemantics.GLOBAL_OUTPUT):
            reason = (
                f"{spec.name} requires whole-dataset/global-output semantics and cannot be partitioned across workers "
                "without changing the feature definition. Run it locally in a separate call or implement a dedicated "
                "distributed algorithm for this feature family."
            )
        elif spec.batch_semantics is BatchSemantics.STRUCTURED_SAMPLE:
            reason = f"{spec.name} is STRUCTURED_SAMPLE and is not representable by the row-partition runtime."
        elif spec.modality == "tabular" and spec.incremental and not spec.stateful:
            reason = (
                f"{spec.name} uses the legacy single-shot incremental convention; fitting independently on workers "
                "would produce worker-local statistics."
            )
        elif spec.stateful:
            encoder = STATEFUL_ENCODERS.get(spec.name)
            if encoder is None or not encoder.distributed_merge_supported:
                reason = f"{spec.name} is stateful but its StatefulEncoder has no distributed merge_states() implementation."
            elif prepared.effective_mode_by_spec.get(spec.name) != "exact":
                # Distributed uniform reservoir sampling requires a mergeable priority/reservoir algorithm.
                # Do not fake it by sampling independently per worker and concatenating samples.
                reason = (
                    f"{spec.name} resolved to mode={prepared.effective_mode_by_spec.get(spec.name)!r}. Distributed "
                    "execution supports state merging in exact mode only; independently merging per-worker reservoirs "
                    "would not preserve the existing uniform-sampling semantics."
                )
            else:
                action = DistributedAction.FIT_THEN_MAP
                stateful_features.append(spec.name)
        elif spec.batch_semantics in (BatchSemantics.ROW_INDEPENDENT, BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED, BatchSemantics.OPAQUE_OBJECT_SAMPLE):
            action = DistributedAction.MAP
            map_features.append(spec.name)
        else:
            reason = f"{spec.name} has unsupported distributed batch semantics {spec.batch_semantics.value!r}."

        if action is DistributedAction.REJECT:
            rejected[spec.name] = reason
        decisions.append(DistributedFeatureDecision(spec.name, action, reason))

    return DistributedPlan(
        runtime=runtime,
        partition_rows=int(partition_rows),
        max_in_flight=int(max_in_flight),
        decisions=decisions,
        map_features=map_features,
        stateful_features=stateful_features,
        rejected_features=rejected,
    )
