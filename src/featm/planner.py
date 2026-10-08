from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .registry import FeatureRegistry, FeatureSpec, Cost
from .resources import ResourceBudget, choose_batch_rows, estimate_dense_output_bytes
from .schema import DatasetProfile, SemanticType


@dataclass(frozen=True)
class FeatureTask:
    spec_name: str
    inputs: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    output_name: str = ""


@dataclass
class ExecutionPlan:
    tasks: list[FeatureTask]
    skipped: list[tuple[str, str]]
    batch_rows: int
    estimated_output_bytes: int


_PRESET_COST = {
    "minimal": Cost.TINY,
    "fast": Cost.LOW,
    "standard": Cost.MEDIUM,
    "comprehensive": Cost.HIGH,
}


class FeaturePlanner:
    def __init__(self, registry: FeatureRegistry) -> None:
        self.registry = registry

    def plan_tabular(
        self,
        profile: DatasetProfile,
        *,
        preset: str,
        budget: ResourceBudget,
        max_output_features: int,
        include_families: set[str] | None = None,
    ) -> ExecutionPlan:
        max_cost = _PRESET_COST[preset]
        tasks: list[FeatureTask] = []
        skipped: list[tuple[str, str]] = []

        numeric = profile.names_for(SemanticType.CONTINUOUS, SemanticType.COUNT)
        categorical = profile.names_for(SemanticType.CATEGORICAL, SemanticType.HIGH_CARDINALITY, SemanticType.BINARY)
        datetimes = profile.names_for(SemanticType.DATETIME)

        specs = self.registry.search(modality="tabular")
        for spec in specs:
            if include_families and spec.family not in include_families:
                continue
            if spec.cost_weight > FeatureSpec(name="_", family="_", modality="_", cost=max_cost).cost_weight:
                skipped.append((spec.name, f"preset excludes cost={spec.cost.value}"))
                continue
            if profile.n_rows < spec.min_rows:
                skipped.append((spec.name, "insufficient rows"))
                continue

            if spec.arity == 1:
                if "numeric" in spec.input_types:
                    cols = numeric
                elif "categorical" in spec.input_types:
                    cols = categorical
                elif "datetime" in spec.input_types:
                    cols = datetimes
                else:
                    cols = []
                for col in cols:
                    tasks.append(FeatureTask(spec.name, (col,), {}, f"{col}__{spec.name.split('.')[-1]}"))
            elif spec.arity == 2 and "numeric" in spec.input_types:
                # Interactions explode combinatorially; cap them before execution.
                for a, b in combinations(numeric, 2):
                    tasks.append(FeatureTask(spec.name, (a, b), {}, f"{a}__{b}__{spec.name.split('.')[-1]}"))
                    if len(tasks) >= max_output_features * 2:
                        break

        # Prefer cheap tasks first, then deterministically trim to requested output cap.
        tasks.sort(key=lambda t: (self.registry.get(t.spec_name).cost_weight, t.spec_name, t.inputs))
        if len(tasks) > max_output_features:
            overflow = len(tasks) - max_output_features
            skipped.append(("planner", f"trimmed {overflow} tasks to max_output_features={max_output_features}"))
            tasks = tasks[:max_output_features]

        output_bytes = estimate_dense_output_bytes(profile.n_rows, len(tasks), itemsize=8)
        batch_rows = choose_batch_rows(
            n_columns_in=profile.n_columns,
            n_columns_out=max(len(tasks), 1),
            budget=budget,
        )
        return ExecutionPlan(tasks, skipped, batch_rows, output_bytes)
