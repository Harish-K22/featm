from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .planner import ExecutionPlan, FeatureTask
from .registry import FeatureRegistry


@dataclass
class TaskResult:
    task: FeatureTask
    status: str
    error: str | None = None


class FaultTolerantExecutor:
    """Execute independent feature tasks without allowing one failure to kill a run."""

    def __init__(self, registry: FeatureRegistry, errors: str = "skip") -> None:
        if errors not in {"skip", "warn", "raise"}:
            raise ValueError("errors must be one of: skip, warn, raise")
        self.registry = registry
        self.errors = errors

    def execute_tabular(
        self,
        df: pd.DataFrame,
        plan: ExecutionPlan,
        *,
        state: dict[str, Any] | None = None,
        fit: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any], list[TaskResult]]:
        state = {} if state is None else state
        outputs: dict[str, pd.Series] = {}
        results: list[TaskResult] = []

        for task in plan.tasks:
            spec = self.registry.get(task.spec_name)
            fn = spec.implementation
            assert fn is not None
            task_key = task.output_name
            try:
                learned_state = state.get(task_key)

                # Stateless streamable features are evaluated in planner-sized row
                # batches. During transform, fitted incremental features can also be
                # replayed in batches because their training state is already frozen.
                can_batch = (
                    spec.streamable
                    and len(df) > plan.batch_rows
                    and (not spec.incremental or not fit)
                )
                if can_batch:
                    pieces: list[pd.Series] = []
                    for start in range(0, len(df), plan.batch_rows):
                        chunk = df.iloc[start : start + plan.batch_rows]
                        value, learned_state = self._run_streamable_chunk(
                            fn=fn,
                            spec=spec,
                            task=task,
                            chunk=chunk,
                            fit=fit,
                            learned_state=learned_state,
                        )
                        pieces.append(value)
                    value = pd.concat(pieces).reindex(df.index)
                    learned = learned_state
                else:
                    series_args = [df[c] for c in task.inputs]
                    value, learned = fn(*series_args, fit=fit, state=learned_state, **task.params)

                if learned is not None:
                    state[task_key] = learned
                if not isinstance(value, pd.Series):
                    value = pd.Series(value, index=df.index)
                # Normalize troublesome infinities at the boundary. Individual features
                # may choose to encode them differently later.
                if pd.api.types.is_numeric_dtype(value):
                    value = value.replace([np.inf, -np.inf], np.nan)
                outputs[task.output_name] = value
                results.append(TaskResult(task, "ok"))
            except MemoryError as exc:
                self._handle(task, exc)
                results.append(TaskResult(task, "skipped", f"MemoryError: {exc}"))
            except Exception as exc:  # feature-level isolation is deliberate
                self._handle(task, exc)
                results.append(TaskResult(task, "skipped", f"{type(exc).__name__}: {exc}"))

        return pd.DataFrame(outputs, index=df.index), state, results


    def _run_streamable_chunk(
        self,
        *,
        fn,
        spec,
        task: FeatureTask,
        chunk: pd.DataFrame,
        fit: bool,
        learned_state: Any,
        min_retry_rows: int = 256,
    ) -> tuple[pd.Series, Any]:
        """Run a streamable feature and halve the batch on MemoryError.

        This makes memory pressure a recoverable condition for stateless transforms
        and frozen-state replay. If even a tiny batch cannot run, control returns to
        the outer feature-level fault isolation logic.
        """
        try:
            series_args = [chunk[c] for c in task.inputs]
            value, learned = fn(
                *series_args,
                fit=False if spec.incremental else fit,
                state=learned_state,
                **task.params,
            )
            if learned is not None and learned_state is None:
                learned_state = learned
            if not isinstance(value, pd.Series):
                value = pd.Series(value, index=chunk.index)
            return value, learned_state
        except MemoryError:
            if len(chunk) <= min_retry_rows:
                raise
            mid = len(chunk) // 2
            left, learned_state = self._run_streamable_chunk(
                fn=fn, spec=spec, task=task, chunk=chunk.iloc[:mid],
                fit=fit, learned_state=learned_state, min_retry_rows=min_retry_rows,
            )
            right, learned_state = self._run_streamable_chunk(
                fn=fn, spec=spec, task=task, chunk=chunk.iloc[mid:],
                fit=fit, learned_state=learned_state, min_retry_rows=min_retry_rows,
            )
            return pd.concat([left, right]), learned_state

    def _handle(self, task: FeatureTask, exc: Exception) -> None:
        if self.errors == "raise":
            raise exc
        if self.errors == "warn":
            warnings.warn(f"Skipping {task.spec_name} on {task.inputs}: {exc}", RuntimeWarning)
