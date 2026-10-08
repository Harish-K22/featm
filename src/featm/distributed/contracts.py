from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DistributedAction(str, Enum):
    """How one selected feature may participate in distributed execution."""

    MAP = "map"
    FIT_THEN_MAP = "fit_then_map"
    REJECT = "reject"


@dataclass(frozen=True)
class DistributedFeatureDecision:
    name: str
    action: DistributedAction
    reason: str = ""


@dataclass
class DistributedPlan:
    runtime: str
    partition_rows: int
    max_in_flight: int
    decisions: list[DistributedFeatureDecision]
    map_features: list[str] = field(default_factory=list)
    stateful_features: list[str] = field(default_factory=list)
    rejected_features: dict[str, str] = field(default_factory=dict)

    @property
    def has_fit_phase(self) -> bool:
        return bool(self.stateful_features)


@dataclass(frozen=True)
class Partition:
    index: int
    frame: Any


@dataclass
class PartitionResult:
    index: int
    n_input_rows: int
    feature_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_errors: dict[str, str] = field(default_factory=dict)
    backend_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    batch_reductions: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FitPartitionResult:
    index: int
    states: dict[str, Any] = field(default_factory=dict)
    feature_errors: dict[str, str] = field(default_factory=dict)


class DistributedExecutionError(RuntimeError):
    pass


class DistributedFeatureUnsupportedError(DistributedExecutionError):
    pass


class DistributedRuntimeUnavailableError(ImportError):
    pass


class DistributedTaskError(DistributedExecutionError):
    def __init__(self, partition_index: int, exc: BaseException):
        super().__init__(f"distributed task for partition {partition_index} failed: {type(exc).__name__}: {exc}")
        self.partition_index = partition_index
        self.original_exception = exc


class DistributedFeatureError(DistributedExecutionError):
    """A single feature raised inside a worker task, under ``errors="raise"``.

    Mirrors ``featm.execution``'s local out-of-core engine: a feature error
    on the FIT phase (any partition) or on the priming (logical partition 0)
    TRANSFORM partition always raises immediately under ``errors="raise"`` --
    the same semantics a small-data/local out-of-core caller already gets.
    The original exception cannot be reliably reconstructed across a process
    boundary (Dask worker processes may not share every exception class), so
    this carries its type name and message as a formatted string instead of
    attempting to re-raise the original object.
    """

    def __init__(self, feature: str, partition_index: int, message: str, *, phase: str):
        super().__init__(f"{feature}: distributed {phase} failed on partition {partition_index}: {message}")
        self.feature = feature
        self.partition_index = partition_index
        self.phase = phase
