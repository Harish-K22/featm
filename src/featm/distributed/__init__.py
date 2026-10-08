"""Step 4 distributed-execution foundation.

The public implementation is intentionally runtime-agnostic: FEATM owns
feature partition semantics and state merging; runtimes such as Dask only
schedule pure partition tasks. Dask imports remain optional/lazy.
"""
from .contracts import (
    DistributedAction,
    DistributedExecutionError,
    DistributedFeatureUnsupportedError,
    DistributedPlan,
    DistributedRuntimeUnavailableError,
    DistributedTaskError,
)
from .execution import distributed_extract

__all__ = [
    "DistributedAction",
    "DistributedExecutionError",
    "DistributedFeatureUnsupportedError",
    "DistributedPlan",
    "DistributedRuntimeUnavailableError",
    "DistributedTaskError",
    "distributed_extract",
]
