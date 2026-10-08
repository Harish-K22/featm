"""Step 3: optional GPU execution backend.

Importing this package never imports CuPy/Torch itself -- every probe in
:mod:`featm.gpu.backend` is lazy. ``import featm`` (which imports this
package to register GPU metadata cross-checks and expose the public
``backend=`` machinery) is therefore safe and silent on a CPU-only machine
with no GPU extras installed.

Public surface re-exported here:

- :func:`featm.gpu.backend.any_gpu_available` -- whether a usable GPU backend exists right now.
- :func:`featm.gpu.backend.device_info` -- one device's name/total/free VRAM.
- :func:`featm.gpu.dispatch.dispatch_one_feature` -- the CPU/GPU dispatch entry point.
- :func:`featm.gpu.registration.gpu_capability_inventory` -- registry-wide GPU coverage counts.
- :func:`featm.gpu.registration.validate_gpu_registry` -- cross-check metadata against real kernels.
"""
from __future__ import annotations

from .backend import any_gpu_available, device_info, preferred_backend
from .dispatch import BackendInfo, dispatch_one_feature, gpu_capable
from .exceptions import (
    GPUBackendError,
    GPUDtypeUnsupportedError,
    GPUError,
    GPUImplementationMissingError,
    GPUNumericalValidationError,
    GPUOutOfMemoryError,
    GPUUnavailableError,
)
from .memory import GPUMemoryBudget
from .registration import gpu_capability_inventory, validate_gpu_registry

__all__ = [
    "any_gpu_available",
    "device_info",
    "preferred_backend",
    "BackendInfo",
    "dispatch_one_feature",
    "gpu_capable",
    "GPUError",
    "GPUUnavailableError",
    "GPUImplementationMissingError",
    "GPUDtypeUnsupportedError",
    "GPUOutOfMemoryError",
    "GPUNumericalValidationError",
    "GPUBackendError",
    "GPUMemoryBudget",
    "gpu_capability_inventory",
    "validate_gpu_registry",
]
