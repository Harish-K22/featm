"""GPU registry cross-validation and capability inventory (Step 3).

``featm.registry.validate_spec`` checks everything derivable from one
``FeatureSpec``'s own fields alone (see its "Step 3: GPU metadata coherence"
section). It cannot also check "does a real kernel exist for this name" --
that answer lives in ``featm.gpu.kernels``, a module the base registry does
not (and must not) import, to avoid a registry-package <-> gpu-package
import cycle. :func:`validate_gpu_registry` is the deferred, whole-registry
version of that check, meant to be called once after both the feature
registry and the GPU kernel modules have finished importing (exactly the
same relationship ``featm.state.STATEFUL_ENCODERS`` has to
``registry.validate_spec`` -- see that module's own comments).
"""
from __future__ import annotations

from ..registry import REGISTRY, RegistryValidationError
from .kernels.reductions import REDUCTION_KERNELS
from .kernels.spectral import SPECTRAL_KERNELS, STANDALONE_SPECTRAL_KERNELS


def _all_kernel_names() -> set[str]:
    return set(REDUCTION_KERNELS) | set(SPECTRAL_KERNELS) | set(STANDALONE_SPECTRAL_KERNELS)


def validate_gpu_registry() -> None:
    """Cross-check every ``gpu_available=True`` spec against the real kernel dicts, both directions.

    Raises :class:`featm.registry.RegistryValidationError` (the same
    exception class ``validate_spec`` uses, for a consistent error type
    across every registry-integrity check) if:

    - a spec claims ``gpu_available=True`` but no kernel dict has an entry
      for its name (metadata claiming a capability the executor cannot
      actually deliver -- exactly the "gpu_available=True but no
      implementation/backend route" contradiction the Step 3 spec names); or
    - a kernel dict has an entry for a name that either does not exist in
      the registry at all, or exists but does not declare
      ``gpu_available=True`` (a kernel nothing will ever route to, or a
      silent mismatch between the two sources of truth).
    """
    errors: list[str] = []
    kernel_names = _all_kernel_names()
    registry_names = {s.name for s in REGISTRY.all()}
    gpu_flagged = {s.name for s in REGISTRY.all() if s.gpu_available}

    missing_kernel = gpu_flagged - kernel_names
    if missing_kernel:
        errors.append(f"gpu_available=True but no registered GPU kernel: {sorted(missing_kernel)}")

    orphan_kernels = kernel_names - registry_names
    if orphan_kernels:
        errors.append(f"GPU kernel registered for unknown feature name(s): {sorted(orphan_kernels)}")

    unflagged_kernels = (kernel_names & registry_names) - gpu_flagged
    if unflagged_kernels:
        errors.append(f"GPU kernel exists but gpu_available=False on the registered spec: {sorted(unflagged_kernels)}")

    if errors:
        raise RegistryValidationError("GPU registry cross-check failed: " + "; ".join(errors))


def gpu_capability_inventory() -> dict:
    """Counts for the Step 3 spec's required "GPU capability inventory" (§C / notebook Section C).

    Returns a plain dict (JSON-serializable) with total/gpu-capable/cpu-only
    counts and per-modality/per-family/per-backend breakdowns -- used by
    ``tests/test_gpu_metadata.py`` and by the Kaggle notebook's Section C.
    """
    all_specs = REGISTRY.all()
    gpu_specs = [s for s in all_specs if s.gpu_available]
    by_modality: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_backend: dict[str, int] = {}
    for s in gpu_specs:
        by_modality[s.modality] = by_modality.get(s.modality, 0) + 1
        key = f"{s.modality}.{s.family}"
        by_family[key] = by_family.get(key, 0) + 1
        be = s.gpu_backend or "unknown"
        by_backend[be] = by_backend.get(be, 0) + 1
    return {
        "total_features": len(all_specs),
        "gpu_capable_features": len(gpu_specs),
        "cpu_only_features": len(all_specs) - len(gpu_specs),
        "gpu_capable_by_modality": dict(sorted(by_modality.items())),
        "gpu_capable_by_family": dict(sorted(by_family.items())),
        "gpu_capable_by_backend": dict(sorted(by_backend.items())),
        "gpu_capable_feature_names": sorted(s.name for s in gpu_specs),
    }
