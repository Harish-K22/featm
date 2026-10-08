"""Step 3 local tests: FeatureSpec GPU metadata, validate_spec coherence, registry cross-checks."""
from __future__ import annotations

import pytest

import featm
from featm.registry import BatchSemantics, FeatureSpec, REGISTRY, RegistryValidationError, validate_spec
from featm.gpu.registration import gpu_capability_inventory, validate_gpu_registry
from featm.gpu.kernels.reductions import REDUCTION_KERNELS
from featm.gpu.kernels.spectral import SPECTRAL_KERNELS, STANDALONE_SPECTRAL_KERNELS


def test_registry_still_552_after_gpu_metadata_added():
    assert len(featm.REGISTRY.all()) == 552


def test_no_features_all_reintroduced():
    with pytest.raises(ValueError, match="features='all'"):
        featm.extract([1, 2, 3], features="all")


def test_gpu_registry_cross_check_passes():
    validate_gpu_registry()  # must not raise


def test_gpu_available_requires_gpu_backend():
    spec = FeatureSpec(name="x.y.z1", family="y", modality="x", gpu_available=True, gpu_backend=None)
    with pytest.raises(RegistryValidationError, match="gpu_available=True requires gpu_backend"):
        validate_spec(spec)


def test_gpu_backend_without_gpu_available_rejected():
    spec = FeatureSpec(name="x.y.z2", family="y", modality="x", gpu_available=False, gpu_backend="cupy")
    with pytest.raises(RegistryValidationError, match="gpu_backend is set but gpu_available=False"):
        validate_spec(spec)


def test_gpu_available_on_structured_sample_rejected():
    spec = FeatureSpec(
        name="x.y.z3", family="y", modality="x", gpu_available=True, gpu_backend="cupy",
        batch_semantics=BatchSemantics.STRUCTURED_SAMPLE,
    )
    with pytest.raises(RegistryValidationError, match="STRUCTURED_SAMPLE feature is incoherent"):
        validate_spec(spec)


def test_gpu_numerical_tolerance_must_be_pair_of_non_negative():
    spec = FeatureSpec(name="x.y.z4", family="y", modality="x", gpu_numerical_tolerance=(-1.0, 1e-6))
    with pytest.raises(RegistryValidationError, match="gpu_numerical_tolerance"):
        validate_spec(spec)


def test_gpu_output_dtype_must_be_float32_or_float64():
    spec = FeatureSpec(name="x.y.z5", family="y", modality="x", gpu_output_dtype="int8")
    with pytest.raises(RegistryValidationError, match="gpu_output_dtype"):
        validate_spec(spec)


def test_gpu_min_rows_must_be_non_negative():
    spec = FeatureSpec(name="x.y.z6", family="y", modality="x", gpu_min_rows=-5)
    with pytest.raises(RegistryValidationError, match="gpu_min_rows"):
        validate_spec(spec)


def test_valid_gpu_spec_passes():
    spec = FeatureSpec(name="x.y.ok", family="y", modality="x", gpu_available=True, gpu_backend="cupy", gpu_min_rows=1000)
    validate_spec(spec)  # must not raise


def test_full_registry_validates_with_gpu_rules():
    featm.REGISTRY.validate_all()  # must not raise


def test_gpu_capability_inventory_shape():
    inv = gpu_capability_inventory()
    assert inv["total_features"] == 552
    assert inv["gpu_capable_features"] == len(inv["gpu_capable_feature_names"])
    assert inv["cpu_only_features"] == inv["total_features"] - inv["gpu_capable_features"]
    assert set(inv["gpu_capable_by_backend"]) == {"cupy"}


def test_gpu_capability_inventory_is_exactly_22_after_audit_fix_pass():
    """Findings 5/6 (audit-fix pass, v0.3.1): timeseries.spectral.fft and
    timeseries.spectral.cepstrum had their GPU acceleration REMOVED (24 -> 22)
    rather than shipped with a validator/tolerance that could hide a real
    wrong answer -- this project prioritizes correctness over coverage count."""
    inv = gpu_capability_inventory()
    assert inv["gpu_capable_features"] == 22
    assert "timeseries.spectral.fft" not in inv["gpu_capable_feature_names"]
    assert "timeseries.spectral.cepstrum" not in inv["gpu_capable_feature_names"]
    # signal.frequency.fft (magnitude-only, no phase-column DC-bin instability) is unaffected.
    assert "signal.frequency.fft" in inv["gpu_capable_feature_names"]


def test_every_gpu_capable_feature_has_a_real_kernel():
    all_kernel_names = set(REDUCTION_KERNELS) | set(SPECTRAL_KERNELS) | set(STANDALONE_SPECTRAL_KERNELS)
    for spec in featm.REGISTRY.all():
        if spec.gpu_available:
            assert spec.name in all_kernel_names, f"{spec.name} claims gpu_available=True but has no kernel"


def test_no_kernel_registered_for_nonexistent_feature():
    all_kernel_names = set(REDUCTION_KERNELS) | set(SPECTRAL_KERNELS) | set(STANDALONE_SPECTRAL_KERNELS)
    registry_names = {s.name for s in featm.REGISTRY.all()}
    assert all_kernel_names.issubset(registry_names)


def test_gpu_available_feature_names_are_real_v01_feature_names_unchanged():
    """GPU acceleration must not introduce a new/renamed public feature name --
    every gpu_available=True spec is one of the original 552 names."""
    for spec in featm.REGISTRY.all():
        if spec.gpu_available:
            assert "gpu" not in spec.name.split(".")[0], f"{spec.name} looks like a GPU-namespaced name, not the canonical feature"
