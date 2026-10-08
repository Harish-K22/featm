import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import featm


def test_registry_is_complete_and_executable():
    # Step 2 adds 3 new out-of-core stateful encoders (tabular.scaling.zscore,
    # tabular.scaling.minmax, tabular.encoding.category_frequency) on top of the
    # original v0.1.0 549-feature universe; see test_v0_1_0_names_still_present
    # for the precise compatibility guarantee that none of the original 549
    # were renamed or removed.
    specs = featm.REGISTRY.all()
    assert len(specs) == 549 + 3
    assert len(featm.list_modalities()) == 18
    assert all(spec.implementation is not None for spec in specs)


def test_v0_1_0_names_still_present():
    """Every one of the original v0.1.0 549 registered names must still resolve.

    This is the precise form of "preserve the 549-feature registry": Step 2 may
    add new features, but must never rename or remove an original one.
    """
    fixture = Path(__file__).parent / "fixtures" / "registry_v0.1.0_names.json"
    original_names = json.loads(fixture.read_text())
    assert len(original_names) == 549
    current_names = {s.name for s in featm.REGISTRY.all()}
    missing = set(original_names) - current_names
    assert not missing, f"{len(missing)} v0.1.0 feature name(s) went missing: {sorted(missing)[:10]}"


def test_registry_validation_catches_malformed_metadata():
    from featm.registry import Cost, FeatureRegistry, FeatureSpec, RegistryValidationError

    registry = FeatureRegistry()
    with pytest.raises(RegistryValidationError, match="dotted lowercase"):
        registry.register(FeatureSpec(name="NotDotted", family="f", modality="m"))
    with pytest.raises(RegistryValidationError, match="output_kind"):
        registry.register(FeatureSpec(name="a.b", family="f", modality="m", output_kind="bogus"))
    with pytest.raises(RegistryValidationError, match="arity"):
        registry.register(FeatureSpec(name="a.c", family="f", modality="m", arity=0))
    with pytest.raises(RegistryValidationError, match="exact_available"):
        registry.register(FeatureSpec(name="a.d", family="f", modality="m", exact_available=False, approximate_available=False))
    # A well-formed spec still registers cleanly and gets sensible complexity/memory defaults.
    ok = registry.register(FeatureSpec(name="a.e", family="f", modality="m", cost=Cost.HIGH))
    from featm.registry import Complexity, MemoryProfile
    assert ok.complexity == Complexity.QUADRATIC
    assert ok.memory_profile == MemoryProfile.LOG_LINEAR
    registry.validate_all()  # should not raise for the one well-formed entry


def test_registry_validate_all_on_real_registry():
    featm.REGISTRY.validate_all()


def test_registry_has_expected_modalities_and_discovery():
    for modality in (
        "tabular", "timeseries", "image", "audio", "video", "signal",
        "graph", "geometry", "pointcloud", "text", "information",
        "topology", "spatial", "manifold", "matrix", "sequence", "event",
        "multimodal",
    ):
        assert featm.list_features(modality=modality)
    assert featm.list_features("*delaunay*")
    assert featm.list_features("*entropy*")
    assert featm.search("triangulation")


def test_no_features_all_shortcut():
    x = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="no features='all' mode"):
        featm.extract(x, features="all")


def test_create_feature_timm_style():
    f = featm.create_feature("timeseries.statistical.mean")
    x = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    out = np.asarray(f(x))
    assert np.allclose(out, [2.0, 5.0])
    assert f.name == "timeseries.statistical.mean"


def test_explicit_timeseries_extraction():
    x = np.array([[1, -1, 1, -1], [1, 1, 1, 1]], dtype=float)
    f = featm.extract(
        x,
        features=[
            "timeseries.statistical.mean",
            "timeseries.complexity.zero_crossing_rate",
        ],
    )
    assert list(f.columns) == [
        "timeseries.statistical.mean",
        "timeseries.complexity.zero_crossing_rate",
    ]
    assert f.loc[0, "timeseries.complexity.zero_crossing_rate"] > f.loc[1, "timeseries.complexity.zero_crossing_rate"]


def test_vector_feature_expands_stable_columns():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    f = featm.extract(x, features=["timeseries.spectral.fft"])
    assert len(f) == 2
    assert len(f.columns) > 1
    assert all(c.startswith("timeseries.spectral.fft__") for c in f.columns)


def test_explicit_image_extraction():
    imgs = np.zeros((3, 8, 8), dtype=np.uint8)
    imgs[1, 2:6, 2:6] = 255
    f = featm.extract(imgs, features=["image.intensity.mean"])
    assert len(f) == 3
    assert "image.intensity.mean" in f
    assert f.loc[1, "image.intensity.mean"] > f.loc[0, "image.intensity.mean"]


def test_delaunay_feature_is_directly_selectable():
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    f = featm.extract(pts, features=["geometry.triangulation.delaunay_degree"])
    assert len(f) == len(pts)
    assert np.nanmin(f.iloc[:, 0]) >= 2


def test_feature_info_reports_optional_requirements():
    spec = featm.feature_info("image.embeddings.timm")
    assert spec.executable
    assert "timm" in spec.requires
    assert "torch" in spec.requires


def test_modality_mismatch_is_rejected():
    x = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="do not belong"):
        featm.extract(
            x,
            modality="image",
            features=["timeseries.statistical.mean"],
        )


def test_mixed_modalities_are_rejected_without_multimodal_feature():
    x = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Mixed-modality"):
        featm.extract(
            x,
            features=["timeseries.statistical.mean", "signal.time_domain.rms"],
        )


def test_legacy_planner_has_no_all_preset():
    with pytest.raises(ValueError, match="no automatic 'all' mode"):
        featm.AutoFeatures(preset="all")


def test_schema_and_legacy_tabular_replay():
    train = pd.DataFrame({
        "area": [100.0, 150.0, 200.0, 250.0],
        "rooms": [1, 2, 3, 4],
        "city": ["A", "A", "B", "C"],
        "sold_at": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
    })
    test = pd.DataFrame({
        "area": [300.0],
        "rooms": [5],
        "city": ["UNSEEN"],
        "sold_at": ["2026-05-01"],
    })
    engine = featm.AutoFeatures(preset="fast", max_output_features=1000)
    xt = engine.fit_transform(train)
    xv = engine.transform(test)
    assert xt.shape[1] > train.shape[1]
    freq_cols = [c for c in xv.columns if c.endswith("__frequency")]
    assert freq_cols
    assert float(xv[freq_cols[0]].iloc[0]) == 0.0
    assert engine.report_.computed > 0


def test_planner_batching_and_memory_retry():
    from featm import ResourceBudget
    from featm.executor import FaultTolerantExecutor
    from featm.planner import ExecutionPlan, FeatureTask
    from featm.registry import Cost, FeatureRegistry, FeatureSpec

    budget = ResourceBudget(memory_bytes=300_000, disk_bytes=1_000_000, cpu_workers=1)
    df = pd.DataFrame({"x": np.arange(5_000, dtype=float), "y": np.arange(5_000, dtype=float) + 1})
    engine = featm.AutoFeatures(preset="fast", max_output_features=20, budget=budget)
    plan = engine.plan(df)
    assert plan.tasks
    assert plan.batch_rows <= len(df)

    registry = FeatureRegistry()

    def fragile(s, *, fit, state=None):
        if len(s) > 600:
            raise MemoryError("simulated batch too large")
        return s.astype(float) + 1.0, None

    registry.register(FeatureSpec(
        name="test.fragile",
        family="test",
        modality="tabular",
        input_types=("numeric",),
        cost=Cost.LOW,
        streamable=True,
        implementation=fragile,
    ))
    plan = ExecutionPlan(
        tasks=[FeatureTask("test.fragile", ("x",), {}, "x__fragile")],
        skipped=[],
        batch_rows=2_000,
        estimated_output_bytes=40_000,
    )
    out, _, results = FaultTolerantExecutor(registry).execute_tabular(df[["x"]], plan, fit=True)
    assert results[0].status == "ok"
    assert out["x__fragile"].iloc[-1] == 5_000.0


def test_cli_registry_commands():
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, "-m", "featm.cli", "list", "*delaunay*"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "geometry.triangulation.delaunay_degree" in result.stdout

    result = subprocess.run(
        [sys.executable, "-m", "featm.cli", "info", "timeseries.statistical.mean", "--json"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert '"name": "timeseries.statistical.mean"' in result.stdout
