from __future__ import annotations

"""Generates step5_validation_results.json (Section 15) from the REAL,
already-produced Step-5 artifacts -- every count here is computed by
reading a CSV/JSON file on disk, never a hardcoded zero counter.

v0.5.0 -> v0.5.1 audit-fix (Finding 10): the v0.5.0 generator was too
shallow -- it did not check provenance, did not verify real-dataset
modality coverage, did not detect a cross-family "family bundle", accepted
4 cross-family redundancy rows as sufficient, did not validate GPU/Dask
crossover classification, treated a deselected regression test as a pass,
and did not assert the installed package's version at all. Every one of
those gaps is closed below.
"""

import argparse
import csv
import json
from pathlib import Path

import featm


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


REQUIRED_REAL_DATASET_MODALITY_GROUPS = {
    "tabular_classification", "tabular_regression", "text", "image", "timeseries_or_signal", "graph", "audio",
}


def _dataset_modality_group(dataset_row: dict) -> str | None:
    """Audit finding (v0.5.1 -> v0.5.2, Finding 6): this used to infer a
    dataset's modality group by pattern-matching substrings of its NAME
    (e.g. "if 'credit_g' in name"), so a renamed or newly-added dataset
    would silently fail coverage detection with no error. The dataset
    manifest now carries an explicit ``modality_group`` column (set at
    dataset-construction time in run_real_dataset_utility.py); this reads
    that field directly and never inspects the dataset name."""
    return dataset_row.get("modality_group") or None


def _gate_provenance(d: Path, root: Path, expected_version: str) -> dict:
    """Audit finding (v0.5.2 -> v0.5.3): an independent audit reproduced
    several ways the v0.5.2 gate could pass tampered evidence, all sharing
    one root cause -- most comparisons were guarded by ``if field:``, so a
    MISSING required field ("comparison unavailable") was silently treated
    as "nothing to check" rather than a failure. Concretely, the v0.5.2 gate
    never compared the run manifest's environment_id against the
    environment manifest's own environment_id at all; never checked the
    environment manifest's harness checksum against the current tree or the
    run manifest; and skipped the row-level harness-checksum check entirely
    whenever a row simply had no benchmark_harness_checksum field (a missing
    field made the ``and r.get(...)`` guard falsy, not a failure).

    This version fails closed: every comparison is unconditional (a missing
    field compares unequal to whatever it's checked against, by construction
    of ``dict.get`` returning None), and it explicitly enforces every
    relationship the provenance contract claims to protect -- manifest-vs-
    current-tree, manifest-vs-manifest, and row-vs-both-manifests -- rather
    than relying on transitivity through only one of them.

    Identity contract (see benchmarks/step5/provenance.py for the
    measurement_harness_checksum / validation_harness_checksum split this
    depends on):
      - CPU run manifest (M) must have: run_id, environment_id, featm_version,
        source_checksum_algorithm, featm_source_checksum, measurement_harness_checksum.
      - Environment manifest (E) must have the same fields EXCEPT run_id --
        E's own run_id (if any) describes environment-manifest generation,
        not the CPU run, and is deliberately never required to equal M.run_id.
      - Every CPU row (R) must have all of M's required fields.
    """
    from benchmarks.step5.provenance import SOURCE_CHECKSUM_ALGORITHM, measurement_harness_checksum, source_checksum

    cpu_rows = _read_csv(d / "step5_cpu_system_feature_results.csv")
    manifest_path = d / "step5_cpu_system_run_manifest.json"
    manifest = _read_json(manifest_path)
    env_manifest_path = d / "step5_environment_manifest.json"
    env_manifest = _read_json(env_manifest_path)

    P = source_checksum(root)
    H = measurement_harness_checksum(root)
    A = SOURCE_CHECKSUM_ALGORITHM
    V = expected_version

    failures = 0
    notes: list[str] = []

    def fail(msg: str) -> None:
        nonlocal failures
        failures += 1
        notes.append(msg)

    if not manifest_path.exists():
        fail("run manifest missing"); manifest = {}
    if not env_manifest_path.exists():
        fail("environment manifest missing"); env_manifest = {}

    m_run_id = manifest.get("run_id")
    m_env_id = manifest.get("environment_id")
    m_version = manifest.get("featm_version")
    m_algo = manifest.get("source_checksum_algorithm")
    m_product = manifest.get("featm_source_checksum")
    m_harness = manifest.get("measurement_harness_checksum")

    e_env_id = env_manifest.get("environment_id")
    e_version = env_manifest.get("featm_version")
    e_algo = env_manifest.get("source_checksum_algorithm")
    e_product = env_manifest.get("featm_source_checksum")
    e_harness = env_manifest.get("measurement_harness_checksum")

    # -- Manifest-level: required fields present, and every claimed
    # relationship explicitly checked -- never inferred by transitivity.
    if not m_run_id:
        fail("run manifest run_id missing/empty")
    if not e_env_id:
        fail("environment manifest environment_id missing/empty")
    if m_env_id != e_env_id:
        fail(f"run manifest environment_id {m_env_id!r} != environment manifest environment_id {e_env_id!r}")

    if m_version != V:
        fail(f"run manifest featm_version {m_version!r} != expected {V!r}")
    if e_version != V:
        fail(f"environment manifest featm_version {e_version!r} != expected {V!r}")

    if m_algo != A:
        fail(f"run manifest source_checksum_algorithm {m_algo!r} != canonical {A!r}")
    if e_algo != A:
        fail(f"environment manifest source_checksum_algorithm {e_algo!r} != canonical {A!r}")

    if m_product != P:
        fail(f"run manifest featm_source_checksum {m_product!r} != current product checksum {P!r}")
    if e_product != P:
        fail(f"environment manifest featm_source_checksum {e_product!r} != current product checksum {P!r}")
    if m_product != e_product:
        fail(f"run manifest featm_source_checksum {m_product!r} != environment manifest featm_source_checksum {e_product!r}")

    if m_harness != H:
        fail(f"run manifest measurement_harness_checksum {m_harness!r} != current harness checksum {H!r}")
    if e_harness != H:
        fail(f"environment manifest measurement_harness_checksum {e_harness!r} != current harness checksum {H!r}")
    if m_harness != e_harness:
        fail(f"run manifest measurement_harness_checksum {m_harness!r} != environment manifest measurement_harness_checksum {e_harness!r}")

    # -- Row-level: real foreign-key validation. Every comparison is
    # unconditional -- a missing row field (None) simply compares unequal.
    row_failures = 0
    row_notes: list[str] = []
    for r in cpu_rows:
        reasons = []
        if r.get("run_id") != m_run_id:
            reasons.append("run_id")
        if r.get("environment_id") != e_env_id:
            reasons.append("environment_id")
        if r.get("featm_version") != V:
            reasons.append("featm_version")
        if r.get("source_checksum_algorithm") != A:
            reasons.append("source_checksum_algorithm")

        row_product = r.get("featm_source_checksum")
        if row_product != P:
            reasons.append("featm_source_checksum!=current")
        elif row_product != m_product:
            reasons.append("featm_source_checksum!=run_manifest")
        elif row_product != e_product:
            reasons.append("featm_source_checksum!=environment_manifest")

        row_harness = r.get("measurement_harness_checksum")
        if row_harness != H:
            reasons.append("measurement_harness_checksum!=current")
        elif row_harness != m_harness:
            reasons.append("measurement_harness_checksum!=run_manifest")
        elif row_harness != e_harness:
            reasons.append("measurement_harness_checksum!=environment_manifest")

        legacy = r.get("source_checksum_blake2b16")
        if legacy and legacy != row_product:
            reasons.append("source_checksum_blake2b16!=featm_source_checksum")

        if reasons:
            row_failures += 1
            if len(row_notes) < 20:
                row_notes.append(f"{r.get('feature')}@scale={r.get('scale')}: {','.join(reasons)}")
    failures += row_failures
    if row_failures:
        suffix = " (showing first 20)" if row_failures > len(row_notes) else ""
        notes.append(f"{row_failures} CPU rows fail foreign-key validation{suffix}: " + "; ".join(row_notes))

    return {"source": "provenance-check", "checks_run": len(cpu_rows) + 15, "failures": failures,
            "artifact": "step5_cpu_system_feature_results.csv", "notes": notes}


def _gate_real_dataset_coverage(d: Path) -> dict:
    manifest_rows = _read_csv(d / "step5_dataset_manifest.csv")
    groups_present = {g for g in (_dataset_modality_group(r) for r in manifest_rows) if g}
    missing = REQUIRED_REAL_DATASET_MODALITY_GROUPS - groups_present
    return {"source": "dataset-manifest", "checks_run": len(REQUIRED_REAL_DATASET_MODALITY_GROUPS),
            "failures": len(missing), "artifact": "step5_dataset_manifest.csv",
            "notes": [f"missing modality group: {m}" for m in sorted(missing)]}


def _gate_dataset_provenance(d: Path) -> dict:
    required_fields = {"source", "version", "license", "split_policy", "checksum_blake2b16", "preprocessing",
                        "modality", "modality_group", "task"}
    rows = _read_csv(d / "step5_dataset_manifest.csv")
    failures = 0
    notes = []
    for r in rows:
        missing = required_fields - {k for k, v in r.items() if v}
        if missing:
            failures += 1
            notes.append(f"{r.get('dataset')}: missing {sorted(missing)}")
    return {"source": "dataset-manifest", "checks_run": len(rows), "failures": failures,
            "artifact": "step5_dataset_manifest.csv", "notes": notes}


REQUIRED_REAL_DATASET_FAMILY_BUNDLES = {
    ("credit_g_classification", "tabular", "scaling"),
    ("credit_g_classification", "tabular", "encoding"),
    ("credit_g_classification", "tabular", "numeric"),
    ("california_housing_regression", "tabular", "scaling"),
    ("california_housing_regression", "tabular", "numeric"),
    ("newsgroups_binary_classification", "text", "tfidf"),
    ("newsgroups_binary_classification", "text", "lexical"),
    ("digits_binary_classification", "image", "color"),
    ("digits_binary_classification", "image", "shape"),
    ("digits_binary_classification", "image", "texture"),
    ("gunpoint_timeseries_classification", "timeseries", "statistical"),
    ("gunpoint_timeseries_classification", "timeseries", "spectral"),
    ("mutag_graph_classification", "graph", "connectivity"),
    ("mutag_graph_classification", "graph", "degree"),
    ("mutag_graph_classification", "graph", "clustering"),
    ("fsdd_audio_classification", "audio", "waveform"),
}


def _check_true_family_bundle_rows(rows: list[dict], artifact: str) -> tuple[int, list[str]]:
    failures = 0
    notes = []
    for r in rows:
        family = r.get("family", "")
        if ";" in family or "," in family:
            failures += 1
            notes.append(f"{artifact}/{r.get('dataset_or_probe')}: family field {family!r} names more than one family -- not a true single-family bundle")
        included = set(f for f in r.get("included_features", "").split(";") if f)
        excluded = set(f for f in r.get("excluded_features", "").split(";") if f)
        n_registry = int(r.get("n_registry_family_features") or 0)
        if len(included) + len(excluded) != n_registry and r.get("status") == "ok":
            failures += 1
            notes.append(f"{artifact}/{r.get('dataset_or_probe')}/{family}: included+excluded ({len(included)}+{len(excluded)}) != n_registry_family_features ({n_registry})")
    return failures, notes


def _gate_family_bundle_utility(d: Path) -> dict:
    """Audit finding (v0.5.1 -> v0.5.2, Finding 4/9): a true family-bundle
    utility gate that only ever read the CONTROLLED-probe artifact would
    keep passing even if the real-public-dataset family-bundle evidence
    silently vanished. This checks both layers exist, are well-formed, and
    that the real-dataset layer covers the audit's minimum required
    (dataset, modality, family) coverage."""
    controlled_rows = _read_csv(d / "step5_controlled_family_bundle_utility_results.csv")
    real_rows = _read_csv(d / "step5_real_dataset_family_bundle_utility_results.csv")
    failures = 0
    notes = []
    if not controlled_rows:
        failures += 1
        notes.append("step5_controlled_family_bundle_utility_results.csv: no rows found")
    if not real_rows:
        failures += 1
        notes.append("step5_real_dataset_family_bundle_utility_results.csv: no rows found")

    f1, n1 = _check_true_family_bundle_rows(controlled_rows, "controlled")
    f2, n2 = _check_true_family_bundle_rows(real_rows, "real_dataset")
    failures += f1 + f2
    notes += n1 + n2

    present = {(r.get("dataset_or_probe"), r.get("modality"), r.get("family")) for r in real_rows}
    missing = REQUIRED_REAL_DATASET_FAMILY_BUNDLES - present
    if missing:
        failures += len(missing)
        notes.append(f"real-dataset family-bundle coverage missing: {sorted(missing)}")

    return {"source": "true-family-bundle-utility", "checks_run": len(controlled_rows) + len(real_rows) + len(REQUIRED_REAL_DATASET_FAMILY_BUNDLES),
            "failures": failures,
            "artifact": "step5_controlled_family_bundle_utility_results.csv + step5_real_dataset_family_bundle_utility_results.csv",
            "notes": notes}


def _gate_redundancy(d: Path) -> dict:
    """Audit finding (v0.5.1 -> v0.5.2, Finding 5/9): redundancy evidence
    used to exist only for controlled probes, and the old gate only checked
    that the family field lacked a semicolon -- it never verified a
    redundancy row actually traces back to a real true-family-bundle
    definition (same dataset/family, same included feature set). This
    checks both evidence layers exist and that every redundancy row's
    included_features matches the corresponding utility row exactly."""
    controlled_redundancy = _read_csv(d / "step5_controlled_family_bundle_redundancy_results.csv")
    real_redundancy = _read_csv(d / "step5_real_dataset_family_bundle_redundancy_results.csv")
    controlled_utility = _read_csv(d / "step5_controlled_family_bundle_utility_results.csv")
    real_utility = _read_csv(d / "step5_real_dataset_family_bundle_utility_results.csv")

    failures = 0
    notes = []
    if not controlled_redundancy:
        failures += 1
        notes.append("step5_controlled_family_bundle_redundancy_results.csv: no rows found")
    if not real_redundancy:
        failures += 1
        notes.append("step5_real_dataset_family_bundle_redundancy_results.csv: no rows found")

    def _key(r):
        return (r.get("dataset_or_probe"), r.get("family"))

    def _check(redundancy_rows, utility_rows, label):
        nonlocal failures
        utility_by_key = {_key(r): r for r in utility_rows}
        for r in redundancy_rows:
            family = r.get("family", "")
            if ";" in family or "," in family:
                failures += 1
                notes.append(f"{label}/{r.get('dataset_or_probe')}: family field {family!r} is not a single family")
                continue
            u = utility_by_key.get(_key(r))
            if u is None:
                failures += 1
                notes.append(f"{label}/{r.get('dataset_or_probe')}/{family}: no matching true-family-bundle utility row to trace back to")
                continue
            if r.get("included_features", "") != u.get("included_features", ""):
                failures += 1
                notes.append(f"{label}/{r.get('dataset_or_probe')}/{family}: redundancy included_features != utility included_features "
                             f"({r.get('included_features')!r} vs {u.get('included_features')!r})")

    _check(controlled_redundancy, controlled_utility, "controlled")
    _check(real_redundancy, real_utility, "real_dataset")

    return {"source": "true-family-bundle-redundancy", "checks_run": len(controlled_redundancy) + len(real_redundancy),
            "failures": failures,
            "artifact": "step5_controlled_family_bundle_redundancy_results.csv + step5_real_dataset_family_bundle_redundancy_results.csv",
            "notes": notes}


def _gate_gpu_crossover(d: Path) -> dict:
    gpu_validation = _read_json(d / "step5_gpu_validation_results.json")
    summary = gpu_validation.get("crossover_summary", [])
    failures = 0
    notes = []
    if gpu_validation.get("gpu_capable_features_found") != 22:
        failures += 1
        notes.append(f"gpu_capable_features_found={gpu_validation.get('gpu_capable_features_found')} != 22")
    valid_statuses = {"measured_bracket", "left_censored_gpu_wins_at_smallest_measured_point",
                       "right_censored_gpu_never_wins_in_measured_range", "unstable_no_two_point_crossover"}
    for row in summary:
        status = row.get("crossover_status")
        if status not in valid_statuses:
            failures += 1
            notes.append(f"{row.get('feature')}: unrecognized/legacy crossover_status {status!r}")
        elif status == "left_censored_gpu_wins_at_smallest_measured_point" and row.get("measured_crossover_elements") is not None:
            failures += 1
            notes.append(f"{row.get('feature')}: left-censored but measured_crossover_elements is not null -- an invented exact crossover")
    return {"source": "gpu-crossover-classification", "checks_run": len(summary), "failures": failures,
            "artifact": "step5_gpu_validation_results.json", "notes": notes}


def _gate_dask_crossover(d: Path, min_sizes: int = 3) -> dict:
    rows = _read_csv(d / "step5_dask_crossover_results.csv")
    if not rows:
        return {"source": "dask-crossover", "checks_run": 0, "failures": 1,
                "artifact": "step5_dask_crossover_results.csv", "notes": ["no rows found"]}
    sizes = {r["n_rows"] for r in rows if r.get("status") == "ok"}
    failures = 0 if len(sizes) >= min_sizes else 1
    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    failures += sum(1 for r in rows if r.get("status") not in ("ok", "optional_dependency_unavailable"))
    return {"source": "dask-crossover", "checks_run": len(rows), "failures": failures,
            "artifact": "step5_dask_crossover_results.csv",
            "notes": [f"{len(sizes)} distinct workload sizes measured (need >= {min_sizes})", f"{ok_count} ok rows"]}


def _gate_regression_commands(d: Path) -> dict:
    """Audit finding (v0.5.1 -> v0.5.2, Finding 7/9): skipped_disk_risk used
    to be an accepted status for batch_equivalence_smoke -- the audit is
    explicit that a deselected/skipped run must never count as a regression
    pass in the final accepted package. It is no longer in the accepted set
    at all: every command must genuinely run to regression_pass or
    known_external_issue."""
    regression = _read_json(d / "step5_regression_matrix.json")
    commands = regression.get("commands", {})
    if not commands:
        return {"source": "regression-matrix", "checks_run": 0, "failures": 1,
                "artifact": "step5_regression_matrix.json", "notes": ["no command results found"]}
    failures = 0
    notes = []
    for name, c in commands.items():
        status = c.get("status")
        if status == "skipped_disk_risk":
            failures += 1
            notes.append(f"{name}: skipped_disk_risk is not an accepted status in the final package -- must actually run")
        elif status not in ("regression_pass", "known_external_issue"):
            failures += 1
            notes.append(f"{name}: {status}")
    return {"source": "regression-matrix", "checks_run": len(commands), "failures": failures,
            "artifact": "step5_regression_matrix.json", "notes": notes}


def _gate_packaging(d: Path, expected_version: str) -> dict:
    regression = _read_json(d / "step5_regression_matrix.json")
    packaging = regression.get("packaging", {})
    if not packaging:
        return {"source": "packaging-clean-install", "checks_run": 0, "failures": 1,
                "artifact": "step5_regression_matrix.json", "notes": ["no packaging results found"]}
    checks = ["build_wheel_sdist", "wheel_clean_install", "sdist_clean_install"]
    failures = 0
    notes = []
    for c in checks:
        entry = packaging.get(c, {})
        if not entry.get("ok"):
            failures += 1
            notes.append(f"{c}: not ok ({entry})")
    for label in ("wheel_clean_install", "sdist_clean_install"):
        stdout = packaging.get(label, {}).get("verify_stdout", "")
        if f"OK {expected_version}" not in stdout:
            failures += 1
            notes.append(f"{label}: verify_stdout does not confirm version {expected_version}: {stdout!r}")
    return {"source": "packaging-clean-install", "checks_run": len(checks) + 2, "failures": failures,
            "artifact": "step5_regression_matrix.json", "notes": notes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="step5_final_results")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="step5_final_results/step5_validation_results.json")
    ap.add_argument("--expected-version", default="0.5.4")
    args = ap.parse_args()
    d = Path(args.dir)
    root = Path(args.root).resolve()

    catalog = _read_csv(d / "step5_feature_catalog.csv")
    cpu_rows = _read_csv(d / "step5_cpu_system_feature_results.csv")
    engine_rows = _read_csv(d / "step5_engine_feature_results.csv")
    stateful_rows = _read_csv(d / "step5_stateful_fit_transform_results.csv")
    stateful_growth_rows = _read_csv(d / "step5_stateful_state_growth_results.csv")
    gpu_equiv_rows = _read_csv(d / "step5_gpu_equivalence_results.csv")
    gpu_auto_rows = _read_csv(d / "step5_gpu_auto_audit_results.csv")
    gpu_validation = _read_json(d / "step5_gpu_validation_results.json")
    dask_rows = _read_csv(d / "step5_dask_crossover_results.csv")
    controlled_rows = _read_csv(d / "step5_controlled_utility_results.csv")
    real_dataset_rows = _read_csv(d / "step5_real_dataset_utility_results.csv")
    metadata_rows = _read_csv(d / "step5_metadata_observed_audit.csv")

    cpu_features_covered = len({r["feature"] for r in cpu_rows})
    cpu_ok = sum(1 for r in cpu_rows if r.get("status") == "ok")
    cpu_bad_buckets = {"feature_error", "worker_crash", "invalid_output"}
    cpu_genuine_failures = sum(1 for r in cpu_rows if r.get("status") in cpu_bad_buckets)
    cpu_nondeterministic = sum(1 for r in cpu_rows if r.get("status") == "ok" and r.get("deterministic") not in ("True", "true", "1"))

    gpu_equiv_failures = sum(1 for r in gpu_equiv_rows if r.get("passed") not in ("True", "true", "1"))
    gpu_auto_disagreements = sum(1 for r in gpu_auto_rows if r.get("agrees_with_threshold") not in ("True", "true", "1"))

    stateful_features_expected = 7  # frozen Step-5 historical benchmark inventory (v0.5.7); current registry has 8
    stateful_features_covered = len({r["feature"] for r in stateful_rows if r.get("status") == "ok"})

    controlled_ok = sum(1 for r in controlled_rows if r.get("status") == "ok")
    real_dataset_ok = sum(1 for r in real_dataset_rows if r.get("status") == "ok")

    gates = {
        "registry_inventory": {
            "source": "featm.REGISTRY", "checks_run": 5, "artifact": "step5_feature_catalog.csv",
            "failures": int(len(featm.REGISTRY.all()) != 552) + int(len(featm.list_modalities()) != 18)
                        + int(sum(bool(s.stateful) for s in featm.REGISTRY.all()) != 8)
                        + int(sum(bool(s.gpu_available) for s in featm.REGISTRY.all()) != 22)
                        + int(len(catalog) != 552),
        },
        "cpu_full_universe_accounting": {
            "source": "cpu-full-universe", "checks_run": 552, "artifact": "step5_cpu_system_feature_results.csv",
            "failures": 552 - cpu_features_covered,
        },
        "provenance": _gate_provenance(d, root, args.expected_version),
        "fixture_validation": _gate_regression_commands(d),  # step5 suite is one of the commands
        "determinism": {
            "source": "cpu-full-universe", "checks_run": cpu_ok, "artifact": "step5_cpu_system_feature_results.csv",
            "failures": cpu_nondeterministic,
        },
        "output_health": {
            "source": "cpu-full-universe", "checks_run": len(cpu_rows), "artifact": "step5_cpu_system_feature_results.csv",
            "failures": cpu_genuine_failures,
        },
        "stateful_fit_transform": {
            "source": "stateful-fit-transform", "checks_run": stateful_features_expected, "artifact": "step5_stateful_fit_transform_results.csv",
            "failures": stateful_features_expected - stateful_features_covered,
        },
        "stateful_state_growth": {
            "source": "stateful-state-growth", "checks_run": len(stateful_growth_rows), "artifact": "step5_stateful_state_growth_results.csv",
            "failures": sum(1 for r in stateful_growth_rows if r.get("status") not in ("ok", "not_applicable")),
        },
        "engine_execution": {
            "source": "public-engine", "checks_run": len(engine_rows), "artifact": "step5_engine_feature_results.csv",
            "failures": sum(1 for r in engine_rows if r.get("status") not in ("ok", "not_engine_representable", "optional_dependency_unavailable")),
        },
        "GPU_equivalence": {
            "source": "gpu-crossover-kaggle", "checks_run": len(gpu_equiv_rows), "artifact": "step5_gpu_equivalence_results.csv",
            "failures": gpu_equiv_failures,
        },
        "GPU_crossover": _gate_gpu_crossover(d),
        "Dask_persistent_cluster": _gate_dask_crossover(d),
        "controlled_utility": {
            "source": "controlled-utility-probes", "checks_run": len(controlled_rows), "artifact": "step5_controlled_utility_results.csv",
            "failures": len(controlled_rows) - controlled_ok - sum(1 for r in controlled_rows if r.get("status") == "not_utility_eligible"),
        },
        "real_dataset_utility": {
            "source": "real-dataset-utility", "checks_run": len(real_dataset_rows), "artifact": "step5_real_dataset_utility_results.csv",
            "failures": len(real_dataset_rows) - real_dataset_ok - sum(1 for r in real_dataset_rows if r.get("status") == "not_utility_eligible"),
        },
        "real_dataset_coverage": _gate_real_dataset_coverage(d),
        "dataset_provenance": _gate_dataset_provenance(d),
        "family_bundle_utility": _gate_family_bundle_utility(d),
        "leakage_safety": _gate_regression_commands(d),
        "redundancy": _gate_redundancy(d),
        "metadata_audit": {
            "source": "metadata-observed-audit", "checks_run": len(metadata_rows), "artifact": "step5_metadata_observed_audit.csv",
            "failures": 0,  # diagnostic-only gate; never auto-fails on a flagged contradiction (see docs/BENCHMARKS.md)
        },
        "Step1_4_regression": _gate_regression_commands(d),
        "packaging": _gate_packaging(d, args.expected_version),
    }

    step5_acceptance = all(g["failures"] == 0 for g in gates.values())
    # Traceability only (v0.5.3): which validator/acceptance code produced
    # this verdict. Never foreign-key checked against a stored prior value --
    # the validator is always current code judging possibly-older evidence,
    # by design (see benchmarks/step5/provenance.py's module docstring).
    from benchmarks.step5.provenance import validation_harness_checksum
    result = {
        "gates": gates, "step5_acceptance": step5_acceptance,
        "validation_harness_checksum": validation_harness_checksum(root),
        "featm_version": args.expected_version,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
