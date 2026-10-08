from __future__ import annotations

"""Adversarial tamper-test matrix for _gate_provenance() (v0.5.2 -> v0.5.3
audit-fix pass).

An independent audit reproduced several ways the v0.5.2 gate could pass
tampered evidence -- all traced to the same root cause: most comparisons
were guarded by ``if field:``, so a MISSING required field was silently
treated as "nothing to check" instead of a failure. The v0.5.2 regression
suite only proved ONE mutation was caught (a wrong row.run_id); that was
not enough to prove the gate is actually fail-closed.

This module builds one fully-valid synthetic (CPU row, run manifest,
environment manifest) triple, confirms it passes with zero failures, then
independently mutates exactly ONE field at a time -- by deleting it or by
setting it to an unrelated value -- and asserts the gate now reports at
least one failure. Every case is checked against the SAME clean baseline
(never against a previously-mutated state), so failures cannot mask or
compound each other.
"""

import csv
import json
from pathlib import Path

import pytest

from benchmarks.step5.gen_validation_results import _gate_provenance
from benchmarks.step5.provenance import make_provenance

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.5.3"
BOGUS = "0" * 32  # a syntactically-plausible but definitely-wrong checksum/id value


def _write_row_csv(d: Path, rows: list[dict]) -> None:
    fields = sorted({k for r in rows for k in r})
    with open(d / "step5_cpu_system_feature_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_manifest(d: Path, name: str, manifest: dict) -> None:
    (d / name).write_text(json.dumps(manifest), encoding="utf-8")


def _baseline(d: Path) -> tuple[dict, dict, dict]:
    """Returns (row, run_manifest, env_manifest) -- all mutually consistent
    and consistent with the ACTUAL current source/harness tree, so the
    unmutated gate call is genuinely clean, not clean by coincidence."""
    prov = make_provenance(ROOT, VERSION)
    row = {"feature": "tabular.scaling.zscore", "scale": "1", "status": "ok", **prov}
    run_manifest = {**prov}
    env_manifest = {**prov}
    _write_row_csv(d, [row])
    _write_manifest(d, "step5_cpu_system_run_manifest.json", run_manifest)
    _write_manifest(d, "step5_environment_manifest.json", env_manifest)
    return row, run_manifest, env_manifest


def test_baseline_triple_is_clean(tmp_path):
    _baseline(tmp_path)
    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] == 0, result


# ---------------------------------------------------------------------------
# Row mutations (cases 1-12): delete or corrupt one required field on the
# single CPU row, leaving both manifests at their correct baseline values.
# ---------------------------------------------------------------------------
ROW_CASES = [
    ("missing_run_id", "run_id", None),
    ("wrong_run_id", "run_id", "run-WRONG-0000000000-0"),
    ("missing_environment_id", "environment_id", None),
    ("wrong_environment_id", "environment_id", "env-WRONG-0.5.3-000000000000-000000000000"),
    ("missing_featm_version", "featm_version", None),
    ("wrong_featm_version", "featm_version", "9.9.9"),
    ("missing_source_checksum_algorithm", "source_checksum_algorithm", None),
    ("wrong_source_checksum_algorithm", "source_checksum_algorithm", "sha256-legacy-v0"),
    ("missing_featm_source_checksum", "featm_source_checksum", None),
    ("wrong_featm_source_checksum", "featm_source_checksum", BOGUS),
    ("missing_measurement_harness_checksum", "measurement_harness_checksum", None),
    ("wrong_measurement_harness_checksum", "measurement_harness_checksum", BOGUS),
]


@pytest.mark.parametrize("case_id,field,bad_value", ROW_CASES, ids=[c[0] for c in ROW_CASES])
def test_row_mutation_is_caught(tmp_path, case_id, field, bad_value):
    row, _run_manifest, _env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0  # sanity: baseline itself is clean

    mutated = dict(row)
    if bad_value is None:
        mutated.pop(field, None)
    else:
        mutated[field] = bad_value
    _write_row_csv(tmp_path, [mutated])

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, f"{case_id} was NOT caught: {result}"


# ---------------------------------------------------------------------------
# CPU run-manifest mutations (cases 13-19).
# ---------------------------------------------------------------------------
RUN_MANIFEST_CASES = [
    ("missing_run_id", "run_id", None),
    ("missing_environment_id", "environment_id", None),
    ("wrong_environment_id", "environment_id", "env-WRONG-0.5.3-000000000000-000000000000"),
    ("missing_featm_version", "featm_version", None),
    ("wrong_featm_version", "featm_version", "9.9.9"),
    ("missing_checksum_algorithm", "source_checksum_algorithm", None),
    ("wrong_checksum_algorithm", "source_checksum_algorithm", "sha256-legacy-v0"),
    ("missing_product_checksum", "featm_source_checksum", None),
    ("wrong_product_checksum", "featm_source_checksum", BOGUS),
    ("missing_harness_checksum", "measurement_harness_checksum", None),
    ("wrong_harness_checksum", "measurement_harness_checksum", BOGUS),
]


@pytest.mark.parametrize("case_id,field,bad_value", RUN_MANIFEST_CASES, ids=[c[0] for c in RUN_MANIFEST_CASES])
def test_run_manifest_mutation_is_caught(tmp_path, case_id, field, bad_value):
    _row, run_manifest, _env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0

    mutated = dict(run_manifest)
    if bad_value is None:
        mutated.pop(field, None)
    else:
        mutated[field] = bad_value
    _write_manifest(tmp_path, "step5_cpu_system_run_manifest.json", mutated)

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, f"{case_id} was NOT caught: {result}"


# ---------------------------------------------------------------------------
# Environment-manifest mutations (cases 20-25).
# ---------------------------------------------------------------------------
ENV_MANIFEST_CASES = [
    ("missing_environment_id", "environment_id", None),
    ("wrong_environment_id", "environment_id", "env-WRONG-0.5.3-000000000000-000000000000"),
    ("missing_featm_version", "featm_version", None),
    ("wrong_featm_version", "featm_version", "9.9.9"),
    ("missing_checksum_algorithm", "source_checksum_algorithm", None),
    ("wrong_checksum_algorithm", "source_checksum_algorithm", "sha256-legacy-v0"),
    ("missing_product_checksum", "featm_source_checksum", None),
    ("wrong_product_checksum", "featm_source_checksum", BOGUS),
    ("missing_harness_checksum", "measurement_harness_checksum", None),
    ("wrong_harness_checksum", "measurement_harness_checksum", BOGUS),
]


@pytest.mark.parametrize("case_id,field,bad_value", ENV_MANIFEST_CASES, ids=[c[0] for c in ENV_MANIFEST_CASES])
def test_environment_manifest_mutation_is_caught(tmp_path, case_id, field, bad_value):
    _row, _run_manifest, env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0

    mutated = dict(env_manifest)
    if bad_value is None:
        mutated.pop(field, None)
    else:
        mutated[field] = bad_value
    _write_manifest(tmp_path, "step5_environment_manifest.json", mutated)

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, f"{case_id} was NOT caught: {result}"


# ---------------------------------------------------------------------------
# Cross-manifest mismatches (cases 26-28): each manifest looks individually
# plausible (non-empty, well-formed) but the two disagree with each other,
# or one silently drifted from the current tree while the other did not.
# This is exactly the class of bug the v0.5.2 gate missed entirely: it
# extracted manifest_env_id but never compared it to env_manifest_id.
# ---------------------------------------------------------------------------
def test_run_and_environment_manifest_reference_different_environment_ids(tmp_path):
    row, run_manifest, env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0

    # Both individually well-formed (non-empty, real-looking) environment_id
    # values -- just DIFFERENT from each other.
    mutated_run_manifest = {**run_manifest, "environment_id": "env-OTHERHOST-0.5.3-000000000000-000000000000"}
    _write_manifest(tmp_path, "step5_cpu_system_run_manifest.json", mutated_run_manifest)

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, f"differing run/environment manifest environment_id was NOT caught: {result}"


def test_environment_manifest_product_checksum_wrong_while_run_manifest_correct(tmp_path):
    row, run_manifest, env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0

    # run_manifest.featm_source_checksum stays == current (correct); only
    # the environment manifest's copy is wrong.
    mutated_env_manifest = {**env_manifest, "featm_source_checksum": BOGUS}
    _write_manifest(tmp_path, "step5_environment_manifest.json", mutated_env_manifest)

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, f"environment-manifest-only product checksum drift was NOT caught: {result}"


def test_environment_manifest_harness_checksum_wrong_while_run_manifest_correct(tmp_path):
    row, run_manifest, env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0

    # run_manifest.measurement_harness_checksum stays == current (correct);
    # only the environment manifest's copy is wrong.
    mutated_env_manifest = {**env_manifest, "measurement_harness_checksum": BOGUS}
    _write_manifest(tmp_path, "step5_environment_manifest.json", mutated_env_manifest)

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, f"environment-manifest-only harness checksum drift was NOT caught: {result}"


# ---------------------------------------------------------------------------
# Section 5's explicit requirement: changing ONLY the algorithm tag, with
# digest text otherwise unchanged, must fail. (Already implied by the
# *_wrong_checksum_algorithm parametrized cases above; named explicitly here
# for direct traceability to that requirement.)
# ---------------------------------------------------------------------------
def test_algorithm_tag_change_alone_fails_even_with_unchanged_digests(tmp_path):
    row, run_manifest, env_manifest = _baseline(tmp_path)
    assert _gate_provenance(tmp_path, ROOT, VERSION)["failures"] == 0

    # Digests (featm_source_checksum, measurement_harness_checksum) are left
    # completely untouched -- only the algorithm label changes.
    mutated_row = {**row, "source_checksum_algorithm": "blake2b16-posix-path-v1"}
    _write_row_csv(tmp_path, [mutated_row])

    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, (
        "a row whose source_checksum_algorithm tag was changed (digests otherwise "
        f"unchanged) must fail provenance validation: {result}"
    )


# ---------------------------------------------------------------------------
# Fail-closed sanity: an entirely missing manifest file must fail, not be
# silently treated as "no constraints to check".
# ---------------------------------------------------------------------------
def test_missing_run_manifest_file_fails(tmp_path):
    _baseline(tmp_path)
    (tmp_path / "step5_cpu_system_run_manifest.json").unlink()
    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, result


def test_missing_environment_manifest_file_fails(tmp_path):
    _baseline(tmp_path)
    (tmp_path / "step5_environment_manifest.json").unlink()
    result = _gate_provenance(tmp_path, ROOT, VERSION)
    assert result["failures"] > 0, result
