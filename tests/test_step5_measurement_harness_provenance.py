from __future__ import annotations

"""Measurement-harness dependency-closure tests (v0.5.3 -> v0.5.4 audit-fix
pass).

Audit finding: worker.py does
``from benchmarks.step5.fixtures import build_call, normalize_output`` --
fixtures.py controls the deterministic seed, every generated benchmark
input's shape/values, scale behavior, dummy-model substitution, and output
normalization. It was classified as validation/reporting code in v0.5.3,
so ``measurement_harness_checksum`` did not change when fixtures.py did
(concretely reproduced: mutating its SEED constant left the checksum
byte-identical). Fixed by moving fixtures.py (and the package
initializer(s) Python actually executes on the ``-m
benchmarks.step5.worker`` import path) into the measurement set.

This module proves two independent things:
1. Mutation tests -- changing each file now claimed to be measurement-
   relevant actually changes measurement_harness_checksum, and changing
   validator-only code does NOT (but DOES change validation_harness_checksum).
2. An import-closure regression test -- so the SAME class of omission (a
   future local helper imported by measurement code but never added to the
   measurement set) fails a test immediately, rather than shipping unnoticed.
"""

import ast
import shutil
from pathlib import Path

import pytest

from benchmarks.step5.provenance import (
    _MEASUREMENT_HARNESS_RELATIVE_PATHS,
    _all_harness_candidate_paths,
    measurement_harness_checksum,
    validation_harness_checksum,
)

ROOT = Path(__file__).resolve().parent.parent


def _copy_harness_tree(tmp_path: Path) -> Path:
    """Copies just the parts of the repo the two harness checksums read
    (benchmarks/ and scripts/) into a scratch root, so a mutation test can
    edit a file without touching the real repository."""
    dest = tmp_path / "repo"
    shutil.copytree(ROOT / "benchmarks", dest / "benchmarks")
    if (ROOT / "scripts").exists():
        shutil.copytree(ROOT / "scripts", dest / "scripts")
    return dest


def _mutate(root: Path, rel_path: str, transform) -> None:
    p = root / rel_path
    p.write_text(transform(p.read_text(encoding="utf-8")), encoding="utf-8")


def _append_marker(text: str) -> str:
    return text + "\n# mutation-test marker -- content changed, meaning unchanged\n"


# ---------------------------------------------------------------------------
# Mutation tests: every file in the measurement set must actually move the
# measurement checksum when it changes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel_path", [
    "benchmarks/step5/fixtures.py",
    "benchmarks/step5/worker.py",
    "benchmarks/step5/run_system_benchmark.py",
    "benchmarks/step5/provenance.py",
])
def test_mutating_a_measurement_file_changes_measurement_checksum(tmp_path, rel_path):
    root = _copy_harness_tree(tmp_path)
    h0 = measurement_harness_checksum(root)
    _mutate(root, rel_path, _append_marker)
    h1 = measurement_harness_checksum(root)
    assert h1 != h0, f"mutating {rel_path} must change measurement_harness_checksum"


def test_mutating_package_initializer_changes_measurement_checksum_if_present(tmp_path):
    root = _copy_harness_tree(tmp_path)
    init_path = root / "benchmarks" / "step5" / "__init__.py"
    assert init_path.exists(), "benchmarks/step5/__init__.py is expected to exist in this repo"
    h0 = measurement_harness_checksum(root)
    _mutate(root, "benchmarks/step5/__init__.py", _append_marker)
    h1 = measurement_harness_checksum(root)
    assert h1 != h0, "mutating benchmarks/step5/__init__.py must change measurement_harness_checksum"


def test_mutating_fixtures_seed_specifically_changes_measurement_checksum(tmp_path):
    """The exact audit reproducer: changing only fixtures.py's SEED constant
    must change measurement_harness_checksum. Confirmed directly against the
    real v0.5.3 source, BEFORE this fix, that it did not (896054a4...
    unchanged before and after)."""
    root = _copy_harness_tree(tmp_path)
    h0 = measurement_harness_checksum(root)
    _mutate(root, "benchmarks/step5/fixtures.py", lambda t: t.replace("SEED = 20260829", "SEED = 20260830"))
    h1 = measurement_harness_checksum(root)
    assert h1 != h0, "mutating fixtures.py's SEED must change measurement_harness_checksum"


def test_mutating_validator_only_file_leaves_measurement_unchanged_but_changes_validation(tmp_path):
    root = _copy_harness_tree(tmp_path)
    m0 = measurement_harness_checksum(root)
    v0 = validation_harness_checksum(root)
    _mutate(root, "benchmarks/step5/gen_validation_results.py", _append_marker)
    m1 = measurement_harness_checksum(root)
    v1 = validation_harness_checksum(root)
    assert m1 == m0, "mutating gen_validation_results.py must NOT change measurement_harness_checksum"
    assert v1 != v0, "mutating gen_validation_results.py MUST change validation_harness_checksum"


def test_mutating_unrelated_reporting_module_leaves_measurement_unchanged_but_changes_validation(tmp_path):
    root = _copy_harness_tree(tmp_path)
    target = "benchmarks/step5/summarize.py"
    assert (root / target).exists()
    m0 = measurement_harness_checksum(root)
    v0 = validation_harness_checksum(root)
    _mutate(root, target, _append_marker)
    m1 = measurement_harness_checksum(root)
    v1 = validation_harness_checksum(root)
    assert m1 == m0, "mutating an unrelated reporting module must NOT change measurement_harness_checksum"
    assert v1 != v0, "mutating an unrelated reporting module MUST change validation_harness_checksum"


# ---------------------------------------------------------------------------
# Partition semantics: measurement and validation together cover exactly the
# candidate universe, with no overlap.
# ---------------------------------------------------------------------------
def test_measurement_and_validation_sets_partition_the_harness_universe():
    all_paths = {p.relative_to(ROOT).as_posix() for p in _all_harness_candidate_paths(ROOT)}
    measurement = all_paths & _MEASUREMENT_HARNESS_RELATIVE_PATHS
    validation = all_paths - _MEASUREMENT_HARNESS_RELATIVE_PATHS
    assert measurement | validation == all_paths
    assert not (measurement & validation)


def test_measurement_set_includes_parent_package_initializers():
    """Every containing package's __init__.py (if it exists on disk) must
    also be in the measurement set for any measurement file -- Python
    executes it on import regardless of whether it currently does anything."""
    for rel in _MEASUREMENT_HARNESS_RELATIVE_PATHS:
        parts = Path(rel).parts
        for i in range(1, len(parts)):
            pkg_init_rel = "/".join(parts[:i]) + "/__init__.py"
            if (ROOT / pkg_init_rel).exists():
                assert pkg_init_rel in _MEASUREMENT_HARNESS_RELATIVE_PATHS, (
                    f"{rel} is in the measurement set, so its parent package initializer "
                    f"{pkg_init_rel} (which Python executes on import) must be too"
                )


# ---------------------------------------------------------------------------
# Import-closure regression test: the actual recurrence guard. Statically
# parses (ast only -- never executes) every measurement file and verifies
# every local benchmarks.* module it transitively imports is ALSO in the
# measurement set. A future `from benchmarks.step5.new_helper import X`
# added to a measurement file, without adding new_helper.py to the set,
# fails this test.
# ---------------------------------------------------------------------------
def _module_name_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_module_path(dotted: str, root: Path) -> Path | None:
    parts = dotted.split(".")
    candidate = root.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    pkg_init = root.joinpath(*parts, "__init__.py")
    if pkg_init.exists():
        return pkg_init
    return None  # e.g. a namespace package with no __init__.py -- nothing to hash


def _local_benchmarks_imports(path: Path, root: Path) -> set[str]:
    """Dotted 'benchmarks.*' module names this file directly imports, via
    AST parsing only (absolute `import benchmarks.x.y` / `from
    benchmarks.x.y import z`, and relative `from . import y` / `from .y
    import z` forms, resolved against the file's own package)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    self_dotted = _module_name_for(path, root)
    package_parts = self_dotted.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("benchmarks."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = package_parts[: len(package_parts) - (node.level - 1)] if node.level > 1 else package_parts
                if node.module:
                    resolved = ".".join(base + node.module.split("."))
                    if resolved.startswith("benchmarks."):
                        found.add(resolved)
                else:
                    for alias in node.names:
                        sub = ".".join(base + [alias.name])
                        if sub.startswith("benchmarks."):
                            found.add(sub)
            elif node.module and node.module.startswith("benchmarks."):
                found.add(node.module)
    return found


def test_measurement_set_import_closure_is_complete():
    seen: set[str] = set()
    frontier: set[str] = set(_MEASUREMENT_HARNESS_RELATIVE_PATHS)
    missing: set[str] = set()
    while frontier:
        rel = frontier.pop()
        seen.add(rel)
        path = ROOT / rel
        if not path.exists():
            continue  # e.g. benchmarks/__init__.py, legitimately absent in this repo
        for dotted in _local_benchmarks_imports(path, ROOT):
            resolved = _resolve_module_path(dotted, ROOT)
            if resolved is None:
                continue
            resolved_rel = resolved.relative_to(ROOT).as_posix()
            if resolved_rel not in _MEASUREMENT_HARNESS_RELATIVE_PATHS:
                missing.add(resolved_rel)
            if resolved_rel not in seen:
                frontier.add(resolved_rel)
    assert not missing, (
        "the following local benchmarks module(s) are (transitively) imported by "
        f"measurement code but are NOT in _MEASUREMENT_HARNESS_RELATIVE_PATHS: {sorted(missing)}"
    )


def test_featm_imports_from_measurement_files_are_not_required_in_the_set():
    """featm.* imports are already protected by featm_source_checksum (all
    of src/featm/**/*.py) -- they must never need to appear inside
    _MEASUREMENT_HARNESS_RELATIVE_PATHS, which is local-benchmarks-module
    only. Sanity check: worker.py does import featm, and the closure test
    above correctly does not flag that as a missing local dependency."""
    worker = ROOT / "benchmarks" / "step5" / "worker.py"
    tree = ast.parse(worker.read_text(encoding="utf-8"))
    imports_featm = any(
        (isinstance(n, ast.Import) and any(a.name == "featm" or a.name.startswith("featm.") for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module and (n.module == "featm" or n.module.startswith("featm.")))
        for n in ast.walk(tree)
    )
    assert imports_featm, "expected worker.py to import featm (sanity check for this test's own premise)"
