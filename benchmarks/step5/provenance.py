from __future__ import annotations

"""Shared run/environment provenance for Step-5 benchmark orchestrators.

Audit finding (v0.5.0 -> v0.5.1): the previously-shipped final CPU evidence
mixed rows from different code snapshots under one manifest with no per-row
way to tell them apart, and the separately-saved environment manifest's
checksum did not match the actually-delivered source. Every raw result row
produced by any Step-5 orchestrator now carries its own run_id/environment_id/
featm_source_checksum/featm_version, computed once per invocation -- never
inherited from a stale file, never silently mismatched.

Audit finding (v0.5.1 -> v0.5.2), two more:

1. **Cross-platform checksum instability.** ``str(Path)`` uses the host OS's
   path separator, so the identical source tree hashed differently on
   Windows (backslash paths) vs. Linux/macOS (forward-slash paths) --
   confirmed directly: the same files gave ``0a07536b...`` on Windows and
   ``c4f6944c...`` on Linux. Fixed by hashing ``Path.as_posix()`` (always
   forward-slash, platform-independent) with explicit null-byte delimiters
   between the path and content and between records, so path/content
   framing is unambiguous. ``SOURCE_CHECKSUM_ALGORITHM`` names this
   explicitly (``"blake2b16-posix-path-v2"``) -- a checksum computed under a
   different algorithm is never silently compared against one computed
   under this one.
2. **Product code vs. benchmark harness code were conflated.** The old
   single ``source_checksum`` covered only ``src/featm/**/*.py`` -- the
   benchmark fixture/timing/evaluation code that actually PRODUCED a result
   row (``benchmarks/step5/**/*.py``, ``scripts/build_step5_*.py``) could
   change after a run with the product-code checksum staying identical,
   and the provenance gate would not notice. A single combined
   ``benchmark_harness_checksum`` was introduced to close this gap.

Audit finding (v0.5.2 -> v0.5.3): a THIRD independent audit found that the
single combined ``benchmark_harness_checksum`` from v0.5.2 covered every
file under ``benchmarks/step5/**`` -- including ``gen_validation_results.py``
itself, which only judges evidence and cannot affect what a CPU measurement
actually produced. This meant editing the *validator* (like this very pass)
would force an otherwise-unnecessary CPU rerun just to keep the harness
checksum consistent. Split into two identities:

- ``measurement_harness_checksum`` -- covers exactly the files whose content
  can affect what ``run_system_benchmark.py`` measures for the CPU sweep
  (``worker.py``, ``run_system_benchmark.py``, ``provenance.py`` itself,
  since it computes the identity stamped onto every row). This is the
  identity carried on every CPU row/manifest and foreign-key validated by
  ``_gate_provenance()``.
- ``validation_harness_checksum`` -- covers everything else under
  ``benchmarks/step5/**/*.py`` plus ``scripts/build_step5_*.py`` (i.e. the
  acceptance/validation/reporting code, including
  ``gen_validation_results.py`` itself). Computed fresh every time the
  validator runs and stamped into ``step5_validation_results.json`` for
  traceability ("this acceptance verdict was computed by validator code
  with checksum X") -- never foreign-key checked against a prior stored
  value, because there is no manifest recording what validator code
  *should* have produced a given CPU row; the validator is always current
  code judging possibly-older evidence, by design.

Together the two checksums partition exactly the same file set the old
single ``benchmark_harness_checksum`` covered -- nothing silently drops out
of coverage, it is simply attributed to whichever identity actually governs
it.

Audit finding (v0.5.3 -> v0.5.4): a FOURTH independent audit found that the
v0.5.3 ``_MEASUREMENT_HARNESS_RELATIVE_PATHS`` set was itself incomplete.
``worker.py`` does ``from benchmarks.step5.fixtures import build_call,
normalize_output`` -- ``fixtures.py`` controls the deterministic seed,
every generated benchmark input's shape/values, scale behavior, dummy-model
substitution, and output normalization. Concretely reproduced: changing
``fixtures.py``'s ``SEED`` constant left ``measurement_harness_checksum``
completely unchanged (``896054a4...`` before and after), even though that
change can alter latency, RSS, ``input_elements``, output shape, the
determinism digest, and status for every row in the CPU sweep. ``fixtures.py``
had been classified as validation/reporting code -- wrong, since it is a
direct, load-bearing import of the measurement path (``python -m
benchmarks.step5.worker``), not something that merely judges evidence
afterward.

Fixed two ways:

1. ``fixtures.py`` and the package initializer(s) Python actually executes
   on the ``-m benchmarks.step5.worker`` import path
   (``benchmarks/step5/__init__.py``; ``benchmarks/__init__.py`` too, IF it
   exists -- it currently does not, and a nonexistent listed file is simply
   never discovered by ``_all_harness_candidate_paths()``, never silently
   "passed" by a missing-file special case) were added to
   ``_MEASUREMENT_HARNESS_RELATIVE_PATHS``.
2. To prevent the same class of omission recurring for some *future* local
   helper (e.g. a hypothetical ``from benchmarks.step5.new_fixture_helper
   import ...``), ``tests/test_step5_measurement_harness_provenance.py``
   statically parses (via ``ast``, never by executing the modules) the
   import closure of every file in ``_MEASUREMENT_HARNESS_RELATIVE_PATHS``
   and asserts every local ``benchmarks.*`` module it (transitively)
   imports is ALSO in that set. A future edit that adds an unlisted local
   import will fail this test, not silently ship an incomplete checksum.
"""

import hashlib
import os
import platform
import time
from pathlib import Path

# Bump whenever the hashing method itself changes -- a checksum computed
# under a different algorithm must never be silently compared against one
# computed under this one. See gen_provenance_bridge.py for cross-version
# migration evidence.
SOURCE_CHECKSUM_ALGORITHM = "blake2b16-posix-path-v2"


def _canonical_tree_checksum(paths: list[Path], root: Path) -> str:
    """BLAKE2b-16 over a sorted set of files, using POSIX-style relative
    paths (platform-independent, unlike ``str(Path)``) and explicit
    null-byte delimiters so path/content/record framing is unambiguous."""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(paths, key=lambda x: x.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def source_checksum(root: Path) -> str:
    """Product-code identity: every src/featm/*.py file."""
    return _canonical_tree_checksum(list((root / "src" / "featm").rglob("*.py")), root)


# Files whose content can affect what a CPU measurement actually produces
# (timing, output shape, status). provenance.py is included because it
# computes the very identity stamped onto every row/manifest -- a change to
# how that identity is computed is itself a change to what a "measurement"
# means. fixtures.py is included because it is a direct import of worker.py
# (the deterministic seed, every generated input's shape/values, scale
# behavior, dummy-model substitution, and output normalization all live
# there -- see the v0.5.3->v0.5.4 module docstring section for the
# concretely-reproduced omission this fixes). The package initializer(s)
# Python actually executes on the `python -m benchmarks.step5.worker` import
# path are included on the same principle: a provenance identity should
# cover the real local Python execution dependency closure, not only files
# that happen to contain substantive code today.
#
# Relative, POSIX-style paths from the repo root -- matched exactly, never
# by basename alone, so a same-named file elsewhere in the tree is never
# accidentally swept in. A listed path that does not exist on disk (e.g.
# benchmarks/__init__.py, which this repo does not have -- it relies on
# implicit namespace packages) is simply never discovered by
# _all_harness_candidate_paths() below and contributes to neither checksum;
# there is no separate "does it exist" branch to get wrong.
#
# tests/test_step5_measurement_harness_provenance.py statically verifies
# (via ast, never by executing these modules) that every local
# benchmarks.* module transitively imported by anything in this set is
# ALSO in this set -- so a future local import added to any measurement
# file, without updating this set, fails that test rather than silently
# shipping an incomplete checksum.
_MEASUREMENT_HARNESS_RELATIVE_PATHS = frozenset({
    "benchmarks/__init__.py",
    "benchmarks/step5/__init__.py",
    "benchmarks/step5/run_system_benchmark.py",
    "benchmarks/step5/worker.py",
    "benchmarks/step5/fixtures.py",
    "benchmarks/step5/provenance.py",
})


def _all_harness_candidate_paths(root: Path) -> list[Path]:
    """Every .py file that is a candidate for EITHER harness identity --
    the measurement/validation split below partitions exactly this set.
    Includes benchmarks/__init__.py explicitly (it sits one level above
    the benchmarks/step5/ rglob root, so it would otherwise never be
    discovered at all) in addition to everything under benchmarks/step5/**
    and scripts/build_step5_*.py."""
    paths = list((root / "benchmarks" / "step5").rglob("*.py"))
    benchmarks_init = root / "benchmarks" / "__init__.py"
    if benchmarks_init.exists():
        paths.append(benchmarks_init)
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        paths.extend(scripts_dir.glob("build_step5_*.py"))
    return paths


def measurement_harness_checksum(root: Path) -> str:
    """Identity of exactly the code that can affect a CPU measurement's
    result (see _MEASUREMENT_HARNESS_RELATIVE_PATHS). This is the identity
    carried on every CPU row/manifest and foreign-key validated by
    _gate_provenance() -- editing validator-only code never changes it."""
    paths = [p for p in _all_harness_candidate_paths(root)
             if p.relative_to(root).as_posix() in _MEASUREMENT_HARNESS_RELATIVE_PATHS]
    return _canonical_tree_checksum(paths, root)


def validation_harness_checksum(root: Path) -> str:
    """Identity of the acceptance/validation/reporting code (everything
    under benchmarks/step5/**/*.py and scripts/build_step5_*.py NOT already
    covered by measurement_harness_checksum) -- e.g. gen_validation_results.py
    itself. Computed fresh at validation time and stamped into
    step5_validation_results.json for traceability only; never foreign-key
    checked against a prior stored value, since the validator is always
    current code judging possibly-older evidence by design."""
    paths = [p for p in _all_harness_candidate_paths(root)
             if p.relative_to(root).as_posix() not in _MEASUREMENT_HARNESS_RELATIVE_PATHS]
    return _canonical_tree_checksum(paths, root)


def make_provenance(root: Path, featm_version: str) -> dict:
    product_checksum = source_checksum(root)
    harness_checksum = measurement_harness_checksum(root)
    run_id = f"run-{platform.node()}-{int(time.time())}-{os.getpid()}"
    # Deterministic given (node, version, both checksums) -- NOT re-randomized
    # per invocation, so the same environment_id is produced whether it's
    # computed by a benchmark orchestrator or gen_environment_manifest.py,
    # as long as the environment is actually the same one.
    environment_id = f"env-{platform.node()}-{featm_version}-{product_checksum[:12]}-{harness_checksum[:12]}"
    return {
        "run_id": run_id, "environment_id": environment_id,
        "source_checksum_algorithm": SOURCE_CHECKSUM_ALGORITHM,
        "featm_source_checksum": product_checksum,
        "measurement_harness_checksum": harness_checksum,
        "featm_version": featm_version,
        # Back-compat alias for older readers/tests expecting this name;
        # always equal to featm_source_checksum, never a separate value.
        "source_checksum_blake2b16": product_checksum,
    }


def warn_if_mixed_checksums(rows: list[dict], current_checksum: str) -> None:
    prior = {r.get("featm_source_checksum") or r.get("source_checksum_blake2b16")
             for r in rows if r.get("featm_source_checksum") or r.get("source_checksum_blake2b16")}
    if prior and prior != {current_checksum}:
        print(f"WARNING: resuming into a file whose prior rows carry a DIFFERENT featm_source_checksum "
              f"({sorted(prior)}) than this run's ({current_checksum}). Prior rows are preserved with their own "
              f"original provenance (never silently relabeled), but this mixes evidence from more than one source "
              f"snapshot under one file -- pass --fresh for a single-snapshot rerun, or treat rows by their own "
              f"run_id/featm_source_checksum as the ground truth per row.")
