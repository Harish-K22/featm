"""Transactional distributed checkpoint/resume (Step 4 completion + audit-fix passes).

Dask tasks may execute more than once (speculative re-execution, task
retries, worker restarts). FEATM cannot make task EXECUTION exactly-once,
so it makes each partition's COMMITTED CONTRIBUTION to fit state / transform
output exactly-once instead, keyed by a deterministic partition index that
is part of a versioned run identity.

Directory layout::

    <directory>/
      distributed_checkpoint.json   -- identity + committed partition indices + phase-done flags + checksums
      fit_state/part-000000.pkl     -- one raw per-partition fit-state contribution, pickled as
                                        {"states": {...}, "feature_errors": {...}} -- see Finding 1 below
      fit_finalized.pkl             -- the one-time finalized (post-merge) fitted states
      output/part-000000.parquet    -- one committed transform output shard (final column schema)

Commit protocol (both fit and transform): compute the partition's
contribution -> write it to a process-unique temp file -> ``os.replace`` to
the deterministic partition filename (atomic on POSIX and Windows) -> ONLY
THEN atomically update the manifest to record that partition index as
committed (with a content checksum). This ordering is the entire
correctness argument:

- A crash between "data file renamed into place" and "manifest updated"
  leaves an ORPHAN data file. Safe: the manifest -- not file presence -- is
  authoritative for "committed"; a resume ignores/overwrites the orphan.
- A crash during the manifest's own atomic write leaves the OLD manifest
  (still fully valid JSON, still correctly reflecting the last successful
  commit) in place, because ``os.replace`` never exposes a partially-written
  file.
- Re-committing the SAME partition index twice (a Dask retry that succeeds
  twice, or a resumed run recomputing something not yet marked committed)
  is idempotent: the second write just overwrites the first at the same
  deterministic filename, and the manifest's committed-index SET does not
  double-count a repeated index.
- A resume against an incompatible source/feature/params/partition-plan/
  mode/errors-policy request is rejected with a clear error (mirrors
  :class:`featm.checkpoint.CheckpointIncompatibleError`) rather than silently
  reusing shards computed against different row boundaries or failure
  semantics.
- Committed artifacts carry a content checksum recorded in the manifest at
  commit time. A resume verifies it before trusting the artifact -- a file
  that still opens/unpickles cleanly but whose *content* was replaced
  (tampered, or corrupted in a way that happens to remain structurally
  valid) is detected, not silently trusted (see Finding 5,
  STEP4_DASK_AUDIT_FIX_HANDOFF.md).

Finding 1 (independent audit, v0.4.0 -> v0.4.1): a fit partition whose
worker task reported ANY feature error must never become indistinguishable,
on resume, from a partition that fully succeeded. Two failure modes were
possible under the old "persist bare ``states``" design:

  - under ``errors="raise"``, the partition was committed (with only the
    features that happened to succeed) *before* the error was raised to the
    caller; on resume, the checkpoint looked like a normal completed
    partition (``feature_errors`` defaults to ``{}`` when reconstructed) and
    the original failure silently vanished, along with the fact that the
    failed feature's contribution for that partition was ever missing;
  - under ``errors="skip"``/``"warn"``, the SAME partial-commit-with-no-error-
    record problem meant a resumed run could not tell that a given partition
    had contributed nothing to a feature that ultimately (correctly) got
    marked failed for the whole run -- an uninterrupted run and a
    resumed run could silently disagree about *why* a feature failed, or
    the resumed run's merge could simply omit a partition's contribution to
    a feature without ever recording that as an error.

Both are fixed by persisting ``feature_errors`` alongside ``states`` for
every committed fit partition, and by never committing a partition at all
when ``errors="raise"`` and it produced any feature error (forcing a
resumed run to re-fit -- and, deterministically, re-raise -- exactly that
partition, rather than resume "successfully" from an incomplete state).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..checkpoint import CheckpointIncompatibleError, _atomic_write_json

DISTRIBUTED_CHECKPOINT_MANIFEST_VERSION = 5
MANIFEST_FILENAME = "distributed_checkpoint.json"


_HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path, chunk_size: int = _HASH_CHUNK_BYTES) -> str:
    """Incrementally hash a file on disk -- never materializes the whole file as one bytes object.

    Finding 4 (final closure audit, v0.4.1 -> v0.4.2): every checkpoint
    checksum in v0.4.1 was computed as ``_sha256_bytes(path.read_bytes())``
    -- for a large checkpoint artifact, that is a full-file in-memory copy
    purely to compute a hash. Reading and hashing in fixed-size chunks keeps
    peak memory for this operation bounded to ``chunk_size`` regardless of
    artifact size.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class DistributedCheckpointArtifactTooLargeError(RuntimeError):
    """Raised when a committed checkpoint artifact's ON-DISK size exceeds a configured limit.

    Finding 4 (final closure audit, v0.4.1 -> v0.4.2): the v0.4.1
    "preflight" (``DistributedCheckpointStateTooLargeError``) checked
    ``len(pickle.dumps(payload))`` -- but that line only runs AFTER
    ``pickle.dumps`` has already allocated the full serialized bytes object,
    so it was never actually a preflight against the memory-duplication risk
    it claimed to guard; the risky allocation had already happened by the
    time the check ran. Fit-state commits now stream directly to a temp file
    (``pickle.dump(payload, f)``, no bytes object ever held in full), which
    eliminates the in-memory duplication this was meant to prevent. What
    remains here is an honest, differently-scoped policy: an upper bound on
    the CHECKPOINT ARTIFACT'S SIZE ON DISK, checked after it is written (via
    ``os.path.getsize`` on the temp file, before it is renamed into place or
    counted as committed) -- a disk-usage guard, not an in-memory preflight.
    """


def _enforce_artifact_size_limit(tmp_path: Path, index: int | str, budget_bytes: int | None) -> None:
    if budget_bytes is None:
        return
    size = tmp_path.stat().st_size
    if size > 2 * budget_bytes:
        tmp_path.unlink(missing_ok=True)
        raise DistributedCheckpointArtifactTooLargeError(
            f"Checkpoint artifact for partition {index} is {size:,} bytes on disk, more than double the configured "
            f"state memory budget ({budget_bytes:,} bytes) used as this checkpoint's disk-size guard. Increase "
            "memory_budget, reduce the corpus/vocabulary cardinality driving this state, or disable checkpointing "
            "for this run. The oversized temp file was not committed."
        )


@dataclass(frozen=True)
class DistributedCheckpointIdentity:
    """Everything that must match between a distributed checkpoint and the run resuming it.

    ``partition_rows`` is included deliberately: it is the row-boundary
    contract every committed shard's identity depends on. A resume that
    requests a different partition size must not silently reuse shards
    computed against different row boundaries -- it must be rejected, the
    same way Step 2's single-machine checkpoint rejects a changed
    ``batch_rows``.

    ``mode`` and ``errors`` are included for the same reason Step 2's local
    :class:`featm.checkpoint.CheckpointIdentity` includes them in its
    ``settings``: both can change the numerical or failure semantics of a
    run. Resuming an ``errors="skip"`` checkpoint as if it were
    ``errors="raise"`` (or vice versa) would let a resumed run silently
    reuse committed work whose failure-handling semantics do not match what
    the caller is asking for now (Finding 6, independent audit).

    ``columns`` uses :func:`featm.sources.canonical_label` (not bare
    ``str()``) for the same reason Step 2's checkpoint does: ``str(0) ==
    str("0")`` but they are distinct, valid pandas column labels. A bare
    ``str()`` identity would let a run mapped to the integer-labeled column
    ``0`` resume against a checkpoint computed from the string-labeled
    column ``"0"`` as if they were the same input (Finding 6).

    ``state_budget_bytes`` (final checkpoint-budget-identity closure pass,
    v0.4.2 -> v0.4.3) is the NORMALIZED, DERIVED value actually handed to
    every stateful encoder --
    ``derive_state_budget_bytes(prepared.budget.memory_bytes)`` -- not the
    raw ``memory_budget=`` text the caller passed in. For an exact stateful
    feature (category-frequency, TF-IDF/BM25/surprisal vocabulary state,
    ...), this budget is not a performance hint: it directly decides whether
    fitting succeeds or raises ``StateBudgetExceededError``, which under
    ``errors="skip"``/``"warn"`` changes the actual selected feature set and
    result shape. Two ``memory_budget=`` inputs that normalize to the SAME
    derived value (and the same ``partition_rows``) remain compatible --
    only the derived value is compared, never the raw text. ``None`` when no
    stateful feature is selected for this run (an unrelated memory_budget
    change must not invalidate a checkpoint whose result never depended on
    the state budget in the first place).
    """

    featm_version: str
    source_fingerprint: str
    schema_fingerprint: str
    features: list[str]
    params: dict[str, dict[str, Any]]
    partition_rows: int
    mode: str
    errors: str
    columns: dict[str, list[str]]
    state_budget_bytes: int | None
    state_budget_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": DISTRIBUTED_CHECKPOINT_MANIFEST_VERSION,
            "featm_version": self.featm_version,
            "source_fingerprint": self.source_fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "features": list(self.features),
            "params": self.params,
            "partition_rows": self.partition_rows,
            "mode": self.mode,
            "errors": self.errors,
            "columns": self.columns,
            "state_budget_bytes": self.state_budget_bytes,
            "state_budget_policy": self.state_budget_policy,
        }


def _stream_pickle_to_temp(path: Path, obj: Any) -> Path:
    """Write ``obj`` directly to a process-unique temp file via ``pickle.dump`` -- never
    via ``pickle.dumps`` -- so no full-size in-memory serialized copy is ever held.

    Returns the temp path, NOT yet renamed into place -- callers hash/size-check it first
    (see ``_sha256_file``/``_enforce_artifact_size_limit``), then ``os.replace`` it themselves,
    keeping the existing "write temp -> verify -> atomic rename -> manifest commit" protocol
    intact (Finding 4, final closure audit).
    """
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tmp


class DistributedCheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fit_state_dir = self.directory / "fit_state"
        self.output_dir = self.directory / "output"
        self.manifest_path = self.directory / MANIFEST_FILENAME

    def load(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise CheckpointIncompatibleError(
                f"Existing distributed checkpoint manifest at {self.manifest_path} is invalid: "
                "its JSON root must be an object. Use a fresh checkpoint directory."
            )
        return manifest

    def _check_compatible(self, existing: dict, identity: DistributedCheckpointIdentity) -> None:
        wanted = identity.to_dict()
        # manifest_version is now part of the comparison (final closure pass,
        # v0.4.2 -> v0.4.3): a checkpoint written under an older manifest
        # schema (e.g. v2, before state_budget_bytes existed in identity) is
        # rejected unambiguously rather than silently compared field-by-field
        # against a manifest missing a key the current schema requires --
        # `existing.get("state_budget_bytes")` on a v2 manifest would return
        # `None` via a bare dict .get(), which could coincidentally equal a
        # stateless run's real `None` and mask exactly the kind of stale
        # semantic reuse this field exists to catch.
        mismatches = [
            f"{key}: checkpoint has {existing.get(key)!r}, requested run has {want!r}"
            for key, want in wanted.items()
            if existing.get(key) != want
        ]
        if mismatches:
            raise CheckpointIncompatibleError(
                f"Distributed checkpoint at {self.directory} is not compatible with this run:\n  "
                + "\n  ".join(mismatches)
                + "\nUse a fresh checkpoint directory, or match the source/features/params/partition_rows/"
                "mode/errors/columns/state_budget_bytes exactly."
            )

    def initialize_or_resume(self, identity: DistributedCheckpointIdentity) -> dict[str, Any]:
        existing = self.load()
        if existing is not None:
            self._check_compatible(existing, identity)
            return existing
        self.fit_state_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = identity.to_dict()
        manifest.update(
            {
                "fit_committed_partitions": [],
                "fit_partition_checksums": {},
                "fit_finalized_checksum": None,
                "fit_failed_features": {},
                "transform_committed_partitions": [],
                "transform_partition_checksums": {},
                "transform_partition_row_counts": {},
                "fit_done": False,
                "transform_done": False,
            }
        )
        _atomic_write_json(self.manifest_path, manifest)
        return manifest

    # --- fit phase --------------------------------------------------------
    def fit_committed_indices(self) -> set[int]:
        manifest = self.load()
        return set(manifest.get("fit_committed_partitions", [])) if manifest else set()

    def commit_fit_partition(
        self,
        index: int,
        states: dict[str, Any],
        *,
        feature_errors: Mapping[str, str] | None = None,
        state_budget_bytes: int | None = None,
    ) -> None:
        """Commit ONE partition's raw fit-state contribution (all stateful features together).

        Persists ``feature_errors`` alongside ``states`` (Finding 1): a
        partition that only PARTIALLY succeeded (some stateful features ok,
        one failed, under ``errors!="raise"``) is committed together with a
        durable record of exactly which feature(s) failed and why, so a
        resumed run reconstructs the identical partial result -- not a
        partition that silently looks fully successful. Callers must not
        call this for a partition under ``errors="raise"`` policy when it
        produced any feature error; see the module docstring's Finding 1
        note and ``distributed_extract``'s ``_commit_as_arrives``.
        """
        payload = {"states": states, "feature_errors": dict(feature_errors or {})}
        path = self.fit_state_dir / f"part-{index:06d}.pkl"
        tmp = _stream_pickle_to_temp(path, payload)
        _enforce_artifact_size_limit(tmp, index, state_budget_bytes)
        checksum = _sha256_file(tmp)
        os.replace(tmp, path)
        manifest = self.load()
        committed = set(manifest.get("fit_committed_partitions", []))
        committed.add(index)
        manifest["fit_committed_partitions"] = sorted(committed)
        manifest.setdefault("fit_partition_checksums", {})[str(index)] = checksum
        _atomic_write_json(self.manifest_path, manifest)

    def load_fit_partition(self, index: int) -> dict[str, Any]:
        """Returns ``{"states": {...}, "feature_errors": {...}}`` for one committed partition."""
        with open(self.fit_state_dir / f"part-{index:06d}.pkl", "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict) and "states" in raw and "feature_errors" in raw:
            return raw
        # Backward compatibility with a v1 (pre-audit-fix) checkpoint that
        # persisted bare states with no feature_errors record at all.
        return {"states": raw, "feature_errors": {}}

    def commit_fit_finalized(self, fitted_states: dict[str, Any], *, failed_features: Mapping[str, str] | None = None) -> None:
        """Persist the ONE-TIME finalized (post-merge, post-``finalize_state``) fitted states.

        Committed once, atomically, after the fit phase's merge completes --
        so a resume that finds ``fit_done()`` true never needs to re-merge
        raw per-partition states at all, just load this single blob.

        ``failed_features`` (feature name -> message) is stored in the
        manifest so a fast-path resume (see ``fit_failed_features``) can
        correctly repopulate which stateful features were excluded from the
        run's output, instead of silently treating every requested stateful
        feature as successful merely because ``fit_done()`` is true.
        """
        path = self.directory / "fit_finalized.pkl"
        tmp = _stream_pickle_to_temp(path, fitted_states)
        checksum = _sha256_file(tmp)
        os.replace(tmp, path)
        manifest = self.load()
        manifest["fit_finalized_checksum"] = checksum
        manifest["fit_failed_features"] = dict(failed_features or {})
        _atomic_write_json(self.manifest_path, manifest)

    def load_fit_finalized(self) -> dict[str, Any]:
        with open(self.directory / "fit_finalized.pkl", "rb") as f:
            return pickle.load(f)

    def fit_failed_features(self) -> dict[str, str]:
        manifest = self.load()
        return dict(manifest.get("fit_failed_features", {})) if manifest else {}

    def mark_fit_done(self) -> None:
        manifest = self.load()
        manifest["fit_done"] = True
        _atomic_write_json(self.manifest_path, manifest)

    def fit_done(self) -> bool:
        manifest = self.load()
        return bool(manifest and manifest.get("fit_done"))

    # --- transform phase ----------------------------------------------------
    def transform_committed_indices(self) -> set[int]:
        manifest = self.load()
        return set(manifest.get("transform_committed_partitions", [])) if manifest else set()

    def commit_transform_partition(self, index: int, frame: Any, *, feature_errors=None) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.output_dir / f"part-{index:06d}.parquet"
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp)
        # Hash the temp file (streamed, fixed-size chunks -- Finding 4) before
        # the atomic rename, so the checksum verifiably reflects exactly the
        # bytes that get published, without ever holding the whole file in memory.
        checksum = _sha256_file(tmp)
        os.replace(tmp, path)
        manifest = self.load()
        committed = set(manifest.get("transform_committed_partitions", []))
        committed.add(index)
        manifest["transform_committed_partitions"] = sorted(committed)
        manifest.setdefault("transform_partition_checksums", {})[str(index)] = checksum
        manifest.setdefault("transform_partition_row_counts", {})[str(index)] = int(len(frame))
        manifest.setdefault("transform_partition_errors", {})[str(index)] = dict(feature_errors or {})
        _atomic_write_json(self.manifest_path, manifest)

    def transform_partition_errors(self, index):
        return dict((self.load() or {}).get("transform_partition_errors", {}).get(str(index), {}))

    def load_transform_partition(self, index: int):
        import pandas as pd

        return pd.read_parquet(self.output_dir / f"part-{index:06d}.parquet")

    def commit_primed_schema(
        self, primed: dict[str, list[str]], output_columns: list[str], *, failed_features: Mapping[str, str] | None = None,
    ) -> None:
        """Persist the per-feature output schema established by the priming (logical partition 0) transform.

        Mirrors ``featm.execution``'s local out-of-core checkpoint, which
        stores the same ``primed_columns``/``ordered_output_columns`` in its
        own manifest -- needed so a resume whose priming partition was
        already committed in a prior run does not need to recompute
        partition 0 just to rediscover the schema.

        ``failed_features`` (Finding 2 follow-on, final closure audit)
        persists WHICH features failed during priming and why, so a resumed
        run that short-circuits via the "checkpoint already recorded zero
        surviving output columns" path (Case B) can still restore an
        identical ``report.failed_features`` -- not just an identical empty
        result with a report that silently forgot why.
        """
        manifest = self.load()
        manifest["primed_columns"] = primed
        manifest["ordered_output_columns"] = output_columns
        manifest["transform_priming_failed_features"] = dict(failed_features or {})
        _atomic_write_json(self.manifest_path, manifest)

    def load_primed_schema(self) -> tuple[dict[str, list[str]], list[str]] | None:
        manifest = self.load()
        if not manifest or "primed_columns" not in manifest:
            return None
        return manifest["primed_columns"], manifest["ordered_output_columns"]

    def transform_priming_failed_features(self) -> dict[str, str]:
        manifest = self.load()
        return dict(manifest.get("transform_priming_failed_features", {})) if manifest else {}

    def mark_transform_done(self) -> None:
        manifest = self.load()
        manifest["transform_done"] = True
        _atomic_write_json(self.manifest_path, manifest)

    def transform_done(self) -> bool:
        manifest = self.load()
        return bool(manifest and manifest.get("transform_done"))

    def validate_committed_shards_integrity(self) -> list[str]:
        """Detect a manifest that claims a partition is committed but whose file is missing, corrupt, or tampered.

        "Opens cleanly" (unpickles / is a structurally valid Parquet file) is
        NOT sufficient proof of integrity -- a valid-but-different pickle or
        Parquet file would pass that check while silently returning wrong
        data. Every committed artifact's content is checksummed at commit
        time (SHA-256, recorded in the manifest); this method re-hashes the
        artifact currently on disk and compares. A checksum recorded by a
        pre-audit-fix (v1) checkpoint will be absent -- treated as a problem
        too, since there is nothing to verify against (Finding 5).
        """
        # Every read here is streamed (Finding 4): structural validity is
        # checked with a streaming pickle.load / a metadata-only ParquetFile
        # open, and the checksum is computed by a separate streaming pass
        # (_sha256_file) -- no artifact is ever materialized as one whole
        # in-memory bytes object merely to validate or hash it, regardless
        # of file size.
        problems: list[str] = []
        manifest = self.load() or {}

        fit_checksums = manifest.get("fit_partition_checksums", {})
        for idx in manifest.get("fit_committed_partitions", []):
            path = self.fit_state_dir / f"part-{idx:06d}.pkl"
            if not path.exists():
                problems.append(f"fit partition {idx} committed in manifest but state file missing: {path}")
                continue
            try:
                with open(path, "rb") as f:
                    pickle.load(f)
            except Exception as exc:
                problems.append(f"fit partition {idx} state file corrupt: {type(exc).__name__}: {exc}")
                continue
            expected = fit_checksums.get(str(idx))
            if expected is None:
                problems.append(f"fit partition {idx} committed but has no recorded checksum (incompatible/old checkpoint)")
            elif _sha256_file(path) != expected:
                problems.append(f"fit partition {idx} content checksum mismatch (file was modified after commit): {path}")

        if manifest.get("fit_done"):
            finalized_path = self.directory / "fit_finalized.pkl"
            if not finalized_path.exists():
                problems.append("fit marked done but fit_finalized.pkl is missing")
            else:
                try:
                    with open(finalized_path, "rb") as f:
                        pickle.load(f)
                except Exception as exc:
                    problems.append(f"fit_finalized.pkl corrupt: {type(exc).__name__}: {exc}")
                else:
                    expected = manifest.get("fit_finalized_checksum")
                    if expected is None:
                        problems.append("fit marked done but fit_finalized.pkl has no recorded checksum (incompatible/old checkpoint)")
                    elif _sha256_file(finalized_path) != expected:
                        problems.append("fit_finalized.pkl content checksum mismatch (file was modified after commit)")

        transform_checksums = manifest.get("transform_partition_checksums", {})
        transform_row_counts = manifest.get("transform_partition_row_counts", {})
        for idx in manifest.get("transform_committed_partitions", []):
            path = self.output_dir / f"part-{idx:06d}.parquet"
            if not path.exists():
                problems.append(f"transform partition {idx} committed in manifest but shard file missing: {path}")
                continue
            try:
                import pyarrow.parquet as pq

                actual_rows = pq.ParquetFile(str(path)).metadata.num_rows
            except Exception as exc:
                problems.append(f"transform partition {idx} shard file corrupt: {type(exc).__name__}: {exc}")
                continue
            expected_checksum = transform_checksums.get(str(idx))
            if expected_checksum is None:
                problems.append(f"transform partition {idx} committed but has no recorded checksum (incompatible/old checkpoint)")
            elif _sha256_file(path) != expected_checksum:
                problems.append(f"transform partition {idx} content checksum mismatch (file was modified after commit): {path}")
            expected_rows = transform_row_counts.get(str(idx))
            if expected_rows is not None and actual_rows != expected_rows:
                problems.append(
                    f"transform partition {idx} row count mismatch: manifest recorded {expected_rows}, file has {actual_rows}"
                )
        return problems
