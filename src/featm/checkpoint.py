"""Checkpoint / resume for long out-of-core extraction runs.

A checkpoint directory holds one small JSON manifest (written atomically —
temp file + ``os.replace``, so a crash mid-write cannot leave a corrupt
manifest that looks valid) plus any fitted-state artifacts and the partial
disk-backed output written so far.

Granularity: checkpointing is at the *batch* level for one execution plan
(the whole set of selected features run together), not at the sub-batch
level and not independently per feature. A run has up to two phases:

- ``fit``   -- only present if at least one selected feature is stateful;
              scans the source once, merging each stateful feature's state,
              checkpointing the running accumulator after every batch so an
              interrupted fit resumes mid-pass rather than from scratch.
- ``transform`` -- scans the source (again, if there was a fit phase) computing
              every selected feature per batch and writing one output shard
              per batch, checkpointing after each shard.

On resume, already-completed batches for the active phase are skipped by
re-iterating the source and discarding them (cheap: no feature computation
runs against a skipped batch) rather than recomputing anything. Compatibility
is verified against a fingerprint of the source, schema, selected features
and parameters, and the relevant execution settings before any work is
reused; an incompatible checkpoint is rejected with a clear explanation
rather than silently reused.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_MANIFEST_VERSION = 3
MANIFEST_FILENAME = "checkpoint.json"


class CheckpointIncompatibleError(RuntimeError):
    """Raised when an existing checkpoint does not match the requested run."""


@dataclass
class CheckpointIdentity:
    """Everything that must match between a checkpoint and the run resuming it."""

    featm_version: str
    source_fingerprint: str
    schema_fingerprint: str
    features: list[str]
    params: dict[str, dict[str, Any]]
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "featm_version": self.featm_version,
            "source_fingerprint": self.source_fingerprint,
            "schema_fingerprint": self.schema_fingerprint,
            "features": list(self.features),
            "params": self.params,
            "settings": self.settings,
        }


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


class CheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.directory / "output"
        self.state_dir = self.directory / "state"
        self.manifest_path = self.directory / MANIFEST_FILENAME

    # --- manifest lifecycle -------------------------------------------------
    def load(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise CheckpointIncompatibleError(
                f"Existing checkpoint manifest at {self.manifest_path} is invalid: "
                "its JSON root must be an object. Use a fresh checkpoint directory."
            )
        return manifest

    def _check_compatible(self, existing: dict[str, Any], identity: CheckpointIdentity) -> None:
        # manifest_version is compared explicitly (local checkpoint
        # replay-history closure pass, v0.4.3 -> v0.4.4): a checkpoint
        # written under an older manifest schema (v1, before failure/
        # degradation history was persisted at all) must be rejected as
        # incompatible on resume, not silently reused -- an old manifest
        # simply has no `fit_failed_features`/`transform_priming_failed_
        # features`/`degraded_batches_by_feature` keys to restore, which
        # would silently reintroduce the exact history-loss bug this pass
        # fixes for any pre-existing checkpoint. A versioned package string
        # (`featm_version`) is not a substitute for this -- the schema, not
        # just the code, changed.
        if existing.get("manifest_version") != CHECKPOINT_MANIFEST_VERSION:
            raise CheckpointIncompatibleError(
                f"Checkpoint at {self.directory} was written under manifest schema version "
                f"{existing.get('manifest_version')!r}, but this version of featm requires "
                f"{CHECKPOINT_MANIFEST_VERSION!r} (older checkpoints do not persist enough failure/degradation "
                "history to resume with a replay-equivalent report). Use a fresh checkpoint directory."
            )
        wanted = identity.to_dict()
        mismatches = []
        for key, want in wanted.items():
            got = existing.get(key)
            if got != want:
                mismatches.append(f"{key}: checkpoint has {got!r}, requested run has {want!r}")
        if mismatches:
            raise CheckpointIncompatibleError(
                f"Checkpoint at {self.directory} is not compatible with this run:\n  "
                + "\n  ".join(mismatches)
                + "\nUse a fresh checkpoint directory, or match the source/features/params/settings exactly."
            )

    def initialize_or_resume(self, identity: CheckpointIdentity, *, has_fit_phase: bool) -> dict[str, Any]:
        """Load a compatible existing manifest, or create a fresh one.

        Raises :class:`CheckpointIncompatibleError` if a checkpoint exists at
        this directory but does not match ``identity``.
        """
        existing = self.load()
        if existing is not None:
            self._check_compatible(existing, identity)
            return existing
        manifest = identity.to_dict()
        manifest.update(
            {
                "manifest_version": CHECKPOINT_MANIFEST_VERSION,
                "phase": "fit" if has_fit_phase else "transform",
                "fit_batches_completed": 0,
                "fit_accumulator_file": None,
                "fit_state_files": {},
                # Local checkpoint replay-history closure pass (v0.4.3 ->
                # v0.4.4): durable failure/degradation metadata, so a resumed
                # (partial or zero-read "done") run's report is
                # replay-equivalent to what an uninterrupted run would have
                # reported -- not silently clean merely because the feature
                # that failed left no successfully-fitted-state file behind
                # to key a restore off of. Each `..._failed_features` value
                # is ``{"kind": "failed" | "optional_dependency", "message": str}``
                # so a restore can call the matching ExecutionReport method.
                "fit_failed_features": {},
                "transform_priming_failed_features": {},
                "degraded_batches_by_feature": {},
                "transform_batches_completed": 0,
                "transform_shards_written": 0,
                "done": False,
            }
        )
        self.save(manifest)
        return manifest

    def save(self, manifest: dict[str, Any]) -> None:
        _atomic_write_json(self.manifest_path, manifest)

    # --- fitted state artifacts ----------------------------------------------
    def save_fit_state(self, feature_name: str, state: Any) -> str:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_filename(feature_name)}.pkl"
        path = self.state_dir / filename
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh)
        os.replace(tmp, path)
        return filename

    def load_fit_state(self, filename: str) -> Any:
        with open(self.state_dir / filename, "rb") as fh:
            return pickle.load(fh)

    def save_fit_accumulators(self, accumulators: dict[str, Any], batch_index: int) -> str:
        """Persist in-progress fit accumulators as a *versioned* snapshot, named by batch index.

        This does NOT update the manifest -- the caller must separately call
        :meth:`save` with the manifest referencing this filename once it wants
        to commit to this snapshot as the new resume point. Two files being
        each individually atomic (this one, and the manifest) does not make
        writing both together atomic: a crash between the two would otherwise
        leave the manifest pointing at (and reprocessing) a batch whose
        accumulator snapshot has already moved past it, double-counting that
        batch's contribution to the fitted state. Versioning by batch index
        means an orphaned snapshot from a crash before the manifest commit is
        simply never referenced and is harmless (cleaned up opportunistically
        by :meth:`cleanup_orphan_accumulator_snapshots`).
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        filename = f"_fit_accumulators-{batch_index:010d}.pkl"
        path = self.state_dir / filename
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            pickle.dump(accumulators, fh)
        os.replace(tmp, path)
        return filename

    def load_fit_accumulators(self, filename: str) -> dict[str, Any] | None:
        """Load exactly the snapshot the manifest committed to -- never "the latest on disk"."""
        path = self.state_dir / filename
        if not path.exists():
            from .matrix import CheckpointCorruptionError
            raise CheckpointCorruptionError(f"Committed fit snapshot is missing: {path}")
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            from .matrix import CheckpointCorruptionError
            raise CheckpointCorruptionError(f"Committed fit snapshot is unreadable: {path}") from exc

    def cleanup_orphan_accumulator_snapshots(self, keep_filename: str | None) -> None:
        """Remove old/orphaned accumulator snapshots, keeping only the manifest-committed one."""
        if not self.state_dir.exists():
            return
        for p in self.state_dir.glob("_fit_accumulators-*.pkl"):
            if keep_filename is None or p.name != keep_filename:
                try:
                    p.unlink()
                except OSError:
                    pass
