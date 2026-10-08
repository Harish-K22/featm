"""Execution report: what actually happened during one extraction run.

One selected feature failing should never silently change the output schema
without a record of it. ``ExecutionReport`` is the single place that records
which features completed/were skipped/failed, whether any ran in approximate
mode, retries/batch reductions caused by memory pressure, per-feature output
columns, and timing -- everything needed to explain the run after the fact.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionReport:
    requested_features: list[str] = field(default_factory=list)
    completed_features: list[str] = field(default_factory=list)
    skipped_features: list[str] = field(default_factory=list)
    failed_features: list[str] = field(default_factory=list)
    optional_dependency_failures: dict[str, str] = field(default_factory=dict)
    approximations_used: dict[str, dict[str, Any]] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    batch_reductions: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    output_columns: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    n_batches: int = 0
    batch_rows: int | None = None
    memory_budget_bytes: int | None = None
    disk_budget_bytes: int | None = None
    output_backend: str = "memory"  # "memory" | "disk"
    mode: str = "exact"  # exact | approximate | auto
    errors_policy: str = "skip"
    cache_stats: dict[str, int] = field(default_factory=dict)
    checkpoint_dir: str | None = None
    resumed: bool = False
    resumed_features: list[str] = field(default_factory=list)
    #: Step 3: per-feature GPU/CPU backend outcome, accumulated across
    #: however many batches this run scanned (see `record_backend_info`).
    #: Empty for a pure `backend="cpu"` run (the default) -- untouched, not
    #: populated-and-ignored, so a report from before Step 3 and a
    #: CPU-only Step 3 run look identical.
    backend_report: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Step 4: coordinator/runtime metadata for distributed runs. Empty for
    #: every ordinary local run so the pre-Step-4 report shape remains
    #: backwards-compatible for callers that do not opt in.
    distributed_report: dict[str, Any] = field(default_factory=dict)

    started_at: float = field(default_factory=time.time)
    total_elapsed_seconds: float = 0.0

    #: Running totals/peaks feeding `cache_stats`, accumulated across however
    #: many separate per-batch cache scopes this run has (fit phase + transform
    #: phase, or just one, or -- for the public featm.extractor API -- across
    #: two entirely separate calls sharing one ExecutionReport is NOT done;
    #: each of .fit()/.transform() gets its own report). Not part of to_dict()
    #: output directly; `cache_stats` (rebuilt by record_cache_stats) is.
    _cache_stats_totals: dict[str, int] = field(
        default_factory=lambda: {"hits": 0, "misses": 0, "evictions": 0, "oversized_skips": 0}, repr=False
    )
    _cache_stats_peak: dict[str, int] = field(default_factory=lambda: {"entries": 0, "bytes": 0}, repr=False)

    # --- recording helpers, used by the engine while it runs -----------------
    def _warn(self, message):
        if self.errors_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def record_completed(self, name: str, *, columns: list[str], elapsed: float) -> None:
        self.completed_features.append(name)
        self.output_columns[name] = columns
        self.timings[name] = self.timings.get(name, 0.0) + elapsed

    def record_skipped(self, name: str, reason: str) -> None:
        self.skipped_features.append(name)
        self.warnings.append(f"{name}: skipped ({reason})")
        self._warn(self.warnings[-1])

    def record_failed(self, name: str, error: str) -> None:
        self.failed_features.append(name)
        self.warnings.append(f"{name}: failed ({error})")
        self._warn(self.warnings[-1])

    def record_optional_dependency_failure(self, name: str, error: str) -> None:
        self.optional_dependency_failures[name] = error
        self.skipped_features.append(name)
        self.warnings.append(f"{name}: optional dependency unavailable ({error})")
        self._warn(self.warnings[-1])

    def record_approximate(self, name: str, params: dict[str, Any]) -> None:
        self.approximations_used[name] = dict(params)

    def record_retry(self, name: str) -> None:
        self.retries[name] = self.retries.get(name, 0) + 1

    def record_batch_reduction(self, name: str) -> None:
        self.batch_reductions[name] = self.batch_reductions.get(name, 0) + 1

    def record_backend_info(self, name: str, info: dict[str, Any]) -> None:
        """Fold in one batch's `featm.gpu.dispatch.BackendInfo.as_dict()` result for one feature.

        Accumulates across batches: `gpu_retries`/timings sum, `gpu_attempted`/
        `cpu_fallback` are True if true for ANY batch. `actual_backend`
        (audit-fix pass, Finding 11) is no longer just the most recent
        batch's value -- a common `backend="auto"` run legitimately runs some
        batches on GPU and a short final batch (below `gpu_min_rows`/
        `gpu_min_elements`) on CPU, and the pre-fix behavior could make a run
        that was overwhelmingly GPU-accelerated look, in the final report, as
        if it simply "used CPU" because that happened to be the LAST batch's
        outcome. `actual_backends_used`/`gpu_batches`/`cpu_batches`/
        `fallback_reasons_seen` are the full per-batch history;
        `actual_backend` becomes the literal string `"mixed"` once both
        `"cpu"` and `"gpu"` have been seen for this feature, so a reader
        cannot mistake a mixed run for a purely-CPU or purely-GPU one by only
        looking at that one field.
        """
        existing = self.backend_report.get(name)
        if existing is None:
            existing = dict(info)
            existing["actual_backends_used"] = [info.get("actual_backend")] if info.get("actual_backend") else []
            existing["gpu_batches"] = 1 if info.get("actual_backend") == "gpu" else 0
            existing["cpu_batches"] = 1 if info.get("actual_backend") == "cpu" else 0
            existing["fallback_reasons_seen"] = [info["fallback_reason"]] if info.get("fallback_reason") else []
            self.backend_report[name] = existing
            return
        existing["gpu_attempted"] = bool(existing.get("gpu_attempted")) or bool(info.get("gpu_attempted"))
        existing["gpu_retries"] = int(existing.get("gpu_retries", 0)) + int(info.get("gpu_retries", 0))
        existing["cpu_fallback"] = bool(existing.get("cpu_fallback")) or bool(info.get("cpu_fallback"))
        existing["gpu_time_seconds"] = (existing.get("gpu_time_seconds") or 0.0) + (info.get("gpu_time_seconds") or 0.0)
        existing["cpu_time_seconds"] = (existing.get("cpu_time_seconds") or 0.0) + (info.get("cpu_time_seconds") or 0.0)

        this_backend = info.get("actual_backend")
        existing["gpu_batches"] = int(existing.get("gpu_batches", 0)) + (1 if this_backend == "gpu" else 0)
        existing["cpu_batches"] = int(existing.get("cpu_batches", 0)) + (1 if this_backend == "cpu" else 0)
        used = list(existing.get("actual_backends_used") or [])
        if this_backend and this_backend not in used:
            used.append(this_backend)
        existing["actual_backends_used"] = used
        existing["actual_backend"] = "mixed" if len(used) > 1 else (used[0] if used else existing.get("actual_backend"))

        reasons = list(existing.get("fallback_reasons_seen") or [])
        if info.get("fallback_reason") and info["fallback_reason"] not in reasons:
            reasons.append(info["fallback_reason"])
        existing["fallback_reasons_seen"] = reasons
        if info.get("fallback_reason"):
            existing["fallback_reason"] = info["fallback_reason"]

        existing["requested_backend"] = info.get("requested_backend", existing.get("requested_backend"))
        existing["gpu_backend_name"] = info.get("gpu_backend_name") or existing.get("gpu_backend_name")
        existing["required_gpu_backend"] = info.get("required_gpu_backend") or existing.get("required_gpu_backend")
        # VRAM planning fields: keep the most recent batch's snapshot (free
        # VRAM at plan time is inherently a point-in-time reading, not
        # something that accumulates across batches).
        for k in ("gpu_total_memory", "gpu_free_memory_at_plan", "gpu_budget", "gpu_device_index", "gpu_planned_batch_rows"):
            if info.get(k) is not None:
                existing[k] = info[k]

    def record_cache_stats(self, stats: dict[str, int]) -> None:
        """Fold in one batch-scoped `featm.cache.SharedIntermediateCache.stats()` result.

        Accumulates hits/misses/evictions/oversized_skips as running totals
        across however many per-batch cache scopes this run has (fit phase,
        transform phase, ...), and entries/bytes as a running peak (the
        largest any single scope reached) -- matching the pre-refactor
        in-`run_extraction` closure this replaces, just as a report method so
        the fit and transform phases (now separate functions) can share it.
        """
        for k in self._cache_stats_totals:
            self._cache_stats_totals[k] += stats.get(k, 0)
        for k in self._cache_stats_peak:
            self._cache_stats_peak[k] = max(self._cache_stats_peak[k], stats.get(k, 0))
        self.cache_stats = {**self._cache_stats_totals, **self._cache_stats_peak}

    def finalize(self) -> "ExecutionReport":
        self.total_elapsed_seconds = time.time() - self.started_at
        return self

    # --- summarization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_features": self.requested_features,
            "completed_features": self.completed_features,
            "skipped_features": self.skipped_features,
            "failed_features": self.failed_features,
            "optional_dependency_failures": self.optional_dependency_failures,
            "approximations_used": self.approximations_used,
            "retries": self.retries,
            "batch_reductions": self.batch_reductions,
            "timings": self.timings,
            "output_columns": self.output_columns,
            "warnings": self.warnings,
            "n_batches": self.n_batches,
            "batch_rows": self.batch_rows,
            "memory_budget_bytes": self.memory_budget_bytes,
            "disk_budget_bytes": self.disk_budget_bytes,
            "output_backend": self.output_backend,
            "mode": self.mode,
            "errors_policy": self.errors_policy,
            "cache_stats": self.cache_stats,
            "checkpoint_dir": self.checkpoint_dir,
            "resumed": self.resumed,
            "resumed_features": self.resumed_features,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "backend_report": self.backend_report,
            "distributed_report": self.distributed_report,
        }

    def summary(self) -> str:
        lines = [
            f"requested={len(self.requested_features)} "
            f"completed={len(self.completed_features)} "
            f"skipped={len(self.skipped_features)} "
            f"failed={len(self.failed_features)}",
            f"mode={self.mode} errors={self.errors_policy} output_backend={self.output_backend} "
            f"batches={self.n_batches} batch_rows={self.batch_rows}",
        ]
        if self.approximations_used:
            lines.append(f"approximated: {sorted(self.approximations_used)}")
        if self.retries:
            lines.append(f"retries: {self.retries}")
        if self.batch_reductions:
            lines.append(f"batch_reductions: {self.batch_reductions}")
        if self.optional_dependency_failures:
            lines.append(f"optional_dependency_failures: {sorted(self.optional_dependency_failures)}")
        lines.append(f"elapsed={self.total_elapsed_seconds:.3f}s")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ExecutionReport({self.summary()!r})"
