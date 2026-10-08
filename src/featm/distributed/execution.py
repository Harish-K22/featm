from __future__ import annotations

import contextlib
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..cache import scoped_cache
from ..execution import (
    PreparedRun,
    _compute_transform_with_retry,
    _prepare_run,
)
from ..engine import _selected_specs
from ..gpu.backend import any_gpu_available
from ..gpu.cache import gpu_cache_budget_bytes, scoped_gpu_cache
from ..gpu.memory import GPUMemoryBudget
from ..matrix import DiskFeatureMatrixWriter, FeatureMatrix
from ..registry import REGISTRY, BatchSemantics
from ..report import ExecutionReport
from ..resources import derive_state_budget_bytes, parse_size, checkpoint_state_budget, check_checkpoint_state_capacity
from ..sources import canonical_label
from ..state import STATEFUL_ENCODERS
from .checkpoint import DistributedCheckpointIdentity, DistributedCheckpointManager
from .contracts import (
    DistributedFeatureError,
    DistributedFeatureUnsupportedError,
    FitPartitionResult,
    Partition,
    PartitionResult,
)
from .dask_runtime import DaskDistributedRuntime
from .local import LocalThreadRuntime
from .partition_provider import resolve_partition_frame, select_partition_provider
from .planner import build_distributed_plan


def _merged_params(spec, params: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    out = dict(spec.default_params)
    out.update((params or {}).get(spec.name, {}))
    return out


def _fit_partition_worker(
    part: Partition,
    *,
    stateful_names: Sequence[str],
    resolved_columns: Mapping[str, list[Any]],
    params: Mapping[str, Mapping[str, Any]] | None,
    state_budget_bytes: int | None,
) -> FitPartitionResult:
    frame = resolve_partition_frame(part.frame)  # a real DataFrame, or read from a Parquet descriptor here on the worker
    states: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in stateful_names:
        spec = REGISTRY.get(name)
        encoder = STATEFUL_ENCODERS[name]
        p = _merged_params(spec, params)
        try:
            state = encoder.init_state(state_budget_bytes=state_budget_bytes, spill_dir=None, params=p)
            col = resolved_columns[name][0]
            states[name] = encoder.update_state(state, frame[col], params=p)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return FitPartitionResult(index=part.index, states=states, feature_errors=errors)


def _transform_partition_worker(
    part: Partition,
    *,
    feature_names: Sequence[str],
    resolved_columns: Mapping[str, list[Any]],
    params: Mapping[str, Mapping[str, Any]] | None,
    fitted_states: Mapping[str, Any],
    backend: str,
    gpu_fallback: str,
    device: str | int | None,
    gpu_memory_budget: str | int | None,
    cache_bytes: int,
) -> PartitionResult:
    frame = resolve_partition_frame(part.frame)  # a real DataFrame, or read from a Parquet descriptor here on the worker
    report = ExecutionReport(requested_features=list(feature_names))
    feature_outputs: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    backend_policy = {
        "backend": backend,
        "gpu_fallback": gpu_fallback,
        "device": device,
        "gpu_memory_budget": gpu_memory_budget,
    }

    gpu_ctx = contextlib.nullcontext()
    if backend != "cpu":
        budget = GPUMemoryBudget.resolve(gpu_memory_budget=gpu_memory_budget, device=device)
        gpu_ctx = scoped_gpu_cache(max_bytes=gpu_cache_budget_bytes(budget))

    with scoped_cache(max_bytes=cache_bytes) as cache, gpu_ctx:
        for name in feature_names:
            spec = REGISTRY.get(name)
            try:
                feature_outputs[name] = _compute_transform_with_retry(
                    spec,
                    frame,
                    dict(resolved_columns),
                    _merged_params(spec, params),
                    dict(fitted_states),
                    report,
                    backend_policy=backend_policy,
                    batch_id=part.index,
                )
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        report.record_cache_stats(cache.stats())

    return PartitionResult(
        index=part.index,
        n_input_rows=len(frame),
        feature_outputs=feature_outputs,
        feature_errors=errors,
        backend_info=dict(report.backend_report),
        retries=dict(report.retries),
        batch_reductions=dict(report.batch_reductions),
        warnings=list(report.warnings),
    )


def _runtime_from_name(
    name: str,
    *,
    n_workers: int,
    scheduler_address: str | None,
    threads_per_worker: int,
    processes: bool,
):
    if name == "local":
        return LocalThreadRuntime(n_workers=n_workers)
    if name == "dask":
        return DaskDistributedRuntime(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            scheduler_address=scheduler_address,
            processes=processes,
        )
    raise ValueError("distributed_backend must be 'local' or 'dask'")


class _ReorderGate:
    """Bounds the logical reassembly window so a slow early partition cannot let the
    runtime race ahead and materialize the entire dataset (Finding 2, independent audit).

    ``max_in_flight`` alone only bounds scheduler-level CONCURRENCY (how many
    tasks may be simultaneously running). It says nothing about how many
    already-*completed* results a coordinator accumulates while waiting for
    one slow early partition to unblock source-order delivery -- and because
    each ``DistributedRuntime.map_stream`` implementation replenishes its
    submission queue every time the caller pulls one more result, a
    coordinator that unconditionally drains the stream (as the v0.4.0
    ``for idx, result in stream:`` loop did) causes the runtime to keep
    submitting and completing partitions indefinitely, even though only
    ``max_in_flight`` were ever concurrently *running*. Reproduced directly:
    with ``max_in_flight=2`` and partition 0 artificially slow among 100
    fast partitions, the v0.4.0 reorder buffer grew to 100 (the whole
    dataset), not 2.

    The fix: track how many results are currently buffered ahead of the
    delivery frontier, and refuse to let the runtime submit MORE work once
    that count reaches ``window``. Passed as ``should_submit_more`` to
    ``map_stream`` (see ``DistributedRuntime.map_stream``'s docstring) --
    submission is naturally gated by consumption (a runtime only submits a
    replacement when the caller pulls a result), so refusing to authorize a
    submission here, rather than refusing to *pull*, avoids deadlocking on
    the one partition (the frontier) needed to make progress: that
    partition was already dispatched during the runtime's initial fill and
    keeps running regardless of whether this gate authorizes anything
    further.
    """

    def __init__(self, window: int) -> None:
        self.window = max(1, int(window))
        self._outstanding = 0  # results currently buffered by the caller, not yet delivered

    def should_submit_more(self) -> bool:
        return self._outstanding < self.window

    def note_buffered(self, count: int) -> None:
        self._outstanding = count


def _ordered_results_with_preloaded(
    stream: Iterable[tuple[int, Any]],
    checkpoint_indices: Iterable[int],
    loader: "Any",
    *,
    gate: "_ReorderGate | None" = None,
    stats: dict | None = None,
) -> Iterable[Any]:
    """Turn out-of-order scheduler completions into deterministic source order.

    Two independent correctness properties, both audit findings against the
    v0.4.0 implementation:

    - Finding 2: the reassembly buffer for FRESH (runtime-produced) results
      is bounded -- ``gate`` (a :class:`_ReorderGate`) is updated with the
      buffer's current size after every fresh arrival, so
      ``should_submit_more`` can refuse further runtime submissions once the
      window is full.
    - Finding 3: checkpoint-resumed partitions are NOT eagerly materialized.
      ``checkpoint_indices`` is just a set of known-committed partition
      indices; ``loader(idx)`` is called to actually read a committed
      partition's data from disk exactly once, at the moment that index
      reaches the delivery frontier -- never upfront, never more than one at
      a time. This is what makes resuming a checkpoint with thousands of
      committed shards bounded-memory instead of "load everything into RAM
      before assembly begins."

    ``preloaded``/eagerly-materialized-dict semantics from the v0.4.0
    signature are gone entirely: a caller with nothing to preload passes
    ``checkpoint_indices=()`` and any ``loader`` (never called).
    """
    next_idx = 0
    buffered: dict[int, Any] = {}
    remaining_checkpoint: set[int] = set(checkpoint_indices)

    def _flush() -> list[Any]:
        nonlocal next_idx
        out: list[Any] = []
        while True:
            if next_idx in buffered:
                out.append(buffered.pop(next_idx))
                next_idx += 1
            elif next_idx in remaining_checkpoint:
                remaining_checkpoint.discard(next_idx)
                out.append(loader(next_idx))
                next_idx += 1
            else:
                break
        return out

    for value in _flush():
        yield value
    for idx, result in stream:
        buffered[idx] = result
        if gate is not None:
            gate.note_buffered(len(buffered))
        if stats is not None:
            stats["max_reorder_buffer_observed"] = max(stats.get("max_reorder_buffer_observed", 0), len(buffered))
        for value in _flush():
            yield value
        if gate is not None:
            gate.note_buffered(len(buffered))
    for value in _flush():
        yield value
    if buffered or remaining_checkpoint:
        missing = min(list(buffered) + list(remaining_checkpoint))
        raise RuntimeError(f"distributed runtime ended with non-contiguous result ordering near partition {missing}")


def _ordered_results(
    stream: Iterable[tuple[int, Any]], *, gate: "_ReorderGate | None" = None, stats: dict | None = None,
) -> Iterable[Any]:
    """Convenience wrapper over :func:`_ordered_results_with_preloaded` for the no-checkpoint case."""
    yield from _ordered_results_with_preloaded(stream, (), lambda idx: None, gate=gate, stats=stats)


def _merge_counter_dict(dst: dict[str, int], src: Mapping[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def distributed_extract(
    source: Any,
    *,
    features: Sequence[str] | str,
    modality: str | None = None,
    columns: Mapping[str, Any] | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str = "exact",
    errors: str = "raise",
    memory_budget: str | int | None = None,
    disk_budget: str | int | None = None,
    output: str = "memory",
    partition_rows: int | None = None,
    distributed_backend: str = "local",
    n_workers: int = 4,
    max_in_flight: int | None = None,
    task_retries: int = 1,
    scheduler_address: str | None = None,
    threads_per_worker: int = 1,
    processes: bool = True,
    backend: str = "cpu",
    gpu_fallback: str = "fallback",
    device: str | int | None = None,
    gpu_memory_budget: str | int | None = None,
    cache_budget: str | int | None = None,
    allow_memory_budget_override: bool = False,
    checkpoint: "str | Any | None" = None,
    allow_multi_worker_gpu: bool = False,
) -> FeatureMatrix:
    """Semantics-first distributed extraction (Dask Distributed, v0.4.1).

    ``distributed_backend='local'`` is the standard-library reference runtime
    used to validate semantics without Dask installed. ``'dask'`` is the real
    Dask Distributed adapter and requires ``featm[distributed]``.

    Exact distributed stateful fits (all 7 stateful encoders), transactional
    checkpoint/resume (fit and transform phases, partition-ID-keyed,
    content-checksummed), and worker-native Parquet I/O are all implemented
    and validated against real multi-process Dask -- see
    ``STEP4_DASK_HANDOFF.md`` and ``STEP4_DASK_AUDIT_FIX_HANDOFF.md`` for the
    full audit trail. Distributed stateful ``mode="approximate"``,
    FULL_DATASET/GLOBAL_OUTPUT features, and multi-node transport remain
    explicitly unsupported/undemonstrated -- unsupported requests fail before
    work begins rather than silently changing semantics.
    """
    feature_names = features if isinstance(features, str) else list(features)
    requested_specs = _selected_specs(feature_names)
    feature_names = [spec.name for spec in requested_specs]
    from ..contracts import validate_params
    extra = set(params or {}) - set(feature_names)
    if extra:
        raise ValueError(f"Parameters supplied for unselected features: {sorted(extra)}")
    for spec in requested_specs:
        validate_params(spec, (params or {}).get(spec.name, {}))
    if not feature_names:
        raise ValueError("Select at least one feature")

    # Stronger-than-Step2 preflight: FULL_DATASET/GLOBAL_OUTPUT and
    # STRUCTURED_SAMPLE features are never distributed, even if the complete
    # source would happen to fit in one local batch. Reject before _prepare_run
    # can reinterpret a caller-specified partition_rows as an ordinary local
    # batch-size constraint.
    pre_rejected: dict[str, str] = {}
    for spec in _selected_specs(feature_names):
        if spec.batch_semantics in (BatchSemantics.FULL_DATASET, BatchSemantics.GLOBAL_OUTPUT):
            pre_rejected[spec.name] = (
                f"{spec.name} has whole-dataset/global-output semantics and cannot be split across distributed workers "
                "without changing its mathematical definition."
            )
        elif spec.batch_semantics is BatchSemantics.STRUCTURED_SAMPLE:
            pre_rejected[spec.name] = (
                f"{spec.name} is STRUCTURED_SAMPLE and is not representable by the row-partition distributed runtime."
            )
    if pre_rejected and errors == "raise":
        raise DistributedFeatureUnsupportedError("; ".join(pre_rejected.values()))
    candidate_names = [n for n in feature_names if n not in pre_rejected]
    if not candidate_names:
        report = ExecutionReport(requested_features=list(feature_names), errors_policy=errors)
        for name, reason in pre_rejected.items():
            report.record_skipped(name, f"distributed execution rejected: {reason}")
        report.distributed_report = {
            "runtime": distributed_backend, "n_workers": n_workers, "partition_rows": partition_rows,
            "rejected_features": dict(pre_rejected),
        }
        matrix = FeatureMatrix.from_pandas(pd.DataFrame())
        matrix.report = report.finalize()
        return matrix
    if output not in {"memory", "disk", "auto"}:
        raise ValueError("output must be memory, disk, or auto")
    # Finding 3 (final closure audit, v0.4.1 -> v0.4.2): every optional
    # integer parameter here used truthiness (`x or default`) to fill in an
    # unset value -- but 0 (and any negative number) is ALSO falsy in Python,
    # so an explicitly invalid `distributed_max_in_flight=0` was silently
    # replaced by the default instead of being rejected. `None` (unset) and
    # an explicit invalid value must be distinguished, for every one of these
    # parameters, not just the ones that happened to already have a
    # `< 1` check below.
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be >= 1")
    if task_retries < 0:
        raise ValueError("task_retries must be >= 0")
    if partition_rows is not None and partition_rows < 1:
        raise ValueError("partition_rows must be >= 1 (or None to use the default)")
    if max_in_flight is not None and max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1 (or None to use the default)")
    if distributed_backend == "dask" and backend in ("gpu", "auto") and not allow_multi_worker_gpu:
        # Finding 8 (independent audit): the v0.4.0 guard only modeled
        # `processes=True and n_workers>1`. That missed real concurrency
        # sources: n_workers=1 with threads_per_worker>1 lets multiple GPU
        # partition tasks run concurrently INSIDE one worker process;
        # processes=False with multiple workers/threads is the same risk
        # without a process boundary; and an externally-supplied
        # `scheduler_address` can have an actual worker/thread count that
        # differs from the caller's `n_workers`/`threads_per_worker` hints.
        # Model concurrency as "how many partition tasks could the scheduler
        # actually run at the same instant" instead of one specific
        # configuration shape.
        probed_n_workers, probed_threads = n_workers, threads_per_worker
        if scheduler_address:
            try:
                from distributed import Client as _ProbeClient

                _probe = _ProbeClient(scheduler_address, timeout="5s")
                try:
                    info = _probe.scheduler_info()
                    workers_info = info.get("workers", {})
                    if workers_info:
                        probed_n_workers = len(workers_info)
                        probed_threads = max((w.get("nthreads", 1) or 1) for w in workers_info.values())
                finally:
                    _probe.close()
            except Exception:
                pass  # best-effort: fall back to the caller's hints rather than fail the preflight on a probe error
        total_concurrent_task_slots = max(1, probed_n_workers) * max(1, probed_threads)

        if total_concurrent_task_slots > 1:
            # Step 3's dispatcher defaults every GPU task to device index 0
            # unless `device=` is pinned -- and pinning the SAME explicit
            # device does not remove the risk either, since every concurrent
            # task slot (a separate worker PROCESS, or a separate THREAD
            # within one worker process -- both are real here) would still
            # target that one device. Only actually relevant if a GPU could
            # be reached at all: an explicit backend="gpu" request is always
            # relevant (that is the caller's stated intent, independent of
            # whether this particular environment currently has a device);
            # backend="auto" is only relevant if a GPU-capable feature was
            # actually selected AND a usable GPU device exists to route to --
            # rejecting a multi-worker CPU-only "auto" run because the word
            # "auto" was passed would be a false positive.
            requested_specs = _selected_specs(feature_names)
            any_gpu_capable_selected = any(getattr(spec, "gpu_available", False) for spec in requested_specs)
            gpu_relevant = any_gpu_capable_selected and (backend == "gpu" or any_gpu_available())
            if gpu_relevant:
                raise ValueError(
                    f"backend={backend!r} with distributed_backend='dask' would allow up to "
                    f"{total_concurrent_task_slots} concurrent partition tasks ({probed_n_workers} worker(s) x "
                    f"{probed_threads} thread(s) each) to run a GPU-capable feature at the same time, all "
                    "defaulting to the same GPU device -- a real oversubscription risk this milestone does not "
                    "manage (no multi-GPU resource routing). Use n_workers=1, threads_per_worker=1 for GPU "
                    "validation, or pass allow_multi_worker_gpu=True to accept the risk explicitly."
                )
    if max_in_flight is None:
        max_in_flight = max(1, 2 * n_workers)

    # Force Step 2 to reason using the same partition size the distributed
    # coordinator will actually use. This preserves required-param guards and
    # all Source/batch-semantics checks before any task is submitted.
    requested_partition_rows = partition_rows if partition_rows is not None else 50_000
    prepared = _prepare_run(
        source,
        features=candidate_names,
        modality=modality,
        columns=columns,
        mode=mode,
        errors=errors,
        memory_budget=memory_budget,
        disk_budget=disk_budget,
        batch_rows=requested_partition_rows,
        cache_budget=cache_budget,
        params={name: value for name, value in (params or {}).items() if name in candidate_names},
        backend=backend,
        gpu_fallback=gpu_fallback,
        device=device,
        gpu_memory_budget=gpu_memory_budget,
    )
    # _prepare_run may shrink/grow its batch for local resource semantics. For
    # distributed work we never exceed the caller-requested partition size;
    # shrinking is allowed because it only creates more row-safe partitions.
    partition_rows = max(1, min(requested_partition_rows, prepared.resolved_batch_rows))
    plan = build_distributed_plan(
        prepared,
        runtime=distributed_backend,
        partition_rows=partition_rows,
        max_in_flight=max_in_flight,
        params=params,
    )
    prepared.report.requested_features = list(feature_names)
    if pre_rejected:
        plan.rejected_features.update(pre_rejected)

    if plan.rejected_features:
        detail = "; ".join(f"{k}: {v}" for k, v in plan.rejected_features.items())
        if errors == "raise":
            raise DistributedFeatureUnsupportedError(detail)
        for name, reason in plan.rejected_features.items():
            prepared.report.record_skipped(name, f"distributed execution rejected: {reason}")

    active_names = [s.name for s in prepared.active_specs if s.name not in plan.rejected_features]
    if not active_names:
        matrix = FeatureMatrix.from_pandas(pd.DataFrame())
        prepared.report.distributed_report = {
            "runtime": distributed_backend,
            "n_workers": n_workers,
            "partition_rows": partition_rows,
            "rejected_features": dict(plan.rejected_features),
        }
        matrix.report = prepared.report.finalize()
        return matrix

    active_specs = [REGISTRY.get(n) for n in active_names]
    needed_columns = []
    seen = set()
    for spec in active_specs:
        for c in prepared.resolved_columns[spec.name]:
            # Pandas labels can be unhashable only in exotic cases; column
            # labels supported by FEATM checkpoint identity are hashable.
            if c not in seen:
                seen.add(c); needed_columns.append(c)

    # Computed here (moved ahead of checkpoint identity construction, final
    # closure pass v0.4.2 -> v0.4.3) because the checkpoint identity needs
    # this NORMALIZED value -- see DistributedCheckpointIdentity's
    # state_budget_bytes docstring for why raw memory_budget text is not
    # the right thing to compare.
    state_budget_bytes = derive_state_budget_bytes(prepared.budget.memory_bytes)

    checkpoint_mgr: DistributedCheckpointManager | None = None
    checkpoint_resumed = False
    if checkpoint is not None:
        # Finding 7 (independent audit): checkpointing unconditionally needs
        # PyArrow for transform-shard commits, but featm[distributed] does not
        # pull it in. Fail clearly, before any fit work starts, rather than
        # doing a full fit pass and only then crashing with a raw
        # ModuleNotFoundError during the first transform commit.
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Distributed checkpointing requires the IO extra: pip install 'featm[io,distributed]'"
            ) from exc
        checkpoint_mgr = DistributedCheckpointManager(checkpoint)
        existing_manifest = checkpoint_mgr.load()
        retained_budget, state_budget_policy = checkpoint_state_budget(
            prepared.budget, memory_budget=memory_budget, stateful=bool(plan.stateful_features),
            existing=existing_manifest,
        )
        if retained_budget is not None:
            state_budget_bytes = retained_budget
        identity = DistributedCheckpointIdentity(
            featm_version=__import__("featm").__version__,
            source_fingerprint=prepared.src.fingerprint(),
            schema_fingerprint=prepared.src.schema().fingerprint(),
            features=list(active_names),
            params={k: dict(v) for k, v in (params or {}).items()},
            partition_rows=int(partition_rows),
            mode=mode,
            errors=errors,
            # Finding 6 (independent audit): canonical_label (not bare str())
            # preserves each column label's Python type in the identity, so an
            # int-labeled column and a same-spelled string-labeled column are
            # never mistaken for the same checkpoint identity.
            columns={name: [canonical_label(c) for c in prepared.resolved_columns[name]] for name in active_names},
            # None when no stateful feature is selected -- see the dataclass
            # docstring. `plan.stateful_features` reflects the FIT-eligible
            # stateful features for this exact run.
            state_budget_bytes=state_budget_bytes if plan.stateful_features else None,
            state_budget_policy=state_budget_policy,
        )
        existing_manifest = checkpoint_mgr.load()
        checkpoint_resumed = existing_manifest is not None
        checkpoint_mgr.initialize_or_resume(identity)
        if checkpoint_resumed:
            integrity_problems = checkpoint_mgr.validate_committed_shards_integrity()
            if integrity_problems:
                raise RuntimeError(
                    "Distributed checkpoint integrity check failed on resume: " + "; ".join(integrity_problems)
                )
        if plan.stateful_features and memory_budget is None and not checkpoint_mgr.transform_done():
            check_checkpoint_state_capacity(prepared.budget, state_budget_bytes)

    # One provider instance for the whole call: a Parquet-native provider
    # computes its row-group -> partition-index boundaries ONCE, so fit and
    # transform phases (different column projections, same row boundaries)
    # deterministically agree on what partition K means without needing to
    # coordinate separately.
    partition_provider = select_partition_provider(prepared, partition_rows)
    reorder_stats: dict[str, int] = {}
    state_merge_seconds = 0.0

    fitted_states: dict[str, Any] = {}
    failed_stateful: set[str] = set()
    fit_partitions = 0
    transform_partitions = 0
    # Finding 9 (independent audit): unambiguous "computed fresh this run" vs
    # "reused from a prior run's checkpoint" counters. The old
    # fit_partitions_completed/transform_partitions_completed counted BOTH
    # together while a separate ..._skipped_from_checkpoint counted reused
    # ones AGAIN -- so a resumed run's report looked like it recomputed
    # everything it actually just reused.
    fit_partitions_computed_this_run = 0
    fit_partitions_reused_from_checkpoint = 0
    transform_partitions_computed_this_run = 0
    transform_partitions_reused_from_checkpoint = 0

    runtime = _runtime_from_name(
        distributed_backend,
        n_workers=n_workers,
        scheduler_address=scheduler_address,
        threads_per_worker=threads_per_worker,
        processes=processes,
    )
    fit_already_committed: set[int] = set()
    try:
        if plan.stateful_features and checkpoint_mgr is not None and checkpoint_mgr.fit_done():
            # Resume path: the fit phase's merge already completed and was
            # committed in a prior run -- skip re-scanning/re-fitting entirely.
            fitted_states = checkpoint_mgr.load_fit_finalized()
            fit_already_committed = checkpoint_mgr.fit_committed_indices()
            fit_partitions_reused_from_checkpoint = len(fit_already_committed)
            # A feature that failed during the ORIGINAL run's fit phase was
            # never in fitted_states, but that fact was never persisted
            # anywhere restorable in v0.4.0 -- so a fast-path resume left
            # `failed_stateful` empty, silently sending the already-failed
            # feature into the transform phase with no fitted state at all.
            # `commit_fit_finalized` now records exactly which features
            # failed and why, so this resume path reproduces the identical
            # feature set (and report) an uninterrupted run would have
            # produced.
            for name, msg in checkpoint_mgr.fit_failed_features().items():
                failed_stateful.add(name)
                prepared.report.record_failed(name, msg)
        elif plan.stateful_features:
            if not prepared.src.replayable:
                raise DistributedFeatureUnsupportedError(
                    "Distributed stateful execution requires a replayable source: one distributed fit pass followed by "
                    "one distributed transform pass. IteratorSource/single-use streams cannot be replayed."
                )
            fit_fn = partial(
                _fit_partition_worker,
                stateful_names=tuple(plan.stateful_features),
                resolved_columns=prepared.resolved_columns,
                params=params,
                state_budget_bytes=state_budget_bytes,
            )
            fit_columns = []
            seen_fit = set()
            for name in plan.stateful_features:
                for c in prepared.resolved_columns[name]:
                    if c not in seen_fit:
                        seen_fit.add(c); fit_columns.append(c)
            merged_raw: dict[str, Any] = {}
            failed_messages: dict[str, str] = {}

            # Checkpoint resume: skip re-submitting partitions whose fit
            # contribution was already committed. Their raw state is loaded
            # LAZILY -- one partition at a time, exactly when the ordered
            # merge below reaches that logical index -- not materialized
            # upfront for every committed partition (Finding 3, independent
            # audit: eagerly loading every committed partition's state
            # defeats out-of-core guarantees for a checkpoint with many
            # committed partitions).
            fit_already_committed = checkpoint_mgr.fit_committed_indices() if checkpoint_mgr is not None else set()

            def _load_fit_preloaded(idx: int) -> FitPartitionResult:
                data = checkpoint_mgr.load_fit_partition(idx)
                return FitPartitionResult(index=idx, states=data["states"], feature_errors=data["feature_errors"])

            fit_gate = _ReorderGate(window=max_in_flight)
            raw_stream = runtime.map_stream(
                fit_fn,
                partition_provider.iter_partitions(fit_columns, skip_indices=frozenset(fit_already_committed)),
                max_in_flight=max_in_flight, retries=task_retries,
                should_submit_more=fit_gate.should_submit_more,
            )

            def _commit_as_arrives(stream: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
                for idx, result in stream:
                    if checkpoint_mgr is not None:
                        if errors == "raise" and result.feature_errors:
                            # Finding 1 (independent audit): do NOT persist any
                            # part of a partition whose fit produced a feature
                            # error under errors="raise". Committing a partial
                            # result here (as v0.4.0 did) would mark the whole
                            # partition index "done" before the error below is
                            # raised to the caller; a resumed run would then
                            # see fit_already_committed include this index,
                            # skip re-fitting it entirely, and silently
                            # finalize a merged state missing this partition's
                            # contribution to the feature that failed -- with
                            # the original error nowhere reproduced. Leaving
                            # it uncommitted forces a resumed run to
                            # re-submit and re-fit exactly this partition,
                            # deterministically reproducing the same failure
                            # (or succeeding, if the underlying cause is now
                            # fixed) -- exactly like an uninterrupted run.
                            pass
                        elif result.states:
                            # Partial success under errors!="raise" (some
                            # stateful features ok, one failed) is committed
                            # together with its feature_errors record, so a
                            # resumed run reconstructs the identical partial
                            # result rather than one that looks fully clean.
                            checkpoint_mgr.commit_fit_partition(
                                idx, result.states, feature_errors=result.feature_errors,
                                state_budget_bytes=state_budget_bytes,
                            )
                    yield idx, result

            for result in _ordered_results_with_preloaded(
                _commit_as_arrives(raw_stream), fit_already_committed, _load_fit_preloaded,
                gate=fit_gate, stats=reorder_stats,
            ):
                fit_partitions += 1
                for name, msg in result.feature_errors.items():
                    # Mirrors featm.execution's local fit-phase loop: ANY
                    # partition's fit failure raises immediately under
                    # errors="raise" -- a failed fit invalidates the whole
                    # feature, there is no "later batch" tolerance the way
                    # transform has (see the transform-phase priming check
                    # below for that distinction).
                    if errors == "raise":
                        raise DistributedFeatureError(name, result.index, msg, phase="fit")
                    failed_messages.setdefault(name, msg)
                _merge_t0 = time.perf_counter()
                for name, raw in result.states.items():
                    if name in failed_messages:
                        continue
                    encoder = STATEFUL_ENCODERS[name]
                    if name not in merged_raw:
                        merged_raw[name] = raw
                    else:
                        try:
                            merged_raw[name] = encoder.merge_states(
                                merged_raw[name], raw, params=_merged_params(REGISTRY.get(name), params)
                            )
                        except Exception as exc:
                            if errors == "raise":
                                raise
                            failed_messages[name] = f"{type(exc).__name__}: {exc}"
                            merged_raw.pop(name, None)
                state_merge_seconds += time.perf_counter() - _merge_t0
            fit_partitions_reused_from_checkpoint = len(fit_already_committed)
            fit_partitions_computed_this_run = fit_partitions - fit_partitions_reused_from_checkpoint
            for name in plan.stateful_features:
                if name in failed_messages or name not in merged_raw:
                    failed_stateful.add(name)
                    prepared.report.record_failed(name, failed_messages.get(name, "distributed fit produced no state"))
                else:
                    try:
                        fitted_states[name] = STATEFUL_ENCODERS[name].finalize_state(
                            merged_raw[name], params=_merged_params(REGISTRY.get(name), params)
                        )
                    except Exception as exc:
                        if errors == "raise":
                            raise
                        failed_stateful.add(name)
                        failed_messages[name] = f"{type(exc).__name__}: {exc}"
                        prepared.report.record_failed(name, failed_messages[name])
            if checkpoint_mgr is not None:
                # Persisted so a later fast-path resume (checkpoint_mgr.fit_done()
                # already true) can restore failed_stateful correctly instead of
                # treating every requested stateful feature as successful merely
                # because the fit phase, as a whole, finished (Finding 1 follow-on).
                checkpoint_mgr.commit_fit_finalized(
                    fitted_states,
                    failed_features={name: failed_messages.get(name, "distributed fit produced no state") for name in failed_stateful},
                )
                checkpoint_mgr.mark_fit_done()

        transform_names = [n for n in active_names if n not in failed_stateful]

        primed: dict[str, list[str]] | None = None
        output_columns: list[str] = []
        if checkpoint_mgr is not None:
            prior_schema = checkpoint_mgr.load_primed_schema()
            if prior_schema is not None:
                primed, output_columns = dict(prior_schema[0]), list(prior_schema[1])
                for name, msg in checkpoint_mgr.transform_priming_failed_features().items():
                    prepared.report.record_failed(name, msg)
        memory_chunks: list[pd.DataFrame] = []
        memory_bytes = 0
        writer: DiskFeatureMatrixWriter | None = None
        disk_budget_bytes = None if disk_budget is None else parse_size(disk_budget)
        use_disk = output == "disk"
        if use_disk:
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise ImportError("distributed output='disk' requires pip install 'featm[io]'") from exc
        degraded: dict[str, int] = {}

        # Finding 2 (final closure audit, v0.4.1 -> v0.4.2), Case A: if EVERY
        # requested feature already failed during the fit phase (under
        # errors!="raise"), there is nothing left to transform. v0.4.1
        # submitted an empty-feature-set transform pass anyway, which built a
        # zero-COLUMN priming frame and then crashed the per-partition
        # row-count check (0 output rows != N input rows) -- a real crash for
        # a case the canonical local engine handles cleanly (a clean empty
        # result, matching this exact scenario: featm.extract(..., errors=
        # "skip") with the only selected feature failing returns shape (0, 0)
        # and records it in failed_features, it does not raise). Skip
        # submitting/scanning entirely; `primed` stays None, which the tail
        # below (see "if primed is None or not output_columns") already
        # produces the correct empty FeatureMatrix for.
        #
        # Case B (every candidate feature fails during transform PRIMING, or
        # a checkpoint already recorded that outcome from a prior run) is
        # handled inside the loop below: once `output_columns` is (or
        # becomes) empty, the per-partition row-count check is skipped rather
        # than crashing, and the loop stops pulling further partitions --
        # every subsequent partition would produce the identical empty
        # result, so there is nothing more to learn by continuing to scan.
        if transform_names and not (primed is not None and not output_columns):
            if primed is not None and use_disk:
                writer = DiskFeatureMatrixWriter(
                    Path(tempfile.mkdtemp(prefix="featm-distributed-")), output_columns, disk_budget_bytes=disk_budget_bytes,
                )
            transform_fn = partial(
                _transform_partition_worker,
                feature_names=tuple(transform_names),
                resolved_columns=prepared.resolved_columns,
                params=params,
                fitted_states=fitted_states,
                backend=backend,
                gpu_fallback=gpu_fallback,
                device=device,
                gpu_memory_budget=gpu_memory_budget,
                cache_bytes=prepared.cache_bytes,
            )
            # Checkpoint resume: skip re-submitting/recomputing already-committed
            # transform partitions. A previously-committed shard is reloaded
            # LAZILY -- one at a time, exactly when the ordered assembly loop
            # below reaches that logical index, never materialized upfront
            # (Finding 3, independent audit) -- and fed into the SAME assembly
            # loop as if it had just been computed, so memory/disk assembly logic
            # never needs to know which partitions are fresh vs reused.
            transform_already_committed = checkpoint_mgr.transform_committed_indices() if checkpoint_mgr is not None else set()

            def _load_transform_preloaded(idx: int) -> tuple[int, str, Any]:
                return (idx, "checkpoint", checkpoint_mgr.load_transform_partition(idx))

            transform_gate = _ReorderGate(window=max_in_flight)
            raw_stream = runtime.map_stream(
                transform_fn,
                (() if checkpoint_mgr is not None and checkpoint_mgr.transform_done() else
                 partition_provider.iter_partitions(needed_columns, skip_indices=frozenset(transform_already_committed))),
                max_in_flight=max_in_flight,
                retries=task_retries,
                should_submit_more=transform_gate.should_submit_more,
            )
            # _ordered_results_with_preloaded yields the bare buffered VALUE, not
            # an (idx, value) pair -- the partition index is embedded inside the
            # tagged tuple itself here so it survives that unwrapping.
            tagged_stream = ((idx, (idx, "fresh", result)) for idx, result in raw_stream)

            for part_idx, kind, payload in _ordered_results_with_preloaded(
                tagged_stream, transform_already_committed, _load_transform_preloaded,
                gate=transform_gate, stats=reorder_stats,
            ):
                transform_partitions += 1
                if kind == "checkpoint":
                    transform_partitions_reused_from_checkpoint += 1
                else:
                    transform_partitions_computed_this_run += 1
                if kind == "checkpoint":
                    # Already-committed shard from a prior run -- its schema was
                    # committed alongside the priming partition, so `primed` must
                    # already be known by the time any checkpoint-kind partition
                    # is reached (partition 0, in logical order, is always
                    # either the priming partition itself or comes after it).
                    if primed is None:
                        raise RuntimeError(
                            f"distributed checkpoint partition {part_idx} has a committed shard but no recorded output "
                            "schema -- the checkpoint directory may be corrupt or from an incompatible run."
                        )
                    frame = payload
                    n = len(frame)
                    for name, error in checkpoint_mgr.transform_partition_errors(part_idx).items():
                        degraded[name] = degraded.get(name, 0) + 1
                        prepared.report.warnings.append(f"{name}: distributed partition {part_idx} failed ({error}); filled with NaN")
                        prepared.report._warn(prepared.report.warnings[-1])
                else:
                    result = payload
                    n = result.n_input_rows
                    for name, info in result.backend_info.items():
                        prepared.report.record_backend_info(name, info)
                    _merge_counter_dict(prepared.report.retries, result.retries)
                    _merge_counter_dict(prepared.report.batch_reductions, result.batch_reductions)
                    prepared.report.warnings.extend(result.warnings)

                    if primed is None:
                        primed = {}
                        priming_failed_messages: dict[str, str] = {}
                        for name in transform_names:
                            if name in result.feature_errors:
                                # Mirrors featm.execution's local transform-phase
                                # priming batch: a feature failure on the FIRST
                                # (schema-establishing) logical partition raises
                                # immediately under errors="raise" -- identical to
                                # small-data extract()'s own behavior. A LATER
                                # partition's failure is deliberately NOT raised here
                                # (see the per-partition degrade loop below), matching
                                # the local engine's "one late batch failure does not
                                # abort an otherwise-successful run" philosophy.
                                if errors == "raise":
                                    raise DistributedFeatureError(name, result.index, result.feature_errors[name], phase="transform")
                                prepared.report.record_failed(name, result.feature_errors[name])
                                priming_failed_messages[name] = result.feature_errors[name]
                                continue
                            expanded = result.feature_outputs.get(name)
                            if expanded is None:
                                msg = "distributed partition 0 produced no output"
                                prepared.report.record_failed(name, msg)
                                priming_failed_messages[name] = msg
                                continue
                            primed[name] = list(expanded)
                            output_columns.extend(primed[name])
                        if checkpoint_mgr is not None:
                            checkpoint_mgr.commit_primed_schema(primed, output_columns, failed_features=priming_failed_messages)
                        if use_disk and output_columns:
                            writer = DiskFeatureMatrixWriter(
                                Path(tempfile.mkdtemp(prefix="featm-distributed-")), output_columns,
                                disk_budget_bytes=disk_budget_bytes,
                            )

                    out: dict[str, np.ndarray] = {}
                    for name, cols in primed.items():
                        expanded = result.feature_outputs.get(name)
                        err = result.feature_errors.get(name)
                        if err is not None or expanded is None:
                            if errors == "raise":
                                raise DistributedFeatureError(name, result.index, err or "no output", phase="transform")
                            degraded[name] = degraded.get(name, 0) + 1
                            prepared.report.warnings.append(
                                f"{name}: distributed partition {result.index} failed ({err or 'no output'}); filled with NaN"
                            )
                            prepared.report._warn(prepared.report.warnings[-1])
                            for c in cols:
                                out[c] = np.full(n, np.nan)
                            continue
                        if list(expanded) != cols:
                            raise DistributedFeatureUnsupportedError(
                                f"{name} emitted schema {list(expanded)} on distributed partition {result.index}, "
                                f"different from partition 0 schema {cols}."
                            )
                        for c in cols:
                            out[c] = np.asarray(expanded[c])
                    frame = pd.DataFrame(out, columns=output_columns)
                    # Case B: zero surviving output columns is a terminal state,
                    # not an arbitrary row-count mismatch -- there is no column to
                    # hold any row's contribution, so the ordinary "did this
                    # partition's frame have the right number of rows" check does
                    # not apply here (mirrors the final row-accounting invariant's
                    # own `shape[1] > 0` exemption below). Row accounting remains
                    # fully enforced whenever at least one feature is active.
                    if output_columns and len(frame) != n:
                        raise RuntimeError(
                            f"distributed partition {result.index} produced {len(frame)} output rows for {n} input rows"
                        )
                    if checkpoint_mgr is not None and output_columns:
                        checkpoint_mgr.commit_transform_partition(part_idx, frame, feature_errors={name: result.feature_errors.get(name, "no output") for name in primed if name in result.feature_errors or name not in result.feature_outputs})

                if not output_columns:
                    # Case B, reached either by discovering it fresh on this
                    # (priming) partition, or by resuming a checkpoint that
                    # already recorded it: every subsequent partition would
                    # produce the identical empty contribution, so stop
                    # scanning the source rather than process the rest.
                    break

                frame_bytes = int(frame.memory_usage(deep=True).sum())
                if not use_disk and output in ("memory", "auto"):
                    projected = memory_bytes + frame_bytes
                    if projected > prepared.budget.memory_bytes:
                        if output == "memory" and not allow_memory_budget_override:
                            raise MemoryError(
                                f"distributed output='memory' accumulated ~{projected:,} bytes, exceeding memory_budget "
                                f"({prepared.budget.memory_bytes:,}). Use output='auto'/'disk', increase memory_budget, or "
                                "pass allow_memory_budget_override=True."
                            )
                        if output == "auto":
                            try:
                                import pyarrow  # noqa: F401
                            except ImportError as exc:
                                raise ImportError(
                                    "distributed output='auto' exceeded memory_budget and needs featm[io] to spill to disk"
                                ) from exc
                            use_disk = True
                            writer = DiskFeatureMatrixWriter(
                                Path(tempfile.mkdtemp(prefix="featm-distributed-")), output_columns,
                                disk_budget_bytes=disk_budget_bytes,
                            )
                            for prior in memory_chunks:
                                writer.write_batch(prior)
                            memory_chunks.clear(); memory_bytes = 0
                if use_disk:
                    assert writer is not None
                    writer.write_batch(frame)
                else:
                    memory_chunks.append(frame); memory_bytes += frame_bytes

        if checkpoint_mgr is not None and (primed is not None or not transform_names):
            # Covers Case B too (primed == {}, output_columns == []): marking
            # transform done lets a resume short-circuit via the
            # checkpoint-loaded-empty-schema branch above instead of
            # re-scanning the source to reconfirm the identical outcome.
            #
            # `not transform_names` covers Case A (final closure pass,
            # v0.4.2 -> v0.4.3): every stateful feature already failed during
            # the fit phase, so the transform submission loop above was
            # skipped entirely and `primed` was never assigned -- but this is
            # still a SUCCESSFUL terminal state (an exception anywhere above
            # this point would have propagated out of the function already,
            # never reaching this line), not an incomplete run. Leaving
            # transform_done=False forever after a genuinely successful
            # all-fit-failed run was inconsistent checkpoint-completion
            # metadata (not a numerical-correctness bug -- a resume already
            # avoids transform work either way, since transform_names is
            # recomputed as empty every time -- but still worth being exact
            # about "did this run finish" for anything that inspects the
            # manifest directly).
            checkpoint_mgr.mark_transform_done()

        if primed is None or not output_columns:
            # Case A (primed never established -- transform_names was empty
            # from the start) or Case B (primed == {}, no feature survived
            # transform priming): a clean empty result, matching the
            # canonical local engine's behavior for "every selected feature
            # failed/was skipped" rather than a row-count crash.
            result_matrix = FeatureMatrix.from_pandas(pd.DataFrame())
        elif use_disk:
            assert writer is not None
            result_matrix = writer.finalize(
                extra_manifest={"distributed_runtime": distributed_backend, "distributed_partitions": transform_partitions}
            )
        else:
            result_matrix = FeatureMatrix.from_pandas(
                pd.concat(memory_chunks, ignore_index=True) if memory_chunks else pd.DataFrame(columns=output_columns)
            )

        # A source with a known row count gives us a strong no-loss/no-duplication invariant.
        if result_matrix.shape[1] > 0 and prepared.src.n_rows is not None and len(result_matrix) != prepared.src.n_rows:
            raise RuntimeError(
                f"distributed row-accounting invariant failed: source has {prepared.src.n_rows} rows but output has "
                f"{len(result_matrix)} rows"
            )

        for name, cols in (primed or {}).items():
            prepared.report.record_completed(name, columns=cols, elapsed=0.0)
        prepared.report.n_batches = transform_partitions + fit_partitions
        prepared.report.batch_rows = partition_rows
        prepared.report.output_backend = "disk" if result_matrix.is_disk_backed else "memory"

        dask_version = distributed_version = None
        # Finding 9 (independent audit): report the SCHEDULER'S truth, not
        # just the caller's input, when it is available -- an internally
        # created LocalCluster's real address was never the caller-supplied
        # `scheduler_address` (usually None), and an externally-supplied
        # client's actual worker/thread count can differ from the `n_workers`/
        # `threads_per_worker` the caller merely *hinted* when constructing
        # this runtime.
        actual_scheduler_address = scheduler_address
        actual_worker_count = n_workers
        actual_threads_by_worker: dict[str, int] | None = None
        if distributed_backend == "dask":
            try:
                import dask as _dask
                import distributed as _distributed

                dask_version, distributed_version = _dask.__version__, _distributed.__version__
            except ImportError:
                pass
            try:
                info = runtime.client.scheduler_info()  # type: ignore[attr-defined]
                actual_scheduler_address = info.get("address", scheduler_address)
                workers_info = info.get("workers", {})
                if workers_info:
                    actual_worker_count = len(workers_info)
                    actual_threads_by_worker = {addr: w.get("nthreads") for addr, w in workers_info.items()}
            except Exception:
                pass  # best-effort: an unreachable/closing scheduler falls back to the caller's inputs, not a crash
        source_rows = prepared.src.n_rows
        output_rows = len(result_matrix)
        row_accounting_ok = source_rows is None or output_rows == source_rows or result_matrix.shape[1] == 0

        prepared.report.distributed_report = {
            "runtime": distributed_backend,
            "dask_version": dask_version,
            "distributed_version": distributed_version,
            "scheduler_address": scheduler_address,
            "actual_scheduler_address": actual_scheduler_address,
            "n_workers": n_workers,
            "actual_worker_count": actual_worker_count,
            "threads_per_worker": threads_per_worker,
            "actual_threads_by_worker": actual_threads_by_worker,
            "processes": processes,
            "partition_rows": partition_rows,
            "partitions_total": transform_partitions,
            "max_in_flight": max_in_flight,
            "max_reorder_buffer_observed": reorder_stats.get("max_reorder_buffer_observed", 0),
            "task_retries": task_retries,
            "state_merge_seconds": round(state_merge_seconds, 6),
            "fit_partitions_computed_this_run": fit_partitions_computed_this_run,
            "fit_partitions_reused_from_checkpoint": fit_partitions_reused_from_checkpoint,
            "transform_partitions_computed_this_run": transform_partitions_computed_this_run,
            "transform_partitions_reused_from_checkpoint": transform_partitions_reused_from_checkpoint,
            "stateful_features": list(plan.stateful_features),
            "map_features": list(plan.map_features),
            "rejected_features": dict(plan.rejected_features),
            "degraded_partitions_by_feature": degraded,
            "source_io": partition_provider.kind,
            "partition_provider": partition_provider.kind,
            "checkpoint_enabled": checkpoint_mgr is not None,
            "checkpoint_resumed": checkpoint_resumed,
            "checkpoint_dir": str(checkpoint_mgr.directory) if checkpoint_mgr is not None else None,
            "source_rows": source_rows,
            "output_rows": output_rows,
            "row_accounting_ok": row_accounting_ok,
            # Finding 9: per-partition backend_info/retries/warnings are only
            # ever recorded for partitions COMPUTED THIS RUN -- a reused
            # checkpoint shard contributes its committed data to the output,
            # but its original run's backend/retry/degradation history was
            # never persisted and is not reconstructed here. Say so
            # explicitly rather than let a resumed run's report imply it is
            # complete historical truth.
            "backend_report_covers_reused_partitions": transform_partitions_reused_from_checkpoint == 0,
        }
        result_matrix.report = prepared.report.finalize()
        return result_matrix
    finally:
        runtime.close()
