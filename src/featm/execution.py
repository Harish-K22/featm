"""Out-of-core execution engine.

Orchestrates everything Step 2 adds on top of the v0.1.0 registry: Source
scanning in bounded, column-projected batches; a resource planner that picks
a batch size and adapts it under memory pressure; per-feature fault
isolation; the exact/approximate protocol; leakage-safe two-pass fitting for
stateful features; a shared intermediate cache per batch; memory- or
disk-backed output; checkpoint/resume; and one execution report describing
exactly what happened.

Scope: this engine's column-projected scanning is naturally suited to
row-oriented tabular data (CSV/TSV/JSONL/Parquet/Arrow/pandas). Two kinds of
selected features are supported per run:

- **tabular** features (``modality == "tabular"``) are applied to the
  explicitly resolved input column(s) per feature (one per ``spec.arity``),
  via ``columns=`` or, when unambiguous, inferred from dtype;
- **whole-batch** features (any other modality, e.g. ``timeseries.*``,
  ``information.*``) receive the batch itself (optionally column-projected
  via ``columns=``), exactly as :func:`featm.extract` already passes a whole
  in-memory array -- this engine just replays that same call once per batch
  and stitches the results together. Which non-tabular modalities this is
  actually a valid/meaningful operation for is documented in
  ``docs/SCALING.md``'s "out-of-core modality support" section -- not every
  whole-batch modality's small-data calling convention is representable as
  row-oriented Source batches. A subset of these (``image.*``/``audio.*``/
  ``video.*``/``graph.*``, ``batch_semantics=OPAQUE_OBJECT_SAMPLE``) needs
  one Source row to BE a real pre-loaded Python object (a ``numpy.ndarray``
  sample, a ``networkx.Graph``) -- genuinely supported, but only from an
  in-memory ``PandasSource`` (fifth audit pass, Finding 1; see
  ``_adapt_object_column`` and that same doc section for exactly which
  modalities work, and how a dense-array modality's per-row objects get
  stacked into the batch tensor its implementation expects). File-backed
  sources (CSV/JSONL/Parquet/Arrow) cannot represent arbitrary Python
  objects in a cell and are rejected clearly for these features, not
  silently treated as unsupported-by-coincidence.

This mirrors the existing small-data ``extract()`` semantics on purpose: an
out-of-core run is "the same extraction, batch by batch," not a separate
programming model. Feature-selection validation (empty selection,
``features="all"``, an explicit ``modality=`` mismatch) reuses exactly the
same code the small-data path uses (:func:`featm.engine._selected_specs`),
so the two paths cannot silently diverge on what counts as a valid request.

Batch-safety contract (audited): a feature is only ever chunked across
multiple batches if that is mathematically safe. Three cases:

- **row-independent** (default) -- each row/sample's output depends only on
  that row; safe to batch arbitrarily, including via the adaptive
  MemoryError-triggered retry/halving below.
- **stateful two-pass** (``spec.stateful``) -- fits mergeable global state
  across all fit batches (:mod:`featm.state`), then transforms with that
  frozen state; safe to batch (fit *and* transform) because ``update_state``
  is a genuine merge and ``transform_batch`` with already-frozen state is
  row-independent. Leakage-safe fitting on one source and transforming a
  *different* source (the common train/test split use case) is exposed
  publicly via :mod:`featm.extractor` (:func:`featm.fit`), which reuses the
  exact same fit-phase/transform-phase logic as this module's single-source
  convenience path (:func:`run_extraction`) via :func:`_prepare_run`,
  :func:`_run_fit_phase`, :func:`_run_transform_phase`.
- **not batchable** (``spec.batchable is False``, driven by
  ``requires_full_materialization=True`` -- geometry/spatial/manifold/
  topology/pointcloud/matrix families, and the legacy v0.1.0
  ``incremental=True`` tabular features that recompute state fresh from
  whatever one call receives) -- the feature's mathematics genuinely
  requires the complete dataset at once. The engine either runs it in one
  batch that covers the *entire* source (growing the batch size to do so
  when the resolved memory budget affordably allows), or refuses to run it
  across multiple batches at all. Critically, this also means such a
  feature's MemoryError is never recovered by splitting the chunk -- a
  smaller sub-chunk would compute a mathematically different (wrong) result
  for exactly the same reason chunking it up front would, so a MemoryError
  on a not-batchable feature is isolated/reported as a normal failure
  instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cache import scoped_cache
from .checkpoint import CheckpointIdentity, CheckpointIncompatibleError, CheckpointManager
from .engine import _expand_result, _selected_specs
from .matrix import (
    CheckpointCorruptionError,
    DiskBudgetExceededError,
    DiskFeatureMatrixWriter,
    FeatureMatrix,
    _estimate_bytes,
    validate_disk_backend_integrity,
)
from .registry import REGISTRY, BatchSemantics, FeatureSpec
from .report import ExecutionReport
from .resources import (
    AdaptiveBatchController,
    ESTIMATED_BYTES_PER_CATEGORY,
    ResourceBudget,
    derive_state_budget_bytes,
    checkpoint_state_budget,
    check_checkpoint_state_capacity,
    estimate_full_materialization_bytes,
    estimate_output_width,
    max_rows_within_budget,
    parse_size,
    plan_batch_rows,
)
from .sampling import ReservoirSampler
from .sources import PandasSource, Source, SourceNotReplayableError, canonical_label, resolve_source
from .state import STATEFUL_ENCODERS
from .contracts import validate_params, numeric_series
import copy

DEFAULT_APPROXIMATE_SAMPLE_SIZE = 200_000
DEFAULT_CACHE_BUDGET_BYTES = 256 * 1024 * 1024
#: Floor for the adaptive MemoryError-triggered retry/halving of a *safe*
#: (row-independent, or frozen-state transform) feature's batch. This is a
#: last-resort floor, not a performance target: a feature that is safe to
#: split at all (see _is_retry_split_safe) should be allowed to shrink all
#: the way down to a single row before giving up, rather than declaring
#: failure at an arbitrary larger floor while genuinely smaller allocations
#: might still have succeeded. A not-batchable/legacy-incremental feature is
#: never subject to this floor at all -- it is never split in the first
#: place, regardless of this constant's value.
_MIN_RETRY_ROWS = 1
#: Safety margin applied once, to the *projected final* output size, when
#: deciding whether output="auto"/"memory" may commit to a memory backend:
#: the eventual pd.concat() of every accumulated batch briefly needs roughly
#: the old chunks plus the newly built merged frame at once, so a projected
#: total comfortably under the budget can still spike past it at that final
#: step. This is deliberately a moderate margin (not 2x): the goal is to
#: catch a projected total that is "just below" the budget (would leave no
#: headroom for that spike), not to penalize a genuinely comfortable case
#: (e.g. 60-70% budget utilization) into an unnecessary disk backend.
_MEMORY_ASSEMBLY_SAFETY_MULTIPLIER = 1.3
#: Conservative per-retained-item byte estimate for a reservoir-sampled
#: value (a Python float or short string object in a list), used only to
#: clamp reservoir capacity to a memory_budget-derived state budget -- see
#: Finding 6 in STEP2_AUDIT3_FIX_HANDOFF.md.
_ESTIMATED_BYTES_PER_RESERVOIR_ITEM = ESTIMATED_BYTES_PER_CATEGORY


class ApproximateModeUnavailableError(ValueError):
    pass


class ColumnResolutionError(ValueError):
    pass


class UnsafeMultiBatchFeatureError(ValueError):
    """A feature's mathematics requires more than one batch can safely give it."""


class RowCountConstraintError(ValueError):
    """A feature's declared min_rows/max_rows_exact constraint was violated."""


class UnsafeOutputModeError(ValueError):
    """output='memory' would risk an unbounded/over-budget in-memory accumulation."""


class FeatureRowAlignmentError(RuntimeError):
    """A row-aligned feature's output did not have exactly len(batch) rows in every emitted column.

    Fourth audit pass, Finding 5: the out-of-core FeatureMatrix path is
    fundamentally row-oriented, and every feature that has not been
    explicitly classified as producing non-row-aligned output must produce
    exactly one output row per input row, in every batch. This is checked
    generically, immediately after each feature's per-batch computation --
    independent of (and a backstop for) whatever ``batch_semantics``
    classification the feature happens to have, so a metadata mistake can
    never silently produce a wrong-shaped result undetected.
    """


class FeatureSchemaDriftError(RuntimeError):
    """A row-aligned feature emitted a different set/order of output columns on a later batch.

    Fourth audit pass, Finding 6: the output schema (column names, in
    order) is fixed once, from the first batch a transform phase actually
    computes -- if a later batch's real output columns differ (a classic
    symptom of a feature whose vocabulary/width is silently data-dependent,
    e.g. an unfitted TF-IDF vectorizer), that is a correctness failure, not
    something to paper over with NaN-filling or silently dropping/ignoring
    the new columns.
    """


# --------------------------------------------------------------------------
# Column resolution: which source column(s) feed which feature (arity-aware,
# validated against the schema before any source iteration begins).
# --------------------------------------------------------------------------
def _dtype_matches(input_type: str, dtype_str: str) -> bool:
    d = dtype_str.lower()
    if input_type == "numeric":
        return any(k in d for k in ("int", "float", "double", "decimal", "uint")) and "datetime" not in d
    if input_type == "categorical":
        return any(k in d for k in ("object", "string", "category", "str", "bool"))
    if input_type == "datetime":
        return "datetime" in d or d == "date"
    return False


def _resolve_columns_for_spec(
    spec: FeatureSpec, schema_columns: Sequence[Any], dtype_of, columns_arg: Mapping[str, Any] | None
) -> list[Any]:
    explicit = (columns_arg or {}).get(spec.name)
    if explicit is not None:
        # A single label (str, int, or any other hashable, non-sequence type --
        # column labels are not always strings, see PandasSource) vs. an
        # ordered sequence of labels for arity > 1. list/tuple is "a sequence
        # of labels"; anything else is "one label", never guessed between.
        cols = list(explicit) if isinstance(explicit, (list, tuple)) else [explicit]
        if spec.modality == "tabular" and len(cols) != spec.arity:
            raise ColumnResolutionError(
                f"{spec.name} has arity={spec.arity} but columns={{{spec.name!r}: {cols!r}}} maps "
                f"{len(cols)} column(s). Pass exactly {spec.arity} column name(s), in the order the "
                f"feature expects them."
            )
        schema_set = set(schema_columns)
        unknown = [c for c in cols if c not in schema_set]
        if unknown:
            raise ColumnResolutionError(
                f"columns={{{spec.name!r}: {cols!r}}} names column(s) not present in the source schema: "
                f"{unknown!r}. Available columns: {list(schema_columns)!r}."
            )
        if spec.input_types == ("numeric",) and any(
            not _dtype_matches("numeric", dtype_of(c)) and not any(
                kind in dtype_of(c).lower() for kind in ("object", "string", "category", "str")
            ) for c in cols
        ):
            raise ColumnResolutionError(f"{spec.name} requires numeric columns, got {[dtype_of(c) for c in cols]}")
        return cols
    if spec.modality != "tabular":
        return list(schema_columns)  # whole-batch feature, no explicit projection given
    candidates = [c for c in schema_columns if any(_dtype_matches(t, dtype_of(c)) for t in spec.input_types)]
    if spec.arity > 1:
        # Ambiguous by construction for arity > 1 -- inferring *which* candidate
        # column plays *which* positional role would be a guess, and a wrong
        # guess silently computes a different (wrong) feature. Always require
        # an explicit, ordered mapping.
        raise ColumnResolutionError(
            f"{spec.name} has arity={spec.arity} and needs an explicit, ordered column mapping "
            f"(candidates by dtype: {candidates or 'none found'}). Pass "
            f"columns={{{spec.name!r}: [col_1, ..., col_{spec.arity}]}}."
        )
    if len(candidates) == 1:
        return candidates
    if not candidates:
        raise ColumnResolutionError(
            f"No source column matches input_types={spec.input_types} for feature {spec.name!r}. "
            f"Pass columns={{{spec.name!r}: 'your_column'}}."
        )
    raise ColumnResolutionError(
        f"Multiple source columns match feature {spec.name!r}: {candidates}. "
        f"Pass columns={{{spec.name!r}: 'the_column'}} to disambiguate."
    )


def _ordered_unique_columns(needed: set, schema_order: Sequence[Any]) -> list[Any]:
    """Deterministic column order without ``sorted()`` -- safe across mixed/non-comparable label types.

    Sorting an arbitrary set of column labels (e.g. mixing int and str
    labels, which pandas allows) can raise ``TypeError`` outright, and even
    when it doesn't, alphabetic/numeric sort order has no particular meaning
    here. The source schema's own declared order is already deterministic
    and meaningful, so columns are emitted in that order; any needed column
    that is somehow not part of the schema (should not happen in practice) is
    appended afterward in first-seen order rather than dropped.
    """
    needed_set = set(needed)
    ordered = [c for c in schema_order if c in needed_set]
    seen = set(ordered)
    extra = [c for c in needed if c not in seen]
    return ordered + extra


def _resolve_effective_mode(spec: FeatureSpec, mode: str, estimated_rows: int | None) -> str:
    """Decide exact/approximate for one feature, honoring exact_available/approximate_available/max_rows_exact.

    Applied to *every* selected feature (not just Step 2 stateful ones): a
    feature with no approximate implementation must reject an explicit
    ``mode="approximate"`` request rather than silently running exact, and
    ``mode="exact"`` must never be let past a declared ``max_rows_exact``
    merely because the feature *also* happens to support approximation --
    exact mode never substitutes approximate for any reason.
    """
    if mode == "exact":
        if not spec.exact_available:
            raise ApproximateModeUnavailableError(f"{spec.name} has no exact implementation")
        if spec.max_rows_exact is not None and estimated_rows is not None and estimated_rows > spec.max_rows_exact:
            raise RowCountConstraintError(
                f"{spec.name} declares max_rows_exact={spec.max_rows_exact}; source has {estimated_rows} "
                "rows and mode='exact' was requested, which never substitutes an approximate result "
                "regardless of whether one is available."
            )
        return "exact"
    if mode == "approximate":
        if not spec.approximate_available:
            raise ApproximateModeUnavailableError(f"{spec.name} has no approximate implementation")
        return "approximate"
    if mode != "auto":
        raise ValueError("mode must be 'exact', 'approximate', or 'auto'")
    if (
        spec.approximate_available
        and spec.approximate_max_exact_rows is not None
        and estimated_rows is not None
        and estimated_rows > spec.approximate_max_exact_rows
    ):
        return "approximate"
    if spec.max_rows_exact is not None and estimated_rows is not None and estimated_rows > spec.max_rows_exact:
        if spec.approximate_available:
            return "approximate"
        raise RowCountConstraintError(
            f"{spec.name} declares max_rows_exact={spec.max_rows_exact} and has no approximate mode; "
            f"source has {estimated_rows} rows, which exceeds it."
        )
    return "exact"


def _is_retry_split_safe(spec: FeatureSpec, *, is_stateful_transform: bool) -> bool:
    """Whether a MemoryError on this feature may be recovered by halving the chunk and retrying.

    Splitting is only a mathematically valid recovery when the feature's
    result does not depend on which rows happen to land in which half --
    i.e. the same "batch-safety contract" that governs ordinary chunking
    (see module docstring), not a separate policy. A not-batchable feature
    or a legacy single-shot-fit tabular feature computing over what it was
    told is the *whole* dataset must not be silently handed half of it on a
    MemoryError; a stateful feature's *transform* step with already-frozen
    state is row-independent and remains safe to split.
    """
    if is_stateful_transform:
        return True
    if not spec.batchable:
        return False
    if spec.modality == "tabular" and spec.incremental and not spec.stateful:
        return False
    return True


#: Modalities whose small-data calling convention expects a dense, already-
#: stacked ``(n, ...)`` numpy array of per-row samples (image.*/audio.*/
#: video.*), as opposed to an iterable of per-row Python objects (text.*/
#: graph.*). Only used to decide HOW to adapt an OPAQUE_OBJECT_SAMPLE
#: feature's resolved column into the argument its implementation expects.
_DENSE_ARRAY_SAMPLE_MODALITIES = frozenset({"image", "audio", "video"})


def _adapt_object_column(spec: FeatureSpec, series: pd.Series) -> Any:
    """Adapt one resolved non-numeric Source column into the argument a whole-batch feature expects.

    Fifth audit pass, Finding 1: ``text.*``/``graph.*`` (and anything else
    whose small-data convention already accepts "an iterable of per-row
    Python objects") work correctly with the column's own ``pandas.Series``
    passed straight through -- iterating it yields real per-row values, and
    each feature's own adapter (``_docs``/``_graphs``/...) does the rest.
    ``image.*``/``audio.*``/``video.*`` are different: their small-data
    convention expects one DENSE, already-stacked ``numpy.ndarray`` of shape
    ``(n, ...)`` (``np.asarray(x)`` inside the feature), matching how a
    caller passes a whole in-memory batch directly to ``featm.extract()``.
    A ``pandas.Series`` of per-row ``numpy.ndarray`` objects does NOT
    auto-stack under ``np.asarray()`` -- it produces a 1-D object array, not
    an ``(n, h, w[, c])`` tensor -- so passing it straight through (the
    pre-fix behavior) always failed with a shape error, even though a
    genuinely compatible in-memory ``PandasSource`` object column is exactly
    what ``docs/SCALING.md`` documents as supported for these modalities.
    This stacks real per-row ``numpy.ndarray`` samples into the dense tensor
    those families' implementations actually expect, and fails clearly (not
    by silently coercing/flattening) when the values are not ndarrays, or
    are ndarrays of inconsistent shape.
    """
    if spec.batch_semantics is not BatchSemantics.OPAQUE_OBJECT_SAMPLE or spec.modality not in _DENSE_ARRAY_SAMPLE_MODALITIES:
        return series  # text.*/graph.*/etc.: pass the Series through unchanged
    values = list(series)
    if not values:
        return series  # empty batch: let the feature's own empty-input handling apply
    bad = [v for v in values if not isinstance(v, np.ndarray)]
    if bad:
        raise ColumnResolutionError(
            f"{spec.name} (modality={spec.modality!r}) requires a PandasSource column of pre-loaded "
            "numpy.ndarray samples (one array per row, e.g. one decoded image/audio clip/video clip) -- "
            f"got a value of type {type(bad[0]).__name__!r} instead. CSV/JSONL/Parquet/Arrow sources "
            f"cannot represent {spec.modality} object samples at all (a file path or raw string is not "
            "a loaded array); only an in-memory PandasSource with real pre-loaded ndarray objects is "
            "supported for this modality. See docs/SCALING.md's out-of-core modality-support table."
        )
    shapes = {v.shape for v in values}
    if len(shapes) > 1:
        raise ColumnResolutionError(
            f"{spec.name}: inconsistent per-row array shapes within one batch: {sorted(shapes)}. Every "
            f"sample passed to a {spec.modality} feature in one out-of-core batch must share the same "
            "shape -- this Source column holds ragged/incompatible arrays, which cannot be safely "
            "stacked into one dense batch tensor without silently coercing or flattening data."
        )
    return np.stack(values)


def _apply_direct_feature(
    spec: FeatureSpec, batch: pd.DataFrame, columns: list[Any], params: dict[str, Any],
    *, backend_policy: "dict | None" = None, batch_id: int = 0,
) -> "tuple[dict[str, np.ndarray], Any]":
    """Compute one feature over one batch. Returns ``(expanded_columns, backend_info_or_None)``.

    ``backend_policy`` (Step 3), when not ``None`` and requesting anything
    other than ``"cpu"``, routes GPU-capable specs through
    ``featm.gpu.dispatch.dispatch_one_feature`` instead of calling
    ``spec.implementation`` directly -- every other feature (the overwhelming
    majority not selected for GPU acceleration this milestone, and every
    tabular/stateful feature) is completely unaffected, still calling
    ``fn(...)`` exactly as before Step 3.
    """
    if (
        backend_policy is not None
        and backend_policy.get("backend", "cpu") != "cpu"
        and spec.modality != "tabular"
    ):
        from .gpu.dispatch import gpu_capable

        if backend_policy.get("backend") == "gpu" or gpu_capable(spec):
            from .gpu.dispatch import dispatch_one_feature

            view = batch[columns] if columns else batch
            arg = view.to_numpy(dtype=float) if all(pd.api.types.is_numeric_dtype(view[c]) for c in view.columns) else _adapt_object_column(spec, view.iloc[:, 0])
            result, info = dispatch_one_feature(
                spec, arg, backend=backend_policy["backend"], gpu_fallback=backend_policy.get("gpu_fallback", "fallback"),
                device=backend_policy.get("device"), gpu_memory_budget=backend_policy.get("gpu_memory_budget"),
                params=params, batch_id=batch_id,
            )
            return _expand_result(spec.name, result), info

    fn = spec.implementation
    if spec.modality == "tabular":
        # arity-aware: pass one positional Series argument per resolved column,
        # matching how e.g. tabular.interactions.product(a, b, ...) is written.
        series_args = [batch[c] for c in columns]
        if spec.input_types == ("numeric",):
            series_args = [numeric_series(arg) for arg in series_args]
        result = fn(*series_args, **params)
        if isinstance(result, tuple) and len(result) == 2:
            result = result[0]
    else:
        view = batch[columns] if columns else batch
        all_numeric = view.shape[1] > 0 and all(pd.api.types.is_numeric_dtype(view[c]) for c in view.columns)
        if all_numeric:
            arg = view.to_numpy(dtype=float)
        elif view.shape[1] == 1:
            # A whole-batch feature whose per-row samples are non-numeric
            # objects (text.*, or an in-memory PandasSource with a
            # pre-loaded-object column such as graph.*/image.*/audio.*/
            # video.* -- see docs/SCALING.md's out-of-core modality-support
            # table) expects an *iterable of per-row values*, matching the
            # small-data extract() calling convention. Passing the raw
            # single-column DataFrame instead of its Series (the pre-fix
            # behavior) is a real correctness bug, not just a style
            # difference: iterating a DataFrame yields its COLUMN LABELS, not
            # row values, so every batch would silently be treated as "one
            # sample" (the column name) regardless of how many rows it
            # actually held -- e.g. a 100-row text column produced a
            # 2-row (one per batch) output computed from the literal string
            # "column_name", not the real text data. Using the column's own
            # Series here iterates real per-row values, matching what
            # text.*/graph.*/etc.'s own small-data adapters
            # (`_docs`/`_graphs`/...) already handle correctly for a Series
            # or list.
            arg = _adapt_object_column(spec, view.iloc[:, 0])
        else:
            raise ColumnResolutionError(
                f"{spec.name} (modality={spec.modality!r}) resolved to {view.shape[1]} non-numeric "
                f"columns ({list(view.columns)!r}) for one out-of-core batch. A whole-batch feature "
                "over non-numeric data expects one column of per-row sample objects (e.g. one string "
                "per row for a text.* feature); multiple non-numeric columns at once is ambiguous "
                f"here. Pass columns={{{spec.name!r}: 'the_single_column'}} to disambiguate, or select "
                "this feature via featm.extract() on in-memory data if it genuinely needs multiple "
                "non-numeric inputs at once."
            )
        result = fn(arg, **params)
    return _expand_result(spec.name, result), None


#: batch_semantics classifications whose output IS expected to be row-aligned
#: with the input batch -- i.e. exactly one output row per input row. A
#: FULL_DATASET feature's single grown-to-whole-source batch legitimately
#: produces a dataset-level summary (often far fewer rows than the input --
#: e.g. one row of aggregate Delaunay-triangulation statistics for a whole
#: point cloud), which is not a bug to catch, it is the entire point of that
#: classification; STRUCTURED_SAMPLE features never reach this check at all
#: (rejected before computation). Everything else claiming batchability is
#: making an implicit row-alignment promise that Finding 5 exists to verify.
_ROW_ALIGNED_BATCH_SEMANTICS = (BatchSemantics.ROW_INDEPENDENT, BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED)


def _validate_row_alignment(spec: FeatureSpec, expanded: dict[str, np.ndarray], expected_n: int) -> None:
    """Finding 5 (fourth audit pass): a row-aligned feature's output must have exactly
    ``expected_n`` rows in every emitted column -- checked generically, immediately after
    computation. Every feature's own emitted columns must also agree with EACH OTHER on
    row count, regardless of batch_semantics (a feature that emits inconsistent-length
    columns is broken regardless of whether its output is meant to be row-aligned).

    This is a backstop, not a substitute for correct metadata: it exists specifically so a
    metadata mistake (a feature that should have been STRUCTURED_SAMPLE but wasn't) still
    cannot silently produce a wrong-shaped, silently-accepted result -- it fails loudly
    instead, isolated per errors= like any other feature-level failure.
    """
    lengths = {k: len(np.asarray(v)) for k, v in expanded.items()}
    if len(set(lengths.values())) > 1:
        raise FeatureRowAlignmentError(
            f"{spec.name} emitted output columns with inconsistent row counts within the same call: "
            f"{lengths}. Every column one feature emits for one call must have the same row count."
        )
    if spec.batch_semantics not in _ROW_ALIGNED_BATCH_SEMANTICS:
        return
    bad = {k: n for k, n in lengths.items() if n != expected_n}
    if bad:
        raise FeatureRowAlignmentError(
            f"{spec.name} produced output with {bad} row(s) for a batch of {expected_n} input row(s) "
            f"(columns: {list(bad)}). A row-aligned feature must emit exactly one output row per input "
            "row in every batch; this is very likely a feature whose implementation implicitly depends on "
            "the whole batch as one collection (e.g. it silently treated a 2-row batch as a structured pair) "
            "rather than genuinely computing one independent value per row. If this feature's output is "
            "intentionally not row-aligned (a genuine whole-dataset summary), it must be classified "
            "requires_full_materialization=True / batch_semantics=FULL_DATASET in the registry, not left to "
            "silently pass through the row-aligned out-of-core path."
        )


def _compute_transform_chunk(
    spec: FeatureSpec, chunk: pd.DataFrame, resolved_columns: dict[str, list[Any]], params_for_spec: dict[str, Any],
    fitted_states: dict[str, Any], *, backend_policy: "dict | None" = None, batch_id: int = 0,
) -> "tuple[dict[str, np.ndarray], Any]":
    unsupported_info = None
    if backend_policy is not None and backend_policy.get("backend", "cpu") != "cpu":
        from .gpu.dispatch import BackendInfo, gpu_capable
        from .gpu.exceptions import GPUImplementationMissingError
        if not gpu_capable(spec):
            requested = backend_policy["backend"]
            if requested == "gpu" and backend_policy.get("gpu_fallback") == "strict":
                raise GPUImplementationMissingError(f"{spec.name} has no GPU implementation; strict GPU request rejected")
            unsupported_info = BackendInfo(requested_backend=requested, actual_backend="cpu",
                cpu_fallback=requested == "gpu", fallback_reason="feature_not_gpu_capable")
            backend_policy = {**backend_policy, "backend": "cpu"}
    if spec.name in fitted_states:
        # Step 3 v1: stateful transform is not GPU-accelerated (no
        # GPU-capable feature in this milestone is also stateful --
        # registry.validate_spec rejects that combination anyway, see
        # "Step 3: GPU metadata coherence"). backend_info stays None here.
        col = resolved_columns[spec.name][0]
        result = STATEFUL_ENCODERS[spec.name].transform_batch(chunk[col], fitted_states[spec.name], params=params_for_spec)
        expanded = _expand_result(spec.name, result)
        info = None
    else:
        expanded, info = _apply_direct_feature(spec, chunk, resolved_columns[spec.name], params_for_spec, backend_policy=backend_policy, batch_id=batch_id)
    _validate_row_alignment(spec, expanded, len(chunk))
    return expanded, unsupported_info if unsupported_info is not None else info


def _compute_transform_with_retry(
    spec: FeatureSpec,
    chunk: pd.DataFrame,
    resolved_columns: dict[str, list[Any]],
    params_for_spec: dict[str, Any],
    fitted_states: dict[str, Any],
    report: ExecutionReport,
    min_rows: int = _MIN_RETRY_ROWS,
    *,
    backend_policy: "dict | None" = None,
    batch_id: int = 0,
) -> dict[str, np.ndarray]:
    """Run one feature on one batch; on MemoryError, halve the batch and retry -- only when safe.

    Recognized memory-allocation failures (``MemoryError`` and numpy's
    ``_ArrayMemoryError``, which subclasses it) are only recoverable by
    splitting for a feature where :func:`_is_retry_split_safe` says so; the
    chunk is split in half and each half retried independently, so a
    transient allocation failure on a large batch does not have to sacrifice
    the whole batch. Halving is bounded by ``min_rows``, so this can never
    loop forever. For a feature where splitting is *not* safe (its
    mathematics depends on the whole chunk, e.g. a Delaunay triangulation),
    the MemoryError is re-raised immediately with no split attempted at all
    -- recovering by splitting would silently compute a chunk-local answer
    and return it as if it were the whole-dataset result, exactly the
    correctness bug the batch-safety contract exists to prevent.
    """
    try:
        expanded, info = _compute_transform_chunk(spec, chunk, resolved_columns, params_for_spec, fitted_states, backend_policy=backend_policy, batch_id=batch_id)
        if info is not None:
            report.record_backend_info(spec.name, info.as_dict())
        return expanded
    except MemoryError:
        if not _is_retry_split_safe(spec, is_stateful_transform=spec.name in fitted_states):
            raise
        report.record_retry(spec.name)
        if len(chunk) <= min_rows:
            raise
        report.record_batch_reduction(spec.name)
        mid = len(chunk) // 2
        left = _compute_transform_with_retry(spec, chunk.iloc[:mid], resolved_columns, params_for_spec, fitted_states, report, min_rows, backend_policy=backend_policy, batch_id=batch_id)
        right = _compute_transform_with_retry(spec, chunk.iloc[mid:], resolved_columns, params_for_spec, fitted_states, report, min_rows, backend_policy=backend_policy, batch_id=batch_id)
        return {k: np.concatenate([np.asarray(left[k]), np.asarray(right[k])]) for k in left}


def _update_state_with_retry(
    spec: FeatureSpec,
    state: Any,
    series: pd.Series,
    report: ExecutionReport,
    min_rows: int = _MIN_RETRY_ROWS,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fit-phase counterpart of :func:`_compute_transform_with_retry`.

    Always safe to split: ``update_state`` is a genuine associative/
    commutative merge (see :mod:`featm.state`), so calling it on two halves
    sequentially is mathematically identical to calling it once on the
    whole chunk -- this is the entire point of the stateful two-pass
    protocol, independent of the batch-safety contract that governs
    ordinary (non-stateful) features.
    """
    encoder = STATEFUL_ENCODERS[spec.name]
    try:
        candidate = copy.deepcopy(state)
        return encoder.update_state(candidate, series, params=params)
    except MemoryError:
        report.record_retry(spec.name)
        if len(series) <= min_rows:
            raise
        report.record_batch_reduction(spec.name)
        mid = len(series) // 2
        state = _update_state_with_retry(spec, state, series.iloc[:mid], report, min_rows, params)
        state = _update_state_with_retry(spec, state, series.iloc[mid:], report, min_rows, params)
        return state


def _real_bytes_per_row(out_frame: dict[str, np.ndarray]) -> int:
    return sum(int(np.asarray(v).dtype.itemsize) for v in out_frame.values()) or 8


def _dataframe_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def _io_backend_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _default_output_dir() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="featm-output-"))


def _materialize_checkpoint_result_to_memory(
    matrix: FeatureMatrix, *, budget: ResourceBudget, allow_memory_budget_override: bool, report: ExecutionReport
) -> FeatureMatrix:
    """Convert a checkpoint's persisted disk result to memory for an ``output="memory"`` caller.

    Fourth audit pass, Finding 9: this conversion previously called
    ``matrix.to_pandas(allow_large=True)`` unconditionally, on both the
    first successful checkpointed run and the zero-read ``done`` fast path
    -- bypassing Finding 8's (third audit pass) contract that an explicit
    ``output="memory"`` respects ``memory_budget`` unless the caller passes
    ``allow_memory_budget_override=True``. Uses the checkpoint's actual
    finalized shape/dtypes (known without materializing anything, from the
    disk-backed matrix's own manifest) rather than a pre-run guess, with the
    same one-time assembly-overhead margin applied elsewhere in this module.
    """
    n_rows, _ = matrix.shape
    est_bytes = _estimate_bytes(n_rows, matrix.dtypes)
    projected = est_bytes * _MEMORY_ASSEMBLY_SAFETY_MULTIPLIER
    if projected > budget.memory_bytes:
        if not allow_memory_budget_override:
            raise UnsafeOutputModeError(
                f"output='memory': materializing the checkpointed disk result (~{est_bytes:,} bytes, "
                f"{n_rows:,} rows x {len(matrix.columns)} columns) would exceed the memory_budget "
                f"({budget.memory_bytes:,} bytes). memory_budget is a hard planner envelope by default; pass "
                "output='disk' to keep the result on disk, increase memory_budget, or pass "
                "allow_memory_budget_override=True to explicitly accept an over-budget in-memory result."
            )
        report.warnings.append(
            f"output='memory': materializing the checkpointed disk result (~{est_bytes:,} bytes) exceeds "
            f"memory_budget ({budget.memory_bytes:,} bytes); proceeding because "
            "allow_memory_budget_override=True was passed."
        )
    return FeatureMatrix.from_pandas(matrix.to_pandas(allow_large=True))


# ==========================================================================
# Shared "prepare" step: everything deterministic and source-schema-dependent
# that both run_extraction() (single-source fit+transform convenience path)
# and featm.extractor's public fit()/FittedFeatureExtractor.transform() (two-
# source leakage-safe path) need, resolved fresh against whichever source is
# being prepared -- a training source and a later, different transform
# source each get their own independent _prepare_run() call, so e.g. column
# order or an unrelated min_rows constraint difference between the two
# sources is never silently assumed to match.
# ==========================================================================
@dataclass
class PreparedRun:
    src: Source
    schema_names: list[Any]
    budget: ResourceBudget
    resolved_columns: dict[str, list[Any]]
    active_specs: list[FeatureSpec]
    effective_mode_by_spec: dict[str, str]
    resolved_batch_rows: int
    stateful_specs: list[FeatureSpec]
    direct_specs: list[FeatureSpec]
    has_fit_phase: bool
    cache_bytes: int
    report: ExecutionReport
    #: Real, source-reported bytes/row used to plan resolved_batch_rows
    #: (Finding 10, fourth audit pass), or None if the source could not
    #: cheaply estimate this and the itemsize-per-column fallback was used.
    #: Kept for the execution-time observability check below.
    planned_bytes_per_row_in: int | None = None
    checkpoint_state_budget_bytes: int | None = None
    #: Step 3: GPU backend policy for this run, threaded through to the
    #: per-batch transform loop's dispatch_one_feature() calls. "cpu"
    #: (default) never touches this run's GPU dispatch code path at all --
    #: identical to every pre-Step-3 out-of-core run.
    backend: str = "cpu"
    gpu_fallback: str = "fallback"
    device: "str | int | None" = None
    gpu_memory_budget: "str | int | None" = None


def _prepare_run(
    source: Any,
    *,
    features: Sequence[str],
    modality: str | None,
    columns: Mapping[str, Any] | None,
    mode: str,
    errors: str,
    memory_budget: "str | int | None",
    disk_budget: "str | int | None",
    batch_rows: int | None,
    cache_budget: "str | int | None",
    params: Mapping[str, Mapping[str, Any]] | None = None,
    backend: str = "cpu",
    gpu_fallback: str = "fallback",
    device: "str | int | None" = None,
    gpu_memory_budget: "str | int | None" = None,
) -> PreparedRun:
    if errors not in {"skip", "warn", "raise"}:
        raise ValueError("errors must be skip, warn, or raise")
    if mode not in {"exact", "approximate", "auto"}:
        raise ValueError("mode must be exact, approximate, or auto")
    if backend not in ("cpu", "gpu", "auto"):
        raise ValueError("backend must be cpu, gpu, or auto")
    if gpu_fallback not in ("fallback", "strict"):
        raise ValueError("gpu_fallback must be fallback or strict")

    # Reuse the exact same feature-selection validation the small-data path
    # uses (empty selection, features="all", REGISTRY lookups) so the two
    # entry points cannot silently diverge on what counts as a valid request.
    specs = _selected_specs(features)
    extra_params = set(params or {}) - {s.name for s in specs}
    if extra_params:
        raise ValueError(f"Parameters supplied for unselected features: {sorted(extra_params)}")
    for spec in specs:
        validate_params(spec, (params or {}).get(spec.name, {}))
    if modality is not None:
        wrong = [s.name for s in specs if s.modality != modality]
        if wrong:
            raise ValueError(f"Selected features do not belong to modality={modality}: {wrong}")

    src: Source = resolve_source(source)
    schema = src.schema()
    schema_names = schema.names()
    budget = ResourceBudget.resolve(memory_budget=memory_budget, disk_budget=disk_budget)

    report = ExecutionReport(
        requested_features=list(features),
        memory_budget_bytes=budget.memory_bytes,
        disk_budget_bytes=budget.disk_bytes,
        mode=mode,
        errors_policy=errors,
    )

    # Cache budget must not silently coexist with (and potentially dwarf) a tiny
    # declared working-memory budget: default to a fraction of it, and cap an
    # explicit override at the overall memory budget -- with NO minimum floor
    # above that cap, so an intentionally tiny memory_budget cannot be
    # silently exceeded by cache retention alone. A very small effective cache
    # budget simply means most/all intermediates end up "too large to retain"
    # (see featm.cache's oversized-value handling) rather than being rejected
    # outright -- the run still completes correctly, just without caching.
    cache_bytes_requested = parse_size(cache_budget)
    if cache_bytes_requested is None:
        cache_bytes = min(DEFAULT_CACHE_BUDGET_BYTES, int(budget.memory_bytes * 0.25))
    else:
        cache_bytes = cache_bytes_requested
    cache_bytes = max(min(cache_bytes, budget.memory_bytes), 1)
    if cache_bytes_requested is not None and cache_bytes != cache_bytes_requested:
        report.warnings.append(
            f"cache_budget={cache_bytes_requested} exceeds the overall memory_budget; "
            f"clamped to {cache_bytes} bytes."
        )

    # --- resolve one input column (or the whole batch) per selected feature,
    # validated against the schema now -- before any source iteration, so a
    # bad explicit column name is isolated to its one feature under
    # errors="skip"/"warn" instead of crashing the whole run. ---------------
    resolved_columns: dict[str, list[Any]] = {}
    for spec in specs:
        try:
            resolved_columns[spec.name] = _resolve_columns_for_spec(spec, schema_names, schema.dtype_of, columns)
        except ValueError as exc:
            if errors == "raise":
                raise
            report.record_skipped(spec.name, str(exc))
    active_specs = [s for s in specs if s.name in resolved_columns]

    # --- enforce min_rows and resolve exact/approximate mode (incl.
    # max_rows_exact) for every selected feature, not only stateful ones ----
    effective_mode_by_spec: dict[str, str] = {}
    checked_specs: list[FeatureSpec] = []
    for spec in active_specs:
        if src.n_rows is not None and spec.min_rows > 1 and src.n_rows < spec.min_rows:
            msg = f"{spec.name} requires at least min_rows={spec.min_rows} rows; source has {src.n_rows}."
            if errors == "raise":
                raise RowCountConstraintError(msg)
            report.record_skipped(spec.name, msg)
            continue
        try:
            effective_mode_by_spec[spec.name] = _resolve_effective_mode(spec, mode, src.n_rows)
        except (ApproximateModeUnavailableError, RowCountConstraintError) as exc:
            if errors == "raise":
                raise
            report.record_skipped(spec.name, str(exc))
            continue
        checked_specs.append(spec)
    active_specs = checked_specs

    # --- batch-size planning, including growing to whole-dataset for FULL_DATASET features ---
    # Finding 10 (fourth audit pass): a real, source-reported deep-memory
    # estimate for the columns this run will actually project, when the
    # source can cheaply provide one (bounded sample / metadata, never a
    # full-dataset read) -- replaces the itemsize-per-column guess, which
    # silently undercounts variable-length strings/objects by orders of
    # magnitude.
    planning_columns = _ordered_unique_columns({c for spec in active_specs for c in resolved_columns[spec.name]}, schema_names)
    try:
        real_bytes_per_row_in = src.estimated_bytes_per_row(planning_columns)
    except Exception:
        real_bytes_per_row_in = None  # best-effort; the itemsize fallback remains correct, just coarser
    resolved_batch_rows = plan_batch_rows(
        schema_n_columns=len(schema_names), specs=active_specs, budget=budget, override=batch_rows,
        bytes_per_row_in=real_bytes_per_row_in,
    )
    needs_whole_dataset = [s for s in active_specs if not s.batchable]
    # Finding 11 (audit 2): per-feature working-memory estimate (curated
    # estimator, or a conservative complexity-derived heuristic for
    # quadratic/cubic features) must gate whole-dataset approval PER SPEC --
    # not just whether to grow the batch size, but whether a batch that
    # already happens to be large enough (e.g. the ordinary per-row-bytes
    # planner already picked a size >= n_rows because the raw table is
    # narrow) is actually safe for THIS feature. A raw-row-bytes-only check
    # would wrongly approve a 100,000 x 2 point table (a few MB raw) for a
    # feature that internally builds an O(n^2) dense distance matrix (tens
    # of GB).
    memory_unsafe_specs: set[str] = set()
    if needs_whole_dataset and src.n_rows is not None:
        for s in needs_whole_dataset:
            est = estimate_full_materialization_bytes(s, src.n_rows, len(schema_names))
            if est > budget.memory_bytes:
                memory_unsafe_specs.add(s.name)
                report.warnings.append(
                    f"{s.name}: estimated working memory for the whole dataset is ~{est:,} bytes "
                    f"(complexity={s.complexity.value}), which exceeds the memory_budget of "
                    f"{budget.memory_bytes:,} bytes; not auto-approved for whole-dataset execution."
                )
    growable_specs = [s for s in needs_whole_dataset if s.name not in memory_unsafe_specs]
    if growable_specs and batch_rows is None and src.n_rows is not None:
        n_out = estimate_output_width(active_specs)
        affordable_rows = max_rows_within_budget(n_columns_in=len(schema_names), n_columns_out=n_out, budget=budget, bytes_per_row_in=real_bytes_per_row_in)
        if src.n_rows <= affordable_rows:
            # The whole dataset fits the declared memory budget (for every
            # not-batchable feature whose own working-memory estimate also
            # fits, per-spec, above): grow the batch for THIS run to cover
            # it, so FULL_DATASET features get the mathematically correct
            # whole-input answer out-of-core instead of being rejected. This
            # does mean every feature in this call shares that one large
            # batch -- select FULL_DATASET features in a separate call if
            # fine-grained streaming for the others matters more than this
            # convenience.
            resolved_batch_rows = max(resolved_batch_rows, src.n_rows)
    report.batch_rows = resolved_batch_rows

    # --- guard: features whose mathematics needs more than one batch can safely give them ---
    single_batch_safe = src.n_rows is not None and src.n_rows <= resolved_batch_rows
    params_map_for_guard: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (params or {}).items()}
    safe_specs: list[FeatureSpec] = []
    for spec in active_specs:
        unsafe_reason = None
        if spec.batch_semantics is BatchSemantics.STRUCTURED_SAMPLE:
            # Finding 4 (fourth audit pass): a nested per-sample structure
            # (e.g. a (2, t) cross-series pair, a {"image":..., "text":...}
            # embedding dict) that a row-oriented Source batch cannot
            # represent AT ALL, regardless of batch size -- growing the
            # batch to cover the whole dataset does not help the way it does
            # for a genuine FULL_DATASET feature, because the source rows
            # still are not independent structured samples. Rejected
            # unconditionally, not just when multi-batch.
            unsafe_reason = (
                f"{spec.name} expects a structured per-sample input (e.g. a paired/triple series, an "
                "embedding-pair dict, or a (n_datasets, n_observations, n_variables) array) that a "
                "row-oriented Source cannot represent -- each Source row is not one independent "
                "structured sample, and growing the batch does not change that. This feature is not "
                "supported via the out-of-core engine at any batch size; select it via featm.extract() "
                "on in-memory data, passing the correctly-shaped structure directly."
            )
        elif spec.batch_semantics is BatchSemantics.OPAQUE_OBJECT_SAMPLE and not isinstance(src, PandasSource):
            # Fifth audit pass, Finding 1: an OPAQUE_OBJECT_SAMPLE feature
            # (image.*/audio.*/video.*/graph.*) needs one Source row to BE a
            # real pre-loaded Python object (a numpy.ndarray sample, a
            # networkx.Graph) -- something only an in-memory PandasSource can
            # represent. CSV/JSONL/Parquet/Arrow cannot deserialize an
            # arbitrary Python object into a cell at all; this is a
            # structural Source-type mismatch, not a batch-partitioning
            # concern, so it is rejected unconditionally (even when the
            # whole source would fit in one batch) rather than only once
            # multi-batch splitting is attempted.
            unsafe_reason = (
                f"{spec.name} (modality={spec.modality!r}) requires a real pre-loaded Python object per "
                f"row (e.g. a decoded numpy.ndarray sample, or a networkx.Graph) -- only an in-memory "
                f"PandasSource can represent this; {type(src).__name__} cannot. File-backed sources "
                "(CSV/JSONL/Parquet/Arrow) do not support arbitrary Python object columns for this "
                "modality. Pass an in-memory pandas.DataFrame with the objects already loaded, or select "
                "this feature via featm.extract() on in-memory data directly."
            )
        elif spec.required_params_for_batching and not single_batch_safe:
            # Finding 2/3 (fourth audit pass): a feature whose *default*
            # (param-absent) behavior implicitly depends on "whichever
            # sample the current call happens to see first/last" -- e.g. an
            # implicit reference sequence or reference time -- is genuinely
            # row-independent and safe to batch once that ambiguity is
            # pinned down by an explicit param, but silently unsafe (batch-
            # local defaults) otherwise. A single-batch run is unaffected: a
            # batch-local default there already equals the whole-dataset
            # default, matching small-data extract() behavior exactly.
            call_params = params_map_for_guard.get(spec.name, {})
            missing = [
                p for p in spec.required_params_for_batching
                if call_params.get(p, spec.default_params.get(p)) is None
            ]
            if missing:
                unsafe_reason = (
                    f"{spec.name} requires explicit param(s) {missing} for safe out-of-core multi-batch "
                    "execution: its default (param-absent) value is implicitly chosen from whatever the "
                    "current call happens to see (e.g. the first row of the first batch), which differs "
                    "per batch and would silently change the result depending on batch_rows. Pass "
                    f"params={{{spec.name!r}: {{{missing[0]!r}: ...}}}} with an explicit value, or select "
                    "this feature via featm.extract() on in-memory data to use its single-call default."
                )
        if unsafe_reason:
            if errors == "raise":
                raise UnsafeMultiBatchFeatureError(unsafe_reason)
            report.record_skipped(spec.name, unsafe_reason)
            continue
        if spec.modality == "tabular" and spec.incremental and not spec.stateful and not single_batch_safe:
            unsafe_reason = (
                f"{spec.name} uses the legacy single-shot fit/state convention (from v0.1.0) and "
                "cannot be safely fit across multiple out-of-core batches: each batch would compute "
                "its own local statistics instead of one fit over the whole training set. Use "
                "featm.extract() on in-memory data, featm.AutoFeatures, or select a Step 2 stateful "
                "feature (tabular.scaling.zscore, tabular.scaling.minmax, "
                "tabular.encoding.category_frequency) for out-of-core scaling/encoding."
            )
        elif not spec.batchable and spec.name in memory_unsafe_specs:
            unsafe_reason = (
                f"{spec.name} requires the complete dataset at once, and its estimated working memory for "
                f"the whole dataset exceeds the memory_budget (see the resource-planning warning above for "
                "the exact estimate/complexity used). Running it anyway could exhaust available memory "
                "well beyond what the declared budget allows, so it is rejected rather than approved on raw "
                "input-row-bytes alone. Increase memory_budget, or select this feature separately via "
                "featm.extract() on in-memory data if you accept the resource risk."
            )
        elif not spec.batchable and not single_batch_safe:
            unsafe_reason = (
                f"{spec.name} requires the complete dataset at once (requires_full_materialization=True) "
                "and the source does not fit in one batch under the current memory_budget/batch_rows "
                "(row count "
                + (f"is {src.n_rows}" if src.n_rows is not None else "is unknown")
                + f", resolved batch size is {resolved_batch_rows}). Chunking it would silently compute a "
                "mathematically different result (e.g. a Delaunay triangulation of a row-subset instead of "
                "the whole point set), so this feature is rejected rather than run incorrectly. Increase "
                "memory_budget/batch_rows to cover the whole source, or select this feature separately via "
                "featm.extract() on in-memory data."
            )
        if unsafe_reason:
            if errors == "raise":
                raise UnsafeMultiBatchFeatureError(unsafe_reason)
            report.record_skipped(spec.name, unsafe_reason)
            continue
        safe_specs.append(spec)
    active_specs = safe_specs

    stateful_specs = [s for s in active_specs if s.stateful and s.name in STATEFUL_ENCODERS]
    direct_specs = [s for s in active_specs if s not in stateful_specs]
    has_fit_phase = bool(stateful_specs)

    return PreparedRun(
        src=src,
        schema_names=schema_names,
        budget=budget,
        resolved_columns=resolved_columns,
        active_specs=active_specs,
        effective_mode_by_spec=effective_mode_by_spec,
        resolved_batch_rows=resolved_batch_rows,
        stateful_specs=stateful_specs,
        direct_specs=direct_specs,
        has_fit_phase=has_fit_phase,
        cache_bytes=cache_bytes,
        report=report,
        planned_bytes_per_row_in=real_bytes_per_row_in,
        backend=backend,
        gpu_fallback=gpu_fallback,
        device=device,
        gpu_memory_budget=gpu_memory_budget,
    )


def _resolve_output_backend(
    prepared: PreparedRun,
    *,
    output: str,
    checkpoint_mgr: "CheckpointManager | None",
    disk_budget: "str | int | None",
) -> tuple[bool, bool, int | None]:
    """Decide memory- vs disk-backed output. Returns (use_disk, io_backend_available, disk_budget_bytes).

    Checkpointed runs always use disk for the *persisted* result (Finding 1 /
    Finding 2 in the third audit pass): a checkpoint must be able to support
    a zero-read resume regardless of the requested return backend, which is
    only possible if something was actually written to durable storage on
    the first run. ``output="memory"`` still controls what is *returned* to
    the caller -- materialized from the persisted disk result after it is
    finalized and validated -- never what gets persisted.
    """
    src = prepared.src
    report = prepared.report
    budget = prepared.budget
    io_backend_available = _io_backend_available()
    use_disk = output == "disk" or checkpoint_mgr is not None
    if output == "auto" and not use_disk:
        if src.n_rows is not None:
            est_bytes = src.n_rows * estimate_output_width(prepared.active_specs) * 8
            use_disk = est_bytes > budget.memory_bytes
        else:
            # Row count is unknown (typical for CSV/JSONL): a memory-backed run
            # would accumulate every batch's output with no way to know ahead of
            # time whether it fits. Defaulting to memory here is exactly the
            # unbounded-growth risk this engine exists to avoid, so default to
            # disk whenever the io backend is available.
            use_disk = io_backend_available
    if output == "memory" and src.n_rows is None and not use_disk:
        # Fifth audit pass, Finding 7: memory_budget is NOT bypassed here --
        # it is still enforced, just per-batch as accumulated output actually
        # grows (see the continuous safety-net checks in
        # _run_transform_phase) instead of pre-computed from a known row
        # count up front (impossible when the row count itself is unknown).
        # The pre-fix wording of this warning ("no upper bound... intentional
        # override") was misleading -- it read as though budget enforcement
        # were disabled entirely, when it is only deferred, and still raises
        # UnsafeOutputModeError once the actual accumulated size exceeds
        # memory_budget unless allow_memory_budget_override=True is passed.
        report.warnings.append(
            "output='memory' was explicitly requested with an unknown source row count (e.g. CSV/JSONL): "
            "the pre-run estimate that would normally catch an over-budget result before scanning any "
            "batches cannot run (there is no row count to estimate from), so memory_budget is instead "
            "enforced per-batch as real output actually accumulates -- this run will still raise "
            "UnsafeOutputModeError partway through if accumulated output exceeds memory_budget, unless "
            "allow_memory_budget_override=True is also passed."
        )
    if output == "auto" and not use_disk and src.n_rows is None and not io_backend_available:
        raise UnsafeOutputModeError(
            "output='auto' cannot safely choose a backend: the source's row count is unknown "
            "(so a memory-backed run has no pre-run bound on how much output it will accumulate) and no "
            "disk-backed IO backend is available (install the optional 'io' extra: "
            "pip install 'featm[io]'). Pass output='memory' explicitly to proceed with memory_budget "
            "enforced per-batch as output actually accumulates instead of pre-checked up front (it will "
            "still raise UnsafeOutputModeError if that budget is exceeded, unless "
            "allow_memory_budget_override=True is also passed), or install featm[io] to allow safe "
            "disk-backed output instead."
        )
    report.output_backend = "disk" if use_disk else "memory"

    # Once disk is required/forced (including whenever checkpointing), verify
    # the backend is actually available NOW, before any batch scanning (fit
    # or transform) begins -- not after an entire fit-phase scan, discovered
    # only when the writer is finally constructed in the transform phase.
    if use_disk and not io_backend_available:
        reason = "checkpointing requires the disk backend to support zero-read resume" if checkpoint_mgr is not None else (
            "output='disk', or output='auto' resolved to disk"
        )
        raise ImportError(
            f"Disk-backed output is required for this run ({reason}) but the optional 'io' extra is not "
            "installed: pip install 'featm[io]'. Failing before scanning any data rather than after an "
            "expensive fit/transform pass."
        )

    # --- disk budget: preflight estimate when possible; always enforced at runtime by the writer ---
    disk_budget_bytes = budget.disk_bytes if disk_budget is not None else None
    if use_disk and disk_budget_bytes is not None and src.n_rows is not None:
        est_out_bytes = src.n_rows * estimate_output_width(prepared.active_specs) * 8
        if est_out_bytes > disk_budget_bytes:
            report.warnings.append(
                f"Preflight estimate: output is estimated at ~{est_out_bytes:,} bytes, which exceeds "
                f"the disk_budget of {disk_budget_bytes:,} bytes. The run will proceed and fail "
                "deterministically once the actual budget is exhausted, rather than being blocked on "
                "an estimate alone (compression and vector-output width are only known once running)."
            )
    if use_disk and disk_budget_bytes is not None:
        try:
            import shutil

            free_bytes = shutil.disk_usage(
                (checkpoint_mgr.output_dir if checkpoint_mgr is not None else Path.cwd())
            ).free
            if disk_budget_bytes > free_bytes:
                report.warnings.append(
                    f"disk_budget={disk_budget_bytes:,} exceeds actual free disk space "
                    f"({free_bytes:,} bytes); effective budget clamped to free space."
                )
                disk_budget_bytes = free_bytes
        except OSError:
            pass  # best-effort; runtime accounting in the writer still enforces the (unclamped) budget

    return use_disk, io_backend_available, disk_budget_bytes


# ==========================================================================
# Fit phase: leakage-safe two-pass stateful fitting, shared by run_extraction
# (single-source convenience path) and featm.extractor.fit() (public
# train/test API).
# ==========================================================================
def _run_fit_phase(
    prepared: PreparedRun,
    *,
    errors: str,
    approximate_sample_size: int,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoint_mgr: "CheckpointManager | None" = None,
    manifest: "dict[str, Any] | None" = None,
) -> tuple[dict[str, Any], set[str]]:
    """Run the fit phase; returns (fitted_states, failed_stateful).

    Only actually scans the source if the fit phase has not already
    completed in a prior (checkpointed) run -- a checkpoint already past
    "fit" loads its finalized state without rescanning batches just to skip
    them. With no checkpoint (``checkpoint_mgr=None``, the public
    ``featm.extractor`` API's usage), always runs a fresh fit scan.
    """
    src = prepared.src
    report = prepared.report
    resolved_columns = prepared.resolved_columns
    stateful_specs = prepared.stateful_specs
    effective_mode_by_spec = prepared.effective_mode_by_spec
    state_budget_bytes = prepared.checkpoint_state_budget_bytes
    if state_budget_bytes is None:
        state_budget_bytes = derive_state_budget_bytes(prepared.budget.memory_bytes)
    spill_dir = checkpoint_mgr.state_dir if checkpoint_mgr is not None else None
    params_map: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (params or {}).items()}

    def merged_fit_params(spec: FeatureSpec) -> dict[str, Any]:
        out = dict(spec.default_params)
        out.update(params_map.get(spec.name, {}))
        return out

    fitted_states: dict[str, Any] = {}
    failed_stateful: set[str] = set()
    fit_already_done = manifest is not None and manifest.get("phase") in ("transform", "done")
    if prepared.has_fit_phase and not fit_already_done:
        accumulators: dict[str, Any] = {}
        reservoirs: dict[str, ReservoirSampler] = {}
        resume_snapshot = manifest.get("fit_accumulator_file") if manifest is not None else None
        resume_progress = checkpoint_mgr.load_fit_accumulators(resume_snapshot) if (checkpoint_mgr is not None and resume_snapshot) else None
        if manifest is not None and manifest.get("fit_batches_completed", 0):
            required = {s.name for s in stateful_specs} - set(manifest.get("fit_failed_features", {}))
            restored = set((resume_progress or {}).get("accumulators", {})) | set((resume_progress or {}).get("reservoirs", {}))
            if not resume_snapshot or not isinstance(resume_progress, dict) or not required <= restored:
                raise CheckpointCorruptionError("Committed fit progress lacks its complete accumulator snapshot")
        # Local checkpoint replay-history closure pass (v0.4.3 -> v0.4.4):
        # seed failed_stateful (and the report) from a PARTIAL prior run's
        # already-committed failure history -- a stateful feature that
        # failed on an earlier, already-checkpointed batch must stay failed
        # on resume, not be silently retried starting from the first
        # not-yet-completed batch (which would skip over whatever batch
        # actually caused the original failure, potentially "succeeding"
        # for a reason a clean run never would have).
        fit_failed_messages: dict[str, dict[str, str]] = dict(manifest.get("fit_failed_features", {})) if manifest is not None else {}
        for _name, _info in fit_failed_messages.items():
            failed_stateful.add(_name)
            if _info.get("kind") == "optional_dependency":
                report.record_optional_dependency_failure(_name, _info.get("message", ""))
            else:
                report.record_failed(_name, _info.get("message", ""))
        for spec in stateful_specs:
            if spec.name in failed_stateful:
                continue
            if effective_mode_by_spec[spec.name] == "approximate":
                # Finding 6 (audit 3): a user-requested approximate_sample_size
                # is a *maximum*, not permission to exceed memory_budget -- the
                # retained reservoir sample is executor working memory like any
                # other. Clamp to a memory_budget-derived state budget and
                # report both the requested and effective capacity.
                sampler, effective_capacity = ReservoirSampler.with_budget(
                    approximate_sample_size,
                    budget_bytes=state_budget_bytes,
                    bytes_per_item=_ESTIMATED_BYTES_PER_RESERVOIR_ITEM,
                    seed=0,
                )
                reservoirs[spec.name] = sampler
                info = {
                    "sample_size_capacity_requested": approximate_sample_size,
                    "sample_size_capacity_effective": effective_capacity,
                }
                if effective_capacity < approximate_sample_size:
                    report.warnings.append(
                        f"{spec.name}: approximate_sample_size={approximate_sample_size} clamped to "
                        f"{effective_capacity} to respect the memory_budget-derived state budget "
                        f"(~{state_budget_bytes:,} bytes)."
                    )
                report.record_approximate(spec.name, info)
            else:
                accumulators[spec.name] = STATEFUL_ENCODERS[spec.name].init_state(
                    state_budget_bytes=state_budget_bytes, spill_dir=spill_dir, params=merged_fit_params(spec)
                )
        if resume_progress is not None:
            accumulators.update(resume_progress.get("accumulators", {}))
            reservoirs.update(resume_progress.get("reservoirs", {}))

        skip_batches = manifest.get("fit_batches_completed", 0) if manifest is not None else 0
        fit_columns = _ordered_unique_columns({c for spec in stateful_specs for c in resolved_columns[spec.name]}, prepared.schema_names)

        for batch_idx, batch in enumerate(src.iter_batches(columns=fit_columns, batch_rows=prepared.resolved_batch_rows)):
            if batch_idx >= skip_batches:
                with scoped_cache(max_bytes=prepared.cache_bytes) as cache:
                    for spec in stateful_specs:
                        if spec.name in failed_stateful:
                            continue
                        col = resolved_columns[spec.name][0]
                        series = batch[col]
                        try:
                            if effective_mode_by_spec[spec.name] == "approximate":
                                vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy() if spec.input_types == ("numeric",) else series.dropna().to_numpy()
                                reservoirs[spec.name].add_batch(vals)
                            else:
                                accumulators[spec.name] = _update_state_with_retry(spec, accumulators[spec.name], series, report, params=merged_fit_params(spec))
                        except Exception as exc:
                            failed_stateful.add(spec.name)
                            if errors == "raise":
                                raise
                            if isinstance(exc, ImportError):
                                report.record_optional_dependency_failure(spec.name, str(exc))
                                fit_failed_messages[spec.name] = {"kind": "optional_dependency", "message": str(exc)}
                            else:
                                msg = f"{type(exc).__name__}: {exc}"
                                report.record_failed(spec.name, msg)
                                fit_failed_messages[spec.name] = {"kind": "failed", "message": msg}
                    report.record_cache_stats(cache.stats())
                if checkpoint_mgr is not None:
                    # Finding 9 (audit 2): write a NEW versioned accumulator
                    # snapshot first; only once that succeeds does the manifest
                    # commit to it as the resume point, and only after THAT
                    # commit are older snapshots cleaned up. A crash between the
                    # snapshot write and the manifest commit leaves the manifest
                    # pointing at the OLD (still consistent) snapshot -- the
                    # new, uncommitted one is simply orphaned and ignored on
                    # resume, never double-applied. fit_failed_features is
                    # committed in this SAME atomic manifest write (local
                    # checkpoint replay-history closure pass) so a crash right
                    # after this batch still leaves the failure durably
                    # recorded, not just the accumulator progress.
                    snapshot_name = checkpoint_mgr.save_fit_accumulators({"accumulators": accumulators, "reservoirs": reservoirs}, batch_idx + 1)
                    manifest["fit_batches_completed"] = batch_idx + 1
                    manifest["fit_accumulator_file"] = snapshot_name
                    manifest["fit_failed_features"] = fit_failed_messages
                    checkpoint_mgr.save(manifest)
                    checkpoint_mgr.cleanup_orphan_accumulator_snapshots(keep_filename=snapshot_name)
            report.n_batches += 1

        for spec in stateful_specs:
            if spec.name in failed_stateful:
                continue
            try:
                encoder = STATEFUL_ENCODERS[spec.name]
                fit_params = merged_fit_params(spec)
                if effective_mode_by_spec[spec.name] == "approximate":
                    sample = reservoirs[spec.name].sample()
                    acc = encoder.update_state(encoder.init_state(params=fit_params), pd.Series(sample), params=fit_params)
                    fitted_states[spec.name] = encoder.finalize_state(acc, params=fit_params)
                    report.approximations_used[spec.name].update(
                        {"sample_size": len(sample), "population_size_seen": reservoirs[spec.name].seen}
                    )
                else:
                    fitted_states[spec.name] = encoder.finalize_state(accumulators[spec.name], params=fit_params)
                if checkpoint_mgr is not None:
                    fname = checkpoint_mgr.save_fit_state(spec.name, fitted_states[spec.name])
                    manifest["fit_state_files"][spec.name] = fname
            except Exception as exc:
                if errors == "raise":
                    raise
                failed_stateful.add(spec.name)
                fitted_states.pop(spec.name, None)
                msg = f"{type(exc).__name__}: {exc}"
                report.record_failed(spec.name, msg)
                fit_failed_messages[spec.name] = {"kind": "failed", "message": msg}
        if checkpoint_mgr is not None:
            manifest["phase"] = "transform"
            manifest["fit_failed_features"] = fit_failed_messages
            checkpoint_mgr.save(manifest)
    elif checkpoint_mgr is not None and manifest is not None and fit_already_done:
        # Restore BOTH the fitted state for features that succeeded AND the
        # failure history for stateful features that failed during the
        # ORIGINAL fit -- not gated on `fit_state_files` being non-empty
        # (the v0.4.3 bug: if EVERY stateful feature failed, that dict is
        # legitimately `{}`, which is falsy, so this whole branch silently
        # never ran at all). `failed_stateful` must be restored BEFORE
        # `_run_transform_phase` computes its own transform-spec selection,
        # so a genuinely-failed feature is not miscategorized there as
        # merely "unfitted" (a different, generic message).
        for name, fname in manifest.get("fit_state_files", {}).items():
            fitted_states[name] = checkpoint_mgr.load_fit_state(fname)
        for name, info in manifest.get("fit_failed_features", {}).items():
            failed_stateful.add(name)
            if info.get("kind") == "optional_dependency":
                report.record_optional_dependency_failure(name, info.get("message", ""))
            else:
                report.record_failed(name, info.get("message", ""))

    return fitted_states, failed_stateful


# ==========================================================================
# Transform phase: applies direct features fresh and stateful features with
# already-frozen state, shared by run_extraction() and
# featm.extractor.FittedFeatureExtractor.transform().
# ==========================================================================
def _run_transform_phase(
    prepared: PreparedRun,
    fitted_states: dict[str, Any],
    failed_stateful: set[str],
    *,
    errors: str,
    output: str,
    use_disk: bool,
    io_backend_available: bool,
    disk_budget_bytes: int | None,
    allow_memory_budget_override: bool,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoint_mgr: "CheckpointManager | None" = None,
    manifest: "dict[str, Any] | None" = None,
) -> FeatureMatrix:
    src = prepared.src
    report = prepared.report
    resolved_columns = prepared.resolved_columns
    budget = prepared.budget
    # Step 3: built once per run, reused for every batch/spec's dispatch
    # call -- "cpu" (the default) makes _apply_direct_feature's GPU branch a
    # no-op, identical to every pre-Step-3 transform phase.
    backend_policy = {
        "backend": prepared.backend, "gpu_fallback": prepared.gpu_fallback,
        "device": prepared.device, "gpu_memory_budget": prepared.gpu_memory_budget,
    }

    # A stateful feature this run selected but that has no fitted state
    # (e.g. the public featm.extractor API's .transform(test_source) selected
    # a feature that was skipped -- too few rows, etc. -- during .fit(train_source))
    # cannot be safely transformed: it is not "batch-unsafe", it simply was
    # never fitted. Isolate it the same way any other feature-level failure
    # is isolated, rather than crashing or silently skipping without saying why.
    unfitted_stateful = [s for s in prepared.stateful_specs if s.name not in fitted_states and s.name not in failed_stateful]
    for spec in unfitted_stateful:
        msg = (
            f"{spec.name} has no fitted state for this transform (it was not successfully fitted during "
            "the corresponding .fit()/fit-phase call -- e.g. it was skipped there, or this is a different "
            "feature selection). A stateful feature can only be transformed with state fitted on a training "
            "source; refit including this feature, or deselect it for this transform."
        )
        if errors == "raise":
            raise UnsafeMultiBatchFeatureError(msg)
        report.record_skipped(spec.name, msg)
        failed_stateful = failed_stateful | {spec.name}

    params_map: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (params or {}).items()}

    def merged_params(spec: FeatureSpec) -> dict[str, Any]:
        out = dict(spec.default_params)
        out.update(params_map.get(spec.name, {}))
        return out

    # ------------------------------------------------------------------
    # No separate priming scan. The first batch this phase actually reads
    # (batch 0 on a fresh run; the first not-yet-completed batch on resume)
    # both discovers output column names AND is fully computed and written --
    # it is never read, discarded, and re-read.
    # ------------------------------------------------------------------
    transform_specs = [s for s in prepared.active_specs if s.name not in failed_stateful]
    transform_columns = _ordered_unique_columns({c for spec in transform_specs for c in resolved_columns[spec.name]}, prepared.schema_names)

    skip_transform_batches = manifest.get("transform_batches_completed", 0) if manifest is not None else 0
    resuming_past_batch_zero = skip_transform_batches > 0 and manifest is not None and "primed_columns" in manifest

    primed_columns: dict[str, list[str]] | None = None
    ordered_output_columns: list[str] | None = None
    primed_specs: list[FeatureSpec] | None = None
    if resuming_past_batch_zero:
        primed_columns = manifest["primed_columns"]
        ordered_output_columns = manifest["ordered_output_columns"]
        primed_specs = [s for s in transform_specs if s.name in primed_columns]
        # Local checkpoint replay-history closure pass: a feature that
        # failed during the ORIGINAL priming batch must still show up as
        # failed in a resumed run's report -- resuming past batch 0 means
        # this run never re-executes the priming batch itself, so nothing
        # else would ever re-discover that failure.
        for _pname, _pinfo in manifest.get("transform_priming_failed_features", {}).items():
            if _pinfo.get("kind") == "optional_dependency":
                report.record_optional_dependency_failure(_pname, _pinfo.get("message", ""))
            else:
                report.record_failed(_pname, _pinfo.get("message", ""))

    writer: DiskFeatureMatrixWriter | None = None
    memory_chunks: list[pd.DataFrame] = []
    memory_chunks_bytes = 0
    schema_drifted: set[str] = set()

    def _make_writer(cols: list[str]) -> DiskFeatureMatrixWriter:
        out_dir = checkpoint_mgr.output_dir if checkpoint_mgr is not None else Path(_default_output_dir())
        start_shard = manifest.get("transform_shards_written", 0) if manifest is not None else 0
        existing_rc = manifest.get("transform_row_counts") if manifest is not None else None
        existing_dt = manifest.get("transform_dtypes") if manifest is not None else None
        existing_bytes = manifest.get("transform_bytes_written", 0) if manifest is not None else 0
        existing_sizes = manifest.get("transform_shard_byte_sizes") if manifest is not None else None
        existing_checksums = manifest.get("transform_shard_checksums") if manifest is not None else None
        return DiskFeatureMatrixWriter(
            out_dir,
            cols,
            start_shard_index=start_shard,
            existing_row_counts=existing_rc,
            existing_dtypes=existing_dt,
            disk_budget_bytes=disk_budget_bytes,
            existing_bytes_written=existing_bytes,
            existing_shard_byte_sizes=existing_sizes,
            existing_shard_checksums=existing_checksums,
        )

    def _save_transform_checkpoint(writer: DiskFeatureMatrixWriter, batch_idx: int) -> None:
        manifest["transform_batches_completed"] = batch_idx + 1
        manifest["transform_shards_written"] = writer.shard_count
        manifest["transform_row_counts"] = writer.row_counts
        manifest["transform_dtypes"] = writer.dtypes
        manifest["transform_bytes_written"] = writer.bytes_written
        manifest["transform_shard_byte_sizes"] = writer.shard_byte_sizes
        manifest["transform_shard_checksums"] = writer.shard_checksums
        # Local checkpoint replay-history closure pass: `degraded` (closed
        # over from the enclosing scope) reflects the CUMULATIVE per-feature
        # degraded-batch count across this whole logical run -- seeded from
        # any prior run's committed count on resume (below), so a resumed
        # run's final report warns about every degraded batch the full
        # logical execution ever saw, not just the ones this process itself
        # computed.
        manifest["degraded_batches_by_feature"] = degraded
        checkpoint_mgr.save(manifest)

    def _unsafe_memory_output(reason_bytes: int, *, kind: str) -> None:
        """Finding 8 (audit 3): output='memory' is a hard budget promise unless overridden.

        output='auto' prefers switching to disk when it can; an *explicit*
        output='memory' cannot silently switch backends (the caller asked for
        memory specifically), so it must either raise or -- only if the
        caller explicitly opted in via allow_memory_budget_override=True --
        proceed with a clear warning. Never silently exceed the budget with
        no signal at all (the pre-fix behavior).
        """
        raise UnsafeOutputModeError(
            f"output='memory': {kind} (~{reason_bytes:,} bytes) exceeds the memory_budget "
            f"({budget.memory_bytes:,} bytes). memory_budget is a hard planner envelope by default; pass "
            "output='disk' or output='auto' to let the engine choose safely, increase memory_budget, or "
            "pass allow_memory_budget_override=True to explicitly accept an over-budget in-memory result."
        )

    if use_disk and resuming_past_batch_zero:
        writer = _make_writer(ordered_output_columns)

    override_warned = False
    degraded: dict[str, int] = dict(manifest.get("degraded_batches_by_feature", {})) if (manifest is not None and resuming_past_batch_zero) else {}
    disk_budget_exceeded: DiskBudgetExceededError | None = None
    for batch_idx, batch in enumerate(src.iter_batches(columns=transform_columns, batch_rows=prepared.resolved_batch_rows)):
        if batch_idx < skip_transform_batches:
            continue

        # Step 3: a matching per-batch GPU-cache scope, opened once here (not
        # once per feature inside dispatch_one_feature -- see that
        # function's docstring), so several GPU-capable features selected
        # together share one device-resident FFT for this batch.
        import contextlib as _contextlib

        from .gpu.cache import gpu_cache_budget_bytes as _gpu_cache_budget_bytes
        from .gpu.cache import scoped_gpu_cache as _scoped_gpu_cache

        _gpu_cache_max_bytes = None
        if prepared.backend != "cpu":
            # Audit-fix pass, Finding 1: derive this batch's shared GPU cache
            # cap from the resolved GPU budget (fresh per batch, since real
            # free VRAM can change over the run), not a fixed constant --
            # see featm.engine.extract's identical reasoning.
            from .gpu.memory import GPUMemoryBudget as _GPUMemoryBudget

            _gpu_cache_max_bytes = _gpu_cache_budget_bytes(
                _GPUMemoryBudget.resolve(gpu_memory_budget=prepared.gpu_memory_budget, device=prepared.device)
            )
        _gpu_cache_ctx = (
            _scoped_gpu_cache(max_bytes=_gpu_cache_max_bytes) if prepared.backend != "cpu" else _contextlib.nullcontext()
        )
        with scoped_cache(max_bytes=prepared.cache_bytes) as cache, _gpu_cache_ctx:
            if primed_columns is None:
                # First batch this call actually sees: discover columns AND
                # compute+write this batch's values in the same pass.
                primed_columns = {}
                ordered_output_columns = []
                out_frame: dict[str, np.ndarray] = {}
                n = len(batch)
                # Finding 10 (fourth audit pass): observability, not
                # enforcement -- a single adversarial giant row is an
                # irreducible lower bound no bounded sample can predict, so
                # this only warns on a gross planning-estimate miss, never
                # blocks the run.
                if n and prepared.planned_bytes_per_row_in is not None:
                    actual_bpr = _dataframe_bytes(batch) / n
                    if actual_bpr > prepared.planned_bytes_per_row_in * 3:
                        report.warnings.append(
                            f"Batch input size estimate was off by >3x: planned ~{prepared.planned_bytes_per_row_in:,} "
                            f"bytes/row, actual first batch measured ~{actual_bpr:,.0f} bytes/row (batch_rows="
                            f"{prepared.resolved_batch_rows:,}). Later batches may hold more actual memory than "
                            "planned for."
                        )
                priming_failed_messages: dict[str, dict[str, str]] = {}
                for spec in transform_specs:
                    try:
                        expanded = _compute_transform_with_retry(spec, batch, resolved_columns, merged_params(spec), fitted_states, report, backend_policy=backend_policy, batch_id=batch_idx)
                        cols = list(expanded.keys())
                        primed_columns[spec.name] = cols
                        ordered_output_columns.extend(cols)
                        for c in cols:
                            out_frame[c] = np.asarray(expanded[c])
                    except Exception as exc:
                        if errors == "raise":
                            raise
                        if isinstance(exc, ImportError):
                            report.record_optional_dependency_failure(spec.name, str(exc))
                            priming_failed_messages[spec.name] = {"kind": "optional_dependency", "message": str(exc)}
                        else:
                            msg = f"{type(exc).__name__}: {exc}"
                            report.record_failed(spec.name, msg)
                            priming_failed_messages[spec.name] = {"kind": "failed", "message": msg}
                primed_specs = [s for s in transform_specs if s.name in primed_columns]
                if checkpoint_mgr is not None:
                    manifest["primed_columns"] = primed_columns
                    manifest["ordered_output_columns"] = ordered_output_columns
                    # Local checkpoint replay-history closure pass: persisted
                    # atomically alongside the schema this priming batch
                    # established -- a transform-priming failure must survive
                    # resume the same way a fit-phase failure now does.
                    manifest["transform_priming_failed_features"] = priming_failed_messages

                # Finding 2 (audit 2): the pre-run output-backend decision used
                # a *guessed* width for undeclared vector/global features
                # (estimate_output_width's conservative constant). Now that
                # this batch's REAL output columns/dtypes are known, re-check
                # an "auto"/"memory" decision against the actual per-row byte
                # cost before committing to memory -- never after accumulating
                # an over-budget result. The safety multiplier is applied
                # HERE, once, against the projected final total: the moment
                # that matters for a genuine peak is the eventual pd.concat()
                # of every accumulated chunk, which briefly needs roughly the
                # old chunks plus the newly built combined frame at once.
                if not use_disk and output in ("auto", "memory") and src.n_rows is not None:
                    real_bpr = _real_bytes_per_row(out_frame)
                    real_est_total = src.n_rows * real_bpr
                    if real_est_total * _MEMORY_ASSEMBLY_SAFETY_MULTIPLIER > budget.memory_bytes:
                        if output == "memory" and allow_memory_budget_override:
                            report.warnings.append(
                                f"output='memory': the actual output width ({len(ordered_output_columns)} "
                                f"columns) makes the real estimated size (~{real_est_total:,} bytes) exceed "
                                f"memory_budget ({budget.memory_bytes:,} bytes); proceeding in-memory because "
                                "allow_memory_budget_override=True was passed."
                            )
                        elif output == "memory":
                            _unsafe_memory_output(real_est_total, kind="the actual output width's estimated final size")
                        elif not io_backend_available:
                            raise UnsafeOutputModeError(
                                f"output='auto': the actual output width ({len(ordered_output_columns)} "
                                f"columns) makes the real estimated size (~{real_est_total:,} bytes) exceed "
                                f"the memory_budget ({budget.memory_bytes:,} bytes), and no disk backend is "
                                "available (pip install 'featm[io]') to spill to instead. Failing now, before "
                                "accumulating any further batches, rather than returning an over-budget result."
                            )
                        else:
                            use_disk = True
                            report.output_backend = "disk"
                            report.warnings.append(
                                f"output='auto': actual output width made the real estimate "
                                f"(~{real_est_total:,} bytes) exceed memory_budget ({budget.memory_bytes:,} "
                                "bytes); switched to disk-backed output before writing the first batch."
                            )
                if use_disk:
                    writer = _make_writer(ordered_output_columns)
            else:
                out_frame = {}
                n = len(batch)
                for spec in primed_specs:
                    try:
                        expanded = _compute_transform_with_retry(spec, batch, resolved_columns, merged_params(spec), fitted_states, report, backend_policy=backend_policy, batch_id=batch_idx)
                        # Finding 6 (fourth audit pass): the output schema is
                        # fixed once, from the primed batch -- a later batch
                        # whose real emitted columns (set AND order) differ
                        # is a correctness failure (classically: an unfitted
                        # vectorizer whose vocabulary silently drifted
                        # between batches), not something to paper over by
                        # dropping new columns or NaN-filling missing ones
                        # while still calling the feature "fine".
                        if list(expanded.keys()) != primed_columns[spec.name]:
                            raise FeatureSchemaDriftError(
                                f"{spec.name} emitted columns {list(expanded.keys())} on batch {batch_idx}, "
                                f"which differs from its primed schema {primed_columns[spec.name]} (from batch "
                                "0). A feature's output schema must be stable across every batch; this is a "
                                "correctness failure (e.g. a corpus-dependent vocabulary that was not properly "
                                "fitted/frozen), not a transient error to recover from."
                            )
                        for c in primed_columns[spec.name]:
                            out_frame[c] = np.asarray(expanded[c])
                    except Exception as exc:
                        if errors == "raise":
                            raise
                        degraded[spec.name] = degraded.get(spec.name, 0) + 1
                        if isinstance(exc, FeatureSchemaDriftError):
                            if errors == "raise":
                                raise
                            schema_drifted.add(spec.name)
                            reason = str(exc)
                        else:
                            reason = "not safe to retry-split (whole-dataset/legacy-fit feature)" if isinstance(exc, MemoryError) and not _is_retry_split_safe(spec, is_stateful_transform=spec.name in fitted_states) else (
                                "exhausted memory retry down to the minimum batch floor" if isinstance(exc, MemoryError) else f"{type(exc).__name__}: {exc}"
                            )
                        report.warnings.append(f"{spec.name}: batch {batch_idx} failed ({reason}); filled with NaN")
                        report._warn(report.warnings[-1])
                        for c in primed_columns[spec.name]:
                            out_frame[c] = np.full(n, np.nan)

            result_df = pd.DataFrame(out_frame, columns=ordered_output_columns)
            report.record_cache_stats(cache.stats())

        if writer is not None:
            try:
                writer.write_batch(result_df)
            except DiskBudgetExceededError as exc:
                disk_budget_exceeded = exc
                break
            if checkpoint_mgr is not None:
                _save_transform_checkpoint(writer, batch_idx)
        else:
            memory_chunks.append(result_df)
            memory_chunks_bytes += _dataframe_bytes(result_df)
            # Finding 2 (audit 2) / Finding 8 (audit 3): a continuous safety
            # net against estimation error -- e.g. the row count was unknown
            # so the batch-0 projection above could not run at all, or actual
            # per-batch output is simply larger than any earlier estimate
            # assumed. Compared against RAW accumulated bytes (no multiplier):
            # the assembly-overhead margin is already spent, once, in the
            # batch-0 projection above against the *final* total: while still
            # accumulating, holding exactly `memory_chunks_bytes` is the real,
            # current peak, not a projection needing extra headroom. Now also
            # applies to an explicit output='memory' (previously only
            # output='auto'), governed by allow_memory_budget_override.
            if output in ("auto", "memory") and memory_chunks_bytes > budget.memory_bytes:
                if output == "memory" and allow_memory_budget_override:
                    if not override_warned:
                        report.warnings.append(
                            f"output='memory': accumulated in-memory output (~{memory_chunks_bytes:,} bytes, "
                            "and growing) has exceeded memory_budget "
                            f"({budget.memory_bytes:,} bytes); continuing because "
                            "allow_memory_budget_override=True was passed. (Reported once; the excess grows "
                            "with every further batch for the remainder of this run.)"
                        )
                        override_warned = True
                elif output == "memory":
                    _unsafe_memory_output(memory_chunks_bytes, kind="accumulated in-memory output")
                elif not io_backend_available:
                    raise UnsafeOutputModeError(
                        f"output='auto': accumulated in-memory output (~{memory_chunks_bytes:,} bytes) has "
                        f"exceeded the memory_budget ({budget.memory_bytes:,} bytes), and no disk backend is "
                        "available (pip install 'featm[io]') to spill to instead."
                    )
                else:
                    report.warnings.append(
                        f"output='auto': accumulated in-memory output exceeded the memory_budget mid-run "
                        f"(~{memory_chunks_bytes:,} bytes accumulated); switching to disk-backed output and "
                        "spilling everything accumulated so far."
                    )
                    use_disk = True
                    report.output_backend = "disk"
                    writer = _make_writer(ordered_output_columns)
                    for chunk in memory_chunks:
                        writer.write_batch(chunk)
                    memory_chunks = []
                    memory_chunks_bytes = 0
                    if checkpoint_mgr is not None:
                        _save_transform_checkpoint(writer, batch_idx)
        report.n_batches += 1

    if disk_budget_exceeded is not None:
        report.finalize()
        raise disk_budget_exceeded

    primed_columns = primed_columns or {}
    ordered_output_columns = ordered_output_columns or []
    primed_specs = primed_specs or []

    for name, cnt in degraded.items():
        report.retries.setdefault(name, 0)
        report.warnings.append(f"{name}: degraded on {cnt} batch(es); output for those rows is NaN")

    for spec in primed_specs:
        if spec.name in schema_drifted:
            # Finding 6 (fourth audit pass): a schema-drifted feature is a
            # correctness failure, not a "completed with some degraded
            # batches" feature (the pre-existing degraded-batch pattern for
            # transient per-batch computation errors) -- it must not be
            # silently reported as completed.
            report.record_failed(spec.name, "output schema drifted between batches (see warnings for details)")
            continue
        report.record_completed(spec.name, columns=primed_columns[spec.name], elapsed=0.0)

    # Findings 1 & 2 (audit 3): the persisted checkpoint must become valid
    # BEFORE the manifest is allowed to say done=True -- finalize (and, for a
    # checkpointed run, validate) the disk-backed result first; only once
    # that succeeds is "done" the final commit point. A crash before this
    # point leaves a resumable, not-done checkpoint; it can never leave a
    # "done" checkpoint whose canonical output was never actually finalized.
    if writer is not None:
        matrix = writer.finalize()
        if checkpoint_mgr is not None:
            problems = validate_disk_backend_integrity(checkpoint_mgr.output_dir)
            if problems:
                raise CheckpointCorruptionError(
                    f"Checkpoint at {checkpoint_mgr.directory} finished writing its output, but the "
                    f"finalized result failed its own integrity check: {problems}. The checkpoint manifest "
                    "was not marked done, so resuming will not silently reuse it."
                )
    else:
        combined = pd.concat(memory_chunks, ignore_index=True) if memory_chunks else pd.DataFrame(columns=ordered_output_columns)
        matrix = FeatureMatrix.from_pandas(combined)

    if manifest is not None:
        manifest["done"] = True
        manifest["phase"] = "done"
        checkpoint_mgr.save(manifest)

    if writer is not None and output == "memory":
        matrix = _materialize_checkpoint_result_to_memory(
            matrix, budget=budget, allow_memory_budget_override=allow_memory_budget_override, report=report
        )
        report.output_backend = "memory"

    report.finalize()
    matrix.report = report
    return matrix


def run_extraction(
    source: Any,
    *,
    features: Sequence[str],
    modality: str | None = None,
    columns: Mapping[str, Any] | None = None,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str = "exact",
    errors: str = "skip",
    memory_budget: "str | int | None" = None,
    disk_budget: "str | int | None" = None,
    output: str = "auto",
    batch_rows: int | None = None,
    checkpoint: "str | Path | None" = None,
    cache_budget: "str | int | None" = None,
    approximate_sample_size: int = DEFAULT_APPROXIMATE_SAMPLE_SIZE,
    allow_memory_budget_override: bool = False,
    backend: str = "cpu",
    gpu_fallback: str = "fallback",
    device: "str | int | None" = None,
    gpu_memory_budget: "str | int | None" = None,
) -> tuple[FeatureMatrix, ExecutionReport]:
    """One-source fit+transform convenience path -- the ``featm.extract()`` out-of-core route.

    For leakage-safe fitting on one source and transforming a *different*
    source (the train/test split use case), see :mod:`featm.extractor`
    (:func:`featm.fit`), which reuses the exact same
    :func:`_prepare_run`/:func:`_run_fit_phase`/:func:`_run_transform_phase`
    machinery this function calls, just against two independently-prepared
    sources instead of one.

    ``backend``/``gpu_fallback``/``device``/``gpu_memory_budget`` (Step 3):
    the fit phase always runs on CPU regardless of ``backend`` (see
    ``docs/SCALING.md``'s GPU section for why -- v1 stateful accumulators
    are not GPU-accelerated); the transform phase dispatches each
    GPU-capable selected feature's per-batch computation through
    ``featm.gpu.dispatch.dispatch_one_feature`` when ``backend != "cpu"``.
    """
    if output not in {"auto", "memory", "disk"}:
        raise ValueError("output must be auto, memory, or disk")

    from . import __version__ as featm_version

    prepared = _prepare_run(
        source,
        features=features,
        modality=modality,
        columns=columns,
        mode=mode,
        errors=errors,
        memory_budget=memory_budget,
        disk_budget=disk_budget,
        batch_rows=batch_rows,
        cache_budget=cache_budget,
        params=params,
        backend=backend,
        gpu_fallback=gpu_fallback,
        device=device,
        gpu_memory_budget=gpu_memory_budget,
    )
    src = prepared.src
    report = prepared.report

    checkpoint_mgr: CheckpointManager | None = CheckpointManager(checkpoint) if checkpoint is not None else None
    use_disk, io_backend_available, disk_budget_bytes = _resolve_output_backend(
        prepared, output=output, checkpoint_mgr=checkpoint_mgr, disk_budget=disk_budget
    )

    manifest: dict[str, Any] | None = None
    if checkpoint_mgr is not None:
        # Every setting that can change numerical results or output schema must
        # be part of the identity, or resuming could silently reuse incompatible
        # partial work (e.g. the same feature names mapped to different columns).
        # Finding 3 (audit 3): canonical_label (not str()) preserves each
        # column label's Python type in this identity -- str(0) == str("0")
        # would otherwise let an int-labeled column and a same-spelled
        # string-labeled column collide into the same checkpoint identity.
        canonical_columns = {
            name: [canonical_label(c) for c in cols] for name, cols in sorted(prepared.resolved_columns.items(), key=lambda kv: kv[0])
        }
        # Checkpoint budget identity closure (v0.4.2 -> v0.4.3, ported from the
        # distributed checkpoint identity fix): memory_budget is not merely a
        # performance/resource hint for an exact stateful feature -- the
        # NORMALIZED, DERIVED state budget handed to every stateful encoder
        # directly decides whether fitting succeeds or raises
        # StateBudgetExceededError, which under errors="skip"/"warn" changes
        # the actual selected feature set and result shape. Without this in
        # identity, a checkpoint fit under a small budget (a stateful feature
        # failed, so `fit_state_files` never recorded that feature at all --
        # see `_run_fit_phase`'s resume branch) resumed under a larger budget
        # reused the small-budget manifest's incomplete `fit_state_files`
        # directly: not merely stale results, but an internally INCOHERENT
        # state (empty output, empty `failed_features`, matching neither the
        # original small-budget run nor a clean large-budget run). `None`
        # when the run has no fit phase at all -- an unrelated memory_budget
        # change must not invalidate a checkpoint whose result never
        # depended on the state budget (matches the distributed checkpoint's
        # policy exactly, for the same reason).
        existing = checkpoint_mgr.load()
        existing_settings = existing.get('settings', {}) if existing is not None else None
        state_budget_bytes, state_budget_policy = checkpoint_state_budget(
            prepared.budget, memory_budget=memory_budget, stateful=prepared.has_fit_phase,
            existing=existing_settings,
        )
        batch_rows_policy = ('auto' if batch_rows is None else 'explicit') if prepared.has_fit_phase else None
        if (existing_settings is not None and state_budget_policy == 'auto'
                and existing_settings.get('state_budget_policy') == 'auto'
                and batch_rows_policy == 'auto' and existing_settings.get('batch_rows_policy') == 'auto'):
            saved_rows = existing_settings.get('batch_rows')
            if type(saved_rows) is not int or saved_rows < 1:
                raise CheckpointCorruptionError('Automatic checkpoint has invalid stored batch boundaries')
            prepared.resolved_batch_rows = saved_rows
            report.batch_rows = saved_rows
        prepared.checkpoint_state_budget_bytes = state_budget_bytes
        identity = CheckpointIdentity(
            featm_version=featm_version,
            source_fingerprint=src.fingerprint(),
            schema_fingerprint=src.schema().fingerprint(),
            features=list(features),
            params={k: dict(v) for k, v in (params or {}).items()},
            settings={
                "batch_rows": prepared.resolved_batch_rows,
                "mode": mode,
                "errors": errors,
                "columns": canonical_columns,
                "approximate_sample_size": approximate_sample_size,
                "state_budget_bytes": state_budget_bytes,
                "state_budget_policy": state_budget_policy,
                "batch_rows_policy": batch_rows_policy,
            },
        )
        manifest = checkpoint_mgr.initialize_or_resume(identity, has_fit_phase=prepared.has_fit_phase)
        report.checkpoint_dir = str(checkpoint_mgr.directory)
        report.resumed = manifest.get("fit_batches_completed", 0) > 0 or manifest.get("transform_batches_completed", 0) > 0
        if report.resumed:
            report.resumed_features = [s.name for s in prepared.active_specs]

        # Zero-read fast path: a checkpoint already marked done reopens its
        # finalized output directly -- but only after confirming every shard
        # the manifest declares actually exists and looks intact (Finding 10,
        # audit 2); a "done" checkpoint whose output was deleted/corrupted
        # afterward must not be silently treated as a valid zero-read reuse.
        if manifest.get("done"):
            problems = validate_disk_backend_integrity(checkpoint_mgr.output_dir)
            if problems:
                raise CheckpointCorruptionError(
                    f"Checkpoint at {checkpoint_mgr.directory} is marked done, but its output has integrity "
                    f"problems and cannot be safely reused as a zero-read result: {problems}. Delete the "
                    "checkpoint directory to start fresh, or restore the missing/corrupt file(s)."
                )
            matrix = FeatureMatrix.open(checkpoint_mgr.output_dir)
            if output == "memory":
                matrix = _materialize_checkpoint_result_to_memory(
                    matrix, budget=prepared.budget, allow_memory_budget_override=allow_memory_budget_override, report=report
                )
                report.output_backend = "memory"
            for spec in prepared.active_specs:
                cols = manifest.get("primed_columns", {}).get(spec.name)
                if cols is not None:
                    report.record_completed(spec.name, columns=cols, elapsed=0.0)
            # Local checkpoint replay-history closure pass (v0.4.3 -> v0.4.4):
            # a zero-read "done" checkpoint's output BYTES are correct on
            # their own, but its ExecutionReport must also be
            # replay-equivalent to the original run's -- not silently
            # "clean" merely because reopening skips the fit/transform code
            # paths that would otherwise re-derive failed_features. Restore
            # every historical failure/degradation bucket the manifest
            # persisted, the same way a partial resume does.
            for name, info in manifest.get("fit_failed_features", {}).items():
                if info.get("kind") == "optional_dependency":
                    report.record_optional_dependency_failure(name, info.get("message", ""))
                else:
                    report.record_failed(name, info.get("message", ""))
            for name, info in manifest.get("transform_priming_failed_features", {}).items():
                if info.get("kind") == "optional_dependency":
                    report.record_optional_dependency_failure(name, info.get("message", ""))
                else:
                    report.record_failed(name, info.get("message", ""))
            for name, cnt in manifest.get("degraded_batches_by_feature", {}).items():
                report.retries.setdefault(name, 0)
                report.warnings.append(f"{name}: degraded on {cnt} batch(es); output for those rows is NaN")
            report.n_batches = manifest.get("transform_shards_written", 0)
            report.cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "oversized_skips": 0, "entries": 0, "bytes": 0}
            report.finalize()
            matrix.report = report
            return matrix, report

    if manifest is not None and prepared.has_fit_phase and memory_budget is None:
        check_checkpoint_state_capacity(prepared.budget, prepared.checkpoint_state_budget_bytes)

    fitted_states, failed_stateful = _run_fit_phase(
        prepared,
        errors=errors,
        approximate_sample_size=approximate_sample_size,
        params=params,
        checkpoint_mgr=checkpoint_mgr,
        manifest=manifest,
    )

    matrix = _run_transform_phase(
        prepared,
        fitted_states,
        failed_stateful,
        errors=errors,
        output=output,
        use_disk=use_disk,
        io_backend_available=io_backend_available,
        disk_budget_bytes=disk_budget_bytes,
        allow_memory_budget_override=allow_memory_budget_override,
        params=params,
        checkpoint_mgr=checkpoint_mgr,
        manifest=manifest,
    )
    return matrix, report
