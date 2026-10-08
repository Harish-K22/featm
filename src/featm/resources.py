from __future__ import annotations

import re
from dataclasses import dataclass
import os

import psutil


_SIZE_UNITS = {
    "b": 1,
    "kb": 1024, "k": 1024,
    "mb": 1024 ** 2, "m": 1024 ** 2,
    "gb": 1024 ** 3, "g": 1024 ** 3,
    "tb": 1024 ** 4, "t": 1024 ** 4,
}
_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")


def parse_size(spec: "str | int | float | None") -> int | None:
    """Parse a human size like ``'2GB'`` / ``'512MB'`` / ``1_000_000`` into bytes.

    ``None`` passes through as ``None`` (meaning "no explicit budget / use auto").
    """
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise TypeError("size must be a number or string, not bool")
    if isinstance(spec, (int, float)):
        return int(spec)
    match = _SIZE_RE.match(str(spec))
    if not match:
        raise ValueError(f"Cannot parse size: {spec!r}")
    value, unit = match.groups()
    unit = unit.lower() or "b"
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown size unit {unit!r} in {spec!r}; expected one of {sorted(_SIZE_UNITS)}")
    return int(float(value) * _SIZE_UNITS[unit])


@dataclass(frozen=True)
class ResourceBudget:
    memory_bytes: int
    disk_bytes: int | None
    cpu_workers: int
    gpu_available: bool = False
    gpu_memory_bytes: int | None = None
    memory_target_fraction: float = 0.65

    @classmethod
    def auto(cls, memory_target_fraction: float = 0.65) -> "ResourceBudget":
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(os.getcwd())
        return cls(
            memory_bytes=int(vm.available * memory_target_fraction),
            disk_bytes=int(disk.free * 0.8),
            cpu_workers=max((os.cpu_count() or 1) - 1, 1),
            gpu_available=False,
            gpu_memory_bytes=None,
            memory_target_fraction=memory_target_fraction,
        )

    @classmethod
    def resolve(
        cls,
        *,
        memory_budget: "str | int | None" = None,
        disk_budget: "str | int | None" = None,
        memory_target_fraction: float = 0.65,
    ) -> "ResourceBudget":
        """Build a budget starting from :meth:`auto`, overridden by any explicit sizes.

        This is the executor *working-memory* budget the planner sizes batches
        against -- it is deliberately not the same thing as total Python
        process RSS, which also includes the interpreter, imported libraries,
        and any data the caller is holding onto outside this call. See
        ``docs/SCALING.md`` for that distinction.
        """
        auto = cls.auto(memory_target_fraction=memory_target_fraction)
        mem = parse_size(memory_budget)
        disk = parse_size(disk_budget)
        return cls(
            memory_bytes=mem if mem is not None else auto.memory_bytes,
            disk_bytes=disk if disk is not None else auto.disk_bytes,
            cpu_workers=auto.cpu_workers,
            gpu_available=auto.gpu_available,
            gpu_memory_bytes=auto.gpu_memory_bytes,
            memory_target_fraction=memory_target_fraction,
        )


@dataclass(frozen=True)
class ResourceEstimate:
    peak_memory_bytes: int
    output_bytes: int
    work_units: int


def estimate_dense_output_bytes(n_rows: int, n_features: int, itemsize: int = 8) -> int:
    return max(n_rows, 0) * max(n_features, 0) * itemsize


class ResourceBudgetTooSmallError(RuntimeError):
    """Raised when even a single row's worth of a batch cannot fit the declared memory budget."""


def _input_bytes_per_row(n_columns_in: int, itemsize: int, bytes_per_row_in: int | None) -> float:
    """Real (source-reported) input bytes/row when available, else the itemsize-per-column guess.

    Fourth audit pass, Finding 10: ``n_columns_in * itemsize`` assumes every
    input cell is a fixed-width ``itemsize``-byte value -- correct for
    numeric dtypes, wildly wrong for variable-length strings/objects (a
    single 10KB text cell is ~1,250x an assumed 8 bytes). When the source can
    report a real, deep-memory-based estimate (``Source.estimated_bytes_per_row``),
    that estimate is trusted instead; the itemsize-per-column guess remains the
    fallback for sources that genuinely cannot estimate this cheaply (e.g.
    ``IteratorSource``).
    """
    return float(bytes_per_row_in) if bytes_per_row_in is not None else n_columns_in * itemsize


def choose_batch_rows(
    *,
    n_columns_in: int,
    n_columns_out: int,
    budget: ResourceBudget,
    itemsize: int = 8,
    safety_multiplier: float = 4.0,
    min_rows: int = 1_000,
    max_rows: int = 1_000_000,
    bytes_per_row_in: int | None = None,
) -> int:
    """Pick a batch size that fits the budget, preferring ``min_rows`` for throughput
    but never lying about the budget to get it.

    ``min_rows`` is a *performance* preference (larger batches amortize
    per-batch overhead), not a safety floor -- silently returning ``min_rows``
    when the budget cannot actually afford it (the pre-fix behavior) meant an
    automatically-planned batch could exceed a declared tiny ``memory_budget``
    by an order of magnitude or more with no warning. If the budget-implied
    row count is below ``min_rows``, this returns the budget-implied count
    instead (as low as 1), and only raises :class:`ResourceBudgetTooSmallError`
    if even a single row does not fit.

    ``bytes_per_row_in``, when given (a real, source-reported estimate --
    see :func:`_input_bytes_per_row`), replaces the ``n_columns_in * itemsize``
    guess for the *input* side of the estimate; the output side still uses
    ``n_columns_out * itemsize`` (output columns are always the engine's own
    numeric feature values, for which the itemsize assumption is accurate).
    """
    # Input + output + temporaries. Conservative because vectorized operations
    # often allocate multiple full-sized intermediates.
    in_bytes = _input_bytes_per_row(n_columns_in, itemsize, bytes_per_row_in)
    bytes_per_row = max((in_bytes + n_columns_out * itemsize) * safety_multiplier, 1)
    budget_rows = int(budget.memory_bytes / bytes_per_row)
    if budget_rows >= min_rows:
        return min(budget_rows, max_rows)
    if budget_rows >= 1:
        return budget_rows
    raise ResourceBudgetTooSmallError(
        f"memory_budget ({budget.memory_bytes:,} bytes) cannot fit even a single row of the planned "
        f"batch (estimated {bytes_per_row:,.0f} bytes/row for {n_columns_in} input + {n_columns_out} "
        "output column(s), with a "
        f"{safety_multiplier:g}x safety multiplier for intermediate allocations). Increase "
        "memory_budget, reduce the number of selected features/columns, or select fewer wide-output "
        "features per call."
    )


def max_rows_within_budget(
    *,
    n_columns_in: int,
    n_columns_out: int,
    budget: ResourceBudget,
    itemsize: int = 8,
    safety_multiplier: float = 4.0,
    bytes_per_row_in: int | None = None,
) -> int:
    """How many rows of ``n_columns_in``+``n_columns_out`` would fit the memory budget, uncapped.

    Same estimate as :func:`choose_batch_rows` but without its ``min_rows``/``max_rows``
    clamping -- used to decide whether a FULL_DATASET feature's entire input can be
    handled in one batch, not to pick an ordinary streaming batch size.
    """
    in_bytes = _input_bytes_per_row(n_columns_in, itemsize, bytes_per_row_in)
    bytes_per_row = max((in_bytes + n_columns_out * itemsize) * safety_multiplier, 1)
    return int(budget.memory_bytes / bytes_per_row)


def estimate_full_materialization_bytes(spec, n_rows: int, n_columns_in: int, itemsize: int = 8) -> int:
    """Estimate the peak working memory a ``requires_full_materialization=True`` feature needs.

    A raw-row-bytes estimate (``n_rows * n_columns_in * itemsize``) is only
    correct for a feature whose working memory genuinely scales linearly with
    input size. Several full-materialization families do not: e.g.
    ``spatial.point_patterns.*``/``spatial.autocorrelation.*`` build a dense
    O(n^2) pairwise distance matrix internally, so 100,000 2-D points (~1.6MB
    raw) can require an ~80GB intermediate. Approving that as "fits in one
    batch" from raw bytes alone would be a real resource-safety failure, not
    just an inaccurate estimate.

    Precedence:

    1. ``spec.memory_estimator(n_rows)`` if the feature declares one -- a
       curated, implementation-verified estimate, trusted as-is.
    2. Otherwise, a conservative heuristic derived from ``spec.complexity``
       (itself defaulted from ``cost`` unless a feature sets it explicitly):
       quadratic/cubic/exponential complexity assumes a dense O(n^2)/O(n^3)
       structure may be built, and the *larger* of that and the raw-bytes
       estimate is used. This is a coarse, unaudited default -- it has not
       been individually verified against every full-materialization
       feature's actual implementation, unlike case 1 -- and is documented as
       such rather than presented as a verified guarantee.
    """
    from .registry import Complexity

    raw_bytes = int(n_rows) * max(n_columns_in, 1) * itemsize
    if spec.memory_estimator is not None:
        return max(int(spec.memory_estimator(n_rows)), 0)
    if spec.complexity == Complexity.QUADRATIC:
        return max(raw_bytes, int(n_rows) * int(n_rows) * itemsize)
    if spec.complexity in (Complexity.CUBIC, Complexity.EXPONENTIAL):
        return max(raw_bytes, int(n_rows) * int(n_rows) * int(n_rows) * itemsize)
    return raw_bytes


#: Conservative per-category estimate for a Python (str key -> int count) dict
#: entry: CPython dict overhead + a typical short-string object + an int
#: object is comfortably in the low hundreds of bytes; used only to decide
#: when to spill an exact frequency count to disk, not as a precise accounting.
ESTIMATED_BYTES_PER_CATEGORY = 200


def derive_state_budget_bytes(memory_budget_bytes: int, *, fraction: float = 0.25, default_cap: int = 256 * 1024 * 1024) -> int:
    """How much of the declared working-memory budget fitted STATE (not batch/cache/output) may use.

    Mirrors the existing cache-budget-derivation policy (a fraction of the
    overall budget, capped at a sane default) so state, cache, and batch
    memory are each bounded fractions of one coherent budget rather than each
    independently being allowed to consume the *entire* memory_budget on its
    own -- see docs/SCALING.md's per-component budget-partition note.
    """
    return max(min(int(memory_budget_bytes * fraction), default_cap), 1)


def checkpoint_state_budget(
    budget: ResourceBudget, *, memory_budget, stateful: bool, existing: dict | None,
) -> tuple[int | None, str | None]:
    """Normalize explicit budgets; retain an automatic checkpoint's original semantics.

    The policy is itself identity-sensitive, so switching from automatic to
    explicit budgeting cannot silently adopt a different interpretation.
    Compatibility of every other identity field is still checked by the manager.
    """
    if not stateful:
        return None, None
    policy = 'auto' if memory_budget is None else 'explicit'
    value = derive_state_budget_bytes(budget.memory_bytes)
    if existing is not None and policy == 'auto' and existing.get('state_budget_policy') == 'auto':
        value = existing.get('state_budget_bytes')
        if type(value) is not int or not 1 <= value <= 256 * 1024 * 1024:
            from .matrix import CheckpointCorruptionError
            raise CheckpointCorruptionError('Automatic checkpoint has an invalid stored state budget; restore the manifest or use a fresh checkpoint directory.')
    return value, policy


def check_checkpoint_state_capacity(budget: ResourceBudget, state_budget_bytes: int) -> None:
    """Reserve half of current working memory for buffers beside retained state.

    This guard applies only when automatic checkpoint reuse needs execution;
    finished checkpoints can reopen their output without allocating fit state.
    It does not reinterpret state/spill thresholds to fit a smaller host.
    """
    if 2 * state_budget_bytes > budget.memory_bytes:
        raise ResourceBudgetTooSmallError(
            f'Automatic checkpoint retains a {state_budget_bytes:,}-byte state budget, '
            f'but current working-memory capacity is only {budget.memory_bytes:,} bytes. '
            'Free memory or resume on a larger machine (at least twice the stored '
            'state budget in working memory); use a fresh checkpoint directory '
            'to deliberately fit with a different state budget.'
        )


def estimate_output_width(specs) -> int:
    """Conservative output-column-count estimate for a set of selected FeatureSpecs.

    Uses declared ``output_names`` when a feature statically knows its width;
    otherwise assumes 1 for scalar output and a conservative fixed guess for
    vector/global output whose width is only known at call time.
    """
    total = 0
    for spec in specs:
        if spec.output_names is not None:
            total += len(spec.output_names)
        elif spec.output_kind == "scalar":
            total += 1
        else:
            total += 8  # conservative guess for undeclared vector/global width
    return max(total, 1)


def plan_batch_rows(
    *,
    schema_n_columns: int,
    specs,
    budget: ResourceBudget,
    override: int | None = None,
    itemsize: int = 8,
    safety_multiplier: float = 4.0,
    min_rows: int = 256,
    max_rows: int = 2_000_000,
    bytes_per_row_in: int | None = None,
) -> int:
    """Choose a bounded batch size for the Step 2 out-of-core engine.

    Generalizes :func:`choose_batch_rows` to arbitrary selected features (not
    only the legacy tabular auto-planner) by estimating output width from
    registry metadata. ``override`` lets a caller pin an exact batch size,
    bypassing estimation (and the budget check) entirely -- an explicit
    ``batch_rows=`` is a deliberate user override, distinct from automatic
    planning, and is honored as given rather than silently adjusted.

    ``min_rows``/``max_rows`` bound the *automatically planned* size only:
    ``min_rows`` is a throughput preference the budget-implied estimate is
    allowed to fall below (down to 1 row) when the declared ``memory_budget``
    genuinely cannot afford it -- see :func:`choose_batch_rows`.

    ``bytes_per_row_in`` (fourth audit pass, Finding 10) is a real,
    source-reported deep-memory estimate of one input row (see
    ``Source.estimated_bytes_per_row``); when given, it replaces the
    ``schema_n_columns * itemsize`` guess, which is only accurate for
    fixed-width numeric columns and can undercount a variable-length
    string/object column by orders of magnitude.
    """
    if override is not None:
        if override < 1:
            raise ValueError("batch_rows override must be >= 1")
        return override
    n_out = estimate_output_width(specs)
    return choose_batch_rows(
        n_columns_in=schema_n_columns,
        n_columns_out=n_out,
        budget=budget,
        itemsize=itemsize,
        safety_multiplier=safety_multiplier,
        min_rows=min_rows,
        max_rows=max_rows,
        bytes_per_row_in=bytes_per_row_in,
    )


class BatchTooSmallError(RuntimeError):
    """Raised when a batch cannot be halved further and still fails."""


class AdaptiveBatchController:
    """Generic halve-on-MemoryError batch-size policy, bounded so it can never loop forever.

    Used by the execution engine: on a recognized memory-allocation failure
    for a batchable/streamable feature, call :meth:`halve` and retry with the
    smaller size. When :meth:`halve` returns ``False`` the floor has been
    reached and the caller must fall back to feature-level isolation instead
    of retrying again.
    """

    def __init__(self, initial_batch_rows: int, *, min_batch_rows: int = 1, max_halvings: int = 30) -> None:
        if initial_batch_rows < 1:
            raise ValueError("initial_batch_rows must be >= 1")
        self.batch_rows = initial_batch_rows
        self.min_batch_rows = max(min_batch_rows, 1)
        self.max_halvings = max_halvings
        self.halvings = 0

    def halve(self) -> bool:
        if self.batch_rows <= self.min_batch_rows or self.halvings >= self.max_halvings:
            return False
        self.batch_rows = max(self.batch_rows // 2, self.min_batch_rows)
        self.halvings += 1
        return True

    def reset(self, batch_rows: int) -> None:
        """Restore a larger batch size (e.g. for the next, independent batch)."""
        self.batch_rows = max(batch_rows, self.min_batch_rows)


def current_process_rss_bytes() -> int:
    """Best-effort current process RSS, for reporting only.

    This is *not* the same quantity as a configured working-memory budget: it
    includes the interpreter, every imported library, and anything else the
    host process is holding, not just the executor's own buffers. See
    ``docs/SCALING.md``.
    """
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def current_process_peak_rss_bytes() -> int:
    """Best-effort *peak* process RSS since process start, for reporting only.

    On Windows, ``peak_wset`` (peak working set) is tracked natively by the OS
    and reported by psutil with no need to poll -- a true peak, not a
    before/after snapshot that could miss a spike in between. Falls back to
    the current RSS on platforms without a native peak counter.
    """
    try:
        mem = psutil.Process(os.getpid()).memory_info()
        return int(getattr(mem, "peak_wset", mem.rss))
    except Exception:
        return 0
