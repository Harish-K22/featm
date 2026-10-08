from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Callable, Iterable, Sequence

import numpy as np


class Cost(str, Enum):
    TINY = "tiny"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


_COST_WEIGHT = {
    Cost.TINY: 1,
    Cost.LOW: 2,
    Cost.MEDIUM: 5,
    Cost.HIGH: 20,
    Cost.EXTREME: 100,
}


class Complexity(str, Enum):
    """Coarse computational complexity class, used by the planner only as a hint."""

    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n^2)"
    CUBIC = "O(n^3)"
    EXPONENTIAL = "O(2^n)"
    UNKNOWN = "unknown"


class MemoryProfile(str, Enum):
    """Coarse working-memory growth profile relative to input size."""

    CONSTANT = "constant"
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    QUADRATIC = "quadratic"
    UNBOUNDED = "unbounded"
    UNKNOWN = "unknown"


class BatchSemantics(str, Enum):
    """Explicit Source/out-of-core batching contract for one feature (fourth audit pass).

    ``batchable``/``requires_full_materialization`` say *whether* a feature
    may be split across batches, but not *why*, and prior audit passes found
    real silent-wrong-answer bugs in features that inherited ``batchable``
    from ``requires_full_materialization=False`` even though their
    mathematics genuinely depends on the whole collection/corpus, or on a
    per-sample structure a row-oriented ``Source`` cannot represent at all.
    This enum makes that reasoning an explicit, auditable field instead of
    something only visible by reading the implementation.

    - ``ROW_INDEPENDENT`` -- one Source row is one independent sample; safe
      to batch arbitrarily. The historical default, still correct for the
      overwhelming majority of the registry (tabular, most timeseries/
      signal/event/sequence/information features).
    - ``DATASET_CONTEXT_ROW_ALIGNED`` -- output is row-aligned, but the
      correct value for one row genuinely depends on dataset/corpus-wide
      context (vocabulary, IDF, global counts, a fitted reference). Only
      actually safe out-of-core when the feature is also ``stateful=True``
      (fit corpus-wide state once, transform with it frozen) or declares
      ``required_params_for_batching`` that pin down what would otherwise be
      an implicit, batch-local default. This label is descriptive/auditable;
      the real enforcement is via those other fields.
    - ``FULL_DATASET`` -- equivalent to ``requires_full_materialization=True``:
      the whole input is genuinely one entity (e.g. a point cloud, a corpus
      that must be seen at once to fit a topic model). Recorded here too so
      the batch-semantics audit/harness has one field to report on.
    - ``STRUCTURED_SAMPLE`` -- one logical sample has a nested shape (e.g.
      ``(2, t)`` for a cross-series pair, a ``{"image": ..., "text": ...}``
      dict) that a row-oriented ``Source`` batch (rows of scalar/array
      columns) cannot represent at all, regardless of batch size -- not even
      growing the batch to cover the whole dataset helps, unlike
      ``FULL_DATASET``. The out-of-core engine rejects selecting a feature
      with this classification unconditionally, with an actionable message;
      it remains fully usable via the small-data in-memory ``extract()``
      path where the caller passes the correctly-shaped structure directly.
    - ``OPAQUE_OBJECT_SAMPLE`` -- one row is one independent, non-numeric
      Python object (a ``networkx.Graph``, or a real per-row ``numpy.ndarray``
      such as one loaded image/audio clip/video clip) that only an in-memory
      ``PandasSource`` with pre-loaded objects can represent -- never a
      file-backed source, which cannot deserialize arbitrary Python objects
      into a cell. Actively enforced (fifth audit pass, Finding 1), not just
      descriptive: the out-of-core engine rejects selecting an
      ``OPAQUE_OBJECT_SAMPLE`` feature against anything other than a
      ``PandasSource``, and adapts the resolved column's per-row objects for
      the feature -- dense ``numpy.ndarray`` samples of a consistent shape
      are stacked into the batch tensor the feature's small-data calling
      convention expects (``(n, ...)``), and any other object type (e.g. a
      ``networkx.Graph``) is passed through as an iterable unchanged, matching
      the small-data ``extract()`` convention exactly. Ragged/inconsistent
      per-row array shapes within one batch fail clearly rather than being
      silently coerced. See ``docs/SCALING.md``'s modality-support table.
    - ``GLOBAL_OUTPUT`` -- the feature's output is not row-aligned with the
      source at all (reserved for future use; no current registration needs
      it -- every registered feature today produces one row of output per
      logical input sample).
    - ``UNAUDITED`` -- default; ``FeatureSpec.__post_init__`` derives a
      concrete value from ``requires_full_materialization``/``stateful``
      (never leaves this classification in place on a constructed spec), so
      no feature is ever actually left unclassified at runtime -- but the
      *derived* value inherits the same "assume row-independent" prior the
      first three audit passes already validated for row-independent
      families, not a promise that every one of the 552 registrations has
      been individually hand-audited for this specific structured-sample
      risk. The registry-wide audit (this pass) explicitly curates the
      families found to need a different classification; the automated
      batch-equivalence harness (``benchmarks/batch_equivalence_smoke.py``)
      exists specifically to catch anything that curation missed.
    """

    ROW_INDEPENDENT = "row_independent"
    DATASET_CONTEXT_ROW_ALIGNED = "dataset_context_row_aligned"
    FULL_DATASET = "full_dataset"
    STRUCTURED_SAMPLE = "structured_sample"
    OPAQUE_OBJECT_SAMPLE = "opaque_object_sample"
    GLOBAL_OUTPUT = "global_output"
    UNAUDITED = "unaudited"


# Sensible defaults so authors do not need to hand-annotate every trivial feature.
# These are intentionally coarse -- they exist for planning/reporting, not billing.
_DEFAULT_COMPLEXITY_BY_COST = {
    Cost.TINY: Complexity.LINEAR,
    Cost.LOW: Complexity.LINEAR,
    Cost.MEDIUM: Complexity.LINEARITHMIC,
    Cost.HIGH: Complexity.QUADRATIC,
    Cost.EXTREME: Complexity.CUBIC,
}
_DEFAULT_MEMORY_PROFILE_BY_COST = {
    Cost.TINY: MemoryProfile.CONSTANT,
    Cost.LOW: MemoryProfile.LINEAR,
    Cost.MEDIUM: MemoryProfile.LINEAR,
    Cost.HIGH: MemoryProfile.LOG_LINEAR,
    Cost.EXTREME: MemoryProfile.QUADRATIC,
}

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*(\.[a-z0-9][a-z0-9_]*){1,}$")
_VALID_OUTPUT_KINDS = {"scalar", "vector", "global", "transformer"}
_VALID_FAILURE_HINTS = {"isolate", "critical"}


class RegistryValidationError(ValueError):
    """Raised when a FeatureSpec's execution metadata is malformed."""


@dataclass
class FeatureOutput:
    """Explicit multi-column result from one selected feature.

    ``values`` must be shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.
    ``names`` are suffixes local to the selected feature. For example a MFCC
    feature can return names ``("c00", ..., "c12")`` and the public extractor
    expands those to ``audio.perceptual.mfcc__c00`` etc.
    """

    values: Any
    names: Sequence[str] | None = None

    def as_array(self) -> np.ndarray:
        arr = np.asarray(self.values)
        if arr.ndim not in (1, 2):
            raise ValueError("FeatureOutput values must be 1D or 2D")
        if self.names is not None and arr.ndim == 2 and len(self.names) != arr.shape[1]:
            raise ValueError("FeatureOutput names length must match output width")
        return arr


@dataclass(frozen=True)
class FeatureSpec:
    """Unified execution contract for one registered, selectable feature.

    Every field has a sensible default so existing/simple registrations do not
    need to hand-annotate metadata that the executor can infer from ``cost``.
    Only fields relevant to a particular feature need to be set explicitly.
    """

    name: str
    family: str
    modality: str
    input_types: tuple[str, ...] = ()
    arity: int = 1
    output_kind: str = "scalar"  # scalar | vector | global | transformer
    cost: Cost = Cost.LOW
    streamable: bool = True
    incremental: bool = False
    approximate: bool = False
    gpu: bool = False
    min_rows: int = 1
    max_rows_exact: int | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    reference: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict, compare=False)
    implementation: Callable[..., Any] | None = field(default=None, compare=False, repr=False)

    # --- Step 2: unified execution-planning metadata (all optional, all defaulted) ---
    stable_id: str | None = None
    subfamily: str = ""
    output_names: tuple[str, ...] | None = None
    stateful: bool = False
    requires_full_materialization: bool = False
    batchable: bool | None = None
    complexity: Complexity = Complexity.UNKNOWN
    memory_profile: MemoryProfile = MemoryProfile.UNKNOWN
    cache_key: str | None = None
    exact_available: bool = True
    approximate_available: bool = False
    approximate_max_exact_rows: int | None = None
    approximate_notes: str = ""
    failure_policy_hint: str = "isolate"
    #: Optional, curated, per-feature working-memory estimator for
    #: ``requires_full_materialization=True`` features: ``(n_rows) -> estimated
    #: peak working bytes``. Only meaningful when the *actual* algorithmic memory
    #: growth (e.g. an O(n^2) dense pairwise-distance matrix) is known and has
    #: been verified against the implementation -- do not set this from a guess.
    #: When unset, the planner falls back to a conservative complexity-derived
    #: heuristic (see ``resources.estimate_full_materialization_bytes``), which
    #: is documented as a coarse, unaudited default, not a verified guarantee.
    memory_estimator: Callable[[int], int] | None = field(default=None, compare=False, repr=False)
    #: Fourth audit pass: explicit Source/out-of-core batching contract. See
    #: :class:`BatchSemantics`. Defaults to ``UNAUDITED``, which
    #: ``__post_init__`` immediately resolves to a concrete classification
    #: derived from ``requires_full_materialization``/``stateful`` -- so
    #: ``spec.batch_semantics`` is never actually ``UNAUDITED`` once a
    #: ``FeatureSpec`` is constructed; that value only exists as the
    #: "not yet explicitly curated" input state.
    batch_semantics: BatchSemantics = BatchSemantics.UNAUDITED
    #: Fourth audit pass: params that MUST be explicitly supplied (non-None,
    #: via ``params={spec.name: {...}}``) for this feature to be treated as
    #: row-independent/safe across MULTIPLE out-of-core batches -- for a
    #: feature whose *default* (param-absent) behavior implicitly depends on
    #: "the whole collection" (e.g. an implicit reference/anchor chosen from
    #: whatever the current call happens to see first), which differs
    #: per-batch and is therefore unsafe to batch silently. A single-batch
    #: run (the whole source fits in one batch) is unaffected -- there, the
    #: implicit default is not batch-dependent in the first place, and
    #: matches small-data ``extract()`` behavior exactly.
    required_params_for_batching: tuple[str, ...] = ()

    # --- Step 3: GPU execution metadata (all optional, all defaulted; a
    # feature with none of these set is simply not GPU-accelerated -- the
    # historical/default state for all but a curated subset of the registry) ---
    #: Whether a real, verified GPU kernel exists for this feature. This is
    #: the field the executor actually branches on (via
    #: ``featm.gpu.dispatch.gpu_capable``, which additionally requires the
    #: kernel to actually be registered in ``featm.gpu.kernels`` -- so
    #: ``gpu_available=True`` alone can never silently claim GPU support with
    #: nothing behind it; ``validate_spec`` also cross-checks this, see below).
    gpu_available: bool = False
    #: Which GPU backend the kernel is ACTUALLY written against -- ``"cupy"``
    #: for every kernel in this milestone. Actively ENFORCED by
    #: ``featm.gpu.dispatch.dispatch_one_feature`` (audit-fix pass, Finding
    #: 4): dispatch requires exactly this backend to be available and never
    #: silently substitutes a different one (e.g. Torch) merely because it
    #: happens to be present, since a CuPy kernel's array-API calls are not
    #: guaranteed to behave identically -- or even exist -- under a different
    #: library. Also surfaced in the GPU capability inventory for humans.
    gpu_backend: str | None = None
    #: Row-count threshold below which ``backend="auto"`` uses CPU even
    #: though a GPU is available and this feature is GPU-capable -- avoids
    #: the "move a 10-row array to the GPU" transfer-overhead trap (Step 3
    #: spec §15). ``None`` falls back to a family-appropriate default set at
    #: registration time (see ``featm.gpu.registration``).
    gpu_min_rows: int | None = None
    #: Audit-fix pass (Finding 10): element-count (``n_rows * n_columns``)
    #: threshold for ``backend="auto"``, for features whose per-row cost
    #: depends materially on sequence length/width, not row count alone (a
    #: row-only threshold would wrongly force a few-rows/huge-width workload
    #: onto CPU, and wrongly send a many-rows/tiny-width workload to the
    #: GPU). ``None`` derives this from ``gpu_min_rows`` at dispatch time --
    #: see ``featm.gpu.memory.estimate_gpu_min_elements``. Set explicitly only
    #: when a feature's real calibration data justifies a different constant
    #: than the shared derivation.
    gpu_min_elements: int | None = None
    #: (rtol, atol) for the CPU-vs-GPU equivalence check performed by
    #: ``featm.gpu.dispatch.dispatch_one_feature(..., validate=True)`` (used
    #: by tests and the Kaggle validation notebook, not by ordinary
    #: production dispatch, which does not pay for a redundant CPU run).
    #: Actively enforced wherever ``validate=True`` is used -- never merely
    #: decorative. Defaults reflect float32-GPU-vs-float64-CPU precision
    #: (Step 3 spec §8), not an arbitrarily loose "make tests pass" value.
    gpu_numerical_tolerance: tuple[float, float] = (1e-4, 1e-6)
    #: The dtype a GPU kernel actually computes in. Every kernel in this
    #: milestone computes in float32 (see ``featm/gpu/kernels/*.py`` module
    #: docstrings) -- host float64 input is cast down explicitly, once, at
    #: the device-transfer boundary, never silently, and the *output* is
    #: cast back to float64 before being handed to the caller so a GPU vs
    #: CPU run of the same call is never distinguishable by dtype alone.
    gpu_output_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.stable_id is None:
            object.__setattr__(self, "stable_id", self.name)
        if self.complexity is Complexity.UNKNOWN:
            object.__setattr__(self, "complexity", _DEFAULT_COMPLEXITY_BY_COST[self.cost])
        if self.memory_profile is MemoryProfile.UNKNOWN:
            object.__setattr__(self, "memory_profile", _DEFAULT_MEMORY_PROFILE_BY_COST[self.cost])
        if self.batchable is None:
            # A feature that requires the whole input materialized at once
            # cannot sensibly be split into batches.
            object.__setattr__(self, "batchable", not self.requires_full_materialization)
        if self.batch_semantics is BatchSemantics.UNAUDITED:
            if self.requires_full_materialization:
                derived = BatchSemantics.FULL_DATASET
            elif self.stateful:
                derived = BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED
            else:
                derived = BatchSemantics.ROW_INDEPENDENT
            object.__setattr__(self, "batch_semantics", derived)

    @property
    def cost_weight(self) -> int:
        return _COST_WEIGHT[self.cost]

    @property
    def executable(self) -> bool:
        return self.implementation is not None

    @property
    def min_samples(self) -> int:
        """Alias of ``min_rows`` using Step 2 terminology."""
        return self.min_rows


def validate_spec(spec: FeatureSpec) -> None:
    """Validate one FeatureSpec's execution metadata, raising early on defects.

    This intentionally checks structural/typing invariants of the unified
    execution contract, not feature-specific numerical correctness.
    """
    errors: list[str] = []

    if not spec.name or not _NAME_RE.match(spec.name):
        errors.append(
            f"name {spec.name!r} must be a dotted lowercase identifier with at "
            "least 2 segments, e.g. 'modality.family.feature'"
        )
    if not spec.family:
        errors.append("family must be non-empty")
    if not spec.modality:
        errors.append("modality must be non-empty")
    if spec.output_kind not in _VALID_OUTPUT_KINDS:
        errors.append(f"output_kind={spec.output_kind!r} must be one of {sorted(_VALID_OUTPUT_KINDS)}")
    if spec.arity < 1:
        errors.append(f"arity must be >= 1, got {spec.arity}")
    if spec.min_rows < 0:
        errors.append(f"min_rows must be >= 0, got {spec.min_rows}")
    if spec.max_rows_exact is not None and spec.max_rows_exact < spec.min_rows:
        errors.append("max_rows_exact must be >= min_rows when set")
    if not isinstance(spec.cost, Cost):
        errors.append(f"cost must be a Cost enum member, got {spec.cost!r}")
    if not isinstance(spec.complexity, Complexity):
        errors.append(f"complexity must be a Complexity enum member, got {spec.complexity!r}")
    if not isinstance(spec.memory_profile, MemoryProfile):
        errors.append(f"memory_profile must be a MemoryProfile enum member, got {spec.memory_profile!r}")
    if spec.failure_policy_hint not in _VALID_FAILURE_HINTS:
        errors.append(f"failure_policy_hint={spec.failure_policy_hint!r} must be one of {sorted(_VALID_FAILURE_HINTS)}")
    if not spec.exact_available and not spec.approximate_available:
        errors.append("a feature must support at least one of exact_available or approximate_available")
    if spec.approximate_max_exact_rows is not None and spec.approximate_max_exact_rows < 0:
        errors.append("approximate_max_exact_rows must be >= 0 when set")
    if not spec.approximate_available and spec.approximate_notes:
        errors.append("approximate_notes set but approximate_available is False")
    if spec.output_names is not None and spec.output_kind not in ("vector", "global"):
        errors.append("output_names may only be set when output_kind is 'vector' or 'global'")
    if not isinstance(spec.tags, tuple) or not all(isinstance(t, str) for t in spec.tags):
        errors.append("tags must be a tuple of str")
    if not isinstance(spec.requires, tuple) or not all(isinstance(r, str) for r in spec.requires):
        errors.append("requires must be a tuple of str")
    if spec.memory_estimator is not None and not callable(spec.memory_estimator):
        errors.append("memory_estimator must be callable (n_rows) -> estimated bytes")
    if spec.memory_estimator is not None and spec.batchable:
        errors.append("memory_estimator is only meaningful for requires_full_materialization=True (not batchable) features")
    if not isinstance(spec.batch_semantics, BatchSemantics):
        errors.append(f"batch_semantics must be a BatchSemantics enum member, got {spec.batch_semantics!r}")
    elif spec.batch_semantics is BatchSemantics.UNAUDITED:
        errors.append("batch_semantics must not remain UNAUDITED after __post_init__ (internal invariant)")
    if not isinstance(spec.required_params_for_batching, tuple) or not all(isinstance(p, str) for p in spec.required_params_for_batching):
        errors.append("required_params_for_batching must be a tuple of str")
    if spec.required_params_for_batching and spec.batch_semantics is BatchSemantics.STRUCTURED_SAMPLE:
        errors.append("required_params_for_batching is meaningless on a STRUCTURED_SAMPLE feature, which is rejected unconditionally out-of-core")
    if spec.batch_semantics is BatchSemantics.FULL_DATASET and not spec.requires_full_materialization:
        errors.append("batch_semantics=FULL_DATASET must be paired with requires_full_materialization=True")
    if spec.batch_semantics is BatchSemantics.STRUCTURED_SAMPLE and spec.requires_full_materialization:
        errors.append("batch_semantics=STRUCTURED_SAMPLE (rejected unconditionally) and requires_full_materialization=True (grown to one whole-dataset batch) are mutually exclusive strategies -- choose one")
    # Fifth audit pass, Finding 5: DATASET_CONTEXT_ROW_ALIGNED and
    # stateful=True must imply each other. DATASET_CONTEXT_ROW_ALIGNED's
    # only currently-supported enforcement mechanism is the StatefulEncoder
    # fit/transform protocol (a mergeable accumulator frozen before
    # transform) -- claiming that classification with stateful=False leaves
    # nothing actually enforcing the "dataset-context" dependency out-of-core
    # (the feature would be run as a plain per-batch direct call, exactly
    # the silent-per-batch-recompute bug the fourth audit pass's Finding 1
    # closed). Symmetrically, stateful=True means this feature's out-of-core
    # safety literally depends on being driven through that same fit/
    # transform protocol; claiming ROW_INDEPENDENT/FULL_DATASET/
    # STRUCTURED_SAMPLE/OPAQUE_OBJECT_SAMPLE instead is not a "carefully
    # documented exception", it is an internally contradictory registration
    # -- the engine only invokes the fit/transform protocol for specs it
    # finds via `stateful and name in STATEFUL_ENCODERS`, and every other
    # classification either bypasses that protocol (ROW_INDEPENDENT/
    # OPAQUE_OBJECT_SAMPLE run a plain per-batch call) or is nonsensical for
    # a feature whose whole point is a *mergeable* dataset-context fit
    # (FULL_DATASET/STRUCTURED_SAMPLE describe a different reason a feature
    # cannot be split across batches at all).
    if spec.batch_semantics is BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED and not spec.stateful:
        errors.append(
            "batch_semantics=DATASET_CONTEXT_ROW_ALIGNED requires stateful=True -- its only supported "
            "out-of-core enforcement mechanism is the StatefulEncoder fit/transform protocol "
            "(featm.state.StatefulEncoder, registered in STATEFUL_ENCODERS); a non-stateful feature "
            "cannot safely claim its output depends on dataset-wide context out-of-core"
        )
    if spec.stateful and spec.batch_semantics is not BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED:
        errors.append(
            f"stateful=True requires batch_semantics=DATASET_CONTEXT_ROW_ALIGNED (or leaving batch_semantics "
            f"unset, which derives it automatically); got {spec.batch_semantics.value!r}. A stateful feature "
            "claiming ROW_INDEPENDENT/FULL_DATASET/STRUCTURED_SAMPLE/OPAQUE_OBJECT_SAMPLE batch semantics "
            "is an internally contradictory registration -- the out-of-core engine only drives the "
            "fit/transform protocol for stateful features, so any other classification either silently "
            "bypasses it or is incoherent with what 'stateful' means"
        )
    # OPAQUE_OBJECT_SAMPLE (fifth audit pass, Finding 1) is only meaningful
    # for a feature the executor can actually adapt: either a dense
    # per-row ndarray/list sample it can stack into a batch tensor (image/
    # audio/video), or an arbitrary pre-loaded Python object it passes
    # through unchanged as an iterable (graph/text-like conventions). It is
    # incoherent to also claim requires_full_materialization=True (there is
    # no whole-dataset-at-once variant of the object-column contract -- see
    # STRUCTURED_SAMPLE for "needs more context than one row" and
    # FULL_DATASET for "needs the whole dataset at once"; OPAQUE_OBJECT_SAMPLE
    # is specifically "one row is one independent object", same batchability
    # story as ROW_INDEPENDENT, just a different Source-representation rule).
    if spec.batch_semantics is BatchSemantics.OPAQUE_OBJECT_SAMPLE and spec.requires_full_materialization:
        errors.append(
            "batch_semantics=OPAQUE_OBJECT_SAMPLE (one row is one independent pre-loaded object) and "
            "requires_full_materialization=True (needs the whole dataset at once) describe different, "
            "mutually exclusive reasons a feature cannot use ordinary row-oriented batching -- choose one"
        )
    if spec.batch_semantics is BatchSemantics.OPAQUE_OBJECT_SAMPLE and spec.stateful:
        errors.append(
            "batch_semantics=OPAQUE_OBJECT_SAMPLE and stateful=True are not currently supported together -- "
            "no registered OPAQUE_OBJECT_SAMPLE feature needs dataset-wide fitted state; if one genuinely "
            "does, it should be reconsidered as DATASET_CONTEXT_ROW_ALIGNED with an object-aware "
            "StatefulEncoder rather than combining both labels on one spec"
        )

    # --- Step 3: GPU metadata coherence. The deeper check ("is there
    # actually a registered kernel for this name") cannot safely happen here
    # -- featm.gpu.kernels populates its kernel dicts as a SEPARATE step
    # after the base registry is built (same reason STATEFUL_ENCODERS is not
    # cross-checked here either: at the moment one FeatureSpec is being
    # validated, during that feature family's own module import, the GPU
    # kernel-registration step may not have run yet). That check lives in
    # featm.gpu.registration.validate_gpu_registry(), called explicitly
    # after all imports complete (see tests/test_gpu_metadata.py). What IS
    # checked here is everything derivable from the spec's own fields alone.
    if spec.gpu_available and spec.gpu_backend is None:
        errors.append("gpu_available=True requires gpu_backend to be set (e.g. 'cupy') -- naming which backend the kernel targets")
    if not spec.gpu_available and spec.gpu_backend is not None:
        errors.append("gpu_backend is set but gpu_available=False -- a GPU backend name with no GPU support claimed is contradictory")
    if not isinstance(spec.gpu_numerical_tolerance, tuple) or len(spec.gpu_numerical_tolerance) != 2:
        errors.append("gpu_numerical_tolerance must be a (rtol, atol) tuple of two floats")
    else:
        rtol, atol = spec.gpu_numerical_tolerance
        if not (isinstance(rtol, (int, float)) and isinstance(atol, (int, float))) or rtol < 0 or atol < 0:
            errors.append(f"gpu_numerical_tolerance=(rtol={spec.gpu_numerical_tolerance[0]!r}, atol={spec.gpu_numerical_tolerance[1]!r}) must be non-negative numbers")
    if spec.gpu_min_rows is not None and spec.gpu_min_rows < 0:
        errors.append(f"gpu_min_rows must be >= 0 when set, got {spec.gpu_min_rows}")
    if spec.gpu_min_elements is not None and spec.gpu_min_elements < 0:
        errors.append(f"gpu_min_elements must be >= 0 when set, got {spec.gpu_min_elements}")
    if spec.gpu_output_dtype not in ("float32", "float64"):
        errors.append(f"gpu_output_dtype={spec.gpu_output_dtype!r} must be 'float32' or 'float64'")
    if spec.gpu_available and spec.batch_semantics is BatchSemantics.STRUCTURED_SAMPLE:
        errors.append(
            "gpu_available=True on a STRUCTURED_SAMPLE feature is incoherent -- the Step 3 GPU engine "
            "reuses the exact same row-oriented Source representation as Step 2 and cannot represent a "
            "structured per-sample input any more than the CPU out-of-core engine can; a STRUCTURED_SAMPLE "
            "feature remains usable via GPU only through featm.extract() on in-memory data with no Source "
            "batching involved, which does not need gpu_available metadata at all"
        )
    if spec.gpu_available and not spec.batchable and spec.batch_semantics is not BatchSemantics.FULL_DATASET:
        errors.append(
            "gpu_available=True with batchable=False must be paired with batch_semantics=FULL_DATASET -- "
            "any other not-batchable classification is a contradiction this registry already rejects "
            "elsewhere; restating it here so a GPU-specific reviewer sees the same rule"
        )

    if errors:
        joined = "; ".join(errors)
        raise RegistryValidationError(f"Invalid FeatureSpec for {spec.name!r}: {joined}")


class FeatureRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._specs:
            raise ValueError(f"Feature already registered: {spec.name}")
        validate_spec(spec)
        self._specs[spec.name] = spec
        return spec

    def validate_all(self) -> None:
        """Re-validate every registered spec; raises on the first defect found."""
        for spec in self._specs.values():
            validate_spec(spec)

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            matches = self.search(name)
            hint = f" Did you mean: {', '.join(x.name for x in matches[:5])}?" if matches else ""
            raise KeyError(f"Unknown feature: {name}.{hint}") from exc

    def all(self) -> list[FeatureSpec]:
        return sorted(self._specs.values(), key=lambda x: x.name)

    def search(
        self,
        query: str | None = None,
        *,
        modality: str | None = None,
        family: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[FeatureSpec]:
        q = (query or "").lower()
        tagset = {t.lower() for t in (tags or [])}
        out: list[FeatureSpec] = []
        wildcard = any(ch in q for ch in "*?[")
        for spec in self._specs.values():
            haystack = " ".join((spec.name, spec.family, spec.description, *spec.tags)).lower()
            if q:
                if wildcard:
                    if not (fnmatch(spec.name.lower(), q) or fnmatch(haystack, q)):
                        continue
                elif q not in haystack:
                    continue
            if modality and spec.modality != modality:
                continue
            if family and spec.family != family:
                continue
            if tagset and not tagset.issubset({x.lower() for x in spec.tags}):
                continue
            out.append(spec)
        return sorted(out, key=lambda x: x.name)

    def modalities(self) -> list[str]:
        return sorted({spec.modality for spec in self._specs.values()})

    def families(self, *, modality: str | None = None) -> list[str]:
        specs = self._specs.values()
        if modality is not None:
            specs = (x for x in specs if x.modality == modality)
        return sorted({x.family for x in specs})


REGISTRY = FeatureRegistry()


def register_feature(**kwargs: Any):
    """Decorator used by built-ins and plugins to register one selectable feature."""

    def deco(func: Callable[..., Any]):
        spec = FeatureSpec(implementation=func, **kwargs)
        REGISTRY.register(spec)
        return func

    return deco


def register_callable(func: Callable[..., Any], **kwargs: Any) -> FeatureSpec:
    """Programmatic equivalent of :func:`register_feature` for feature factories."""
    spec = FeatureSpec(implementation=func, **kwargs)
    return REGISTRY.register(spec)
