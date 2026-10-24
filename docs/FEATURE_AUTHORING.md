# Authoring a featm feature

The v0.1 contract is intentionally small. A feature is a callable plus metadata registered under a stable name.

## Scalar feature

```python
import numpy as np
from featm import Cost, register_feature

@register_feature(
    name="timeseries.statistical.example_mean_abs",
    modality="timeseries",
    family="statistical",
    input_types=("numeric",),
    output_kind="scalar",
    cost=Cost.TINY,
    description="Mean absolute value of each input series.",
    tags=("statistics", "magnitude"),
)
def mean_abs(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    return np.nanmean(np.abs(X), axis=1)
```

For sample-oriented modalities, the first dimension should represent independent samples and the returned first dimension must match it.

## Fixed-vector feature

Use `FeatureOutput` when one selectable feature naturally produces several dimensions.
This complete example computes three intensity summaries for a batch of grayscale images:

```python
import numpy as np
from featm import Cost, FeatureOutput, register_feature

@register_feature(
    name="example.vector_descriptor",
    modality="image",
    family="example",
    output_kind="vector",
    cost=Cost.LOW,
)
def descriptor(images):
    images = np.asarray(images, dtype=float)  # shape (n_images, height, width)
    values = np.column_stack([images.mean((1, 2)), images.min((1, 2)), images.max((1, 2))])
    return FeatureOutput(values, names=("a", "b", "c"))
```

`featm.extract` expands this to:

```text
example.vector_descriptor__a
example.vector_descriptor__b
example.vector_descriptor__c
```

## Tabular compatibility features

The legacy tabular planner expects registered tabular transforms to follow its fit/state calling convention.
This is a schematic interface; replace the placeholder before executing it:

```python
from featm import register_feature

@register_feature(
    name="tabular.numeric.example",
    modality="tabular",
    family="numeric",
    input_types=("numeric",),
    streamable=True,
)
def transform(series, *, fit=False, state=None):
    values = ...
    return values, state
```

New non-tabular registry features do not need those keyword arguments.

## Optional dependencies

Do not import heavy optional packages at top-level module import time. Import them inside the selected feature and raise a concise installation hint if unavailable.

This optional-import snippet is schematic: `some_optional_library` and the
feature body are placeholders.

```python
def feature(X):
    try:
        import some_optional_library
    except ImportError as exc:
        raise ImportError("Install 'some_optional_library' to use ...") from exc
    ...
```

Declare the dependency in `FeatureSpec.requires` as documentation/runtime metadata too.

## Metadata guidelines

- Use a stable dotted name: `modality.family.feature`.
- Keep `family` broad enough to be useful for browsing.
- Mark computational `cost` honestly.
- Use `output_kind="vector"` for a fixed descriptor vector.
- Set `requires` for optional packages/models.
- Put useful discovery terms in `tags` and `description`.
- Add a reference when the implementation follows a particular paper/standard.
- Do not mark a feature `streamable=True` when splitting observations changes the mathematical definition.

Metadata is validated at registration time (`featm.registry.validate_spec`); a
malformed spec raises `RegistryValidationError` immediately on import rather
than being silently accepted. This includes internal-coherence checks
between fields, not just per-field type/range checks (fifth audit pass,
Finding 5): `batch_semantics=DATASET_CONTEXT_ROW_ALIGNED` and `stateful=True`
must imply each other (a stateful feature claiming any other
`batch_semantics`, or a non-stateful feature claiming
`DATASET_CONTEXT_ROW_ALIGNED`, is rejected -- the out-of-core engine's only
supported enforcement mechanism for "depends on dataset-wide context" is the
`StatefulEncoder` fit/transform protocol, so the two claims must always
travel together); `batch_semantics=OPAQUE_OBJECT_SAMPLE` cannot also claim
`requires_full_materialization=True` or `stateful=True` (no registered
feature needs either combination, and both would be internally
contradictory with what `OPAQUE_OBJECT_SAMPLE` means). If you leave
`batch_semantics`/`stateful` at their defaults and only set one of them
explicitly when you have a real reason to, `__post_init__`'s automatic
derivation already keeps them coherent -- these checks exist to catch a
hand-written registration that overrides one field without updating the
other to match.

## Step 2 execution-planning metadata (optional, all sensibly defaulted)

You do not need to set any of these for a typical feature -- `complexity` and
`memory_profile` are inferred from `cost` if left unset, and `batchable`
defaults to `not requires_full_materialization`. Set them when they are
genuinely informative:

- `complexity=Complexity.QUADRATIC` (etc.) if you know a tighter bound than the `cost`-derived default.
- `cache_key="my_family.shared_thing"` if this feature depends on the same
  expensive intermediate as other selected features in its family -- see
  `timeseries.spectral.*` (`_utils.spectral_parts`) and
  `geometry.triangulation.*` (`_tri_data`) for the pattern: have the shared
  helper check `featm.cache.current_cache()` and call `.get_or_compute(key, ...)`,
  falling back to a plain call when there is no active cache scope (e.g. the
  helper is used standalone/in a test). Only set `cache_key` on features that
  actually route through that helper -- do not claim sharing that is not real.
- `approximate_available=True` with `approximate_max_exact_rows=...` and
  `approximate_notes="..."` only when you have implemented and tested a
  mathematically defensible approximation (see `featm/state.py`'s three
  stateful encoders for the only current example: reservoir-sampling the fit
  rows). Do not set this just to satisfy the interface.
- `requires_full_materialization=True` if the feature genuinely cannot be
  computed from anything less than the whole input at once (rare -- most
  features operate row-by-row or batch-by-batch already).

## Batch-safety metadata (`batch_semantics` / `required_params_for_batching`) -- read this before adding a whole-batch feature

The fourth independent audit pass found a real, previously-undiscovered class
of silent-wrong-answer bug: a feature marked (by default!) as safe to batch
whose actual mathematics depended on the *whole collection* it was called
with -- a corpus-wide word distribution, a `.rank()`/`.value_counts()` over
the whole column, an implicit "first row" reference, a structured `(2, t)`
pair a row-oriented `Source` cannot represent at all. **Before registering
any feature whose implementation is not a pure, independent function of
that one row/sample's own input, read this section and set the right
metadata explicitly** -- `batchable=True` is still the default, and an
unaudited default is not a safety guarantee.

Ask, in order:

1. **Does this feature's result for one sample depend on any OTHER sample in
   the same call (a corpus statistic, a whole-column rank/count/entropy, an
   implicit reference chosen from "whichever sample the call happens to see
   first/last", a running total, a lag/shift relative to the previous row)?**
   If the dependency is genuinely *mergeable* across batches (summable
   counts, running extrema, a document-frequency table -- see "Stateful
   (fit/transform) tabular features" below) -- implement it as a
   `StatefulEncoder` (`stateful=True`). If the dependency can be pinned down
   by an explicit, required, transform-time parameter instead of an implicit
   default (an explicit `reference=`, `reference_time=`, etc.) -- register
   `required_params_for_batching=("that_param",)`; the engine then rejects
   multi-batch execution when that param is left at its implicit default,
   rather than silently using a batch-local one (see
   `sequence.distance.edit_distance`/`event.timing.recency` for the pattern).
   If neither applies and the dependency genuinely needs the whole dataset
   at once (an NMF/ICA-style joint decomposition, a global normalization) --
   set `requires_full_materialization=True` (see "Step 2 execution-planning
   metadata" above); do **not** leave it as a plain row-independent feature
   that happens to work by coincidence on small inputs.
2. **Does one logical sample have a nested/structured shape** (a `(2, t)`
   cross-series pair, a `{"image": ..., "text": ...}` embedding dict, a
   `(n_datasets, n_observations, n_variables)` array) **that a row-oriented
   `Source` (rows of scalar/array columns) cannot represent at all**, at any
   batch size? Set `batch_semantics=BatchSemantics.STRUCTURED_SAMPLE`
   explicitly. This is a stronger statement than
   `requires_full_materialization=True`: growing the batch to cover the
   whole dataset does not help (the rows still are not independent
   structured samples), so the out-of-core engine rejects selecting this
   feature unconditionally, not just when the source does not fit in one
   batch. See `information.dependence.*`, `multimodal.*`,
   `timeseries.cross_series.*`, `signal.synchronization.phase_locking` for
   examples, and `docs/SCALING.md`'s modality-support table for the full
   picture of which modalities are/are not row-Source-representable at all.
3. **Is one logical sample a whole, independent, non-numeric Python object**
   (a real `numpy.ndarray` like a decoded image/audio clip/video clip, or a
   `networkx.Graph`) **that only an in-memory `PandasSource` column can hold
   pre-loaded** -- as opposed to something a row-oriented file format
   (CSV/JSONL/Parquet/Arrow) could ever represent in a cell? Set
   `batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE` explicitly. Unlike
   `STRUCTURED_SAMPLE`, this IS genuinely batch-safe -- each row is still one
   independent sample -- it just needs adaptation before reaching your
   implementation: a dense per-row `numpy.ndarray` (image/audio/video
   conventions) is stacked into the `(n, ...)` batch tensor your
   implementation's `np.asarray(x)` call already expects
   (`execution._adapt_object_column`); any other object type (a
   `networkx.Graph`) is passed through as an iterable unchanged, matching the
   small-data convention. The engine rejects selecting an
   `OPAQUE_OBJECT_SAMPLE` feature against anything but a `PandasSource`
   before any batch is read, and fails clearly (not by silently coercing) on
   ragged/inconsistent per-row shapes within one batch. See `image.*`/
   `audio.*`/`video.*`/`graph.*`'s `_reg()` helpers for the pattern (fifth
   audit pass, Finding 1 -- this classification existed in v0.2.4 but was
   purely descriptive; the engine had no code path that actually adapted an
   object column, so these modalities' documented "supported" claim was not
   true until this pass).
4. **Otherwise** -- a pure function of that one row/sample's own input,
   independent of any other row and of call order -- no new metadata is
   needed; `batch_semantics` derives to `ROW_INDEPENDENT` automatically.

**If your implementation is internally randomized (a random walk, an ensemble
decomposition, anything that draws from an RNG), seed that RNG from the
sample's own content, never from a shared generator created once outside your
per-sample loop.** A shared RNG's draw sequence for sample *i* depends on how
many other samples were processed before it in the same call -- exactly the
kind of hidden whole-collection dependency this section exists to rule out,
just expressed through RNG state instead of through the returned value
directly. `graph.embeddings.deepwalk`'s random-walk generator had exactly
this bug (fifth audit pass, Finding 2): a `numpy.random.Generator` created
once before the per-graph loop meant the same graph's embedding silently
differed depending on which other graphs shared its out-of-core batch. Fixed
with `featm.features._utils.content_seed(data: bytes) -> int`: hash the
sample's own canonical content (never Python's per-process-salted `hash()`,
which is not reproducible across runs) and seed a fresh `numpy.random.
default_rng(...)` per sample from it -- see `graph.full._graph_content_seed`
(canonical node/edge encoding) and `signal.full.sig_eemd` (raw array bytes
fed straight to `content_seed`) for two different canonicalization strategies
using the same shared primitive. `tabular.categorical.embeddings` already
used this exact pattern before this pass and was confirmed, not just
assumed, to be unaffected.

**Verify your assumption, do not just assert it.** The "whole-column"/RNG
bugs found across the fourth and fifth audit passes
(`tabular.numeric.rank`/`.percentile`/`.ecdf` via `pandas.Series.rank()`,
`tabular.categorical.entropy` via a whole-column entropy broadcast to every
row, `tabular.categorical.transition` via `.shift(1)` across a batch
boundary, `tabular.relational.entity_graph` via `.value_counts()`,
`signal.decomposition.ica` via a jointly-estimated `FastICA` unmixing
matrix, `graph.embeddings.deepwalk`'s shared-RNG bug above, and
`image.*`/`audio.*`/`video.*` simply never working at all from the
documented-supported `PandasSource` object-column path) were all discovered
empirically, not by reading metadata -- **run
`benchmarks/batch_equivalence_smoke.py`** (or add your feature's modality to
its per-modality sample builders if not yet covered, including the
`image`/`audio`/`video`/`graph` object-column builders added in the fifth
audit pass) to compare your feature's output across `batch_rows=1`, a small
value, and one batch covering the whole source; if the values differ, your
feature is not what you classified it as, regardless of what you intended.

## Stateful (fit/transform) tabular features

There are two different conventions for a tabular feature that needs fitted
state, and they are not interchangeable:

**Legacy convention** (`incremental=True`, `stateful` left `False`): the
calling convention is `fn(series, *, fit=False, state=None) -> (values, state)`,
and `fit=True` **recomputes state from whatever is passed to that one call**.
This is correct for the legacy `AutoFeatures` planner (which always fits on
the whole in-memory DataFrame in a single call) and for the small-data
`extract()` path, but the Step 2 out-of-core engine will refuse to run such a
feature across more than one batch (it would silently compute independent
per-batch-local statistics instead of one global fit) rather than produce a
wrong answer.

**Step 2 stateful protocol** (`stateful=True`): implement
`featm.state.StatefulEncoder`. This schematic class shows lifecycle signatures;
the `...` bodies do not implement a working encoder. `merge_states` is needed
only when supporting distributed fitting:

```python
from featm.state import StatefulEncoder

class MyEncoder(StatefulEncoder):
    def init_state(self, *, state_budget_bytes=None, spill_dir=None, params=None): ...  # fresh, empty accumulator
    def update_state(self, state, batch, *, params=None): ...  # fold the next batch into the accumulator
    def merge_states(self, left, right, *, params=None): ...  # distributed fitting: preserve partition order
    def finalize_state(self, state, *, params=None): ...  # freeze accumulator -> transform-ready state (default: identity)
    def transform_batch(self, batch, state, *, params=None): ...  # apply frozen state; never mutate it
```

`update_state` must accumulate batches; distributed `merge_states` must combine
partial accumulators in source partition order (not every encoder is commutative).
Use the existing implementations as references (see `ZScoreEncoder`'s Welford/Chan
combination, `MinMaxEncoder`'s running extrema, `FrequencyEncoder`'s summed
counts, and the text corpus-statistics encoders' mergeable term/document-
frequency accumulator -- `SurprisalEncoder`/`Bm25Encoder`/`TfidfEncoder`/
`SimilarityEncoder`) so the out-of-core engine can drive a real multi-batch
fit without ever materializing the whole training set. `params` (fourth
audit pass) is this feature's merged per-call `params={name: {...}}` dict,
threaded through every stage so an encoder can read call-time configuration
-- e.g. `Bm25Encoder`'s `query` (transform-time only; corpus statistics are
fit-time) or `TfidfEncoder`'s `max_features` (fit-time, decides which terms
the frozen vocabulary keeps); most encoders ignore it (default `None`).
`state_budget_bytes`, when given, is a real budget your accumulator should
enforce if its size can scale with input cardinality (see
`BoundedFrequencyCounter`'s real-per-key-byte-estimate pattern, not a fixed
per-item average -- Finding 11, fourth audit pass) -- raise
`featm.state.StateBudgetExceededError` before silently exceeding it, never
after. Register it with the existing `register_feature` decorator, wrapped
via `featm.state._stateful_adapter` so the *same* encoder also works as a
plain single-shot feature on the small-data path. This registration fragment
assumes the schematic `MyEncoder` above has been fully implemented;
`_stateful_adapter` is an internal integration helper, not a separate public API:

```python
from featm import register_feature
from featm.state import _stateful_adapter

register_feature(
    name="tabular.scaling.my_thing", family="scaling", modality="tabular",
    input_types=("numeric",), stateful=True, incremental=True, streamable=True,
)(_stateful_adapter(MyEncoder()))
```

and add it to `featm.state.STATEFUL_ENCODERS` so the out-of-core engine can
find it by name.

## Adding a GPU kernel for an existing feature (Step 3)

GPU acceleration is a *backend* for an existing registered feature, never a
new feature name -- there is no `gpu.timeseries.spectral_entropy`, only
`timeseries.spectral.spectral_entropy` with a GPU implementation available
via `backend="gpu"`/`"auto"`. To add one:

1. **Write the kernel** in `featm/gpu/kernels/reductions.py` (a plain
   elementwise/per-row reduction over a `(n_rows, n_steps)` array) or
   `.../spectral.py` (anything that can share the centered-FFT primitive via
   `gpu_spectral_parts()`), or a new kernel module if neither shared
   primitive fits. It must be a direct translation of the CPU
   implementation's exact formula (diff them side by side), not a
   reimplementation -- see any existing kernel's docstring for the pattern
   (each names precisely which CPU feature it mirrors). Accept `(a, xp)` (or
   `(a, xp, *, batch_id, **kwargs)` if it shares the GPU cache) where `xp` is
   the CuPy module -- never `import cupy` directly in the kernel body, so the
   same code stays import-safe if a future pass adds a Torch backend.
2. **Register it** in the appropriate `*_KERNELS` dict (`REDUCTION_KERNELS`
   / `SPECTRAL_KERNELS` / `STANDALONE_SPECTRAL_KERNELS`), keyed by the exact
   existing feature name.
3. **Add the metadata** at the feature's own `register_feature()`/`_reg()`
   call site (not retrofitted after the fact) -- `gpu_available=True,
   gpu_backend="cupy", gpu_min_rows=<a real, justified threshold>`.
   `gpu_backend` is **actively enforced** (audit-fix pass, v0.3.1): it must
   name the backend the kernel is ACTUALLY written against
   (`"cupy"` for every kernel today) -- dispatch requires exactly that
   backend to be available and will never silently substitute a different
   one, even if it happens to be present. If the feature's cost depends
   materially on sequence length (not just row count), also consider setting
   `gpu_min_elements` explicitly rather than relying on the shared
   `gpu_min_rows * 64` derivation (`featm.gpu.memory.
   estimate_gpu_min_elements`) -- see `docs/SCALING.md`'s "Backend API"
   section for what that derivation assumes. Pick `gpu_min_rows` from the
   same family of features already accelerated (see `docs/SCALING.md`'s "GPU
   execution (Step 3, audit-fix pass v0.3.1)" section for the current
   defaults), not an arbitrary guess -- and expect to recalibrate it from
   the Kaggle benchmark notebook once real timings exist.
4. **Declare a tolerance carefully -- per-COLUMN, not just per-feature.** If
   the default `gpu_numerical_tolerance=(1e-4, 1e-6)` is not appropriate for
   this specific formula, investigate and justify a wider one (never loosen
   merely to make a test pass -- see `docs/SCALING.md`'s "CPU is the
   mathematical reference" section). **Be especially careful with a vector
   output that mixes columns of genuinely different scale/kind** (e.g.
   magnitude and phase) -- `featm.gpu.dispatch._validate_against_cpu`
   applies ONE `(rtol, atol)` to the whole output vector via `np.allclose`,
   so a tolerance widened to accommodate one column's known instability can
   silently stop catching a real error in a DIFFERENT, otherwise-stable
   column. This is exactly how `timeseries.spectral.fft`/`.cepstrum` lost
   their GPU acceleration in the v0.3.1 audit-fix pass -- see
   the historical audit record Findings 5-6 for the full writeup and the
   adversarial test that caught it. If you cannot get every column within a
   single tight tolerance, prefer NOT accelerating the feature over widening
   the tolerance until the validator stops meaning anything.
5. **Estimate its GPU memory footprint honestly.** If the new kernel's family
   coexists with more device-resident arrays per row than a plain reduction
   (the way the shared-FFT spectral family does), update
   `featm.gpu.memory.estimate_gpu_bytes_per_row`'s family detection rather
   than letting the universal reduction multiplier silently underestimate
   it -- an underestimate defeats the point of `gpu_memory_budget=`
   (Finding 1, audit-fix pass): proactive VRAM-budget chunking would think a
   batch fits when it does not.
6. **Cross-check the registry**: `featm.gpu.registration.validate_gpu_registry()`
   must pass (catches a metadata/kernel-dict name mismatch in either
   direction) and `featm.REGISTRY.validate_all()` must still pass (catches
   metadata incoherence, e.g. `gpu_available=True` on a `STRUCTURED_SAMPLE`
   feature).
7. **Test it** the same way `tests/test_gpu_dispatch.py` tests the existing
   22: a fake-CuPy-backend local test for control flow (dispatch routing,
   OOM retry if the feature is split-safe, proactive VRAM chunking if
   relevant), plus adding the feature to the Kaggle validation notebook's
   Section D/E (CPU<->GPU equivalence, batch-partition equivalence) for the
   real-CUDA numerical check -- a local mocked-backend test alone is never
   sufficient to claim GPU support (see the historical audit record's
   explicit distinction between local-only and Kaggle-confirmed status for
   the discipline this project follows).

## Tests expected

At minimum, a new feature family should test:

- registry visibility;
- deterministic output shape;
- numerical sanity on a tiny example;
- degenerate input behavior;
- optional-dependency failure behavior when applicable;
- vector output names when applicable.

The repository also runs `benchmarks/smoke_universe.py`, which attempts every registered feature on small synthetic data and treats missing optional dependencies separately from unexpected failures.

For any feature selectable via the Step 2 out-of-core engine (i.e. any
non-`requires_full_materialization` feature reachable from a row-oriented
`Source`), also run `benchmarks/batch_equivalence_smoke.py` (fourth audit
pass, Finding 8) -- it must report zero `mismatches`/`row_alignment_failures`/
`schema_drift_failures` for your feature. If your feature's modality is not
yet covered by that script's per-modality sample builders, add one rather
than leaving your feature silently unexercised.
