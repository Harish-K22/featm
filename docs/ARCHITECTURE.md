# featm architecture

## Goal

`featm` is primarily a **registry of numerical feature representations**, inspired by the ergonomics of `timm`.

```text
                 FeatureRegistry
                       │
          ┌────────────┼────────────┐
          │            │            │
       browse        inspect      create
          │            │            │
          ▼            ▼            ▼
 list_features   feature_info   create_feature
                                      │
                                      ▼
                                  selected
                                  feature
```

For batch extraction:

```text
input samples
     │
     ▼
explicit feature names
     │
     ▼
FeatureSpec lookup
     │
     ├─ validate modality
     ├─ load optional backend only if needed
     └─ merge default/user parameters
     │
     ▼
feature implementations
     │
     ▼
scalar or FeatureOutput vector
     │
     ▼
DataFrame with deterministic names
```

There is deliberately no automatic `features="all"` path.

For large/out-of-core extraction (v0.2), the same registry and the same feature
implementations are reused by a second execution path -- see "Two execution
paths" below and `SCALING.md` for the full design.

## Feature registry contract

Every `FeatureSpec` is validated at registration time (`featm.registry.validate_spec`,
called from `FeatureRegistry.register`) against a unified execution-metadata
contract. Core v0.1 fields:

- canonical name; family; modality;
- accepted semantic input types; arity;
- output kind;
- coarse computational cost;
- streamability/incremental hints used by the legacy tabular planner;
- approximation/GPU capability flags;
- minimum row count / optional exact ceiling;
- description and tags;
- optional dependency requirements;
- reference/provenance field;
- default parameters;
- implementation callable.

v0.2 adds execution-planning fields, all optional and all sensibly defaulted
from `cost` so existing/trivial registrations never need to change:

- `stable_id`, `subfamily`, `output_names` (declared static output width when known);
- `stateful` (participates in the Step 2 two-pass fit/transform lifecycle -- see `SCALING.md`);
- `requires_full_materialization`, `batchable` (defaults to `not requires_full_materialization`);
- `complexity` (`Complexity` enum, e.g. `O(n)`/`O(n log n)`/`O(n^2)`) and `memory_profile`
  (`MemoryProfile` enum), both defaulted from `cost` if unset;
- `cache_key` -- documents which shared intermediate (if any) this feature depends on,
  e.g. `"timeseries.spectral_parts"`; `None` if it does not share one;
- `exact_available` / `approximate_available` / `approximate_max_exact_rows` / `approximate_notes`;
- `failure_policy_hint` (`"isolate"` or `"critical"`, documentation-only in this release).

`FeatureRegistry.validate_all()` re-validates every registered spec (used as a
registry-wide sanity test); malformed metadata is caught at import time via
`register()`, not silently accepted.

The implementation is never a catalogue-only placeholder in this universe.

## Scalar and vector outputs

A feature can return a one-dimensional result directly:

```python
return values
```

or a fixed-width vector:

```python
from featm import FeatureOutput

return FeatureOutput(
    values=matrix,
    names=("c00", "c01", "c02"),
)
```

The extractor expands the latter into deterministic columns such as:

```text
audio.perceptual.mfcc__c00
audio.perceptual.mfcc__c01
audio.perceptual.mfcc__c02
```

This is important for descriptors such as MFCC, HOG, Hu moments, spectral summaries, graph spectra, and learned embeddings.

## Optional dependencies

Optional scientific libraries are imported inside their feature implementations. Importing `featm` therefore does not require the complete scientific stack.

A feature that needs an unavailable backend raises a concise error when selected. `extract(..., errors="skip"|"warn")` can isolate that failure; `errors="raise"` is useful for development.

## Modality separation

`extract()` accepts explicit features from one modality at a time. Mixed-modality feature definitions belong in the `multimodal` namespace instead of silently coercing unrelated data types.

## Compatibility planner

`AutoFeatures` remains for earlier tabular experiments. It is deliberately secondary to the registry API and supports only bounded presets (`minimal`, `fast`, `standard`, `comprehensive`), never `all`.

## Two execution paths

`featm.extract()` dispatches to one of two implementations:

```text
featm.extract(X, features=[...], **kwargs)
     │
     ├─ plain array/DataFrame/list, no Step 2 kwargs
     │     -> v0.1 in-memory path (engine.py), returns pandas.DataFrame
     │        (byte-identical to v0.1 behavior)
     │
     └─ file path / Source, or memory_budget/disk_budget/output/
        batch_rows/checkpoint/cache_budget/columns/mode!="exact"
              -> out-of-core engine (execution.py), returns FeatureMatrix
                 with a .report (ExecutionReport)
```

Both paths share the same `FeatureRegistry`, the same feature implementation
callables, and the same shared intermediate cache (`featm/cache.py`) -- an
out-of-core run computes the identical per-batch result the in-memory path
would for that batch, just scanned from a `Source` in bounded pieces with a
resource planner, fault isolation, optional disk output, and checkpointing
layered around it. See `SCALING.md` for the complete design: sources,
`FeatureMatrix`, the resource planner and adaptive batching, per-feature
fault isolation, the shared cache, the exact/approximate protocol, the
stateful two-pass lifecycle, and checkpoint/resume.

## GPU execution backend (Step 3)

`featm/gpu/` is a self-contained package layered on top of both execution
paths, not a third path: `backend="gpu"`/`"auto"` still goes through
`engine.extract()`/`execution.run_extraction()` exactly as before, just with
`featm.gpu.dispatch.dispatch_one_feature()` deciding CPU-vs-GPU per selected
feature per batch. Nothing in `featm/gpu/` is imported at CUDA-touching
import time -- `featm/gpu/backend.py`'s probes (`cupy_available()`,
`torch_cuda_available()`) import CuPy/Torch lazily, inside a function body,
cached per-process, so `import featm` never imports a CUDA library or emits
a CUDA warning on a machine with no GPU extras installed.

```text
featm/gpu/
├── backend.py       lazy CuPy/Torch detection, device info, xp_module()
├── memory.py        GPUMemoryBudget (VRAM planner, separate from host RAM/disk/cache budgets)
├── cache.py         GPUIntermediateCache -- device-resident shared-intermediate cache, per-batch scoped
├── dispatch.py       dispatch_one_feature(): the CPU/GPU decision + CUDA-OOM halve-and-retry loop
├── exceptions.py     GPUUnavailableError / GPUImplementationMissingError / GPUOutOfMemoryError / ...
├── registration.py   validate_gpu_registry() (kernel <-> registry cross-check), gpu_capability_inventory()
└── kernels/
    ├── reductions.py  elementwise reductions (mean/std/min/max/median/range/rms/energy/...)
    └── spectral.py    shared-FFT spectral family (centroid/dominant_frequency/flatness/fft/...)
```

`dispatch_one_feature(spec, X, *, backend, gpu_fallback, device, gpu_memory_budget, params, batch_id, validate=False)`
is the single function both execution paths call -- `engine.py`'s small-data
loop and `execution.py`'s per-batch transform loop never diverge on backend
policy because they route through the identical function. It returns
`(result, BackendInfo)`, where `result` is exactly what the CPU
implementation would have returned (a plain array or a `FeatureOutput`) --
callers do not need backend-specific handling of the *result*, only of the
reporting.

A GPU-capable feature is one where `FeatureSpec.gpu_available=True` **and**
a real kernel exists in `featm.gpu.kernels`'s dicts
(`REDUCTION_KERNELS`/`SPECTRAL_KERNELS`/`STANDALONE_SPECTRAL_KERNELS`) **and**
`gpu_backend` names the backend the kernel actually requires (audit-fix pass,
v0.3.1: `FeatureSpec.gpu_backend` is now actively enforced by dispatch --
`gpu_available=True` alone is never sufficient, and neither is "some GPU
backend happens to be available" -- `featm.gpu.dispatch.gpu_capable()`/
`require_gpu(backend=spec.gpu_backend)` check all of this, and
`featm.gpu.registration.validate_gpu_registry()` cross-checks the whole
registry against the whole kernel set in both directions). Inside
`dispatch_one_feature`, the resolved GPU device (`featm.gpu.backend.
device_context`) is one coherent execution context for VRAM planning, host<->
device transfer, kernel execution, cache allocation, and OOM cleanup; the
resolved `GPUMemoryBudget` is used to proactively size (and, when necessary,
chunk) the batch before any allocation is attempted, not merely reported. See
`SCALING.md`'s "GPU execution (Step 3, audit-fix pass v0.3.1)" section for the
full backend/fallback/memory/OOM contract, and the historical audit record
for exactly which 22 features are GPU-accelerated and why that scope was
chosen (down from v0.3.0's 24 -- two were removed for correctness reasons).
