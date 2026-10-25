# Advanced execution reference

This complements [scaling](SCALING.md), [Dask](DISTRIBUTED.md) and
[public API guidance](API.md). It describes the audited execution model and
the v0.5.11 missingness-entropy correction without expanding platform certification.

## Batch safety and input contracts

The unit of a sample belongs to the selected modality. Ordinary time-series
rows are samples, not a request to create sliding windows. Sources include
in-memory pandas data, CSV/TSV/JSONL, supported Parquet/Arrow data and iterator
sources. File-backed scalar columns do not deserialize image/audio/graph
objects. Preloaded objects need an appropriate in-memory object column.
Structured pairs/dictionaries cannot be represented by an arbitrary scalar row.

| Declared semantics | Local source execution | Distributed partitions |
| --- | --- | --- |
| Row independent | Split-safe within the method's input contract | Map independently |
| Dataset-context, stateful | Fit source statistics before frozen transform | Merge raw partition fit states before transform |
| Context with required parameters | Required reference/configuration must be pinned for batching | Same requirement; distribution is inherently multi-call |
| Opaque object sample | Appropriate preloaded pandas object columns | Eligible objects can be mapped; serialization cost applies |
| Full dataset | Whole-source materialization if affordable; otherwise reject | Reject naive partitioning |
| Structured sample | Not representable by the row-oriented source adapter | Reject |

Legacy non-stateful incremental methods are not distributed using worker-local
fit statistics. Distributed stateful approximate/reservoir mode is rejected:
independent worker reservoirs do not reproduce a uniform global sample. Local
approximate stateful execution remains explicitly reported as approximate and
does not avoid scanning every source row. Non-batchable whole-source algorithms
are not silently split when allocation fails; a different per-chunk answer is
not a valid recovery. Minimum input sizes and fixed output schema still apply.

Stateful fitting requires a replayable source for a later transform pass.
An iterator factory and a one-shot consumed generator have different replay
properties. Do not assume any iterator can support a multi-pass workflow.

## State consistency and leakage

Use `fit(train, ...)` then `.transform(test)` to separate training statistics
from test samples. `extract()` fits and transforms its own source. The eight
stateful methods include missingness entropy, scaling, category frequencies
and corpus statistics.
Distributed fitting merges states in logical source order, freezes one global
state and applies that same state to each transform partition. Test samples do
not enter a saved training state. Persistence must be loaded only from trusted
sources; Python-serialized state is not an untrusted-input sandbox.

## Memory, output and materialization

Working-memory budgets guide estimated batches, bounded fitted state, cache
and output choices. They are not a hard total-process RSS limit and do not
erase the cost of already resident inputs or third-party libraries. Explicit
`batch_rows` chooses that size rather than invoking automatic planning.
High-cardinality state may exceed its state budget even if batches are small.

Memory and disk-backed `FeatureMatrix` output are supported. Automatic output
may spill from retained chunks to disk. Checkpoints persist canonical output
to disk regardless of the caller's preferred output mode; IO must be installed
before a checkpoint run begins. Disk-backed conversion to one pandas/NumPy
object can be rejected by a materialization guard. Prefer `iter_batches()` for
large results. Disk budget and checkpoint/output storage are distinct from
Python process memory; make sufficient filesystem space available.

## Local checkpoint identity and resume

Identity binds product version, source/schema fingerprints, ordered methods,
parameters and normalized execution settings, including typed column mappings,
batch boundaries, mode/error policy, approximate sample size and fitted-state
budget policy/value where applicable. Integer and string labels must not alias.
Stateless results do not depend on an irrelevant fitted-state budget.

Automatic budgets retain the original effective state limit across changing
available RAM. Automatically selected local stateful batch boundaries are also
retained. Explicit budgets remain identity-sensitive. Partial execution checks
current capacity rather than shrinking previously accepted semantics. A fully
completed checkpoint can reopen committed output without reading input batches.

Fit progress uses manifest-committed accumulator snapshots; transform progress
uses committed shards. Orphan snapshots from a crash are not authoritative.
State/manifest writes use atomic replacement, but the manifest commit decides
which progress may be reused. Corrupt/incomplete committed fit state cannot be
silently treated as a clean fit. Existing JSON roots must be objects; null,
arrays/scalars, malformed JSON, old schema or product/identity mismatches fail
closed. Invalid manifests are preserved for review; choose a fresh directory
instead of editing them to force compatibility.

## Dask partition ordering and backpressure

Workers may finish in any order. Results are reassembled and committed in
logical partition order, preserving row order and source-sensitive state.
The coordinator gates replacement submissions on the undelivered-result
window as well as limiting in-flight tasks. This bounds reassembly growth by
the configured window rather than the total partition count, even if an early
partition is slow. Reports expose the observed maximum reorder buffer.
For a known row count and at least one surviving output column, the assembled
output must have the source row count; silent dropped/duplicated rows are errors.

Dask task transport retries are distinct from feature error policy. A permanent
infrastructure failure is not converted into a successful arbitrary result.
Retry/commit handling avoids merging a partition twice. Worker-native Parquet
plans can retain source projection and read workers' partitions rather than
materializing the entire source at the coordinator. Keep FEATM versions,
optional dependencies and source access consistent across workers.

## Distributed checkpoint and failure history

Distributed identity also binds partition size and canonical column mappings,
mode/error policy and applicable state budget. Fit and transform commits use
logical partition IDs with checksums. Resume trusts committed fit states and
output shards, not merely whatever files appear latest on disk. Uncommitted
files can be recomputed; corrupt committed artifacts cannot be counted as good
progress. Completed and partial resumes restore recorded failure/degradation
history instead of reporting a falsely clean run.

Under `raise`, a feature error propagates. Under `warn`/`skip`, rejected or
fit/priming-failed methods may be absent; later batch/partition failures can be
represented by NaNs under an established schema and recorded in the report.
When no method survives fitting/priming, an intentional `(0, 0)` result is
possible rather than a row-loss success. A mixed selection still keeps all
rows for surviving methods. Structural checkpoint corruption raises under
every ordinary feature error policy; it is not an opportunity to emit NaNs.

## Operational limits

Dask serialization/scheduling can dominate small jobs. Distribution does not
make a whole-dataset algorithm partition-safe or install missing optional
libraries. GPU choice is independent of Dask: only registered CuPy kernels
have acceleration, fallback is reported, and declared CPU/GPU tolerances are
not bitwise equality. Strict GPU mode rejects an unavailable implementation.
No new GPU hardware/platform certification is inferred by these contracts.
See [compatibility](COMPATIBILITY.md) for the retained support boundaries.
