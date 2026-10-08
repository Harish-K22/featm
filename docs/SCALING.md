# Larger data and checkpointing

Install `"featm[io]==0.5.7"` for Parquet/Arrow sources, checkpoints and
disk-backed output. Checkpointing needs IO even with an in-memory DataFrame.
The base package rejects that operation with an installation hint before
scanning; it remains usable for unrelated base features.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import featm

name = "tabular.scaling.zscore"
with TemporaryDirectory() as directory:
    directory = Path(directory)
    source = directory / "values.csv"
    pd.DataFrame({"amount": np.arange(20.0)}).to_csv(source, index=False)
    options = dict(
        features=[name], columns={name: "amount"}, batch_rows=5,
        checkpoint=directory / "checkpoint", memory_budget="128MB",
        errors="raise",
    )
    first = featm.extract(source, **options)
    resumed = featm.extract(source, **options)
    pd.testing.assert_frame_equal(first.to_pandas(), resumed.to_pandas())
    for batch in resumed.iter_batches(batch_rows=7):
        print(batch.shape)
```

Only compatible source, schema, selected methods/parameters and execution
settings can reuse a checkpoint. v0.5.7 rejects existing non-object JSON roots
without overwriting the manifest. Malformed JSON, older schemas and product
version mismatches also fail closed. A missing manifest permits a fresh run.
Never edit a checkpoint to force compatibility; use a new directory.

Automatic state budgets retain their original effective limit on resume.
Partial resumption still checks current capacity; an insufficient host must
free memory or use a larger machine. Explicit budgets remain part of identity.
State snapshots and failure/degradation history are persisted transactionally.
Only load artifacts from trusted sources.

## Memory and modality limits

Budgets guide batch planning, fitted state, caches and guarded materialization;
they are not a total-process RSS ceiling. Automatically planned batches obey
the budget estimate; supplying `batch_rows` explicitly chooses that batch size.
Python, optional libraries and an already resident input use memory too.

Batching is allowed only when the feature's declared semantics preserve its
meaning. Corpus-wide stateful text methods fit across the source before
transforming. Whole-point-cloud or whole-matrix algorithms may require full
materialization. Structured pairs and multimodal dictionaries cannot be
represented by arbitrary CSV cells. Preloaded graph/image/audio objects can
be used in appropriate in-memory object columns; file paths are not decoded
automatically as arbitrary objects.

There is no new windowing infrastructure or relational DFS in v0.5.7.
Use explicit, correctly constructed samples. Advanced registration semantics
are described in [feature authoring](FEATURE_AUTHORING.md).

## GPU execution

Only 22 registered methods have CuPy kernels. Strict GPU mode rejects an
unsupported method; fallback mode runs it on CPU and records the reason.
Automatic backend selection may choose CPU for small inputs. VRAM budgeting
can chunk split-safe features, and CPU/GPU comparisons use declared numerical
tolerances rather than promising bitwise equality. See
[GPU installation](INSTALLATION.md#gpu) and [tested scope](COMPATIBILITY.md).

## Advanced contracts

See [the execution reference](EXECUTION_REFERENCE.md) for batch safety, state
consistency, output ordering/backpressure, checkpoint identity and committed
progress, failure replay, materialization and operational limits.
