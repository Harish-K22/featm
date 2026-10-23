# Distributed extraction with Dask (v0.5.9)

Install `"featm[distributed,io]"` for Dask with checkpoint/Parquet IO.
Ordinary imports do not eagerly require Dask. CPU/GPU is a separate choice
inside each worker; installing Dask does not install CUDA or certify a new GPU.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import featm

def main():
    name = "tabular.scaling.zscore"
    data = pd.DataFrame({"value": np.arange(23.0)})
    with TemporaryDirectory() as directory:
        options = dict(
            features=[name], columns={name: "value"},
            execution="distributed", distributed_backend="dask",
            distributed_workers=2, distributed_partition_rows=5,
            checkpoint=Path(directory) / "checkpoint", memory_budget="128MB",
            errors="raise",
        )
        first = featm.extract(data, **options)
        resumed = featm.extract(data, **options)
        pd.testing.assert_frame_equal(first.to_pandas(), resumed.to_pandas())
        print(resumed.report.to_dict())

if __name__ == "__main__":
    main()
```

Run this as a script with the main guard, especially on Windows where workers
spawn separate processes. Workers need the same FEATM version and the optional
dependencies for the selected features. The completed example was tested with
the installed v0.5.7 wheel and actual multiprocess Dask, not a threaded imitation.

Eligible row-independent features retain source order across partitions.
Stateful methods merge partition fit state before frozen transformation.
Checkpoint commits preserve output/failure history and allow compatible reuse;
invalid roots, identity mismatches or unsafe retained state fail closed.
Native Parquet sources can be read by workers without sending the complete
input from the coordinator. No scheduler choice makes a non-batchable algorithm
safe to partition.

Start with explicit methods and inspect `.report.distributed_report` or
`.report.to_dict()`. Failed features may be isolated under `warn`/`skip`, but
structurally invalid checkpoints still raise. Small jobs may be faster locally;
Dask overhead is workload-dependent. See [scaling](SCALING.md),
[architecture](ARCHITECTURE.md) and [compatibility](COMPATIBILITY.md).

## Advanced contracts

See [the execution reference](EXECUTION_REFERENCE.md) for batch safety, state
consistency, output ordering/backpressure, checkpoint identity and committed
progress, failure replay, materialization and operational limits.
