from __future__ import annotations

"""Public-engine benchmark worker for Source-safe features.

This lane deliberately uses ``featm.extract`` rather than calling the kernel
implementation directly. It measures FEATM's local execution overhead and
batching machinery separately from raw kernel latency.
"""

import argparse
import json
import math
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd

import featm
from featm.execution import UnsafeMultiBatchFeatureError
from benchmarks.batch_equivalence_smoke import _build_sample, _reason_excluded
from benchmarks.step5.worker import PeakRSS, _classify_missing_dependency, _looks_like_asset_error, _looks_like_wrapped_import_error


def _repeat_rows(df: pd.DataFrame, scale: int) -> pd.DataFrame:
    if scale <= 1:
        return df.reset_index(drop=True)
    return pd.concat([df] * scale, ignore_index=True)


def run(feature: str, scale: int, partitions: int, warmups: int, repeats: int) -> dict[str, Any]:
    spec=featm.REGISTRY.get(feature)
    reason=_reason_excluded(spec)
    base={"feature":feature,"modality":spec.modality,"family":spec.family,"subfamily":spec.subfamily,
          "lane":"engine_extract_one_batch" if partitions<=1 else f"engine_extract_{partitions}way",
          "scale":scale,"stateful":bool(spec.stateful),"batch_semantics":spec.batch_semantics.value,"status":"error"}
    if reason:
        return {**base,"status":"not_engine_representable","error_type":"ExcludedByHarness","error_message":reason}
    try:
        df, columns, params=_build_sample(spec)
        df=_repeat_rows(df,scale)
        n=len(df)
        batch_rows=n if partitions<=1 else max(1,math.ceil(n/partitions))
        def call():
            return featm.extract(df,features=[feature],columns=columns or None,params=params,
                                 batch_rows=batch_rows,output="memory",errors="raise")
        for _ in range(max(0,warmups)): call()
        times=[]; peak=0; final=None
        for _ in range(max(1,repeats)):
            with PeakRSS() as rss:
                t0=time.perf_counter(); final=call(); times.append(time.perf_counter()-t0)
            peak=max(peak,rss.delta)
        frame=final.to_pandas()
        ts=sorted(times); p90=ts[min(len(ts)-1,max(0,math.ceil(.9*len(ts))-1))]
        return {**base,"status":"ok","logical_rows":n,"batch_rows":batch_rows,"partitions_requested":partitions,
                "median_seconds":float(statistics.median(times)),"min_seconds":float(min(times)),"p90_seconds":float(p90),
                "peak_rss_delta_bytes":int(peak),"output_rows":int(frame.shape[0]),"output_cols":int(frame.shape[1]),
                "failed_features":";".join(final.report.failed_features),"skipped_features":";".join(final.report.skipped_features),
                "error_type":"","error_message":""}
    except (ImportError,ModuleNotFoundError) as exc:
        return {**base,"status":_classify_missing_dependency(exc),"error_type":type(exc).__name__,"error_message":str(exc)[:500]}
    except UnsafeMultiBatchFeatureError as exc:
        # A CORRECT Step-2 safety rejection (the public engine refusing to run
        # this feature through batched/out-of-core execution because doing so
        # would silently produce a batch-local, not dataset-correct, answer)
        # -- not a benchmark-harness or feature bug. _reason_excluded() above
        # only checks static spec properties and does not catch every legacy
        # single-shot/structured-input convention that execution.py itself
        # enforces at call time, so this is caught here instead of
        # mislabeling a working safety guard as feature_error.
        return {**base,"status":"not_engine_representable","error_type":type(exc).__name__,"error_message":str(exc)[:500]}
    except MemoryError as exc:
        return {**base,"status":"resource_limit","error_type":type(exc).__name__,"error_message":str(exc)[:500]}
    except BaseException as exc:
        if _looks_like_wrapped_import_error(exc):
            status=_classify_missing_dependency(exc)
        elif _looks_like_asset_error(exc):
            status="external_asset_unavailable"
        else:
            status="feature_error"
        return {**base,"status":status,"error_type":type(exc).__name__,"error_message":str(exc)[:500]}


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--feature',required=True);ap.add_argument('--scale',type=int,default=1)
    ap.add_argument('--partitions',type=int,default=1);ap.add_argument('--warmups',type=int,default=1);ap.add_argument('--repeats',type=int,default=3)
    a=ap.parse_args(); rec=run(a.feature,a.scale,a.partitions,a.warmups,a.repeats);print(json.dumps(rec,sort_keys=True))
    explained={'ok','optional_dependency_unavailable','external_asset_unavailable','not_engine_representable','resource_limit'}
    return 0 if rec['status'] in explained else 2
if __name__=='__main__': raise SystemExit(main())
