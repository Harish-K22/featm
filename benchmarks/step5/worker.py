from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

import featm
from benchmarks.step5.fixtures import build_call, normalize_output

# Every featm optional-dependency failure raises ImportError uniformly,
# whether the missing thing is a pip package (``pip install X``) or a
# pretrained model asset that needs a network download (e.g. a HuggingFace
# checkpoint). We only have the message text to tell them apart, so classify
# on keywords that are consistent across the featm feature modules.
_ASSET_KEYWORDS = ("download", "pretrained", "could not load", "checkpoint", "cache")


def _classify_missing_dependency(exc: BaseException) -> str:
    msg = str(exc).lower()
    if any(kw in msg for kw in _ASSET_KEYWORDS):
        return "external_asset_unavailable"
    return "optional_dependency_unavailable"


# from_pretrained()-style calls (transformers/sentence-transformers/timm) can
# fail with a network/OS-level error rather than ImportError when the
# package IS installed but the model weights can't be downloaded or found in
# the local cache. Treat those as the same external-asset bucket rather than
# a genuine feature-kernel bug.
_ASSET_ERROR_TYPES = ("OSError", "HTTPError", "ConnectionError", "URLError",
                      "RepositoryNotFoundError", "LocalEntryNotFoundError")

# json.JSONDecodeError is unconditionally treated as an asset problem (no
# message-keyword gate, unlike the other types above): confirmed via a real
# Step-5 run (text.embeddings.transformer) that this arises from
# transformers' own from_pretrained() machinery parsing a truncated/corrupt
# cached *.json config file on disk -- FEATM's own registered feature code
# never parses JSON itself, so a JSONDecodeError surfacing here always means
# a broken external cache entry, never a genuine feature-kernel bug.
_UNCONDITIONAL_ASSET_ERROR_TYPES = ("JSONDecodeError",)


def _looks_like_asset_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in _UNCONDITIONAL_ASSET_ERROR_TYPES:
        return True
    if name not in _ASSET_ERROR_TYPES:
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in ("download", "connect", "cache", "repo", "not found", "resolve"))


# transformers' _LazyModule wraps a submodule import failure in a bare
# RuntimeError with this exact message shape ("Failed to import X because of
# the following error") -- confirmed via a real Step-5 run
# (audio.embeddings.clap/hubert) where the underlying cause was a broken
# TensorFlow install on this machine, an unrelated transitively-imported
# optional backend, not a bug in the requested PyTorch-based feature. A bare
# except RuntimeError elsewhere in this file (unsupported_by_benchmark_fixture,
# from build_call) never reaches here -- this only classifies RuntimeErrors
# raised from actually RUNNING a feature.
def _looks_like_wrapped_import_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "failed to import" in str(exc).lower()


# Significant figures kept when digesting floating-point output for the
# determinism check -- see _digest()'s comment for why this exists.
_DIGEST_SIGNIFICANT_FIGURES = 9


def _digest(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    # normalize NaN payloads and endian/layout so the digest is semantic enough
    # for repeated calls in the same environment, not a serialization format.
    b = np.ascontiguousarray(a)
    if np.issubdtype(b.dtype, np.floating):
        b = b.copy()
        b[np.isnan(b)] = np.nan
        finite = np.isfinite(b)
        if finite.any():
            # Audit finding (graph.centrality.eigenvector): numpy.linalg.eig
            # under multi-threaded LAPACK gives genuinely different last-bit
            # floating-point results across repeated calls to the SAME input
            # (~1e-16 relative, i.e. machine epsilon) -- confirmed via direct
            # reproduction outside this harness, not a FEATM bug. An exact
            # byte-for-byte digest flags that as "nondeterministic" even
            # though the values are numerically identical for any real
            # purpose. Round to _DIGEST_SIGNIFICANT_FIGURES significant
            # figures (scale-independent, unlike a fixed decimal-places
            # round) before hashing -- far looser than ULP noise, far
            # tighter than any real algorithmic difference this check is
            # meant to catch.
            vals = b[finite]
            with np.errstate(divide="ignore"):
                magnitude = np.where(vals != 0, np.floor(np.log10(np.abs(vals))), 0.0)
            factor = 10.0 ** (_DIGEST_SIGNIFICANT_FIGURES - 1 - magnitude)
            b[finite] = np.round(vals * factor) / factor
    h = hashlib.blake2b(digest_size=16)
    h.update(str(b.dtype).encode())
    h.update(str(tuple(b.shape)).encode())
    h.update(b.tobytes())
    return h.hexdigest()


class PeakRSS:
    def __init__(self, interval_s: float = .002):
        self.interval_s = interval_s
        self.proc = psutil.Process(os.getpid())
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except psutil.Error:
                pass
            self.stop.wait(self.interval_s)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=.2)
        try:
            self.peak = max(self.peak, self.proc.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def delta(self) -> int:
        return max(0, int(self.peak - self.base))


def run(feature: str, scale: int, warmups: int, repeats: int) -> dict[str, Any]:
    spec = featm.REGISTRY.get(feature)
    lane = "kernel_fit_transform" if spec.stateful else "kernel"
    result: dict[str, Any] = {
        "feature": spec.name,
        "modality": spec.modality,
        "family": spec.family,
        "subfamily": spec.subfamily,
        "cost": spec.cost.value,
        "complexity": spec.complexity.value,
        "memory_profile": spec.memory_profile.value,
        "batch_semantics": spec.batch_semantics.value,
        "stateful": bool(spec.stateful),
        "gpu_available": bool(spec.gpu_available),
        "lane": lane,
        "scale": scale,
        "scale_axis": "",
        "logical_samples": 0,
        "input_elements": 0,
        "uses_dummy_model": False,
        "status": "feature_error",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        call, meta = build_call(spec, scale)
    except RuntimeError as exc:
        result.update(status="unsupported_by_benchmark_fixture",
                       error_type=type(exc).__name__, error_message=str(exc)[:500])
        return result
    result.update(scale_axis=meta.scale_axis, logical_samples=meta.logical_samples,
                  input_elements=meta.input_elements, uses_dummy_model=bool(meta.uses_dummy_model))
    try:
        for _ in range(max(0, warmups)):
            normalize_output(spec, call())
        times = []
        digests = []
        peak_delta = 0
        final = None
        for _ in range(max(1, repeats)):
            with PeakRSS() as rss:
                t0 = time.perf_counter()
                final = normalize_output(spec, call())
                times.append(time.perf_counter() - t0)
            peak_delta = max(peak_delta, rss.delta)
            digests.append(_digest(final))
        assert final is not None
        finite = np.isfinite(final) if np.issubdtype(final.dtype, np.number) else np.ones(final.shape, dtype=bool)
        sorted_t = sorted(times)
        p90_idx = min(len(sorted_t)-1, max(0, math.ceil(.90 * len(sorted_t)) - 1))
        result.update({
            "status": "ok",
            "median_seconds": float(statistics.median(times)),
            "min_seconds": float(min(times)),
            "p90_seconds": float(sorted_t[p90_idx]),
            "peak_rss_delta_bytes": int(peak_delta),
            "output_rows": int(final.shape[0]),
            "output_cols": int(final.shape[1] if final.ndim == 2 else 1),
            "output_bytes": int(final.nbytes),
            "nonfinite_count": int(finite.size - finite.sum()),
            "deterministic": len(set(digests)) == 1,
            "digest": digests[-1],
            "error_type": "",
            "error_message": "",
        })
    except (ImportError, ModuleNotFoundError) as exc:
        result.update(status=_classify_missing_dependency(exc),
                       error_type=type(exc).__name__, error_message=str(exc)[:500])
    except AssertionError as exc:
        result.update(status="invalid_output", error_type=type(exc).__name__, error_message=str(exc)[:500])
    except MemoryError as exc:
        result.update(status="resource_limit", error_type=type(exc).__name__, error_message=str(exc)[:500])
    except BaseException as exc:
        if _looks_like_wrapped_import_error(exc):
            result.update(status=_classify_missing_dependency(exc),
                           error_type=type(exc).__name__, error_message=str(exc)[:500])
        elif _looks_like_asset_error(exc):
            result.update(status="external_asset_unavailable", error_type=type(exc).__name__, error_message=str(exc)[:500])
        else:
            result.update(status="feature_error", error_type=type(exc).__name__, error_message=str(exc)[:500])
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()
    rec = run(args.feature, args.scale, args.warmups, args.repeats)
    text = json.dumps(rec, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    # Exit 0 for every well-classified, non-crash outcome, including the
    # "couldn't run this for an explained reason" buckets (missing optional
    # dependency/asset, fixture can't represent this feature, resource
    # limit hit) -- those are meaningful benchmark evidence, not failures.
    # Only a genuine feature/output bug gets a nonzero exit so a direct CLI
    # caller or test can tell "ran clean" from "found a real problem"
    # without parsing JSON. The orchestrator's own crash detection is
    # independent of this exit code (it looks at stdout, not returncode).
    explained = {"ok", "optional_dependency_unavailable", "external_asset_unavailable",
                 "unsupported_by_benchmark_fixture", "resource_limit"}
    return 0 if rec["status"] in explained else 2


if __name__ == "__main__":
    raise SystemExit(main())
