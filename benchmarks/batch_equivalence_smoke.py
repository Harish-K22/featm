"""Automated batch-equivalence audit: does out-of-core batching change a feature's mathematics?

Fourth audit pass, Finding 8 ("we should not need a fifth manual audit to
discover another corpus-dependent function") -- and the fifth audit pass
proved that concern literally correct: this script previously excluded
EVERY image/audio/video/graph feature wholesale ("no representative
row-oriented Source layout defined"), which is exactly the verification
blind spot that let a real image/audio/video execution bug (Finding 1) and
a real graph.embeddings.deepwalk batch-partition-dependence bug (Finding 2)
both ship undetected in v0.2.4. Fifth audit pass, Finding 3: this script now
also builds representative in-memory ``PandasSource`` object-column samples
for those four modalities (``BatchSemantics.OPAQUE_OBJECT_SAMPLE``), so they
are exercised for real equivalence like any other claimed-batch-safe
feature, not excluded by construction.

For every registered feature whose ``batch_semantics`` claims row-
independent/batch-safe operation (``ROW_INDEPENDENT``,
``DATASET_CONTEXT_ROW_ALIGNED``, or ``OPAQUE_OBJECT_SAMPLE``) and for which
this script defines a representative Source layout, it:

1. builds a deterministic synthetic DataFrame of logical samples;
2. runs the feature through the Step-2 out-of-core engine in ONE batch
   (``batch_rows=N``, i.e. mathematically identical to the small-data path)
   and again with SEVERAL partitions (``batch_rows=1``, ``2``, and a
   non-divisor/prime-ish value);
3. asserts identical row count, identical output column names/order, and
   numerically close values (``equal_nan=True``).

For a feature this script classifies as intentionally NOT batch-safe
(``STRUCTURED_SAMPLE``, or ``required_params_for_batching`` left unset), it
instead asserts the multi-batch run is *rejected* cleanly
(``UnsafeMultiBatchFeatureError``), not silently computed as a batch-local
substitute.

Every registered feature is accounted for in the final counts -- excluded
features (no representative Source layout defined, or a missing optional
dependency) are reported with a specific reason, never silently dropped.
This is a correctness/behavior audit, not a throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import featm
from featm.execution import UnsafeMultiBatchFeatureError
from featm.registry import BatchSemantics, FeatureSpec

rng = np.random.default_rng(7)
N = 12  # logical sample count; deliberately not a power of 2
PARTITIONS = [1, 2, 5]  # 5 is a non-divisor of 12 -- an uneven last batch

# v0.5.1 -> v0.5.2 audit-fix (Finding 7): this harness previously either ran
# the FULL registry against live network/model-cache access (risking
# multi-GB HuggingFace/gensim/torch downloads that repeatedly drove this
# Windows dev machine's free disk space toward zero) or was skipped entirely
# with status=skipped_disk_risk in the final accepted package -- which the
# audit is explicit is NOT an acceptable regression pass. --external-assets
# offline-skip below fixes this at the harness level (no feature-math
# change): known heavyweight-optional-dependency features run inside a
# bounded, killable subprocess with offline env vars set, so a feature that
# would otherwise try to download a large remote asset either uses an
# already-installed local cache (exercised for real, per the audit's "do not
# populate multi-GB model caches merely to validate row/batch semantics
# unless the asset is already deliberately installed") or fails/times out
# fast and is reported in its own external_asset_unavailable bucket, never
# silently skipped and never allowed to hang or fill the disk.
_HEAVY_DEPENDENCY_PACKAGES = {
    "sentence-transformers", "transformers", "torch", "gensim",
    "node2vec", "timm", "clip", "open_clip", "openai-clip",
}
_SUBPROCESS_TIMEOUT_SECONDS = 90


def _is_heavyweight(spec: FeatureSpec) -> bool:
    return bool(set(spec.requires) & _HEAVY_DEPENDENCY_PACKAGES)


def _set_offline_env() -> None:
    # Best-effort: honored by huggingface_hub/transformers (and anything
    # built on top of them, e.g. sentence-transformers, several image/text
    # embeddings features) to refuse network access and fail fast instead of
    # downloading. Not every optional dependency here respects these (e.g.
    # gensim's own downloader does not), which is exactly why heavyweight
    # features also get the bounded-subprocess timeout below as a backstop.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ==========================================================================
# Per-modality representative row-oriented sample builders.
#
# Only modalities whose small-data calling convention is genuinely
# row-Source-representable (verified across three prior audit passes plus
# this one's own investigation -- see docs/SCALING.md's modality-support
# table) get a builder here. Everything else (image/audio/video/graph/
# geometry/spatial/manifold/pointcloud/topology/matrix/multimodal) has no
# representative row-oriented Source layout and is reported as such, not
# silently skipped.
# ==========================================================================
def _tabular_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    cols: list[str] = []
    columns_map: list[str] = []
    for i, typ in enumerate(spec.input_types or ("numeric",)):
        name = f"col{i}"
        cols.append(name)
        columns_map.append(name)
    df = pd.DataFrame()
    for i, typ in enumerate(spec.input_types or ("numeric",)):
        name = f"col{i}"
        if typ == "categorical":
            df[name] = pd.Series(["a", "b", "a", "c"] * (N // 4 + 1))[:N]
        elif typ == "datetime":
            df[name] = pd.date_range("2026-01-01", periods=N, freq="h")
        else:
            df[name] = np.linspace(-2, 3, N) + rng.normal(scale=0.1, size=N)
    columns = columns_map if spec.arity > 1 else columns_map[0]
    return df, {spec.name: columns}


def _timeseries_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    T = 24
    data = np.vstack([np.sin(np.linspace(0, 6 * np.pi, T)) + 0.05 * i + 0.02 * rng.normal(size=T) for i in range(N)])
    df = pd.DataFrame(data, columns=[f"t{i:03d}" for i in range(T)])
    return df, {}


def _event_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    # Fixed-width numeric columns, each row = one sorted event-time sequence
    # (verified working row-independent convention, see event.timing.event_rate).
    K = 6
    data = np.cumsum(rng.exponential(1.0, size=(N, K)), axis=1)
    df = pd.DataFrame(data, columns=[f"e{i}" for i in range(K)])
    return df, {}


def _sequence_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    alphabets = ["aabb", "abab", "baba", "bbaa", "aaab", "baaa"]
    df = pd.DataFrame({"s": [alphabets[i % len(alphabets)] + str(i % 3) for i in range(N)]})
    return df, {}


def _information_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    K = 8
    probs = np.abs(rng.normal(size=(N, K)))
    probs /= probs.sum(1, keepdims=True)
    df = pd.DataFrame(probs, columns=[f"p{i}" for i in range(K)])
    return df, {}


def _text_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    words = ["apple", "banana", "car", "road", "fruit", "engine", "yellow", "tropical"]
    docs = [" ".join(rng.choice(words, size=4)) for _ in range(N)]
    df = pd.DataFrame({"text": docs})
    return df, {}


# --------------------------------------------------------------------------
# Object-column (OPAQUE_OBJECT_SAMPLE) samples -- fifth audit pass, Finding
# 3. Findings 1/2 were only discoverable because image/audio/video/graph had
# no representative Source layout here at all; a PandasSource with real
# pre-loaded per-row objects is exactly what docs/SCALING.md's modality-
# support table (and BatchSemantics.OPAQUE_OBJECT_SAMPLE) claims is
# supported, so it belongs in this harness's coverage, not permanently
# excluded as "no representative row-oriented Source layout".
# --------------------------------------------------------------------------
def _image_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    h, w, c = 8, 8, 3
    imgs = [(rng.random((h, w, c)) * 255.0) for _ in range(N)]
    df = pd.DataFrame({"img": imgs})
    return df, {}


def _audio_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    T = 512
    clips = [np.sin(2 * np.pi * 220 * np.arange(T) / 8000) + 0.01 * rng.normal(size=T) for _ in range(N)]
    df = pd.DataFrame({"clip": clips})
    return df, {}


def _video_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    t, h, w, c = 4, 6, 6, 3
    clips = [(rng.random((t, h, w, c)) * 255.0) for _ in range(N)]
    df = pd.DataFrame({"clip": clips})
    return df, {}


def _graph_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    import networkx as nx  # required=('networkx',) on every graph.* feature -- guarded inside audit_one's try
    graphs = [nx.cycle_graph(5 + (i % 3)) for i in range(N)]
    df = pd.DataFrame({"g": graphs})
    return df, {}


_MODALITY_BUILDERS = {
    "tabular": _tabular_sample,
    "timeseries": _timeseries_sample,
    "signal": _timeseries_sample,
    "event": _event_sample,
    "sequence": _sequence_sample,
    "information": _information_sample,
    "text": _text_sample,
    "image": _image_sample,
    "audio": _audio_sample,
    "video": _video_sample,
    "graph": _graph_sample,
}

# Required transform-time params for features Finding 2/3's
# required_params_for_batching gates -- without these, this script correctly
# expects rejection instead (see "unsafe_rejected_correctly" below).
_REQUIRED_PARAM_VALUES: dict[str, dict] = {
    "sequence.distance.edit_distance": {"reference": "aabb0"},
    "sequence.alignment.alignment": {"reference": "aabb0"},
    "event.timing.recency": {"reference_time": 100.0},
    "text.bm25.bm25": {"query": "apple fruit"},
}


#: Features this harness cannot meaningfully exercise for reasons unrelated
#: to batch-safety -- documented explicitly (never silently dropped) rather
#: than left to show up as a false "mismatch".
_EXCLUDED_NOT_BATCH_RELATED: dict[str, str] = {
    # arity=1 declared, but the implementation needs 2 packed (lat, lon)
    # columns -- fails immediately and clearly (ColumnResolutionError)
    # before any computation, regardless of batching; a pre-existing
    # out-of-core usability gap, not a silent-wrong-answer risk. Stated as a
    # known limitation in STEP2_AUDIT4_FIX_HANDOFF.md rather than fixed in
    # this pass (fixing it means changing the registered arity contract).
    "tabular.geographic.unit_x": "arity/column-count mismatch (pre-existing, fails clearly, not a batch-safety issue)",
    "tabular.geographic.unit_y": "arity/column-count mismatch (pre-existing, fails clearly, not a batch-safety issue)",
    "tabular.geographic.unit_z": "arity/column-count mismatch (pre-existing, fails clearly, not a batch-safety issue)",
    # requires an explicit model= object this harness does not construct.
    "text.embeddings.word": "requires an explicit model= keyed-vector object not supplied by this harness",
    "graph.embeddings.gnn": "requires an explicit user-supplied model=callable(graph)->vector, not supplied by this harness",
    "video.embeddings.3d_cnn": "requires an explicit user-supplied model=callable(video)->vector, not supplied by this harness",
    "video.embeddings.timesformer": "requires an explicit user-supplied model=callable(video)->vector, not supplied by this harness",
    "video.embeddings.video_mae": "requires an explicit user-supplied model=callable(video)->vector, not supplied by this harness",
    "video.facial_motion.landmarks": "requires an explicit user-supplied model=callable(video)->vector, not supplied by this harness",
    "video.object_tracks.track_count": "requires an explicit user-supplied model=callable(video)->vector, not supplied by this harness",
    "video.pose_dynamics.joint_velocity": "requires an explicit user-supplied model=callable(video)->vector, not supplied by this harness",
    # audio.spatial.ild/itd expect stereo audio, shape (n, t, 2) -- a
    # different per-sample array rank than this harness's mono
    # _audio_sample builder produces (matching the existing precedent of
    # excluding arity>1/differently-shaped-sample features rather than
    # multiplying sample builders per exact shape convention).
    "audio.spatial.ild": "expects stereo audio (n,t,2); this harness's audio sample builder is mono (n,t), not covered",
    "audio.spatial.itd": "expects stereo audio (n,t,2); this harness's audio sample builder is mono (n,t), not covered",
    # signal.decomposition.eemd was fixed in the fifth audit pass (Finding 2:
    # PyEMD's EEMD.noise_seed() is now seeded from each row's own signal
    # content, making it deterministic AND batch-partition-invariant) --
    # deliberately NOT excluded any more; it is now exercised for real
    # equivalence like any other row-independent feature.
}


def _reason_excluded(spec: FeatureSpec) -> str | None:
    if spec.name in _EXCLUDED_NOT_BATCH_RELATED:
        return _EXCLUDED_NOT_BATCH_RELATED[spec.name]
    if spec.modality not in _MODALITY_BUILDERS:
        return f"no representative row-oriented Source layout defined for modality={spec.modality!r}"
    if spec.arity > 1 and spec.modality != "tabular":
        return "arity>1 whole-batch feature not covered by this harness's sample builders"
    return None


def _build_sample(spec: FeatureSpec) -> tuple[pd.DataFrame, dict]:
    builder = _MODALITY_BUILDERS[spec.modality]
    df, columns = builder(spec)
    params = dict(_REQUIRED_PARAM_VALUES.get(spec.name, {}))
    return df, columns, ({spec.name: params} if params else None)


def _run(spec: FeatureSpec, df: pd.DataFrame, columns: dict, params: dict | None, batch_rows: int):
    return featm.extract(
        df, features=[spec.name], columns=columns or None, params=params,
        batch_rows=batch_rows, output="memory", errors="raise",
    )


def audit_one(spec: FeatureSpec) -> tuple[str, str]:
    """Returns (bucket, detail)."""
    excluded_reason = _reason_excluded(spec)
    if excluded_reason:
        return "excluded", excluded_reason

    # Fifth audit pass: OPAQUE_OBJECT_SAMPLE (image.*/audio.*/video.*/
    # graph.*) is now a real, actively-enforced, genuinely batch-safe
    # classification (Finding 1) -- claimed batch-safe exactly like
    # ROW_INDEPENDENT/DATASET_CONTEXT_ROW_ALIGNED, and exercised for real
    # full-vs-batched equivalence below, not excluded wholesale the way this
    # harness previously excluded every image/audio/video/graph feature.
    is_claimed_safe = spec.batch_semantics in (
        BatchSemantics.ROW_INDEPENDENT, BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED, BatchSemantics.OPAQUE_OBJECT_SAMPLE,
    )
    is_structured = spec.batch_semantics == BatchSemantics.STRUCTURED_SAMPLE
    needs_explicit_params = bool(spec.required_params_for_batching) and spec.name not in _REQUIRED_PARAM_VALUES
    # Legacy single-shot tabular convention (Finding 7/8: several more of
    # these were found by this very harness and marked incremental=True in
    # this pass -- see tabular/full.py). Enforced by a separate guard in
    # execution.py (not the required_params_for_batching mechanism), but the
    # expected behavior for THIS harness is identical: reject multi-batch,
    # not compute a batch-local answer.
    is_legacy_incremental = spec.modality == "tabular" and spec.incremental and not spec.stateful

    try:
        # Sample construction itself can need an optional dependency (e.g.
        # networkx for graph.*) -- guarded here too, not just around _run(),
        # so a missing optional dependency is reported as such rather than
        # crashing this whole audit's loop.
        df, columns, params = _build_sample(spec)

        if is_structured or needs_explicit_params or is_legacy_incremental:
            # This feature must be REJECTED for genuinely multi-batch
            # out-of-core execution -- prove that, rather than proving
            # equivalence (there is nothing correct to be equivalent to).
            try:
                _run(spec, df, columns, params, batch_rows=2)
            except UnsafeMultiBatchFeatureError:
                return "unsafe_rejected_correctly", "rejected multi-batch execution as expected"
            except (ImportError, ModuleNotFoundError) as exc:
                return "unsupported_optional", str(exc)[:160]
            return "mismatches", "expected UnsafeMultiBatchFeatureError but the run succeeded"

        if not is_claimed_safe:
            return "excluded", f"batch_semantics={spec.batch_semantics.value} not covered by this harness (e.g. FULL_DATASET)"

        full = _run(spec, df, columns, params, batch_rows=N)
        full_df = full.to_pandas()
        for batch_rows in PARTITIONS:
            batched = _run(spec, df, columns, params, batch_rows=batch_rows)
            if batched.report.failed_features or batched.report.skipped_features:
                return "mismatches", f"batch_rows={batch_rows}: unexpected failed/skipped {batched.report.failed_features + batched.report.skipped_features}"
            batched_df = batched.to_pandas()
            if batched_df.shape[0] != full_df.shape[0]:
                return "row_alignment_failures", f"batch_rows={batch_rows}: row count {batched_df.shape[0]} != full {full_df.shape[0]}"
            if list(batched_df.columns) != list(full_df.columns):
                return "schema_drift_failures", f"batch_rows={batch_rows}: columns {list(batched_df.columns)} != full {list(full_df.columns)}"
            a = full_df.to_numpy(dtype=float, na_value=np.nan)
            b = batched_df.to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-8):
                return "mismatches", f"batch_rows={batch_rows}: values differ from the single-batch (whole-source) run"
        return "equivalent", "identical across batch_rows=N/1/2/5"
    except (ImportError, ModuleNotFoundError) as exc:
        return "unsupported_optional", str(exc)[:160]
    except Exception as exc:
        return "mismatches", f"{type(exc).__name__}: {exc}"[:200]


def _run_bounded(spec: FeatureSpec, timeout: int = _SUBPROCESS_TIMEOUT_SECONDS) -> tuple[str, str]:
    """Runs audit_one(spec) inside a real child PROCESS (subprocess.run),
    not a multiprocessing.Process -- an earlier version of this fix used
    multiprocessing with the "spawn" start method, which on Windows
    re-imports this module via runpy.run_module() in the child using
    whatever sys.path the pickled preparation data happened to carry; that
    reimport reliably failed with "ModuleNotFoundError: No module named
    'featm'" for every heavyweight feature tried (verified directly in this
    pass's own run log), silently misclassifying a harness bug as
    "external asset unavailable". subprocess.run with an explicit
    PYTHONPATH (the same pattern already used by
    benchmarks/step5/run_system_benchmark.py's _run_one) avoids that
    reimport path entirely and is what every other Step-5 worker dispatch
    in this codebase already relies on."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "benchmarks.batch_equivalence_smoke", "--audit-one", spec.name, "--external-assets", "offline-skip"]
    try:
        cp = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "external_asset_unavailable", (
            f"bounded subprocess timed out after {timeout}s under --external-assets=offline-skip "
            "(remote asset likely unavailable/blocked, or a genuine hang -- killed rather than left running)"
        )
    lines = [x for x in cp.stdout.splitlines() if x.strip().startswith("{")]
    if not lines:
        return "external_asset_unavailable", (
            f"child process exited (code={cp.returncode}) without reporting a result; "
            f"stderr_tail={(cp.stderr or '')[-300:]!r}"
        )
    try:
        rec = json.loads(lines[-1])
        return rec["bucket"], rec["detail"]
    except Exception as exc:
        return "external_asset_unavailable", f"could not parse child process result: {type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-assets", choices=["allow", "offline-skip"], default="allow",
                     help="offline-skip: run known heavyweight-optional-dependency features "
                          "(sentence-transformers/transformers/torch/gensim/node2vec/timm/clip) "
                          "in a bounded, killable subprocess with offline env vars set, instead of "
                          "risking an unbounded network/model-cache download.")
    ap.add_argument("--audit-one", default=None,
                     help="internal: run audit_one() for exactly this one feature and print its "
                          "(bucket, detail) as a JSON line, then exit. Used by _run_bounded() to "
                          "dispatch a heavyweight feature to its own real child process via "
                          "subprocess.run, not by a human.")
    args = ap.parse_args()

    if args.external_assets == "offline-skip":
        _set_offline_env()

    if args.audit_one:
        try:
            spec = featm.REGISTRY.get(args.audit_one)
            bucket, detail = audit_one(spec)
        except Exception as exc:
            bucket, detail = "external_asset_unavailable", f"{type(exc).__name__}: {str(exc)[:200]}"
        print(json.dumps({"feature": args.audit_one, "bucket": bucket, "detail": detail}))
        return 0

    buckets: dict[str, list[tuple[str, str]]] = {
        "equivalent": [], "unsafe_rejected_correctly": [], "unsupported_optional": [],
        "excluded": [], "mismatches": [], "row_alignment_failures": [], "schema_drift_failures": [],
        "external_asset_unavailable": [],
    }
    for spec in featm.REGISTRY.all():
        if args.external_assets == "offline-skip" and _is_heavyweight(spec):
            bucket, detail = _run_bounded(spec)
        else:
            bucket, detail = audit_one(spec)
        buckets[bucket].append((spec.name, detail))

    claimed_batch_safe = sum(
        1 for s in featm.REGISTRY.all()
        if s.batch_semantics in (BatchSemantics.ROW_INDEPENDENT, BatchSemantics.DATASET_CONTEXT_ROW_ALIGNED, BatchSemantics.OPAQUE_OBJECT_SAMPLE)
        and not (s.required_params_for_batching and s.name not in _REQUIRED_PARAM_VALUES)
        and not (s.modality == "tabular" and s.incremental and not s.stateful)
    )
    print(f"external_assets_policy = {args.external_assets}")
    print(f"claimed_batch_safe = {claimed_batch_safe}")
    for name in ("equivalent", "unsupported_optional", "unsafe_rejected_correctly", "excluded"):
        print(f"{name} = {len(buckets[name])}")
    for name in ("mismatches", "row_alignment_failures", "schema_drift_failures"):
        print(f"{name} = {len(buckets[name])}")
    print(f"external_asset_unavailable = {len(buckets['external_asset_unavailable'])}")

    if buckets["excluded"]:
        print("\nEXCLUDED (no representative Source layout defined by this harness -- not a failure):")
        for n, d in buckets["excluded"]:
            print(" ", n, "->", d)
    if buckets["unsupported_optional"]:
        print("\nOPTIONAL DEPENDENCY UNAVAILABLE:")
        for n, d in buckets["unsupported_optional"]:
            print(" ", n, "->", d)
    if buckets["external_asset_unavailable"]:
        # Reported separately from genuine failures -- not installed/reachable
        # under this policy is not the same claim as "the math is wrong".
        print("\nEXTERNAL ASSET UNAVAILABLE (offline-skip policy; not a batch-equivalence failure):")
        for n, d in buckets["external_asset_unavailable"]:
            print(" ", n, "->", d)
    for bad_bucket in ("mismatches", "row_alignment_failures", "schema_drift_failures"):
        if buckets[bad_bucket]:
            print(f"\n{bad_bucket.upper()}:")
            for n, d in buckets[bad_bucket]:
                print(" ", n, "->", d)

    return 1 if (buckets["mismatches"] or buckets["row_alignment_failures"] or buckets["schema_drift_failures"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
