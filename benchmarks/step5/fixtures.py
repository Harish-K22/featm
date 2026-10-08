from __future__ import annotations

"""Deterministic synthetic fixtures for the Step-5 benchmark suite.

The calling conventions intentionally mirror ``benchmarks/smoke_universe.py``
(the already-audited full-registry smoke harness).  Step 5 adds a scale factor
but does not invent a second interpretation of FEATM modality shapes.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from featm.registry import FeatureOutput, FeatureSpec

SEED = 20260829


@dataclass(frozen=True)
class FixtureMeta:
    scale: int
    scale_axis: str
    logical_samples: int
    input_elements: int
    # Step-5 completion audit: a feature whose real signature takes a
    # caller-supplied model=callable (embedding-style features that need a
    # real trained model) is exercised here against a trivial STAND-IN
    # (``dummy_model``/``DummyVectors``/a lambda), not a real model. This is
    # necessary to exercise the wrapper/dispatch code path at all without a
    # multi-GB model download, but its timing/RSS/output numbers describe
    # the DUMMY model's cost, not a real embedding pipeline's -- callers
    # must not read these as "real embedding feature performance". True by
    # default; ``build_call`` sets it False for every case that does NOT
    # substitute a fixture model.
    uses_dummy_model: bool = False


class DummyVectors:
    vector_size = 6

    def __getitem__(self, key: str) -> np.ndarray:
        # Process-independent seed: Python's builtin hash() is randomized.
        import hashlib

        digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little") & 0xFFFFFFFF
        return np.random.default_rng(seed).normal(size=self.vector_size)


def dummy_model(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.array([a.mean(), a.std(), a.min(), a.max()], dtype=float)


def _rng(scale: int) -> np.random.Generator:
    return np.random.default_rng(SEED + int(scale) * 1009)


def _n(base: int, scale: int, *, cap: int | None = None) -> int:
    value = max(1, int(base * max(1, scale)))
    return min(value, cap) if cap else value


def _tabular_args(spec: FeatureSpec, scale: int, rng: np.random.Generator):
    n = _n(64, scale, cap=100_000)
    numeric = pd.Series(np.linspace(-2, 3, n) + rng.normal(scale=.1, size=n))
    cats = np.array(["a", "b", "a", "c", "d", "b", "e", "a"], dtype=object)
    categorical = pd.Series(np.resize(cats, n))
    dt = pd.Series(pd.date_range("2026-01-01", periods=n, freq="min"))
    points = rng.normal(size=(n, 2))
    args = []
    for typ in spec.input_types or ("numeric",):
        if typ == "numeric":
            args.append(numeric)
        elif typ == "categorical":
            args.append(categorical)
        elif typ == "datetime":
            args.append(dt)
        elif typ == "geographic":
            args.append(points)
        else:
            args.append(numeric)
    elements = sum(np.asarray(a).size for a in args)
    return args, FixtureMeta(scale, "rows", n, int(elements))


def build_call(spec: FeatureSpec, scale: int = 1):
    """Return ``(callable, FixtureMeta)`` for one deterministic benchmark case.

    The callable executes ONE feature kernel invocation.  Stateful features
    perform an in-memory fit then transform using the same adapter convention
    as ``smoke_universe.py``; the Step-5 worker reports them as lane
    ``kernel_fit_transform`` so they are never confused with stateless kernel
    latency.
    """

    rng = _rng(scale)
    fn = spec.implementation
    if fn is None:
        raise RuntimeError(f"{spec.name} has no implementation")

    if spec.modality == "tabular":
        if spec.family == "geographic":
            n = _n(64, scale, cap=100_000)
            points = rng.normal(size=(n, 2))
            return lambda: fn(points, fit=False, state=None), FixtureMeta(scale, "rows", n, int(points.size))
        args, meta = _tabular_args(spec, scale, rng)
        if spec.stateful:
            return lambda: fn(*args, fit=True, state=None), meta
        return lambda: fn(*args, fit=True, state=None), meta

    if spec.modality in {"timeseries", "signal"}:
        n = _n(4, scale, cap=4096)
        width = 128
        base = np.linspace(0, 8 * np.pi, width)
        series = np.vstack([np.sin(base) + 0.05 * rng.normal(size=width) + i * 1e-4 for i in range(n)])
        pair = np.stack([series, np.roll(series, 3, axis=1)], axis=1)
        triple = np.stack([series, np.roll(series, 2, axis=1), np.roll(series, 5, axis=1)], axis=1)
        if spec.modality == "timeseries":
            x = triple if ("conditional" in spec.name or "interaction" in spec.name) else (pair if spec.family == "cross_series" else series)
        else:
            x = pair if "phase_locking" in spec.name else series
        return lambda: fn(x), FixtureMeta(scale, "samples", n, int(np.asarray(x).size))

    if spec.modality == "audio":
        n = _n(3, scale, cap=512)
        width = 4096
        t = np.arange(width) / 22050
        audio = np.vstack([np.sin(2 * np.pi * (220 + i % 5) * t) + .01 * rng.normal(size=width) for i in range(n)])
        x = np.stack([audio, np.roll(audio, 2, axis=1)], axis=-1) if spec.family == "spatial" else audio
        return lambda: fn(x), FixtureMeta(scale, "clips", n, int(np.asarray(x).size))

    if spec.modality == "image":
        n = _n(3, scale, cap=512)
        side = 32
        images = (rng.random((n, side, side, 3)) * 255).astype(np.uint8)
        return lambda: fn(images), FixtureMeta(scale, "images", n, int(images.size))

    if spec.modality == "video":
        n = _n(2, scale, cap=128)
        video = (rng.random((n, 6, 24, 24, 3)) * 255).astype(np.uint8)
        dummy = "user-model" in spec.requires
        kw = {"model": dummy_model} if dummy else {}
        return lambda: fn(video, **kw), FixtureMeta(scale, "videos", n, int(video.size), uses_dummy_model=dummy)

    if spec.modality == "graph":
        import networkx as nx

        n_graphs = _n(3, scale, cap=128)
        nodes = min(18 + 2 * max(0, scale - 1), 80)
        graphs = []
        for i in range(n_graphs):
            g = nx.watts_strogatz_graph(nodes, min(4, nodes - 1), .2, seed=SEED + i)
            nx.set_node_attributes(g, {n: str(n % 3) for n in g}, "label")
            graphs.append(g)
        kw = {}
        dummy = spec.name == "graph.embeddings.gnn"
        if dummy:
            kw["model"] = lambda g: np.array([len(g), g.number_of_edges(), nx.density(g)])
        elements = sum(g.number_of_nodes() + g.number_of_edges() for g in graphs)
        return lambda: fn(graphs, **kw), FixtureMeta(scale, "graphs", n_graphs, int(elements), uses_dummy_model=dummy)

    if spec.modality in {"geometry", "manifold"}:
        n = _n(32, scale, cap=4096)
        points = rng.normal(size=(n, 2))
        return lambda: fn(points), FixtureMeta(scale, "points", n, int(points.size))

    if spec.modality == "pointcloud":
        n_clouds = _n(2, scale, cap=64)
        points_per = min(_n(40, scale), 2048)
        clouds = [rng.normal(size=(points_per + (i % 3), 3)) + i * .1 for i in range(n_clouds)]
        dummy = "user-model" in spec.requires
        kw = {"model": dummy_model} if dummy else {}
        return lambda: fn(clouds, **kw), FixtureMeta(scale, "clouds", n_clouds, int(sum(c.size for c in clouds)), uses_dummy_model=dummy)

    if spec.modality == "matrix":
        n = _n(3, scale, cap=512)
        matrices = rng.normal(size=(n, 5, 5))
        return lambda: fn(matrices), FixtureMeta(scale, "matrices", n, int(matrices.size))

    if spec.modality == "sequence":
        n = _n(3, scale, cap=4096)
        seeds = ["ababcaaab", "xyzxyzxy", "mississippi", "abababab", "aabbccdd"]
        seqs = [seeds[i % len(seeds)] + str(i % 7) for i in range(n)]
        return lambda: fn(seqs), FixtureMeta(scale, "sequences", n, int(sum(len(s) for s in seqs)))

    if spec.modality == "text":
        n = _n(3, scale, cap=4096)
        base = [
            "Feature engineering discovers useful mathematical structure.",
            "Graphs topology and signals can expose different patterns.",
            "A small text document for testing feature extraction.",
            "Spectral statistical lexical geometric robust feature benchmark.",
        ]
        docs = [base[i % len(base)] + f" token{i % 13}" for i in range(n)]
        kw = {}
        dummy = spec.name == "text.embeddings.word"
        if spec.name == "text.bm25.bm25":
            kw["query"] = "feature graphs"
        if dummy:
            kw["model"] = DummyVectors()
        if spec.stateful:
            return lambda: fn(docs, fit=True, state=None, **kw), FixtureMeta(scale, "documents", n, int(sum(len(d) for d in docs)), uses_dummy_model=dummy)
        return lambda: fn(docs, **kw), FixtureMeta(scale, "documents", n, int(sum(len(d) for d in docs)), uses_dummy_model=dummy)

    if spec.modality == "event":
        n = _n(2, scale, cap=512)
        if spec.family == "survival":
            k = 8
            durations = np.sort(rng.uniform(0.1, 10, size=(n, k)), axis=1)
            observed = rng.random((n, k)) > .35
            x = {"durations": durations, "observed": observed}
            elements = durations.size + observed.size
        else:
            k = 30
            x = [np.cumsum(rng.exponential(1.0 + (i % 3) * .1, size=k)) for i in range(n)]
            elements = sum(a.size for a in x)
        return lambda: fn(x), FixtureMeta(scale, "event_series", n, int(elements))

    if spec.modality == "multimodal":
        n = _n(4, scale, cap=1024)
        emb_a = rng.normal(size=(n, 16))
        emb_b = rng.normal(size=(n, 16))
        x = {
            "image": emb_a,
            "text": emb_b,
            "audio": emb_a * .8,
            "video": emb_a * 1.1,
            "tabular": rng.normal(size=(n, 5)),
            "temporal": rng.normal(size=(n, 7)),
        }
        return lambda: fn(x), FixtureMeta(scale, "samples", n, int(sum(np.asarray(v).size for v in x.values())))

    if spec.modality == "information":
        n = _n(3, scale, cap=2048)
        probs = np.abs(rng.normal(size=(n, 12)))
        probs /= probs.sum(1, keepdims=True)
        paired_probs = np.stack([probs, np.roll(probs, 1, axis=1)], axis=1)
        base_ts = rng.normal(size=(n, 80))
        pair_ts = np.stack([base_ts, np.roll(base_ts, 3, axis=1)], axis=1)
        triple_ts = np.stack([base_ts, np.roll(base_ts, 2, axis=1), np.roll(base_ts, 5, axis=1)], axis=1)
        info_multi = rng.normal(size=(max(2, n), 80, 4))
        if spec.family == "entropy":
            x = pair_ts if (spec.name.endswith(".joint") or spec.name.endswith(".conditional")) else probs
        elif spec.family == "dependence":
            x = triple_ts if ("conditional" in spec.name or "interaction" in spec.name) else pair_ts
        elif spec.family == "divergence":
            x = pair_ts if spec.name.endswith("variation_of_information") else paired_probs
        elif spec.family == "multivariate":
            x = info_multi
        elif spec.family == "pid":
            x = triple_ts
        else:
            x = probs
        return lambda: fn(x), FixtureMeta(scale, "samples", n, int(np.asarray(x).size))

    if spec.modality == "spatial":
        n = _n(32, scale, cap=4096)
        coords = rng.normal(size=(n, 2))
        x = {"coordinates": coords, "values": np.sin(coords[:, 0]) + rng.normal(scale=.1, size=n)}
        return lambda: fn(x), FixtureMeta(scale, "points", n, int(coords.size + n))

    if spec.modality == "topology":
        n_clouds = _n(2, scale, cap=32)
        points_per = min(_n(40, scale), 1024)
        clouds = [rng.normal(size=(points_per, 3)) + i * .2 for i in range(n_clouds)]
        return lambda: fn(clouds), FixtureMeta(scale, "clouds", n_clouds, int(sum(c.size for c in clouds)))

    raise RuntimeError(f"No Step-5 fixture for modality={spec.modality!r}")


def normalize_output(spec: FeatureSpec, result: Any) -> np.ndarray:
    if spec.modality == "tabular" and isinstance(result, tuple):
        result = result[0]
    elif spec.stateful and isinstance(result, tuple):
        result = result[0]
    if isinstance(result, FeatureOutput):
        arr = result.as_array()
    else:
        arr = np.asarray(result)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim not in (1, 2):
        raise AssertionError(f"bad output ndim={arr.ndim}")
    return arr
