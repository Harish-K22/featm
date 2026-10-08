from __future__ import annotations

"""Step-5 real public-dataset utility matrix (Section 10).

A small, reproducible curated matrix -- NOT dozens of datasets. Reuses the
exact same leakage-safe fit-on-train-fold-only pattern as
utility_probes.py's ``_feature_matrix`` (train-fold-only fit, frozen
transform onto both folds; never independently fit on validation data).

Covered (all real, real provenance recorded to step5_dataset_manifest.csv) --
all six modalities named by the Step-5 task (v0.5.1 audit-fix pass added the
last three; a v0.5.0 disclosed gap, not silently left unfixed):
  - Tabular classification: OpenML "credit-g" (German Credit, numeric+categorical)
  - Tabular regression:     sklearn's California Housing
  - Text classification:    20 Newsgroups (2-category subset)
  - Image classification:   sklearn's digits (2-digit subset, 8x8 grayscale)
  - Time series/signal:     UCR GunPoint (bundled with sktime, no download)
  - Graph classification:   MUTAG (TUDataset raw files, parsed directly into
                             NetworkX -- no torch_geometric/DGL dependency)
  - Audio classification:   Free Spoken Digit Dataset (FSDD), a small subset
                             (digits 0 vs 1, ~40 clips/class) fetched as
                             individual small .wav files from its public
                             GitHub repo -- deliberately avoids any large
                             archive/model download

FSDD clips have genuinely variable length (different word durations); passed
with batch_rows=1 (RealDataset.batch_rows) since featm.extract's batching
machinery correctly refuses to silently stack ragged per-row arrays into one
dense batch tensor -- this is not a workaround for a bug, it is honoring an
existing safety guard by giving it a batch size that matches the data.
"""

import argparse
import csv
import hashlib
import io
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import featm


@dataclass(frozen=True)
class RealDataset:
    name: str
    task: str  # "classification" | "regression"
    frame: pd.DataFrame
    y: np.ndarray
    features: tuple[str, ...]
    columns: dict
    provenance: dict
    batch_rows: "int | None" = None  # force e.g. 1 for ragged per-row arrays (variable-length audio clips)


def _checksum(values: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(values).tobytes(), digest_size=8).hexdigest()


def _credit_g() -> RealDataset:
    from sklearn.datasets import fetch_openml
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    y = (df["class"] == "bad").astype(int).to_numpy()  # "bad credit risk" = positive class
    frame = df[["duration", "credit_amount", "age", "checking_status"]].copy()
    frame["checking_status"] = frame["checking_status"].astype(str)
    features = ("tabular.scaling.zscore", "tabular.encoding.category_frequency", "tabular.numeric.abs")
    columns = {"tabular.scaling.zscore": "duration", "tabular.encoding.category_frequency": "checking_status",
               "tabular.numeric.abs": "credit_amount"}
    provenance = {"name": "credit-g", "source": "OpenML (https://www.openml.org/d/31)", "version": "1",
                  "license": "Public Domain (CC0, per OpenML dataset page)", "retrieval_date": time.strftime("%Y-%m-%d"),
                  "raw_rows": len(df), "split_policy": "5-fold StratifiedKFold, shuffle, seed=123",
                  "checksum_blake2b16": _checksum(y), "preprocessing": "class->{bad:1,good:0}; 3 numeric + 1 categorical column subset",
                  "modality": "tabular", "modality_group": "tabular_classification", "task": "classification"}
    return RealDataset("credit_g_classification", "classification", frame, y, features, columns, provenance)


def _california_housing() -> RealDataset:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    y = df["MedHouseVal"].to_numpy()
    frame = df[["MedInc", "HouseAge", "AveRooms"]].copy()
    features = ("tabular.scaling.zscore", "tabular.numeric.abs", "tabular.numeric.logarithmic")
    columns = {"tabular.scaling.zscore": "MedInc", "tabular.numeric.abs": "HouseAge", "tabular.numeric.logarithmic": "AveRooms"}
    provenance = {"name": "california_housing", "source": "sklearn.datasets.fetch_california_housing (StatLib/Pace&Barry 1997)",
                  "version": "sklearn-bundled", "license": "Public domain (US Census derived)", "retrieval_date": time.strftime("%Y-%m-%d"),
                  "raw_rows": len(df), "split_policy": "5-fold KFold, shuffle, seed=123",
                  "checksum_blake2b16": _checksum(y), "preprocessing": "3-column numeric subset, target=MedHouseVal",
                  "modality": "tabular", "modality_group": "tabular_regression", "task": "regression"}
    return RealDataset("california_housing_regression", "regression", frame, y, features, columns, provenance)


def _newsgroups_binary() -> RealDataset:
    from sklearn.datasets import fetch_20newsgroups
    cats = ["sci.space", "rec.sport.hockey"]
    data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
    docs = [d for d in data.data if d.strip()]
    y = np.array([data.target[i] for i, d in enumerate(data.data) if d.strip()])
    frame = pd.DataFrame({"text": docs})
    features = ("text.tfidf.tfidf", "text.lexical.word_count")
    provenance = {"name": "20newsgroups_sci_space_vs_hockey", "source": "sklearn.datasets.fetch_20newsgroups (UCI/20 Newsgroups)",
                  "version": "sklearn-bundled", "license": "Public domain / widely redistributed research corpus",
                  "retrieval_date": time.strftime("%Y-%m-%d"), "raw_rows": len(docs),
                  "split_policy": "5-fold StratifiedKFold, shuffle, seed=123", "checksum_blake2b16": _checksum(y),
                  "preprocessing": "2-category subset (sci.space vs rec.sport.hockey), headers/footers/quotes removed, empty docs dropped",
                  "modality": "text", "modality_group": "text", "task": "classification"}
    return RealDataset("newsgroups_binary_classification", "classification", frame, y, features, {}, provenance)


def _digits_binary() -> RealDataset:
    from sklearn.datasets import load_digits
    data = load_digits()
    mask = np.isin(data.target, [3, 8])
    X = data.images[mask]
    y = (data.target[mask] == 8).astype(int)
    imgs = [np.repeat((row * 16.0)[:, :, None], 3, axis=2) for row in X]  # 0..16 -> pseudo-RGB, matches the image Source convention
    frame = pd.DataFrame({"img": imgs})
    features = ("image.color.red_mean", "image.shape.hu_moments", "image.texture.lbp")
    provenance = {"name": "digits_3_vs_8", "source": "sklearn.datasets.load_digits (UCI ML hand-written digits, bundled)",
                  "version": "sklearn-bundled", "license": "Public domain (UCI)", "retrieval_date": time.strftime("%Y-%m-%d"),
                  "raw_rows": int(mask.sum()), "split_policy": "5-fold StratifiedKFold, shuffle, seed=123",
                  "checksum_blake2b16": _checksum(y), "preprocessing": "digits 3 vs 8 subset, 8x8 grayscale replicated to pseudo-RGB",
                  "modality": "image", "modality_group": "image", "task": "classification"}
    return RealDataset("digits_binary_classification", "classification", frame, y, features, {}, provenance)


def _gunpoint_timeseries() -> RealDataset:
    from sktime.datasets import load_gunpoint
    X, y_raw = load_gunpoint(return_type="numpy2d")  # bundled with sktime -- no network download
    y = (y_raw.astype(str) == "2").astype(int)
    frame = pd.DataFrame(X, columns=[f"t{i}" for i in range(X.shape[1])])
    features = ("timeseries.statistical.mean", "timeseries.statistical.std",
                "timeseries.spectral.centroid", "timeseries.spectral.spectral_entropy")
    provenance = {"name": "gunpoint_timeseries", "source": "UCR Time Series Archive 'GunPoint', bundled via sktime.datasets.load_gunpoint",
                  "version": "sktime-bundled", "license": "Public research archive (UCR/UEA Time Series Classification Repository)",
                  "retrieval_date": time.strftime("%Y-%m-%d"), "raw_rows": len(frame),
                  "split_policy": "5-fold StratifiedKFold, shuffle, seed=123", "checksum_blake2b16": _checksum(y),
                  "preprocessing": "class '2' vs '1' (gun-draw vs point gesture), no resampling",
                  "modality": "timeseries", "modality_group": "timeseries_or_signal", "task": "classification"}
    return RealDataset("gunpoint_timeseries_classification", "classification", frame, y, features, {}, provenance)


def _mutag_graph() -> RealDataset:
    import networkx as nx
    url = "https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))

    def _lines(name):
        return zf.read(f"MUTAG/{name}").decode("utf-8").strip().splitlines()

    graph_indicator = [int(x) for x in _lines("MUTAG_graph_indicator.txt")]
    graph_labels = [int(x) for x in _lines("MUTAG_graph_labels.txt")]
    node_labels = [int(x) for x in _lines("MUTAG_node_labels.txt")]
    edges = [tuple(int(v) for v in line.split(",")) for line in _lines("MUTAG_A.txt")]

    n_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(n_graphs)]
    node_to_graph = {}
    for global_node_idx, g_idx in enumerate(graph_indicator, start=1):
        graphs[g_idx - 1].add_node(global_node_idx, label=node_labels[global_node_idx - 1])
        node_to_graph[global_node_idx] = g_idx - 1
    for u, v in edges:
        gi = node_to_graph[u]
        graphs[gi].add_edge(u, v)

    y = (np.array(graph_labels) == 1).astype(int)  # MUTAG label convention: 1 = mutagenic, -1 = not
    frame = pd.DataFrame({"g": graphs})
    features = ("graph.connectivity.density", "graph.degree.average_degree",
                "graph.clustering.average_clustering", "graph.clustering.transitivity")
    provenance = {"name": "mutag_graph_classification", "source": "TUDataset MUTAG (https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip)",
                  "version": "TUDataset (Kersting et al.)", "license": "Public research benchmark (TU Dortmund graph kernel datasets)",
                  "retrieval_date": time.strftime("%Y-%m-%d"), "raw_rows": n_graphs,
                  "split_policy": "5-fold StratifiedKFold, shuffle, seed=123", "checksum_blake2b16": _checksum(y),
                  "preprocessing": "parsed directly from TUDataset raw format (*_A.txt/_graph_indicator.txt/_graph_labels.txt/_node_labels.txt) into NetworkX graphs, no torch_geometric/DGL dependency",
                  "modality": "graph", "modality_group": "graph", "task": "classification"}
    return RealDataset("mutag_graph_classification", "classification", frame, y, features, {}, provenance)


def _fsdd_audio(n_per_class: int = 40) -> RealDataset:
    import json as _json

    from scipy.io import wavfile
    listing_url = "https://api.github.com/repos/Jakobovski/free-spoken-digit-dataset/contents/recordings"
    req = urllib.request.Request(listing_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        entries = _json.loads(r.read())
    names_by_digit = {"0": [], "1": []}
    for e in entries:
        for d in names_by_digit:
            if e["name"].startswith(f"{d}_"):
                names_by_digit[d].append(e)
    clips, labels = [], []
    for digit, label in (("0", 0), ("1", 1)):
        for e in sorted(names_by_digit[digit], key=lambda x: x["name"])[:n_per_class]:
            wav_req = urllib.request.Request(e["download_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(wav_req, timeout=30) as wr:
                wav_bytes = wr.read()
            sr, samples = wavfile.read(io.BytesIO(wav_bytes))
            clips.append(samples.astype(np.float64) / (np.iinfo(samples.dtype).max if np.issubdtype(samples.dtype, np.integer) else 1.0))
            labels.append(label)
    y = np.array(labels)
    frame = pd.DataFrame({"clip": clips})
    features = ("audio.waveform.rms", "audio.waveform.zero_crossing_rate",
                "audio.waveform.crest_factor", "audio.waveform.dynamic_range")
    provenance = {"name": "fsdd_audio_digits_0_vs_1", "source": "Free Spoken Digit Dataset (https://github.com/Jakobovski/free-spoken-digit-dataset)",
                  "version": "master (fetched per-file, no bulk archive)", "license": "Creative Commons Attribution-ShareAlike 4.0 (per FSDD repo)",
                  "retrieval_date": time.strftime("%Y-%m-%d"), "raw_rows": len(y),
                  "split_policy": "5-fold StratifiedKFold, shuffle, seed=123", "checksum_blake2b16": _checksum(y),
                  "preprocessing": f"digit '0' vs '1', {n_per_class} clips/class, individual .wav files, int-PCM normalized to [-1,1], no librosa/model dependency (audio.waveform family only)",
                  "modality": "audio", "modality_group": "audio", "task": "classification"}
    return RealDataset("fsdd_audio_classification", "classification", frame, y, features, {}, provenance, batch_rows=1)


def _feature_matrix(feature: str, ds: RealDataset, train: pd.DataFrame, test: pd.DataFrame):
    spec = featm.REGISTRY.get(feature)
    br = ds.batch_rows
    if spec.stateful:
        fitted = featm.fit(train, features=[feature], columns=ds.columns or None, errors="raise", batch_rows=br)
        Xtr = fitted.transform(train, output="memory", batch_rows=br).to_pandas().to_numpy()
        Xte = fitted.transform(test, output="memory", batch_rows=br).to_pandas().to_numpy()
    else:
        Xtr = featm.extract(train, features=[feature], columns=ds.columns or None, errors="raise", output="memory", batch_rows=br).to_pandas().to_numpy()
        Xte = featm.extract(test, features=[feature], columns=ds.columns or None, errors="raise", output="memory", batch_rows=br).to_pandas().to_numpy()
    return Xtr, Xte, spec


def _score_classification(Xtr, Xte, ytr, yte) -> float:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))


def _score_regression(Xtr, Xte, ytr, yte) -> tuple[float, float]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return float(mean_squared_error(yte, p) ** 0.5), float(r2_score(yte, p))


def _folds(ds: RealDataset, k=5):
    if ds.task == "classification":
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(k, shuffle=True, random_state=123).split(ds.frame, ds.y)
    from sklearn.model_selection import KFold
    return KFold(k, shuffle=True, random_state=123).split(ds.frame)


def evaluate_individual(ds: RealDataset, folds=5) -> list[dict]:
    rows = []
    for feature in ds.features:
        spec = featm.REGISTRY.get(feature)
        try:
            aucs, rmses, r2s = [], [], []
            for tr, te in _folds(ds, folds):
                train = ds.frame.iloc[tr].reset_index(drop=True)
                test = ds.frame.iloc[te].reset_index(drop=True)
                Xtr, Xte, _ = _feature_matrix(feature, ds, train, test)
                if ds.task == "classification":
                    aucs.append(_score_classification(Xtr, Xte, ds.y[tr], ds.y[te]))
                else:
                    rmse, r2 = _score_regression(Xtr, Xte, ds.y[tr], ds.y[te])
                    rmses.append(rmse); r2s.append(r2)
            if ds.task == "classification":
                rows.append({"dataset": ds.name, "task": ds.task, "feature": feature, "modality": spec.modality, "family": spec.family,
                             "folds": folds, "metric": "roc_auc", "mean_score": float(np.mean(aucs)), "std_score": float(np.std(aucs)), "status": "ok"})
            else:
                rows.append({"dataset": ds.name, "task": ds.task, "feature": feature, "modality": spec.modality, "family": spec.family,
                             "folds": folds, "metric": "rmse+r2", "mean_score": float(np.mean(rmses)), "std_score": float(np.std(rmses)),
                             "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)), "status": "ok"})
        except Exception as exc:
            rows.append({"dataset": ds.name, "task": ds.task, "feature": feature, "modality": spec.modality, "family": spec.family,
                         "folds": folds, "status": "not_utility_eligible", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return rows


def evaluate_bundle(ds: RealDataset, folds=5) -> dict:
    modalities = {featm.REGISTRY.get(f).modality for f in ds.features}
    families = {featm.REGISTRY.get(f).family for f in ds.features}
    base = {"dataset": ds.name, "task": ds.task, "bundle": "+".join(ds.features),
            "modality": next(iter(modalities)) if len(modalities) == 1 else ";".join(sorted(modalities)),
            "family": next(iter(families)) if len(families) == 1 else ";".join(sorted(families)),
            "n_features": len(ds.features), "folds": folds}
    try:
        aucs, rmses, r2s = [], [], []
        for tr, te in _folds(ds, folds):
            train = ds.frame.iloc[tr].reset_index(drop=True)
            test = ds.frame.iloc[te].reset_index(drop=True)
            Xtr_parts, Xte_parts = [], []
            for feature in ds.features:
                Xtr, Xte, _ = _feature_matrix(feature, ds, train, test)
                Xtr_parts.append(Xtr); Xte_parts.append(Xte)
            Xtr_all = np.concatenate(Xtr_parts, axis=1); Xte_all = np.concatenate(Xte_parts, axis=1)
            if ds.task == "classification":
                aucs.append(_score_classification(Xtr_all, Xte_all, ds.y[tr], ds.y[te]))
            else:
                rmse, r2 = _score_regression(Xtr_all, Xte_all, ds.y[tr], ds.y[te])
                rmses.append(rmse); r2s.append(r2)
        if ds.task == "classification":
            return {**base, "metric": "roc_auc", "mean_score": float(np.mean(aucs)), "std_score": float(np.std(aucs)), "status": "ok"}
        return {**base, "metric": "rmse+r2", "mean_score": float(np.mean(rmses)), "std_score": float(np.std(rmses)),
                "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)), "status": "ok"}
    except Exception as exc:
        return {**base, "status": "not_utility_eligible", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility-out", default="step5_results/step5_real_dataset_utility_results.csv")
    ap.add_argument("--bundle-out", default="step5_results/step5_real_dataset_bundle_utility_results.csv")
    ap.add_argument("--manifest-out", default="step5_results/step5_dataset_manifest.csv")
    args = ap.parse_args()

    builders = (_credit_g, _california_housing, _newsgroups_binary, _digits_binary,
                _gunpoint_timeseries, _mutag_graph, _fsdd_audio)
    datasets = []
    for b in builders:
        print(f"loading {b.__name__} ...", flush=True)
        datasets.append(b())

    rows, bundle_rows, manifest_rows = [], [], []
    for ds in datasets:
        print(f"evaluating {ds.name} ({len(ds.frame)} rows) ...", flush=True)
        rows.extend(evaluate_individual(ds))
        bundle_rows.append(evaluate_bundle(ds))
        manifest_rows.append({"dataset": ds.name, **ds.provenance})

    out = Path(args.utility_out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    bout = Path(args.bundle_out)
    bfields = sorted({k for r in bundle_rows for k in r})
    with bout.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bfields, extrasaction="ignore"); w.writeheader(); w.writerows(bundle_rows)

    mout = Path(args.manifest_out)
    mfields = sorted({k for r in manifest_rows for k in r})
    with mout.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mfields, extrasaction="ignore"); w.writeheader(); w.writerows(manifest_rows)

    print(f"Wrote {out} ({len(rows)} rows), {bout} ({len(bundle_rows)} rows), {mout} ({len(manifest_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
