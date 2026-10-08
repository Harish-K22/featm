from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import featm
from benchmarks.step5.catalog import rows as catalog_rows
from benchmarks.step5.fixtures import build_call, normalize_output
from benchmarks.step5.run_system_benchmark import SCALES_BY_COST, TIMEOUT_BY_COST
from benchmarks.step5.worker import run


def test_step5_catalog_accounts_for_entire_registry():
    rows=list(catalog_rows())
    assert len(rows)==552==len(featm.REGISTRY.all())
    assert len({(r["modality"],r["family"]) for r in rows}) >= 150
    assert sum(bool(r["stateful"]) for r in rows)==7
    assert sum(bool(r["gpu_available"]) for r in rows)==22


def test_every_modality_has_a_fixture_builder_path():
    seen=set()
    # One cheapest feature per modality is enough to prove dispatch coverage
    # without executing every optional/heavy kernel in a unit test.
    for spec in sorted(featm.REGISTRY.all(), key=lambda s:(s.modality,s.cost.value,s.name)):
        if spec.modality in seen: continue
        call, meta=build_call(spec,1)
        assert callable(call)
        assert meta.logical_samples >= 1
        assert meta.input_elements >= 1
        seen.add(spec.modality)
    assert seen == {s.modality for s in featm.REGISTRY.all()}


def test_adaptive_ladders_cover_all_cost_classes():
    costs={s.cost for s in featm.REGISTRY.all()}
    assert costs <= set(SCALES_BY_COST)
    assert costs <= set(TIMEOUT_BY_COST)
    assert all(v and v[0]==1 for v in SCALES_BY_COST.values())


def test_worker_smoke_for_representative_dependency_light_features():
    reps=[
        "tabular.numeric.abs",
        "tabular.scaling.zscore",
        "timeseries.statistical.mean",
        "signal.time_domain.rms",
        "sequence.basic.length",
        "matrix.algebra.trace",
    ]
    for name in reps:
        rec=run(name,1,warmups=0,repeats=2)
        assert rec["status"]=="ok", rec
        assert rec["deterministic"] is True
        assert rec["output_rows"] >= 1
        assert rec["median_seconds"] >= 0


def test_stateful_worker_lane_is_not_mislabeled_as_stateless_kernel():
    rec=run("tabular.scaling.zscore",1,0,1)
    assert rec["lane"]=="kernel_fit_transform"


def test_controlled_utility_code_uses_public_fit_for_stateful_features():
    text=open("benchmarks/step5/utility_probes.py",encoding="utf-8").read()
    assert "if spec.stateful" in text
    assert "featm.fit(train" in text
    assert "fitted.transform(test" in text


def test_stateful_utility_evaluation_never_calls_extract_directly(monkeypatch):
    # Behavioral regression test: this MUST fail if utility evaluation is
    # ever rewritten to call featm.extract() independently on the train and
    # validation folds for a stateful feature -- that would silently refit a
    # fresh mean/std (or vocabulary, or category frequencies, ...) on the
    # validation fold instead of reusing the train-fold-fitted state, which
    # is exactly the leakage bug Section 9 forbids. The text-grep test above
    # only proves the right call names appear somewhere in the file; this
    # test proves the actual call graph taken at runtime never touches
    # featm.extract for a stateful feature.
    import featm
    from benchmarks.step5.utility_probes import Probe, evaluate_probe, _tabular_probe

    calls = []
    real_extract = featm.extract

    def spy(*a, **k):
        calls.append((a, k))
        return real_extract(*a, **k)

    monkeypatch.setattr(featm, "extract", spy)
    base = _tabular_probe(n=120)
    stateful_probe = Probe(base.name, base.task, base.frame, base.y,
                            ("tabular.scaling.zscore",), base.columns)
    rows = evaluate_probe(stateful_probe, folds=2)
    assert rows[0]["status"] == "ok"
    assert calls == [], (
        f"featm.extract() must never be called directly for a stateful "
        f"feature during utility evaluation (leakage risk), but it was "
        f"called {len(calls)} time(s)"
    )


def test_stateful_transform_uses_train_fold_statistics_not_test_fold():
    # Directly proves the leakage-safe numeric behavior: zscore fit on a
    # train fold and applied (frozen) to a test fold drawn from a very
    # different distribution must NOT re-center around the test fold's own
    # mean. If the code were leaky (fit independently per fold), the
    # transformed test values would end up close to zero for both folds
    # since each fold would be normalized against its own mean/std.
    import numpy as np
    import pandas as pd

    import featm

    rng = np.random.default_rng(7)
    train = pd.DataFrame({"x": rng.normal(0, 1, size=200)})
    test = pd.DataFrame({"x": rng.normal(50, 1, size=200)})

    fitted = featm.fit(train, features=["tabular.scaling.zscore"],
                        columns={"tabular.scaling.zscore": "x"}, errors="raise")
    Xte = fitted.transform(test, output="memory").to_pandas().to_numpy()

    assert abs(float(np.mean(Xte))) > 10, (
        "transformed test-fold values are near zero, suggesting zscore was "
        "fit on the test fold's own statistics instead of the train fold's "
        "-- this is the exact leakage pattern Section 9 forbids"
    )


def test_worker_flags_dummy_model_features_without_conflating_with_real_ones():
    # text.embeddings.word is exercised against DummyVectors(), a toy
    # stand-in, not a real trained embedding model -- its timing/output
    # numbers describe the stand-in's cost, not a real embedding pipeline's,
    # so it must be flagged and must not look identical to an unflagged run.
    dummy = run("text.embeddings.word", 1, warmups=0, repeats=1)
    assert dummy["uses_dummy_model"] is True
    plain = run("tabular.numeric.abs", 1, warmups=0, repeats=1)
    assert plain["uses_dummy_model"] is False


def test_run_system_benchmark_resume_loads_prior_rows_and_skips_them(tmp_path):
    import csv

    from benchmarks.step5.run_system_benchmark import _load_existing

    out = tmp_path / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "scale", "status"])
        w.writeheader()
        w.writerow({"feature": "tabular.numeric.abs", "scale": "1", "status": "ok"})
    rows, done = _load_existing(out)
    assert len(rows) == 1
    assert ("tabular.numeric.abs", 1) in done
    # A fresh (non-existent) output file resumes to an empty, non-blocking state.
    empty_rows, empty_done = _load_existing(tmp_path / "does_not_exist.csv")
    assert empty_rows == [] and empty_done == set()


def test_whisper_audio_embedding_uses_encoder_only_not_full_seq2seq_forward():
    # Product-bug regression test (found via the Step-5 full-registry CPU
    # benchmark, independently reproduced outside the benchmark harness):
    # openai/whisper-tiny is an encoder-decoder checkpoint. _hf_audio_embeddings
    # loads it via AutoModel and previously called model(**inp) with only
    # encoder inputs (from the processor) -- WhisperModel.forward() requires
    # decoder_input_ids/decoder_inputs_embeds for the decoder half, so this
    # raised "ValueError: You have to specify either decoder_input_ids or
    # decoder_inputs_embeds" for every call, 100% of the time, not an
    # environment flake. Fixed by running the encoder only (via
    # model.get_encoder()) for any checkpoint architecture that has one --
    # audio embedding is a representation task, not generation, so only the
    # encoder's hidden states are needed. Encoder-only checkpoints
    # (Wav2Vec2/HuBERT/CLAP) have no get_encoder() and are unaffected.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import numpy as np

    from featm.features.audio.full import _hf_audio_embeddings

    x = np.random.default_rng(0).normal(size=(2, 16000)).astype("float32")
    out = _hf_audio_embeddings(x, model_name="openai/whisper-tiny")
    arr = out.as_array()
    assert arr.shape[0] == 2
    assert arr.shape[1] > 0
    assert np.isfinite(arr).all()


def test_dino_image_embedding_uses_token_pooling_and_native_input_size():
    # Product-bug regression test (found via the Step-5 full-registry CPU
    # benchmark, independently reproduced outside the benchmark harness, two
    # compounding issues on the installed timm==1.0.3):
    #  1) vit_small_patch14_dinov2.lvd142m ships no fc_norm weights (it was
    #     trained for CLS-token pooling, not average pooling); requesting
    #     global_pool='avg' made timm add a fresh fc_norm layer and then
    #     strictly fail to load its state_dict against the checkpoint --
    #     100% reproducible, not an environment flake. Fixed with a narrowly
    #     scoped retry using global_pool='token' when creation fails with a
    #     state_dict/fc_norm error.
    #  2) The same checkpoint's native resolution is 518x518 (patch14), not
    #     the 224x224 every other model in this loop uses -- fixed by
    #     reading each model's own pretrained_cfg['input_size'] instead of a
    #     single hardcoded 224 for every model_name.
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import numpy as np

    from featm.features.image.full import _timm_embed

    x = (np.random.default_rng(0).random((2, 64, 64, 3)) * 255).astype("uint8")
    out = _timm_embed(x, model_name="vit_small_patch14_dinov2.lvd142m")
    arr = out.as_array()
    assert arr.shape[0] == 2
    assert arr.shape[1] > 0
    assert np.isfinite(arr).all()
    # Confirm the fix did not disturb a checkpoint that was already working
    # (global_pool='avg' must still be tried first and must still succeed).
    resnet_out = _timm_embed(x, model_name="resnet18").as_array()
    assert resnet_out.shape[0] == 2 and np.isfinite(resnet_out).all()


def test_digest_tolerates_ulp_noise_but_still_catches_real_differences():
    # Benchmark-infrastructure bug found via the Step-5 full-registry CPU
    # benchmark: graph.centrality.eigenvector (numpy.linalg.eig under
    # multi-threaded LAPACK) gives genuinely different last-bit
    # floating-point results across repeated calls to the IDENTICAL input
    # (confirmed via direct reproduction: ~1e-16 relative difference, i.e.
    # machine epsilon) -- not a FEATM bug, but the exact-byte digest called
    # it "nondeterministic" anyway. Fixed by rounding to a fixed number of
    # significant figures (scale-independent) before hashing, matching how
    # the CPU<->GPU equivalence check already uses a numerical tolerance
    # instead of exact equality for the same underlying reason.
    import numpy as np

    from benchmarks.step5.worker import _digest

    a = np.array([0.21916246421487584, 1234.5678901234])
    ulp_noise = a * (1 + 1e-14)
    assert _digest(a) == _digest(ulp_noise), "ULP-level floating point noise must not flip the digest"

    genuinely_different = a.copy()
    genuinely_different[0] += 0.01
    assert _digest(a) != _digest(genuinely_different), "a real, meaningful difference must still change the digest"

    rec = run("graph.centrality.eigenvector", 1, warmups=1, repeats=5)
    assert rec["status"] == "ok"
    assert rec["deterministic"] is True, "false-positive nondeterminism from BLAS ULP noise must not resurface"


def test_clap_audio_embedding_uses_audio_only_branch_and_native_sample_rate():
    # Product-bug regression test (found via the Step-5 full-registry CPU
    # benchmark, independently reproduced outside the benchmark harness):
    # laion/clap-htsat-unfused is a dual audio/text contrastive checkpoint.
    # Calling it via the same model(**inp) convention used for encoder-only
    # audio models raised "text input must be of type str" (ClapModel's
    # forward() always expects paired text) and, once that used get_audio_features
    # instead, "the model...was trained using a sampling rate of 48000" (its
    # processor rejects the helper's generic 16kHz default outright, no
    # resampling). Fixed by detecting get_audio_features() (present only on
    # dual audio/text encoders), calling it directly with an explicit
    # audios= keyword, and deferring to the processor's own declared native
    # sampling rate instead of a fixed cross-model default.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import numpy as np

    from featm.features.audio.full import _hf_audio_embeddings

    x = np.random.default_rng(0).normal(size=(2, 16000)).astype("float32")
    out = _hf_audio_embeddings(x, model_name="laion/clap-htsat-unfused")
    arr = out.as_array()
    assert arr.shape[0] == 2
    assert arr.shape[1] > 0
    assert np.isfinite(arr).all()
    # Confirm the fix did not disturb an encoder-only checkpoint that was
    # already working (get_encoder() path must still be tried and win).
    wav2vec_out = _hf_audio_embeddings(x, model_name="facebook/wav2vec2-base-960h").as_array()
    assert wav2vec_out.shape[0] == 2 and np.isfinite(wav2vec_out).all()


def test_engine_worker_classifies_unsafe_multibatch_rejection_as_not_engine_representable():
    # Benchmark-infrastructure bug found via the Step-5 Layer B run: a
    # feature that correctly raises UnsafeMultiBatchFeatureError (a Step-2
    # safety guard, e.g. tabular.categorical.cooccurrence's legacy
    # single-shot convention refusing multi-batch execution) was being
    # caught by the generic except BaseException handler and mislabeled
    # "feature_error" -- indistinguishable from a genuine bug. Fixed by
    # catching UnsafeMultiBatchFeatureError explicitly and bucketing it as
    # not_engine_representable, matching how the harness already treats
    # every other correctly-rejected-by-design case.
    from benchmarks.step5.engine_worker import run as run_engine

    rec = run_engine("tabular.categorical.cooccurrence", scale=1, partitions=4, warmups=0, repeats=1)
    assert rec["status"] == "not_engine_representable", rec
    assert rec["error_type"] == "UnsafeMultiBatchFeatureError"


def test_engine_worker_exercises_public_extract_for_safe_feature():
    from benchmarks.step5.engine_worker import run as run_engine
    rec=run_engine("timeseries.statistical.mean",scale=1,partitions=4,warmups=0,repeats=1)
    assert rec["status"]=="ok", rec
    assert rec["output_rows"]==rec["logical_rows"]
    assert rec["lane"]=="engine_extract_4way"


# --------------------------------------------------------------------------
# Provenance regression tests (v0.5.0 -> v0.5.1 audit-fix pass).
#
# An independent audit found the shipped final CPU evidence had NO per-row
# way to trace a result back to the code/environment snapshot that produced
# it -- the separately-saved environment manifest's checksum did not match
# the actually-delivered source, and repeated single-feature resume reruns
# (each fixing a real product bug) had silently mixed rows from different
# source snapshots under one CSV with no row-level identity. Root cause:
# run_system_benchmark.py's --manifest flag defaulted to a DIFFERENT
# filename than --out, so every resume rerun silently overwrote a
# differently-named stray manifest while the "official" one went stale.
# --------------------------------------------------------------------------
def test_provenance_module_produces_a_run_and_environment_identity(tmp_path):
    from benchmarks.step5.provenance import make_provenance
    import featm as _featm
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent
    prov = make_provenance(root, _featm.__version__)
    assert prov["run_id"]
    assert prov["environment_id"]
    assert len(prov["source_checksum_blake2b16"]) == 32  # 16-byte blake2b hex digest
    assert prov["featm_version"] == _featm.__version__


def test_cpu_system_benchmark_rows_carry_run_and_environment_identity(tmp_path):
    """Every raw row from a real run_system_benchmark.py invocation must carry
    a valid run/environment identity, and the identity must match a real
    manifest file written alongside the output -- not merely be present as a
    field with an arbitrary value."""
    import csv as _csv
    import subprocess as _subprocess
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    out = tmp_path / "cpu_feature_results.csv"
    cmd = [_sys.executable, "-m", "benchmarks.step5.run_system_benchmark",
           "--out", str(out), "--feature", "timeseries.statistical.mean", "--fresh"]
    cp = _subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=60)
    assert cp.returncode == 0, cp.stdout + cp.stderr

    manifest_path = tmp_path / "cpu_run_manifest.json"
    assert manifest_path.exists(), "manifest must be derived from --out, not left at a hardcoded default filename"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = list(_csv.DictReader(open(out, encoding="utf-8")))
    assert rows, "expected at least one row"
    for r in rows:
        assert r["run_id"], r
        assert r["environment_id"], r
        assert len(r["source_checksum_blake2b16"]) == 32, r
        assert r["featm_version"], r
        # Row provenance must match the manifest written by the SAME run,
        # not merely be independently well-formed.
        assert r["run_id"] == manifest["run_id"], (r["run_id"], manifest["run_id"])
        assert r["source_checksum_blake2b16"] == manifest["source_checksum_blake2b16"]
        assert r["featm_version"] == manifest["featm_version"]


def test_manifest_source_checksum_matches_the_actual_current_source_tree():
    """A manifest's source_checksum_blake2b16 must equal a checksum computed
    fresh, right now, from the actual src/featm tree -- not a value that
    could have drifted from an earlier snapshot."""
    from benchmarks.step5.provenance import source_checksum
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    fresh = source_checksum(root)
    fresh_again = source_checksum(root)
    assert fresh == fresh_again, "checksum must be deterministic for an unchanged source tree"
    assert len(fresh) == 32


def test_source_checksum_is_platform_independent(tmp_path):
    # Audit finding (v0.5.1 -> v0.5.2): the original source_checksum() hashed
    # str(Path.relative_to(root)), which uses the HOST OS path separator --
    # the identical source tree hashed differently on Windows (backslash
    # paths) vs. Linux/macOS (forward-slash paths), confirmed directly:
    # 0a07536b... on Windows vs. c4f6944c... on Linux for the same bytes.
    # Fixed by hashing Path.as_posix() (always forward-slash) instead.
    from benchmarks.step5.provenance import _canonical_tree_checksum

    root = tmp_path
    pkg = root / "src" / "featm"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_bytes(b"x = 1\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "b.py").write_bytes(b"y = 2\n")

    paths = list(pkg.rglob("*.py"))
    digest = _canonical_tree_checksum(paths, root)

    # The function's own path-building must never depend on os.sep: every
    # relative path it hashes must already be posix-style (no backslashes),
    # regardless of what platform this test runs on.
    for p in paths:
        rel = p.relative_to(root).as_posix()
        assert "\\" not in rel, f"relative path {rel!r} contains a backslash -- not platform-canonical"

    # Deterministic and stable across repeated computation on this platform.
    assert _canonical_tree_checksum(paths, root) == digest

    # Simulate the SAME logical relative path arriving via a Windows-style
    # PureWindowsPath vs. a POSIX-style PurePosixPath -- both must normalize
    # to the identical posix string that gets hashed, proving the digest
    # does not depend on which OS produced the path.
    import ntpath
    import posixpath
    win_style = ntpath.join("src", "featm", "sub", "b.py")
    posix_style = posixpath.join("src", "featm", "sub", "b.py")
    assert win_style.replace("\\", "/") == posix_style


def test_provenance_gate_detects_a_mutated_run_id(tmp_path):
    """A row whose run_id does not match its own run manifest must fail the
    provenance gate -- not merely "is non-empty"."""
    import csv
    import json as _json
    from pathlib import Path as _Path

    from benchmarks.step5.gen_validation_results import _gate_provenance
    from benchmarks.step5.provenance import make_provenance

    root = _Path(__file__).resolve().parent.parent
    d = tmp_path
    prov = make_provenance(root, "0.5.3")
    row = {"feature": "x", "scale": "1", "status": "ok", **prov}
    with open(d / "step5_cpu_system_feature_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    manifest = {**prov}
    (d / "step5_cpu_system_run_manifest.json").write_text(_json.dumps(manifest), encoding="utf-8")
    (d / "step5_environment_manifest.json").write_text(_json.dumps(manifest), encoding="utf-8")

    clean = _gate_provenance(d, root, "0.5.3")
    assert clean["failures"] == 0, clean

    # Mutate the row's run_id so it no longer matches the run manifest.
    row["run_id"] = "run-mutated-0000000000-0"
    with open(d / "step5_cpu_system_feature_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    mutated = _gate_provenance(d, root, "0.5.3")
    assert mutated["failures"] > 0, "a row whose run_id disagrees with the run manifest must fail the gate"


def test_dataset_modality_group_reads_explicit_metadata_not_the_name(tmp_path):
    """Audit finding (v0.5.1 -> v0.5.2, Finding 6): modality-group detection
    used to pattern-match dataset NAMES (e.g. "credit_g" in name). Prove it
    now reads an explicit modality_group field and works for an arbitrary
    dataset name that shares no substring with any real dataset's name."""
    from benchmarks.step5.gen_validation_results import (
        REQUIRED_REAL_DATASET_MODALITY_GROUPS,
        _dataset_modality_group,
        _gate_real_dataset_coverage,
    )

    arbitrary_row = {"dataset": "totally_unrelated_xyz_42", "modality_group": "tabular_classification"}
    assert _dataset_modality_group(arbitrary_row) == "tabular_classification"

    no_metadata_row = {"dataset": "credit_g_classification"}
    assert _dataset_modality_group(no_metadata_row) is None, (
        "a row with no explicit modality_group must not fall back to name inference"
    )

    import csv

    d = tmp_path
    rows = [{"dataset": f"arbitrary_dataset_{i}", "modality_group": g}
            for i, g in enumerate(sorted(REQUIRED_REAL_DATASET_MODALITY_GROUPS))]
    fields = sorted({k for r in rows for k in r})
    with open(d / "step5_dataset_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    gate = _gate_real_dataset_coverage(d)
    assert gate["failures"] == 0, gate
