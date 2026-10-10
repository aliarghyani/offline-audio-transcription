from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_alignment import fake_engine, fake_extract
from test_alignment import reviewed as reviewed_fixture  # noqa: F401

from session_pipeline import alignment as a
from session_pipeline import diarization as d
from session_pipeline import model_store as store
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.cli import run
from session_pipeline.models import sha256_file


@pytest.fixture
def aligned(request, project_config):
    sid, path, source = request.getfixturevalue("reviewed_fixture")

    def partial(originals, unit, clip):
        returned = fake_engine(originals, unit, clip)
        if unit["index"] == 0:
            returned[0] = {"words": []}
            returned[1] = {"words": []}
        return returned

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: partial, extractor=fake_extract
    )
    aligned_path = Path(report["aligned_json"])
    return sid, path, source, aligned_path


@pytest.fixture
def local_model(aligned, project_config):
    root = project_config.application.root
    lock = store.load_model_lock(root)
    spec = next(s for s in lock["models"] if s["role"] == "diarization")
    for file in spec["files"]:
        data = b"synthetic model asset: never load real weights"
        if file["path"] == "config.yaml":
            data = (
                b"pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n"
                b"  params:\n    segmentation: $model/segmentation\n"
                b"    embedding: $model/embedding\n    plda: $model/plda\n"
            )
        file.update(
            size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=None
        )
        path = store.model_path(root, spec) / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    store.write_json(root / "config/models.lock.json", lock)
    return store.model_path(root, spec)


def turns():
    return [
        {"start": 0, "end": 120, "speaker": "SPEAKER_00"},
        {"start": 100, "end": 170, "speaker": "SPEAKER_01"},
    ]


def execute(config, sid, **kwargs):
    return d.run_diarization(
        config, sid, engine_factory=lambda _: lambda clip: turns(), extractor=fake_extract, **kwargs
    )


def test_plan_binds_full_aligned_parent_and_preserves_both_failed_segments(aligned, project_config):
    sid, _, _, parent = aligned
    p, plan = d.make_plan(project_config, sid)
    assert d.make_plan(project_config, sid) == (p, plan)
    assert plan["processed_interval"] == [0, 620]
    assert plan["configuration"]["min_speakers"] == plan["configuration"]["max_speakers"] == 2
    assert plan["model"] == d.MODEL and plan["configuration"]["device"] == "cpu"
    assert plan["parent_checksums"]["aligned_asr"] == read_artifact(parent)["artifact_checksum"]
    refs = plan["aligned_segment_references"]
    assert len(refs) == 6
    assert [r["reviewed_segment_id"] for r in refs] == [
        s["reviewed_segment_id"] for s in read_artifact(parent)["segments"]
    ]
    failed = [r for r in refs if r["alignment_status"] == "ALIGNMENT_FAILED"]
    assert len(failed) == 2 and all(
        r["timing_policy"] == "REVIEWED_SEGMENT_FALLBACK" for r in failed
    )
    assert [r["reviewed_interval"] for r in failed] == [[0, 2], [100, 102]]
    assert not plan["speaker_identity_assignment"]
    assert d.output_path(project_config, sid, plan).is_relative_to(parent.parents[2] / "diarized")


def test_missing_model_fails_before_factory_and_audio_decode(aligned, project_config):
    sid, _, _, _ = aligned
    factory, decode = Mock(side_effect=AssertionError("model load")), Mock()
    report = d.run_diarization(project_config, sid, engine_factory=factory, extractor=decode)
    assert report["state"] == "FAILED" and report["output_path"] is None
    assert report["failure_code"] == "ModelLockError" and not report["model_provisioned"]
    factory.assert_not_called()
    decode.assert_not_called()


@pytest.mark.parametrize("mutation", ["wrong_revision", "corrupt", "nonlocal_config"])
def test_pinned_local_model_contract_enforced_before_inference(
    aligned, local_model, project_config, mutation
):
    sid, _, _, _ = aligned
    root = project_config.application.root
    if mutation == "wrong_revision":
        lock = store.load_model_lock(root)
        spec = next(s for s in lock["models"] if s["role"] == "diarization")
        spec.update(revision="a" * 40, destination="models/diarization/" + "a" * 40)
        store.write_json(root / "config/models.lock.json", lock)
        with pytest.raises(ValueError, match="approved exact revision"):
            d.make_plan(project_config, sid)
        return
    if mutation == "corrupt":
        path = local_model / "embedding/pytorch_model.bin"
        path.write_bytes(b"x" * path.stat().st_size)
    else:
        lock = store.load_model_lock(root)
        spec = next(s for s in lock["models"] if s["role"] == "diarization")
        path = local_model / "config.yaml"
        data = path.read_bytes().replace(b"$model/embedding", b"remote/embedding")
        path.write_bytes(data)
        file = next(f for f in spec["files"] if f["path"] == "config.yaml")
        file.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        store.write_json(root / "config/models.lock.json", lock)
    factory = Mock(side_effect=AssertionError("no inference"))
    report = d.run_diarization(project_config, sid, engine_factory=factory, extractor=Mock())
    assert report["state"] == "FAILED" and report["model_status"] == "UNKNOWN"
    factory.assert_not_called()


def test_overlap_preserved_complete_result_reused_and_inputs_immutable(
    aligned, local_model, project_config
):
    sid, _, _, parent = aligned
    *_, files = d.load_input(project_config, sid)
    before = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in files}
    report = execute(project_config, sid)
    assert report["state"] == "SUCCEEDED_WITH_WARNINGS" and report["full_diarization_complete"]
    assert (
        report["model_provisioned"]
        and report["verified_revision"] == d.MODEL["diarization"]["revision"]
    )
    output = Path(report["output_path"])
    document = read_artifact(output)
    assert document["turns"][0]["end"] > document["turns"][1]["start"]
    assert document["qa"]["overlap_seconds"] == 20
    assert document["qa"]["speech_union_seconds"] == 170
    assert document["qa"]["summed_turn_seconds"] == 190
    assert document["qa"]["speaker_confidence"] is None
    assert document["execution"]["realtime_factor"] == document["execution"]["wall_seconds"] / 620
    assert len(document["execution"]["model_inventory"]) > 0
    assert (
        document["execution"]["immutable_input_checksums"][
            parent.relative_to(project_config.application.root).as_posix()
        ]
        == before[parent][0]
    )
    assert len(document["aligned_segment_references"]) == 6
    assert not document["transcript_attribution_performed"]
    original = output.read_bytes()
    # A successful result can be reused even if the model is later unavailable.
    (local_model / "embedding/pytorch_model.bin").unlink()
    report = d.run_diarization(
        project_config,
        sid,
        engine_factory=Mock(side_effect=AssertionError("load")),
        extractor=Mock(side_effect=AssertionError("decode")),
    )
    assert report["full_diarization_complete"] and output.read_bytes() == original
    assert before == {p: (sha256_file(p), p.stat().st_mtime_ns) for p in files}
    output.unlink()
    report = d.run_diarization(
        project_config, sid, engine_factory=Mock(side_effect=AssertionError("load"))
    )
    assert report["full_diarization_complete"] and output.read_bytes() == original


@pytest.mark.parametrize(
    "items,error",
    [
        ([], "EMPTY_DIARIZATION"),
        ([{"start": 0, "end": 1, "speaker": "SPEAKER_00"}], "MISSING_SPEAKER"),
        ([{"start": 0, "end": 1, "speaker": "SPEAKER_02"}], "INVALID_SPEAKER_LABEL"),
        ([{"start": 0, "end": 1, "speaker": "named-person"}], "INVALID_SPEAKER_LABEL"),
        ([{"start": -1, "end": 1, "speaker": "SPEAKER_00"}], "OUT_OF_BOUNDS_TURN"),
        ([{"start": 0, "end": 621, "speaker": "SPEAKER_00"}], "OUT_OF_BOUNDS_TURN"),
        ([{"start": 2, "end": 2, "speaker": "SPEAKER_00"}], "OUT_OF_BOUNDS_TURN"),
        ([{"start": float("nan"), "end": 2, "speaker": "SPEAKER_00"}], "OUT_OF_BOUNDS_TURN"),
        (list(reversed(turns())), "INVALID_ORDERING"),
        (
            [{"start": 0, "end": 1, "speaker": "SPEAKER_00", "duration": 10}],
            "INVALID_TURN_DURATION",
        ),
    ],
)
def test_turn_validation_distinguishes_contract_errors(aligned, project_config, items, error):
    sid, _, _, _ = aligned
    _, plan = d.make_plan(project_config, sid)
    qa = d.evaluate_turns(items, plan)
    assert not qa["validation_pass"] and qa["error_counts"][error] > 0


def test_true_third_speaker_blocks_canonical_publication(aligned, local_model, project_config):
    sid, _, _, _ = aligned
    result = turns() + [{"start": 171, "end": 172, "speaker": "SPEAKER_02"}]
    report = d.run_diarization(
        project_config, sid, engine_factory=lambda _: lambda clip: result, extractor=fake_extract
    )
    assert report["state"] == "FAILED" and report["output_path"] is None
    assert report["qa"]["error_counts"]["INVALID_SPEAKER_LABEL"] == 1


def test_short_turn_and_uncovered_intervals_are_warnings_not_confidence(aligned, project_config):
    sid, _, _, _ = aligned
    _, plan = d.make_plan(project_config, sid)
    qa = d.evaluate_turns(
        [
            {"start": 0, "end": 0.05, "speaker": "SPEAKER_00"},
            {"start": 100, "end": 102, "speaker": "SPEAKER_01"},
        ],
        plan,
    )
    assert qa["validation_pass"] and qa["has_warnings"]
    assert qa["warning_counts"]["SUSPICIOUS_SHORT_TURN"] == 1
    assert qa["warning_counts"]["LONG_UNCOVERED_INTERVAL"] == 2
    assert qa["warning_counts"]["UNCOVERED_REVIEWED_BOUNDS"] > 0
    assert qa["speaker_confidence"] is None


@pytest.mark.parametrize("mutation", ["checksum", "coverage", "text", "timing", "source", "smoke"])
def test_aligned_parent_tamper_is_rejected(aligned, project_config, mutation):
    sid, _, _, path = aligned
    parent = read_artifact(path)
    if mutation == "checksum":
        parent["artifact_checksum"] = "0" * 64
    elif mutation == "coverage":
        parent["segments"].pop()
    elif mutation == "text":
        parent["segments"][0]["text"] = "invented wording"
    elif mutation == "timing":
        parent["segments"][0]["start"] = -1
    elif mutation == "source":
        parent["source_sha256"] = "0" * 64
    else:
        parent["configuration"]["mode"] = "SMOKE"
    atomic_write_json(path, parent if mutation == "checksum" else seal_artifact(parent))
    with pytest.raises(ValueError):
        d.make_plan(project_config, sid)


def test_smoke_is_separate_and_never_full(aligned, local_model, project_config):
    sid, _, _, _ = aligned
    full_path, full = d.make_plan(project_config, sid)
    report = execute(project_config, sid, smoke=True)
    assert report["state"] == "SUCCEEDED" and not report["full_diarization_complete"]
    assert report["processed_interval"] == [0, 180]
    assert "smoke" in Path(report["output_path"]).parts
    assert report["fingerprint"] != full["stage_fingerprint"]
    assert not d.checkpoint_path(full_path).exists()
    assert not read_artifact(Path(report["output_path"]))["full_diarization_complete"]
    assert not d.status_report(project_config, sid)["processing_complete"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupted_whole_job_is_explicit_and_retry_archives_attempt(
    aligned, local_model, project_config, interrupt
):
    sid, _, _, _ = aligned
    with pytest.raises(interrupt):
        d.run_diarization(
            project_config,
            sid,
            engine_factory=lambda _: Mock(side_effect=interrupt),
            extractor=fake_extract,
        )
    p, plan = d.make_plan(project_config, sid)
    cp = read_artifact(d.checkpoint_path(p))
    assert cp["status"] == "INTERRUPTED" and not (p.parent / "derived/audio.wav").exists()
    assert d.status_report(project_config, sid)["state"] == "INTERRUPTED"
    report = execute(project_config, sid)
    assert report["full_diarization_complete"]
    assert read_artifact(d.checkpoint_path(p))["attempt"] == 2
    assert read_artifact(p.parent / "attempts" / f"{cp['artifact_checksum']}.json") == cp


def test_status_recovers_stale_running_without_mutation(aligned, local_model, project_config):
    sid, _, _, _ = aligned
    execute(project_config, sid)
    p, _ = d.make_plan(project_config, sid)
    cp = read_artifact(d.checkpoint_path(p))
    cp.update(status="RUNNING")
    atomic_write_json(d.checkpoint_path(p), seal_artifact(cp))
    before = d.checkpoint_path(p).read_bytes()
    work = project_config.application.work_path / sid
    with d.exclusive_lock(work / "diarization/.worker.lock"):
        assert d.status_report(project_config, sid)["state"] == "RUNNING"
    assert d.status_report(project_config, sid)["state"] == "INTERRUPTED"
    assert d.checkpoint_path(p).read_bytes() == before


def test_execution_errors_sanitized_token_not_persisted(
    aligned, local_model, project_config, monkeypatch
):
    sid, _, _, _ = aligned
    token = "synthetic-private-token"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", token)

    def failing(path):
        assert "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["PYANNOTE_METRICS_ENABLED"] == "0"
        raise RuntimeError(token + "private audio")

    report = d.run_diarization(project_config, sid, engine_factory=failing, extractor=fake_extract)
    assert report["state"] == "FAILED" and report["failure_code"] == "RuntimeError"
    assert token not in json.dumps(report)
    for p in (project_config.application.work_path / sid / "diarization").rglob("*.json"):
        assert token not in p.read_text(encoding="utf-8") and "private audio" not in p.read_text(
            encoding="utf-8"
        )


def test_status_and_plan_never_import_model_access_network_or_write_status(
    aligned, local_model, project_config, monkeypatch
):
    sid, _, _, _ = aligned
    d.make_plan(project_config, sid)
    root = project_config.application.root
    baseline = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "pyannote",
            "whisperx",
            "transformers",
            "openai",
            "anthropic",
            "faster_whisper",
        }:
            raise AssertionError("model/LLM import forbidden")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(store, "_download", Mock(side_effect=AssertionError("download")))
    with d.deny_network():
        report = d.status_report(project_config, sid)
        assert d.make_plan(project_config, sid)[0].exists()
    assert report["state"] == "PENDING" and not report["models_loaded"]
    assert baseline == {
        p: (sha256_file(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


def test_offline_adapter_cpu_pinned_path_two_speakers_and_regular_overlap(
    local_model, monkeypatch, tmp_path
):
    import numpy as np

    torch = ModuleType("torch")
    torch.device = lambda name: name
    torch.set_num_threads = Mock()
    torch.inference_mode = d.deny_network
    torch.from_numpy = lambda samples: samples
    module = ModuleType("pyannote.audio")
    annotation = SimpleNamespace(
        itertracks=lambda **kwargs: iter(
            [
                (SimpleNamespace(start=0, end=3), "track0", "SPEAKER_00"),
                (SimpleNamespace(start=2, end=4), "track1", "SPEAKER_01"),
            ]
        )
    )
    pipeline = Mock(
        return_value=SimpleNamespace(
            speaker_diarization=annotation,
            exclusive_speaker_diarization=annotation,
        )
    )
    loader = Mock(return_value=pipeline)
    module.Pipeline = SimpleNamespace(from_pretrained=loader)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    engine = d.LocalDiarizer(local_model)
    loader.assert_called_once_with(
        str(local_model), token=False, cache_dir=local_model.parents[1] / ".hf-cache"
    )
    pipeline.to.assert_called_once_with("cpu")
    clip = tmp_path / "synthetic.wav"
    with wave.open(str(clip), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.array([-32768, 0, 32767], dtype="<i2").tobytes())
    output = engine(clip)
    audio = pipeline.call_args.args[0]
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].shape == (1, 3)
    assert audio["waveform"].dtype == np.float32
    assert audio["waveform"][0].tolist() == [-1, 0, 32767 / 32768]
    assert pipeline.call_args.kwargs == {"min_speakers": 2, "max_speakers": 2}
    assert output["turns"][0]["end"] > output["turns"][1]["start"]
    assert output["exclusive_turns"] == output["turns"]
    assert pipeline.embedding_batch_size == 8 and pipeline.segmentation_batch_size == 4
    assert set(output["turns"][0]) == {"start", "end", "speaker", "duration"}


@pytest.mark.parametrize(
    "channels,sample_rate,frames", [(2, 16000, 3), (1, 8000, 3), (1, 16000, 0)]
)
def test_adapter_refuses_invalid_or_empty_pcm(monkeypatch, tmp_path, channels, sample_rate, frames):
    torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    engine = d.LocalDiarizer.__new__(d.LocalDiarizer)
    engine.pipeline = Mock(side_effect=AssertionError("inference must not start"))
    clip = tmp_path / "invalid.wav"
    with wave.open(str(clip), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00" * channels * frames)
    with pytest.raises(ValueError, match="diarization PCM"):
        engine(clip)
    engine.pipeline.assert_not_called()


def test_cli_defaults_plan_and_status_no_plan_creation(
    aligned, project_config, monkeypatch, capsys
):
    sid, _, _, _ = aligned
    monkeypatch.setattr("session_pipeline.cli.load_config", lambda **_: project_config)
    assert run(["diarize", sid, "--status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "NOT_PLANNED"
    assert not (project_config.application.work_path / sid / "diarization").exists()
    assert run(["diarize", sid, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["state"] == "PLANNED" and not report["models_loaded"]
    assert run(["diarize", sid, "--status", "--fingerprint", report["fingerprint"], "--json"]) == 0


def test_model_binding_stable_when_provisioner_pins_git_blob_sha256(aligned, project_config):
    spec = d.model_spec(project_config)
    before = d.model_binding(spec)
    for f in spec["files"]:
        if f["sha256"] is None:
            f["sha256"] = "a" * 64
    assert d.model_binding(spec) == before


def test_existing_gated_provisioner_uses_exact_revision_and_never_saves_token(
    aligned, local_model, project_config, monkeypatch
):
    root = project_config.application.root
    token = "synthetic-one-time-token"
    monkeypatch.setenv("HF_TOKEN", token)
    path = local_model / "config.yaml"
    original = path.read_bytes()
    path.unlink()
    calls = []

    def preflight(spec, value):
        assert spec["revision"] == d.MODEL["diarization"]["revision"] and value == token
        calls.append("preflight")

    def download(spec, file, project_root, value, **kwargs):
        assert file["path"] == "config.yaml" and value == token
        assert spec["revision"] == d.MODEL["diarization"]["revision"]
        path.write_bytes(original)
        calls.append("download")

    monkeypatch.setattr(store, "_access_preflight", preflight)
    monkeypatch.setattr(store, "_download", download)
    with d.deny_network():
        result = store.models_report(root, action="provision", role="diarization")
        assert result["models_ready"] and result["models"][0]["verification"] == "SHA256"
        monkeypatch.delenv("HF_TOKEN")
        assert store.local_model_path(root, "diarization") == local_model
    assert calls == ["preflight", "download"] and token not in json.dumps(result)
    assert token not in (local_model / ".provision.json").read_text(encoding="utf-8")
    assert token not in (root / "config/models.lock.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", ["parent", "third_speaker", "output"])
def test_completed_result_corruption_not_reused(aligned, local_model, project_config, mutation):
    sid, _, _, _ = aligned
    report = execute(project_config, sid)
    p, plan = d.make_plan(project_config, sid)
    cp = read_artifact(d.checkpoint_path(p))
    if mutation == "output":
        output = Path(report["output_path"])
        output.write_bytes(b"corrupt")
        assert d.status_report(project_config, sid)["state"] == "INVALID_OUTPUT"
        with pytest.raises(ValueError):
            execute(project_config, sid)
        return
    result = cp["result"]
    if mutation == "parent":
        result["parent_checksums"]["aligned_asr"] = "0" * 64
    else:
        result["turns"].append({"start": 171, "end": 172, "speaker": "SPEAKER_02", "duration": 1})
    cp["result"] = seal_artifact(result)
    atomic_write_json(d.checkpoint_path(p), seal_artifact(cp))
    assert d.status_report(project_config, sid)["state"] == "INVALID_CHECKPOINT"
    with pytest.raises(ValueError):
        execute(project_config, sid)


def test_model_and_all_configured_hf_caches_are_ignored():
    root = Path(__file__).resolve().parents[1]
    for path in (
        "models/diarization/test/config.yaml",
        "models/.hf-cache/hub/test",
        "models/diarization/test/.cache/huggingface/download/test",
        ".hf_cache/hub/test",
        ".huggingface/token",
        ".cache/huggingface/hub/test",
    ):
        assert (
            subprocess.run(["git", "check-ignore", "-q", path], cwd=root, check=False).returncode
            == 0
        )


def test_diarization_module_import_has_no_heavy_dependencies():
    code = """
import builtins
real = builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','pyannote','whisperx','transformers',
                              'openai','anthropic','faster_whisper'}:
        raise AssertionError('heavy import')
    return real(name,*args,**kwargs)
builtins.__import__=guard
import session_pipeline.diarization
print('PASS')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "PASS"


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_raw_before_validation_and_clipping(
    aligned, local_model, project_config, monkeypatch, boundary
):
    sid, _, _, _ = aligned
    payload = turns()
    if boundary == "start":
        payload[0]["start"] = -0.025
    else:
        payload[1]["end"] = 620.025
    payload = {
        "turns": payload,
        "exclusive_turns": [
            {"start": -0.025, "end": 100, "speaker": "SPEAKER_00"},
            {"start": 100, "end": 620.025, "speaker": "SPEAKER_01"},
        ],
    }
    p, plan = d.make_plan(project_config, sid)
    originals = d.load_input(project_config, sid)[-1]
    baseline = {v: (v.read_bytes(), v.stat().st_mtime_ns) for v in originals}
    evaluate = d.evaluate_turns

    def checked(turns, plan):
        assert read_artifact(d.raw_path(p))["turns"] == payload["turns"]
        return evaluate(turns, plan)

    monkeypatch.setattr(d, "evaluate_turns", checked)
    report = d.run_diarization(
        project_config, sid, engine_factory=lambda _: lambda clip: payload, extractor=fake_extract
    )
    raw = d.read_raw(p, plan)
    result = read_artifact(Path(report["output_path"]))
    assert raw["turns"] == payload["turns"]
    assert raw["exclusive_turns"] == payload["exclusive_turns"]
    assert result["exclusive_turns"][0]["start"] == 0
    assert result["exclusive_turns"][1]["end"] == 620
    assert result["raw_anomalies"]["RAW_OUT_OF_BOUNDS_TURN"] == 1
    assert result["raw_anomalies"]["CLIPPED_TO_AUDIO_BOUNDS"] == 1
    clipping = result["raw_anomalies"]["clipped_turns"][0]
    assert clipping["raw_start"] == payload["turns"][clipping["turn_index"]]["start"]
    assert clipping["raw_end"] == payload["turns"][clipping["turn_index"]]["end"]
    assert all(0 <= t["start"] < t["end"] <= 620 for t in result["turns"])
    assert result["qa"]["error_counts"]["OUT_OF_BOUNDS_TURN"] == 0
    assert result["qa"]["overlap_seconds"] > 0
    assert report["state"] == "SUCCEEDED_WITH_WARNINGS"
    assert report["raw_turn_count"] == report["regular_turn_count"] == 2
    assert report["exclusive_turn_count"] == 2 and report["clipping_count"] == 1
    assert baseline == {v: (v.read_bytes(), v.stat().st_mtime_ns) for v in originals}


@pytest.mark.parametrize("invalid", ["outside", "third", "one", "order", "duration"])
def test_invalid_output_is_saved_and_reused_without_inference(
    aligned, local_model, project_config, invalid
):
    sid, _, _, _ = aligned
    payload = turns()
    if invalid == "outside":
        payload.append({"start": 621, "end": 622, "speaker": "SPEAKER_00"})
    elif invalid == "third":
        payload.append({"start": 171, "end": 172, "speaker": "SPEAKER_02"})
    elif invalid == "one":
        payload = payload[:1]
    elif invalid == "order":
        payload.reverse()
    else:
        payload[0]["duration"] = 3
    report = d.run_diarization(
        project_config, sid, engine_factory=lambda _: lambda clip: payload, extractor=fake_extract
    )
    p, plan = d.make_plan(project_config, sid)
    raw = d.read_raw(p, plan)
    before = d.raw_path(p).read_bytes(), d.raw_path(p).stat().st_mtime_ns
    assert raw["turns"] == payload
    assert report["state"] == "FAILED"
    assert report["execution_state"] == "RAW_PERSISTED_VALIDATION_FAILED"
    assert report["inference_completed"] and report["raw_inference_persisted"]
    assert report["output_path"] is None
    with d.deny_network():
        again = d.run_diarization(
            project_config,
            sid,
            engine_factory=Mock(side_effect=AssertionError("inference")),
            extractor=Mock(side_effect=AssertionError("decode")),
        )
    assert again["state"] == "FAILED"
    assert before == (d.raw_path(p).read_bytes(), d.raw_path(p).stat().st_mtime_ns)


@pytest.mark.parametrize("interrupt_after_raw", [False, True])
def test_interrupt_raw_recovery(
    aligned, local_model, project_config, monkeypatch, interrupt_after_raw
):
    sid, _, _, _ = aligned
    p, plan = d.make_plan(project_config, sid)
    original = d.raw_validation
    if interrupt_after_raw:
        monkeypatch.setattr(d, "raw_validation", Mock(side_effect=KeyboardInterrupt))
        factory = Mock(return_value=lambda clip: turns())
    else:
        factory = Mock(return_value=Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        d.run_diarization(project_config, sid, engine_factory=factory, extractor=fake_extract)
    report = d.status_report(project_config, sid)
    assert report["state"] == "INTERRUPTED"
    assert report["raw_inference_persisted"] == interrupt_after_raw
    assert report["inference_completed"] == interrupt_after_raw
    assert d.raw_path(p).exists() == interrupt_after_raw
    if interrupt_after_raw:
        monkeypatch.setattr(d, "raw_validation", original)
        # Simulate power loss leaving stale RUNNING after the atomic raw write.
        cp = read_artifact(d.checkpoint_path(p))
        cp["status"] = "RUNNING"
        atomic_write_json(d.checkpoint_path(p), seal_artifact(cp))
        assert d.status_report(project_config, sid)["state"] == "INTERRUPTED"
        recovered = d.run_diarization(
            project_config,
            sid,
            engine_factory=Mock(side_effect=AssertionError("model")),
            extractor=Mock(side_effect=AssertionError("decode")),
        )
        assert recovered["full_diarization_complete"]


def test_status_no_model_access_and_canonical_warnings(
    aligned, local_model, project_config, monkeypatch
):
    sid, _, _, _ = aligned
    report = execute(project_config, sid)
    result = read_artifact(Path(report["output_path"]))
    assert result["qa"]["warning_counts"]["UNCOVERED_REVIEWED_BOUNDS"] > 0
    assert result["qa"]["uncovered_reviewed_segment_ids"]
    assert result["turns"] == [dict(t, duration=t["end"] - t["start"]) for t in turns()]
    monkeypatch.setattr(d, "inspect_model", Mock(side_effect=AssertionError("model access")))
    monkeypatch.setattr(d, "LocalDiarizer", Mock(side_effect=AssertionError("load")))
    monkeypatch.setattr(store, "_download", Mock(side_effect=AssertionError("download")))
    with d.deny_network():
        status = d.status_report(project_config, sid)
    assert status["raw_inference_persisted"] and status["inference_completed"]
    assert status["validation_state"] == "PASSED" and not status["models_loaded"]


def test_success_without_warnings(aligned, local_model, project_config):
    sid, _, _, _ = aligned
    payload = [
        {"start": 0, "end": 310, "speaker": "SPEAKER_00"},
        {"start": 310, "end": 620, "speaker": "SPEAKER_01"},
    ]
    report = d.run_diarization(
        project_config, sid, engine_factory=lambda _: lambda clip: payload, extractor=fake_extract
    )
    assert report["state"] == "SUCCEEDED" and report["processing_complete"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_is_preserved_before_failure(aligned, local_model, project_config, value):
    sid, _, _, _ = aligned
    payload = turns()
    payload[0]["start"] = value
    report = d.run_diarization(
        project_config, sid, engine_factory=lambda _: lambda clip: payload, extractor=fake_extract
    )
    p, plan = d.make_plan(project_config, sid)
    raw = d.read_raw(p, plan)
    assert raw["turns"][0]["start"] == {"__raw_nonfinite_float__": repr(value)}
    assert report["raw_inference_persisted"] and report["validation_state"] == "FAILED"
    assert not report["processing_complete"]


def test_raw_write_failure_is_not_misreported_as_inference_failure(
    aligned, local_model, project_config, monkeypatch
):
    sid, _, _, _ = aligned
    write = d.atomic_write_json

    def fail_raw(path, document):
        if path.name == "raw-inference.json":
            raise OSError("synthetic disk failure")
        return write(path, document)

    monkeypatch.setattr(d, "atomic_write_json", fail_raw)
    report = execute(project_config, sid)
    assert report["state"] == "FAILED" and report["inference_completed"]
    assert not report["raw_inference_persisted"]
    assert report["execution_state"] == "INFERENCE_COMPLETED_RAW_PERSISTENCE_FAILED"


def test_legacy_failed_checkpoint_without_raw_remains_explicit(aligned, project_config):
    sid, _, _, _ = aligned
    p, plan = d.make_plan(project_config, sid)
    payload = turns()
    payload[1]["end"] = 621
    cp = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "diarization_checkpoint",
            "status": "FAILED",
            "attempt": 1,
            "plan_checksum": plan["artifact_checksum"],
            "parent_checksums": plan["parent_checksums"],
            "stage_fingerprint": plan["stage_fingerprint"],
            "failure_phase": "OUTPUT_VALIDATION",
            "qa": d.evaluate_turns(payload, plan),
        }
    )
    atomic_write_json(d.checkpoint_path(p), cp)
    report = d.status_report(project_config, sid)
    assert report["inference_completed"] and not report["raw_inference_persisted"]
    assert report["execution_state"] == "INFERENCE_COMPLETED_RAW_NOT_PERSISTED"
    assert not report["processing_complete"]
