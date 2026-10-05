from __future__ import annotations

import copy
import json
import socket
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest

from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_quality import (
    RULES,
    QualityThresholds,
    audit,
    create_review,
    distribution,
    pad,
    session_paths,
)
from session_pipeline.asr_retry import (
    DECODE,
    MODEL,
    deny_network,
    extract_clip,
    launch_worker,
    local_engine,
    make_plan,
    owned_segments,
    run_retry,
)
from session_pipeline.cli import run
from session_pipeline.models import session_id_from_sha256, sha256_file


def segment(start=20.0, end=21.0, text="synthetic original", **diagnostics):
    return {
        "start": start,
        "end": end,
        "text": text,
        "temperature": 0.0,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "words": [{"start": start, "end": end, "word": text, "probability": 0.9}],
        **diagnostics,
    }


def synthetic_raw(segments):
    return {"source_sha256": "a" * 64, "probe": {"duration_seconds": 120.0}, "segments": segments}


@pytest.fixture
def canonical(project_config):
    root = project_config.application.root
    source = root / "sessions/synthetic.wav"
    source.write_bytes(b"synthetic fixture only; extractor is injected")
    sha = sha256_file(source)
    sid = session_id_from_sha256(sha)
    raw_path, work, _ = session_paths(project_config, sid)
    raw = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "asr",
            "status": "SUCCEEDED",
            "session_id": sid,
            "source_sha256": sha,
            "source": {"sha256": sha},
            "model": MODEL,
            "probe": {"duration_seconds": 120.0},
            "segment_count": 2,
            "segments": [segment(20, 21, temperature=0.8), segment(50, 51, temperature=1.0)],
        }
    )
    atomic_write_json(raw_path, raw)
    atomic_write_json(
        work / "integrity.json",
        seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "integrity",
                "status": "SUCCEEDED",
                "decode": {"success": True},
                "source": {
                    "sha256": sha,
                    "original_filename": source.name,
                    "size_bytes": source.stat().st_size,
                },
            }
        ),
    )
    txt = raw_path.with_suffix(".txt")
    txt.write_text("immutable text fixture", encoding="utf-8")
    (root / "config/asr-quality.json").write_text(
        json.dumps(asdict(QualityThresholds(unknown_speech_gap_seconds=120)))
    )
    return sid, raw_path, source, txt


def test_all_explicit_flags_and_determinism():
    repeated = "one two three four five " * 3
    low = [{"start": 20.0, "end": 20.1, "word": "x", "probability": 0.1}] * 2
    raw = synthetic_raw(
        [
            segment(
                20, 20.1, repeated, temperature=0.6, avg_logprob=-1.1, no_speech_prob=0.9, words=low
            )
        ]
    )
    result = audit(raw, QualityThresholds(), speech=[[0, 10]], boundaries=[20])
    assert all(result["entry_flag_counts"][flag] > 0 for flag in RULES)
    assert result == audit(raw, QualityThresholds(), speech=[[0, 10]], boundaries=[20])
    assert result["entries"][0]["original_text"] == repeated
    assert "confidence" not in result


def test_conservative_edges_missing_values_and_gaps():
    raw = synthetic_raw(
        [segment(0, 1, temperature=0.59, avg_logprob=-1, words=[{"probability": 0.01}])]
    )
    result = audit(raw, QualityThresholds(), speech=[])
    assert result["flagged_segment_count"] == 0
    assert result["entry_flag_counts"]["SUSPICIOUS_GAP"] == 0
    assert audit(raw, QualityThresholds())["entry_flag_counts"]["SUSPICIOUS_GAP"] == 1
    assert distribution([None, "unknown"])["missing"] == 2
    assert distribution([1, 2, 3])["median"] == 2


def test_repetition_marks_preceding_boundary():
    text = "synthetic repeated text only"
    raw = synthetic_raw([segment(10, 11, text), segment(12, 13, text)])
    result = audit(raw, QualityThresholds(), speech=[], boundaries=[10])
    assert "BOUNDARY_RISK" in result["entries"][0]["reasons"]
    assert result["segment_flag_counts"]["SUSPICIOUS_REPETITION"] == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"low_word_min_count": True},
        {"low_word_probability": 1.1},
        {"high_temperature": float("nan")},
        {"gap_seconds": 0},
        {"retry_max_target_seconds": 0},
        {"retry_padding_seconds": 11},
    ],
)
def test_invalid_thresholds(settings):
    with pytest.raises(ValueError):
        QualityThresholds(**settings)


def test_padding_clamping():
    assert pad([1, 119], 5, 120) == [0, 120]
    with pytest.raises(ValueError):
        pad([-1, 3], 5, 120)


def test_manifest_identity_and_configuration(canonical, project_config):
    sid, raw_path, source, txt = canonical
    before = [p.read_bytes() for p in (raw_path, source, txt)]
    path, review = create_review(project_config, sid)
    assert review["entries"][0]["original_text"] == "synthetic original"
    assert review["entries"][0]["suggested_retry_interval"] == [15, 26]
    assert create_review(project_config, sid)[1]["artifact_checksum"] == review["artifact_checksum"]
    thresholds = asdict(QualityThresholds(high_temperature=1, unknown_speech_gap_seconds=120))
    (project_config.application.root / "config/asr-quality.json").write_text(json.dumps(thresholds))
    changed = create_review(project_config, sid)[1]
    assert changed["stage_fingerprint"] != review["stage_fingerprint"]
    assert len(changed["entries"]) == 1
    assert path.is_relative_to(project_config.application.work_path)
    assert before == [p.read_bytes() for p in (raw_path, source, txt)]


def test_candidate_preservation_resume_and_failure(canonical, project_config):
    sid, raw_path, source, txt = canonical
    create_review(project_config, sid)
    path, plan = make_plan(project_config, sid)
    assert len(plan["windows"]) == 2
    baseline = [p.read_bytes() for p in (raw_path, source, txt)]
    calls = []

    def extractor(original, derived, interval):
        assert original == source
        assert derived.is_relative_to(path.parent)
        calls.append(interval)

    def inference(engine, clip, offset):
        if len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        return [segment(offset + 5, offset + 6, "synthetic retry")], {}

    with pytest.raises(RuntimeError):
        run_retry(
            project_config,
            sid,
            path,
            engine_factory=lambda _: object(),
            extractor=extractor,
            inference=inference,
        )
    candidate = path.parent / "candidate-0000.json"
    before = candidate.read_bytes()
    assert read_artifact(path.parent / "progress.json")["completed_windows"] == 1
    result = run_retry(
        project_config,
        sid,
        path,
        engine_factory=lambda _: object(),
        extractor=extractor,
        inference=lambda engine, clip, offset: (
            [segment(offset + 5, offset + 6, "synthetic retry")],
            {},
        ),
    )
    assert result["completed_windows"] == 2
    assert len(calls) == 3
    assert before == candidate.read_bytes()
    stored = read_artifact(candidate)
    assert stored["candidate_a"]["segments"][0]["text"] == "synthetic original"
    assert stored["candidate_b"]["segments"][0]["text"] == "synthetic retry"
    assert stored["review_required"] and not stored["canonical_replacement"]

    def forbidden(*args):
        pytest.fail("completed checkpoint must not reload model or repeat inference")

    run_retry(
        project_config,
        sid,
        path,
        engine_factory=forbidden,
        extractor=forbidden,
        inference=forbidden,
    )
    assert baseline == [p.read_bytes() for p in (raw_path, source, txt)]


def test_retry_identity_tampering_budget_and_stale_parent(canonical, project_config):
    sid, raw_path, _, _ = canonical
    _, review = create_review(project_config, sid)
    path, plan = make_plan(project_config, sid)
    subset, _ = make_plan(project_config, sid, [review["entries"][0]["entry_id"]])
    assert subset != path
    assert make_plan(project_config, sid, max_seconds=1)[1]["execution_allowed"] is False
    assert make_plan(project_config, sid, [])[1]["windows"] == []
    modified = copy.deepcopy(plan)
    modified["windows"][0]["decoded_interval"] = [0, 120]
    atomic_write_json(path, seal_artifact(modified))
    with pytest.raises(ValueError, match="semantics"):
        run_retry(project_config, sid, path, engine_factory=lambda _: pytest.fail("model load"))
    raw = read_artifact(raw_path)
    raw["segments"][0]["temperature"] = 0.9
    atomic_write_json(raw_path, seal_artifact(raw))
    with pytest.raises(ValueError, match="stale"):
        make_plan(project_config, sid)


def test_bad_retry_timestamps_fail_without_candidate(canonical, project_config):
    sid, _, _, _ = canonical
    create_review(project_config, sid)
    path, _ = make_plan(project_config, sid)
    with pytest.raises(ValueError, match="timestamps"):
        run_retry(
            project_config,
            sid,
            path,
            engine_factory=lambda _: object(),
            extractor=lambda *args: None,
            inference=lambda *args: ([segment(0, 120)], {}),
        )
    assert not (path.parent / "candidate-0000.json").exists()


def test_word_ownership_does_not_duplicate_boundary():
    value = segment(
        0,
        3,
        words=[{"start": 0, "end": 2, "word": "first"}, {"start": 2, "end": 3, "word": "second"}],
    )
    assert owned_segments([value], [0, 1]) == []
    assert len(owned_segments([value], [1, 3])[0]["words"]) == 2


def test_status_and_planning_no_network_no_heavy_imports(canonical, project_config, monkeypatch):
    sid, _, _, _ = canonical
    monkeypatch.chdir(project_config.application.root)
    import session_pipeline.cli as cli

    monkeypatch.setattr(cli, "load_config", lambda **kwargs: project_config)
    with deny_network():
        assert run(["qa", "asr", sid, "--json"]) == 0
        assert run(["qa", "status", sid, "--json"]) == 0
        import session_pipeline.asr_retry as retry

        monkeypatch.setattr(retry, "launch_worker", lambda *args: pytest.fail("worker started"))
        assert run(["qa", "retry", sid, "--json"]) == 0
        with pytest.raises(RuntimeError, match="network disabled"):
            socket.create_connection(("example.com", 443))
    root = Path(__file__).resolve().parents[1]
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'openai', 'anthropic', 'torch', 'faster_whisper', 'whisperx',
                            'transformers', 'huggingface_hub'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import session_pipeline.asr_quality
import session_pipeline.asr_retry
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_local_worker_ownership(canonical, project_config, monkeypatch):
    sid, _, _, _ = canonical
    create_review(project_config, sid)
    path, _ = make_plan(project_config, sid)

    def fake_run(command, *, env, check):
        assert command[:3] == [sys.executable, "-m", "session_pipeline.asr_retry"]
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert "HF_TOKEN" not in env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert launch_worker(project_config, sid, path) == 0


def test_local_model_loader_never_uses_remote_name(project_root, monkeypatch):
    import session_pipeline.asr_retry as retry

    local = project_root / "models" / MODEL["asr"]["revision"]
    calls = []

    def factory(path, **settings):
        calls.append((path, settings))
        return types.SimpleNamespace(model=types.SimpleNamespace(compute_type="int8_float32"))

    monkeypatch.setattr(retry, "local_model_path", lambda root, role: local)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=factory))
    assert local_engine(project_root).model.compute_type == "int8_float32"
    path, settings = calls[0]
    assert path == str(local)
    assert settings["local_files_only"] and settings["use_auth_token"] is False
    assert settings["device"] == "cpu" and settings["compute_type"] == "int8"
    assert DECODE["language"] == "fa" and DECODE["initial_prompt"] is None
    monkeypatch.setattr(retry, "local_model_path", lambda root, role: project_root / "wrong")
    with pytest.raises(ValueError, match="revision"):
        local_engine(project_root)
    assert len(calls) == 1


def test_derived_clip_preserves_tiny_source(tiny_wav, tmp_path):
    import shutil
    import wave

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    before = tiny_wav.read_bytes()
    derived = tmp_path / "retry/derived.wav"
    extract_clip(tiny_wav, derived, [0.05, 0.15])
    assert tiny_wav.read_bytes() == before
    with wave.open(str(derived), "rb") as stream:
        assert stream.getnframes() / stream.getframerate() == pytest.approx(0.1)


def test_package_change_changes_retry_identity(canonical, project_config, monkeypatch):
    import session_pipeline.asr_retry as retry

    sid, _, _, _ = canonical
    create_review(project_config, sid)
    path, _ = make_plan(project_config, sid)
    monkeypatch.setattr(retry, "package_identity", lambda: {"synthetic-package": "changed"})
    changed, _ = make_plan(project_config, sid)
    assert changed != path
