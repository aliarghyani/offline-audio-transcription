from __future__ import annotations

import copy
import http.client
import json
import subprocess
import sys
import threading
import wave
from dataclasses import asdict
from pathlib import Path

import pytest

from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_quality import QualityThresholds, create_review, session_paths
from session_pipeline.asr_retry import MODEL, deny_network, make_plan, owned_segments, run_retry
from session_pipeline.human_review import (
    DecisionConflict,
    ReviewError,
    boundary_issues,
    build_bundle,
    context_fingerprint,
    load_bundle,
    load_context,
    load_decisions,
    reconcile_candidate,
    review_entries,
    review_status,
    save_decision,
    validate_target_windows,
)
from session_pipeline.models import session_id_from_sha256, sha256_file
from session_pipeline.review_merge import merge_review
from session_pipeline.review_server import create_server


def seg(start, end, text, temperature=0.0):
    return {
        "start": start,
        "end": end,
        "text": text,
        "temperature": temperature,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
        "words": [{"start": start, "end": end, "word": text, "probability": 0.9}],
    }


def wav(path, duration):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * round(duration * 16000))


@pytest.fixture
def completed(project_config):
    root = project_config.application.root
    source = root / "sessions/synthetic.wav"
    wav(source, 30)
    sha = sha256_file(source)
    sid = session_id_from_sha256(sha)
    raw_path, work, _ = session_paths(project_config, sid)
    segments = [
        seg(0, 1, "before"),
        seg(10, 12, "original one", 0.8),
        seg(13, 14, "between"),
        seg(20, 21, "original two", 0.8),
        seg(29, 30, "after"),
    ]
    raw = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "asr",
            "status": "SUCCEEDED",
            "session_id": sid,
            "source_sha256": sha,
            "source": {"sha256": sha},
            "model": MODEL,
            "probe": {"duration_seconds": 30.0},
            "segment_count": len(segments),
            "segments": segments,
        }
    )
    atomic_write_json(raw_path, raw)
    raw_path.with_suffix(".txt").write_text("immutable original fixture", encoding="utf-8")
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
    (root / "config/asr-quality.json").write_text(json.dumps(asdict(QualityThresholds())))
    create_review(project_config, sid)
    plan_path, plan = make_plan(project_config, sid)

    def extractor(source_path, path, interval):
        assert source_path == source
        wav(path, interval[1] - interval[0])

    def inference(engine, path, offset):
        target = next(
            w["target_interval"] for w in plan["windows"] if w["decoded_interval"][0] == offset
        )
        return [
            seg(offset, offset + 1, "PAD BEFORE"),
            seg(*target, "retry target"),
            seg(target[1] + 1, target[1] + 2, "PAD AFTER"),
        ], {"language": "fa"}

    run_retry(
        project_config,
        sid,
        plan_path,
        engine_factory=lambda _: object(),
        extractor=extractor,
        inference=inference,
    )
    return sid, plan_path, raw_path, source


def choose_all(config, sid, choices):
    context = load_context(config, sid)
    bundle = load_bundle(context)
    decisions = load_decisions(context, bundle)
    for entry, choice in zip(bundle["entries"], choices, strict=True):
        decisions = save_decision(
            config,
            sid,
            context.plan["stage_fingerprint"],
            entry["review_id"],
            choice,
            decisions["artifact_checksum"],
        )
    return decisions


def test_persisted_decisions_build_resume_and_conflict(completed, project_config):
    sid, _, _, _ = completed
    report = build_bundle(project_config, sid)
    assert report["entry_count"] == report["unresolved_count"] == 2
    assert report["audio_ready_count"] == 2 and report["b_selectable_count"] == 2
    context = load_context(project_config, sid)
    bundle = load_bundle(context)
    original = load_decisions(context, bundle)
    new = save_decision(
        project_config,
        sid,
        context.plan["stage_fingerprint"],
        bundle["entries"][0]["review_id"],
        "USE_A",
        original["artifact_checksum"],
    )
    with pytest.raises(DecisionConflict):
        save_decision(
            project_config,
            sid,
            context.plan["stage_fingerprint"],
            bundle["entries"][1]["review_id"],
            "USE_B",
            original["artifact_checksum"],
        )
    assert build_bundle(project_config, sid)["decision_counts"]["USE_A"] == 1
    assert load_decisions(load_context(project_config, sid), bundle) == new


def test_merge_refuses_incomplete_no_auto_acceptance(completed, project_config):
    sid, _, raw_path, _ = completed
    build_bundle(project_config, sid)
    with pytest.raises(ReviewError, match="incomplete"):
        merge_review(project_config, sid)
    assert not (raw_path.parent / "reviewed").exists()
    assert review_status(project_config, sid)["decision_counts"]["USE_B"] == 0


@pytest.mark.parametrize("choices", [("USE_A", "USE_A"), ("USE_B", "USE_A"), ("USE_A", "UNCLEAR")])
def test_selections_context_exclusion_provenance_and_raw_preservation(
    completed, project_config, choices
):
    sid, _, raw_path, source = completed
    baseline = [p.read_bytes() for p in (raw_path, raw_path.with_suffix(".txt"), source)]
    build_bundle(project_config, sid)
    decisions = choose_all(project_config, sid, choices)
    result = merge_review(project_config, sid)
    output = read_artifact(Path(result["reviewed_json"]))
    text = Path(result["reviewed_txt"]).read_text(encoding="utf-8")
    assert "PAD BEFORE" not in text and "PAD AFTER" not in text
    assert "before" in text and "between" in text and "after" in text
    if choices[0] == "USE_B":
        assert "retry target" in text and "original one" not in text
    else:
        assert "original one" in text and "retry target" not in text
    assert "original two" in text
    assert ("[UNRESOLVED" in text) == (choices[1] == "UNCLEAR")
    assert output["qa_acceptance"]["status"] == "PENDING"
    assert output["qa_acceptance"]["alignment_ready"] is False
    assert (
        output["provenance"]["raw_artifact_checksum"]
        == read_artifact(raw_path)["artifact_checksum"]
    )
    assert output["provenance"]["decision_manifest_checksum"] == decisions["artifact_checksum"]
    assert output["provenance"]["source_audio_sha256"] == sha256_file(source)
    assert [s["choice"] for s in output["provenance"]["spans"]] == list(choices)
    assert output["rendered_txt_sha256"] == sha256_file(Path(result["reviewed_txt"]))
    assert baseline == [p.read_bytes() for p in (raw_path, raw_path.with_suffix(".txt"), source)]
    output_before = Path(result["reviewed_json"]).read_bytes()
    assert merge_review(project_config, sid) == result
    assert Path(result["reviewed_json"]).read_bytes() == output_before


def test_changed_decision_creates_new_lineage(completed, project_config):
    sid, _, _, _ = completed
    build_bundle(project_config, sid)
    choose_all(project_config, sid, ("USE_A", "USE_A"))
    first = merge_review(project_config, sid)
    before = Path(first["reviewed_json"]).read_bytes()
    choose_all(project_config, sid, ("USE_B", "USE_A"))
    second = merge_review(project_config, sid)
    assert first["reviewed_json"] != second["reviewed_json"]
    assert Path(first["reviewed_json"]).read_bytes() == before


def test_overlap_rejected_padding_overlap_allowed():
    windows = [
        {"window_index": 0, "target_interval": [10, 12], "decoded_interval": [5, 17]},
        {"window_index": 1, "target_interval": [13, 15], "decoded_interval": [8, 20]},
    ]
    validate_target_windows(windows, 30)
    windows[1]["target_interval"] = [11, 15]
    with pytest.raises(ReviewError, match="overlapping"):
        validate_target_windows(windows, 30)


def test_boundary_crossing_and_empty_are_not_silently_selected(completed, project_config):
    sid, _, _, _ = completed
    c = load_context(project_config, sid).candidates[0]
    assert boundary_issues(c) == []  # Timed padded context is safe when owned words fit.
    bad = copy.deepcopy(c)
    bad["candidate_b"]["raw_segments"][1]["words"][0]["start"] = 9.9
    assert "B_WORD_CROSSES_TARGET" in boundary_issues(bad)
    bad = copy.deepcopy(c)
    bad["candidate_a"]["segments"][0]["start"] = 9
    assert "A_SEGMENT_CROSSES_TARGET" in boundary_issues(bad)
    bad = copy.deepcopy(c)
    bad["candidate_b"]["raw_segments"][1]["words"] = []
    bad["candidate_b"]["raw_segments"][1]["start"] = 9
    assert "B_UNTIMED_SEGMENT_CROSSES_TARGET" in boundary_issues(bad)
    bad["candidate_b"]["segments"] = []
    assert "B_TARGET_EMPTY" in boundary_issues(bad)
    split = copy.deepcopy(c)
    words = [
        seg(5, 6, "context")["words"][0],
        seg(10, 12, "target")["words"][0],
        seg(16, 17, "context")["words"][0],
    ]
    split["candidate_b"]["raw_segments"] = [
        {**seg(5, 17, "context target context"), "words": words}
    ]
    split["candidate_b"]["segments"] = owned_segments(
        split["candidate_b"]["raw_segments"], [10, 12]
    )
    assert (
        boundary_issues(split) == []
    )  # A segment can be split only at clean timed word boundaries.


@pytest.mark.parametrize("empty", [True, False])
def test_unsafe_b_choice_refused_but_human_a_or_unclear_allowed(completed, project_config, empty):
    sid, plan_path, _, _ = completed
    path = plan_path.parent / "candidate-0000.json"
    c = read_artifact(path)
    if empty:
        c["candidate_b"]["raw_segments"] = []
        c["candidate_b"]["segments"] = []
        c["candidate_b_empty"] = True
    else:
        c["candidate_b"]["raw_segments"][1]["words"][0]["start"] = 9.9
        c["candidate_b"]["segments"] = owned_segments(
            c["candidate_b"]["raw_segments"], c["window"]["target_interval"]
        )
    atomic_write_json(path, seal_artifact(c))
    build_bundle(project_config, sid)
    context = load_context(project_config, sid)
    bundle = load_bundle(context)
    d = load_decisions(context, bundle)
    if empty:
        with pytest.raises(ReviewError, match="no usable target text"):
            save_decision(
                project_config,
                sid,
                context.plan["stage_fingerprint"],
                bundle["entries"][0]["review_id"],
                "USE_B",
                d["artifact_checksum"],
            )
    else:
        saved = choose_all(project_config, sid, ("USE_B", "USE_A"))
        assert (
            saved["decisions"][bundle["entries"][0]["review_id"]]["merge_status"]
            == "MANUAL_BOUNDARY_REQUIRED"
        )
        assert review_status(project_config, sid)["manual_boundary_selected_count"] == 1
        with pytest.raises(ReviewError, match="window 1"):
            merge_review(project_config, sid)
        assert not (context.raw_path.parent / "reviewed").exists()
    choose_all(project_config, sid, ("UNCLEAR", "USE_A"))
    assert merge_review(project_config, sid)["unresolved_count"] == 1


def test_corrupt_candidate_and_audio_fail_closed(completed, project_config):
    sid, plan_path, _, _ = completed
    build_bundle(project_config, sid)
    clip = plan_path.parent / "derived/retry-0000.wav"
    payload = bytearray(clip.read_bytes())
    payload[-1] = 1
    clip.write_bytes(payload)
    with pytest.raises(ReviewError, match="audio changed"):
        review_status(project_config, sid)
    c_path = plan_path.parent / "candidate-0000.json"
    c = read_artifact(c_path)
    c["candidate_a"]["segments"][0]["text"] = "altered original"
    atomic_write_json(c_path, seal_artifact(c))
    with pytest.raises(ReviewError, match="preserve"):
        load_context(project_config, sid)


def test_offline_build_status_and_import_boundary(completed, project_config):
    sid, _, _, _ = completed
    with deny_network():
        assert build_bundle(project_config, sid)["entry_count"] == 2
        assert review_status(project_config, sid)["cloud_llm_required"] is False
    script = """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'openai','anthropic','torch','faster_whisper','whisperx',
                              'transformers','huggingface_hub'}: raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import session_pipeline.human_review
import session_pipeline.review_merge
import session_pipeline.review_server
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_http_local_state_save_restart_security_and_audio_range(completed, project_config):
    sid, _, _, _ = completed
    build_bundle(project_config, sid)
    server = create_server(project_config, sid, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("GET", "/api/state")
        response = conn.getresponse()
        assert response.status == 200
        state = json.loads(response.read())
        conn.request("GET", "/audio/0", headers={"Range": "bytes=0-43"})
        response = conn.getresponse()
        assert response.status == 206 and response.getheader("Content-Range").startswith(
            "bytes 0-43/"
        )
        assert response.read()[:4] == b"RIFF"
        body = json.dumps(
            {
                "review_id": state["bundle"]["entries"][0]["review_id"],
                "choice": "USE_B",
                "expected_checksum": state["decisions"]["artifact_checksum"],
            }
        )
        conn.request("POST", "/api/decision", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        headers = {
            "Content-Type": "application/json",
            "X-Review-Token": state["save_token"],
            "Origin": f"http://127.0.0.1:{server.server_port}",
        }
        conn.request("POST", "/api/decision", body, headers)
        response = conn.getresponse()
        assert response.status == 200
        response.read()
        conn.request("POST", "/api/decision", body, headers)
        response = conn.getresponse()
        assert response.status == 409
        response.read()
        conn.request("GET", "/api/state", headers={"Host": "attacker.example"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.request("GET", "/audio/0", headers={"Range": "bytes=999999999-"})
        response = conn.getresponse()
        assert response.status == 416
        response.read()
        conn.request("GET", "/../../.env")
        response = conn.getresponse()
        assert response.status == 404
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    restarted = create_server(project_config, sid, port=0)
    restarted.server_close()
    assert review_status(project_config, sid)["decision_counts"]["USE_B"] == 1


def downgrade_to_v1(config, sid):
    context = load_context(config, sid)
    bundle = load_bundle(context)
    decisions = load_decisions(context, bundle)
    bundle.update(
        review_version=1,
        review_fingerprint=context_fingerprint(context, 1),
        entries=review_entries(context, legacy=True),
    )
    bundle = seal_artifact(bundle)
    decisions.pop("decision_version")
    decisions.update(
        bundle_checksum=bundle["artifact_checksum"],
        review_fingerprint=context_fingerprint(context, 1),
    )
    for item in decisions["decisions"].values():
        item.pop("merge_status")
    decisions = seal_artifact(decisions)
    atomic_write_json(context.directory / "bundle.json", bundle)
    atomic_write_json(context.directory / "decisions.json", decisions)
    return context, bundle, decisions


@pytest.mark.parametrize("choices", [("UNCLEAR", "UNCLEAR"), ("USE_B", "USE_A")])
def test_existing_human_decisions_survive_upgrade(completed, project_config, choices):
    sid, _, _, _ = completed
    build_bundle(project_config, sid)
    choose_all(project_config, sid, choices)
    context, old_bundle, old_decisions = downgrade_to_v1(project_config, sid)
    build_bundle(project_config, sid)
    bundle = load_bundle(context)
    decisions = load_decisions(context, bundle)
    assert bundle["review_version"] == decisions["decision_version"] == 2
    assert decisions["revision"] == old_decisions["revision"]
    assert decisions["updated_at"] == old_decisions["updated_at"]
    assert {
        k: {name: v[name] for name in ("choice", "updated_at")}
        for k, v in decisions["decisions"].items()
    } == old_decisions["decisions"]
    assert read_artifact(context.directory / "migration-v1/bundle.json") == old_bundle
    assert read_artifact(context.directory / "migration-v1/decisions.json") == old_decisions
    assert read_artifact(context.directory / "migration-v2.json")["status"] == "SUCCEEDED"
    build_bundle(project_config, sid)
    assert load_decisions(context, bundle) == decisions
    entry = bundle["entries"][0]
    reset = save_decision(
        project_config,
        sid,
        context.plan["stage_fingerprint"],
        entry["review_id"],
        "UNREVIEWED",
        decisions["artifact_checksum"],
    )
    assert reset["decisions"][entry["review_id"]]["choice"] == "UNREVIEWED"


@pytest.mark.parametrize("failure_point", ["decision_write", "completion_marker"])
def test_upgrade_recovers_interruption_without_losing_choices(
    completed, project_config, monkeypatch, failure_point
):
    import session_pipeline.human_review as review

    sid, _, _, _ = completed
    build_bundle(project_config, sid)
    choose_all(project_config, sid, ("UNCLEAR", "UNCLEAR"))
    context, _, old_decisions = downgrade_to_v1(project_config, sid)
    real_write = review.atomic_write_json

    def failing(path, document):
        if (
            failure_point == "decision_write"
            and path == context.directory / "decisions.json"
            and document.get("decision_version") == 2
        ):
            raise RuntimeError("synthetic interrupted decision commit")
        if (
            failure_point == "completion_marker"
            and path == context.directory / "migration-v2.json"
            and document.get("status") == "SUCCEEDED"
        ):
            raise RuntimeError("synthetic interrupted completion marker")
        return real_write(path, document)

    monkeypatch.setattr(review, "atomic_write_json", failing)
    with pytest.raises(RuntimeError, match="synthetic"):
        build_bundle(project_config, sid)
    monkeypatch.setattr(review, "atomic_write_json", real_write)
    if failure_point == "completion_marker":
        # A legitimate human save after pair commit must survive journal recovery.
        bundle = load_bundle(context)
        decisions = load_decisions(context, bundle)
        save_decision(
            project_config,
            sid,
            context.plan["stage_fingerprint"],
            bundle["entries"][0]["review_id"],
            "USE_A",
            decisions["artifact_checksum"],
        )
    report = build_bundle(project_config, sid)
    assert report["decision_counts"]["UNCLEAR"] == (2 if failure_point == "decision_write" else 1)
    backup = read_artifact(context.directory / "migration-v1/decisions.json")
    assert backup == old_decisions
    assert read_artifact(context.directory / "migration-v2.json")["status"] == "SUCCEEDED"


@pytest.mark.parametrize("state", ["SAFE", "AUTO_TRIMMED", "MANUAL_BOUNDARY_REQUIRED", "UNUSABLE"])
def test_reconciliation_states_and_merge(completed, project_config, state):
    sid, plan_path, raw_path, source = completed
    before = [p.read_bytes() for p in (raw_path, raw_path.with_suffix(".txt"), source)]
    path = plan_path.parent / "candidate-0000.json"
    candidate = read_artifact(path)
    if state == "SAFE":
        candidate["candidate_b"]["raw_segments"] = [seg(10, 12, "retry target")]
    elif state == "MANUAL_BOUNDARY_REQUIRED":
        candidate["candidate_b"]["raw_segments"][1]["words"][0]["start"] = 9.9
    elif state == "UNUSABLE":
        candidate["candidate_b"]["raw_segments"] = []
    candidate["candidate_b"]["segments"] = owned_segments(
        candidate["candidate_b"]["raw_segments"], candidate["window"]["target_interval"]
    )
    candidate["candidate_b_empty"] = not candidate["candidate_b"]["segments"]
    atomic_write_json(path, seal_artifact(candidate))
    eligibility = reconcile_candidate(candidate)
    assert eligibility["merge_status"] == state
    report = build_bundle(project_config, sid)
    context = load_context(project_config, sid)
    bundle = load_bundle(context)
    assert bundle["entries"][0]["b_selectable"] == (state != "UNUSABLE")
    if state == "UNUSABLE":
        assert report["b_unusable_count"] == 1
    else:
        choose_all(project_config, sid, ("USE_B", "USE_A"))
        if state == "MANUAL_BOUNDARY_REQUIRED":
            assert build_bundle(project_config, sid)["manual_boundary_selected_count"] == 1
            assert (
                load_decisions(context, bundle)["decisions"][bundle["entries"][0]["review_id"]][
                    "choice"
                ]
                == "USE_B"
            )
            with pytest.raises(ReviewError, match="window 1"):
                merge_review(project_config, sid)
            assert not (raw_path.parent / "reviewed").exists()
        else:
            result = merge_review(project_config, sid)
            text = Path(result["reviewed_txt"]).read_text(encoding="utf-8")
            assert "retry target" in text
            assert "PAD BEFORE" not in text and "PAD AFTER" not in text
            output = read_artifact(Path(result["reviewed_json"]))
            assert output["provenance"]["spans"][0]["merge_status"] == state
    assert before == [p.read_bytes() for p in (raw_path, raw_path.with_suffix(".txt"), source)]


def test_word_timing_ambiguity_never_auto_merges(completed, project_config):
    sid, _, _, _ = completed
    c = copy.deepcopy(load_context(project_config, sid).candidates[0])
    c["candidate_b"]["raw_segments"][1]["words"] = [
        {"start": 10, "end": 11.5, "word": "first"},
        {"start": 11, "end": 12, "word": "second"},
    ]
    assert reconcile_candidate(c)["merge_status"] == "MANUAL_BOUNDARY_REQUIRED"
    c["candidate_b"]["raw_segments"][1]["words"] = [{"start": 10, "end": 10, "word": "word"}]
    assert reconcile_candidate(c)["merge_status"] == "MANUAL_BOUNDARY_REQUIRED"
