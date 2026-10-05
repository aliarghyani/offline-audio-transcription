from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest
from test_human_review import completed as completed_fixture
from test_human_review import seg

from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_retry import deny_network
from session_pipeline.human_review import build_bundle, load_context
from session_pipeline.models import sha256_file
from session_pipeline.retry_recovery import auto_merge, build_recovery, evaluate, recover, splice
from session_pipeline.review_server import create_server


@pytest.fixture
def completed(project_config):
    return completed_fixture.__wrapped__(project_config)


def candidate(words, target=(10, 10.1), old=None):
    return {
        "artifact_checksum": "test",
        "window": {"window_index": 0, "target_interval": list(target), "decoded_interval": [8, 14]},
        "candidate_b": {"raw_segments": [seg(*w) for w in words], "segments": old or []},
    }


def test_midpoint_outside_empty_b_recovers_whole_word_without_context_duplicates():
    c = candidate([(8, 9, "PAD BEFORE"), (9.8, 10.12, " recovered"), (11, 12, "PAD AFTER")])
    assert (9.8 + 10.12) / 2 < 10
    originals = [
        seg(8, 9, "neighbor before"),
        seg(10, 10.1, "wrong"),
        seg(11, 12, "neighbor after"),
    ]
    e = recover(c, originals)
    assert e["classification"] == "RECOVERED_FROM_CONTEXT"
    assert e["effective_interval"] == [9.8, 10.12]
    assert e["retry_word_ids"] == ["1:0"]
    result = splice(originals, [e])
    assert [s["text"] for s in result] == ["neighbor before", "recovered", "neighbor after"]
    assert result[1]["words"][0]["start"] == 9.8
    assert result[1]["words"][0]["end"] == 10.12


def test_multiple_overlapping_retry_words_and_both_crossing_edges():
    c = candidate([(9.9, 10.04, " first"), (10.02, 10.2, " last")])
    e = recover(c, [seg(10, 10.1, "old")])
    assert e["classification"] == "RECOVERED_FROM_CONTEXT"
    assert e["effective_interval"] == [9.9, 10.2]
    assert len(e["retry_word_ids"]) == 2
    assert [s["text"] for s in splice([seg(10, 10.1, "old")], [e])] == ["first", "last"]


def test_partial_original_segment_preserves_both_neighboring_word_runs():
    original = seg(9, 12, " before wrong after")
    original["words"] = [
        {"start": a, "end": b, "word": t}
        for a, b, t in [(9, 9.8, " before"), (10, 10.1, " wrong"), (11, 12, " after")]
    ]
    e = recover(candidate([(9.9, 10.2, " correct")]), [original])
    result = splice([original], [e])
    assert [s["text"] for s in result] == ["before", "correct", "after"]
    assert [s["review_origin"]["candidate"] for s in result] == ["A", "B", "A"]


def test_joint_boundary_closure_includes_only_words_in_shared_effective_span():
    c = candidate([(8, 9, "context"), (9.9, 10.2, "one"), (10.2, 10.5, "two"), (11, 12, "context")])
    e = recover(c, [seg(10, 10.4, "original whole word")])
    assert e["effective_interval"] == [9.9, 10.5]
    assert e["retry_word_ids"] == ["1:0", "2:0"]


def test_no_positive_overlap_including_endpoint_touch_and_zero_duration():
    e = recover(candidate([(9, 10, "before"), (10.1, 11, "after"), (10.05, 10.05, "zero")]), [])
    # Out-of-order metadata is corruption; chronological endpoint-only evidence is no overlap.
    assert e["classification"] == "CORRUPT"
    e = recover(candidate([(9, 10, "before"), (10.05, 10.05, "zero"), (10.1, 11, "after")]), [])
    assert e["classification"] == "NO_TARGET_OVERLAP"


@pytest.mark.parametrize("start,end", [(None, 10.2), (10.2, 9.9), (float("nan"), 10.2)])
def test_invalid_timestamps_are_corrupt(start, end):
    c = candidate([(9.9, 10.2, "retry")])
    c["candidate_b"]["raw_segments"][0]["words"][0].update(start=start, end=end)
    assert recover(c, [seg(10, 10.1, "old")])["classification"] == "CORRUPT"


def test_missing_word_timestamps_cannot_be_guessed():
    c = candidate([(9.9, 10.2, "retry")])
    del c["candidate_b"]["raw_segments"][0]["words"]
    assert recover(c, [seg(10, 10.1, "old")])["classification"] == "CORRUPT"


def test_conflicting_expanded_windows_preserve_a(completed, project_config):
    sid, _, _, _ = completed
    context = load_context(project_config, sid)
    context.candidates = [
        candidate([(9.9, 10.2, "left")]),
        candidate([(10.15, 10.5, "right")], target=(10.3, 10.4)),
    ]
    context.candidates[1]["artifact_checksum"] = "second"
    context.candidates[1]["window"]["window_index"] = 1
    context.raw["segments"] = [seg(10, 10.1, "A1"), seg(10.3, 10.4, "A2")]
    entries, counts = evaluate(context)
    assert counts["CORRUPT"] == 2
    assert [s["text"] for s in splice(context.raw["segments"], entries)] == ["A1", "A2"]


def test_auto_policy_idempotence_raw_immutability_no_network_and_ui(completed, project_config):
    sid, _, raw, source = completed
    build_bundle(project_config, sid)
    context = load_context(project_config, sid)
    human_bytes = (context.directory / "decisions.json").read_bytes()
    baseline = {
        p: (sha256_file(p), p.stat().st_mtime_ns) for p in [raw, raw.with_suffix(".txt"), source]
    }
    with deny_network():
        counts = build_recovery(project_config, sid)
        first = auto_merge(project_config, sid)
        second = auto_merge(project_config, sid)
    assert counts["DIRECT_TARGET"] == 2
    assert first == second and first["USE_B"] == 2 and first["USE_A"] == 0
    assert (context.directory / "decisions.json").read_bytes() == human_bytes
    assert (context.directory / "before-auto-prefer-b/decisions.json").read_bytes() == human_bytes
    for p, expected in baseline.items():
        assert (sha256_file(p), p.stat().st_mtime_ns) == expected
    merged = read_artifact(Path(first["reviewed_json"]))
    assert merged["qa_acceptance"]["alignment_ready"] is True
    assert merged["provenance"]["human_verified"] is False
    assert [s["text"] for s in merged["segments"]] == [
        "before",
        "retry target",
        "between",
        "retry target",
        "after",
    ]
    server = create_server(project_config, sid, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/state")
        response = connection.getresponse()
        state = json.loads(response.read())
        assert response.status == 200
        assert state["bundle"]["read_only"] is True
        assert state["bundle"]["review_mode"] == "AUTO_PREFER_B"
        assert all(d["choice"] == "USE_B" for d in state["decisions"]["decisions"].values())
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_corrupt_checkpoint_falls_back_without_overwriting_it(completed, project_config):
    sid, plan, _, _ = completed
    path = plan.parent / "candidate-0000.json"
    path.write_bytes(b"corrupt checkpoint")
    build_recovery(project_config, sid)
    result = auto_merge(project_config, sid)
    assert result["counts"]["CORRUPT"] == 1
    assert result["USE_A"] == result["USE_B"] == 1
    assert path.read_bytes() == b"corrupt checkpoint"


def test_recovered_context_is_shown_in_ui_state(completed, project_config):
    sid, plan, _, _ = completed
    path = plan.parent / "candidate-0000.json"
    c = read_artifact(path)
    target = c["window"]["target_interval"]
    c["candidate_b"]["raw_segments"] = [seg(target[0] - 4, target[0] + 0.1, "recovered")]
    c["candidate_b"]["segments"] = []
    c["candidate_b_empty"] = True
    atomic_write_json(path, seal_artifact(c))
    build_bundle(project_config, sid)
    build_recovery(project_config, sid)
    from session_pipeline.human_review import load_bundle, load_decisions
    from session_pipeline.retry_recovery import recovery_ui_state

    context = load_context(project_config, sid)
    bundle = load_bundle(context)
    bundle, _ = recovery_ui_state(context, bundle, load_decisions(context, bundle))
    entry = bundle["entries"][0]
    assert entry["candidate_source"] == "RECOVERED_FROM_CONTEXT"
    assert entry["text_b"] == "recovered" and entry["b_selectable"]


def test_expansion_outside_evidence_is_corrupt():
    c = candidate([(9.9, 10.2, "retry")])
    e = recover(c, [seg(7, 10.1, "long original")])
    assert e["classification"] == "CORRUPT"


def test_new_boundary_duplicate_falls_back_but_preexisting_repeat_is_retained(
    completed, project_config
):
    sid, _, _, _ = completed
    context = load_context(project_config, sid)
    context.candidates = [candidate([(10, 10.1, "repeat")])]
    context.raw["segments"] = [seg(9, 9.9, "repeat"), seg(10, 10.1, "different")]
    entries, counts = evaluate(context)
    assert counts["CORRUPT"] == 1
    assert [s["text"] for s in splice(context.raw["segments"], entries)] == ["repeat", "different"]
    context.raw["segments"][1] = seg(10, 10.1, "repeat")
    _, counts = evaluate(context)
    assert counts["RECOVERED_FROM_CONTEXT"] == 1
