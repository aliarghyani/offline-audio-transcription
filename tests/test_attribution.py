from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import wave

import pytest

from session_pipeline import attribution as a
from session_pipeline import speaker_review as review
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.final_render import plan_turns
from session_pipeline.models import sha256_file


def turn(start, end, speaker="SPEAKER_00"):
    return {"start": start, "end": end, "speaker": speaker, "duration": end - start}


def segment(index=0, text="one two", intervals=((1, 2), (2, 3)), *, failed=False):
    words = []
    cursor = 0
    for i, (word, interval) in enumerate(zip(text.split(), intervals, strict=True)):
        left = text.index(word, cursor)
        cursor = left + len(word)
        words.append(
            {
                "word": word,
                "start": interval[0],
                "end": interval[1],
                "character_interval": [left, cursor],
                "reviewed_word_index": i,
                "status": "UNALIGNED_WORD" if failed else "ALIGNED",
            }
        )
    return {
        "reviewed_segment_id": f"segment-{index}",
        "reviewed_segment_index": index,
        "text": text,
        "start": 1,
        "end": 3,
        "reviewed_interval": [1, 3],
        "status": "ALIGNMENT_FAILED" if failed else "ALIGNED",
        "words": words,
    }


def attribute(segments, regular, exclusive=None, uncovered=()):
    return a.attribute_segments(
        {"segments": segments},
        {
            "turns": regular,
            "exclusive_turns": exclusive,
            "qa": {"uncovered_reviewed_segment_ids": list(uncovered)},
        },
    )


def assigned(result):
    return [w["attribution"]["speaker"] for s in result for w in s["words"]]


def test_exclusive_preferred_and_regular_overlap_preserved():
    result, qa, diagnostic = attribute(
        [segment()], [turn(0, 5, "SPEAKER_01"), turn(1, 4)], [turn(0, 5)]
    )
    assert assigned(result) == ["SPEAKER_00"] * 2
    assert qa["overlap_affected_words"] == 2
    assert diagnostic["regular_overlap_regions"] == [[1, 4]]
    assert len(result[0]["words"][0]["attribution"]["regular_turn_overlaps"]) == 2


def test_largest_positive_turn_overlap_and_no_confidence():
    result, _, _ = attribute(
        [segment(text="one", intervals=((1, 3),))],
        [turn(0, 5)],
        [turn(1, 1.5), turn(1.5, 3, "SPEAKER_01")],
    )
    word = result[0]["words"][0]["attribution"]
    assert word["speaker"] == "SPEAKER_01"
    assert word["overlap_duration"] == 1.5
    assert word["overlap_fraction"] == 0.75
    assert "confidence" not in word


def test_regular_fallback_no_overlap_and_exact_tie():
    result, _, _ = attribute([segment()], [turn(1, 2)], [turn(10, 12)])
    assert assigned(result) == ["SPEAKER_00", None]
    assert result[0]["words"][0]["attribution"]["assignment_method"] == "REGULAR_MAX_TURN_OVERLAP"
    result, qa, _ = attribute(
        [segment(text="one", intervals=((1, 3),))],
        [turn(0, 5)],
        [turn(1, 2), turn(2, 3, "SPEAKER_01")],
    )
    assert assigned(result) == [None]  # Do not fall back after an exclusive tie.
    assert qa["ambiguous_words"] == 1
    assert result[0]["words"][0]["attribution"]["ambiguity_status"] == "EXACT_TIE"


def test_nested_regular_turn_index_does_not_miss_outer_overlap():
    result, _, _ = attribute(
        [segment(text="one", intervals=((8, 9),))], [turn(0, 10), turn(1, 2, "SPEAKER_01")]
    )
    assert assigned(result) == ["SPEAKER_00"]


def test_maximum_individual_turn_is_not_sum_of_disjoint_turns():
    result, _, _ = attribute(
        [segment(text="one", intervals=((1, 4),))],
        [turn(0, 5)],
        [turn(1, 1.8), turn(1.8, 3, "SPEAKER_01"), turn(3, 4)],
    )
    assert assigned(result) == ["SPEAKER_01"]


def test_cli_plan_execute_status_and_review_status_are_local(lineage, monkeypatch, capsys):
    from session_pipeline import cli

    config, sid, _, _ = lineage
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: config)
    assert cli.run(["attribute", sid, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "PLANNED"
    raw, _, _ = session_paths_for_test(config, sid)
    assert not (raw.parent / "attributed").exists()
    assert cli.run(["attribute", sid, "--execute", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total_segments"] == 1
    assert cli.run(["attribute", sid, "--status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total_words"] == 2
    assert cli.run(["speakers", "status", sid, "--json"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["identity_mapping_status"] == "HUMAN_MAPPING_REQUIRED"
    )


def test_states_and_former_uncovered_bounds_re_evaluated():
    s0, s1, s2, s3 = [segment(i) for i in range(4)]
    for word in s2["words"]:
        word["start"] += 10
        word["end"] += 10
    s3["words"][1].update(start=11, end=12)
    result, qa, _ = attribute(
        [s0, s1, s2, s3],
        [turn(1, 2), turn(2, 3, "SPEAKER_01")],
        uncovered=[s0["reviewed_segment_id"], s2["reviewed_segment_id"], s3["reviewed_segment_id"]],
    )
    assert [s["attribution"]["state"] for s in result] == [
        "MIXED_SPEAKER",
        "MIXED_SPEAKER",
        "UNASSIGNED",
        "PARTIAL",
    ]
    assert qa["former_uncovered_bounds"] == {
        "total": 3,
        "FULLY_ATTRIBUTABLE": 1,
        "PARTIALLY_ATTRIBUTABLE": 1,
        "UNASSIGNED": 1,
    }


def test_reviewed_failed_bounds_are_not_word_timestamps():
    s = segment(failed=True, intervals=((None, None), (None, None)))
    result, qa, _ = attribute([s], [turn(0, 4)], [turn(0, 4)])
    assert assigned(result) == ["SPEAKER_00"] * 2
    assert qa["fallback_timing_words"] == 2 and qa["total_timed_words"] == 0
    for word in result[0]["words"]:
        assert word["start"] is None and word["end"] is None
        assert word["attribution"]["timing_source"] == "REVIEWED_SEGMENT_FALLBACK"
    result, _, _ = attribute([s], [turn(0, 4)], [turn(0, 2.5), turn(2.5, 4, "SPEAKER_01")])
    assert assigned(result) == [None, None]  # Not majority-assigned.
    result, _, _ = attribute([s], [turn(0, 4), turn(1, 2, "SPEAKER_01")], [turn(0, 4)])
    assert assigned(result) == [None, None]
    s["status"] = "ALIGNED"
    result, qa, _ = attribute([s], [turn(0, 4)])
    assert assigned(result) == [None, None] and qa["fallback_timing_words"] == 0


def test_mixed_with_unassigned_word_is_still_mixed():
    s = segment(text="one two three", intervals=((1, 2), (2, 3), (4, 5)))
    result, _, _ = attribute([s], [turn(1, 2), turn(2, 3, "SPEAKER_01")])
    assert result[0]["attribution"]["state"] == "MIXED_SPEAKER"
    assert result[0]["attribution"]["is_partial"] is True


def test_gap_candidates_are_diagnostic_only():
    s = segment(text="one", intervals=((2, 3),))
    result, qa, _ = attribute([s], [turn(0, 1), turn(4, 5)], [turn(0, 1), turn(4, 5)])
    word = result[0]["words"][0]["attribution"]
    assert word["speaker"] is None
    diagnostic = word["fallback_diagnostics"]
    assert diagnostic["same_neighbor_speaker"] is True
    assert diagnostic["previous_distance_seconds"] == diagnostic["next_distance_seconds"] == 1
    assert diagnostic["gap_duration_seconds"] == 3
    assert diagnostic["assignment_applied"] is False
    assert qa["fallback_candidates"]["same_neighbor_speaker"] == 1


def test_overlap_ambiguity_does_not_report_neighbor_window_as_a_speech_gap():
    s = segment(failed=True, intervals=((None, None), (None, None)))
    result, _, _ = attribute(
        [s], [turn(0, 4), turn(1, 2, "SPEAKER_01")], [turn(0, 4), turn(4, 5, "SPEAKER_01")]
    )
    diagnostic = result[0]["words"][0]["attribution"]["fallback_diagnostics"]
    assert diagnostic["exclusive_overlap"] is True
    assert diagnostic["gap_status"] == "INSIDE_EXCLUSIVE_SPEECH"
    assert diagnostic["gap_duration_seconds"] is None


def test_all_2810_ids_text_and_original_word_fields_preserved():
    source = [segment(i, text="علی مشاور", intervals=((1, 2), (2, 3))) for i in range(2810)]
    before = copy.deepcopy(source)
    result, _, _ = attribute(source, [turn(0, 5)])
    assert source == before
    assert [s["reviewed_segment_id"] for s in result] == [s["reviewed_segment_id"] for s in before]
    for old, new in zip(before, result, strict=True):
        assert {k: v for k, v in new.items() if k != "attribution" and k != "words"} == {
            k: v for k, v in old.items() if k != "words"
        }
        for ow, nw in zip(old["words"], new["words"], strict=True):
            assert {k: v for k, v in nw.items() if k != "attribution"} == ow
    assert assigned(result) == ["SPEAKER_00"] * 5620  # Names in text have no effect.


@pytest.fixture
def lineage(project_config):
    source = project_config.application.sessions_path / "synthetic.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000 * 100)
    sha = sha256_file(source)
    sid = "s-" + sha[:20]
    afp, dfp = "a" * 64, "b" * 64
    raw, work, _ = session_paths_for_test(project_config, sid)
    apath = raw.parent / "aligned" / afp / "transcript.aligned.fa.json"
    dpath = raw.parent / "diarized" / dfp / "diarization.json"
    aligned = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "aligned_asr",
            "status": "SUCCEEDED",
            "configuration": {"mode": "FULL"},
            "session_id": sid,
            "stage_fingerprint": afp,
            "source_sha256": sha,
            "source": {
                "original_filename": source.name,
                "sha256": sha,
                "size_bytes": source.stat().st_size,
            },
            "probe": {"duration_seconds": 100},
            "qa": {"full_alignment_complete": True, "coverage_validation_pass": True},
            "segments": [segment()],
            "segment_count": 1,
        }
    )
    atomic_write_json(apath, aligned)
    plan = seal_artifact(
        {
            "schema_version": 1,
            "stage_fingerprint": dfp,
            "aligned_file_sha256": sha256_file(apath),
            "aligned_fingerprint": afp,
            "parent_checksums": {"aligned_asr": aligned["artifact_checksum"]},
        }
    )
    atomic_write_json(work / "diarization" / dfp / "plan.json", plan)
    turns = [
        turn(0, 15),
        turn(15, 30, "SPEAKER_01"),
        turn(35, 50),
        turn(50, 65, "SPEAKER_01"),
        turn(70, 85),
        turn(85, 100, "SPEAKER_01"),
    ]
    diarized = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "diarization",
            "status": "SUCCEEDED",
            "configuration": {"mode": "FULL"},
            "full_diarization_complete": True,
            "session_id": sid,
            "stage_fingerprint": dfp,
            "aligned_fingerprint": afp,
            "parent_checksums": plan["parent_checksums"],
            "plan_checksum": plan["artifact_checksum"],
            "source_sha256": sha,
            "source_duration_seconds": 100,
            "processed_interval": [0, 100],
            "qa": {"validation_pass": True, "uncovered_reviewed_segment_ids": []},
            "exclusive_qa": {"validation_pass": True},
            "turns": turns,
            "exclusive_turns": turns,
        }
    )
    atomic_write_json(dpath, diarized)
    return project_config, sid, apath, dpath


def session_paths_for_test(config, sid):
    from session_pipeline.asr_quality import session_paths

    return session_paths(config, sid)


def test_deterministic_lineage_upstream_unchanged_and_tamper_rejected(lineage):
    config, sid, apath, dpath = lineage
    before = [p.read_bytes() for p in (apath, dpath)]
    p1, d1 = a.make_document(config, sid)
    p2, d2 = a.make_document(config, sid)
    assert p1 == p2 and d1 == d2
    a.execute(config, sid)
    assert a.status(config, sid)["total_words"] == 2
    assert [p.read_bytes() for p in (apath, dpath)] == before
    assert a.execute(config, sid)["fingerprint"] == p1.parent.name
    damaged = json.loads(dpath.read_text("utf-8"))
    damaged["aligned_fingerprint"] = "c" * 64
    atomic_write_json(dpath, seal_artifact(damaged))
    with pytest.raises(ValueError):
        a.load_attributed(config, sid)
    assert p1.read_bytes()  # Existing output preserved after failure.


def test_parent_checksum_and_exclusive_overlap_rejected(lineage):
    config, sid, apath, dpath = lineage
    doc = read_artifact(apath)
    doc["segments"][0]["text"] = "tampered"
    atomic_write_json(apath, doc)
    with pytest.raises(ValueError, match="checksum"):
        a.make_document(config, sid)
    with pytest.raises(ValueError, match="overlapping"):
        a.validate_turns([turn(0, 3), turn(2, 4, "SPEAKER_01")], 5, exclusive=True)


def test_sample_selection_excludes_overlap_short_turns_and_spreads(lineage):
    config, sid, _, dpath = lineage
    d = read_artifact(dpath)
    regular = sorted(
        d["turns"] + [turn(4, 5, "SPEAKER_01"), turn(39, 39.05)],
        key=lambda t: (t["start"], t["end"], t["speaker"]),
    )
    evidence = {"regular_turns": regular, "exclusive_turns": d["exclusive_turns"]}
    samples = review.select_samples(evidence, 100)
    assert samples == review.select_samples(evidence, 100)
    assert len(samples) == 6
    for speaker in a.SPEAKERS:
        chosen = [s for s in samples if s["speaker"] == speaker]
        assert len(chosen) == 3
        assert len({int(s["start"] / (100 / 3)) for s in chosen}) == 3
    for s in samples:
        assert 6 <= s["duration_seconds"] <= 15
        assert not a.intersects([s["start"], s["end"]], [[4, 5], [39, 39.05]])
    with pytest.raises(ValueError):
        review.select_samples({"regular_turns": regular, "exclusive_turns": None}, 100)


@pytest.fixture
def bundle(lineage, monkeypatch):
    config, sid, _, _ = lineage
    a.execute(config, sid)

    def extract(source, path, interval):
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * round((interval[1] - interval[0]) * 16000))

    monkeypatch.setattr(review, "extract_clip", extract)
    report = review.build_bundle(config, sid)
    return config, sid, report["fingerprint"]


def test_human_bijection_no_default_atomic_save_conflict_and_history(bundle):
    config, sid, fp = bundle
    directory, manifest, _ = review.load_bundle(config, sid, fp)
    assert review.load_mapping(directory, manifest) is None
    assert review.status(config, sid)["identity_mapping_status"] == "HUMAN_MAPPING_REQUIRED"
    for mapping in (
        {},
        {"SPEAKER_00": "علی", "SPEAKER_01": "علی"},
        {"SPEAKER_00": "علی", "SPEAKER_01": "unknown"},
    ):
        with pytest.raises(ValueError):
            review.save_mapping(config, sid, fp, mapping, None, human_confirmed=True)
    mapping = {"SPEAKER_00": "علی", "SPEAKER_01": "مشاور"}
    with pytest.raises(ValueError, match="confirmation"):
        review.save_mapping(config, sid, fp, mapping, None, human_confirmed=False)
    result = review.save_mapping(config, sid, fp, mapping, None, human_confirmed=True)
    with pytest.raises(review.MappingConflict):
        review.save_mapping(config, sid, fp, mapping, None, human_confirmed=True)
    review.save_mapping(
        config,
        sid,
        fp,
        {"SPEAKER_00": "مشاور", "SPEAKER_01": "علی"},
        result["artifact_checksum"],
        human_confirmed=True,
    )
    assert review.load_mapping(directory, manifest)["revision"] == 2
    assert (
        review.mapping_path(directory).parent / "history" / (result["artifact_checksum"] + ".json")
    ).exists()
    assert review.build_bundle(config, sid)["identity_mapping_status"] == "HUMAN_CONFIRMED"


def test_review_server_loopback_state_range_csrf_and_explicit_choice(bundle):
    config, sid, fp = bundle
    server = review.create_server(config, sid, fp, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, body=None, headers=None):
        req = urllib.request.Request(base + path, data=body, headers=headers or {})
        return urllib.request.urlopen(req, timeout=10)

    try:
        with request("/api/state") as response:
            state = json.load(response)
        assert state["mapping"] is None and state["bundle"]["default_mapping"] is None
        with request("/") as response:
            html = response.read().decode()
            assert 'dir="rtl"' in html and 'type="checkbox"' in html
            assert "https://" not in html and "http://" not in html
        with request("/audio/SPEAKER_00-1", headers={"Range": "bytes=0-15"}) as response:
            assert response.status == 206 and len(response.read()) == 16
        with pytest.raises(urllib.error.HTTPError) as error:
            request("/api/state", headers={"Host": "evil.example"})
        assert error.value.code == 403
        payload = json.dumps(
            {
                "mapping": {"SPEAKER_00": "علی", "SPEAKER_01": "مشاور"},
                "expected_checksum": None,
                "human_confirmed": True,
            }
        ).encode()
        with pytest.raises(urllib.error.HTTPError) as error:
            request("/api/mapping", payload, {"Content-Type": "application/json"})
        assert error.value.code == 403
        with request(
            "/api/mapping",
            payload,
            {
                "Content-Type": "application/json",
                "Origin": base,
                "X-Review-Token": state["save_token"],
            },
        ) as response:
            assert json.load(response)["mapping"]["human_confirmed"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_status_review_imports_and_network_independent(bundle, monkeypatch):
    config, sid, fp = bundle
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("no network or model loading permitted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(review, "extract_clip", forbidden)
    assert a.status(config, sid)["unassigned_words"] == 0
    assert review.status(config, sid)["sample_count"] == 6
    server = review.create_server(config, sid, fp, 0)
    server.server_close()
    code = """
import sys, socket
def forbidden(*args, **kwargs):
    raise AssertionError('network forbidden')
socket.socket.connect = forbidden
from session_pipeline import attribution, speaker_review, final_render
heavy = {'torch', 'whisperx', 'pyannote', 'transformers'}
assert not any(n.split('.')[0] in heavy for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_renderer_requires_mapping_and_preserves_mixed_text_and_neutral_label():
    s = segment(text="one two three", intervals=((1, 2), (2, 3), (4, 5)))
    result, _, _ = attribute([s], [turn(1, 2), turn(2, 3, "SPEAKER_01")])
    doc = {
        "segments": result,
        "artifact_checksum": "a" * 64,
        "diarization_evidence": {
            "regular_turns": [turn(1, 2), turn(2, 3, "SPEAKER_01")],
            "exclusive_turns": None,
        },
    }
    with pytest.raises(ValueError, match="human identity"):
        plan_turns(doc, None)
    mapping = {
        "human_confirmed": True,
        "identity_inference_used": False,
        "attribution_checksum": "a" * 64,
        "mapping": {"SPEAKER_00": "علی", "SPEAKER_01": "مشاور"},
    }
    turns = plan_turns(doc, mapping)
    assert [t["label"] for t in turns] == ["علی", "مشاور", "نامشخص"]
    assert "".join(p["text"] for t in turns for p in t["pieces"]) == s["text"]
    assert [p["word_indices"][0] for t in turns for p in t["pieces"]] == [0, 1, 2]


def test_renderer_does_not_group_across_intervening_detected_speaker():
    s = segment(intervals=((1, 2), (4, 5)))
    timeline = [turn(1, 2), turn(2, 4, "SPEAKER_01"), turn(4, 5)]
    result, _, _ = attribute([s], timeline, timeline)
    doc = {
        "segments": result,
        "artifact_checksum": "a" * 64,
        "diarization_evidence": {"regular_turns": timeline, "exclusive_turns": timeline},
    }
    mapping = {
        "human_confirmed": True,
        "identity_inference_used": False,
        "attribution_checksum": "a" * 64,
        "mapping": {"SPEAKER_00": "علی", "SPEAKER_01": "مشاور"},
    }
    turns = plan_turns(doc, mapping)
    assert len(turns) == 2 and turns[0]["speaker"] == turns[1]["speaker"] == "SPEAKER_00"
    assert "".join(p["text"] for t in turns for p in t["pieces"]) == s["text"]


def test_fingerprint_changes_with_policy(lineage, monkeypatch):
    config, sid, _, _ = lineage
    first, _ = a.make_document(config, sid)
    monkeypatch.setitem(a.POLICY, "version", 2)
    second, _ = a.make_document(config, sid)
    assert first != second
