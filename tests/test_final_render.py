from __future__ import annotations

import copy
import json
import socket
import subprocess
import sys

import pytest
from test_attribution import attribute, segment, turn
from test_attribution import bundle as bundle
from test_attribution import lineage as lineage

from session_pipeline import final_render as final
from session_pipeline import speaker_review
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.models import sha256_file


def inputs(segments, timeline, regular=None):
    result, qa, _ = attribute(segments, regular or timeline, timeline)
    document = {
        "segments": result,
        "artifact_checksum": "a" * 64,
        "qa": qa,
        "diarization_evidence": {"regular_turns": regular or timeline, "exclusive_turns": timeline},
    }
    mapping = {
        "human_confirmed": True,
        "identity_inference_used": False,
        "attribution_checksum": document["artifact_checksum"],
        "mapping": {"SPEAKER_00": "مشاور", "SPEAKER_01": "علی"},
    }
    return document, mapping


def check(document, turns, duration=100):
    qa = final.validate_coverage(document["segments"], turns, duration)
    text, spans = final.render_txt(turns)
    assert final.validate_txt(document["segments"], turns, text, spans)
    assert qa["source_text_sha256"] == qa["reconstructed_text_sha256"]
    return text


def test_only_human_mapping_and_neutral_unknown_labels():
    document, mapping = inputs(
        [segment(text="علی مشاور سه", intervals=((1, 2), (2, 3), (5, 6)))],
        [turn(1, 2), turn(2, 3, "SPEAKER_01")],
    )
    with pytest.raises(ValueError):
        final.plan_turns(document, None)
    turns = final.plan_turns(document, mapping)
    assert [t["label"] for t in turns] == ["مشاور", "علی", "نامشخص"]
    assert [t["speaker"] for t in turns] == ["SPEAKER_00", "SPEAKER_01", None]
    check(document, turns)


def test_exact_mixed_slices_preserve_punctuation_whitespace_and_unicode():
    original = "  يک،\tدو\nسه؟  "
    document, mapping = inputs(
        [segment(text=original, intervals=((1, 2), (2, 3), (3, 4)))],
        [turn(0, 2), turn(2, 4, "SPEAKER_01")],
    )
    turns = final.plan_turns(document, mapping)
    assert len(turns) == 2
    assert "".join(p["text"] for t in turns for p in t["pieces"]) == original
    assert document["segments"][0]["attribution"]["state"] == "MIXED_SPEAKER"
    check(document, turns)


@pytest.mark.parametrize("broken", [None, [0, 99], [0, 2]])
def test_no_unsafe_word_reconstruction_when_offsets_unproven(broken):
    document, mapping = inputs([segment()], [turn(1, 2), turn(2, 3, "SPEAKER_01")])
    document["segments"][0]["words"][0]["character_interval"] = broken
    turns = final.plan_turns(document, mapping)
    assert len(turns) == 1
    assert turns[0]["label"] == "مشاور / علی"
    piece = turns[0]["pieces"][0]
    assert piece["text"] == document["segments"][0]["text"]
    assert piece["source_status"] == "MIXED_SPEAKER"
    assert piece["render_strategy"] == "EXACT_PARENT_BLOCK"
    check(document, turns)


def test_partial_parent_block_preserves_unknown_content():
    document, mapping = inputs([segment(intervals=((1, 2), (5, 6)))], [turn(1, 2)])
    document["segments"][0]["words"][0].pop("character_interval")
    turns = final.plan_turns(document, mapping)
    assert turns[0]["label"] == "مشاور / نامشخص"
    assert turns[0]["pieces"][0]["source_status"] == "PARTIAL"
    assert turns[0]["pieces"][0]["word_indices"] == [0, 1]
    check(document, turns)


def test_failed_alignment_unknown_run_retains_reviewed_timing():
    s = segment(failed=True, intervals=((None, None), (None, None)))
    document, mapping = inputs([s], [turn(0, 2), turn(2, 5, "SPEAKER_01")])
    turns = final.plan_turns(document, mapping)
    assert len(turns) == 1 and turns[0]["label"] == "نامشخص"
    assert turns[0]["start"] == 1 and turns[0]["end"] == 3
    assert turns[0]["pieces"][0]["timing_source"] == "REVIEWED_SEGMENT_FALLBACK"
    check(document, turns)


def test_no_dropped_duplicated_or_changed_text_and_word_indices():
    document, mapping = inputs([segment()], [turn(0, 5)])
    turns = final.plan_turns(document, mapping)
    for corruption in ("text", "indices", "missing", "duplicate"):
        bad = copy.deepcopy(turns)
        if corruption == "text":
            bad[0]["pieces"][0]["text"] += "!"
        elif corruption == "indices":
            bad[0]["pieces"][0]["word_indices"] = [0, 0]
        elif corruption == "missing":
            bad = []
        else:
            bad[0]["pieces"] *= 2
        with pytest.raises(ValueError):
            final.validate_coverage(document["segments"], bad, 100)
    text, spans = final.render_txt(turns)
    with pytest.raises(ValueError):
        final.validate_txt(document["segments"], turns, text + "extra", spans)


def test_all_2810_segment_ids_preserved_without_mutating_input():
    document, mapping = inputs([segment(i) for i in range(2810)], [turn(0, 100)])
    # Give each source segment valid ordered times.
    for i, s in enumerate(document["segments"]):
        s.update(start=i * 3, end=i * 3 + 2, reviewed_interval=[i * 3, i * 3 + 2])
        for j, word in enumerate(s["words"]):
            word.update(start=i * 3 + j, end=i * 3 + j + 1)
            word["attribution"]["attribution_interval"] = [word["start"], word["end"]]
    before = copy.deepcopy(document)
    turns = final.plan_turns(document, mapping, 10000)
    assert (
        final.validate_coverage(document["segments"], turns, 10000)["represented_segments"] == 2810
    )
    assert document == before
    assert {p["segment_id"] for t in turns for p in t["pieces"]} == {
        s["reviewed_segment_id"] for s in document["segments"]
    }


def test_conservative_grouping_gap_threshold_and_detected_change():
    s0 = segment(0, text="one", intervals=((1, 2),))
    s1 = segment(1, text="two", intervals=((2.5, 3),))
    s1.update(start=2.5, end=3, reviewed_interval=[2.5, 3])
    document, mapping = inputs([s0, s1], [turn(0, 10)])
    assert len(final.plan_turns(document, mapping)) == 1
    document["segments"][1]["words"][0]["attribution"]["attribution_interval"] = [3, 4]
    assert len(final.plan_turns(document, mapping)) == 2
    document, mapping = inputs([s0, s1], [turn(0, 2), turn(2, 2.5, "SPEAKER_01"), turn(2.5, 10)])
    assert len(final.plan_turns(document, mapping)) == 2


def test_no_grouping_across_regular_overlap_or_mixed_boundary():
    s0 = segment(0)
    s1 = segment(1, text="three", intervals=((3.1, 4),))
    s1.update(start=3.1, end=4, reviewed_interval=[3.1, 4])
    timeline = [turn(0, 2), turn(2, 3, "SPEAKER_01"), turn(3, 10)]
    document, mapping = inputs([s0, s1], timeline)
    assert len(final.plan_turns(document, mapping)) == 3
    document, mapping = inputs(
        [segment()], [turn(0, 10)], [turn(0, 10), turn(1.5, 2.5, "SPEAKER_01")]
    )
    assert len(final.plan_turns(document, mapping)) == 2


def test_invalid_timestamp_bounds_and_regression_fail():
    document, mapping = inputs([segment()], [turn(0, 10)])
    document["segments"][0]["words"][0]["attribution"]["attribution_interval"] = [-1, 2]
    with pytest.raises(ValueError, match="time bounds"):
        final.plan_turns(document, mapping, 10)
    document, mapping = inputs([segment()], [turn(1, 2), turn(2, 3, "SPEAKER_01")])
    turns = final.plan_turns(document, mapping)
    with pytest.raises(ValueError, match="timestamp"):
        final.validate_coverage(document["segments"], list(reversed(turns)), 10)


@pytest.fixture
def confirmed(bundle):
    config, sid, fp = bundle
    speaker_review.save_mapping(
        config, sid, fp, {"SPEAKER_00": "مشاور", "SPEAKER_01": "علی"}, None, human_confirmed=True
    )
    return config, sid, fp


def test_persisted_mapping_missing_corrupt_unconfirmed_or_duplicate_refuses(bundle):
    config, sid, fp = bundle
    directory, manifest, _ = speaker_review.load_bundle(config, sid, fp)
    path = speaker_review.mapping_path(directory)
    with pytest.raises(ValueError, match="persisted"):
        final.execute(config, sid, fp)
    valid = speaker_review.save_mapping(
        config, sid, fp, {"SPEAKER_00": "مشاور", "SPEAKER_01": "علی"}, None, human_confirmed=True
    )
    for mode in ("unconfirmed", "duplicate", "corrupt"):
        invalid = copy.deepcopy(valid)
        if mode == "unconfirmed":
            invalid["human_confirmed"] = False
        else:
            invalid["mapping"]["SPEAKER_01"] = "مشاور"
        atomic_write_json(path, invalid if mode == "corrupt" else seal_artifact(invalid))
        with pytest.raises(ValueError):
            final.execute(config, sid, fp)
    assert manifest["attribution_fingerprint"] == fp


def test_deterministic_output_idempotence_and_upstream_immutability(confirmed):
    config, sid, fp = confirmed
    before = {
        p: sha256_file(p)
        for directory in (config.application.root / "outputs", config.application.root / "work")
        for p in directory.rglob("*.json")
    }
    path, document, text = final.build_document(config, sid, fp)
    assert final.build_document(config, sid, fp) == (path, document, text)
    result = final.execute(config, sid, fp)
    assert result["consultant_words"] == 2 and result["ali_words"] == result["unknown_words"] == 0
    contents = path.read_bytes(), path.with_suffix(".txt").read_bytes()
    assert final.execute(config, sid, fp) == result
    assert final.status(config, sid, path.parent.name) == result
    assert contents == (path.read_bytes(), path.with_suffix(".txt").read_bytes())
    assert all(sha256_file(p) == sha for p, sha in before.items())
    assert read_artifact(path)["mapping_source"] == "HUMAN_CONFIRMED"


def test_final_corruption_is_not_overwritten_and_status_is_read_only(confirmed):
    config, sid, fp = confirmed
    result = final.execute(config, sid, fp)
    from pathlib import Path

    path = Path(result["txt_path"])
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        final.status(config, sid, result["final_fingerprint"])
    with pytest.raises(ValueError):
        final.execute(config, sid, fp)
    assert path.read_text("utf-8") == "changed"


def test_final_cli_and_status_no_models_or_network(confirmed, monkeypatch, capsys):
    from session_pipeline import cli

    config, sid, fp = confirmed
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: config)

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert cli.run(["final", "render", sid, "--attributed-fingerprint", fp, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (
        cli.run(["final", "status", sid, "--fingerprint", report["final_fingerprint"], "--json"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["txt_text_preservation_pass"] is True
    code = """
import sys, socket
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('network'))
from session_pipeline import final_render
heavy = {'torch', 'whisperx', 'pyannote', 'transformers'}
assert not any(n.split('.')[0] in heavy for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_changed_human_confirmation_changes_fingerprint(confirmed):
    config, sid, fp = confirmed
    p1, _, _ = final.build_document(config, sid, fp)
    directory, bundle, _ = speaker_review.load_bundle(config, sid, fp)
    old = speaker_review.load_mapping(directory, bundle)
    speaker_review.save_mapping(
        config,
        sid,
        fp,
        {"SPEAKER_00": "علی", "SPEAKER_01": "مشاور"},
        old["artifact_checksum"],
        human_confirmed=True,
    )
    p2, _, _ = final.build_document(config, sid, fp)
    assert p1 != p2
