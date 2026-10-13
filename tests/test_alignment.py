from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_human_review import seg, wav

from session_pipeline import alignment as a
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_retry import deny_network
from session_pipeline.cli import run
from session_pipeline.models import session_id_from_sha256, sha256_file


@pytest.fixture
def reviewed(project_config):
    root = project_config.application.root
    origin = Path(__file__).resolve().parents[1]
    (root / "config/models.lock.json").write_bytes(
        (origin / "config/models.lock.json").read_bytes()
    )
    source = root / "sessions/synthetic.wav"
    wav(source, 620)
    sha = sha256_file(source)
    sid = session_id_from_sha256(sha)
    output = root / "outputs" / sid
    path = output / "reviewed" / ("a" * 64) / "transcript.reviewed.fa.json"
    path.parent.mkdir(parents=True)
    text = "authoritative synthetic reviewed fixture\n"
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    segments = [
        seg(0, 2, "اول دوم"),
        seg(100, 102, "سه چهار"),
        seg(299, 302, "پنج شش"),
        seg(310, 313, "هفت هشت"),
        seg(610, 612, "نه ده"),
        seg(618, 620, "پایان"),
    ]
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "reviewed_asr",
            "status": "SUCCEEDED",
            "session_id": sid,
            "source_sha256": sha,
            "source": {"sha256": sha},
            "stage_fingerprint": "a" * 64,
            "probe": {"duration_seconds": 620},
            "segments": segments,
            "segment_count": len(segments),
            "qa_acceptance": {"alignment_ready": True},
            "unresolved": [],
            "rendered_txt_sha256": sha256_file(path.with_suffix(".txt")),
        }
    )
    atomic_write_json(path, document)
    atomic_write_json(
        output / "reviewed/latest.json",
        seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "reviewed_asr_pointer",
                "session_id": sid,
                "reviewed_json": path.relative_to(output).as_posix(),
                "artifact_checksum_of_reviewed_json": document["artifact_checksum"],
            }
        ),
    )
    atomic_write_json(
        root / "work" / sid / "integrity.json",
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
    for suffix in ("json", "txt"):
        (output / f"transcript.raw.fa.{suffix}").write_bytes(b"RAW MUST NOT BE PARSED OR WRITTEN")
    return sid, path, source


def fake_engine(originals, unit, clip):
    assert clip.name.endswith(".wav")
    offset = unit["decoded_interval"][0]
    returned = []
    for s in originals:
        words = a.tokens(s["text"])
        width = (s["end"] - s["start"]) / len(words)
        returned.append(
            {
                "words": [
                    {
                        "word": text,
                        "start": s["start"] + i * width - offset,
                        "end": s["start"] + (i + 1) * width - offset,
                        "score": 0.75,
                    }
                    for i, (text, _, _) in enumerate(words)
                ]
            }
        )
    return returned


def fake_extract(source, clip, interval):
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"synthetic PCM substitute for injected engine")


def execute(config, sid, **kwargs):
    return a.run_alignment(
        config, sid, engine_factory=lambda _: fake_engine, extractor=fake_extract, **kwargs
    )


def test_deterministic_plan_exact_ownership_and_no_model_load(
    reviewed, project_config, monkeypatch
):
    sid, path, _ = reviewed
    monkeypatch.setattr(a, "LocalAligner", Mock(side_effect=AssertionError("model load")))
    with deny_network():
        p1, plan = a.make_plan(project_config, sid)
        p2, same = a.make_plan(project_config, sid)
    assert p1 == p2 and plan == same
    assert plan["parent_checksums"]["reviewed_asr"] == read_artifact(path)["artifact_checksum"]
    assert plan["reviewed_relative_path"].endswith("transcript.reviewed.fa.json")
    assert len(plan["units"]) == 3
    assert plan["planned_audio_seconds"] == 622
    assert plan["owned_audio_seconds"] == 620
    assert [n for u in plan["units"] for n in u["segment_indices"]] == list(range(6))
    assert len({u["unit_id"] for u in plan["units"]}) == 3
    assert plan["units"][0]["decoded_interval"] == [0, 302]
    assert plan["units"][1]["segment_indices"] == [3]
    assert plan["units"][-1]["ownership_interval"] == [600, 620]


def test_full_resume_idempotence_absolute_timestamps_and_immutability(reviewed, project_config):
    sid, path, source = reviewed
    raw = path.parents[2] / "transcript.raw.fa.json"
    files = [path, path.with_suffix(".txt"), source, raw, raw.with_suffix(".txt")]
    baseline = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in files}
    report = execute(project_config, sid)
    assert report["full_alignment_complete"] and report["completed_units"] == 3
    out = Path(report["aligned_json"])
    old = out.read_bytes()
    doc = read_artifact(out)
    assert doc["segments"][3]["words"][0]["start"] == 310
    assert doc["segments"][3]["text"] == "هفت هشت"
    assert doc["segments"][-1]["words"][-1]["end"] == 620
    require_no_load = Mock(side_effect=AssertionError("completed run must not load model"))
    second = a.run_alignment(
        project_config,
        sid,
        engine_factory=require_no_load,
        extractor=Mock(side_effect=AssertionError("decode")),
    )
    assert second["full_alignment_complete"] and out.read_bytes() == old
    assert all((sha256_file(p), p.stat().st_mtime_ns) == value for p, value in baseline.items())


def test_failed_unit_retry_skips_completed_units_and_keeps_sanitized_errors(
    reviewed, project_config
):
    sid, _, _ = reviewed

    def failing(originals, unit, clip):
        if unit["index"] == 1:
            raise RuntimeError("private transcript text must never reach failure artifacts")
        return fake_engine(originals, unit, clip)

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: failing, extractor=fake_extract
    )
    assert report["completed_units"] == 2 and report["failed_units"] == 1
    assert report["aligned_json"] is None and not report["full_alignment_complete"]
    p, plan = a.make_plan(project_config, sid)
    checkpoints = [a.checkpoint_path(p, u) for u in plan["units"]]
    original = [x.read_bytes() for x in checkpoints]
    assert "private transcript text" not in checkpoints[1].read_text(encoding="utf-8")
    calls = []

    def retry(originals, unit, clip):
        calls.append(unit["index"])
        return fake_engine(originals, unit, clip)

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: retry, extractor=fake_extract
    )
    assert calls == [1] and report["full_alignment_complete"]
    assert checkpoints[0].read_bytes() == original[0] and checkpoints[2].read_bytes() == original[2]
    assert read_artifact(checkpoints[1])["attempt"] == 2


def test_interrupted_running_unit_is_resumed(reviewed, project_config):
    sid, _, _ = reviewed
    p, plan = a.make_plan(project_config, sid)

    def interrupted(originals, unit, clip):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        a.run_alignment(
            project_config, sid, engine_factory=lambda _: interrupted, extractor=fake_extract
        )
    cp = read_artifact(a.checkpoint_path(p, plan["units"][0]))
    assert cp["status"] == "INTERRUPTED"
    assert a.status_report(project_config, sid)["unit_status_counts"]["INTERRUPTED"] == 1
    assert execute(project_config, sid)["full_alignment_complete"]


def test_smoke_separate_lineage_never_full_and_bounded(reviewed, project_config):
    sid, _, _ = reviewed
    report = execute(project_config, sid, smoke=True)
    assert report["mode_processing_complete"] and not report["full_alignment_complete"]
    assert report["planned_audio_seconds"] == 180 and report["completed_segment_count"] == 2
    assert "/smoke/" in Path(report["aligned_json"]).as_posix()
    assert read_artifact(Path(report["aligned_json"]))["qa"]["full_alignment_complete"] is False
    status = a.status_report(project_config, sid)
    assert status["status"] == "NOT_PLANNED" and not status["full_alignment_complete"]


def test_partial_alignment_preserves_unaligned_tokens_and_score_meaning(reviewed, project_config):
    sid, _, _ = reviewed

    def partial(originals, unit, clip):
        result = fake_engine(originals, unit, clip)
        result[0]["words"][-1] = {"word": result[0]["words"][-1]["word"]}
        return result

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: partial, extractor=fake_extract
    )
    # Last unit has a second fully aligned segment, so a partial last word is retained safely.
    assert report["unaligned_word_count"] == 3 and report["qa_counts"]["PARTIAL_ALIGNMENT"] == 3
    assert report["full_alignment_complete"]
    assert report["partial_units"] == 3 and report["succeeded_units"] == 0
    doc = read_artifact(Path(report["aligned_json"]))
    w = doc["segments"][0]["words"][-1]
    assert w["word"] == "دوم" and w["start"] is None and w["status"] == "UNALIGNED_WORD"


def test_model_revision_is_exact_and_never_resolves_main(reviewed, project_config):
    sid, _, _ = reviewed
    lock = project_config.application.root / "config/models.lock.json"
    data = json.loads(lock.read_text())
    next(m for m in data["models"] if m["role"] == "alignment")["revision"] = "0" * 40
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        a.make_plan(project_config, sid)


@pytest.mark.parametrize("change", ["json", "txt", "source"])
def test_immutable_parent_or_source_changes_refuse_execution(reviewed, project_config, change):
    sid, path, source = reviewed
    p, _ = a.make_plan(project_config, sid)
    if change == "json":
        doc = read_artifact(path)
        doc["segments"][0]["text"] = "different"
        atomic_write_json(path, seal_artifact(doc))
    elif change == "txt":
        path.with_suffix(".txt").write_text("different")
    else:
        data = bytearray(source.read_bytes())
        data[-1] = 1
        source.write_bytes(data)
    with pytest.raises(ValueError):
        execute(project_config, sid)
    assert not (p.parent / "units").exists()


def test_status_is_read_only_and_imports_no_ml_or_llm(reviewed, project_config, monkeypatch):
    sid, _, _ = reviewed
    a.make_plan(project_config, sid)
    root = project_config.application.root
    baseline = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "torchaudio",
            "whisperx",
            "transformers",
            "openai",
            "anthropic",
            "faster_whisper",
        }:
            raise AssertionError("model or LLM import forbidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with deny_network():
        report = a.status_report(project_config, sid)
    assert report["completed_units"] == 0 and not report["models_loaded"]
    assert baseline == {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "kind,flag",
    [
        ("bounds", "OUT_OF_BOUNDS"),
        ("order", "TIMESTAMP_ORDER_ERROR"),
        ("duration", "SUSPICIOUS_WORD_DURATION"),
        ("missing", "UNALIGNED_WORD"),
        ("wording", "ALIGNMENT_FAILED"),
    ],
)
def test_qa_serialization_and_authoritative_text(kind, flag):
    original = seg(310, 312, "first second")
    unit = {"unit_id": "test", "decoded_interval": [300, 600]}
    returned = {
        "words": [
            {"word": "first", "start": 10, "end": 11, "score": 0.2},
            {"word": "second", "start": 11, "end": 12, "score": 0.3},
        ]
    }
    if kind == "bounds":
        returned["words"][0]["start"] = -400
    if kind == "order":
        returned["words"][1]["start"] = 9
    if kind == "duration":
        returned["words"][0]["end"] = 10.001
    if kind == "missing":
        returned["words"][1] = {"word": "second"}
    if kind == "wording":
        returned["words"][0]["word"] = "invented"
    out = a.normalize_segment(original, returned, unit, 0, "segment", 620)
    assert flag in out["qa_flags"]
    assert out["text"] == original["text"] and [w["word"] for w in out["words"]] == [
        "first",
        "second",
    ]
    assert json.loads(json.dumps(out)) == out


def test_invalid_timing_prevents_full_publish_and_reports_failed_unit(reviewed, project_config):
    sid, _, _ = reviewed

    def invalid(originals, unit, clip):
        result = fake_engine(originals, unit, clip)
        if unit["index"] == 1:
            result[0]["words"][0]["start"] = -9999
        return result

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: invalid, extractor=fake_extract
    )
    assert report["failed_units"] == 1 and not report["full_alignment_complete"]
    assert report["qa_counts"]["OUT_OF_BOUNDS"] == 1 and report["aligned_json"] is None


def test_checkpoint_parent_tamper_is_not_reused(reviewed, project_config):
    sid, _, _ = reviewed
    execute(project_config, sid)
    p, plan = a.make_plan(project_config, sid)
    cp_path = a.checkpoint_path(p, plan["units"][0])
    cp = read_artifact(cp_path)
    cp["parent_checksums"]["reviewed_asr"] = "0" * 64
    atomic_write_json(cp_path, seal_artifact(cp))
    status = a.status_report(project_config, sid)
    assert (
        status["unit_status_counts"]["INVALID_CHECKPOINT"] == 1
        and not status["full_alignment_complete"]
    )
    with pytest.raises(ValueError):
        execute(project_config, sid)


def test_default_cli_plans_only_and_status_never_creates_plan(
    reviewed, project_config, monkeypatch, capsys
):
    sid, _, _ = reviewed
    monkeypatch.setattr("session_pipeline.cli.load_config", lambda **_: project_config)
    assert run(["align", sid, "--status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "NOT_PLANNED"
    assert run(["align", sid, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLANNED" and report["real_alignment_executed"] is False
    assert run(["align", sid, "--status", "--fingerprint", report["fingerprint"], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["completed_units"] == 0


def test_offline_adapter_loads_only_pinned_cpu_alignment_model(project_config, monkeypatch):
    local = project_config.application.root / "models/alignment" / a.MODEL["alignment"]["revision"]
    monkeypatch.setattr(
        a,
        "local_model_path",
        lambda root, role: local if role == "alignment" else pytest.fail("ASR"),
    )
    torch = ModuleType("torch")
    torch.set_num_threads = Mock()
    wx = ModuleType("whisperx")
    wx.__path__ = []
    module = ModuleType("whisperx.alignment")
    model = Mock()
    module.load_align_model = Mock(return_value=(model, {"language": "fa"}))
    wx.alignment = module
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "whisperx.alignment", module)
    a.LocalAligner(project_config.application.root)
    module.load_align_model.assert_called_once_with(
        language_code="fa", device="cpu", model_name=str(local), model_cache_only=True
    )
    model.eval.assert_called_once()


def test_whisperx_sentence_adapter_never_downloads_nltk_and_uses_no_interpolation(tmp_path):
    # Real PCM and an injected WhisperX callable; no real model instantiated.
    clip = tmp_path / "fixture.wav"
    wav(clip, 2)
    original = seg(300, 302, "one two")
    unit = {"decoded_interval": [300, 302]}
    module = SimpleNamespace(
        nltk_load=Mock(side_effect=AssertionError("dataset lookup")),
        nltk=SimpleNamespace(download=Mock(side_effect=AssertionError("download"))),
        logger=SimpleNamespace(disabled=False),
    )

    def align(transcript, model, metadata, audio, device, **options):
        assert module.nltk_load("anything").span_tokenize("one two") == [(0, 7)]
        assert options["interpolate_method"] == "ignore" and device == "cpu"
        assert transcript == [{"start": 0, "end": 2, "text": "one two"}]
        return {"word_segments": [{"word": "one", "start": 0, "end": 1}, {"word": "two"}]}

    module.align = align
    engine = a.LocalAligner.__new__(a.LocalAligner)
    engine.wx = module
    engine.model = object()
    engine.metadata = {}
    with deny_network():
        returned = engine([original], unit, clip)
    assert returned[0]["words"][1] == {"word": "two"}
    assert not module.logger.disabled


def test_alignment_import_has_no_model_or_llm_dependency():
    code = """
import builtins
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchaudio', 'whisperx', 'transformers',
                              'faster_whisper', 'openai', 'anthropic'}:
        raise AssertionError('inference dependency imported at module startup')
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
import session_pipeline.alignment
print('PASS')
"""
    import subprocess

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "PASS"


def test_global_order_error_blocks_completion_even_when_unit_checkpoints_exist(
    reviewed, project_config
):
    sid, _, _ = reviewed
    p, plan = a.make_plan(project_config, sid)
    _, document, _, _, _ = a.load_input(project_config, sid)
    results = []
    for unit in plan["units"]:
        originals = [document["segments"][n] for n in unit["segment_indices"]]
        returned = fake_engine(originals, unit, Path("fake.wav"))
        results.extend(
            a.normalize_segment(s, r, unit, n, key, 620)
            for s, r, n, key in zip(
                originals, returned, unit["segment_indices"], unit["segment_ids"], strict=True
            )
        )
    # A whole segment in unit 0 crosses 300s, while the next unit starts at 300s.
    results[2]["words"][-1].update(start=301, end=302)
    results[3]["words"][0].update(start=300, end=301)
    with pytest.raises(ValueError, match="global alignment timestamp order"):
        a.validate_coverage(plan, document, results)


def isolated_failure(originals, unit, clip):
    result = fake_engine(originals, unit, clip)
    if unit["index"] == 0:
        result[0] = {"words": []}
    return result


def legacy_content_checkpoint(config, sid):
    report = a.run_alignment(
        config, sid, engine_factory=lambda _: isolated_failure, extractor=fake_extract
    )
    Path(report["aligned_json"]).unlink()
    p, plan = a.make_plan(config, sid)
    cp_path = a.checkpoint_path(p, plan["units"][0])
    cp = read_artifact(cp_path)
    cp.update(status="FAILED", failure_code="ReviewError")
    cp.pop("execution_completed", None)
    for segment in cp["segments"]:
        segment.pop("alignment_status", None)
        segment["alignment_diagnostics"].pop("failure_reason", None)
    atomic_write_json(cp_path, seal_artifact(cp))
    return p, plan, cp_path


def test_isolated_content_failure_is_partial_with_complete_canonical_coverage(
    reviewed, project_config
):
    sid, path, source = reviewed
    raw = path.parents[2] / "transcript.raw.fa.json"
    immutable = [path, path.with_suffix(".txt"), source, raw, raw.with_suffix(".txt")]
    before = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in immutable}
    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: isolated_failure, extractor=fake_extract
    )
    assert report["succeeded_units"] == 2 and report["partial_units"] == 1
    assert report["failed_units"] == 0 and report["completed_units"] == 3
    assert report["mode_processing_complete"] and report["full_alignment_complete"]
    assert report["coverage_validation_pass"] and report["alignment_has_warnings"]
    assert report["qa_counts"]["ALIGNMENT_FAILED"] == 1
    assert report["qa_counts"]["UNALIGNED_WORD"] == 1 and report["unaligned_word_count"] == 2
    doc = read_artifact(Path(report["aligned_json"]))
    p, plan = a.make_plan(project_config, sid)
    assert [s["reviewed_segment_id"] for s in doc["segments"]] == [
        key for u in plan["units"] for key in u["segment_ids"]
    ]
    originals = read_artifact(path)["segments"]
    assert len(doc["segments"]) == len(originals)
    for aligned, original in zip(doc["segments"], originals, strict=True):
        assert aligned["text"].encode("utf-8") == original["text"].encode("utf-8")
        assert (aligned["start"], aligned["end"]) == (original["start"], original["end"])
    failed = doc["segments"][0]
    assert failed["alignment_status"] == "ALIGNMENT_FAILED"
    assert failed["alignment_diagnostics"]["failure_reason"] == "LEXICAL_MAPPING_FAILED"
    assert all(
        w["start"] is None and w["end"] is None and w["score"] is None for w in failed["words"]
    )
    assert failed["review_origin"] == originals[0].get("review_origin")
    a.validate_coverage(plan, read_artifact(path), doc["segments"])
    assert doc["qa"]["partial_units"] == 1 and doc["qa"]["alignment_has_warnings"]
    assert before == {p: (sha256_file(p), p.stat().st_mtime_ns) for p in immutable}


def test_legacy_content_finalization_reuses_checkpoints_without_inference_or_network(
    reviewed, project_config, monkeypatch
):
    sid, path, _ = reviewed
    p, plan, failed_path = legacy_content_checkpoint(project_config, sid)
    old = failed_path.read_bytes()
    old_checksum = read_artifact(failed_path)["artifact_checksum"]
    successful = {
        a.checkpoint_path(p, u): a.checkpoint_path(p, u).read_bytes() for u in plan["units"][1:]
    }
    before = (path.read_bytes(), path.with_suffix(".txt").read_bytes())
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "torchaudio",
            "whisperx",
            "transformers",
            "faster_whisper",
            "openai",
            "anthropic",
        }:
            raise AssertionError("model or LLM import forbidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(a, "make_plan", Mock(side_effect=AssertionError("no new plan")))
    monkeypatch.setattr(a, "LocalAligner", Mock(side_effect=AssertionError("no model")))
    monkeypatch.setattr(a, "extract_clip", Mock(side_effect=AssertionError("no decode")))
    with deny_network():
        status = a.status_report(project_config, sid, plan_path=p)
        assert status["partial_units"] == 1 and status["full_alignment_complete"]
        assert failed_path.read_bytes() == old  # Status is read-only.
        report = a.finalize_alignment(project_config, sid, plan_path=p)
        artifact = Path(report["aligned_json"])
        canonical = artifact.read_bytes()
        assert a.finalize_alignment(project_config, sid, plan_path=p)["full_alignment_complete"]
        assert artifact.read_bytes() == canonical
    recovered = read_artifact(failed_path)
    assert recovered["status"] == "PARTIAL" and recovered["attempt"] == 1
    assert recovered["recovered_from_checksum"] == old_checksum
    assert (p.parent / "recovery" / f"{old_checksum}.json").read_bytes() == old
    assert successful == {cp: cp.read_bytes() for cp in successful}
    assert before == (path.read_bytes(), path.with_suffix(".txt").read_bytes())


@pytest.mark.parametrize(
    "failure",
    ["execution", "engine_diagnostic", "coverage", "timestamp", "corrupt", "model_parent"],
)
def test_finalization_never_promotes_infrastructure_or_invalid_results(
    reviewed, project_config, failure
):
    sid, _, _ = reviewed
    p, plan, cp_path = legacy_content_checkpoint(project_config, sid)
    cp = read_artifact(cp_path)
    if failure == "execution":
        cp["failure_code"] = "RuntimeError"
    elif failure == "engine_diagnostic":
        cp["segments"][0]["alignment_diagnostics"]["engine_failure_code"] = "MemoryError"
    elif failure == "coverage":
        cp["segments"].pop()
    elif failure == "timestamp":
        cp["segments"][1]["words"][0]["start"] = -10
    elif failure == "model_parent":
        cp["parent_checksums"]["model_lock"] = "0" * 64
    atomic_write_json(cp_path, seal_artifact(cp))
    if failure == "corrupt":
        cp_path.write_bytes(b"invalid json")
    before = cp_path.read_bytes()
    report = a.finalize_alignment(project_config, sid, plan_path=p)
    assert not report["mode_processing_complete"] and not report["coverage_validation_pass"]
    assert report["aligned_json"] is None and cp_path.read_bytes() == before
    assert not (p.parent / "recovery").exists()


def test_stale_running_status_is_interrupted_and_live_worker_remains_running(
    reviewed, project_config
):
    sid, _, _ = reviewed
    completed = execute(project_config, sid)
    # Simulate interruption before canonical publication, not damage to a
    # published lineage (which must still be refused).
    Path(completed["aligned_json"]).unlink()
    p, plan = a.make_plan(project_config, sid)
    cp_path = a.checkpoint_path(p, plan["units"][0])
    cp = read_artifact(cp_path)
    cp.update(status="RUNNING", segments=[])
    atomic_write_json(cp_path, seal_artifact(cp))
    before = cp_path.read_bytes()
    with a.exclusive_lock(project_config.application.work_path / sid / "alignment/.worker.lock"):
        assert a.status_report(project_config, sid)["unit_status_counts"]["RUNNING"] == 1
    assert a.status_report(project_config, sid)["unit_status_counts"]["INTERRUPTED"] == 1
    assert cp_path.read_bytes() == before
    assert not a.finalize_alignment(project_config, sid, plan_path=p)["mode_processing_complete"]
    assert execute(project_config, sid)["full_alignment_complete"]


def test_complete_partial_resume_skips_all_model_execution(reviewed, project_config):
    sid, _, _ = reviewed
    first = a.run_alignment(
        project_config, sid, engine_factory=lambda _: isolated_failure, extractor=fake_extract
    )
    artifact = Path(first["aligned_json"]).read_bytes()
    report = a.run_alignment(
        project_config,
        sid,
        engine_factory=Mock(side_effect=AssertionError("partial must not retry")),
        extractor=Mock(side_effect=AssertionError("no decode")),
    )
    assert report["partial_units"] == 1 and report["full_alignment_complete"]
    assert Path(report["aligned_json"]).read_bytes() == artifact


def test_finalize_cli_uses_existing_exact_plan(reviewed, project_config, monkeypatch, capsys):
    sid, _, _ = reviewed
    p, plan, _ = legacy_content_checkpoint(project_config, sid)
    monkeypatch.setattr("session_pipeline.cli.load_config", lambda **_: project_config)
    monkeypatch.setattr(a, "run_alignment", Mock(side_effect=AssertionError("no inference")))
    assert (
        run(["align", sid, "--finalize", "--fingerprint", plan["stage_fingerprint"], "--json"]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["partial_units"] == 1 and report["aligned_json"]


def test_adapter_execution_exception_propagates_as_failed_unit(tmp_path):
    clip = tmp_path / "fixture.wav"
    wav(clip, 2)
    module = SimpleNamespace(
        nltk_load=Mock(),
        nltk=SimpleNamespace(download=Mock()),
        logger=SimpleNamespace(disabled=False),
        align=Mock(side_effect=MemoryError("private")),
    )
    engine = a.LocalAligner.__new__(a.LocalAligner)
    engine.wx, engine.model, engine.metadata = module, object(), {}
    with pytest.raises(MemoryError):
        engine([seg(0, 2, "one two")], {"decoded_interval": [0, 2]}, clip)


def test_multiple_lexical_failures_in_one_unit_publish_json_and_txt(reviewed, project_config):
    sid, path, _ = reviewed

    def several(originals, unit, clip):
        result = fake_engine(originals, unit, clip)
        if unit["index"] == 0:
            result[0] = {"words": []}
            result[1] = {"words": [{"word": "unrelated"}]}
        return result

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: several, extractor=fake_extract
    )
    assert report["status"] == "SUCCEEDED_WITH_WARNINGS"
    assert report["partial_units"] == 1 and report["failed_units"] == 0
    assert report["coverage_validation_pass"] and report["full_alignment_complete"]
    assert report["unaligned_word_count"] == 4
    doc = read_artifact(Path(report["aligned_json"]))
    assert doc["status"] == "SUCCEEDED_WITH_WARNINGS"
    originals = read_artifact(path)["segments"]
    assert [s["text"] for s in doc["segments"]] == [s["text"] for s in originals]
    assert Path(report["aligned_txt"]).read_bytes() == (
        "\n".join(s["text"] for s in originals) + "\n"
    ).encode("utf-8")
    for s in doc["segments"][:2]:
        assert s["status"] == "ALIGNMENT_FAILED"
        assert all(w["start"] is None and w["end"] is None for w in s["words"])


def test_zero_duration_is_isolated_before_engine_and_text_stays_immutable(reviewed, project_config):
    sid, path, source = reviewed
    doc = read_artifact(path)
    doc["segments"][1]["end"] = doc["segments"][1]["start"]
    atomic_write_json(path, seal_artifact(doc))
    pointer = read_artifact(path.parents[1] / "latest.json")
    pointer["artifact_checksum_of_reviewed_json"] = read_artifact(path)["artifact_checksum"]
    atomic_write_json(path.parents[1] / "latest.json", seal_artifact(pointer))
    baseline = {
        p: (sha256_file(p), p.stat().st_mtime_ns) for p in (path, source, path.with_suffix(".txt"))
    }

    def positive_only(originals, unit, clip):
        assert all(s["end"] > s["start"] for s in originals)
        return fake_engine(originals, unit, clip)

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: positive_only, extractor=fake_extract
    )
    failed = read_artifact(Path(report["aligned_json"]))["segments"][1]
    assert report["status"] == "SUCCEEDED_WITH_WARNINGS" and report["full_alignment_complete"]
    assert failed["start"] == failed["end"] == 100
    assert failed["text"] == doc["segments"][1]["text"]
    assert failed["alignment_diagnostics"]["failure_reason"] == "ZERO_DURATION_REVIEWED_SEGMENT"
    assert all(w["start"] is None and w["end"] is None for w in failed["words"])
    assert baseline == {p: (sha256_file(p), p.stat().st_mtime_ns) for p in baseline}


def test_review_error_aborted_unit_never_recovers_as_lexical_warning(reviewed, project_config):
    from session_pipeline.human_review import ReviewError

    sid, _, _ = reviewed

    def abort(originals, unit, clip):
        if unit["index"] == 2:
            raise ReviewError("zero-duration alignment segment")
        return fake_engine(originals, unit, clip)

    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: abort, extractor=fake_extract
    )
    p, plan = a.make_plan(project_config, sid)
    cp_path = a.checkpoint_path(p, plan["units"][2])
    cp = read_artifact(cp_path)
    assert cp["processed_audio_seconds"] == 0 and cp["failure_code"] == "ReviewError"
    assert all(
        s["alignment_diagnostics"]["failure_reason"] == "ENGINE_EXECUTION_FAILED"
        for s in cp["segments"]
    )
    before = cp_path.read_bytes()
    finalized = a.finalize_alignment(project_config, sid, plan_path=p)
    assert report["failed_units"] == finalized["failed_units"] == 1
    assert finalized["status"] == "FAILED" and finalized["aligned_json"] is None
    assert before == cp_path.read_bytes()


def test_quality_adapter_finalizes_persisted_partial_without_model_or_retry(
    reviewed, project_config, monkeypatch
):
    from session_pipeline import workflow

    sid, path, source = reviewed
    p, _, _ = legacy_content_checkpoint(project_config, sid)
    monkeypatch.setattr(a, "run_alignment", Mock(side_effect=AssertionError("no inference")))
    monkeypatch.setattr(a, "make_plan", Mock(side_effect=AssertionError("no new plan")))
    monkeypatch.setattr(a, "LocalAligner", Mock(side_effect=AssertionError("no model")))
    adapter = workflow.LocalStages(project_config, source, sid, "auto-prefer-retry")
    with deny_network():
        result = adapter.alignment({"REVIEW": {"reviewed_fingerprint": path.parent.name}})
    assert result["state"] == "SUCCEEDED_WITH_WARNINGS"
    assert result["warning_counts"]["failed_segments"] == 1
    assert result["warning_counts"]["unaligned_words"] == 2
    assert Path(result["txt_path"]).is_file()
    assert a.status_report(project_config, sid, plan_path=p)["full_alignment_complete"]


def test_adapter_zero_duration_skips_only_that_segment_and_continues(tmp_path):
    clip = tmp_path / "fixture.wav"
    wav(clip, 2)
    module = SimpleNamespace(
        nltk_load=Mock(),
        nltk=SimpleNamespace(download=Mock()),
        logger=SimpleNamespace(disabled=False),
        align=Mock(return_value={"word_segments": [{"word": "retained", "start": 1, "end": 2}]}),
    )
    engine = a.LocalAligner.__new__(a.LocalAligner)
    engine.wx, engine.model, engine.metadata = module, object(), {}
    with deny_network():
        result = engine(
            [seg(0, 0, "zero"), seg(1, 2, "retained")], {"decoded_interval": [0, 2]}, clip
        )
    assert result[0]["failure_code"] == "ZERO_DURATION_REVIEWED_SEGMENT"
    assert result[0]["words"] == []
    assert result[1]["words"][0]["word"] == "retained"
    assert module.align.call_count == 1


def test_explicit_incomplete_execution_cannot_be_promoted(reviewed, project_config):
    sid, _, _ = reviewed
    p, _, cp_path = legacy_content_checkpoint(project_config, sid)
    cp = read_artifact(cp_path)
    cp["execution_completed"] = False
    atomic_write_json(cp_path, seal_artifact(cp))
    before = cp_path.read_bytes()
    report = a.finalize_alignment(project_config, sid, plan_path=p)
    assert report["status"] == "FAILED" and report["aligned_json"] is None
    assert cp_path.read_bytes() == before


def test_legacy_succeeded_canonical_with_warnings_reused_byte_for_byte(reviewed, project_config):
    sid, _, _ = reviewed
    report = a.run_alignment(
        project_config, sid, engine_factory=lambda _: isolated_failure, extractor=fake_extract
    )
    path = Path(report["aligned_json"])
    doc = read_artifact(path)
    doc["status"] = "SUCCEEDED"
    atomic_write_json(path, seal_artifact(doc))
    before = path.read_bytes(), path.stat().st_mtime_ns
    finalized = a.finalize_alignment(project_config, sid)
    assert finalized["status"] == "SUCCEEDED_WITH_WARNINGS"
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
