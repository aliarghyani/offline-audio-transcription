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
    assert cp["status"] == "RUNNING"
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
