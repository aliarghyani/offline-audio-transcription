"""Synthetic product acceptance: never load/download real models or private audio."""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest
from test_alignment import fake_engine, fake_extract
from test_human_review import completed as completed_fixture
from test_human_review import seg, wav

from session_pipeline import alignment, cli, diarization, raw_asr, speaker_review, workflow
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.models import build_session_source


@pytest.fixture
def product(project_config):
    root = project_config.application.root
    origin = Path(__file__).resolve().parents[1]
    for name in ("models.lock.json", "asr-quality.json"):
        (root / "config" / name).write_bytes((origin / "config" / name).read_bytes())
    source = root / "sessions" / "future session.wav"
    wav(source, 100)
    return project_config, source


class SyntheticStages(workflow.LocalStages):
    calls = []
    confirmed = False
    missing = None
    fail = None
    interrupt = None

    def model_ready(self, role):
        return role != self.missing

    def inference_needed(self, stage, results):
        return True

    def run(self, stage, results):
        self.calls.append((self.sid, stage))
        if stage == self.fail:
            raise ValueError("synthetic failure")
        if stage == self.interrupt:
            raise KeyboardInterrupt
        if stage in {"INGEST", "INTEGRITY"}:
            return super().run(stage, results)
        if stage == "IDENTITY_MAPPING" and not self.confirmed:
            return {
                "state": "WAITING_FOR_USER",
                "gate": "SPEAKER_IDENTITY",
                "review_command": f"uv run --no-sync session-pipeline review {self.sid} --open",
            }
        path = self.work / "synthetic" / (stage + ".json")
        value = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "synthetic_stage",
                "status": "SUCCEEDED",
                "session_id": self.sid,
                "stage": stage,
            }
        )
        atomic_write_json(path, value)
        result = {"state": "SUCCEEDED", "artifacts": [str(path)]}
        for key in (
            "retry_fingerprint",
            "reviewed_fingerprint",
            "aligned_fingerprint",
            "diarization_fingerprint",
            "attributed_fingerprint",
        ):
            result[key] = "a" * 64
        if stage == "ASR_QA":
            result.update(state="SUCCEEDED_WITH_WARNINGS", warnings=["synthetic triage warning"])
        if stage in {"ASR", "ALIGNMENT", "FINAL_RENDER"}:
            txt = path.with_suffix(".txt")
            txt.write_text("synthetic output\n", encoding="utf-8")
            result.update(json_path=str(path), txt_path=str(txt))
        return result


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(SyntheticStages, "calls", [])
    monkeypatch.setattr(SyntheticStages, "confirmed", False)
    monkeypatch.setattr(SyntheticStages, "missing", None)
    monkeypatch.setattr(SyntheticStages, "fail", None)
    monkeypatch.setattr(SyntheticStages, "interrupt", None)
    return SyntheticStages


@pytest.mark.parametrize("profile,expected", [("fast", 3), ("quality", 7), ("full", 11)])
def test_planning_no_models_no_writes(product, profile, expected, monkeypatch):
    config, source = product
    monkeypatch.setattr(
        workflow.LocalStages, "model_ready", lambda *a: pytest.fail("model inspect")
    )
    result = workflow.transcribe_one(config, source, profile=profile, dry_run=True)
    assert sum(s["state"] == "PENDING" for s in result["stages"].values()) == expected
    assert not config.application.work_path.exists()


def test_fast_resume_and_lookup(product, synthetic):
    config, source = product
    result = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert result["state"] == "SUCCEEDED"
    assert Path(result["txt_path"]).is_file()
    before = len(synthetic.calls)
    resumed = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert len(synthetic.calls) == before + 1  # ingest only; ASR and integrity verified/reused
    assert resumed["stages"]["ASR"]["reused"]
    receipt = workflow.receipt_path(config, result["session_id"])
    baseline = receipt.read_bytes(), receipt.stat().st_mtime_ns
    report = workflow.status(config, source)
    assert report["txt_path"] == result["txt_path"]
    assert baseline == (receipt.read_bytes(), receipt.stat().st_mtime_ns)


def test_quality_graph_warning_and_profile_upgrade(product, synthetic):
    config, source = product
    workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    result = workflow.transcribe_one(config, source, profile="quality", adapter_factory=synthetic)
    assert result["state"] == "SUCCEEDED_WITH_WARNINGS"
    assert result["review_policy"] == "auto-prefer-retry"
    assert result["stages"]["ASR"]["reused"]
    assert [s for _, s in synthetic.calls].count("ASR") == 1
    assert result["stages"]["DIARIZATION"]["state"] == "SKIPPED"


def test_full_identity_wait_resume_final(product, synthetic):
    config, source = product
    waiting = workflow.transcribe_one(config, source, adapter_factory=synthetic)
    assert waiting["state"] == "WAITING_FOR_USER"
    sid = waiting["session_id"]
    assert waiting["review_command"].endswith(f"review {sid} --open")
    assert waiting["stages"]["FINAL_RENDER"]["state"] == "PENDING"
    assert workflow.selected_review(config, sid, "SPEAKER_IDENTITY") == "a" * 64
    synthetic.confirmed = True
    done = workflow.transcribe_one(config, source, adapter_factory=synthetic)
    assert done["state"] == "SUCCEEDED_WITH_WARNINGS"
    assert done["stages"]["DIARIZATION"]["reused"]
    assert [s for _, s in synthetic.calls].count("ASR") == 1
    assert [s for _, s in synthetic.calls].count("ALIGNMENT") == 1
    assert Path(done["txt_path"]).is_file()


def test_stale_running_recovery_and_ctrl_c(product, synthetic):
    config, source = product
    synthetic.interrupt = "ASR"
    stopped = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert stopped["state"] == "INTERRUPTED"
    path = workflow.receipt_path(config, stopped["session_id"])
    old = read_artifact(path)
    old["stages"]["ASR"]["state"] = "RUNNING"  # simulate terminal/power loss
    old["state"] = "RUNNING"
    atomic_write_json(path, seal_artifact(old))
    before = path.read_bytes()
    assert workflow.status(config, source)["state"] == "INTERRUPTED"
    assert before == path.read_bytes()
    synthetic.interrupt = None
    done = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert done["state"] == "SUCCEEDED"
    assert done["stages"]["ASR"]["recovered_from"] == "INTERRUPTED"


def test_missing_model_blocks_without_inference(product, synthetic):
    config, source = product
    synthetic.missing = "asr"
    result = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert result["state"] == "BLOCKED_MODEL_PROVISIONING"
    assert result["provision_command"].endswith("models provision --role asr")
    assert "ASR" not in [s for _, s in synthetic.calls]


def test_gated_diarization_block(product, synthetic):
    config, source = product
    synthetic.missing = "diarization"
    result = workflow.transcribe_one(config, source, adapter_factory=synthetic)
    assert result["state"] == "BLOCKED_MODEL_PROVISIONING"
    assert result["stopped_at"] == "DIARIZATION"
    assert result["provision_command"].endswith("models provision --role diarization")
    assert "DIARIZATION" not in [s for _, s in synthetic.calls]


def test_current_settings_change_invalidates_status_and_downstream(product, synthetic):
    config, source = product
    done = workflow.transcribe_one(config, source, profile="quality", adapter_factory=synthetic)
    path = config.application.root / "config/asr-quality.json"
    thresholds = json.loads(path.read_text("utf-8"))
    thresholds["high_temperature"] = 0.8
    path.write_text(json.dumps(thresholds), encoding="utf-8")
    status = workflow.status(config, source)
    assert status["state"] == "FAILED" and status["stages"]["ASR_QA"]["state"] == "FAILED"
    resumed = workflow.transcribe_one(config, source, profile="quality", adapter_factory=synthetic)
    assert resumed["stages"]["ASR"]["reused"]
    assert not resumed["stages"]["ASR_QA"]["reused"]
    assert done["stages"]["ASR_QA"]["fingerprint"] != resumed["stages"]["ASR_QA"]["fingerprint"]


def test_failed_stage_stops_and_bad_binding_prevents_reuse(product, synthetic):
    config, source = product
    synthetic.fail = "ASR"
    failed = workflow.transcribe_one(config, source, adapter_factory=synthetic)
    assert failed["state"] == "FAILED" and failed["stopped_at"] == "ASR"
    assert "ASR_QA" not in [s for _, s in synthetic.calls]
    synthetic.fail = None
    done = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    Path(done["txt_path"]).write_bytes(b"changed output")
    assert workflow.status(config, source)["state"] == "FAILED"
    resumed = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert not resumed["stages"]["ASR"]["reused"]


def test_batch_deterministic_sequential_stop_on_gate(product, synthetic):
    config, source = product
    second = source.with_name("z.wav")
    wav(second, 101)
    paths = workflow.audio_inputs(source.parent)
    assert paths == [source, second]
    result = workflow.transcribe(config, source.parent, adapter_factory=synthetic)
    assert result["state"] == "WAITING_FOR_USER"
    assert result["unprocessed"] == [str(second)]
    assert len({sid for sid, _ in synthetic.calls}) == 1
    synthetic.calls.clear()
    fast = workflow.transcribe(config, source.parent, profile="fast", adapter_factory=synthetic)
    assert len(fast["sessions"]) == 2
    ids = [sid for sid, _ in synthetic.calls]
    assert ids == sorted(ids, key=lambda sid: ids.index(sid))


def test_external_input_copy_preserves_original_and_duplicate_discovery(product, synthetic):
    config, source = product
    external = config.application.root / "external.wav"
    external.write_bytes(source.read_bytes())
    before = external.read_bytes(), external.stat().st_mtime_ns
    first = workflow.transcribe_one(config, external, profile="fast", adapter_factory=synthetic)
    ingest = first["stages"]["INGEST"]["result"]
    staged = Path(ingest["source"])
    assert staged.parent == config.application.sessions_path
    assert staged.name.startswith(build_session_source(external).source_sha256)
    assert before == (external.read_bytes(), external.stat().st_mtime_ns)
    assert len(workflow.audio_inputs(source.parent)) == 1
    resumed = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert resumed["stages"]["ASR"]["reused"]


def test_cli_json_gate_review_and_default_summary(product, synthetic, monkeypatch, capsys):
    config, source = product
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: config)
    original = workflow.transcribe_one
    monkeypatch.setattr(
        workflow, "transcribe_one", lambda *a, **k: original(*a, **k, adapter_factory=synthetic)
    )
    assert cli.run(["transcribe", str(source), "--json"]) == 7
    waiting = json.loads(capsys.readouterr().out)["sessions"][0]
    calls = []
    monkeypatch.setattr(speaker_review, "serve_review", lambda *a, **k: calls.append((a, k)))
    assert cli.run(["review", waiting["session_id"], "--open"]) == 0
    assert calls[0][0][2] == "a" * 64
    assert cli.run(["status", waiting["session_id"]]) == 7
    assert "WAITING_FOR_USER" in capsys.readouterr().out
    assert cli.run(["transcribe", str(source), "--profile", "fast"]) == 0
    text = capsys.readouterr().out
    assert "TXT:" in text and "segments" not in text


def test_no_network_cloud_reference_and_private_ignored(product, synthetic, monkeypatch):
    config, source = product
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network"))
    reference = config.application.root / "reference_transcripts" / "history.wav"
    reference.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="historical"):
        workflow.transcribe_one(config, reference, adapter_factory=synthetic)
    result = workflow.transcribe_one(config, source, profile="fast", adapter_factory=synthetic)
    assert not result["cloud_llm_required"] and not result["network_required"]
    origin = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "sessions/test.wav",
            "outputs/test.json",
            "work/test.json",
            "models/test.bin",
            "reference_transcripts/x.txt",
        ],
        cwd=origin,
        capture_output=True,
        check=True,
    )
    assert len(ignored.stdout.splitlines()) == 5


def test_raw_native_output_reused_without_loading(product, monkeypatch):
    config, source = product
    identity = build_session_source(source)
    adapter = workflow.LocalStages(config, source, identity.session_id, "auto-prefer-retry")
    ingest = adapter.ingest({})
    adapter.integrity({"INGEST": ingest})
    calls = []

    def inference(engine, audio, offset, *, checkpoint):
        calls.append(audio)
        assert offset == 0 and engine == "synthetic"
        segments = [seg(1, 2, "synthetic text")]
        checkpoint(segments)
        return segments, {"language": "fa"}

    done = raw_asr.execute(
        config, identity.session_id, engine_factory=lambda _: "synthetic", inference=inference
    )
    Path(done["json_path"]).unlink()  # native output survives a publication interruption
    raw_asr.execute(
        config,
        identity.session_id,
        engine_factory=lambda _: pytest.fail("repeated ASR"),
        inference=inference,
    )
    assert len(calls) == 1
    assert read_artifact(Path(done["json_path"]))["model"]["asr"]["revision"]


@pytest.mark.parametrize(
    "policy,partial,raw_recovery",
    [
        (policy, partial, False)
        for policy in ("auto-prefer-retry", "conservative", "human")
        for partial in (False, True)
    ]
    + [("auto-prefer-retry", False, True)],
)
def test_real_adapters_synthetic_full_pipeline_identity_resume(
    product, monkeypatch, policy, partial, raw_recovery
):
    """Actual artifact engines/QA/merge/renderer, with only ML and audio extraction injected."""
    config, source = product
    calls = {"asr": 0, "alignment": 0, "diarization": 0}
    original_asr = raw_asr.execute

    def native(engine, audio, offset, *, checkpoint):
        calls["asr"] += 1
        segments = [
            seg(1, 3, "first words"),
            seg(16, 18, "second words"),
            seg(36, 38, "third words"),
            seg(51, 53, "fourth words"),
            seg(71, 73, "fifth words"),
            seg(86, 88, "last words"),
        ]
        return segments, {"language": "fa"}

    monkeypatch.setattr(
        raw_asr,
        "execute",
        lambda *a, **k: original_asr(
            *a, **k, engine_factory=lambda _: "synthetic", inference=native
        ),
    )
    monkeypatch.setattr(workflow.LocalStages, "model_ready", lambda *a: True)
    original_align = alignment.run_alignment

    def align_engine(originals, unit, clip):
        calls["alignment"] += 1
        result = fake_engine(originals, unit, clip)
        if partial:
            result[0] = {"words": []}
        return result

    monkeypatch.setattr(
        alignment,
        "run_alignment",
        lambda *a, **k: original_align(
            *a, **k, engine_factory=lambda _: align_engine, extractor=fake_extract
        ),
    )
    # Diarization still runs its genuine raw-persistence and canonical validation lifecycle.
    from session_pipeline import model_store

    spec = diarization.model_spec(config)
    model_path = model_store.model_path(config.application.root, spec)
    monkeypatch.setattr(diarization, "local_model_path", lambda *a: model_path)
    monkeypatch.setattr(
        diarization,
        "inspect_model",
        lambda *a, **k: {"status": "PROVISIONED", "files": [{"synthetic": True}]},
    )
    original_diarize = diarization.run_diarization

    def diarize_engine(clip):
        calls["diarization"] += 1
        turns = [
            {
                "start": i * 15 + (5 if i > 1 else 0),
                "end": min(100, (i + 1) * 15 + (5 if i > 1 else 0)),
                "speaker": f"SPEAKER_0{i % 2}",
            }
            for i in range(6)
        ]
        return {"turns": turns, "exclusive_turns": turns}

    monkeypatch.setattr(
        diarization,
        "run_diarization",
        lambda *a, **k: original_diarize(
            *a, **k, engine_factory=lambda _: diarize_engine, extractor=fake_extract
        ),
    )
    monkeypatch.setattr(
        speaker_review,
        "extract_clip",
        lambda source, clip, interval: wav(clip, interval[1] - interval[0]),
    )
    if raw_recovery:
        canonical = diarization.canonical_turns
        failures = [True]

        def fail_after_raw(*args, **kwargs):
            if failures:
                failures.pop()
                raise ValueError("synthetic post-inference validation interruption")
            return canonical(*args, **kwargs)

        monkeypatch.setattr(diarization, "canonical_turns", fail_after_raw)
        failed = workflow.transcribe_one(config, source, policy=policy)
        assert failed["state"] == "FAILED" and failed["stopped_at"] == "DIARIZATION"
        raw_files = list(
            (config.application.work_path / failed["session_id"] / "diarization").glob(
                "*/raw-inference.json"
            )
        )
        assert len(raw_files) == 1
        persisted = raw_files[0].read_bytes()
        monkeypatch.setattr(workflow.LocalStages, "model_ready", lambda *a: False)
    # Retry should have no target windows for this high-quality short synthetic recording.
    waiting = workflow.transcribe_one(config, source, policy=policy)
    assert waiting["state"] == "WAITING_FOR_USER", waiting
    if raw_recovery:
        assert persisted == raw_files[0].read_bytes() and calls["diarization"] == 1
    sid = waiting["session_id"]
    fp = workflow.selected_review(config, sid, "SPEAKER_IDENTITY")
    speaker_review.save_mapping(
        config, sid, fp, {"SPEAKER_00": "علی", "SPEAKER_01": "مشاور"}, None, human_confirmed=True
    )
    before = calls.copy()
    done = workflow.transcribe_one(config, source, policy=policy)
    assert done["state"] in workflow.COMPLETE, done
    assert calls == before == {"asr": 1, "alignment": 1, "diarization": 1}
    final = read_artifact(Path(done["json_path"]))
    assert final["mapping_source"] == "HUMAN_CONFIRMED" and final["segment_count"] == 6
    assert "last words" in Path(done["txt_path"]).read_text("utf-8")
    assert workflow.status(config, sid)["state"] in workflow.COMPLETE
    assert not final["ml_inference_used"] and not final["cloud_llm_used"]
    assert not done["stages"]["REVIEW"]["result"]["human_verified"]  # no A/B targets
    if partial:
        assert done["stages"]["ALIGNMENT"]["state"] == "SUCCEEDED_WITH_WARNINGS"
        assert final["segments"][0]["text"] == "first words"
    # Lost workflow progress must still reuse completed engine/native outputs.
    receipt = workflow.receipt_path(config, sid)
    saved = read_artifact(receipt)
    for stage in ("ASR", "ALIGNMENT", "DIARIZATION"):
        saved["stages"][stage]["state"] = "RUNNING"
    saved["state"] = "RUNNING"
    atomic_write_json(receipt, seal_artifact(saved))
    monkeypatch.setattr(workflow.LocalStages, "model_ready", lambda *a: False)
    recovery = workflow.transcribe_one(config, source, policy=policy)
    assert recovery["state"] in workflow.COMPLETE, recovery.get("error")
    assert calls == before
    receipt.unlink()
    legacy = workflow.status(config, sid)
    assert legacy["recognition"] == "READ_ONLY_PRE_PRODUCTIZATION_FINAL"
    assert legacy["txt_path"] == done["txt_path"]
    assert not receipt.exists()


def test_review_policy_nonempty_target_lineage_and_human_gate(project_config, monkeypatch):
    # The existing fixture uses only generated audio and injected retry inference.
    from session_pipeline import human_review

    config = project_config
    sid, plan, raw_path, source = completed_fixture.__wrapped__(config)
    conservative = workflow.reviewed_policy(config, sid, plan.parent.name, "conservative")
    doc = read_artifact(Path(conservative["reviewed_json"]))
    assert doc["segments"] == read_artifact(raw_path)["segments"]
    assert doc["qa_acceptance"]["alignment_ready"]
    assert not doc["provenance"]["human_verified"]
    monkeypatch.setattr(
        human_review,
        "extract_clip",
        lambda source, clip, interval: wav(clip, interval[1] - interval[0]),
    )
    waiting = workflow.reviewed_policy(config, sid, plan.parent.name, "human")
    assert waiting["state"] == "WAITING_FOR_USER" and waiting["gate"] == "ASR_REVIEW"
    context = human_review.load_context(config, sid, plan.parent.name)
    bundle = human_review.load_bundle(context)
    decisions = human_review.load_decisions(context, bundle)
    for entry in bundle["entries"]:
        decisions = human_review.save_decision(
            config,
            sid,
            plan.parent.name,
            entry["review_id"],
            "USE_A",
            decisions["artifact_checksum"],
        )
    human = workflow.reviewed_policy(config, sid, plan.parent.name, "human")
    doc = read_artifact(Path(human["reviewed_json"]))
    assert doc["qa_acceptance"]["alignment_ready"] and doc["provenance"]["human_verified"]
    assert doc["segments"] != []
    adapter = workflow.LocalStages(config, source, sid, "auto-prefer-retry")
    automatic = adapter.review({"ASR_RETRY": {"retry_fingerprint": plan.parent.name}})
    reviewed = read_artifact(Path(automatic["json_path"]))
    assert reviewed["provenance"]["review_mode"] == "AUTO_PREFER_B"
    assert not automatic["human_verified"]
