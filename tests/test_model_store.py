from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from session_pipeline import cli, doctor
from session_pipeline import model_store as store


@pytest.fixture
def model_fixture(project_root, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    original = Path(__file__).resolve().parents[1] / "config/models.lock.json"
    lock = json.loads(original.read_text(encoding="utf-8"))
    contents = {}
    for spec in lock["models"]:
        role = spec["role"]
        spec["revision"] = "1" * 40
        spec["destination"] = f"models/{role}/{spec['revision']}"
        for file in spec["files"]:
            if file["path"].endswith(".json"):
                data = json.dumps({"model_type": "wav2vec2", "synthetic": True}).encode()
            elif file["path"] == "config.yaml":
                data = (
                    b"pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n"
                    b"  params:\n    segmentation: $model/segmentation\n"
                    b"    embedding: $model/embedding\n    plda: $model/plda\n"
                )
            else:
                data = b"synthetic test asset, never loaded"
            contents[(role, file["path"])] = data
            file.update({"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                         "git_blob_sha1": None})
    store.write_json(project_root / "config/models.lock.json", lock)
    return project_root, lock, contents


def populate(fixture, roles=tuple(store.APPROVED)):
    root, lock, contents = fixture
    for spec in lock["models"]:
        if spec["role"] not in roles:
            continue
        for file in spec["files"]:
            path = store.model_path(root, spec) / file["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[(spec["role"], file["path"])])


def test_lock_parsing_and_relative_paths(model_fixture):
    root, _, _ = model_fixture
    lock = store.load_model_lock(root)
    assert {s["repo_id"] for s in lock["models"]} == set(store.APPROVED.values())
    for spec in lock["models"]:
        assert store.model_path(root, spec).is_relative_to(root / "models")
        assert "flax_model.msgpack" not in {f["path"] for f in spec["files"]}


@pytest.mark.parametrize("revision", ["main", "v1", "a" * 39, "A" * 40, None])
def test_lock_rejects_floating_or_invalid_revisions(model_fixture, revision):
    root, lock, _ = model_fixture
    lock["models"][0]["revision"] = revision
    store.write_json(root / "config/models.lock.json", lock)
    with pytest.raises(store.ModelLockError, match="immutable"):
        store.load_model_lock(root)


@pytest.mark.parametrize("mutation", ["repo", "path", "flax", "checksum", "required", "token"])
def test_lock_rejects_unapproved_contract_changes(model_fixture, mutation):
    root, lock, _ = model_fixture
    spec = lock["models"][0]
    if mutation == "repo":
        spec["repo_id"] = "unapproved/mirror"
    elif mutation == "path":
        spec["destination"] = "../sessions"
    elif mutation == "flax":
        spec["files"][0]["path"] = "flax_model.msgpack"
    elif mutation == "checksum":
        spec["files"][0]["sha256"] = None
    elif mutation == "required":
        spec["files"] = []
    else:
        spec["token"] = "fake-test-secret"
    store.write_json(root / "config/models.lock.json", lock)
    with pytest.raises(store.ModelLockError):
        store.load_model_lock(root)


def test_status_and_verify_never_download_or_call_preflight(model_fixture, monkeypatch):
    root, _, _ = model_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("network operation forbidden")

    monkeypatch.setattr(store, "_download", forbidden)
    monkeypatch.setattr(store, "_access_preflight", forbidden)
    assert store.models_report(root)["models_ready"] is False
    populate(model_fixture)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    result = store.models_report(root, action="verify")
    assert result["models_ready"] is True
    assert result["models_loaded"] is False
    assert store.models_report(root)["models_ready"] is True
    assert store.local_model_path(root, "asr").is_dir()


def test_absent_token_does_not_block_public_roles(model_fixture, monkeypatch):
    root, _, contents = model_fixture

    def download(spec, file, root, token, **kwargs):
        assert spec["role"] != "diarization"
        assert token is False
        path = store.model_path(root, spec) / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[(spec["role"], file["path"])])

    monkeypatch.setattr(store, "_download", download)
    report = store.models_report(root, action="provision")
    states = {m["role"]: m["status"] for m in report["models"]}
    assert states == {"asr": "PROVISIONED", "alignment": "PROVISIONED",
                      "diarization": "BLOCKED_AUTH"}


def test_gated_failure_and_receipt_never_expose_token(model_fixture, monkeypatch):
    root, lock, _ = model_fixture
    token = "fake-test-secret-only"
    monkeypatch.setenv("HF_TOKEN", token)

    class AccessError(Exception):
        response = SimpleNamespace(status_code=403)

    def denied(spec, value):
        assert value == token
        raise AccessError(f"Authorization: Bearer {token}")

    monkeypatch.setattr(store, "_access_preflight", denied)
    result = store.models_report(root, action="provision", role="diarization")
    assert result["models"][0]["status"] == "BLOCKED_ACCESS_CONDITIONS"
    assert token not in json.dumps(result) + store.human_models_report(result)
    spec = next(s for s in lock["models"] if s["role"] == "diarization")
    assert token not in (store.model_path(root, spec) / ".provision.json").read_text()


def test_partial_download_resumes_valid_files(model_fixture, monkeypatch):
    root, lock, contents = model_fixture
    calls = []

    def download(spec, file, root, token, **kwargs):
        calls.append(file["path"])
        if len(calls) == 2:
            raise RuntimeError("fake-test-secret must not enter diagnostics")
        path = store.model_path(root, spec) / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[(spec["role"], file["path"])])

    monkeypatch.setattr(store, "_download", download)
    first = store.models_report(root, action="provision", role="asr")
    assert first["models"][0]["status"] == "PARTIAL"
    assert "fake-test-secret" not in json.dumps(first)
    completed_first = lock["models"][0]["files"][0]["path"]
    calls.clear()

    def resume(spec, file, root, token, **kwargs):
        calls.append(file["path"])
        path = store.model_path(root, spec) / file["path"]
        path.write_bytes(contents[(spec["role"], file["path"])])

    monkeypatch.setattr(store, "_download", resume)
    second = store.models_report(root, action="provision", role="asr")
    assert second["models_ready"] is True
    assert completed_first not in calls


def test_corrupt_critical_file_detected_even_at_same_size(model_fixture):
    root, lock, _ = model_fixture
    populate(model_fixture, roles=("asr",))
    assert store.models_report(root, action="verify", role="asr")["models_ready"]
    path = store.model_path(root, lock["models"][0]) / "model.bin"
    path.write_bytes(b"x" * path.stat().st_size)
    result = store.models_report(root, action="verify", role="asr")["models"][0]
    assert result["status"] == "CORRUPT"
    assert "model.bin" in result["corrupt_files"]
    with pytest.raises(store.ModelLockError, match="unavailable"):
        store.local_model_path(root, "asr")


def test_metadata_changes_invalidate_quick_readiness(model_fixture):
    root, lock, _ = model_fixture
    populate(model_fixture, roles=("asr",))
    store.models_report(root, action="verify", role="asr")
    path = store.model_path(root, lock["models"][0]) / "model.bin"
    path.write_bytes(path.read_bytes())
    assert store.models_report(root, role="asr")["models"][0]["status"] == "PARTIAL"


def test_git_blob_validation_pins_sha256_after_provision(model_fixture, monkeypatch):
    root, lock, contents = model_fixture
    populate(model_fixture, roles=("asr",))
    file = lock["models"][0]["files"][0]
    data = contents[("asr", file["path"])]
    file.update({"sha256": None,
                 "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
    store.write_json(root / "config/models.lock.json", lock)
    assert store.models_report(root, action="provision", role="asr")["models_ready"]
    updated = store.load_model_lock(root)["models"][0]["files"][0]
    assert updated["sha256"] == hashlib.sha256(data).hexdigest()


def test_range_download_resumes_only_the_pinned_official_artifact(model_fixture, monkeypatch):
    import huggingface_hub
    import requests

    root, lock, _ = model_fixture
    spec = lock["models"][0]
    data = b"synthetic-range-data" * 8
    file = {"path": "model.bin", "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}
    cache = root / "models/.hf-cache/asr" / spec["revision"]
    cache.mkdir(parents=True)
    partial = cache / f"model.bin.{file['sha256']}.incomplete"
    partial.write_bytes(data[:8])
    calls = []

    def metadata(url, **kwargs):
        assert spec["revision"] in url and spec["repo_id"] in url
        assert kwargs["token"] is False
        return SimpleNamespace(commit_hash=spec["revision"], size=len(data),
                               etag=file["sha256"], location="https://official.test/signed")

    class Response:
        status_code = 206
        headers = {"Content-Range": f"bytes 8-{len(data)-1}/{len(data)}"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_content(self, size):
            yield data[8:]

    def get(url, **kwargs):
        calls.append(kwargs["headers"]["Range"])
        return Response()

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(requests, "get", get)
    store._download_ranges(spec, file, root, False)
    assert calls == [f"bytes=8-{len(data)-1}"]
    assert not partial.exists()
    assert (store.model_path(root, spec) / "model.bin").read_bytes() == data


def test_offline_environment_and_missing_model_guard(model_fixture, monkeypatch):
    root, _, _ = model_fixture
    monkeypatch.setenv("HF_TOKEN", "fake-test-secret")
    environment = store.local_only_environment(root)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in environment
    assert environment["HF_HOME"] == str(root / "models/.hf-cache")
    with pytest.raises(store.ModelLockError, match="unavailable"):
        store.local_model_path(root, "asr")


def test_doctor_distinguishes_software_models_and_source_readiness(
    model_fixture, project_config, monkeypatch,
):
    from test_runtime_doctor import _mock_doctor

    root, _, _ = model_fixture
    populate(model_fixture, roles=("asr", "alignment"))
    store.models_report(root, action="verify")
    _mock_doctor(monkeypatch)
    report = doctor.build_report(project_config)
    assert report["software_ready"] is True
    assert report["models_ready"] is False
    assert report["transcription_ready"] is False
    assert report["model_readiness"]["models"][-1]["status"] == "BLOCKED_AUTH"


def test_models_cli_blocked_exit_and_secret_redaction(
    model_fixture, project_config, monkeypatch, capsys,
):
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: project_config)
    monkeypatch.setenv("HF_TOKEN", "fake-test-secret")
    assert cli.run(["models", "status", "--json"]) == 6
    output = capsys.readouterr().out
    assert "fake-test-secret" not in output
    assert json.loads(output)["models_ready"] is False


def test_model_and_cache_paths_are_git_ignored():
    root = Path(__file__).resolve().parents[1]
    for path in ("models/asr/test/model.bin", "models/.hf-cache/hub/test", ".env"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=root, check=False)
        assert result.returncode == 0


def test_remote_head_drift_is_reported_without_changing_lock(model_fixture, monkeypatch):
    import huggingface_hub

    root, _, _ = model_fixture
    path = root / "config/models.lock.json"
    before = path.read_bytes()

    def info(repo, **kwargs):
        assert repo in store.APPROVED.values()
        assert kwargs == {"revision": "main", "token": False}
        return SimpleNamespace(sha="2" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: SimpleNamespace(model_info=info))
    report = store.check_remote_revisions(root)
    assert all(model["changed"] for model in report["models"])
    assert report["lock_updated"] is False
    assert report["model_files_downloaded"] is False
    assert path.read_bytes() == before


def test_invalid_receipt_cannot_claim_readiness(model_fixture):
    root, lock, _ = model_fixture
    populate(model_fixture, roles=("asr",))
    spec = lock["models"][0]
    store.write_json(store.model_path(root, spec) / ".provision.json", {"files": ["bad"]})
    assert store.models_report(root, role="asr")["models_ready"] is False


def test_unexpected_alignment_configuration_is_corrupt(model_fixture):
    root, lock, _ = model_fixture
    populate(model_fixture, roles=("alignment",))
    spec = next(s for s in lock["models"] if s["role"] == "alignment")
    path = store.model_path(root, spec) / "config.json"
    data = b'{"model_type":"not-wav2vec2"}'
    path.write_bytes(data)
    file = next(f for f in spec["files"] if f["path"] == "config.json")
    file.update({"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    store.write_json(root / "config/models.lock.json", lock)
    result = store.models_report(root, action="verify", role="alignment")["models"][0]
    assert result["status"] == "CORRUPT"
    assert result["config_errors"] == ["UNEXPECTED_ALIGNMENT_MODEL_TYPE"]


def test_model_path_cannot_escape_into_sessions(model_fixture):
    root, lock, _ = model_fixture
    spec = dict(lock["models"][0], destination="models/../sessions")
    with pytest.raises(store.ModelLockError, match="escapes"):
        store.model_path(root, spec)


def test_all_models_ready_does_not_make_damaged_source_ready(
    model_fixture, project_config, monkeypatch,
):
    from test_runtime_doctor import _mock_doctor

    root, _, _ = model_fixture
    populate(model_fixture)
    store.models_report(root, action="verify")
    _mock_doctor(monkeypatch)
    report = doctor.build_report(project_config)
    assert report["models_ready"] is True
    assert report["transcription_ready"] is False
