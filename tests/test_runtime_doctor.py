from __future__ import annotations

import ast
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from session_pipeline import cli, doctor, runtime
from session_pipeline.config import SecretSettings, load_config


def test_package_metadata_handles_missing_distribution(monkeypatch):
    def version(name):
        if name == "whisperx":
            return "3.8.6"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    versions = runtime.package_versions()
    assert versions["whisperx"] == "3.8.6"
    assert versions["torch"] is None


@pytest.mark.parametrize("override", [
    {}, {"version": "2.8.0"}, {"version": "2.8.0+cu128"},
    {"cuda_runtime": "12.8"}, {"cuda_available": True}, {"cuda_device_count": 1},
])
def test_cpu_only_requires_build_and_all_cuda_checks(override):
    details = {
        "version": "2.8.0+cpu", "cuda_runtime": None,
        "cuda_available": False, "cuda_device_count": 0,
    }
    details.update(override)
    assert runtime.cpu_only_torch(details) is (not override)


def test_incomplete_torch_diagnostics_cannot_pass():
    assert runtime.cpu_only_torch({
        "version": "2.8.0+cpu", "cuda_available": False, "cuda_device_count": 0,
    }) is False


@pytest.mark.parametrize("types,expected", [({"int8", "float32"}, "PASS"), ({"float32"}, "FAIL")])
def test_ctranslate_requires_cpu_int8(monkeypatch, types, expected):
    calls = []

    def supported(device):
        calls.append(device)
        return types

    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: SimpleNamespace(
        get_supported_compute_types=supported,
    ))
    assert runtime.probe_module("ctranslate2")["status"] == expected
    assert calls == ["cpu"]


def test_shared_detection_requires_all_dll_families(tmp_path, monkeypatch):
    for family in runtime.DLL_FAMILIES[:-1]:
        (tmp_path / f"{family}-1.dll").write_bytes(b"test fixture")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert runtime.find_shared_runtime() == []
    (tmp_path / "swscale-1.dll").write_bytes(b"test fixture")
    candidates = runtime.find_shared_runtime()
    assert len(candidates) == 1
    assert all(candidates[0]["libraries"].values())


def test_inaccessible_path_is_not_positive_dll_evidence(monkeypatch, tmp_path):
    def denied(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "glob", denied)
    assert not any(runtime.shared_libraries(tmp_path).values())


def test_probe_environment_removes_tokens_and_disables_network_telemetry(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-test-secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "another-fake-secret")
    environment = runtime.offline_environment()
    assert "HF_TOKEN" not in environment
    assert "HUGGING_FACE_HUB_TOKEN" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYANNOTE_METRICS_ENABLED"] == "0"
    assert runtime.os.environ["HF_TOKEN"] == "fake-test-secret"


def test_failed_torchcodec_probe_preserves_diagnostics(monkeypatch):
    def run(command, **kwargs):
        assert command[1:4] == ["-B", "-m", "session_pipeline.runtime"]
        assert command[4:6] == ["--probe", "torchcodec"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return subprocess.CompletedProcess(
            command, 1, json.dumps({"import": "FAIL", "error": "missing DLL"}),
            "original traceback and warning",
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)
    result = runtime.run_probe("torchcodec", "C:/trusted/bin")
    assert result["status"] == "FAIL"
    assert result["error"] == "missing DLL"
    assert result["stderr"] == "original traceback and warning"


def test_probe_timeout_is_a_failure(monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    assert runtime.run_probe("torch")["status"] == "FAIL"


def test_whisperx_checks_inspect_software_without_constructing_models(monkeypatch):
    class Backend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("must not load a model")

    class Diarization:
        def __init__(self):
            raise AssertionError("must not load a model")

        def __call__(self, audio, num_speakers=None):
            raise AssertionError("must not diarize")

    def assign(audio, fill_nearest=False):
        raise AssertionError("must not assign words")

    modules = {
        "whisperx.alignment": SimpleNamespace(DEFAULT_ALIGN_MODELS_HF={
            "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
        }),
        "whisperx.diarize": SimpleNamespace(
            DiarizationPipeline=Diarization, assign_word_speakers=assign,
        ),
        "whisperx.asr": SimpleNamespace(WhisperModel=Backend, FasterWhisperPipeline=Backend),
        "faster_whisper": SimpleNamespace(WhisperModel=Backend),
    }
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _: "3.8.6")
    monkeypatch.setattr(runtime.inspect, "getsource", lambda method: (
        'pyannote/speaker-diarization-community-1' if method is Diarization.__init__
        else 'torch.device("cpu")'
    ))
    result = runtime.whisperx_capabilities()
    assert result["status"] == "PASS"
    assert result["loaders_called"] is False


def _mock_doctor(monkeypatch):
    versions = dict.fromkeys(runtime.PACKAGES.values(), "test-version")
    versions.update({
        "whisperx": "3.8.6", "torch": "2.8.0+cpu", "torchaudio": "2.8.0+cpu",
        "torchvision": "0.23.0+cpu", "torchcodec": "0.7.0",
    })
    monkeypatch.setattr(runtime, "package_versions", lambda: versions)
    monkeypatch.setattr(runtime, "run_probe", lambda *args: {"status": "PASS", "stderr": ""})
    monkeypatch.setattr(doctor, "shared_runtime_info", lambda: [{
        "directory": "C:/trusted/bin", "compatible_major": True,
    }])
    monkeypatch.setattr(doctor, "executable_info", lambda name: {
        "status": "PASS", "path": f"C:/tools/{name}.exe", "version": "fixture",
    })
    return versions


def test_doctor_serialization_presence_only_and_no_model_claim(project_config, monkeypatch):
    _mock_doctor(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "fake-test-secret")
    report = doctor.build_report(project_config)
    serialized = json.dumps(report)
    assert "fake-test-secret" not in serialized + doctor.human_report(report)
    assert report["hf_token"] == "present"
    assert report["models_loaded"] is False
    assert report["software_status"] == "PASS"
    assert report["status"] == "WARN"
    assert report["model_download_readiness"]["gated_access"] == "UNKNOWN"


def test_doctor_missing_dependency_is_failure(project_config, monkeypatch):
    versions = _mock_doctor(monkeypatch)
    versions["torchcodec"] = None
    report = doctor.build_report(project_config)
    assert report["software_status"] == "FAIL"
    assert report["probes"]["torchcodec"]["error"] == "package not installed"


def test_current_cli_does_not_obtain_token_values(project_root, monkeypatch, capsys):
    def forbidden(cls):
        raise AssertionError("current CLI must not request secret values")

    monkeypatch.setattr(SecretSettings, "from_environment", classmethod(forbidden))
    config = load_config(project_root, include_secrets=False)
    assert config.secrets.hf_token is None
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: load_config(project_root, **kwargs))
    assert cli.run(["config", "show", "--json"]) == 0
    assert "secrets" not in json.loads(capsys.readouterr().out)


def test_doctor_cli_failure_exit_and_json(project_config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: project_config)
    monkeypatch.setattr(cli, "build_report", lambda config: {"software_status": "FAIL"})
    assert cli.run(["doctor", "--json"]) == 5
    assert json.loads(capsys.readouterr().out)["software_status"] == "FAIL"


def test_validation_source_has_no_inference_or_download_calls():
    forbidden = {
        "from_pretrained", "load_model", "load_align_model", "snapshot_download",
        "hf_hub_download", "transcribe", "align", "WhisperModel", "DiarizationPipeline",
    }
    for module in (runtime, doctor):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert not calls.intersection(forbidden)
