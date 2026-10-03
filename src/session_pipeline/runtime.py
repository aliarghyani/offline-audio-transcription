"""Isolated, offline software probes. Never instantiate an inference model."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import traceback
import wave
from pathlib import Path
from typing import Any

PACKAGES = {
    "whisperx": "whisperx",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "torchcodec": "torchcodec",
    "faster_whisper": "faster-whisper",
    "ctranslate2": "ctranslate2",
    "pyannote.audio": "pyannote-audio",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
}
DLL_FAMILIES = ("avcodec", "avformat", "avutil", "swresample", "swscale")


def package_versions() -> dict[str, str | None]:
    result = {}
    for distribution in PACKAGES.values():
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(name, None)
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYANNOTE_METRICS_ENABLED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "DO_NOT_TRACK": "1",
    })
    return environment


def shared_libraries(directory: Path) -> dict[str, list[str]]:
    try:
        return {
            family: sorted(str(path.resolve()) for path in directory.glob(f"{family}-*.dll"))
            for family in DLL_FAMILIES
        }
    except OSError:
        # An inaccessible PATH entry is not evidence of available libraries.
        return {family: [] for family in DLL_FAMILIES}


def find_shared_runtime() -> list[dict[str, Any]]:
    """Inspect PATH and the trusted per-user Gyan WinGet installation only."""
    candidates = {Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry}
    local = os.environ.get("LOCALAPPDATA")
    if local:
        packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
        candidates.update(packages.glob("Gyan.FFmpeg.Shared_*/*/bin"))
    result = []
    for directory in sorted(candidates, key=str):
        libraries = shared_libraries(directory)
        if all(libraries.values()):
            result.append({"directory": str(directory.resolve()), "libraries": libraries})
    return result


def cpu_only_torch(details: dict[str, Any]) -> bool:
    return (
        {"version", "cuda_runtime", "cuda_available", "cuda_device_count"} <= details.keys()
        and str(details.get("version", "")).endswith("+cpu")
        and details.get("cuda_runtime") is None
        and details.get("cuda_available") is False
        and type(details.get("cuda_device_count")) is int
        and details.get("cuda_device_count") == 0
    )


def synthetic_decode() -> dict[str, Any]:
    """Decode only a generated tone, then remove the temporary directory."""
    torch = importlib.import_module("torch")
    decoder_type = importlib.import_module("torchcodec.decoders").AudioDecoder
    values = [int(8000 * math.sin(2 * math.pi * 440 * index / 16000)) for index in range(1600)]
    with tempfile.TemporaryDirectory(prefix="session-pipeline-doctor-") as temporary:
        path = Path(temporary) / "synthetic.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(struct.pack("<1600h", *values))
        decoder = decoder_type(str(path))
        samples = decoder.get_all_samples()
        expected = torch.tensor(values, dtype=torch.float32) / 32768
        shape = list(samples.data.shape)
        error = (
            (samples.data[0] - expected).abs().max().item()
            if shape == [1, 1600] else None
        )
        success = (
            shape == [1, 1600] and samples.sample_rate == 16000
            and error is not None and error <= 1 / 32768
        )
        result = {
            "status": "PASS" if success else "FAIL", "shape": shape,
            "sample_rate": samples.sample_rate, "duration_seconds": 0.1,
            "max_sample_error": error, "input": "temporary synthetic PCM16 tone",
        }
        # Release the decoder/file handle before TemporaryDirectory cleanup on Windows.
        del decoder, samples
    result["temporary_removed"] = not Path(temporary).exists()
    if not result["temporary_removed"]:
        result["status"] = "FAIL"
    return result


def whisperx_capabilities() -> dict[str, Any]:
    """Inspect mappings, signatures and source; do not call a model loader."""
    alignment = importlib.import_module("whisperx.alignment")
    diarize = importlib.import_module("whisperx.diarize")
    asr = importlib.import_module("whisperx.asr")
    backend = importlib.import_module("faster_whisper")
    speaker_parameters = inspect.signature(diarize.DiarizationPipeline.__call__).parameters
    assignment = inspect.signature(diarize.assign_word_speakers).parameters
    initializer = inspect.getsource(diarize.DiarizationPipeline.__init__)
    pipeline_source = inspect.getsource(asr.FasterWhisperPipeline.__init__)
    result = {
        "version": importlib.metadata.version("whisperx"),
        "persian_alignment_model": alignment.DEFAULT_ALIGN_MODELS_HF.get("fa"),
        "diarization_wrapper": True,
        "community_1_default": "pyannote/speaker-diarization-community-1" in initializer,
        "exact_num_speakers": "num_speakers" in speaker_parameters,
        "fill_nearest_default": assignment["fill_nearest"].default,
        "cpu_device_supported": 'torch.device("cpu")' in pipeline_source,
        "faster_whisper_backend": issubclass(asr.WhisperModel, backend.WhisperModel),
        "loaders_called": False,
    }
    success = (
        result["version"] == "3.8.6"
        and result["persian_alignment_model"] == "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
        and result["community_1_default"] and result["exact_num_speakers"]
        and result["fill_nearest_default"] is False and result["cpu_device_supported"]
        and result["faster_whisper_backend"]
    )
    result["status"] = "PASS" if success else "FAIL"
    return result


def probe_module(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    details: dict[str, Any] = {"import": "PASS"}
    if module_name == "torch":
        details.update({
            "version": module.__version__,
            "cuda_runtime": module.version.cuda,
            "cuda_available": module.cuda.is_available(),
            "cuda_device_count": module.cuda.device_count(),
        })
        details["cpu_only"] = cpu_only_torch(details)
        details["status"] = "PASS" if details["cpu_only"] else "FAIL"
    elif module_name == "ctranslate2":
        types = sorted(module.get_supported_compute_types("cpu"))
        details.update({
            "cpu_compute_types": types, "status": "PASS" if "int8" in types else "FAIL",
        })
    elif module_name == "torchcodec":
        details["functional_decode"] = synthetic_decode()
        details["status"] = details["functional_decode"]["status"]
        # Installed TorchCodec exposes this diagnostic without creating a model.
        core = importlib.import_module("torchcodec._core")
        details["ffmpeg_library_versions"] = core.get_ffmpeg_library_versions()
    elif module_name == "whisperx":
        details["capabilities"] = whisperx_capabilities()
        details["status"] = details["capabilities"]["status"]
    return details


def run_probe(module_name: str, directory: str | None = None) -> dict[str, Any]:
    command = [sys.executable, "-B", "-m", "session_pipeline.runtime", "--probe", module_name]
    if directory:
        command.extend(["--ffmpeg-dir", directory])
    try:
        completed = subprocess.run(
            command, env=offline_environment(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "FAIL", "import": "FAIL", "error": str(error)}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"status": "FAIL", "import": "FAIL", "error": "invalid probe output"}
        result["stdout"] = completed.stdout
    result["stderr"] = completed.stderr
    result["returncode"] = completed.returncode
    if completed.returncode:
        result["status"] = "FAIL"
    else:
        result.setdefault("status", "PASS")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=PACKAGES, required=True)
    parser.add_argument("--ffmpeg-dir", type=Path)
    args = parser.parse_args()
    handles = []
    try:
        if args.ffmpeg_dir and sys.platform == "win32":
            handles.append(os.add_dll_directory(str(args.ffmpeg_dir.resolve())))
        with contextlib.redirect_stdout(sys.stderr):
            result = probe_module(args.probe)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = {"status": "FAIL", "import": "FAIL", "error": f"{type(error).__name__}: {error}"}
    finally:
        for handle in handles:
            handle.close()
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(1 if result.get("status") == "FAIL" else 0)


if __name__ == "__main__":
    main()
