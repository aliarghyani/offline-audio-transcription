"""Read-only environment report with offline, isolated ML software checks."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from session_pipeline import __version__, runtime
from session_pipeline.config import AppConfig


def executable_info(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"status": "FAIL", "path": None, "version": "UNKNOWN"}
    try:
        completed = subprocess.run(
            [path, "-version"], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15, check=False,
        )
        lines = (completed.stdout or completed.stderr).splitlines()
        return {
            "status": "PASS" if completed.returncode == 0 and lines else "FAIL",
            "path": path, "version": lines[0] if lines else "UNKNOWN",
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "FAIL", "path": path, "version": "UNKNOWN", "error": str(error)}


def shared_runtime_info() -> list[dict[str, Any]]:
    candidates = runtime.find_shared_runtime()
    for candidate in candidates:
        executable = Path(candidate["directory"]) / "ffmpeg.exe"
        try:
            completed = subprocess.run(
                [str(executable), "-version"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15, check=False,
            )
            line = completed.stdout.splitlines()[0] if completed.stdout else "UNKNOWN"
            match = re.match(r"ffmpeg version (\d+)\.", line)
            candidate["version"] = line
            candidate["compatible_major"] = bool(
                completed.returncode == 0 and match and int(match[1]) in (4, 5, 6, 7)
            )
        except (OSError, subprocess.TimeoutExpired):
            candidate.update({"version": "UNKNOWN", "compatible_major": False})
    return candidates


def build_report(config: AppConfig) -> dict[str, Any]:
    versions = runtime.package_versions()
    shared = shared_runtime_info()
    selected = next((item["directory"] for item in shared if item["compatible_major"]), None)
    probes = {}
    for module, distribution in runtime.PACKAGES.items():
        probes[module] = (
            runtime.run_probe(module, selected) if versions[distribution] is not None
            else {"status": "FAIL", "import": "FAIL", "error": "package not installed"}
        )
    executables = {name: executable_info(name) for name in ("ffmpeg", "ffprobe")}
    failed = any(probe["status"] == "FAIL" for probe in probes.values())
    failed |= any(info["status"] == "FAIL" for info in executables.values())
    expected = {
        "whisperx": "3.8.6", "torch": "2.8.0+cpu", "torchaudio": "2.8.0+cpu",
        "torchvision": "0.23.0+cpu", "torchcodec": "0.7.0",
    }
    pins_ok = all(versions[name] == version for name, version in expected.items())
    failed |= not pins_ok
    # Presence only. Never obtain the token's value or include AppConfig.secrets.
    token_present = "HF_TOKEN" in runtime.os.environ
    warnings = ["Model weights were not loaded; model access, revisions and RAM fit are untested."]
    if not token_present:
        warnings.append("HF_TOKEN absent; future gated model access requires separate preparation.")
    if not selected:
        warnings.append("No compatible shared FFmpeg directory discovered; see TorchCodec probe.")
    for name, probe in probes.items():
        if probe.get("stderr"):
            warnings.append(f"{name} emitted diagnostics; see its stderr field.")
    return {
        "schema_version": 1,
        "status": "FAIL" if failed else "WARN" if warnings else "PASS",
        "software_status": "FAIL" if failed else "PASS",
        "python_version": platform.python_version(),
        "pipeline_version": __version__,
        "package_versions": versions,
        "pinned_versions_status": "PASS" if pins_ok else "FAIL",
        "probes": probes,
        "executables": executables,
        "ffmpeg_shared_runtimes": shared,
        "selected_shared_directory": selected,
        "configured_ml": asdict(config.ml),
        "hf_token": "present" if token_present else "absent",
        "model_download_readiness": {
            "status": "WARN", "network_access": "disabled in probes",
            "gated_access": "UNKNOWN", "revisions_pinned": False,
            "downloads_authorized": False,
        },
        "models_loaded": False,
        "warnings": warnings,
    }


def human_report(report: dict[str, Any]) -> str:
    lines = [f"Environment: {report['status']} (software: {report['software_status']})"]
    lines.append(
        f"Python {report['python_version']}; session-pipeline {report['pipeline_version']}"
    )
    for module, distribution in runtime.PACKAGES.items():
        probe = report["probes"][module]
        version = report["package_versions"][distribution] or "UNKNOWN"
        lines.append(f"{probe['status']}: {distribution} {version}")
        for key in ("error", "cuda_runtime", "cuda_available", "cuda_device_count",
                    "cpu_compute_types", "functional_decode", "capabilities",
                    "ffmpeg_library_versions"):
            if key in probe:
                lines.append(f"  {key}: {probe[key]}")
        if probe.get("stderr"):
            lines.append("  diagnostics:\n" + probe["stderr"])
    for name, info in report["executables"].items():
        lines.append(f"{info['status']}: {name} {info['path']}\n  {info['version']}")
    lines.append(f"Shared FFmpeg directory: {report['selected_shared_directory'] or 'UNKNOWN'}")
    lines.append(f"Configured ML: {report['configured_ml']}")
    lines.append(f"HF_TOKEN: {report['hf_token']}; models loaded: {report['models_loaded']}")
    lines.append(f"Model download readiness: {report['model_download_readiness']}")
    lines.extend(f"WARN: {warning}" for warning in report["warnings"])
    return "\n".join(lines)
