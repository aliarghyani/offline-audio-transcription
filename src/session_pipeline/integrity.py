"""Read-only source validation by hashing, probing, and full decode to null."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from session_pipeline.models import (
    SessionSource,
    SourceChangedError,
    build_session_source,
    sha256_file,
)

PROBE_TIMEOUT_SECONDS = 120
DECODE_TIMEOUT_SECONDS = 4 * 60 * 60
DIAGNOSTIC_LIMIT = 4096


class IntegrityFailure(StrEnum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NOT_A_FILE = "NOT_A_FILE"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    FFPROBE_UNAVAILABLE = "FFPROBE_UNAVAILABLE"
    FFPROBE_FAILED = "FFPROBE_FAILED"
    PROBE_INVALID_JSON = "PROBE_INVALID_JSON"
    MISSING_AUDIO_STREAM = "MISSING_AUDIO_STREAM"
    FFMPEG_UNAVAILABLE = "FFMPEG_UNAVAILABLE"
    DECODE_FAILED = "DECODE_FAILED"
    PROCESS_TIMEOUT = "PROCESS_TIMEOUT"


@dataclass
class IntegrityResult:
    source: SessionSource | None = None
    probe: dict[str, Any] = field(
        default_factory=lambda: {
            "success": False,
            "format_name": None,
            "duration_seconds": None,
            "streams": [],
        }
    )
    decode: dict[str, Any] = field(
        default_factory=lambda: {"attempted": False, "success": False, "ffmpeg_exit_code": None}
    )
    failure_codes: list[IntegrityFailure] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.failure_codes and self.probe["success"] and self.decode["success"]


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value) if value is not None else None
        return number if number is not None and math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def _normalized_probe(document: dict[str, Any]) -> dict[str, Any]:
    raw_format = document.get("format") or {}
    raw_streams = document.get("streams") or []
    if not isinstance(raw_format, dict) or not isinstance(raw_streams, list):
        raise ValueError("malformed ffprobe document")
    streams = []
    for raw in raw_streams:
        if not isinstance(raw, dict):
            raise ValueError("malformed ffprobe stream")
        streams.append(
            {
                "index": _int_or_none(raw.get("index")),
                "codec_type": raw.get("codec_type"),
                "codec_name": raw.get("codec_name"),
                "sample_rate": _int_or_none(raw.get("sample_rate")),
                "channels": _int_or_none(raw.get("channels")),
                "channel_layout": raw.get("channel_layout"),
                "bit_rate": _int_or_none(raw.get("bit_rate")),
            }
        )
    return {
        "success": True,
        "format_name": raw_format.get("format_name"),
        "duration_seconds": _float_or_none(raw_format.get("duration")),
        "streams": streams,
    }


def _run_probe(executable: str, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            executable,
            "-hide_banner",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=PROBE_TIMEOUT_SECONDS,
        check=False,
    )


def _run_decode(executable: str, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            executable,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=DECODE_TIMEOUT_SECONDS,
        check=False,
    )


def inspect_audio(path: Path) -> IntegrityResult:
    """Perform every safe gate possible; never write to the source or a media output."""
    result = IntegrityResult()
    if not path.exists():
        result.failure_codes.append(IntegrityFailure.FILE_NOT_FOUND)
        return result
    if not path.is_file():
        result.failure_codes.append(IntegrityFailure.NOT_A_FILE)
        return result
    try:
        result.source = build_session_source(path)
    except (SourceChangedError, FileNotFoundError):
        result.failure_codes.append(IntegrityFailure.SOURCE_CHANGED)
        return result
    try:
        initial_mtime_ns = result.source.source_path.stat().st_mtime_ns
    except OSError:
        result.failure_codes.append(IntegrityFailure.SOURCE_CHANGED)
        return result

    try:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            result.failure_codes.append(IntegrityFailure.FFPROBE_UNAVAILABLE)
            return result
        try:
            process = _run_probe(ffprobe, result.source.source_path)
        except subprocess.TimeoutExpired:
            result.failure_codes.append(IntegrityFailure.PROCESS_TIMEOUT)
            result.diagnostics["probe"] = "ffprobe timed out"
            return result
        result.diagnostics["probe"] = (process.stderr or "")[-DIAGNOSTIC_LIMIT:]
        result.diagnostics["ffprobe_exit_code"] = process.returncode
        if process.returncode != 0:
            result.failure_codes.append(IntegrityFailure.FFPROBE_FAILED)
            return result
        try:
            document = json.loads(process.stdout)
            result.probe = _normalized_probe(document)
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            result.failure_codes.append(IntegrityFailure.PROBE_INVALID_JSON)
            return result
        if not any(stream["codec_type"] == "audio" for stream in result.probe["streams"]):
            result.failure_codes.append(IntegrityFailure.MISSING_AUDIO_STREAM)
            return result

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            result.failure_codes.append(IntegrityFailure.FFMPEG_UNAVAILABLE)
            return result
        result.decode["attempted"] = True
        try:
            decoded = _run_decode(ffmpeg, result.source.source_path)
        except subprocess.TimeoutExpired:
            result.failure_codes.append(IntegrityFailure.PROCESS_TIMEOUT)
            result.diagnostics["decode"] = "ffmpeg decode timed out"
            return result
        result.diagnostics["decode"] = (decoded.stderr or "")[-DIAGNOSTIC_LIMIT:]
        result.decode["ffmpeg_exit_code"] = decoded.returncode
        result.decode["success"] = decoded.returncode == 0
        if decoded.returncode != 0:
            result.failure_codes.append(IntegrityFailure.DECODE_FAILED)
    finally:
        if result.source is not None:
            try:
                current = result.source.source_path.stat()
                changed = (
                    current.st_size != result.source.size_bytes
                    or current.st_mtime_ns != initial_mtime_ns
                    or sha256_file(result.source.source_path) != result.source.source_sha256
                )
            except OSError:
                changed = True
            if changed and IntegrityFailure.SOURCE_CHANGED not in result.failure_codes:
                result.failure_codes.append(IntegrityFailure.SOURCE_CHANGED)
    return result
