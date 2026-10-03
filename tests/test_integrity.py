from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from session_pipeline import integrity
from session_pipeline.integrity import IntegrityFailure, inspect_audio
from session_pipeline.models import sha256_file


def test_complete_positive_decode_of_synthetic_unicode_fixture(tiny_wav: Path) -> None:
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg tools unavailable")
    initial_hash = sha256_file(tiny_wav)
    initial_stat = tiny_wav.stat()
    result = inspect_audio(tiny_wav)
    assert result.success
    assert result.failure_codes == []
    assert result.probe["duration_seconds"] == pytest.approx(0.2)
    assert result.probe["streams"][0]["codec_type"] == "audio"
    assert result.decode == {"attempted": True, "success": True, "ffmpeg_exit_code": 0}
    assert sha256_file(tiny_wav) == initial_hash
    assert tiny_wav.stat().st_mtime_ns == initial_stat.st_mtime_ns


def test_ffprobe_failure_is_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "bad.m4a"
    source.write_bytes(b"not media")
    monkeypatch.setattr(integrity.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        integrity,
        "_run_probe",
        lambda executable, path: subprocess.CompletedProcess([], 1, "", "moov atom not found"),
    )
    result = inspect_audio(source)
    assert result.failure_codes == [IntegrityFailure.FFPROBE_FAILED]
    assert result.decode["attempted"] is False
    assert result.diagnostics["probe"] == "moov atom not found"


def test_missing_audio_stream_blocks_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(integrity.shutil, "which", lambda name: name)
    output = json.dumps({"format": {"duration": "1.0"}, "streams": [{"codec_type": "video"}]})
    monkeypatch.setattr(
        integrity,
        "_run_probe",
        lambda executable, path: subprocess.CompletedProcess([], 0, output, ""),
    )
    monkeypatch.setattr(
        integrity,
        "_run_decode",
        lambda executable, path: pytest.fail("decode should not run without audio"),
    )
    result = inspect_audio(source)
    assert result.failure_codes == [IntegrityFailure.MISSING_AUDIO_STREAM]
    assert result.decode["attempted"] is False


def test_ffmpeg_decode_failure_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "broken.wav"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(integrity.shutil, "which", lambda name: name)
    output = json.dumps({"format": {"duration": "1.0"}, "streams": [{"codec_type": "audio"}]})
    monkeypatch.setattr(
        integrity,
        "_run_probe",
        lambda executable, path: subprocess.CompletedProcess([], 0, output, ""),
    )
    monkeypatch.setattr(
        integrity,
        "_run_decode",
        lambda executable, path: subprocess.CompletedProcess([], 1, None, "decode failed"),
    )
    result = inspect_audio(source)
    assert result.failure_codes == [IntegrityFailure.DECODE_FAILED]
    assert result.decode == {"attempted": True, "success": False, "ffmpeg_exit_code": 1}


def test_source_change_after_probe_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "changed.wav"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(integrity.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        integrity,
        "_run_probe",
        lambda executable, path: subprocess.CompletedProcess([], 1, "", "probe failed"),
    )
    monkeypatch.setattr(integrity, "sha256_file", lambda path: "0" * 64)
    result = inspect_audio(source)
    assert IntegrityFailure.SOURCE_CHANGED in result.failure_codes
