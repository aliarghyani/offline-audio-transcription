from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from session_pipeline.config import load_config


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    original = Path(__file__).resolve().parents[1] / "config" / "defaults.json"
    document = json.loads(original.read_text(encoding="utf-8"))
    (config_dir / "defaults.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "sessions").mkdir()
    (tmp_path / "reference_transcripts").mkdir()
    return tmp_path


@pytest.fixture
def project_config(project_root: Path):
    return load_config(project_root)


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    path = tmp_path / "نمونه صوتی.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 3200)
    return path
