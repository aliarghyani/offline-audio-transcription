from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_pipeline.config import ConfigurationError, load_config
from session_pipeline.discovery import discover_sessions
from session_pipeline.models import build_session_source, session_id_from_sha256, sha256_file


def test_streaming_hash_unicode_path_and_stable_identity(tmp_path: Path) -> None:
    first = tmp_path / "جلسه یک.m4a"
    second = tmp_path / "unrelated name.m4a"
    content = (b"known test bytes" * 1000) + b"end"
    first.write_bytes(content)
    second.write_bytes(content)
    import hashlib

    expected = hashlib.sha256(content).hexdigest()
    assert sha256_file(first, chunk_size=17) == expected
    a = build_session_source(first)
    b = build_session_source(second)
    assert a.session_id == b.session_id == session_id_from_sha256(expected)
    assert a.original_filename == "جلسه یک.m4a"
    assert a.source_path == first.resolve()
    assert a.to_dict(include_path=False).get("source_path") is None


def test_discover_only_sessions_and_never_reads_reference(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = project_root / "sessions" / "فایل آزمایشی.M4A"
    session.write_bytes(b"synthetic metadata fixture")
    (project_root / "sessions" / "notes.txt").write_text("ignored", encoding="utf-8")
    reference = project_root / "reference_transcripts" / "historical.txt"
    reference.write_text("not used", encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if reference == path:
            raise AssertionError("reference transcript was read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    found = discover_sessions(project_root / "sessions")
    assert len(found) == 1
    assert found[0].original_filename == session.name


def test_configuration_defaults_and_secret_exclusion(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "never-display-this-token"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    config = load_config(project_root)
    public = json.dumps(config.public_dict(), ensure_ascii=False)
    assert config.secrets.hf_token == sentinel
    assert sentinel not in public
    assert "HF_TOKEN" not in public
    assert sentinel not in repr(config)
    assert config.ml.language == "fa"
    assert config.ml.num_speakers == 2
    assert config.ml.fill_nearest is False
    assert config.pipeline.chunk_duration_seconds is None


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("ml", "language", "en"),
        ("ml", "num_speakers", 3),
        ("ml", "fill_nearest", True),
        ("pipeline", "batch_size", -1),
        ("pipeline", "chunk_duration_seconds", 0),
        ("application", "sessions_dir", "../reference_transcripts"),
    ],
)
def test_config_rejects_invalid_values(
    project_root: Path, section: str, key: str, bad_value: object
) -> None:
    path = project_root / "config" / "defaults.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document[section][key] = bad_value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(project_root)
