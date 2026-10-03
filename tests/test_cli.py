from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import pytest

from session_pipeline import cli


def test_cli_version_and_config_show_never_show_token(
    project_config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HF_TOKEN", "not-for-output")
    monkeypatch.setattr(cli, "load_config", lambda: project_config)
    assert cli.run(["version", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "0.1.0"
    assert cli.run(["config", "show", "--json"]) == 0
    output = capsys.readouterr().out
    assert "not-for-output" not in output
    assert json.loads(output)["ml"]["language"] == "fa"


def test_cli_discover_ignores_reference_and_missing_input_exits_nonzero(
    project_root: Path, project_config, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: project_config)
    (project_root / "sessions" / "فایل.m4a").write_bytes(b"synthetic")
    (project_root / "reference_transcripts" / "old.txt").write_text(
        "not inspected", encoding="utf-8"
    )
    assert cli.run(["discover", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["count"] == 1
    assert output["sources"][0]["original_filename"] == "فایل.m4a"
    assert cli.run(["integrity", str(project_root / "sessions" / "missing.wav"), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["failure_codes"] == ["FILE_NOT_FOUND"]


def test_cli_positive_fixture_publishes_valid_artifact(
    tiny_wav: Path, project_config, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
) -> None:
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg tools unavailable")
    from session_pipeline.artifacts import read_artifact

    monkeypatch.setattr(cli, "load_config", lambda: project_config)
    assert cli.run(["integrity", str(tiny_wav), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["result"] == "PASS"
    artifact_path = project_config.application.work_path / output["session_id"] / "integrity.json"
    artifact = read_artifact(artifact_path)
    assert artifact["decode"]["success"] is True
    assert "source_path" not in artifact["source"]


def test_discover_survives_legacy_windows_console_encoding(
    project_root: Path, project_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project_root / "sessions" / "جلسه.m4a").write_bytes(b"synthetic")
    monkeypatch.setattr(cli, "load_config", lambda: project_config)
    output = io.BytesIO()
    legacy_stream = io.TextIOWrapper(output, encoding="cp1252", write_through=True)
    monkeypatch.setattr(sys, "stdout", legacy_stream)
    assert cli.run(["discover", "--json"]) == 0
    document = json.loads(output.getvalue().decode("utf-8"))
    assert document["sources"][0]["original_filename"] == "جلسه.m4a"
