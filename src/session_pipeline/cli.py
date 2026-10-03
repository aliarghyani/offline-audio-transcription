"""Small local CLI for discovery, configuration, and source integrity only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from session_pipeline import __version__
from session_pipeline.artifacts import atomic_write_json, integrity_artifact
from session_pipeline.config import ConfigurationError, load_config
from session_pipeline.discovery import discover_sessions
from session_pipeline.doctor import build_report, human_report
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.integrity import inspect_audio
from session_pipeline.models import SourceChangedError

EXIT_SUCCESS = 0
EXIT_INPUT_INVALID = 3
EXIT_CONFIGURATION_ERROR = 4
EXIT_ENVIRONMENT_INVALID = 5


def _configure_console_streams() -> None:
    # Windows PowerShell may expose a legacy output encoding even when the
    # filesystem path itself is valid Unicode. UTF-8 preserves Persian paths
    # in both human and JSON output, including redirected output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def _emit(value: Any, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="session-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version", help="show the installed pipeline version")
    version.add_argument("--json", action="store_true")
    discover = commands.add_parser("discover", help="list candidate audio beneath sessions/")
    discover.add_argument("--json", action="store_true")
    integrity = commands.add_parser("integrity", help="validate one source without modifying it")
    integrity.add_argument("audio_path", type=Path)
    integrity.add_argument("--json", action="store_true")
    config = commands.add_parser("config", help="inspect non-secret configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    show = config_commands.add_parser("show", help="show effective non-secret settings")
    show.add_argument("--json", action="store_true")
    doctor = commands.add_parser("doctor", help="check ML software offline without loading models")
    doctor.add_argument("--json", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    _configure_console_streams()
    args = _parser().parse_args(argv)
    if args.command == "version":
        _emit(
            {"version": __version__} if args.json else f"session-pipeline {__version__}",
            json_mode=args.json,
        )
        return EXIT_SUCCESS
    try:
        config = load_config(include_secrets=False)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR
    if args.command == "doctor":
        report = build_report(config)
        _emit(report if args.json else human_report(report), json_mode=args.json)
        return EXIT_ENVIRONMENT_INVALID if report["software_status"] == "FAIL" else EXIT_SUCCESS
    if args.command == "config":
        public = config.public_dict()
        _emit(
            public if args.json else json.dumps(public, ensure_ascii=False, indent=2),
            json_mode=args.json,
        )
        return EXIT_SUCCESS
    if args.command == "discover":
        try:
            sources = discover_sessions(config.application.sessions_path)
        except (OSError, SourceChangedError) as error:
            print(f"Discovery failed: {type(error).__name__}", file=sys.stderr)
            return EXIT_INPUT_INVALID
        public = [source.to_dict() for source in sources]
        if args.json:
            _emit({"count": len(public), "sources": public}, json_mode=True)
        else:
            print(f"Found {len(public)} candidate audio file(s).")
            for source in sources:
                print(f"{source.session_id}  {source.source_path}  {source.size_bytes} bytes")
        return EXIT_SUCCESS
    if args.command == "integrity":
        try:
            result = inspect_audio(args.audio_path)
        except OSError as error:
            print(f"Integrity I/O error: {type(error).__name__}", file=sys.stderr)
            return EXIT_INPUT_INVALID
        if result.source is not None:
            fingerprint = stage_fingerprint(
                stage="integrity",
                source_sha256=result.source.source_sha256,
                relevant_config={},
                parent_checksums={},
                pipeline_version=__version__,
                stage_schema_version=1,
            )
            artifact = integrity_artifact(result, fingerprint)
            work_dir = config.application.work_path / result.source.session_id
            try:
                atomic_write_json(work_dir / "integrity.json", artifact)
                # Raw tool messages and absolute paths stay in ignored local work only.
                atomic_write_json(work_dir / "integrity.diagnostics.json", result.diagnostics)
            except OSError as error:
                print(f"Artifact write failed: {type(error).__name__}", file=sys.stderr)
                return EXIT_CONFIGURATION_ERROR
        else:
            artifact = None
        public = {
            "result": "PASS" if result.success else "FAIL",
            "failure_codes": [code.value for code in result.failure_codes],
            "session_id": result.source.session_id if result.source else None,
            "artifact": artifact,
        }
        if args.json:
            _emit(public, json_mode=True)
        else:
            print(public["result"])
            if result.source:
                print(f"Session: {result.source.session_id}")
            if result.failure_codes:
                print("Failure codes: " + ", ".join(public["failure_codes"]))
        return EXIT_SUCCESS if result.success else EXIT_INPUT_INVALID
    return EXIT_CONFIGURATION_ERROR


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
