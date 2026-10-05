"""Local CLI for environment, integrity, model files and ASR quality review."""

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
from session_pipeline.model_store import (
    APPROVED,
    ModelLockError,
    check_remote_revisions,
    human_models_report,
    models_report,
)
from session_pipeline.models import SourceChangedError

EXIT_SUCCESS = 0
EXIT_INPUT_INVALID = 3
EXIT_CONFIGURATION_ERROR = 4
EXIT_ENVIRONMENT_INVALID = 5
EXIT_MODELS_NOT_READY = 6


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
    align = commands.add_parser(
        "align", help="offline reviewed-transcript alignment; default is plan only"
    )
    align.add_argument("session_id")
    mode = align.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true")
    align.add_argument("--smoke", action="store_true", help="separate first-180-second lineage")
    align.add_argument("--reviewed-fingerprint", help="pin a reviewed input instead of latest")
    align.add_argument("--fingerprint", help="existing alignment plan fingerprint (status only)")
    align.add_argument("--json", action="store_true")
    models = commands.add_parser("models", help="manage pinned model files without inference")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    for name in ("status", "provision", "verify"):
        model_command = model_commands.add_parser(name)
        model_command.add_argument("--role", choices=APPROVED)
        model_command.add_argument("--json", action="store_true")
    updates = model_commands.add_parser("check-updates", help="explicit remote metadata check")
    updates.add_argument("--json", action="store_true")
    qa = commands.add_parser("qa", help="offline ASR review and separate retry candidates")
    qa_commands = qa.add_subparsers(dest="qa_command", required=True)
    review = qa_commands.add_parser("review", help="local human A/B review")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    for name in ("build", "serve", "status"):
        command = review_commands.add_parser(name)
        command.add_argument("session_id")
        command.add_argument("--retry-fingerprint")
        command.add_argument("--json", action="store_true")
        if name == "serve":
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--open", action="store_true")
    merge = qa_commands.add_parser("merge", help="merge completed human decisions into new lineage")
    merge.add_argument("session_id")
    merge.add_argument("--retry-fingerprint")
    merge.add_argument("--json", action="store_true")
    recovery = qa_commands.add_parser("recover", help="offline retry word recovery")
    recovery.add_argument("session_id")
    recovery.add_argument("--retry-fingerprint")
    recovery.add_argument(
        "--auto-prefer-b",
        action="store_true",
        help="explicitly apply automatic B choices and create reviewed output",
    )
    recovery.add_argument("--json", action="store_true")
    for name in ("asr", "status", "retry"):
        command = qa_commands.add_parser(name)
        command.add_argument("session_id")
        command.add_argument("--json", action="store_true")
        if name == "asr":
            command.add_argument("--thresholds", type=Path)
        if name == "retry":
            command.add_argument("--entry", action="append")
            command.add_argument("--max-seconds", type=float, default=900)
            command.add_argument(
                "--execute", action="store_true", help="run local inference; default is plan only"
            )
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
    if args.command == "align":
        from session_pipeline.alignment import make_plan, plan_report, run_alignment, status_report
        from session_pipeline.asr_quality import session_paths

        try:
            if args.fingerprint:
                import re

                if not args.status or not re.fullmatch(r"[0-9a-f]{64}", args.fingerprint):
                    raise ValueError("--fingerprint requires --status and a SHA-256 fingerprint")
                from session_pipeline.alignment import path_fingerprint
                from session_pipeline.artifacts import read_artifact

                _, work, _ = session_paths(config, args.session_id)
                plan_path = work / "alignment" / args.fingerprint / "plan.json"
                old = read_artifact(plan_path)
                args.reviewed_fingerprint = path_fingerprint(old)
                args.smoke = old["configuration"]["mode"] == "SMOKE"
            else:
                plan_path = None
            if args.status:
                report = status_report(
                    config,
                    args.session_id,
                    smoke=args.smoke,
                    reviewed_fingerprint=args.reviewed_fingerprint,
                    plan_path=plan_path,
                )
            elif args.execute:
                report = run_alignment(
                    config,
                    args.session_id,
                    smoke=args.smoke,
                    reviewed_fingerprint=args.reviewed_fingerprint,
                )
            else:
                path, plan = make_plan(
                    config,
                    args.session_id,
                    smoke=args.smoke,
                    reviewed_fingerprint=args.reviewed_fingerprint,
                )
                report = plan_report(config, path, plan)
            _emit(report if args.json else json.dumps(report, indent=2), json_mode=args.json)
            return (
                EXIT_SUCCESS
                if not args.execute or report["mode_processing_complete"]
                else EXIT_INPUT_INVALID
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Alignment failed: {type(error).__name__}", file=sys.stderr)
            return EXIT_INPUT_INVALID
    if args.command == "qa":
        from session_pipeline.artifacts import read_artifact
        from session_pipeline.asr_quality import create_review, load_canonical, session_paths

        try:
            if args.qa_command == "recover":
                from session_pipeline.retry_recovery import auto_merge, build_recovery

                report = build_recovery(config, args.session_id, args.retry_fingerprint)
                if args.auto_prefer_b:
                    report.update(auto_merge(config, args.session_id, args.retry_fingerprint))
                _emit(report if args.json else json.dumps(report, indent=2), json_mode=args.json)
                return EXIT_SUCCESS
            if args.qa_command == "review":
                from session_pipeline.human_review import build_bundle, review_status

                if args.review_command == "serve":
                    from session_pipeline.review_server import serve_review

                    serve_review(
                        config, args.session_id, args.retry_fingerprint, args.port, args.open
                    )
                    return EXIT_SUCCESS
                operation = build_bundle if args.review_command == "build" else review_status
                report = operation(config, args.session_id, args.retry_fingerprint)
                _emit(report if args.json else json.dumps(report, indent=2), json_mode=args.json)
                return EXIT_SUCCESS
            if args.qa_command == "merge":
                from session_pipeline.review_merge import merge_review

                report = merge_review(config, args.session_id, args.retry_fingerprint)
                _emit(report if args.json else json.dumps(report, indent=2), json_mode=args.json)
                return EXIT_SUCCESS
            if args.qa_command == "retry":
                from session_pipeline.asr_retry import launch_worker, make_plan

                path, plan = make_plan(config, args.session_id, args.entry, args.max_seconds)
                report = {
                    key: plan[key]
                    for key in (
                        "target_audio_seconds",
                        "decoded_audio_seconds",
                        "execution_allowed",
                        "execution_blockers",
                        "canonical_replacement",
                        "cloud_llm_required",
                    )
                }
                report.update(
                    plan_path=str(path),
                    window_count=len(plan["windows"]),
                    mode="execute" if args.execute else "plan_only",
                )
                _emit(report if args.json else json.dumps(report, indent=2), json_mode=args.json)
                if args.execute:
                    if not plan["execution_allowed"]:
                        return EXIT_INPUT_INVALID
                    return launch_worker(config, args.session_id, path)
                return EXIT_SUCCESS
            if args.qa_command == "asr":
                path, review = create_review(config, args.session_id, args.thresholds)
            else:
                raw_path, _, path = session_paths(config, args.session_id)
                raw = load_canonical(raw_path, args.session_id)
                review = read_artifact(path)
                if (
                    review.get("status") != "SUCCEEDED"
                    or review.get("artifact_type") != "asr_review"
                    or review.get("session_id") != args.session_id
                    or review.get("source_sha256") != raw["source_sha256"]
                    or review["parent_checksums"]["canonical_asr"] != raw["artifact_checksum"]
                ):
                    raise ValueError("review is stale")
            report = {
                key: review[key]
                for key in (
                    "segment_count",
                    "distributions",
                    "temperature_counts",
                    "repeated_text_groups",
                    "segment_flag_counts",
                    "entry_flag_counts",
                    "flagged_segment_count",
                    "proposed_retry_union_seconds",
                    "cloud_llm_required",
                )
            }
            report.update(review_path=str(path), entry_count=len(review["entries"]))
            _emit(report if args.json else json.dumps(report, indent=2), json_mode=args.json)
            return EXIT_SUCCESS
        except (OSError, ValueError, KeyError, TypeError) as error:
            from session_pipeline.human_review import ReviewError

            detail = str(error) if isinstance(error, ReviewError) else type(error).__name__
            print(f"QA failed: {detail}", file=sys.stderr)
            return EXIT_INPUT_INVALID
    if args.command == "models":
        try:
            if args.model_command == "check-updates":
                report = check_remote_revisions(config.application.root)
                _emit(report, json_mode=args.json)
                return EXIT_SUCCESS
            report = models_report(
                config.application.root,
                action=args.model_command,
                role=args.role,
            )
        except ModelLockError as error:
            print(f"Model configuration error: {error}", file=sys.stderr)
            return EXIT_CONFIGURATION_ERROR
        _emit(report if args.json else human_models_report(report), json_mode=args.json)
        return EXIT_SUCCESS if report["models_ready"] else EXIT_MODELS_NOT_READY
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
