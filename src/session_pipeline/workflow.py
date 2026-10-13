"""Sequential, offline orchestration. Stage engines own their inference/checkpoints."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import time
from pathlib import Path

from session_pipeline import __version__
from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    integrity_artifact,
    read_artifact,
    seal_artifact,
)
from session_pipeline.asr_quality import load_canonical, session_paths
from session_pipeline.asr_retry import DECODE, exclusive_lock
from session_pipeline.discovery import SUPPORTED_AUDIO_EXTENSIONS, discover_sessions
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.integrity import inspect_audio
from session_pipeline.model_store import inspect_model, load_model_lock
from session_pipeline.models import build_session_source, sha256_file
from session_pipeline.product_defaults import (
    DEFAULT_PROFILE,
    DEFAULT_REVIEW_POLICY,
    OFFLINE,
    RETRY_BUDGET_SECONDS,
    REVIEW_POLICIES,
)
from session_pipeline.review_merge import write_text_atomic

STAGES = (
    "INGEST",
    "INTEGRITY",
    "ASR",
    "ASR_QA",
    "ASR_RETRY",
    "REVIEW",
    "ALIGNMENT",
    "DIARIZATION",
    "ATTRIBUTION",
    "IDENTITY_MAPPING",
    "FINAL_RENDER",
)
PROFILES = {"fast": STAGES[:3], "quality": STAGES[:7], "full": STAGES}
COMPLETE = {"SUCCEEDED", "SUCCEEDED_WITH_WARNINGS", "SKIPPED"}
MODEL_STAGES = {
    "ASR": "asr",
    "ASR_RETRY": "asr",
    "ALIGNMENT": "alignment",
    "DIARIZATION": "diarization",
}


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def receipt_path(config, sid):
    return session_paths(config, sid)[1] / "workflow.json"


def resolve_session(value):
    if re.fullmatch(r"s-[0-9a-f]{20}", str(value)):
        return str(value)
    return build_session_source(Path(value)).session_id


def audio_inputs(path):
    path = Path(path)
    if path.is_dir():
        # Deduplicate identical files deterministically, including ingested copies.
        sources = discover_sessions(path)
        unique = {}
        for source in sources:
            unique.setdefault(source.source_sha256, source.source_path)
        return list(unique.values())
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError("input must be an existing supported audio file or directory")
    return [path.resolve()]


def bindings(config, result):
    """Commit every output/evidence file returned by the adapter, not just a state flag."""
    paths = list(result.get("artifacts", []))
    paths += [result[k] for k in ("json_path", "txt_path") if result.get(k)]
    return [
        {
            "path": str(Path(p).resolve().relative_to(config.application.root)),
            "sha256": sha256_file(Path(p)),
        }
        for p in sorted(set(paths))
    ]


def verified(config, entry, fingerprint):
    if entry.get("fingerprint") != fingerprint or entry.get("state") not in COMPLETE:
        return False
    # Success without evidence is never a reusable expensive-stage checkpoint.
    if not entry.get("bindings"):
        return False
    try:
        for binding in entry["bindings"]:
            path = (config.application.root / binding["path"]).resolve()
            if not path.is_relative_to(config.application.root.resolve()):
                return False
            if sha256_file(path) != binding["sha256"]:
                return False
            if path.suffix == ".json":
                read_artifact(path)
        return True
    except (OSError, ValueError):
        return False


def settings(config, stage, policy):
    """Only relevant settings enter each stage; upgrading profiles reuses ASR."""
    if stage in MODEL_STAGES:
        role = MODEL_STAGES[stage]
        spec = next(
            m for m in load_model_lock(config.application.root)["models"] if m["role"] == role
        )
        from session_pipeline import alignment, diarization
        from session_pipeline.asr_retry import package_identity

        engine_settings = {
            "ASR": DECODE,
            "ASR_RETRY": DECODE,
            "ALIGNMENT": alignment.SETTINGS,
            "DIARIZATION": diarization.SETTINGS,
        }
        packages = (
            alignment.packages()
            if stage == "ALIGNMENT"
            else diarization.configuration(False)["package_versions"]
            if stage == "DIARIZATION"
            else package_identity()
        )
        return {
            "model": spec,
            "settings": engine_settings[stage],
            "packages": packages,
            "budget_seconds": RETRY_BUDGET_SECONDS if stage == "ASR_RETRY" else None,
        }
    if stage == "ASR_QA":
        return json.loads((config.application.root / "config/asr-quality.json").read_text("utf-8"))
    if stage == "REVIEW":
        return {"review_policy": policy, "version": 1}
    if stage == "ATTRIBUTION":
        from session_pipeline.attribution import POLICY

        return POLICY
    if stage == "FINAL_RENDER":
        from session_pipeline.final_render import POLICY

        return POLICY
    return {"version": 1}


class LocalStages:
    """Thin adapters. No ML imports occur during planning/status."""

    def __init__(self, config, source, sid, policy):
        self.config, self.source, self.sid, self.policy = config, source, sid, policy
        self.raw, self.work, self.qa = session_paths(config, sid)

    def model_ready(self, role):
        spec = next(
            m for m in load_model_lock(self.config.application.root)["models"] if m["role"] == role
        )
        return inspect_model(self.config.application.root, spec)["status"] == "PROVISIONED"

    def inference_needed(self, stage, results):
        """Completed engine checkpoints can be published even after models are removed."""
        if stage == "ASR":
            from session_pipeline.raw_asr import identity

            integrity = read_artifact(self.work / "integrity.json")
            fp = identity(self.config, integrity)
            native = self.work / "asr" / fp / "raw-inference.json"
            return not (self.raw.exists() or native.exists())
        if stage == "ASR_RETRY":
            from session_pipeline.human_review import load_context

            path, plan, _ = self.retry_plan()
            if not plan["windows"]:
                return False
            try:
                load_context(self.config, self.sid, path.parent.name)
                return False
            except (OSError, ValueError, KeyError, TypeError):
                return True
        if stage == "ALIGNMENT":
            from session_pipeline.alignment import status_report

            report = status_report(
                self.config,
                self.sid,
                reviewed_fingerprint=results["REVIEW"]["reviewed_fingerprint"],
            )
            return not report.get("full_alignment_complete")
        if stage == "DIARIZATION":
            from session_pipeline.diarization import status_report

            report = status_report(
                self.config,
                self.sid,
                aligned_fingerprint=results["ALIGNMENT"]["aligned_fingerprint"],
            )
            return not (
                report.get("full_diarization_complete") or report.get("raw_inference_persisted")
            )
        return True

    def run(self, stage, results):
        operation = getattr(self, stage.lower())
        return operation(results)

    def ingest(self, _):
        source = build_session_source(self.source)
        if source.session_id != self.sid:
            raise ValueError("source changed during ingest")
        sessions = self.config.application.sessions_path.resolve()
        sessions.mkdir(parents=True, exist_ok=True)
        path = self.work / "ingest.json"
        if path.exists():
            old = read_artifact(path)
            target = Path(old["staged_source"])
            if (
                old["source_sha256"] == source.source_sha256
                and target.resolve().parent == sessions
                and target.is_file()
                and sha256_file(target) == source.source_sha256
            ):
                return {"state": "SUCCEEDED", "source": str(target), "artifacts": [str(path)]}
        # Existing stage contracts use a direct child of sessions. External/nested
        # audio gets an automatic content-addressed copy; the supplied file is untouched.
        target = source.source_path
        if target.parent != sessions:
            target = sessions / (source.source_sha256 + source.source_path.suffix.lower())
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".ingest.tmp")
                try:
                    shutil.copyfile(source.source_path, temporary)
                    if sha256_file(temporary) != source.source_sha256:
                        raise ValueError("source changed during ingest copy")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        if sha256_file(target) != source.source_sha256:
            raise ValueError("ingested source changed")
        document = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "workflow_ingest",
                "status": "SUCCEEDED",
                "session_id": self.sid,
                "source_sha256": source.source_sha256,
                "original_filename": source.original_filename,
                "original_path": str(source.source_path),
                "staged_source": str(target),
            }
        )
        if not path.exists() or read_artifact(path) != document:
            atomic_write_json(path, document)
        return {"state": "SUCCEEDED", "source": str(target), "artifacts": [str(path)]}

    def integrity(self, results):
        path = self.work / "integrity.json"
        source = Path(results["INGEST"]["source"])
        sha = sha256_file(source)
        if path.exists():
            old = read_artifact(path)
            if old["status"] == "SUCCEEDED" and old["source"]["sha256"] == sha:
                return {
                    "state": "SUCCEEDED",
                    "duration": old["probe"]["duration_seconds"],
                    "artifacts": [str(path)],
                }
        result = inspect_audio(source)
        if not result.source or result.source.session_id != self.sid:
            raise ValueError("audio changed or cannot be identified")
        fp = stage_fingerprint(
            stage="integrity",
            source_sha256=sha,
            relevant_config={},
            parent_checksums={},
            pipeline_version=__version__,
            stage_schema_version=1,
        )
        atomic_write_json(path, integrity_artifact(result, fp))
        return {
            "state": "SUCCEEDED" if result.success else "FAILED",
            "duration": result.probe.get("duration_seconds"),
            "warnings": [code.value for code in result.failure_codes],
            "artifacts": [str(path)],
        }

    def asr(self, _):
        from session_pipeline.raw_asr import execute

        return execute(self.config, self.sid)

    def asr_qa(self, _):
        from session_pipeline.asr_quality import create_review

        path, review = create_review(self.config, self.sid)
        flags = review["flagged_segment_count"]
        entries = len(review["entries"])
        return {
            "state": "SUCCEEDED_WITH_WARNINGS" if entries else "SUCCEEDED",
            "warnings": [f"{flags} ASR segments; {entries} entries flagged for triage"]
            if entries
            else [],
            "artifacts": [str(path)],
        }

    def asr_retry(self, _):
        from session_pipeline.asr_retry import run_retry

        path, plan, omitted = self.retry_plan()
        run_retry(self.config, self.sid, path)
        return {
            "state": "SUCCEEDED_WITH_WARNINGS" if omitted else "SUCCEEDED",
            "retry_fingerprint": path.parent.name,
            "warnings": [f"{omitted} triage entries retained outside bounded retry budget"]
            if omitted
            else [],
            "artifacts": [
                str(path),
                str(path.parent / "progress.json"),
                *[str(p) for p in sorted(path.parent.glob("candidate-*.json"))],
            ],
        }

    def retry_plan(self):
        from session_pipeline.asr_retry import make_plan

        # Select a deterministic bounded prefix. Never expand to a full ASR rerun.
        review = read_artifact(self.qa)
        selected = []
        for entry in review["entries"]:
            _, trial = make_plan(
                self.config, self.sid, [*selected, entry["entry_id"]], RETRY_BUDGET_SECONDS
            )
            if trial["execution_allowed"]:
                selected.append(entry["entry_id"])
        path, plan = make_plan(self.config, self.sid, selected, RETRY_BUDGET_SECONDS)
        omitted = len(review["entries"]) - len(selected)
        return path, plan, omitted

    def review(self, results):
        fp = results["ASR_RETRY"]["retry_fingerprint"]
        retry_plan = read_artifact(self.work / "qa/retries" / fp / "plan.json")
        if self.policy == "auto-prefer-retry" and retry_plan["windows"]:
            from session_pipeline.retry_recovery import auto_merge, build_recovery

            build_recovery(self.config, self.sid, fp)
            report = auto_merge(self.config, self.sid, fp)
        else:
            report = reviewed_policy(self.config, self.sid, fp, self.policy)
            if report.get("state") == "WAITING_FOR_USER":
                return report
        path = Path(report["reviewed_json"])
        reviewed = read_artifact(path)
        return {
            "state": "SUCCEEDED",
            "reviewed_fingerprint": path.parent.name,
            "json_path": str(path),
            "txt_path": str(path.with_suffix(".txt")),
            "human_verified": reviewed["qa_acceptance"].get("human_verified", False),
        }

    def alignment(self, results):
        from session_pipeline.alignment import finalize_alignment, run_alignment

        report = finalize_alignment(
            self.config, self.sid, reviewed_fingerprint=results["REVIEW"]["reviewed_fingerprint"]
        )
        if not report.get("full_alignment_complete"):
            report = run_alignment(
                self.config,
                self.sid,
                reviewed_fingerprint=results["REVIEW"]["reviewed_fingerprint"],
            )
        if not report["full_alignment_complete"] or not report.get("aligned_json"):
            return {
                "state": "FAILED",
                "warnings": ["alignment incomplete; completed units retained"],
                "warning_counts": alignment_warning_counts(report),
            }
        path = Path(report["aligned_json"])
        txt = Path(report["aligned_txt"])
        warning = report["alignment_has_warnings"]
        return {
            "state": "SUCCEEDED_WITH_WARNINGS" if warning else "SUCCEEDED",
            "aligned_fingerprint": path.parent.name,
            "json_path": str(path),
            "txt_path": str(txt),
            "warning_counts": alignment_warning_counts(report),
            "warnings": ["alignment contains partial/unaligned words; original text preserved"]
            if warning
            else [],
        }

    def diarization(self, results):
        from session_pipeline.diarization import run_diarization

        report = run_diarization(
            self.config, self.sid, aligned_fingerprint=results["ALIGNMENT"]["aligned_fingerprint"]
        )
        return {
            "state": report["state"],
            "diarization_fingerprint": report["fingerprint"],
            "artifacts": [report["output_path"]] if report.get("output_path") else [],
            "warnings": [f"diarization attribution warnings: {report['attribution_warnings']}"]
            if report.get("state") == "SUCCEEDED_WITH_WARNINGS"
            else [],
        }

    def attribution(self, results):
        from session_pipeline.attribution import execute

        report = execute(
            self.config,
            self.sid,
            results["ALIGNMENT"]["aligned_fingerprint"],
            results["DIARIZATION"]["diarization_fingerprint"],
        )
        return {
            "state": report["state"],
            "attributed_fingerprint": report["fingerprint"],
            "artifacts": [report["artifact_path"]],
            "warnings": [f"{report['unassigned_words']} words have unknown speaker"]
            if report["unassigned_words"]
            else [],
        }

    def identity_mapping(self, results):
        from session_pipeline import speaker_review

        fp = results["ATTRIBUTION"]["attributed_fingerprint"]
        directory = speaker_review.bundle_directory(self.config, self.sid, fp)
        if not (directory / "bundle.json").exists():
            speaker_review.build_bundle(self.config, self.sid, fp)
        _, bundle, _ = speaker_review.load_bundle(self.config, self.sid, fp)
        mapping = speaker_review.load_mapping(directory, bundle)
        if mapping is None:
            return {
                "state": "WAITING_FOR_USER",
                "gate": "SPEAKER_IDENTITY",
                "review_command": f"uv run --no-sync session-pipeline review {self.sid} --open",
            }
        return {
            "state": "SUCCEEDED",
            "mapping_status": "HUMAN_CONFIRMED",
            "artifacts": [
                str(speaker_review.mapping_path(directory)),
                str(directory / "bundle.json"),
                *[str(directory / s["filename"]) for s in bundle["samples"]],
            ],
        }

    def final_render(self, results):
        from session_pipeline.final_render import execute

        report = execute(self.config, self.sid, results["ATTRIBUTION"]["attributed_fingerprint"])
        return {
            "state": report["state"],
            "json_path": report["json_path"],
            "txt_path": report["txt_path"],
            "warnings": report["warnings"],
        }


def reviewed_policy(config, sid, retry_fp, policy):
    """Reuse human merge or retain A, then publish a separate structural acceptance lineage."""
    import copy

    from session_pipeline import human_review, review_merge

    raw_path, work, qa_path = session_paths(config, sid)
    raw = load_canonical(raw_path, sid)
    plan_path = work / "qa/retries" / retry_fp / "plan.json"
    plan = read_artifact(plan_path)
    has_targets = bool(plan["windows"])
    if has_targets:
        context = human_review.load_context(config, sid, retry_fp)
        parents = context.parents.copy()
    else:
        from session_pipeline.asr_retry import make_plan

        expected_path, expected = make_plan(config, sid, [], RETRY_BUDGET_SECONDS)
        progress = read_artifact(plan_path.parent / "progress.json")
        if (
            plan_path != expected_path
            or plan != expected
            or progress["status"] != "SUCCEEDED"
            or not progress["completed_windows"] == progress["expected_windows"] == 0
        ):
            raise ValueError("empty retry lineage is invalid")
        parents = {
            "canonical_asr": raw["artifact_checksum"],
            "review": read_artifact(qa_path)["artifact_checksum"],
            "plan": plan["artifact_checksum"],
            "progress": progress["artifact_checksum"],
        }
    human_verified = policy == "human" and has_targets
    if human_verified:
        if not (context.directory / "bundle.json").exists():
            human_review.build_bundle(config, sid, retry_fp)
        bundle = human_review.load_bundle(context)
        decisions = human_review.load_decisions(context, bundle)
        if any(d["choice"] in {"UNREVIEWED", "UNCLEAR"} for d in decisions["decisions"].values()):
            return {
                "state": "WAITING_FOR_USER",
                "gate": "ASR_REVIEW",
                "review_command": f"uv run --no-sync session-pipeline review {sid} --open",
            }
        merged = review_merge.merge_review(config, sid, retry_fp)
        base_path = Path(merged["reviewed_json"])
        base = read_artifact(base_path)
        parents["human_merge"] = base["artifact_checksum"]
        segments = copy.deepcopy(base["segments"])
    else:
        segments = copy.deepcopy(raw["segments"])
    duration = raw["probe"]["duration_seconds"]
    if not segments:
        raise ValueError("no speech segments to align")
    previous = 0.0
    for s in segments:
        if not previous <= s["start"] <= s["end"] <= duration:
            raise ValueError("reviewed timeline structurally invalid")
        previous = s["start"]
    if segments[0]["start"] > raw["segments"][0]["start"] or max(s["end"] for s in segments) < max(
        s["end"] for s in raw["segments"]
    ):
        raise ValueError("review lost session extent")
    fp = stage_fingerprint(
        stage="product_review",
        source_sha256=raw["source_sha256"],
        relevant_config={"policy": policy, "version": 1},
        parent_checksums=parents,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    path = raw_path.parent / "reviewed" / fp / "transcript.reviewed.fa.json"
    text = review_merge.render_reviewed(segments, [])
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "reviewed_asr",
            "status": "SUCCEEDED",
            "session_id": sid,
            "source_sha256": raw["source_sha256"],
            "source": raw["source"],
            "probe": raw["probe"],
            "model": raw["model"],
            "stage_fingerprint": fp,
            "provenance": {
                "review_policy": policy,
                "human_verified": human_verified,
                "parent_checksums": parents,
            },
            "segments": segments,
            "segment_count": len(segments),
            "unresolved": [],
            "rendered_txt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "qa_acceptance": {
                "status": "STRUCTURAL_PASS",
                "alignment_ready": True,
                "human_verified": human_verified,
                "speech_accuracy_verified": False,
            },
        }
    )
    if path.exists() and read_artifact(path) != document:
        raise ValueError("reviewed lineage changed")
    if not path.exists():
        write_text_atomic(path.with_suffix(".txt"), text)
        atomic_write_json(path, document)
    return {"reviewed_json": str(path)}


def transcribe_one(
    config,
    audio,
    *,
    profile=DEFAULT_PROFILE,
    policy=DEFAULT_REVIEW_POLICY,
    dry_run=False,
    adapter_factory=LocalStages,
    progress=None,
):
    if profile not in PROFILES or policy not in REVIEW_POLICIES:
        raise ValueError("invalid profile or review policy")
    if OFFLINE is not True:
        raise ValueError("only offline transcription is supported")
    source = build_session_source(Path(audio))
    historical = (config.application.root / "reference_transcripts").resolve()
    if source.source_path.is_relative_to(historical):
        raise ValueError("historical reference directory is not a transcription input")
    sid = source.session_id
    path = receipt_path(config, sid)
    report = {
        "schema_version": 1,
        "artifact_type": "transcription_workflow",
        "session_id": sid,
        "source_sha256": source.source_sha256,
        "profile": profile,
        "review_policy": policy,
        "state": "PENDING",
        "source_duration_seconds": None,
        "stages": {},
        "warnings": [],
        "cloud_llm_required": False,
        "network_required": False,
        "offline": OFFLINE,
    }
    if dry_run:
        report["stages"] = {
            s: {"state": "PENDING" if s in PROFILES[profile] else "SKIPPED"} for s in STAGES
        }
        report["dry_run"] = True
        return report
    adapter = adapter_factory(config, source.source_path, sid, policy)
    roles = sorted({MODEL_STAGES[s] for s in PROFILES[profile] if s in MODEL_STAGES})
    # Read-only inventory before stage execution; no model load or network.
    report["model_readiness"] = {role: adapter.model_ready(role) for role in roles}
    with exclusive_lock(path.parent / ".workflow.lock"):
        old = read_artifact(path) if path.exists() else {}
        if old and old.get("source_sha256") != source.source_sha256:
            raise ValueError("session prefix collision")
        old_stages = old.get("stages", {})
        results, parent_bindings = {}, {}

        def save():
            atomic_write_json(path, seal_artifact(report))

        for stage in STAGES:
            if stage not in PROFILES[profile]:
                report["stages"][stage] = {"state": "SKIPPED", "runtime_seconds": 0.0}
                continue
            fp = stage_fingerprint(
                stage=f"workflow_{stage.lower()}",
                source_sha256=source.source_sha256,
                relevant_config=settings(config, stage, policy),
                parent_checksums={k: digest(v) for k, v in parent_bindings.items()},
                pipeline_version=__version__,
                stage_schema_version=1,
            )
            previous = old_stages.get(stage, {})
            # INGEST rechecks original source; identity rechecks current human mapping.
            reusable = stage not in {"INGEST", "IDENTITY_MAPPING"} and verified(
                config, previous, fp
            )
            if reusable:
                entry = {**previous, "reused": True}
                result = entry["result"]
            else:
                entry = {
                    "state": "RUNNING",
                    "fingerprint": fp,
                    "pid": os.getpid(),
                    "runtime_seconds": 0.0,
                    "reused": False,
                    "recovered_from": previous.get("state"),
                }
                if entry["recovered_from"] == "RUNNING":
                    entry["recovered_from"] = "INTERRUPTED"
                report["stages"][stage] = entry
                report["state"] = "RUNNING"
                save()
                if progress:
                    progress(stage)
                started = time.perf_counter()
                try:
                    role = MODEL_STAGES.get(stage)
                    if (
                        role
                        and not report["model_readiness"][role]
                        and adapter.inference_needed(stage, results)
                    ):
                        result = {
                            "state": "BLOCKED_MODEL_PROVISIONING",
                            "model_role": role,
                            "provision_command": "uv run --no-sync session-pipeline "
                            f"models provision --role {role}",
                            "warnings": [
                                f"local pinned {role} model not ready; provision once "
                                "(or run models verify if files are already local)"
                            ],
                        }
                    else:
                        # Engine progress is captured; normal CLI and --json remain concise/valid.
                        from unittest.mock import patch

                        from session_pipeline.asr_retry import deny_network
                        from session_pipeline.model_store import local_only_environment

                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            patch.dict(
                                os.environ,
                                local_only_environment(config.application.root),
                                clear=True,
                            ),
                            deny_network(),
                        ):
                            result = adapter.run(stage, results)
                    entry.update(
                        state=result["state"], result=result, bindings=bindings(config, result)
                    )
                except KeyboardInterrupt:
                    entry.update(state="INTERRUPTED", result={"state": "INTERRUPTED"}, bindings=[])
                    report["state"] = "INTERRUPTED"
                    entry["runtime_seconds"] = time.perf_counter() - started
                    save()
                    return report
                except Exception as error:
                    result = {
                        "state": "FAILED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    entry.update(state="FAILED", result=result, bindings=[])
                entry["runtime_seconds"] = time.perf_counter() - started
            report["stages"][stage] = entry
            results[stage] = result
            if stage == "ALIGNMENT":
                report["warning_counts"] = result.get("warning_counts", {})
            parent_bindings[stage] = entry.get("bindings", [])
            report["warnings"].extend(result.get("warnings", []))
            if stage == "INTEGRITY":
                report["source_duration_seconds"] = result.get("duration")
            if result["state"] not in COMPLETE:
                report.update(state=result["state"], stopped_at=stage)
                for key in ("gate", "review_command", "provision_command", "error", "error_type"):
                    if key in result:
                        report[key] = result[key]
                for pending in STAGES[STAGES.index(stage) + 1 :]:
                    report["stages"][pending] = {
                        "state": "PENDING" if pending in PROFILES[profile] else "SKIPPED"
                    }
                save()
                return report
        last = results[PROFILES[profile][-1]]
        report.update(
            state="SUCCEEDED_WITH_WARNINGS"
            if report["warnings"]
            or any(r["state"] == "SUCCEEDED_WITH_WARNINGS" for r in results.values())
            else "SUCCEEDED",
            json_path=last.get("json_path"),
            txt_path=last.get("txt_path"),
        )
        save()
        return report


def alignment_warning_counts(report):
    return {
        "unaligned_words": report["unaligned_word_count"],
        "failed_segments": report["segment_status_counts"]["ALIGNMENT_FAILED"],
        "partial_segments": report["segment_status_counts"]["PARTIAL_ALIGNMENT"],
        "suspicious_word_durations": report["word_qa_counts"]["SUSPICIOUS_WORD_DURATION"],
        "partial_units": report["partial_units"],
        "failed_units": report["failed_units"],
    }


def transcribe(config, value, *, continue_on_error=False, **options):
    paths = audio_inputs(Path(value))
    if not paths:
        raise ValueError("no supported audio files found")
    reports = []
    for path in paths:
        try:
            report = transcribe_one(config, path, **options)
        except (OSError, ValueError, KeyError, TypeError) as error:
            try:
                sid = build_session_source(path).session_id
            except OSError:
                sid = None
            report = {
                "state": "FAILED",
                "session_id": sid,
                "profile": options.get("profile", DEFAULT_PROFILE),
                "stages": {},
                "warnings": [],
                "error_type": type(error).__name__,
                "error": str(error),
            }
        report["source_path"] = str(path)
        reports.append(report)
        if report["state"] not in COMPLETE and not report.get("dry_run"):
            if continue_on_error and report["state"] == "FAILED":
                continue
            break  # Gates and interruptions still stop; failures require explicit opt-in.
    failures = [
        {k: r.get(k) for k in ("source_path", "session_id", "stopped_at", "error_type", "error")}
        for r in reports
        if r["state"] == "FAILED"
    ]
    summary = {
        "succeeded": sum(r["state"] == "SUCCEEDED" for r in reports),
        "succeeded_with_warnings": sum(r["state"] == "SUCCEEDED_WITH_WARNINGS" for r in reports),
        "failed": len(failures),
        "waiting_blocking": sum(r["state"] not in COMPLETE | {"FAILED"} for r in reports),
        # Reuse is supplementary to the completion counts above.
        "skipped_reused": sum(
            r["state"] == "SKIPPED"
            or r["state"] in COMPLETE
            and all(
                e.get("reused")
                for s, e in r["stages"].items()
                if s != "INGEST" and e["state"] != "SKIPPED"
            )
            for r in reports
        ),
        "unprocessed": len(paths) - len(reports),
    }
    return {
        "state": "INTERRUPTED"
        if reports[-1]["state"] == "INTERRUPTED"
        else "FAILED"
        if failures
        else reports[-1]["state"]
        if reports[-1]["state"] not in COMPLETE
        else "SUCCEEDED_WITH_WARNINGS"
        if any(r["state"] == "SUCCEEDED_WITH_WARNINGS" for r in reports)
        else "SUCCEEDED",
        "sessions": reports,
        "batch_policy": "SEQUENTIAL_CONTINUE_ON_ERROR_STOP_ON_GATE"
        if continue_on_error
        else "SEQUENTIAL_STOP_ON_GATE_OR_FAILURE",
        "summary": summary,
        "failures": failures,
        "discovered_count": len(paths),
        "unprocessed": [str(p) for p in paths[len(reports) :]],
    }


def selected_review(config, sid, gate):
    """Resolve the persisted gate without asking users for internal fingerprints."""
    report = read_artifact(receipt_path(config, sid))
    stage, key = (
        ("ATTRIBUTION", "attributed_fingerprint")
        if gate == "SPEAKER_IDENTITY"
        else ("ASR_RETRY", "retry_fingerprint")
    )
    return report["stages"][stage]["result"][key]


def status(config, value):
    """Read-only verification, including final lineages created before productization."""
    sid = resolve_session(value)
    path = receipt_path(config, sid)
    if path.exists():
        report = read_artifact(path)
        report.pop("artifact_checksum", None)
        # A lock probe detects a surviving workflow process, independent of PID reuse.
        locked = False
        lock = path.parent / ".workflow.lock"
        if lock.exists():
            try:
                with exclusive_lock(lock):
                    pass
            except OSError:
                locked = True
        parents = {}
        invalid_dependency = False
        for stage in STAGES:
            if stage not in report["stages"]:
                continue
            entry = report["stages"][stage]
            if entry["state"] == "RUNNING" and not locked:
                entry["state"] = "INTERRUPTED"
                report["state"] = "INTERRUPTED"
            elif entry["state"] in COMPLETE and entry["state"] != "SKIPPED":
                expected = stage_fingerprint(
                    stage=f"workflow_{stage.lower()}",
                    source_sha256=report["source_sha256"],
                    relevant_config=settings(config, stage, report["review_policy"]),
                    parent_checksums={k: digest(v) for k, v in parents.items()},
                    pipeline_version=__version__,
                    stage_schema_version=1,
                )
                if invalid_dependency or not verified(config, entry, expected):
                    entry["state"] = "FAILED"
                    report.update(state="FAILED", error=f"{stage} artifact verification failed")
                    invalid_dependency = True
            if entry["state"] != "SKIPPED":
                parents[stage] = entry.get("bindings", [])
        if report.get("json_path") and report["profile"] == "full" and report["state"] in COMPLETE:
            from session_pipeline.final_render import status as final_status

            final_status(config, sid, Path(report["json_path"]).parent.name)
        return report
    raw, work, _ = session_paths(config, sid)
    candidates = sorted((raw.parent / "final").glob("*/transcript.final.fa.json"))
    if len(candidates) > 1:
        raise ValueError("multiple historical final lineages; use final status --fingerprint")
    if candidates:
        from session_pipeline.final_render import status as final_status

        final = final_status(config, sid, candidates[0].parent.name)
        document = read_artifact(candidates[0])
        # The final verifier checks parent artifacts, human mapping and exact TXT.
        stages = {
            s: {
                "state": "SUCCEEDED",
                "runtime_seconds": None,
                "evidence": "VALIDATED_HISTORICAL_LINEAGE",
            }
            for s in STAGES
        }
        stages["INGEST"]["state"] = "SKIPPED"
        for s, p in [
            ("INTEGRITY", work / "integrity.json"),
            ("ASR_QA", work / "qa/asr-review.json"),
        ]:
            evidence = read_artifact(p)
            stages[s]["state"] = evidence["status"]
            if s == "ASR_QA" and evidence.get("flagged_segment_count"):
                stages[s]["state"] = "SUCCEEDED_WITH_WARNINGS"
        stages["ASR"]["state"] = load_canonical(raw, sid)["status"]
        for s, parent in [
            ("ALIGNMENT", "aligned_asr"),
            ("DIARIZATION", "diarization"),
            ("ATTRIBUTION", "attribution"),
        ]:
            evidence = read_artifact(
                config.application.root / document["input_bindings"][parent]["relative_path"]
            )
            stages[s]["state"] = evidence["status"]
            if s == "ALIGNMENT" and evidence.get("qa", {}).get("alignment_has_warnings"):
                stages[s]["state"] = "SUCCEEDED_WITH_WARNINGS"
        stages["IDENTITY_MAPPING"].update(mapping_status="HUMAN_CONFIRMED")
        stages["FINAL_RENDER"]["state"] = final["state"]
        return {
            "session_id": sid,
            "profile": "full",
            "state": final["state"],
            "review_policy": "historical:"
            + document.get("attribution_parent_provenance", {}).get(
                "review_policy", "persisted-reviewed-lineage"
            ),
            "source_duration_seconds": document["source_duration_seconds"],
            "stages": stages,
            "warnings": final["warnings"],
            "json_path": final["json_path"],
            "txt_path": final["txt_path"],
            "recognition": "READ_ONLY_PRE_PRODUCTIZATION_FINAL",
            "models_loaded": False,
        }
    return {
        "session_id": sid,
        "profile": None,
        "state": "PENDING",
        "warnings": [],
        "stages": {s: {"state": "PENDING"} for s in STAGES},
        "recognition": "NO_WORKFLOW_OR_FINAL_RECEIPT",
        "models_loaded": False,
    }


def human_summary(report):
    lines = [
        f"Session: {report['session_id']}   Profile: {report['profile']}   State: {report['state']}"
    ]
    if report.get("source_duration_seconds") is not None:
        lines.append(f"Source duration: {report['source_duration_seconds']:.3f} seconds")
    for stage in STAGES:
        if stage not in report["stages"]:
            continue
        entry = report["stages"][stage]
        runtime = entry.get("runtime_seconds")
        suffix = f"  {runtime:.2f}s" if runtime is not None else ""
        suffix += "  reused" if entry.get("reused") else ""
        suffix += "  HUMAN_CONFIRMED" if entry.get("mapping_status") == "HUMAN_CONFIRMED" else ""
        lines.append(f"{stage:18} {entry['state']}{suffix}")
    lines.extend(f"Warning: {w}" for w in report.get("warnings", []))
    if report.get("warning_counts"):
        lines.append(
            "Alignment warnings: "
            + ", ".join(f"{key}={value}" for key, value in report["warning_counts"].items())
        )
    for key, label in [
        ("json_path", "JSON"),
        ("txt_path", "TXT"),
        ("review_command", "Review"),
        ("provision_command", "Provision"),
        ("error", "Error"),
    ]:
        if report.get(key):
            lines.append(f"{label}: {report[key]}")
    return "\n".join(lines)
