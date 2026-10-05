"""Pinned, offline two-speaker diarization. Planning/status never import ML."""

from __future__ import annotations

import copy
import gc
import importlib.metadata
import os
import re
import sys
import time
import wave
from collections import Counter
from unittest.mock import patch

from session_pipeline import __version__
from session_pipeline import alignment as alignment_contract
from session_pipeline.artifacts import (
    atomic_write_json,
    read_artifact,
    seal_artifact,
    validate_artifact,
)
from session_pipeline.asr_quality import number, session_paths
from session_pipeline.asr_retry import deny_network, exclusive_lock, extract_clip
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.human_review import require
from session_pipeline.model_store import (
    inspect_model,
    load_model_lock,
    local_model_path,
    local_only_environment,
)
from session_pipeline.models import sha256_file, utc_timestamp

MODEL = {
    "diarization": {
        "repo_id": "pyannote/speaker-diarization-community-1",
        "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    }
}
SETTINGS = {
    "diarization_contract_version": 1,
    "strategy": "WHOLE_RECORDING_GLOBAL_CLUSTERING",
    "device": "cpu",
    "cpu_threads": 4,
    "min_speakers": 2,
    "max_speakers": 2,
    "segmentation_batch_size": 4,
    "embedding_batch_size": 8,
    "sample_rate": 16000,
    "channels": 1,
    "audio_input": "DERIVED_PCM_WAVEFORM",
    "smoke_seconds": 180,
    "output_policy": "REGULAR_OVERLAP_PRESERVING",
    "short_turn_seconds": 0.1,
    "long_uncovered_seconds": 30,
    "uncovered_reviewed_seconds": 1,
}
SPEAKERS = {"SPEAKER_00", "SPEAKER_01"}
ERRORS = (
    "EMPTY_DIARIZATION",
    "MISSING_SPEAKER",
    "INVALID_SPEAKER_LABEL",
    "OUT_OF_BOUNDS_TURN",
    "INVALID_ORDERING",
    "INVALID_TURN_DURATION",
)
WARNINGS = ("SUSPICIOUS_SHORT_TURN", "LONG_UNCOVERED_INTERVAL", "UNCOVERED_REVIEWED_BOUNDS")


def model_spec(config):
    require(
        config.ml.diarization_model == MODEL["diarization"]["repo_id"]
        and config.ml.num_speakers == 2
        and config.ml.device == "cpu",
        "diarization requires the approved CPU two-speaker configuration",
    )
    spec = next(
        m for m in load_model_lock(config.application.root)["models"] if m["role"] == "diarization"
    )
    require(
        {"repo_id": spec["repo_id"], "revision": spec["revision"]} == MODEL["diarization"],
        "diarization lock differs from approved exact revision",
    )
    return spec


def model_binding(spec):
    # Provisioning fills missing SHA256s after validating pinned Git blobs. Bind
    # those stable blob identities so this enrichment does not invalidate a plan.
    return alignment_contract.digest(
        {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "files": [
                {
                    "path": f["path"],
                    "required": f["required"],
                    "size": f["size_bytes"],
                    "identity": f["git_blob_sha1"] or f["sha256"],
                }
                for f in spec["files"]
            ],
        }
    )


def configuration(smoke):
    versions = {}
    for name in ("pyannote-audio", "pyannote-core", "pyannote-pipeline", "torch", "torchcodec"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "UNKNOWN"
    return {**SETTINGS, "mode": "SMOKE" if smoke else "FULL", "package_versions": versions}


def load_input(config, sid, aligned_fingerprint=None):
    raw, work, _ = session_paths(config, sid)
    output = raw.parent
    if aligned_fingerprint is None:
        candidates = sorted((output / "aligned").glob("*/transcript.aligned.fa.json"))
        require(
            len(candidates) == 1,
            "specify --aligned-fingerprint for exactly one full aligned parent",
        )
        path = candidates[0].resolve()
    else:
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", aligned_fingerprint)), "invalid aligned fingerprint"
        )
        path = (output / "aligned" / aligned_fingerprint / "transcript.aligned.fa.json").resolve()
    require(path.is_relative_to((output / "aligned").resolve()), "aligned parent escapes lineage")
    aligned = read_artifact(path)
    require(
        aligned.get("artifact_type") == "aligned_asr"
        and aligned.get("status") == "SUCCEEDED"
        and aligned["session_id"] == sid
        and aligned["stage_fingerprint"] == path.parent.name
        and aligned["configuration"]["mode"] == "FULL"
        and aligned["qa"]["full_alignment_complete"] is True
        and aligned["qa"]["coverage_validation_pass"] is True,
        "invalid or incomplete full aligned parent",
    )
    alignment_plan_path = work / "alignment" / path.parent.name / "plan.json"
    alignment_plan = read_artifact(alignment_plan_path)
    reviewed_path, reviewed, source, _, integrity = alignment_contract.load_input(
        config, sid, alignment_contract.path_fingerprint(alignment_plan)
    )
    alignment_contract.validate_plan(
        config,
        alignment_plan,
        reviewed_path,
        reviewed,
        integrity,
        alignment_contract.model_spec(config),
    )
    require(
        aligned["plan_checksum"] == alignment_plan["artifact_checksum"]
        and aligned["parent_checksums"] == alignment_plan["parent_checksums"]
        and aligned["source_sha256"] == reviewed["source_sha256"]
        and aligned["probe"] == reviewed["probe"]
        and aligned["source"] == reviewed["source"]
        and aligned["model"] == alignment_contract.MODEL
        and aligned["configuration"] == alignment_plan["configuration"]
        and len(aligned["segments"]) == aligned["segment_count"] == reviewed["segment_count"],
        "aligned parent lineage changed",
    )
    alignment_contract.validate_coverage(alignment_plan, reviewed, aligned["segments"])
    # Keep all reviewed segment references, including untimed-word fallback bounds.
    references = [
        {
            "reviewed_segment_id": s["reviewed_segment_id"],
            "reviewed_segment_index": s["reviewed_segment_index"],
            "alignment_status": s["status"],
            "reviewed_interval": [s["start"], s["end"]],
            "timing_policy": "REVIEWED_SEGMENT_FALLBACK"
            if s["status"] == "ALIGNMENT_FAILED"
            else "ALIGNED_WORDS_WHERE_AVAILABLE",
        }
        for s in aligned["segments"]
    ]
    immutable = [
        path,
        reviewed_path,
        reviewed_path.with_suffix(".txt"),
        source,
        alignment_plan_path,
    ]
    immutable.extend(p for p in (raw, raw.with_suffix(".txt")) if p.exists())
    return path, aligned, source, work, references, immutable


def plan_document(config, sid, *, smoke=False, aligned_fingerprint=None):
    path, aligned, _, work, references, _ = load_input(config, sid, aligned_fingerprint)
    spec = model_spec(config)
    settings = configuration(smoke)
    duration = aligned["probe"]["duration_seconds"]
    require(number(duration) and duration > 0, "invalid diarization source duration")
    interval = [0, min(duration, SETTINGS["smoke_seconds"]) if smoke else duration]
    parents = {"aligned_asr": aligned["artifact_checksum"], "model_lock": model_binding(spec)}
    fp = stage_fingerprint(
        stage="local_two_speaker_diarization",
        source_sha256=aligned["source_sha256"],
        relevant_config={"settings": settings, "interval": interval},
        parent_checksums=parents,
        models=MODEL,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "diarization_plan",
            "status": "PLANNED",
            "session_id": sid,
            "source_sha256": aligned["source_sha256"],
            "source_duration_seconds": duration,
            "processed_interval": interval,
            "stage_fingerprint": fp,
            "configuration": settings,
            "model": MODEL,
            "parent_checksums": parents,
            "aligned_fingerprint": path.parent.name,
            "aligned_relative_path": path.relative_to(config.application.root).as_posix(),
            "aligned_file_sha256": sha256_file(path),
            "aligned_segment_references": references,
            "speaker_identity_assignment": False,
            "resume_policy": "REUSE_COMPLETE_OR_RESTART_WHOLE_JOB",
        }
    )
    destination = work / "diarization" / fp / "plan.json"
    require(
        destination.resolve().is_relative_to(work.resolve()),
        "diarization plan escapes session work storage",
    )
    return destination, document


def make_plan(config, sid, *, smoke=False, aligned_fingerprint=None):
    path, plan = plan_document(config, sid, smoke=smoke, aligned_fingerprint=aligned_fingerprint)
    if path.exists():
        require(read_artifact(path) == plan, "existing diarization plan changed")
    else:
        atomic_write_json(path, plan)
    return path, plan


def read_plan(config, sid, path):
    _, work, _ = session_paths(config, sid)
    require(
        path.resolve().is_relative_to((work / "diarization").resolve()),
        "invalid diarization plan path",
    )
    plan = read_artifact(path)
    expected_path, expected = plan_document(
        config,
        sid,
        smoke=plan["configuration"]["mode"] == "SMOKE",
        aligned_fingerprint=plan["aligned_fingerprint"],
    )
    require(
        path.resolve() == expected_path.resolve() and plan == expected,
        "diarization plan or immutable aligned parent changed",
    )
    return plan


def output_path(config, sid, plan):
    raw, _, _ = session_paths(config, sid)
    namespace = raw.parent / "diarized"
    if plan["configuration"]["mode"] == "SMOKE":
        namespace /= "smoke"
    path = namespace / plan["stage_fingerprint"] / "diarization.json"
    require(
        path.resolve().is_relative_to((raw.parent / "diarized").resolve())
        and path.resolve().is_relative_to(raw.parent.resolve()),
        "diarization output escapes lineage",
    )
    return path


def merged_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def evaluate_turns(turns, plan):
    """Validate regular turns; simultaneous speakers remain overlapping intervals."""
    errors, warnings = Counter(), Counter()
    labels = set()
    lower, upper = plan["processed_interval"]
    previous = None
    valid = []
    require(isinstance(turns, list), "invalid diarization engine result")
    if not turns:
        errors["EMPTY_DIARIZATION"] += 1
    for turn in turns:
        start, end, speaker = turn.get("start"), turn.get("end"), turn.get("speaker")
        if isinstance(speaker, str) and re.fullmatch(r"SPEAKER_[0-9]{2}", speaker):
            labels.add(speaker)
        if speaker not in SPEAKERS:
            errors["INVALID_SPEAKER_LABEL"] += 1
        if not (number(start) and number(end) and lower <= start < end <= upper):
            errors["OUT_OF_BOUNDS_TURN"] += 1
            continue
        key = (start, end, speaker if isinstance(speaker, str) else "")
        if previous is not None and key < previous:
            errors["INVALID_ORDERING"] += 1
        previous = key
        if "duration" in turn and turn["duration"] != end - start:
            errors["INVALID_TURN_DURATION"] += 1
        if end - start < SETTINGS["short_turn_seconds"]:
            warnings["SUSPICIOUS_SHORT_TURN"] += 1
        valid.append((start, end, speaker))
    if labels != SPEAKERS:
        errors["MISSING_SPEAKER"] += len(SPEAKERS - labels)
    coverage = merged_intervals((a, b) for a, b, _ in valid)
    cursor = lower
    gaps = []
    for a, b in coverage:
        if cursor < a:
            gaps.append([cursor, a])
        cursor = max(cursor, b)
    if cursor < upper:
        gaps.append([cursor, upper])
    warnings["LONG_UNCOVERED_INTERVAL"] = sum(
        b - a > SETTINGS["long_uncovered_seconds"] for a, b in gaps
    )
    uncovered = []
    for ref in plan["aligned_segment_references"]:
        a, b = ref["reviewed_interval"]
        a, b = max(a, lower), min(b, upper)
        if b <= a:
            continue
        covered = sum(max(0, min(b, y) - max(a, x)) for x, y in coverage)
        if b - a - covered > SETTINGS["uncovered_reviewed_seconds"]:
            uncovered.append(ref["reviewed_segment_id"])
    warnings["UNCOVERED_REVIEWED_BOUNDS"] = len(uncovered)
    # Sweep speaker activity rather than flattening regular output into exclusive turns.
    events = []
    for a, b, speaker in valid:
        events.extend([(a, 1, speaker), (b, -1, speaker)])
    active = Counter()
    overlaps = []
    previous_time = lower
    for timestamp, changes in _group_events(events):
        speakers = sorted(s for s, n in active.items() if n > 0 and s in SPEAKERS)
        if timestamp > previous_time and len(speakers) > 1:
            overlaps.append({"start": previous_time, "end": timestamp, "speakers": speakers})
        for delta, speaker in changes:
            active[speaker] += delta
        previous_time = timestamp
    speech_seconds = sum(b - a for a, b in coverage)
    return {
        "validation_pass": not any(errors.values()),
        "expected_speaker_labels": sorted(SPEAKERS),
        "detected_speaker_labels": sorted(labels),
        "speaker_count": len(labels),
        "turn_count": len(turns),
        "error_counts": {key: errors[key] for key in ERRORS},
        "warning_counts": {key: warnings[key] for key in WARNINGS},
        "has_warnings": any(warnings.values()),
        "coverage_intervals": coverage,
        "uncovered_intervals": gaps,
        "speech_union_seconds": speech_seconds,
        "summed_turn_seconds": sum(b - a for a, b, _ in valid),
        "duration_coverage_fraction": speech_seconds / (upper - lower),
        "overlap_intervals": overlaps,
        "overlap_seconds": sum(o["end"] - o["start"] for o in overlaps),
        "uncovered_reviewed_segment_ids": uncovered,
        "uncovered_measurement": "TIME_WITHOUT_MODEL_TURNS_NOT_VERIFIED_SILENCE",
        "speaker_confidence": None,
    }


def _group_events(events):
    from itertools import groupby

    for timestamp, group in groupby(sorted(events, key=lambda e: e[0]), key=lambda e: e[0]):
        yield timestamp, [(delta, speaker) for _, delta, speaker in group]


class LocalDiarizer:
    """Only explicit execution constructs pyannote; all subordinate paths are local."""

    def __init__(self, path):
        require(
            path.is_dir() and path.name == MODEL["diarization"]["revision"],
            "wrong local diarization revision",
        )
        import torch
        from pyannote.audio import Pipeline

        torch.set_num_threads(SETTINGS["cpu_threads"])
        self.pipeline = Pipeline.from_pretrained(
            str(path), token=False, cache_dir=path.parents[1] / ".hf-cache"
        )
        require(self.pipeline is not None, "local diarization pipeline could not load")
        self.pipeline.to(torch.device("cpu"))
        self.pipeline.segmentation_batch_size = SETTINGS["segmentation_batch_size"]
        self.pipeline.embedding_batch_size = SETTINGS["embedding_batch_size"]

    def __call__(self, clip):
        import numpy as np
        import torch

        # Reuse the FFmpeg-derived PCM rather than relying on TorchCodec DLL
        # discovery in this process. pyannote supports waveform dictionaries.
        with wave.open(str(clip), "rb") as stream:
            require(
                stream.getnchannels() == 1
                and stream.getframerate() == SETTINGS["sample_rate"]
                and stream.getsampwidth() == 2,
                "invalid diarization PCM",
            )
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        require(samples.size > 0, "empty diarization PCM")
        waveform = torch.from_numpy(samples.astype(np.float32)[None, :] / 32768)
        with torch.inference_mode():
            result = self.pipeline(
                {"waveform": waveform, "sample_rate": SETTINGS["sample_rate"]},
                min_speakers=2,
                max_speakers=2,
            )
        # Regular diarization preserves overlap. Never substitute exclusive output.
        annotation = result.speaker_diarization
        turns = [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
                "duration": float(turn.end) - float(turn.start),
            }
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        return sorted(turns, key=lambda t: (t["start"], t["end"], t["speaker"]))


def checkpoint_path(plan_path):
    path = plan_path.parent / "checkpoint.json"
    require(
        path.resolve().is_relative_to(plan_path.parent.resolve()),
        "diarization checkpoint escapes lineage",
    )
    return path


def read_checkpoint(plan_path, plan):
    cp = read_artifact(checkpoint_path(plan_path))
    require(
        cp.get("artifact_type") == "diarization_checkpoint"
        and cp.get("plan_checksum") == plan["artifact_checksum"]
        and cp.get("parent_checksums") == plan["parent_checksums"]
        and cp.get("stage_fingerprint") == plan["stage_fingerprint"]
        and cp.get("status") in {"RUNNING", "INTERRUPTED", "FAILED", "SUCCEEDED"},
        "diarization checkpoint identity changed",
    )
    if cp["status"] == "SUCCEEDED":
        result = validate_artifact(cp["result"])
        require(
            result.get("artifact_type") == "diarization"
            and result.get("status") == "SUCCEEDED"
            and result["session_id"] == plan["session_id"]
            and result["stage_fingerprint"] == plan["stage_fingerprint"]
            and result["aligned_fingerprint"] == plan["aligned_fingerprint"]
            and result["processed_interval"] == plan["processed_interval"]
            and result["source_duration_seconds"] == plan["source_duration_seconds"]
            and result["full_diarization_complete"] == (plan["configuration"]["mode"] == "FULL")
            and result["speaker_identities_assigned"] is False
            and result["transcript_attribution_performed"] is False
            and result["plan_checksum"] == plan["artifact_checksum"]
            and result["parent_checksums"] == plan["parent_checksums"]
            and result["model"] == MODEL
            and result["configuration"] == plan["configuration"]
            and result["source_sha256"] == plan["source_sha256"]
            and result["aligned_segment_references"] == plan["aligned_segment_references"]
            and result["qa"] == evaluate_turns(result["turns"], plan)
            and result["qa"]["validation_pass"],
            "invalid completed diarization result",
        )
    return cp


def worker_active(work):
    path = work / "diarization/.worker.lock"
    if not path.exists():
        return False
    try:
        with exclusive_lock(path):
            return False
    except OSError:
        return True


def publish(config, sid, plan, result):
    path = output_path(config, sid, plan)
    if path.exists():
        require(read_artifact(path) == result, "existing diarization output changed")
    else:
        atomic_write_json(path, result)
    return path


def status_report(config, sid, *, smoke=False, aligned_fingerprint=None, fingerprint=None):
    if fingerprint:
        require(bool(re.fullmatch(r"[0-9a-f]{64}", fingerprint)), "invalid diarization fingerprint")
        _, work, _ = session_paths(config, sid)
        selected = read_plan(config, sid, work / "diarization" / fingerprint / "plan.json")
        require(
            aligned_fingerprint is None or aligned_fingerprint == selected["aligned_fingerprint"],
            "status aligned fingerprint conflicts with selected plan",
        )
        aligned_fingerprint = selected["aligned_fingerprint"]
    _, _, _, work, _, _ = load_input(config, sid, aligned_fingerprint)
    model = inspect_model(config.application.root, model_spec(config), full=True)
    if fingerprint:
        require(bool(re.fullmatch(r"[0-9a-f]{64}", fingerprint)), "invalid diarization fingerprint")
        path = work / "diarization" / fingerprint / "plan.json"
        plan = read_plan(config, sid, path)
    else:
        path, plan = plan_document(
            config, sid, smoke=smoke, aligned_fingerprint=aligned_fingerprint
        )
        if path.exists():
            plan = read_plan(config, sid, path)
    state = "PENDING" if path.exists() else "NOT_PLANNED"
    result, cp = None, None
    if checkpoint_path(path).exists():
        try:
            cp = read_checkpoint(path, plan)
            state = cp["status"]
            if state == "RUNNING" and not worker_active(work):
                state = "INTERRUPTED"
            if state == "SUCCEEDED":
                result = cp["result"]
        except (OSError, ValueError, KeyError, TypeError):
            state = "INVALID_CHECKPOINT"
            cp = None
    out = output_path(config, sid, plan)
    published = False
    if result and out.exists():
        try:
            published = read_artifact(out) == result
            if not published:
                state = "INVALID_OUTPUT"
        except ValueError:
            state = "INVALID_OUTPUT"
    qa = result["qa"] if result else (cp.get("qa") if cp else None)
    return {
        "session_id": sid,
        "model_provisioned": model["status"] == "PROVISIONED",
        "model_status": model["status"],
        "pinned_revision": MODEL["diarization"]["revision"],
        "verified_revision": model["revision"] if model["status"] == "PROVISIONED" else None,
        "model_verification": model["verification"],
        "model_missing_required": model["missing_required"],
        "aligned_fingerprint": plan["aligned_fingerprint"],
        "aligned_checksum": plan["parent_checksums"]["aligned_asr"],
        "fingerprint": plan["stage_fingerprint"],
        "mode": plan["configuration"]["mode"],
        "configuration": plan["configuration"],
        "strategy": SETTINGS["strategy"],
        "source_duration_seconds": plan["source_duration_seconds"],
        "processed_interval": plan["processed_interval"],
        "state": state,
        "plan_path": str(path) if path.exists() else None,
        "output_path": str(out) if published else None,
        "expected_output_path": str(out),
        "processing_complete": state == "SUCCEEDED",
        "full_diarization_complete": published
        and state == "SUCCEEDED"
        and plan["configuration"]["mode"] == "FULL",
        "detected_speaker_labels": qa["detected_speaker_labels"] if qa else [],
        "turn_count": qa["turn_count"] if qa else 0,
        "duration_coverage": qa["duration_coverage_fraction"] if qa else None,
        "qa": qa,
        "failure_code": cp.get("failure_code") if cp else None,
        "failure_phase": cp.get("failure_phase") if cp else None,
        "models_loaded": False,
        "network_required": False,
        "hf_token_required_for_runtime": False,
        "speaker_identities_assigned": False,
    }


def run_diarization(
    config,
    sid,
    *,
    smoke=False,
    aligned_fingerprint=None,
    engine_factory=LocalDiarizer,
    extractor=extract_clip,
):
    plan_path, plan = make_plan(config, sid, smoke=smoke, aligned_fingerprint=aligned_fingerprint)
    _, aligned, source, work, _, immutable = load_input(config, sid, plan["aligned_fingerprint"])
    baseline = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in immutable}
    require(baseline[source][0] == aligned["source_sha256"], "source audio SHA-256 mismatch")
    engine = None
    # Exclude existing same-session alignment/ASR runners without changing their contracts.
    with (
        exclusive_lock(work / "diarization/.worker.lock"),
        exclusive_lock(work / "alignment/.worker.lock"),
        exclusive_lock(work / "qa/.retry.lock"),
        patch.dict(os.environ, local_only_environment(config.application.root), clear=True),
        deny_network(),
    ):
        cp_path = checkpoint_path(plan_path)
        old = read_checkpoint(plan_path, plan) if cp_path.exists() else None
        if old and old["status"] == "SUCCEEDED":
            publish(config, sid, plan, old["result"])
            return status_report(
                config,
                sid,
                aligned_fingerprint=plan["aligned_fingerprint"],
                fingerprint=plan["stage_fingerprint"],
            )
        if old:
            archive = plan_path.parent / "attempts" / f"{old['artifact_checksum']}.json"
            require(
                archive.resolve().is_relative_to(plan_path.parent.resolve()),
                "diarization attempt archive escapes lineage",
            )
            if archive.exists():
                require(read_artifact(archive) == old, "diarization attempt archive changed")
            else:
                atomic_write_json(archive, old)
        cp = {
            "schema_version": 1,
            "artifact_type": "diarization_checkpoint",
            "status": "RUNNING",
            "plan_checksum": plan["artifact_checksum"],
            "parent_checksums": plan["parent_checksums"],
            "stage_fingerprint": plan["stage_fingerprint"],
            "attempt": old["attempt"] + 1 if old else 1,
            "started_at": utc_timestamp(),
            "immutable_input_checksums": {
                p.relative_to(config.application.root).as_posix(): sha
                for p, (sha, _) in baseline.items()
            },
        }
        atomic_write_json(cp_path, seal_artifact(cp))
        started = time.perf_counter()
        clip = plan_path.parent / "derived/audio.wav"
        phase = "MODEL_VERIFICATION"
        try:
            read_plan(config, sid, plan_path)
            # Hash-verify every required pinned file before factory/inference/decoding.
            local_path = local_model_path(config.application.root, "diarization")
            require(
                local_path.name == MODEL["diarization"]["revision"],
                "wrong local diarization revision",
            )
            require(
                not clip.is_symlink()
                and clip.resolve().is_relative_to(plan_path.parent.resolve())
                and clip.resolve() != source.resolve(),
                "unsafe derived diarization clip",
            )
            inventory = inspect_model(config.application.root, model_spec(config), full=True)
            require(
                inventory["status"] == "PROVISIONED", "pinned local diarization model unavailable"
            )
            phase = "AUDIO_DECODING"
            extractor(source, clip, plan["processed_interval"])
            phase = "MODEL_LOADING"
            engine = engine_factory(local_path)
            phase = "INFERENCE"
            turns = engine(clip)
            phase = "OUTPUT_VALIDATION"
            qa = evaluate_turns(turns, plan)
            cp["qa"] = qa
            require(qa["validation_pass"], "diarization turn contract failed")
            require(
                all(
                    sha256_file(p) == sha and p.stat().st_mtime_ns == modified
                    for p, (sha, modified) in baseline.items()
                ),
                "immutable diarization input changed",
            )
            read_plan(config, sid, plan_path)
            final_inventory = inspect_model(config.application.root, model_spec(config), full=True)
            require(
                final_inventory["status"] == "PROVISIONED"
                and final_inventory["files"] == inventory["files"],
                "diarization model changed during execution",
            )
            seconds = plan["processed_interval"][1] - plan["processed_interval"][0]
            wall_seconds = time.perf_counter() - started
            result = seal_artifact(
                {
                    "schema_version": 1,
                    "artifact_type": "diarization",
                    "status": "SUCCEEDED",
                    "session_id": sid,
                    "source_sha256": plan["source_sha256"],
                    "source_duration_seconds": plan["source_duration_seconds"],
                    "processed_interval": plan["processed_interval"],
                    "stage_fingerprint": plan["stage_fingerprint"],
                    "aligned_fingerprint": plan["aligned_fingerprint"],
                    "parent_checksums": plan["parent_checksums"],
                    "plan_checksum": plan["artifact_checksum"],
                    "configuration": plan["configuration"],
                    "model": MODEL,
                    "turns": [
                        {
                            "start": t["start"],
                            "end": t["end"],
                            "speaker": t["speaker"],
                            "duration": t["end"] - t["start"],
                        }
                        for t in turns
                    ],
                    "aligned_segment_references": copy.deepcopy(plan["aligned_segment_references"]),
                    "qa": qa,
                    "speaker_identities_assigned": False,
                    "transcript_attribution_performed": False,
                    "full_diarization_complete": not smoke,
                    "mode_processing_complete": True,
                    "execution": {
                        "attempt": cp["attempt"],
                        "started_at": cp["started_at"],
                        "finished_at": utc_timestamp(),
                        "wall_seconds": wall_seconds,
                        "source_audio_seconds": plan["source_duration_seconds"],
                        "processed_audio_seconds": seconds,
                        "realtime_factor": wall_seconds / seconds,
                        "speaker_count": qa["speaker_count"],
                        "turn_count": qa["turn_count"],
                        "total_speech_seconds": qa["speech_union_seconds"],
                        "model_inventory": final_inventory["files"],
                        "immutable_input_checksums": {
                            p.relative_to(config.application.root).as_posix(): sha
                            for p, (sha, _) in baseline.items()
                        },
                    },
                }
            )
            cp.update(status="SUCCEEDED", result=result)
        except (KeyboardInterrupt, SystemExit):
            cp.update(status="INTERRUPTED", failure_code="PROCESS_INTERRUPTED")
            raise
        except Exception as error:
            cp.update(status="FAILED", failure_code=type(error).__name__, failure_phase=phase)
            print(f"Diarization failed: {type(error).__name__}", file=sys.stderr)
        finally:
            cp.update(finished_at=utc_timestamp(), wall_seconds=time.perf_counter() - started)
            atomic_write_json(cp_path, seal_artifact(cp))
            try:
                if clip.exists():
                    require(
                        clip.resolve().is_relative_to((plan_path.parent / "derived").resolve()),
                        "derived diarization clip escapes lineage",
                    )
                    clip.unlink()
            finally:
                del engine
                gc.collect()
        if cp["status"] == "SUCCEEDED":
            publish(config, sid, plan, cp["result"])
    return status_report(
        config,
        sid,
        aligned_fingerprint=plan["aligned_fingerprint"],
        fingerprint=plan["stage_fingerprint"],
    )
