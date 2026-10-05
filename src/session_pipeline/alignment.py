"""Resumable, local Persian forced alignment. Planning/status never import ML."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import math
import os
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from session_pipeline import __version__
from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact,
    seal_artifact,
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
    model_path,
)
from session_pipeline.models import session_id_from_sha256, sha256_file, utc_timestamp

MODEL = {
    "alignment": {
        "repo_id": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
        "revision": "234714078a1398a9db88194c5a40fefe6f376dc1",
    }
}
SETTINGS = {
    "language": "fa",
    "device": "cpu",
    "cpu_threads": 4,
    "unit_seconds": 300,
    "smoke_seconds": 180,
    "max_segment_seconds": 60,
    "sample_rate": 16000,
    "interpolate_method": "ignore",
    "sentence_policy": "ONE_REVIEWED_SEGMENT_NO_NLTK_DOWNLOAD",
    "suspicious_word_min_seconds": 0.02,
    "suspicious_word_max_seconds": 3.0,
    "alignment_contract_version": 1,
}
QA_CODES = (
    "ALIGNED",
    "UNALIGNED_WORD",
    "PARTIAL_ALIGNMENT",
    "ALIGNMENT_FAILED",
    "OUT_OF_BOUNDS",
    "TIMESTAMP_ORDER_ERROR",
    "SUSPICIOUS_WORD_DURATION",
)


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def packages():
    result = {}
    for name in ("whisperx", "torch", "torchaudio", "transformers", "numpy", "pandas", "nltk"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "UNKNOWN"
    return result


def model_spec(config):
    spec = next(
        m for m in load_model_lock(config.application.root)["models"] if m["role"] == "alignment"
    )
    require(
        {"repo_id": spec["repo_id"], "revision": spec["revision"]} == MODEL["alignment"],
        "alignment lock differs from approved exact Persian revision",
    )
    return spec


def load_input(config, sid, reviewed_fingerprint=None):
    raw, work, _ = session_paths(config, sid)
    output = raw.parent
    if reviewed_fingerprint is None:
        pointer = read_artifact(output / "reviewed/latest.json")
        require(
            pointer.get("artifact_type") == "reviewed_asr_pointer"
            and pointer.get("session_id") == sid,
            "invalid reviewed pointer",
        )
        path = (output / pointer["reviewed_json"]).resolve()
    else:
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", reviewed_fingerprint)),
            "invalid reviewed fingerprint",
        )
        path = (
            output / "reviewed" / reviewed_fingerprint / "transcript.reviewed.fa.json"
        ).resolve()
        pointer = None
    require(
        path.is_relative_to((output / "reviewed").resolve())
        and path.name == "transcript.reviewed.fa.json",
        "alignment requires reviewed JSON lineage",
    )
    reviewed = read_artifact(path)
    sha = reviewed["source_sha256"]
    require(
        reviewed.get("artifact_type") == "reviewed_asr"
        and reviewed.get("status") == "SUCCEEDED"
        and reviewed.get("session_id") == sid
        and session_id_from_sha256(sha) == sid
        and reviewed["source"]["sha256"] == sha
        and reviewed["stage_fingerprint"] == path.parent.name,
        "invalid reviewed input identity",
    )
    if pointer:
        require(
            pointer["artifact_checksum_of_reviewed_json"] == reviewed["artifact_checksum"],
            "reviewed pointer checksum mismatch",
        )
    require(
        reviewed["qa_acceptance"].get("alignment_ready") is True and not reviewed.get("unresolved"),
        "reviewed input is not structurally ready",
    )
    duration = reviewed["probe"]["duration_seconds"]
    require(number(duration) and duration > 0, "invalid source duration")
    segments = reviewed["segments"]
    require(
        isinstance(segments, list)
        and bool(segments)
        and len(segments) == reviewed["segment_count"],
        "reviewed segment count mismatch",
    )
    previous = 0
    for s in segments:
        a, b = s["start"], s["end"]
        require(
            number(a)
            and number(b)
            and previous <= a <= b <= duration
            and isinstance(s["text"], str),
            "invalid reviewed segment timeline",
        )
        previous = a
    txt = path.with_suffix(".txt")
    require(
        txt.is_file() and sha256_file(txt) == reviewed["rendered_txt_sha256"],
        "reviewed TXT checksum mismatch",
    )
    integrity = read_artifact(work / "integrity.json")
    require(
        integrity.get("status") == "SUCCEEDED"
        and integrity["decode"]["success"]
        and integrity["source"]["sha256"] == sha,
        "source integrity mismatch",
    )
    name = integrity["source"]["original_filename"]
    require(
        isinstance(name, str) and Path(name).name == name and "/" not in name and "\\" not in name,
        "invalid source filename",
    )
    source = (config.application.sessions_path / name).resolve()
    require(
        source.is_relative_to(config.application.sessions_path.resolve())
        and source.stat().st_size == integrity["source"]["size_bytes"],
        "invalid source file",
    )
    # Read-only status checks identity/size; execute additionally hashes the audio.
    return path, reviewed, source, work, integrity


def build_units(reviewed, smoke):
    duration = reviewed["probe"]["duration_seconds"]
    limit = min(duration, SETTINGS["smoke_seconds"]) if smoke else duration
    units = []
    for index in range(math.ceil(limit / SETTINGS["unit_seconds"])):
        start, end = (
            index * SETTINGS["unit_seconds"],
            min(limit, (index + 1) * SETTINGS["unit_seconds"]),
        )
        indices = [
            n
            for n, s in enumerate(reviewed["segments"])
            if start <= s["start"] < end and (not smoke or s["end"] <= limit)
        ]
        unit = {
            "index": index,
            "ownership_interval": [start, end],
            "decoded_interval": [
                start,
                max([end] + [reviewed["segments"][n]["end"] for n in indices]),
            ],
            "segment_indices": indices,
            "segment_ids": [
                digest({"reviewed_checksum": reviewed["artifact_checksum"], "segment_index": n})
                for n in indices
            ],
        }
        # Include a zero-duration segment at the full session endpoint exactly once.
        if not smoke and index == math.ceil(limit / SETTINGS["unit_seconds"]) - 1:
            for n, s in enumerate(reviewed["segments"]):
                if s["start"] == duration:
                    unit["segment_indices"].append(n)
                    unit["segment_ids"].append(
                        digest(
                            {"reviewed_checksum": reviewed["artifact_checksum"], "segment_index": n}
                        )
                    )
        unit["unit_id"] = digest({"reviewed_checksum": reviewed["artifact_checksum"], "unit": unit})
        units.append(unit)
    owned = [n for u in units for n in u["segment_indices"]]
    expected = [
        n
        for n, s in enumerate(reviewed["segments"])
        if not smoke or s["start"] < limit and s["end"] <= limit
    ]
    require(
        sorted(owned) == expected and len(set(owned)) == len(owned), "segment ownership mismatch"
    )
    require(
        units[0]["ownership_interval"][0] == 0 and units[-1]["ownership_interval"][1] == limit,
        "beginning/tail ownership missing",
    )
    return units


def make_plan(config, sid, *, smoke=False, reviewed_fingerprint=None):
    path, reviewed, source, work, integrity = load_input(config, sid, reviewed_fingerprint)
    spec = model_spec(config)
    units = build_units(reviewed, smoke)
    configuration = {
        **SETTINGS,
        "mode": "SMOKE" if smoke else "FULL",
        "package_versions": packages(),
    }
    parents = {
        "reviewed_asr": reviewed["artifact_checksum"],
        "integrity": integrity["artifact_checksum"],
        "model_lock": digest(spec),
    }
    fp = stage_fingerprint(
        stage="persian_forced_alignment",
        source_sha256=reviewed["source_sha256"],
        relevant_config={"settings": configuration, "units": units},
        parent_checksums=parents,
        models=MODEL,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    plan = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "alignment_plan",
            "status": "PLANNED",
            "session_id": sid,
            "source_sha256": reviewed["source_sha256"],
            "stage_fingerprint": fp,
            "parent_checksums": parents,
            "model": MODEL,
            "model_local_path": spec["destination"],
            "configuration": configuration,
            "reviewed_relative_path": path.relative_to(config.application.root).as_posix(),
            "reviewed_file_sha256": sha256_file(path),
            "reviewed_txt_sha256": sha256_file(path.with_suffix(".txt")),
            "source_relative_path": source.relative_to(config.application.root).as_posix(),
            "duration_seconds": reviewed["probe"]["duration_seconds"],
            "units": units,
            "planned_audio_seconds": sum(
                u["decoded_interval"][1] - u["decoded_interval"][0] for u in units
            ),
            "owned_audio_seconds": sum(
                u["ownership_interval"][1] - u["ownership_interval"][0] for u in units
            ),
            "owned_segment_count": sum(len(u["segment_indices"]) for u in units),
            "execution_blockers": ["reviewed segment exceeds conservative 60-second memory limit"]
            if any(
                reviewed["segments"][n]["end"] - reviewed["segments"][n]["start"]
                > SETTINGS["max_segment_seconds"]
                for u in units
                for n in u["segment_indices"]
            )
            else [],
            "cloud_llm_required": False,
        }
    )
    directory = work / "alignment" / fp
    require(directory.resolve().is_relative_to(work.resolve()), "alignment work escapes session")
    plan_path = directory / "plan.json"
    if plan_path.exists():
        require(read_artifact(plan_path) == plan, "alignment plan changed")
    else:
        atomic_write_json(plan_path, plan)
    return plan_path, plan


def tokens(text):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def normalize_segment(original, returned, unit, segment_index, segment_id, duration):
    """Attach timings to authoritative text tokens, never copy generated wording."""
    expected = tokens(original["text"])
    supplied = returned.get("words", [])
    require(isinstance(supplied, list), "invalid alignment words")
    mismatch = [w.get("word") for w in supplied] != [w for w, _, _ in expected]
    lexical_failure = mismatch or returned.get("failed", False)
    words, issues = [], set()
    previous = None
    offset = unit["decoded_interval"][0]
    for i, (text, char_start, char_end) in enumerate(expected):
        entry = {
            "word": text,
            "reviewed_word_index": i,
            "character_interval": [char_start, char_end],
            "start": None,
            "end": None,
            "score": None,
            "status": "UNALIGNED_WORD",
            "qa_flags": [],
        }
        candidate = {} if lexical_failure else supplied[i]
        a, b = candidate.get("start"), candidate.get("end")
        if number(a) and number(b):
            a, b = a + offset, b + offset
            entry.update(start=a, end=b, status="ALIGNED")
            score = candidate.get("score")
            entry["score"] = score if number(score) else None
            if not (
                0 <= a <= b <= duration
                and unit["decoded_interval"][0] <= a <= b <= unit["decoded_interval"][1]
            ):
                entry["qa_flags"].append("OUT_OF_BOUNDS")
            if previous is not None and a < previous or b < a:
                entry["qa_flags"].append("TIMESTAMP_ORDER_ERROR")
            previous = a
            if (
                b - a < SETTINGS["suspicious_word_min_seconds"]
                or b - a > SETTINGS["suspicious_word_max_seconds"]
            ):
                entry["qa_flags"].append("SUSPICIOUS_WORD_DURATION")
        else:
            entry["qa_flags"].append("UNALIGNED_WORD")
            if a is not None or b is not None:
                entry["qa_flags"].append("OUT_OF_BOUNDS")
        issues.update(entry["qa_flags"])
        words.append(entry)
    aligned = sum(w["status"] == "ALIGNED" for w in words)
    status = (
        "ALIGNMENT_FAILED"
        if lexical_failure or expected and aligned == 0
        else ("PARTIAL_ALIGNMENT" if aligned < len(words) else "ALIGNED")
    )
    issues.add(status)
    return {
        "reviewed_segment_index": segment_index,
        "reviewed_segment_id": segment_id,
        "text": original["text"],
        "start": original["start"],
        "end": original["end"],
        "review_origin": copy.deepcopy(original.get("review_origin")),
        "words": words,
        "status": status,
        "qa_flags": sorted(issues),
        "alignment_unit_id": unit["unit_id"],
        "alignment_diagnostics": {
            "lexical_mapping_failed": mismatch,
            "engine_failure_code": returned.get("failure_code"),
        },
    }


def validate_segments(segments, reviewed, unit, duration):
    require(
        [s["reviewed_segment_index"] for s in segments] == unit["segment_indices"]
        and [s["reviewed_segment_id"] for s in segments] == unit["segment_ids"],
        "checkpoint segment ownership changed",
    )
    previous = None
    for s in segments:
        original = reviewed["segments"][s["reviewed_segment_index"]]
        require(
            s["text"] == original["text"]
            and s["start"] == original["start"]
            and s["end"] == original["end"]
            and s["alignment_unit_id"] == unit["unit_id"],
            "checkpoint rewrites reviewed text or timestamps",
        )
        require(
            [w["word"] for w in s["words"]] == [w for w, _, _ in tokens(original["text"])],
            "checkpoint word ownership changed",
        )
        for w in s["words"]:
            a, b = w["start"], w["end"]
            if a is None and b is None:
                continue
            require(
                number(a)
                and number(b)
                and 0 <= a <= b <= duration
                and unit["decoded_interval"][0] <= a <= b <= unit["decoded_interval"][1],
                "aligned word outside source/unit bounds",
            )
            require(previous is None or a >= previous, "aligned word timestamp order error")
            previous = a


def checkpoint_path(plan_path, unit):
    return plan_path.parent / "units" / f"{unit['unit_id']}.json"


def checkpoint_identity(plan, unit):
    return {
        "plan": plan["artifact_checksum"],
        "reviewed_asr": plan["parent_checksums"]["reviewed_asr"],
        "model_lock": plan["parent_checksums"]["model_lock"],
        "unit": digest(unit),
    }


def read_checkpoint(path, plan, unit, reviewed):
    cp = read_artifact(path)
    require(
        cp.get("artifact_type") == "alignment_unit"
        and cp.get("session_id") == plan["session_id"]
        and cp.get("source_sha256") == plan["source_sha256"]
        and cp.get("model") == MODEL
        and cp.get("parent_checksums") == checkpoint_identity(plan, unit)
        and cp.get("unit") == unit
        and cp.get("status") in {"RUNNING", "FAILED", "SUCCEEDED"},
        "alignment checkpoint identity mismatch",
    )
    if cp["status"] == "SUCCEEDED":
        validate_segments(cp["segments"], reviewed, unit, plan["duration_seconds"])
        require(
            not any(s["status"] == "ALIGNMENT_FAILED" for s in cp["segments"]),
            "successful checkpoint hides alignment failure",
        )
    return cp


def status_report(config, sid, *, smoke=False, reviewed_fingerprint=None, plan_path=None):
    # In memory only: status never writes a plan, touches a model, or imports ML.
    path, reviewed, _, work, integrity = load_input(config, sid, reviewed_fingerprint)
    spec = model_spec(config)
    candidates = []
    if plan_path is None:
        for p in sorted((work / "alignment").glob("*/plan.json")):
            plan = read_artifact(p)
            if plan["parent_checksums"]["reviewed_asr"] == reviewed["artifact_checksum"] and (
                plan["configuration"]["mode"] == ("SMOKE" if smoke else "FULL")
            ):
                candidates.append(p)
        require(len(candidates) <= 1, "multiple alignment plans; specify --fingerprint")
        plan_path = candidates[0] if candidates else None
    if plan_path is None:
        return {
            "session_id": sid,
            "reviewed_checksum": reviewed["artifact_checksum"],
            "reviewed_path": str(path),
            "model": MODEL,
            "status": "NOT_PLANNED",
            "planned_unit_count": 0,
            "completed_units": 0,
            "failed_units": 0,
            "full_alignment_complete": False,
            "models_loaded": False,
        }
    require(
        plan_path.resolve().is_relative_to((work / "alignment").resolve()),
        "invalid alignment plan path",
    )
    plan = read_artifact(plan_path)
    require(plan_path.parent.name == plan["stage_fingerprint"], "alignment plan directory mismatch")
    validate_plan(config, plan, path, reviewed, integrity, spec)
    counts = Counter()
    qa = Counter()
    known = []
    audio, processed, count, elapsed = 0.0, 0.0, 0, 0.0
    unaligned = 0
    segments = []
    for unit in plan["units"]:
        cp_path = checkpoint_path(plan_path, unit)
        cp = None
        state = "PENDING"
        if cp_path.exists():
            try:
                cp = read_checkpoint(cp_path, plan, unit, reviewed)
                state = cp["status"]
            except (ValueError, KeyError, TypeError):
                state = "INVALID_CHECKPOINT"
        counts[state] += 1
        known.append({"unit_id": unit["unit_id"], "status": state})
        if cp:
            elapsed += cp.get("wall_seconds", 0)
            processed += cp.get("processed_audio_seconds", 0)
            for s in cp.get("segments", []):
                qa.update(s["qa_flags"])
                unaligned += sum(w["status"] == "UNALIGNED_WORD" for w in s["words"])
        if state == "SUCCEEDED":
            count += len(unit["segment_indices"])
            audio += unit["ownership_interval"][1] - unit["ownership_interval"][0]
            segments.extend(cp["segments"])
    complete = counts["SUCCEEDED"] == len(plan["units"])
    ordered = True
    coverage_errors = []
    if complete:
        try:
            validate_coverage(plan, reviewed, segments)
        except ValueError as error:
            ordered = False
            coverage_errors.append(str(error))
            qa["TIMESTAMP_ORDER_ERROR"] += 1
    return {
        "session_id": sid,
        "reviewed_path": str(path),
        "reviewed_checksum": reviewed["artifact_checksum"],
        "model": MODEL,
        "model_local_path": str(model_path(config.application.root, spec)),
        "plan_path": str(plan_path),
        "fingerprint": plan["stage_fingerprint"],
        "mode": plan["configuration"]["mode"],
        "planned_unit_count": len(plan["units"]),
        "planned_audio_seconds": plan["planned_audio_seconds"],
        "completed_units": counts["SUCCEEDED"],
        "failed_units": counts["FAILED"],
        "unit_status_counts": dict(counts),
        "units": known,
        "completed_owned_audio_seconds": audio,
        "processed_audio_seconds": processed,
        "owned_segment_count": plan["owned_segment_count"],
        "completed_segment_count": count,
        "unaligned_word_count": unaligned,
        "qa_counts": dict(qa),
        "wall_seconds": elapsed,
        "mode_processing_complete": complete and ordered,
        "coverage_validation_pass": complete and ordered,
        "coverage_errors": coverage_errors,
        "full_alignment_complete": complete and ordered and plan["configuration"]["mode"] == "FULL",
        "models_loaded": False,
        "cloud_llm_required": False,
    }


def validate_plan(config, plan, path, reviewed, integrity, spec):
    configuration = {
        **SETTINGS,
        "mode": plan["configuration"]["mode"],
        "package_versions": packages(),
    }
    require(configuration["mode"] in {"FULL", "SMOKE"}, "invalid alignment mode")
    units = build_units(reviewed, configuration["mode"] == "SMOKE")
    parents = {
        "reviewed_asr": reviewed["artifact_checksum"],
        "integrity": integrity["artifact_checksum"],
        "model_lock": digest(spec),
    }
    fp = stage_fingerprint(
        stage="persian_forced_alignment",
        source_sha256=reviewed["source_sha256"],
        relevant_config={"settings": configuration, "units": units},
        parent_checksums=parents,
        models=MODEL,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    require(
        plan.get("artifact_type") == "alignment_plan"
        and plan.get("status") == "PLANNED"
        and plan["stage_fingerprint"] == fp
        and plan["parent_checksums"] == parents
        and plan["units"] == units
        and plan["configuration"] == configuration
        and plan["model"] == MODEL
        and plan["source_sha256"] == reviewed["source_sha256"]
        and plan["session_id"] == reviewed["session_id"]
        and plan["duration_seconds"] == reviewed["probe"]["duration_seconds"]
        and plan["reviewed_relative_path"] == path.relative_to(config.application.root).as_posix()
        and plan["reviewed_file_sha256"] == sha256_file(path)
        and plan["reviewed_txt_sha256"] == sha256_file(path.with_suffix(".txt")),
        "alignment plan or immutable reviewed parent changed",
    )


def validate_coverage(plan, reviewed, segments):
    expected = [n for u in plan["units"] for n in u["segment_indices"]]
    actual = [s["reviewed_segment_index"] for s in segments]
    require(
        actual == expected and len(set(actual)) == len(actual), "alignment segment coverage failed"
    )
    if plan["configuration"]["mode"] == "FULL":
        require(actual == list(range(len(reviewed["segments"]))), "full reviewed coverage failed")
    previous = None
    for unit in plan["units"]:
        own = [s for s in segments if s["alignment_unit_id"] == unit["unit_id"]]
        validate_segments(own, reviewed, unit, plan["duration_seconds"])
    for s in segments:
        for w in s["words"]:
            if w["start"] is not None:
                require(
                    previous is None or w["start"] >= previous,
                    "global alignment timestamp order error",
                )
                previous = w["start"]
    require(plan["units"][0]["ownership_interval"][0] == 0, "beginning not processed")
    if plan["configuration"]["mode"] == "FULL":
        require(
            plan["units"][-1]["ownership_interval"][1] == plan["duration_seconds"],
            "tail not processed",
        )


class WholeSegmentSentence:
    def span_tokenize(self, text):
        return [(0, len(text))]


class LocalAligner:
    """Only constructed by explicit execute; no ASR imports or model loading."""

    def __init__(self, root):
        path = local_model_path(root, "alignment")
        require(path.name == MODEL["alignment"]["revision"], "wrong local alignment revision")
        import torch
        import whisperx.alignment as wx

        torch.set_num_threads(SETTINGS["cpu_threads"])
        self.wx = wx
        self.model, self.metadata = wx.load_align_model(
            language_code="fa", device="cpu", model_name=str(path), model_cache_only=True
        )
        self.model.eval()

    def __call__(self, originals, unit, clip):
        import numpy as np

        with wave.open(str(clip), "rb") as stream:
            require(
                stream.getnchannels() == 1
                and stream.getframerate() == 16000
                and stream.getsampwidth() == 2,
                "invalid alignment PCM",
            )
            audio = (
                np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
            require(
                abs(
                    len(audio) / 16000 - (unit["decoded_interval"][1] - unit["decoded_interval"][0])
                )
                <= 0.12,
                "alignment clip duration mismatch",
            )
        result = []
        # WhisperX's default Punkt fallback downloads data; each authoritative
        # reviewed segment is its own sentence here. The patch is process-local.
        with (
            patch.object(self.wx, "nltk_load", return_value=WholeSegmentSentence()),
            patch.object(
                self.wx.nltk, "download", side_effect=RuntimeError("NLTK download forbidden")
            ),
            patch.object(self.wx.logger, "disabled", True),
        ):
            for original in originals:
                local = {
                    "start": original["start"] - unit["decoded_interval"][0],
                    "end": original["end"] - unit["decoded_interval"][0],
                    "text": original["text"],
                }
                if local["end"] <= local["start"]:
                    result.append(
                        {"words": [], "failed": True, "failure_code": "ZERO_DURATION_SEGMENT"}
                    )
                    continue
                try:
                    aligned = self.wx.align(
                        [local],
                        self.model,
                        self.metadata,
                        audio,
                        "cpu",
                        interpolate_method="ignore",
                        return_char_alignments=False,
                        print_progress=False,
                    )
                    result.append({"words": aligned["word_segments"]})
                except Exception as error:
                    # Transcript-bearing exception strings stay out of logs and artifacts.
                    result.append(
                        {"words": [], "failed": True, "failure_code": type(error).__name__}
                    )
        return result


def run_alignment(
    config,
    sid,
    *,
    smoke=False,
    reviewed_fingerprint=None,
    engine_factory=LocalAligner,
    extractor=extract_clip,
):
    plan_path, plan = make_plan(config, sid, smoke=smoke, reviewed_fingerprint=reviewed_fingerprint)
    path, reviewed, source, _, integrity = load_input(config, sid, path_fingerprint(plan))
    spec = model_spec(config)
    validate_plan(config, plan, path, reviewed, integrity, spec)
    require(not plan["execution_blockers"], "alignment plan has execution blockers")
    immutable = [path, path.with_suffix(".txt"), source]
    raw = path.parents[2] / "transcript.raw.fa.json"
    immutable.extend(p for p in (raw, raw.with_suffix(".txt")) if p.exists())
    baseline = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in immutable}
    require(baseline[source][0] == reviewed["source_sha256"], "source audio SHA-256 mismatch")
    engine = None
    with (
        exclusive_lock(config.application.work_path / sid / "alignment/.worker.lock"),
        patch.dict(os.environ, local_only_environment(config.application.root), clear=True),
        deny_network(),
    ):
        try:
            for unit in plan["units"]:
                cp_path = checkpoint_path(plan_path, unit)
                old = read_checkpoint(cp_path, plan, unit, reviewed) if cp_path.exists() else None
                if old and old["status"] == "SUCCEEDED":
                    continue
                checkpoint = {
                    "schema_version": 1,
                    "artifact_type": "alignment_unit",
                    "status": "RUNNING",
                    "session_id": sid,
                    "source_sha256": plan["source_sha256"],
                    "model": MODEL,
                    "unit": unit,
                    "parent_checksums": checkpoint_identity(plan, unit),
                    "attempt": old.get("attempt", 0) + 1 if old else 1,
                    "started_at": utc_timestamp(),
                    "segments": [],
                    "processed_audio_seconds": 0,
                    "wall_seconds": 0,
                }
                atomic_write_json(cp_path, seal_artifact(checkpoint))
                started = time.perf_counter()
                clip = plan_path.parent / "derived" / f"{unit['unit_id']}.wav"
                require(
                    clip.resolve().is_relative_to(plan_path.parent.resolve())
                    and clip.resolve() != source.resolve()
                    and not clip.is_symlink(),
                    "unsafe derived alignment clip destination",
                )
                try:
                    if unit["segment_indices"]:
                        if engine is None:
                            engine = engine_factory(config.application.root)
                        extractor(source, clip, unit["decoded_interval"])
                        originals = [reviewed["segments"][n] for n in unit["segment_indices"]]
                        returned = engine(originals, unit, clip)
                        require(
                            isinstance(returned, list) and len(returned) == len(originals),
                            "alignment engine lost reviewed segments",
                        )
                        checkpoint["segments"] = [
                            normalize_segment(s, r, unit, n, segment_id, plan["duration_seconds"])
                            for s, r, n, segment_id in zip(
                                originals,
                                returned,
                                unit["segment_indices"],
                                unit["segment_ids"],
                                strict=True,
                            )
                        ]
                    checkpoint["processed_audio_seconds"] = (
                        unit["decoded_interval"][1] - unit["decoded_interval"][0]
                    )
                    validate_segments(
                        checkpoint["segments"], reviewed, unit, plan["duration_seconds"]
                    )
                    require(
                        not any(s["status"] == "ALIGNMENT_FAILED" for s in checkpoint["segments"]),
                        "unit contains failed segment alignment",
                    )
                    checkpoint["status"] = "SUCCEEDED"
                except Exception as error:
                    checkpoint.update(status="FAILED", failure_code=type(error).__name__)
                    if not checkpoint["segments"]:
                        checkpoint["segments"] = [
                            normalize_segment(
                                reviewed["segments"][n],
                                {"words": [], "failed": True, "failure_code": type(error).__name__},
                                unit,
                                n,
                                segment_id,
                                plan["duration_seconds"],
                            )
                            for n, segment_id in zip(
                                unit["segment_indices"], unit["segment_ids"], strict=True
                            )
                        ]
                finally:
                    # Only our ignored derived PCM is ephemeral; never the original.
                    if clip.exists():
                        require(
                            clip.resolve().is_relative_to((plan_path.parent / "derived").resolve()),
                            "derived clip escapes alignment directory",
                        )
                        clip.unlink()
                    checkpoint.update(
                        wall_seconds=(old.get("wall_seconds", 0) if old else 0)
                        + time.perf_counter()
                        - started,
                        finished_at=utc_timestamp(),
                    )
                    checkpoint["processed_audio_seconds"] += (
                        old.get("processed_audio_seconds", 0) if old else 0
                    )
                    atomic_write_json(cp_path, seal_artifact(checkpoint))
                report = status_report(
                    config,
                    sid,
                    smoke=smoke,
                    reviewed_fingerprint=path_fingerprint(plan),
                    plan_path=plan_path,
                )
                atomic_write_json(
                    plan_path.parent / "progress.json",
                    seal_artifact(
                        {
                            "schema_version": 1,
                            "artifact_type": "alignment_progress",
                            "plan_checksum": plan["artifact_checksum"],
                            "updated_at": utc_timestamp(),
                            **report,
                        }
                    ),
                )
                print(
                    f"Alignment unit {unit['index'] + 1}/{len(plan['units'])}: "
                    f"{checkpoint['status']}",
                    file=sys.stderr,
                    flush=True,
                )
            require(
                all(
                    sha256_file(p) == sha and p.stat().st_mtime_ns == mtime
                    for p, (sha, mtime) in baseline.items()
                ),
                "immutable source/raw/reviewed input changed during alignment",
            )
            current = load_input(config, sid, path_fingerprint(plan))
            validate_plan(config, plan, current[0], current[1], current[4], model_spec(config))
            report = status_report(
                config,
                sid,
                smoke=smoke,
                reviewed_fingerprint=path_fingerprint(plan),
                plan_path=plan_path,
            )
            if report["mode_processing_complete"]:
                segments = [
                    s
                    for u in plan["units"]
                    for s in read_checkpoint(checkpoint_path(plan_path, u), plan, u, reviewed)[
                        "segments"
                    ]
                ]
                validate_coverage(plan, reviewed, segments)
                output_root = path.parents[2] / "aligned"
                if smoke:
                    output_root = output_root / "smoke"
                output = output_root / plan["stage_fingerprint"]
                require(
                    output.resolve().is_relative_to((path.parents[2] / "aligned").resolve()),
                    "aligned output escapes lineage",
                )
                aligned_path = output / "transcript.aligned.fa.json"
                document = seal_artifact(
                    {
                        "schema_version": 1,
                        "artifact_type": "aligned_asr",
                        "status": "SUCCEEDED",
                        "session_id": sid,
                        "source_sha256": plan["source_sha256"],
                        "model": MODEL,
                        "stage_fingerprint": plan["stage_fingerprint"],
                        "parent_checksums": plan["parent_checksums"],
                        "plan_checksum": plan["artifact_checksum"],
                        "configuration": plan["configuration"],
                        "source": reviewed["source"],
                        "probe": reviewed["probe"],
                        "segments": segments,
                        "segment_count": len(segments),
                        "qa": {
                            k: report[k]
                            for k in (
                                "unaligned_word_count",
                                "qa_counts",
                                "coverage_validation_pass",
                                "full_alignment_complete",
                            )
                        },
                        "unit_checksums": {
                            u["unit_id"]: read_checkpoint(
                                checkpoint_path(plan_path, u), plan, u, reviewed
                            )["artifact_checksum"]
                            for u in plan["units"]
                        },
                        "reviewed_text_authoritative": True,
                        "cloud_llm_required": False,
                    }
                )
                if aligned_path.exists():
                    require(
                        read_artifact(aligned_path) == document, "existing aligned lineage changed"
                    )
                else:
                    atomic_write_json(aligned_path, document)
                report["aligned_json"] = str(aligned_path)
            else:
                report["aligned_json"] = None
        finally:
            del engine
            gc.collect()
    return report


def path_fingerprint(plan):
    return Path(plan["reviewed_relative_path"]).parent.name


def plan_report(config, plan_path, plan):
    spec = model_spec(config)
    return {
        "session_id": plan["session_id"],
        "status": "PLANNED",
        "plan_path": str(plan_path),
        "fingerprint": plan["stage_fingerprint"],
        "mode": plan["configuration"]["mode"],
        "reviewed_input": plan["reviewed_relative_path"],
        "reviewed_checksum": plan["parent_checksums"]["reviewed_asr"],
        "source_duration_seconds": plan["duration_seconds"],
        "unit_count": len(plan["units"]),
        "planned_audio_seconds": plan["planned_audio_seconds"],
        "owned_audio_seconds": plan["owned_audio_seconds"],
        "owned_segment_count": plan["owned_segment_count"],
        "model": MODEL,
        "model_local_path": str(model_path(config.application.root, spec)),
        "model_file_status": inspect_model(config.application.root, spec)["status"],
        "execution_blockers": plan["execution_blockers"],
        "real_alignment_executed": False,
    }
