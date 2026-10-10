"""Bounded, resumable local retry candidates. Canonical ASR is never a write target."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

from session_pipeline import __version__
from session_pipeline.artifacts import (
    Checkpoint,
    atomic_write_json,
    checkpoint_reusable,
    read_artifact,
    seal_artifact,
)
from session_pipeline.asr_quality import (
    QualityThresholds,
    load_canonical,
    number,
    pad,
    session_paths,
    union,
)
from session_pipeline.config import AppConfig, load_config
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.model_store import local_model_path, local_only_environment
from session_pipeline.models import StageStatus, sha256_file, utc_timestamp
from session_pipeline.product_defaults import ASR_DECODE

MODEL = {
    "asr": {
        "repo_id": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    }
}
DECODE = ASR_DECODE


def package_identity() -> dict[str, str]:
    versions = {}
    for name in ("faster-whisper", "ctranslate2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "UNKNOWN"
    return versions


def make_plan(
    config: AppConfig,
    session_id: str,
    entry_ids: list[str] | None = None,
    max_seconds: float = 900.0,
) -> tuple[Path, dict[str, Any]]:
    raw_path, work, manifest_path = session_paths(config, session_id)
    raw = load_canonical(raw_path, session_id)
    manifest = read_artifact(manifest_path)
    if (
        manifest.get("artifact_type") != "asr_review"
        or manifest["status"] != "SUCCEEDED"
        or manifest["session_id"] != session_id
        or manifest["source_sha256"] != raw["source_sha256"]
        or manifest["parent_checksums"]["canonical_asr"] != raw["artifact_checksum"]
    ):
        raise ValueError("review is stale or belongs to a different canonical result")
    if raw["model"] != MODEL:
        raise ValueError("retry requires the approved original pinned large-v3 model")
    entries = manifest["entries"]
    if entry_ids is not None:
        known = {e["entry_id"] for e in entries}
        if set(entry_ids) - known:
            raise ValueError("unknown review entry ID")
        entries = [e for e in entries if e["entry_id"] in entry_ids]
    duration = raw["probe"]["duration_seconds"]
    thresholds = manifest["thresholds"]
    QualityThresholds(**thresholds)
    targets = union([e["interval"] for e in entries])
    windows = []
    for start, end in targets:
        cursor = start
        while cursor < end:
            right = min(end, cursor + thresholds["retry_max_target_seconds"])
            owned = [cursor, right]
            windows.append(
                {
                    "window_index": len(windows),
                    "target_interval": owned,
                    "decoded_interval": pad(owned, thresholds["retry_padding_seconds"], duration),
                    "entry_ids": [
                        e["entry_id"]
                        for e in entries
                        if e["interval"][1] > cursor and e["interval"][0] < right
                    ],
                    "original_segment_indices": [
                        n
                        for n, s in enumerate(raw["segments"])
                        if s["end"] > cursor and s["start"] < right
                    ],
                }
            )
            cursor = right
    decoded_seconds = sum(b - a for a, b in [w["decoded_interval"] for w in windows])
    target_seconds = sum(b - a for a, b in targets)
    if not number(max_seconds) or not 0 < max_seconds <= 1800:
        raise ValueError("retry budget must be (0,1800] seconds")
    blockers = []
    if decoded_seconds > max_seconds:
        blockers.append("selected workload exceeds decoded-audio budget; select fewer --entry IDs")
    if target_seconds >= duration * 0.5 and windows:
        blockers.append("retry targets cover at least half the source; narrow the selected entries")
    if any(b - a >= duration for a, b in [w["decoded_interval"] for w in windows]):
        blockers.append("a padded clip would cover the entire recording")
    decoded_union = union([w["decoded_interval"] for w in windows])
    if windows and sum(b - a for a, b in decoded_union) >= duration:
        blockers.append("combined padded clips would cover the entire recording")
    parents = {"canonical_asr": raw["artifact_checksum"], "review": manifest["artifact_checksum"]}
    relevant = {
        "decode": DECODE,
        "windows": windows,
        "budget_seconds": max_seconds,
        "package_versions": package_identity(),
        "selected_entry_ids": sorted(e["entry_id"] for e in entries),
    }
    fp = stage_fingerprint(
        stage="asr_targeted_retry",
        source_sha256=raw["source_sha256"],
        relevant_config=relevant,
        parent_checksums=parents,
        models=MODEL,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    path = work / "qa/retries" / fp / "plan.json"
    if path.exists():
        old = read_artifact(path)
        if old.get("stage_fingerprint") == fp and old.get("status") == "SUCCEEDED":
            if (
                old.get("configuration") != relevant
                or old.get("windows") != windows
                or old.get("parent_checksums") != parents
                or old.get("model") != MODEL
                or old.get("execution_allowed") != (not blockers)
                or old.get("execution_blockers") != blockers
                or old.get("source_sha256") != raw["source_sha256"]
                or old.get("session_id") != session_id
            ):
                raise ValueError("existing retry plan semantics changed")
            return path, old
    plan = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "asr_retry_plan",
            "status": "SUCCEEDED",
            "session_id": session_id,
            "source_sha256": raw["source_sha256"],
            "stage_fingerprint": fp,
            "parent_checksums": parents,
            "model": MODEL,
            "configuration": relevant,
            "windows": windows,
            "created_at": utc_timestamp(),
            "target_audio_seconds": target_seconds,
            "decoded_audio_seconds": decoded_seconds,
            "execution_allowed": not blockers,
            "execution_blockers": blockers,
            "canonical_replacement": False,
            "cloud_llm_required": False,
        }
    )
    atomic_write_json(path, plan)
    return path, plan


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """OS releases the lock after a crash; the inert lock file may remain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def retry_fingerprint(
    plan: dict[str, Any], window: dict[str, Any], parent_checksums: dict[str, str]
) -> str:
    return stage_fingerprint(
        stage="asr_retry_window",
        source_sha256=plan["source_sha256"],
        relevant_config={
            "window": window,
            "decode": DECODE,
            "packages": plan["configuration"]["package_versions"],
        },
        parent_checksums=parent_checksums,
        models=MODEL,
        pipeline_version=__version__,
        stage_schema_version=1,
    )


def extract_clip(source: Path, clip: Path, interval: list[float]) -> None:
    start, end = interval
    clip.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(end - start),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(clip),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError("derived retry clip decode failed")
    with wave.open(str(clip), "rb") as stream:
        actual = stream.getnframes() / stream.getframerate()
        if (
            stream.getnchannels() != 1
            or stream.getframerate() != 16000
            or abs(actual - (end - start)) > 0.12
        ):
            raise ValueError("derived clip metadata/duration mismatch")


@contextlib.contextmanager
def deny_network():
    import socket

    def blocked(*args, **kwargs):
        raise RuntimeError("network disabled for local ASR retry")

    saved = (socket.create_connection, socket.socket.connect, socket.socket.connect_ex)
    socket.create_connection = socket.socket.connect = socket.socket.connect_ex = blocked
    try:
        yield
    finally:
        socket.create_connection, socket.socket.connect, socket.socket.connect_ex = saved


def local_engine(root: Path):
    path = local_model_path(root, "asr")  # Checks lock, files and full payload checksums offline.
    if path.name != MODEL["asr"]["revision"]:
        raise ValueError("wrong local model revision")
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(path),
        device=DECODE["device"],
        compute_type=DECODE["compute_type"],
        cpu_threads=DECODE["cpu_threads"],
        num_workers=DECODE["num_workers"],
        local_files_only=True,
        use_auth_token=False,
    )
    if model.model.compute_type not in {"int8", "int8_float32"}:
        raise ValueError("requested INT8 runtime unavailable")
    return model


def transcribe_clip(
    engine, clip: Path, offset: float, *, checkpoint=None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    settings = {
        key: DECODE[key]
        for key in (
            "language",
            "beam_size",
            "best_of",
            "word_timestamps",
            "vad_filter",
            "condition_on_previous_text",
            "initial_prompt",
            "hotwords",
            "temperature",
        )
    }
    iterator, info = engine.transcribe(str(clip), **settings)
    segments = []
    for native in iterator:  # Consume the lazy iterator completely before marking success.
        segment = asdict(native)
        segment["start"] += offset
        segment["end"] += offset
        for word in segment.get("words") or []:
            word["start"] += offset
            word["end"] += offset
        segments.append(segment)
        if checkpoint is not None:
            checkpoint(segments)
    return segments, {
        "elapsed_seconds": time.perf_counter() - started,
        "decoded_duration": info.duration,
        "language": info.language,
        "actual_compute_type": engine.model.compute_type,
    }


def owned_segments(segments: list[dict[str, Any]], target: list[float]) -> list[dict[str, Any]]:
    result = []
    start, end = target
    for s in segments:
        words = [w for w in s.get("words") or [] if start <= (w["start"] + w["end"]) / 2 < end]
        if words:
            result.append(
                {
                    **s,
                    "start": max(start, words[0]["start"]),
                    "end": min(end, words[-1]["end"]),
                    "words": words,
                    "text": "".join(w["word"] for w in words).strip(),
                }
            )
        elif not s.get("words") and start <= (s["start"] + s["end"]) / 2 < end:
            result.append({**s, "start": max(start, s["start"]), "end": min(end, s["end"])})
    return result


def run_retry(
    config: AppConfig,
    session_id: str,
    plan_path: Path,
    *,
    engine_factory=local_engine,
    extractor=extract_clip,
    inference=transcribe_clip,
) -> dict[str, Any]:
    raw_path, work, review_path = session_paths(config, session_id)
    plan_path = plan_path.resolve()
    if not plan_path.is_relative_to((work / "qa/retries").resolve()):
        raise ValueError("retry plan escapes session retry directory")
    plan = read_artifact(plan_path)
    if not plan["execution_allowed"]:
        raise ValueError("retry workload blocked: " + "; ".join(plan["execution_blockers"]))
    # Recompute the approved plan, never trust an edited execution_allowed boolean.
    selected = plan["configuration"]["selected_entry_ids"]
    _, current = make_plan(config, session_id, selected, plan["configuration"]["budget_seconds"])
    if current["artifact_checksum"] != plan["artifact_checksum"]:
        raise ValueError("retry plan identity changed")
    integrity_path = work / "integrity.json"
    integrity = read_artifact(integrity_path)
    if (
        integrity["status"] != "SUCCEEDED"
        or not integrity["decode"]["success"]
        or integrity["source"]["sha256"] != plan["source_sha256"]
    ):
        raise ValueError("valid matching integrity parent required")
    raw = load_canonical(raw_path, session_id)
    name = integrity["source"]["original_filename"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("invalid source filename")
    source = (config.application.sessions_path / name).resolve()
    if not source.is_relative_to(config.application.sessions_path.resolve()):
        raise ValueError("source escapes sessions directory")
    baseline = source.stat()

    def unchanged():
        stat = source.stat()
        if (
            stat.st_size != integrity["source"]["size_bytes"]
            or stat.st_mtime_ns != baseline.st_mtime_ns
            or sha256_file(source) != plan["source_sha256"]
            or read_artifact(raw_path)["artifact_checksum"] != raw["artifact_checksum"]
        ):
            raise ValueError("source or canonical ASR changed during retry")

    unchanged()
    parents = {
        "canonical_asr": raw_path,
        "review": review_path,
        "plan": plan_path,
        "integrity": integrity_path,
    }
    hashes = {key: read_artifact(path)["artifact_checksum"] for key, path in parents.items()}

    def parents_unchanged():
        if any(
            read_artifact(path)["artifact_checksum"] != hashes[key] for key, path in parents.items()
        ):
            raise ValueError("retry parent changed")

    directory = plan_path.parent

    def save_progress(status, completed, error_type=None):
        atomic_write_json(
            directory / "progress.json",
            seal_artifact(
                {
                    "schema_version": 1,
                    "artifact_type": "asr_retry_progress",
                    "status": status,
                    "session_id": session_id,
                    "source_sha256": plan["source_sha256"],
                    "created_at": utc_timestamp(),
                    "parent_checksums": hashes,
                    "completed_windows": completed,
                    "expected_windows": len(plan["windows"]),
                    "error_type": error_type,
                    "canonical_replacement": False,
                }
            ),
        )

    with exclusive_lock(work / "qa/.retry.lock"), deny_network():
        engine = None
        completed = 0
        try:
            for window in plan["windows"]:
                path = directory / f"candidate-{window['window_index']:04}.json"
                fp = retry_fingerprint(plan, window, hashes)
                if path.exists():
                    value = read_artifact(path)
                    checkpoint = Checkpoint(
                        StageStatus(value["status"]),
                        value["stage_fingerprint"],
                        value["artifact_checksum"],
                        value["parent_checksums"],
                    )
                    if checkpoint_reusable(
                        checkpoint,
                        expected_fingerprint=fp,
                        artifact_path=path,
                        required_parents=parents,
                    ):
                        completed += 1
                        save_progress("RUNNING", completed)
                        continue
                if engine is None:
                    engine = engine_factory(config.application.root)
                save_progress("RUNNING", completed)
                clip = directory / f"derived/retry-{window['window_index']:04}.wav"
                if not clip.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("derived clip escapes retry directory")
                extractor(source, clip, window["decoded_interval"])
                retry, diagnostics = inference(engine, clip, window["decoded_interval"][0])
                left, right = window["decoded_interval"]
                previous = left
                for segment in retry:
                    for timed in [segment, *(segment.get("words") or [])]:
                        a, b = timed["start"], timed["end"]
                        if not number(a) or not number(b) or not left <= a <= b <= right + 0.12:
                            raise ValueError("invalid retry timestamps")
                    if segment["start"] < previous:
                        raise ValueError("unordered retry segments")
                    previous = segment["start"]
                unchanged()
                parents_unchanged()
                candidate_a = copy.deepcopy(
                    [raw["segments"][n] for n in window["original_segment_indices"]]
                )
                candidate_b = owned_segments(retry, window["target_interval"])
                document = seal_artifact(
                    {
                        "schema_version": 1,
                        "artifact_type": "asr_retry_candidate",
                        "status": "SUCCEEDED",
                        "session_id": session_id,
                        "source_sha256": plan["source_sha256"],
                        "stage_fingerprint": fp,
                        "parent_checksums": hashes,
                        "created_at": utc_timestamp(),
                        "model": MODEL,
                        "configuration": DECODE,
                        "window": window,
                        "candidate_a": {"origin": "canonical_raw_asr", "segments": candidate_a},
                        "candidate_b": {
                            "origin": "local_retry",
                            "raw_segments": retry,
                            "segments": candidate_b,
                            "diagnostics": diagnostics,
                        },
                        "review_required": True,
                        "canonical_replacement": False,
                        "candidate_b_empty": not candidate_b,
                    }
                )
                atomic_write_json(path, document)
                completed += 1
                save_progress("RUNNING", completed)
                print(
                    json.dumps(
                        {
                            "event": "retry_checkpointed",
                            "completed": completed,
                            "expected": len(plan["windows"]),
                        }
                    ),
                    flush=True,
                )
            unchanged()
            parents_unchanged()
            save_progress("SUCCEEDED", completed)
        except Exception as error:
            save_progress("FAILED", completed, type(error).__name__)
            raise
    return {
        "status": "SUCCEEDED",
        "completed_windows": completed,
        "canonical_replacement": False,
        "directory": str(directory),
    }


def launch_worker(config: AppConfig, session_id: str, plan_path: Path) -> int:
    """The user's CLI owns this local process; no agent, service or cloud runtime is needed."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "session_pipeline.asr_retry",
            session_id,
            "--plan",
            str(plan_path),
            "--root",
            str(config.application.root),
        ],
        env=local_only_environment(config.application.root),
        check=False,
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    environment = local_only_environment(args.root)
    os.environ.clear()
    os.environ.update(environment)
    try:
        result = run_retry(
            load_config(args.root, include_secrets=False), args.session_id, args.plan
        )
    except Exception as error:
        print(json.dumps({"status": "FAILED", "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(3) from None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
