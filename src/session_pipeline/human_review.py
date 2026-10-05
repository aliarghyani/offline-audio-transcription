"""Local human decisions over the existing ASR retry contract. No inference."""

from __future__ import annotations

import copy
import hashlib
import re
import wave
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from session_pipeline import __version__
from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact,
    seal_artifact,
    validate_artifact,
)
from session_pipeline.asr_quality import load_canonical, number, session_paths
from session_pipeline.asr_retry import (
    DECODE,
    MODEL,
    exclusive_lock,
    extract_clip,
    owned_segments,
    retry_fingerprint,
)
from session_pipeline.config import AppConfig
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.models import sha256_file, utc_timestamp

REVIEW_VERSION = 2
MERGE_STATES = ("SAFE", "AUTO_TRIMMED", "MANUAL_BOUNDARY_REQUIRED", "UNUSABLE")
CHOICES = ("UNREVIEWED", "USE_A", "USE_B", "UNCLEAR")
EPSILON = 1e-6  # Arithmetic tolerance only; not a speech-boundary adjustment.


class ReviewError(ValueError):
    """A review/merge contract cannot safely be satisfied."""


class DecisionConflict(ReviewError):
    """Another tab/process updated the decision file; reload before saving."""


@dataclass
class RetryContext:
    raw_path: Path
    work: Path
    plan_path: Path
    raw: dict[str, Any]
    plan: dict[str, Any]
    review: dict[str, Any]
    integrity: dict[str, Any]
    candidates: list[dict[str, Any]]
    parents: dict[str, str]

    @property
    def directory(self) -> Path:
        return self.work / "qa/human-review" / self.plan["stage_fingerprint"]


def require(value: bool, message: str) -> None:
    if not value:
        raise ReviewError(message)


def interval(value: Any, duration: float) -> tuple[float, float]:
    require(isinstance(value, list) and len(value) == 2, "invalid interval")
    start, end = value
    require(
        number(start) and number(end) and 0 <= start < end <= duration, "invalid interval limits"
    )
    return start, end


def intersects(start: float, end: float, target: list[float]) -> bool:
    return end > target[0] and start < target[1]


def validate_target_windows(windows: list[dict[str, Any]], duration: float) -> None:
    previous = 0.0
    for index, window in enumerate(windows):
        require(
            type(window["window_index"]) is int and window["window_index"] == index,
            "retry window indices must be contiguous",
        )
        start, end = interval(window["target_interval"], duration)
        left, right = interval(window["decoded_interval"], duration)
        require(
            start >= previous and left <= start < end <= right, "overlapping/invalid retry targets"
        )
        previous = end


def validate_segments(segments: Any, decoded: list[float]) -> None:
    require(isinstance(segments, list), "candidate segments must be a list")
    previous = decoded[0]
    for segment in segments:
        require(
            isinstance(segment, dict) and isinstance(segment.get("text"), str),
            "invalid candidate text",
        )
        a, b = segment.get("start"), segment.get("end")
        require(
            number(a) and number(b) and previous <= a <= b <= decoded[1] + 0.12,
            "invalid candidate timeline",
        )
        previous = a
        words = segment.get("words") or []
        require(isinstance(words, list), "invalid candidate words")
        word_previous = decoded[0]
        for word in words:
            require(
                isinstance(word, dict) and isinstance(word.get("word"), str),
                "invalid candidate word",
            )
            a, b = word.get("start"), word.get("end")
            require(
                number(a) and number(b) and word_previous <= a <= b <= decoded[1] + 0.12,
                "invalid candidate word timeline",
            )
            word_previous = a


def load_context(
    config: AppConfig, sid: str, fingerprint: str | None = None, *, recovery: bool = False
) -> RetryContext:
    raw_path, work, review_path = session_paths(config, sid)
    raw = load_canonical(raw_path, sid)
    review = read_artifact(review_path)
    integrity = read_artifact(work / "integrity.json")
    require(
        review.get("status") == "SUCCEEDED"
        and review.get("artifact_type") == "asr_review"
        and review.get("session_id") == sid
        and review.get("source_sha256") == raw["source_sha256"]
        and review["parent_checksums"]["canonical_asr"] == raw["artifact_checksum"],
        "quality review is stale",
    )
    require(
        integrity.get("status") == "SUCCEEDED"
        and integrity["decode"]["success"]
        and integrity["source"]["sha256"] == raw["source_sha256"],
        "integrity parent mismatch",
    )
    if fingerprint is not None:
        require(bool(re.fullmatch(r"[0-9a-f]{64}", fingerprint)), "invalid retry fingerprint")
        plan_path = work / "qa/retries" / fingerprint / "plan.json"
    else:
        plans = []
        for path in sorted((work / "qa/retries").glob("*/plan.json")):
            progress_path = path.parent / "progress.json"
            if progress_path.exists() and read_artifact(progress_path).get("status") == "SUCCEEDED":
                plans.append(path)
        require(
            len(plans) == 1, "exactly one completed retry required; specify --retry-fingerprint"
        )
        plan_path = plans[0]
    require(
        plan_path.resolve().is_relative_to((work / "qa/retries").resolve()), "plan escapes work"
    )
    plan = read_artifact(plan_path)
    parents = {"canonical_asr": raw["artifact_checksum"], "review": review["artifact_checksum"]}
    require(
        plan.get("artifact_type") == "asr_retry_plan"
        and plan.get("status") == "SUCCEEDED"
        and plan.get("session_id") == sid
        and plan.get("source_sha256") == raw["source_sha256"]
        and plan.get("model") == raw.get("model") == MODEL
        and plan.get("parent_checksums") == parents,
        "retry plan parent/model mismatch",
    )
    configuration = plan["configuration"]
    require(
        configuration["decode"] == DECODE and configuration["windows"] == plan["windows"],
        "retry plan configuration mismatch",
    )
    expected = stage_fingerprint(
        stage="asr_targeted_retry",
        source_sha256=raw["source_sha256"],
        relevant_config=configuration,
        parent_checksums=parents,
        models=MODEL,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    require(
        expected == plan["stage_fingerprint"] == plan_path.parent.name,
        "retry plan fingerprint mismatch",
    )
    parents.update(plan=plan["artifact_checksum"], integrity=integrity["artifact_checksum"])
    progress = read_artifact(plan_path.parent / "progress.json")
    require(
        progress.get("artifact_type") == "asr_retry_progress"
        and progress.get("status") == "SUCCEEDED"
        and progress.get("session_id") == sid
        and progress.get("source_sha256") == raw["source_sha256"]
        and progress.get("parent_checksums") == parents
        and progress["completed_windows"] == progress["expected_windows"] == len(plan["windows"]),
        "retry completion/progress mismatch",
    )
    validate_target_windows(plan["windows"], raw["probe"]["duration_seconds"])
    candidates = []
    previous = 0.0
    known_entries = {entry["entry_id"] for entry in review["entries"]}
    for index, window in enumerate(plan["windows"]):
        require(
            type(window["window_index"]) is int and window["window_index"] == index,
            "retry window indices must be contiguous",
        )
        start, end = interval(window["target_interval"], raw["probe"]["duration_seconds"])
        left, right = interval(window["decoded_interval"], raw["probe"]["duration_seconds"])
        require(
            start >= previous and left <= start < end <= right, "overlapping/invalid retry targets"
        )
        previous = end
        require(set(window["entry_ids"]).issubset(known_entries), "unknown quality entry")
        expected_indices = [
            n
            for n, s in enumerate(raw["segments"])
            if intersects(s["start"], s["end"], window["target_interval"])
        ]
        require(expected_indices == window["original_segment_indices"], "original span mismatch")
        path = plan_path.parent / f"candidate-{index:04}.json"
        require(
            path.resolve().is_relative_to(plan_path.parent.resolve()), "candidate escapes retry"
        )
        try:
            c = read_artifact(path)
            require(
                c.get("artifact_type") == "asr_retry_candidate"
                and c.get("status") == "SUCCEEDED"
                and c.get("session_id") == sid
                and c.get("source_sha256") == raw["source_sha256"]
                and c.get("model") == MODEL
                and c.get("configuration") == DECODE
                and c.get("window") == window
                and c.get("parent_checksums") == parents
                and c.get("stage_fingerprint") == retry_fingerprint(plan, window, parents)
                and c.get("canonical_replacement") is False,
                "candidate identity mismatch",
            )
            require(
                c["candidate_a"]["segments"] == [raw["segments"][n] for n in expected_indices],
                "candidate A does not preserve canonical original",
            )
            validate_segments(c["candidate_b"]["raw_segments"], window["decoded_interval"])
            b = owned_segments(c["candidate_b"]["raw_segments"], window["target_interval"])
            require(
                c["candidate_b"]["segments"] == b and c["candidate_b_empty"] == (not b),
                "candidate B target ownership mismatch",
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            if not recovery:
                raise
            # Preserve bad checkpoints; represent them as explicit fallback evidence.
            c = {
                "artifact_checksum": sha256_file(path)
                if path.exists()
                else hashlib.sha256(f"missing-candidate-{index}".encode()).hexdigest(),
                "window": window,
                "candidate_a": {"segments": [raw["segments"][n] for n in expected_indices]},
                "candidate_b": {"raw_segments": [], "segments": []},
                "candidate_b_empty": True,
                "_recovery_error": f"invalid retry checkpoint: {type(error).__name__}",
            }
        candidates.append(c)
    require(bool(candidates), "no retry windows to review")
    parents = {
        **parents,
        **{f"candidate_{n:04}": c["artifact_checksum"] for n, c in enumerate(candidates)},
    }
    return RetryContext(
        raw_path, work, plan_path, raw, plan, review, integrity, candidates, parents
    )


def boundary_issues(c: dict[str, Any]) -> list[str]:
    target = c["window"]["target_interval"]
    start, end = target
    issues = set()
    b = c["candidate_b"]["segments"]
    if not b or not any(s["text"].strip() for s in b):
        issues.add("B_TARGET_EMPTY")
    for s in c["candidate_a"]["segments"]:
        if s["start"] < start - EPSILON or s["end"] > end + EPSILON:
            issues.add("A_SEGMENT_CROSSES_TARGET")
        for w in s.get("words") or []:
            if w["start"] < start - EPSILON or w["end"] > end + EPSILON:
                issues.add("A_WORD_CROSSES_TARGET")
    for s in c["candidate_b"]["raw_segments"]:
        if not intersects(s["start"], s["end"], target):
            continue
        if not s.get("words"):
            if s["start"] < start - EPSILON or s["end"] > end + EPSILON:
                issues.add("B_UNTIMED_SEGMENT_CROSSES_TARGET")
        for w in s.get("words") or []:
            if intersects(w["start"], w["end"], target) and (
                w["start"] < start - EPSILON or w["end"] > end + EPSILON
            ):
                issues.add("B_WORD_CROSSES_TARGET")
    return sorted(issues)


def reconcile_candidate(c: dict[str, Any]) -> dict[str, Any]:
    """Separate usable human evidence from deterministic automatic replacement."""
    start, end = c["window"]["target_interval"]
    issues = set(boundary_issues(c))
    trimmed = []
    outside = False
    previous_word_end = None
    word_ids = []
    for segment_index, s in enumerate(c["candidate_b"]["raw_segments"]):
        words = s.get("words") or []
        if not words:
            if start - EPSILON <= s["start"] <= s["end"] <= end + EPSILON:
                trimmed.append(copy.deepcopy(s))
            else:
                outside = True
            continue
        selected = []
        for word_index, word in enumerate(words):
            a, b = word["start"], word["end"]
            relevant = intersects(a, b, [start, end]) or start <= a == b < end
            if relevant:
                if a < start - EPSILON or b > end + EPSILON:
                    issues.add("B_WORD_CROSSES_TARGET")
                if a == b:
                    issues.add("B_AMBIGUOUS_WORD_TIMING")
                if previous_word_end is not None and a < previous_word_end - EPSILON:
                    issues.add("B_OVERLAPPING_WORD_TIMING")
                previous_word_end = b
            if start - EPSILON <= a <= b <= end + EPSILON and start <= (a + b) / 2 < end:
                selected.append(copy.deepcopy(word))
                word_ids.append(f"{segment_index}:{word_index}")
            else:
                outside = True
        if selected:
            trimmed.append(
                {
                    **copy.deepcopy(s),
                    "start": max(start, selected[0]["start"]),
                    "end": min(end, selected[-1]["end"]),
                    "words": selected,
                    "text": "".join(w["word"] for w in selected).strip(),
                }
            )
    usable = any(s["text"].strip() for s in c["candidate_b"]["segments"])
    if not usable:
        status = "UNUSABLE"
    elif issues:
        status = "MANUAL_BOUNDARY_REQUIRED"
    else:
        status = "AUTO_TRIMMED" if outside else "SAFE"
    return {
        "merge_status": status,
        "b_usable": usable,
        "merge_issues": sorted(issues),
        "replacement_segments": trimmed if status in {"SAFE", "AUTO_TRIMMED"} else [],
        "automatic_word_ids": word_ids if status in {"SAFE", "AUTO_TRIMMED"} else [],
    }


def context_fingerprint(context: RetryContext, version: int = REVIEW_VERSION) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "version": version,
                "source": context.raw["source_sha256"],
                "parents": context.parents,
            }
        )
    ).hexdigest()


def source_path(config: AppConfig, context: RetryContext) -> Path:
    name = context.integrity["source"]["original_filename"]
    require(
        isinstance(name, str) and Path(name).name == name and "/" not in name and "\\" not in name,
        "invalid source filename",
    )
    path = (config.application.sessions_path / name).resolve()
    require(
        path.is_relative_to(config.application.sessions_path.resolve()), "source escapes sessions"
    )
    require(
        path.stat().st_size == context.integrity["source"]["size_bytes"]
        and sha256_file(path) == context.raw["source_sha256"],
        "source audio hash mismatch",
    )
    return path


def clip_path(context: RetryContext, index: int) -> Path:
    require(type(index) is int and 0 <= index < len(context.candidates), "invalid audio entry")
    path = context.plan_path.parent / f"derived/retry-{index:04}.wav"
    require(path.resolve().is_relative_to(context.plan_path.parent.resolve()), "clip escapes retry")
    return path


def clip_valid(path: Path, window: dict[str, Any]) -> bool:
    try:
        with wave.open(str(path), "rb") as stream:
            actual = stream.getnframes() / stream.getframerate()
            expected = window["decoded_interval"][1] - window["decoded_interval"][0]
            return (
                stream.getnchannels() == 1
                and stream.getframerate() == 16000
                and stream.getsampwidth() == 2
                and abs(actual - expected) <= 0.12
            )
    except (OSError, EOFError, wave.Error):
        return False


def segments_text(segments: list[dict[str, Any]]) -> str:
    return "\n".join(s["text"] for s in segments)


def initial_decisions(context: RetryContext, bundle: dict[str, Any]) -> dict[str, Any]:
    return seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "human_review_decisions",
            "decision_version": REVIEW_VERSION,
            "session_id": context.raw["session_id"],
            "source_sha256": context.raw["source_sha256"],
            "bundle_checksum": bundle["artifact_checksum"],
            "review_fingerprint": context_fingerprint(context),
            "revision": 0,
            "updated_at": utc_timestamp(),
            "decisions": {
                e["review_id"]: {
                    "choice": "UNREVIEWED",
                    "updated_at": None,
                    "merge_status": e["merge_status"],
                }
                for e in bundle["entries"]
            },
        }
    )


def load_decisions(context: RetryContext, bundle: dict[str, Any]) -> dict[str, Any]:
    d = read_artifact(context.directory / "decisions.json")
    validate_decisions(context, bundle, d)
    return d


def validate_decisions(
    context: RetryContext, bundle: dict[str, Any], d: dict[str, Any], version: int = REVIEW_VERSION
) -> None:
    require(
        d.get("artifact_type") == "human_review_decisions"
        and d.get("session_id") == context.raw["session_id"]
        and d.get("source_sha256") == context.raw["source_sha256"]
        and d.get("bundle_checksum") == bundle["artifact_checksum"]
        and d.get("review_fingerprint") == context_fingerprint(context, version)
        and d.get("decision_version", 1) == version
        and type(d.get("revision")) is int
        and d["revision"] >= 0
        and set(d["decisions"]) == {e["review_id"] for e in bundle["entries"]},
        "decision manifest identity mismatch",
    )
    for entry in bundle["entries"]:
        choice = d["decisions"][entry["review_id"]]["choice"]
        require(choice in CHOICES, "invalid decision")
        require(choice != "USE_B" or entry["b_selectable"], "unusable B decision")
        if version == REVIEW_VERSION:
            require(
                d["decisions"][entry["review_id"]].get("merge_status") == entry["merge_status"],
                "decision merge status changed",
            )


def load_bundle(context: RetryContext) -> dict[str, Any]:
    bundle = read_artifact(context.directory / "bundle.json")
    validate_bundle(context, bundle)
    return bundle


def validate_bundle(
    context: RetryContext, bundle: dict[str, Any], version: int = REVIEW_VERSION
) -> None:
    require(
        bundle.get("artifact_type") == "human_review_bundle"
        and bundle.get("status") == "SUCCEEDED"
        and bundle.get("review_fingerprint") == context_fingerprint(context, version)
        and bundle.get("review_version") == version
        and bundle.get("parent_checksums") == context.parents,
        "review bundle is stale or needs upgrade; run qa review build",
    )
    require(
        bundle["entries"] == review_entries(context, legacy=version == 1),
        "review bundle entries changed",
    )
    require(
        bundle.get("audio_checksums")
        == {str(i): sha256_file(clip_path(context, i)) for i in range(len(context.candidates))},
        "review audio changed; preserve the bundle for investigation",
    )


def review_entries(context: RetryContext, *, legacy: bool = False) -> list[dict[str, Any]]:
    flags = {e["entry_id"]: e["reasons"] for e in context.review["entries"]}
    result = []
    for index, c in enumerate(context.candidates):
        issues = boundary_issues(c)
        eligibility = reconcile_candidate(c)
        result.append(
            {
                "review_id": c["artifact_checksum"],
                "window_index": index,
                "target_interval": c["window"]["target_interval"],
                "decoded_interval": c["window"]["decoded_interval"],
                "quality_flags": sorted(
                    {r for eid in c["window"]["entry_ids"] for r in flags[eid]}
                ),
                "candidate_a": copy.deepcopy(c["candidate_a"]),
                "candidate_b": copy.deepcopy(c["candidate_b"]),
                "text_a": segments_text(c["candidate_a"]["segments"]),
                "text_b": segments_text(c["candidate_b"]["segments"]),
                "b_selectable": not issues if legacy else eligibility["b_usable"],
                "merge_issues": issues if legacy else eligibility["merge_issues"],
                "audio_url": f"/audio/{index}",
            }
        )
        if not legacy:
            result[-1].update(eligibility)
    return result


def upgrade_bundle(context: RetryContext) -> None:
    """Locked, journaled v1->v2 migration; user choices/timestamps never change."""
    directory = context.directory
    bundle_path, decisions_path = directory / "bundle.json", directory / "decisions.json"
    journal_path = directory / "migration-v2.json"
    if journal_path.exists():
        journal = read_artifact(journal_path)
        require(
            journal.get("artifact_type") == "human_review_migration", "invalid migration journal"
        )
        if journal["status"] == "PREPARED":
            new_bundle = validate_artifact(journal["new_bundle"])
            new_decisions = validate_artifact(journal["new_decisions"])
            validate_bundle(context, new_bundle)
            validate_decisions(context, new_bundle, new_decisions)
            require(
                read_artifact(bundle_path)["artifact_checksum"]
                in {journal["old_bundle_checksum"], new_bundle["artifact_checksum"]},
                "bundle changed during migration",
            )
            current_decisions = read_artifact(decisions_path)
            if current_decisions.get("decision_version") == REVIEW_VERSION:
                # A writer may have saved a newer human choice after both files
                # committed but before the migration completion marker was saved.
                validate_decisions(context, new_bundle, current_decisions)
                new_decisions = current_decisions
            else:
                require(
                    current_decisions["artifact_checksum"] == journal["old_decision_checksum"],
                    "decisions changed during migration",
                )
            atomic_write_json(bundle_path, new_bundle)
            atomic_write_json(decisions_path, new_decisions)
            journal["status"] = "SUCCEEDED"
            atomic_write_json(journal_path, seal_artifact(journal))
    old_bundle = read_artifact(bundle_path)
    if old_bundle.get("review_version") == REVIEW_VERSION:
        return
    require(old_bundle.get("review_version") == 1, "unsupported review version")
    validate_bundle(context, old_bundle, version=1)
    old_decisions = read_artifact(decisions_path)
    validate_decisions(context, old_bundle, old_decisions, version=1)
    # Preserve the exact pre-upgrade sealed documents as immutable local backups.
    for name, document in (("bundle.json", old_bundle), ("decisions.json", old_decisions)):
        backup = directory / "migration-v1" / name
        if backup.exists():
            require(read_artifact(backup) == document, "migration backup already differs")
        else:
            atomic_write_json(backup, document)
    new_bundle = copy.deepcopy(old_bundle)
    new_bundle.update(
        review_version=REVIEW_VERSION,
        review_fingerprint=context_fingerprint(context),
        entries=review_entries(context),
    )
    new_bundle = seal_artifact(new_bundle)
    new_decisions = copy.deepcopy(old_decisions)
    new_decisions.update(
        decision_version=REVIEW_VERSION,
        bundle_checksum=new_bundle["artifact_checksum"],
        review_fingerprint=context_fingerprint(context),
    )
    for entry in new_bundle["entries"]:
        new_decisions["decisions"][entry["review_id"]]["merge_status"] = entry["merge_status"]
    new_decisions["migration"] = {
        "from_version": 1,
        "to_version": REVIEW_VERSION,
        "old_decision_checksum": old_decisions["artifact_checksum"],
        "migrated_at": utc_timestamp(),
    }
    new_decisions = seal_artifact(new_decisions)
    validate_decisions(context, new_bundle, new_decisions)
    journal = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "human_review_migration",
            "status": "PREPARED",
            "old_bundle_checksum": old_bundle["artifact_checksum"],
            "old_decision_checksum": old_decisions["artifact_checksum"],
            "new_bundle": new_bundle,
            "new_decisions": new_decisions,
        }
    )
    atomic_write_json(journal_path, journal)
    upgrade_bundle(context)  # Resume through the same validated journal commit path.


def build_bundle(config: AppConfig, sid: str, fingerprint: str | None = None) -> dict[str, Any]:
    context = load_context(config, sid, fingerprint)
    source = source_path(config, context)
    baseline = source.stat().st_mtime_ns
    for index, c in enumerate(context.candidates):
        clip = clip_path(context, index)
        if not clip_valid(clip, c["window"]):
            require(not clip.exists(), "existing audio clip invalid; preserve it for investigation")
            extract_clip(source, clip, c["window"]["decoded_interval"])
            require(clip_valid(clip, c["window"]), "review clip invalid")
    require(
        source.stat().st_mtime_ns == baseline
        and sha256_file(source) == context.raw["source_sha256"],
        "source changed during review build",
    )
    directory = context.directory
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".decisions.lock"):
        if (directory / "bundle.json").exists():
            upgrade_bundle(context)
            bundle = load_bundle(context)
            if (directory / "decisions.json").exists():
                load_decisions(context, bundle)
            else:
                atomic_write_json(directory / "decisions.json", initial_decisions(context, bundle))
        else:
            require(not (directory / "decisions.json").exists(), "decisions exist without bundle")
            bundle = seal_artifact(
                {
                    "schema_version": 1,
                    "artifact_type": "human_review_bundle",
                    "status": "SUCCEEDED",
                    "session_id": sid,
                    "source_sha256": context.raw["source_sha256"],
                    "review_version": REVIEW_VERSION,
                    "review_fingerprint": context_fingerprint(context),
                    "retry_plan_fingerprint": context.plan["stage_fingerprint"],
                    "parent_checksums": context.parents,
                    "created_at": utc_timestamp(),
                    "entries": review_entries(context),
                    "audio_checksums": {
                        str(i): sha256_file(clip_path(context, i))
                        for i in range(len(context.candidates))
                    },
                    "cloud_llm_required": False,
                }
            )
            atomic_write_json(directory / "bundle.json", bundle)
            atomic_write_json(directory / "decisions.json", initial_decisions(context, bundle))
        for name in ("index.html", "app.js", "style.css"):
            (directory / name).write_bytes(
                files("session_pipeline").joinpath("review_ui", name).read_bytes()
            )
    return review_status(config, sid, context.plan["stage_fingerprint"])


def review_status(config: AppConfig, sid: str, fingerprint: str | None = None) -> dict[str, Any]:
    context = load_context(config, sid, fingerprint)
    bundle = load_bundle(context)
    decisions = load_decisions(context, bundle)
    counts = {
        choice: sum(d["choice"] == choice for d in decisions["decisions"].values())
        for choice in CHOICES
    }
    return {
        "session_id": sid,
        "retry_plan_fingerprint": context.plan["stage_fingerprint"],
        "review_directory": str(context.directory),
        "decision_path": str(context.directory / "decisions.json"),
        "entry_count": len(bundle["entries"]),
        "decision_counts": counts,
        "unresolved_count": counts["UNCLEAR"] + counts["UNREVIEWED"],
        "review_complete": counts["UNREVIEWED"] == 0,
        "empty_b_count": sum(c["candidate_b_empty"] for c in context.candidates),
        "b_selectable_count": sum(e["b_selectable"] for e in bundle["entries"]),
        "b_usable_count": sum(e["b_usable"] for e in bundle["entries"]),
        "b_unusable_count": sum(not e["b_usable"] for e in bundle["entries"]),
        "merge_status_counts": {
            status: sum(e["merge_status"] == status for e in bundle["entries"])
            for status in MERGE_STATES
        },
        "manual_boundary_selected_count": sum(
            decisions["decisions"][e["review_id"]]["choice"] == "USE_B"
            and e["merge_status"] == "MANUAL_BOUNDARY_REQUIRED"
            for e in bundle["entries"]
        ),
        "audio_ready_count": sum(
            clip_valid(clip_path(context, i), c["window"]) for i, c in enumerate(context.candidates)
        ),
        "cloud_llm_required": False,
    }


def save_decision(
    config: AppConfig,
    sid: str,
    fingerprint: str,
    review_id: str,
    choice: str,
    expected_checksum: str,
) -> dict[str, Any]:
    require(choice in CHOICES, "invalid decision choice")
    context = load_context(config, sid, fingerprint)
    bundle = load_bundle(context)
    entry = next((e for e in bundle["entries"] if e["review_id"] == review_id), None)
    require(entry is not None, "unknown review entry")
    require(choice != "USE_B" or entry["b_selectable"], "B has no usable target text")
    with exclusive_lock(context.directory / ".decisions.lock"):
        decisions = load_decisions(context, bundle)
        if decisions["artifact_checksum"] != expected_checksum:
            raise DecisionConflict("decisions changed; reload before saving")
        decisions["decisions"][review_id] = {
            "choice": choice,
            "updated_at": utc_timestamp(),
            "merge_status": entry["merge_status"],
        }
        decisions["revision"] += 1
        decisions["updated_at"] = utc_timestamp()
        decisions = seal_artifact(decisions)
        atomic_write_json(context.directory / "decisions.json", decisions)
    return decisions
