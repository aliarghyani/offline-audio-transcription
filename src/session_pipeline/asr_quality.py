"""Deterministic local ASR review. Signals are triage evidence, not confidence scores."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from session_pipeline import __version__
from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact,
    seal_artifact,
)
from session_pipeline.config import AppConfig
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.models import session_id_from_sha256, utc_timestamp

RULES = (
    "HIGH_TEMPERATURE",
    "LOW_AVG_LOGPROB",
    "LOW_WORD_PROBABILITY",
    "SUSPICIOUS_REPETITION",
    "SUSPICIOUS_SHORT_SEGMENT",
    "SUSPICIOUS_GAP",
    "BOUNDARY_RISK",
    "POSSIBLE_HALLUCINATION",
)
EXPERIMENTAL = RULES[2:]


@dataclass(frozen=True)
class QualityThresholds:
    high_temperature: float = 0.6
    low_avg_logprob: float = -1.0
    low_word_probability: float = 0.2
    low_word_min_count: int = 2
    low_word_min_fraction: float = 0.2
    short_segment_seconds: float = 0.35
    repetition_min_characters: int = 20
    repetition_nearby_seconds: float = 15.0
    repetition_ngram_words: int = 5
    repetition_min_occurrences: int = 3
    gap_seconds: float = 15.0
    gap_min_speech_seconds: float = 3.0
    unknown_speech_gap_seconds: float = 30.0
    boundary_margin_seconds: float = 0.5
    hallucination_no_speech_probability: float = 0.8
    hallucination_avg_logprob: float = -1.0
    retry_padding_seconds: float = 5.0
    retry_max_target_seconds: float = 60.0

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if "logprob" not in key and value < 0:
                raise ValueError(f"{key} must be non-negative")
            if key.endswith(("_count", "_characters", "_words", "_occurrences")):
                if type(value) is not int or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
        for key in (
            "high_temperature",
            "low_word_probability",
            "low_word_min_fraction",
            "hallucination_no_speech_probability",
        ):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be within [0, 1]")
        if not 0 < self.retry_max_target_seconds <= 90 or self.retry_padding_seconds > 10:
            raise ValueError("retry target must be (0,90] seconds and padding at most 10 seconds")
        if self.gap_seconds <= 0 or self.unknown_speech_gap_seconds < self.gap_seconds:
            raise ValueError("gap thresholds must be positive and ordered")


def load_thresholds(path: Path) -> QualityThresholds:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("quality thresholds must be an object")
    return QualityThresholds(**document)


def number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def union(intervals: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def pad(interval: list[float], padding: float, duration: float) -> list[float]:
    start, end = interval
    if not all(number(v) for v in (start, end, padding, duration)):
        raise ValueError("interval values must be finite")
    if not 0 <= start <= end <= duration or padding < 0:
        raise ValueError("invalid interval or padding")
    return [max(0.0, start - padding), min(duration, end + padding)]


def distribution(values: list[Any]) -> dict[str, Any]:
    available = sorted(float(v) for v in values if number(v))
    result: dict[str, Any] = {"available": len(available), "missing": len(values) - len(available)}
    if not available:
        return {**result, "min": None, "p05": None, "median": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        position = (len(available) - 1) * fraction
        left = math.floor(position)
        right = math.ceil(position)
        return available[left] + (available[right] - available[left]) * (position - left)

    return {
        **result,
        "min": available[0],
        "p05": percentile(0.05),
        "median": statistics.median(available),
        "p95": percentile(0.95),
        "max": available[-1],
        "mean": statistics.fmean(available),
    }


def comparison_text(text: str) -> str:
    # Comparison only: original text is always stored byte-for-byte as a string.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def session_paths(config: AppConfig, session_id: str) -> tuple[Path, Path, Path]:
    if not re.fullmatch(r"s-[0-9a-f]{20}", session_id):
        raise ValueError("invalid session ID")
    app = config.application
    directories = [
        app.sessions_path.resolve(),
        app.work_path.resolve(),
        (app.root / app.outputs_dir).resolve(),
    ]
    historical = (app.root / "reference_transcripts").resolve()
    if any(
        a.is_relative_to(b)
        for i, a in enumerate(directories)
        for j, b in enumerate(directories)
        if i != j
    ) or any(path.is_relative_to(historical) for path in directories):
        raise ValueError("session, work, output and historical directories must be separate")
    raw = app.root / app.outputs_dir / session_id / "transcript.raw.fa.json"
    work = app.work_path / session_id
    for path in (raw, work):
        if not path.resolve().is_relative_to(app.root.resolve()):
            raise ValueError("session path escapes repository")
    return raw, work, work / "qa/asr-review.json"


def load_canonical(path: Path, session_id: str) -> dict[str, Any]:
    raw = read_artifact(path)
    if raw.get("status") != "SUCCEEDED" or raw.get("artifact_type") != "asr":
        raise ValueError("successful canonical ASR required")
    sha = raw["source_sha256"]
    if raw["session_id"] != session_id or session_id_from_sha256(sha) != session_id:
        raise ValueError("canonical source/session identity mismatch")
    if raw["source"]["sha256"] != sha:
        raise ValueError("canonical source hash mismatch")
    duration = raw["probe"]["duration_seconds"]
    if not number(duration) or duration <= 0:
        raise ValueError("invalid source duration")
    segments = raw["segments"]
    if not isinstance(segments, list) or len(segments) != raw["segment_count"]:
        raise ValueError("invalid segment count")
    previous = 0.0
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if not number(start) or not number(end) or not previous <= start <= end <= duration:
            raise ValueError("invalid ordered segment interval")
        if not isinstance(segment["text"], str) or not isinstance(segment.get("words", []), list):
            raise ValueError("invalid segment text/words")
        for word in segment.get("words", []):
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise ValueError("invalid word metadata")
            a, b = word.get("start"), word.get("end")
            if not number(a) or not number(b) or not 0 <= a <= b <= duration:
                raise ValueError("invalid word interval")
        previous = start
    return raw


def audit(
    raw: dict[str, Any],
    thresholds: QualityThresholds,
    speech: list[list[float]] | None = None,
    boundaries: list[float] | None = None,
) -> dict[str, Any]:
    segments = raw["segments"]
    duration = raw["probe"]["duration_seconds"]
    t = thresholds
    reasons: list[set[str]] = [set() for _ in segments]
    low_words = []
    normalized = [comparison_text(s["text"]) for s in segments]
    for index, segment in enumerate(segments):
        flags = reasons[index]
        if number(segment.get("temperature")) and segment["temperature"] >= t.high_temperature:
            flags.add("HIGH_TEMPERATURE")
        if number(segment.get("avg_logprob")) and segment["avg_logprob"] < t.low_avg_logprob:
            flags.add("LOW_AVG_LOGPROB")
        probabilities = [w.get("probability") for w in segment.get("words", [])]
        valid = [v for v in probabilities if number(v)]
        low = sum(v < t.low_word_probability for v in valid)
        low_words.append(low)
        if low >= t.low_word_min_count and low / max(1, len(valid)) >= t.low_word_min_fraction:
            flags.add("LOW_WORD_PROBABILITY")
        if segment["text"].strip() and segment["end"] - segment["start"] < t.short_segment_seconds:
            flags.add("SUSPICIOUS_SHORT_SEGMENT")
        if (
            number(segment.get("no_speech_prob"))
            and number(segment.get("avg_logprob"))
            and segment["no_speech_prob"] >= t.hallucination_no_speech_probability
            and segment["avg_logprob"] < t.hallucination_avg_logprob
            and segment["text"].strip()
        ):
            flags.add("POSSIBLE_HALLUCINATION")
        tokens = normalized[index].split()
        n = t.repetition_ngram_words
        grams = Counter(tuple(tokens[pos : pos + n]) for pos in range(len(tokens) - n + 1))
        if grams and max(grams.values()) >= t.repetition_min_occurrences:
            flags.add("SUSPICIOUS_REPETITION")
        if (
            index
            and len(normalized[index]) >= t.repetition_min_characters
            and normalized[index] == normalized[index - 1]
            and segment["start"] - segments[index - 1]["end"] <= t.repetition_nearby_seconds
        ):
            flags.add("SUSPICIOUS_REPETITION")
            reasons[index - 1].add("SUSPICIOUS_REPETITION")
    # Evaluate after repetition can add a signal to a preceding segment.
    for segment, flags in zip(segments, reasons, strict=True):
        # Boundary proximity alone requires another signal.
        if (
            flags
            and boundaries
            and any(
                segment["start"] - t.boundary_margin_seconds
                <= b
                <= segment["end"] + t.boundary_margin_seconds
                for b in boundaries
            )
        ):
            flags.add("BOUNDARY_RISK")
    entries = []
    for index, flags in enumerate(reasons):
        if not flags:
            continue
        segment = segments[index]
        interval = [segment["start"], segment["end"]]
        entries.append(
            {
                "segment_indices": [index],
                "interval": interval,
                "original_text": segment["text"],
                "reasons": sorted(flags),
                "diagnostics": {
                    key: segment.get(key)
                    for key in ("temperature", "avg_logprob", "no_speech_prob", "compression_ratio")
                },
                "words": segment.get("words", []),
                "low_word_count": low_words[index],
            }
        )
    gaps = []
    cursor = 0.0
    for start, end in union([[s["start"], s["end"]] for s in segments]) + [[duration, duration]]:
        if start - cursor >= t.gap_seconds:
            detected = (
                None
                if speech is None
                else sum(max(0, min(start, b) - max(cursor, a)) for a, b in union(speech))
            )
            suspicious = (detected is not None and detected >= t.gap_min_speech_seconds) or (
                detected is None and start - cursor >= t.unknown_speech_gap_seconds
            )
            gap = {
                "interval": [cursor, start],
                "duration_seconds": start - cursor,
                "vad_speech_seconds": detected,
                "suspicious": suspicious,
            }
            gaps.append(gap)
            if suspicious:
                entries.append(
                    {
                        "segment_indices": [],
                        "interval": [cursor, start],
                        "original_text": "",
                        "reasons": ["SUSPICIOUS_GAP"],
                        "diagnostics": gap,
                    }
                )
        cursor = max(cursor, end)
    for entry in entries:
        entry["timestamp"] = entry["interval"][0]
        entry["suggested_retry_interval"] = pad(
            entry["interval"], t.retry_padding_seconds, duration
        )
        entry["experimental_reasons"] = [f for f in entry["reasons"] if f in EXPERIMENTAL]
        entry["entry_id"] = hashlib.sha256(
            canonical_json_bytes(
                {
                    "source": raw["source_sha256"],
                    "thresholds": asdict(t),
                    **entry,
                }
            )
        ).hexdigest()[:20]
    words = [w.get("probability") for s in segments for w in s.get("words", [])]
    distributions = {
        key: distribution([s.get(key) for s in segments])
        for key in ("temperature", "avg_logprob", "no_speech_prob", "compression_ratio")
    }
    distributions.update(
        word_probability=distribution(words),
        segment_duration=distribution([s["end"] - s["start"] for s in segments]),
        words_per_segment=distribution([len(s.get("words", [])) for s in segments]),
    )
    durations = union([e["suggested_retry_interval"] for e in entries])
    return {
        "thresholds": asdict(t),
        "experimental_rules": list(EXPERIMENTAL),
        "segment_count": len(segments),
        "distributions": distributions,
        "temperature_counts": dict(Counter(str(s.get("temperature")) for s in segments)),
        "repeated_text_groups": sum(v > 1 for v in Counter(normalized).values()),
        "gaps": gaps,
        "entries": entries,
        "segment_flag_counts": {rule: sum(rule in f for f in reasons) for rule in RULES},
        "entry_flag_counts": {rule: sum(rule in e["reasons"] for e in entries) for rule in RULES},
        "flagged_segment_count": sum(bool(f) for f in reasons),
        "proposed_retry_union_seconds": sum(b - a for a, b in durations),
        "proposed_retry_intervals": durations,
        "cloud_llm_required": False,
    }


def create_review(
    config: AppConfig, session_id: str, thresholds_path: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    raw_path, work, destination = session_paths(config, session_id)
    raw = load_canonical(raw_path, session_id)
    thresholds = load_thresholds(
        thresholds_path or config.application.root / "config/asr-quality.json"
    )
    speech: list[list[float]] | None = None
    boundaries = []
    parents = {"canonical_asr": raw["artifact_checksum"]}
    for name in ("coverage", "final_verification"):
        path = work / f"{name}.json"
        if path.exists():
            evidence = read_artifact(path)
            if evidence.get("source_sha256") != raw["source_sha256"]:
                raise ValueError("verification source mismatch")
            if name == "final_verification" and (
                evidence.get("parent_checksums", {}).get("asr") != raw["artifact_checksum"]
                or evidence.get("parent_checksums", {}).get("coverage") != parents.get("coverage")
            ):
                raise ValueError("final verification is stale")
            parents[name] = evidence["artifact_checksum"]
    if raw.get("chunk_artifacts"):
        speech = []
        for ref in raw["chunk_artifacts"]:
            path = (config.application.root / ref["relative_path"]).resolve()
            if not path.is_relative_to(work.resolve()):
                raise ValueError("chunk reference escapes session work directory")
            chunk = read_artifact(path)
            if (
                chunk["artifact_checksum"] != ref["checksum"]
                or chunk["status"] != "SUCCEEDED"
                or chunk["source_sha256"] != raw["source_sha256"]
            ):
                raise ValueError("invalid chunk evidence")
            if "speech_intervals" not in chunk:
                speech = None
            elif speech is not None:
                speech.extend(chunk["speech_intervals"])
            start = chunk["owned_interval"][0]
            if start > 0:
                boundaries.append(start)
    findings = audit(raw, thresholds, speech, boundaries)
    fingerprint = stage_fingerprint(
        stage="asr_quality",
        source_sha256=raw["source_sha256"],
        relevant_config=asdict(thresholds),
        parent_checksums=parents,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    if destination.exists():
        old = read_artifact(destination)
        if old.get("stage_fingerprint") == fingerprint and old.get("status") == "SUCCEEDED":
            if old.get("parent_checksums") != parents or any(
                old.get(key) != value for key, value in findings.items()
            ):
                raise ValueError("existing review semantics changed")
            return destination, old
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "asr_review",
            "status": "SUCCEEDED",
            "session_id": session_id,
            "source_sha256": raw["source_sha256"],
            "source_duration_seconds": raw["probe"]["duration_seconds"],
            "stage_fingerprint": fingerprint,
            "parent_checksums": parents,
            "created_at": utc_timestamp(),
            **findings,
        }
    )
    atomic_write_json(destination, document)
    return destination, document
