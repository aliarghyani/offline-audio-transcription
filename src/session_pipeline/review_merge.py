"""Deterministic, human-authorized merge into a new immutable reviewed lineage."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact,
    seal_artifact,
)
from session_pipeline.asr_retry import exclusive_lock
from session_pipeline.config import AppConfig
from session_pipeline.human_review import (
    intersects,
    load_bundle,
    load_context,
    load_decisions,
    reconcile_candidate,
    require,
    source_path,
)
from session_pipeline.models import sha256_file, utc_timestamp

MERGE_VERSION = 2


def render_reviewed(segments: list[dict[str, Any]], unresolved: list[dict[str, Any]]) -> str:
    lines = []
    events = [(s["start"], 1, s) for s in segments]
    events.extend((entry["target_interval"][0], 0, entry) for entry in unresolved)
    for _, kind, event in sorted(events, key=lambda item: (item[0], item[1])):
        if kind == 0:
            a, b = event["target_interval"]
            lines.append(f"[UNRESOLVED {a:.3f}–{b:.3f} seconds; original retained]")
        else:
            lines.append(unicodedata.normalize("NFC", event["text"]))
    return "\n".join(lines) + "\n"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def merge_review(config: AppConfig, sid: str, fingerprint: str | None = None) -> dict[str, Any]:
    context = load_context(config, sid, fingerprint)
    source = source_path(config, context)
    source_mtime = source.stat().st_mtime_ns
    bundle = load_bundle(context)
    raw_files = [context.raw_path, context.raw_path.with_suffix(".txt")]
    raw_hashes = {str(path): sha256_file(path) for path in raw_files if path.exists()}
    with exclusive_lock(context.directory / ".decisions.lock"):
        decisions = load_decisions(context, bundle)
        manual = [
            e
            for e in bundle["entries"]
            if decisions["decisions"][e["review_id"]]["choice"] == "USE_B"
            and e["merge_status"] == "MANUAL_BOUNDARY_REQUIRED"
        ]
        require(
            not manual,
            "manual boundary resolution required for USE_B entries: "
            + ", ".join(f"window {e['window_index'] + 1} [{e['review_id']}]" for e in manual),
        )
        require(
            all(d["choice"] != "UNREVIEWED" for d in decisions["decisions"].values()),
            "merge refuses incomplete review",
        )
        identity = {
            "merge_version": MERGE_VERSION,
            "parents": context.parents,
            "bundle_checksum": bundle["artifact_checksum"],
            "decision_manifest_checksum": decisions["artifact_checksum"],
        }
        merge_fp = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        directory = context.raw_path.parent / "reviewed" / merge_fp
        require(
            directory.resolve().is_relative_to(context.raw_path.parent.resolve()),
            "reviewed output escapes output directory",
        )
        removed = set()
        replacements = []
        spans = []
        unresolved = []
        for entry, candidate in zip(bundle["entries"], context.candidates, strict=True):
            choice = decisions["decisions"][entry["review_id"]]["choice"]
            span = {
                "review_id": entry["review_id"],
                "window_index": entry["window_index"],
                "target_interval": entry["target_interval"],
                "choice": choice,
                "candidate_checksum": candidate["artifact_checksum"],
                "merge_status": entry["merge_status"],
            }
            spans.append(span)
            if choice == "USE_B":
                reconciled = reconcile_candidate(candidate)
                require(
                    reconciled["merge_status"] in {"SAFE", "AUTO_TRIMMED"},
                    "unusable or unresolved B replacement",
                )
                span["automatic_word_ids"] = reconciled["automatic_word_ids"]
                target = candidate["window"]["target_interval"]
                indices = {
                    n
                    for n, s in enumerate(context.raw["segments"])
                    if intersects(s["start"], s["end"], target)
                }
                require(
                    not removed.intersection(indices), "overlapping canonical replacement spans"
                )
                removed.update(indices)
                for segment in reconciled["replacement_segments"]:
                    replacement = copy.deepcopy(segment)
                    replacement["review_origin"] = {"candidate": "B", **span}
                    replacements.append(replacement)
            elif choice == "UNCLEAR":
                unresolved.append({**span, "status": "UNRESOLVED", "original_retained": True})
        segments = []
        for index, original in enumerate(context.raw["segments"]):
            if index not in removed:
                segment = copy.deepcopy(original)
                segment["review_origin"] = {"candidate": "A", "canonical_segment_index": index}
                segments.append(segment)
        segments.extend(replacements)
        segments.sort(key=lambda s: s["start"])  # Stable: preserve canonical order on tied starts.
        text = render_reviewed(segments, unresolved)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        json_path, txt_path = (
            directory / "transcript.reviewed.fa.json",
            directory / "transcript.reviewed.fa.txt",
        )
        provenance = {
            **identity,
            "raw_artifact_relative_path": context.raw_path.relative_to(
                config.application.root
            ).as_posix(),
            "raw_artifact_checksum": context.raw["artifact_checksum"],
            "retry_plan_fingerprint": context.plan["stage_fingerprint"],
            "source_audio_sha256": context.raw["source_sha256"],
            "spans": spans,
        }
        document = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "reviewed_asr",
                "status": "SUCCEEDED",
                "session_id": sid,
                "source_sha256": context.raw["source_sha256"],
                "source": copy.deepcopy(context.raw["source"]),
                "probe": copy.deepcopy(context.raw["probe"]),
                "model": copy.deepcopy(context.raw["model"]),
                "stage_fingerprint": merge_fp,
                "created_at": utc_timestamp(),
                "provenance": provenance,
                "segments": segments,
                "segment_count": len(segments),
                "word_count": sum(len(s.get("words") or []) for s in segments),
                "unresolved": unresolved,
                "text_normalization": "NFC only; UTF-8; newline-separated segments",
                "rendered_txt_sha256": text_sha,
                "qa_acceptance": {
                    "status": "PENDING",
                    "alignment_ready": False,
                    "unresolved_count": len(unresolved),
                },
                "cloud_llm_required": False,
            }
        )
        # Revalidate all lineage parents while holding the decision writer lock.
        current = load_context(config, sid, context.plan["stage_fingerprint"])
        require(current.parents == context.parents, "retry lineage changed during merge")
        require(
            source.stat().st_mtime_ns == source_mtime
            and sha256_file(source) == context.raw["source_sha256"]
            and all(sha256_file(Path(path)) == sha for path, sha in raw_hashes.items()),
            "raw artifacts/source changed during merge",
        )
        if json_path.exists():
            previous = read_artifact(json_path)
            require(
                previous["provenance"] == provenance
                and previous["segments"] == segments
                and previous["rendered_txt_sha256"] == text_sha,
                "existing reviewed lineage changed",
            )
            document = previous
        if txt_path.exists():
            require(sha256_file(txt_path) == text_sha, "existing reviewed TXT changed")
        else:
            write_text_atomic(txt_path, text)
        if not json_path.exists():
            atomic_write_json(json_path, document)  # JSON commits a complete, checksummed TXT pair.
        atomic_write_json(
            context.raw_path.parent / "reviewed/latest.json",
            seal_artifact(
                {
                    "schema_version": 1,
                    "artifact_type": "reviewed_asr_pointer",
                    "session_id": sid,
                    "reviewed_json": json_path.relative_to(context.raw_path.parent).as_posix(),
                    "artifact_checksum_of_reviewed_json": document["artifact_checksum"],
                    "qa_acceptance": "PENDING",
                    "alignment_ready": False,
                }
            ),
        )
    return {
        "reviewed_json": str(json_path),
        "reviewed_txt": str(txt_path),
        "unresolved_count": len(unresolved),
        "qa_acceptance": "PENDING",
        "alignment_ready": False,
        "raw_modified": False,
    }
