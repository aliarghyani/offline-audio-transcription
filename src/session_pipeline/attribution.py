"""Deterministic anonymous attribution. Only sealed local artifacts are inputs."""

from __future__ import annotations

import copy
import re
from bisect import bisect_left, bisect_right
from collections import Counter

from session_pipeline import __version__
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_quality import number, session_paths, union
from session_pipeline.asr_retry import exclusive_lock
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.models import sha256_file

SPEAKERS = ("SPEAKER_00", "SPEAKER_01")
POLICY = {
    "version": 1,
    "diagnostics_version": 2,
    "primary": "EXCLUSIVE_MAX_TURN_OVERLAP",
    "secondary": "REGULAR_MAX_TURN_OVERLAP_IF_NO_EXCLUSIVE_OVERLAP",
    "tie": "UNASSIGNED",
    "reviewed_fallback": "SINGLE_SUPPORTED_SPEAKER_WITHOUT_REGULAR_OVERLAP",
    "nearest_assignment": False,
    "identity_inference": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_interval(start, end):
    return number(start) and number(end) and 0 <= start < end


def overlap(start, end, turn):
    return max(0.0, min(end, turn["end"]) - max(start, turn["start"]))


class TurnIndex:
    """Positive-overlap lookup preserving original indices, including nested turns."""

    def __init__(self, turns):
        self.turns = turns
        self.starts = [t["start"] for t in turns]
        self.prefix_ends = []
        maximum = 0
        for t in turns:
            maximum = max(maximum, t["end"])
            self.prefix_ends.append(maximum)

    def __iter__(self):
        return iter(self.turns)

    def __bool__(self):
        return bool(self.turns)

    def candidates(self, interval):
        start, end = interval
        left = bisect_right(self.prefix_ends, start)
        right = bisect_left(self.starts, end)
        return [(i, self.turns[i]) for i in range(left, right) if self.turns[i]["end"] > start]


def checked_path(namespace, fingerprint, filename):
    require(
        isinstance(fingerprint, str) and bool(re.fullmatch(r"[0-9a-f]{64}", fingerprint)),
        "invalid lineage fingerprint",
    )
    path = (namespace / fingerprint / filename).resolve()
    require(path.is_relative_to(namespace.resolve()), "lineage escapes storage")
    return path


def select_path(namespace, fingerprint, filename):
    if fingerprint is None:
        paths = sorted(namespace.glob(f"*/{filename}"))
        require(len(paths) == 1, "select exactly one lineage with an explicit fingerprint")
        fingerprint = paths[0].parent.name
    return checked_path(namespace, fingerprint, filename)


def validate_turns(turns, duration, *, exclusive=False):
    require(isinstance(turns, list) and bool(turns), "empty or invalid diarization turns")
    previous = None
    previous_end = 0
    for turn in turns:
        start, end, speaker = turn["start"], turn["end"], turn["speaker"]
        require(
            valid_interval(start, end) and end <= duration and speaker in SPEAKERS,
            "invalid diarization interval or label",
        )
        key = (start, end, speaker)
        require(previous is None or previous <= key, "unordered diarization turns")
        if exclusive:
            require(start >= previous_end, "exclusive diarization contains overlapping turns")
        previous, previous_end = key, end


def load_parents(config, sid, aligned_fingerprint=None, diarization_fingerprint=None):
    raw, work, _ = session_paths(config, sid)
    dpath = select_path(raw.parent / "diarized", diarization_fingerprint, "diarization.json")
    diarized = read_artifact(dpath)
    afp = aligned_fingerprint or diarized["aligned_fingerprint"]
    apath = checked_path(raw.parent / "aligned", afp, "transcript.aligned.fa.json")
    aligned = read_artifact(apath)
    require(
        aligned["artifact_type"] == "aligned_asr"
        and aligned["status"] == "SUCCEEDED"
        and aligned["configuration"]["mode"] == "FULL"
        and aligned["qa"]["full_alignment_complete"] is True
        and aligned["qa"]["coverage_validation_pass"] is True,
        "incomplete aligned parent",
    )
    require(
        diarized["artifact_type"] == "diarization"
        and diarized["status"] in {"SUCCEEDED", "SUCCEEDED_WITH_WARNINGS"}
        and diarized["configuration"]["mode"] == "FULL"
        and diarized["full_diarization_complete"] is True
        and diarized["qa"]["validation_pass"] is True,
        "incomplete diarization parent",
    )
    require(
        aligned["session_id"] == diarized["session_id"] == sid
        and aligned["stage_fingerprint"] == apath.parent.name
        and diarized["stage_fingerprint"] == dpath.parent.name
        and diarized["aligned_fingerprint"] == apath.parent.name
        and diarized["parent_checksums"]["aligned_asr"] == aligned["artifact_checksum"]
        and aligned["source_sha256"] == diarized["source_sha256"],
        "parent fingerprints/checksums or source identity disagree",
    )
    # Verify the recorded file hash too, without importing any model contract.
    plan_path = work / "diarization" / dpath.parent.name / "plan.json"
    plan = read_artifact(plan_path)
    require(
        plan["stage_fingerprint"] == dpath.parent.name
        and plan["artifact_checksum"] == diarized["plan_checksum"]
        and plan["aligned_file_sha256"] == sha256_file(apath)
        and plan["aligned_fingerprint"] == apath.parent.name
        and plan["parent_checksums"] == diarized["parent_checksums"],
        "diarization plan parent binding changed",
    )
    duration = aligned["probe"]["duration_seconds"]
    require(
        number(duration)
        and duration > 0
        and diarized["source_duration_seconds"] == duration
        and diarized["processed_interval"] == [0, duration],
        "inconsistent duration",
    )
    validate_turns(diarized["turns"], duration)
    if diarized.get("exclusive_turns") is not None:
        validate_turns(diarized["exclusive_turns"], duration, exclusive=True)
        require(
            diarized["exclusive_qa"]["validation_pass"] is True, "exclusive turn validation failed"
        )
    ids = [s["reviewed_segment_id"] for s in aligned["segments"]]
    require(len(ids) == len(set(ids)) == aligned["segment_count"], "invalid segment identity")
    return apath, aligned, dpath, diarized


def overlap_regions(turns):
    """Sweep regular turns; count distinct active speakers, including nested turns."""
    events = {}
    for t in turns:
        events.setdefault(t["start"], []).append((t["speaker"], 1))
        events.setdefault(t["end"], []).append((t["speaker"], -1))
    active, regions = Counter(), []
    previous = 0
    for timestamp, changes in sorted(events.items()):
        if timestamp > previous and sum(v > 0 for v in active.values()) > 1:
            regions.append([previous, timestamp])
        for speaker, delta in changes:
            active[speaker] += delta
        previous = timestamp
    return union(regions)


def intersects(interval, regions):
    if interval is None:
        return False
    start, end = interval
    return any(min(end, b) > max(start, a) for a, b in regions)


def choose(interval, turns, method, *, conservative=False):
    candidates = Counter()
    relevant = turns.candidates(interval) if isinstance(turns, TurnIndex) else enumerate(turns)
    for _, t in relevant:
        amount = overlap(*interval, t)
        if amount > 0:
            candidates[t["speaker"]] = max(candidates[t["speaker"]], amount)
    best = max(candidates.values(), default=0)
    winners = [speaker for speaker, amount in candidates.items() if amount == best]
    ambiguous = bool(best) and (len(winners) != 1 or (conservative and len(candidates) != 1))
    return {
        "speaker": winners[0] if best and not ambiguous else None,
        "assignment_status": "AMBIGUOUS" if ambiguous else "ASSIGNED" if best else "UNASSIGNED",
        "assignment_method": method if best else "NO_OVERLAP",
        "ambiguity_status": "MULTIPLE_SUPPORTED_SPEAKERS"
        if ambiguous and conservative
        else "EXACT_TIE"
        if ambiguous
        else "NONE",
        "overlap_duration": best,
        "overlap_fraction": best / (interval[1] - interval[0]),
        "speaker_max_turn_overlaps": dict(sorted(candidates.items())),
    }


def diagnostics(interval, exclusive, regular, regions):
    if interval is None:
        return {"diagnostic_status": "NO_VALID_TEMPORAL_BOUNDS", "assignment_applied": False}
    start, end = interval
    prev = [t for t in exclusive if t["end"] <= start]
    nxt = [t for t in exclusive if t["start"] >= end]
    previous = max(prev, key=lambda t: (t["end"], t["start"])) if prev else None
    following = min(nxt, key=lambda t: (t["start"], t["end"])) if nxt else None
    same = previous["speaker"] == following["speaker"] if previous and following else None
    exclusive_overlap = any(overlap(start, end, t) > 0 for t in exclusive)
    neighbor_span = following["start"] - previous["end"] if previous and following else None
    return {
        "diagnostic_status": "AVAILABLE" if exclusive else "EXCLUSIVE_UNAVAILABLE",
        "previous_exclusive_turn": copy.deepcopy(previous),
        "next_exclusive_turn": copy.deepcopy(following),
        "previous_distance_seconds": start - previous["end"] if previous else None,
        "next_distance_seconds": following["start"] - end if following else None,
        "previous_speaker": previous["speaker"] if previous else None,
        "next_speaker": following["speaker"] if following else None,
        "same_neighbor_speaker": same,
        "gap_duration_seconds": None if exclusive_overlap else neighbor_span,
        "neighbor_window_duration_seconds": neighbor_span,
        "gap_status": "INSIDE_EXCLUSIVE_SPEECH"
        if exclusive_overlap
        else "EXCLUSIVE_GAP"
        if previous and following
        else "MISSING_NEIGHBOR",
        "regular_overlap": any(overlap(start, end, t) > 0 for t in regular),
        "inside_known_overlap_region": intersects(interval, regions),
        "exclusive_overlap": exclusive_overlap,
        "assignment_applied": False,
    }


def attribute_segments(aligned, diarized):
    regular = TurnIndex(diarized["turns"])
    exclusive = TurnIndex(diarized.get("exclusive_turns") or [])
    regions = overlap_regions(regular)
    result, unassigned, segment_candidates = [], [], []
    timed = fallback = affected = 0
    for s in aligned["segments"]:
        segment = copy.deepcopy(s)
        assigned_words = []
        for index, word in enumerate(s["words"]):
            valid = valid_interval(word.get("start"), word.get("end"))
            reviewed = (
                not valid
                and s.get("status") == "ALIGNMENT_FAILED"
                and word.get("start") is None
                and word.get("end") is None
            )
            interval = (
                [word["start"], word["end"]]
                if valid
                else (list(s["reviewed_interval"]) if reviewed else None)
            )
            if interval is not None:
                require(valid_interval(*interval), "invalid word/fallback bounds")
            timed += valid
            fallback += reviewed
            overlap_affected = intersects(interval, regions)
            affected += overlap_affected
            if interval is None:
                assignment = {
                    "speaker": None,
                    "assignment_status": "UNASSIGNED",
                    "assignment_method": "NO_VALID_TIMING",
                    "ambiguity_status": "NONE",
                    "overlap_duration": 0,
                    "overlap_fraction": None,
                    "speaker_max_turn_overlaps": {},
                }
            else:
                assignment = choose(
                    interval, exclusive, "EXCLUSIVE_MAX_TURN_OVERLAP", conservative=reviewed
                )
                if assignment["overlap_duration"] == 0:
                    assignment = choose(
                        interval, regular, "REGULAR_MAX_TURN_OVERLAP", conservative=reviewed
                    )
                if reviewed and overlap_affected and assignment["speaker"] is not None:
                    assignment.update(
                        speaker=None,
                        assignment_status="AMBIGUOUS",
                        ambiguity_status="REVIEWED_BOUNDS_OVERLAP",
                    )
            assignment.update(
                timing_source="ALIGNED_WORD"
                if valid
                else "REVIEWED_SEGMENT_FALLBACK"
                if reviewed
                else "UNAVAILABLE",
                attribution_interval=interval,
                overlap_affected=overlap_affected,
                regular_turn_overlaps=[
                    {
                        "turn_index": i,
                        "speaker": t["speaker"],
                        "overlap_seconds": overlap(*interval, t),
                    }
                    for i, t in (regular.candidates(interval) if interval else [])
                    if overlap(*interval, t) > 0
                ],
            )
            new_word = copy.deepcopy(word)
            new_word["attribution"] = assignment
            if assignment["speaker"] is None:
                diagnostic = diagnostics(interval, exclusive, regular, regions)
                assignment["fallback_diagnostics"] = diagnostic
                unassigned.append(
                    {
                        "reviewed_segment_id": s["reviewed_segment_id"],
                        "word_index": index,
                        **diagnostic,
                    }
                )
            assigned_words.append(new_word)
        segment["words"] = assigned_words
        speakers = sorted(
            {
                w["attribution"]["speaker"]
                for w in assigned_words
                if w["attribution"]["speaker"] is not None
            }
        )
        missing = sum(w["attribution"]["speaker"] is None for w in assigned_words)
        state = (
            "MIXED_SPEAKER"
            if len(speakers) > 1
            else "UNASSIGNED"
            if not speakers
            else "PARTIAL"
            if missing
            else "SINGLE_SPEAKER"
        )
        segment["attribution"] = {
            "state": state,
            "speakers": speakers,
            "speaker": speakers[0] if state == "SINGLE_SPEAKER" else None,
            "unassigned_word_count": missing,
            "is_partial": bool(missing),
            "has_speaker_change": len(speakers) > 1,
        }
        if missing or not assigned_words:
            candidate = diagnostics(s["reviewed_interval"], exclusive, regular, regions)
            segment["attribution"]["fallback_diagnostics"] = candidate
            segment_candidates.append(
                {"reviewed_segment_id": s["reviewed_segment_id"], **candidate}
            )
        result.append(segment)
    words = [w for s in result for w in s["words"]]
    counts = Counter(w["attribution"]["speaker"] or "UNASSIGNED" for w in words)
    states = Counter(s["attribution"]["state"] for s in result)
    former_ids = diarized["qa"].get("uncovered_reviewed_segment_ids", [])
    require(len(former_ids) == len(set(former_ids)), "duplicate uncovered segment references")
    by_id = {s["reviewed_segment_id"]: s for s in result}
    require(all(sid in by_id for sid in former_ids), "unknown uncovered segment reference")
    former = []
    for sid in former_ids:
        s = by_id[sid]
        missing = s["attribution"]["unassigned_word_count"]
        count = len(s["words"])
        classification = (
            "FULLY_ATTRIBUTABLE"
            if count and missing == 0
            else ("PARTIALLY_ATTRIBUTABLE" if count > missing else "UNASSIGNED")
        )
        former.append(
            {
                "reviewed_segment_id": sid,
                "classification": classification,
                "word_count": count,
                "unassigned_words": missing,
            }
        )
    former_counts = Counter(s["classification"] for s in former)
    assigned = len(words) - counts["UNASSIGNED"]
    qa = {
        "total_segments": len(result),
        "total_words": len(words),
        "total_timed_words": timed,
        "assigned_words": {speaker: counts[speaker] for speaker in SPEAKERS},
        "unassigned_words": counts["UNASSIGNED"],
        "segment_counts": {
            key: states[key] for key in ("SINGLE_SPEAKER", "MIXED_SPEAKER", "PARTIAL", "UNASSIGNED")
        },
        "attribution_coverage_percent": 100 * assigned / len(words) if words else 0,
        "timed_word_coverage_percent": 100
        * sum(
            w["attribution"]["speaker"] is not None
            and w["attribution"]["timing_source"] == "ALIGNED_WORD"
            for w in words
        )
        / timed
        if timed
        else 0,
        "fallback_timing_words": fallback,
        "overlap_affected_words": affected,
        "ambiguous_words": sum(w["attribution"]["assignment_status"] == "AMBIGUOUS" for w in words),
        "identity_mapping_status": "HUMAN_MAPPING_REQUIRED",
        "former_uncovered_bounds": {
            "total": len(former),
            **{
                key: former_counts[key]
                for key in ("FULLY_ATTRIBUTABLE", "PARTIALLY_ATTRIBUTABLE", "UNASSIGNED")
            },
        },
        "fallback_candidates": {
            "words": len(unassigned),
            "segments": len(segment_candidates),
            "same_neighbor_speaker": sum(
                d.get("same_neighbor_speaker") is True for d in unassigned
            ),
            "different_neighbor_speakers": sum(
                d.get("same_neighbor_speaker") is False for d in unassigned
            ),
            "missing_neighbor": sum(d.get("same_neighbor_speaker") is None for d in unassigned),
            "regular_overlap": sum(d.get("regular_overlap", False) for d in unassigned),
            "known_overlap_region": sum(
                d.get("inside_known_overlap_region", False) for d in unassigned
            ),
            "gap_duration_buckets": {
                str(limit): sum(
                    d.get("gap_duration_seconds") is not None and d["gap_duration_seconds"] <= limit
                    for d in unassigned
                )
                for limit in (0.1, 0.5, 1, 2, 5)
            },
            "assignment_applied": False,
        },
    }
    return (
        result,
        qa,
        {
            "regular_overlap_regions": regions,
            "former_uncovered_bounds": former,
            "unassigned_words": unassigned,
            "unassigned_segments": segment_candidates,
        },
    )


def make_document(config, sid, aligned_fingerprint=None, diarization_fingerprint=None):
    apath, aligned, dpath, diarized = load_parents(
        config, sid, aligned_fingerprint, diarization_fingerprint
    )
    parents = {
        "aligned_asr": aligned["artifact_checksum"],
        "diarization": diarized["artifact_checksum"],
    }
    fp = stage_fingerprint(
        stage="deterministic_speaker_attribution",
        source_sha256=aligned["source_sha256"],
        relevant_config=POLICY,
        parent_checksums=parents,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    segments, qa, diagnostic = attribute_segments(aligned, diarized)
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "anonymous_speaker_attribution",
            "stage_fingerprint": fp,
            "session_id": sid,
            "status": "SUCCEEDED_WITH_WARNINGS" if qa["unassigned_words"] else "SUCCEEDED",
            "source_sha256": aligned["source_sha256"],
            "source": copy.deepcopy(aligned["source"]),
            "parent_checksums": parents,
            "parents": {
                name: {
                    "fingerprint": obj["stage_fingerprint"],
                    "checksum": obj["artifact_checksum"],
                    "file_sha256": sha256_file(path),
                    "relative_path": path.relative_to(config.application.root).as_posix(),
                }
                for name, path, obj in [
                    ("aligned_asr", apath, aligned),
                    ("diarization", dpath, diarized),
                ]
            },
            "configuration": POLICY,
            "pipeline_version": __version__,
            "segment_count": len(segments),
            "segments": segments,
            "qa": qa,
            "diagnostics": diagnostic,
            "diarization_evidence": {
                "regular_turns": copy.deepcopy(diarized["turns"]),
                "exclusive_turns": copy.deepcopy(diarized.get("exclusive_turns")),
                "primary_source": "EXCLUSIVE" if diarized.get("exclusive_turns") else "REGULAR",
            },
            "ml_inference_used": False,
            "cloud_llm_used": False,
            "speaker_identities_assigned": False,
        }
    )
    raw, _, _ = session_paths(config, sid)
    path = checked_path(raw.parent / "attributed", fp, "transcript.attributed.fa.json")
    return path, document


def execute(config, sid, aligned_fingerprint=None, diarization_fingerprint=None):
    path, document = make_document(config, sid, aligned_fingerprint, diarization_fingerprint)
    _, work, _ = session_paths(config, sid)
    with exclusive_lock(work / "attribution" / ".write.lock"):
        # Revalidate parents immediately before publication; no upstream writes.
        for parent in document["parents"].values():
            p = config.application.root / parent["relative_path"]
            current = read_artifact(p)
            require(
                sha256_file(p) == parent["file_sha256"]
                and current["artifact_checksum"] == parent["checksum"]
                and current["stage_fingerprint"] == parent["fingerprint"],
                "parent changed during attribution",
            )
        if path.exists():
            require(read_artifact(path) == document, "existing anonymous attribution changed")
        else:
            atomic_write_json(path, document)
    return {
        "state": document["status"],
        "fingerprint": path.parent.name,
        "artifact_path": str(path),
        **document["qa"],
    }


def load_attributed(config, sid, fingerprint=None):
    raw, _, _ = session_paths(config, sid)
    path = select_path(raw.parent / "attributed", fingerprint, "transcript.attributed.fa.json")
    document = read_artifact(path)
    parents = document["parents"]
    expected_path, expected = make_document(
        config, sid, parents["aligned_asr"]["fingerprint"], parents["diarization"]["fingerprint"]
    )
    require(path == expected_path and document == expected, "attribution or parent lineage changed")
    return path, document


def status(config, sid, fingerprint=None):
    path, document = load_attributed(config, sid, fingerprint)
    return {
        "state": document["status"],
        "fingerprint": path.parent.name,
        "artifact_path": str(path),
        **document["qa"],
    }
