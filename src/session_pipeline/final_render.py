"""Deterministic final rendering from anonymous attribution and human confirmation."""

from __future__ import annotations

import copy
import hashlib
from collections import Counter

from session_pipeline import __version__, attribution, speaker_review
from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact,
    seal_artifact,
)
from session_pipeline.asr_quality import session_paths
from session_pipeline.asr_retry import exclusive_lock
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.models import sha256_file
from session_pipeline.review_merge import write_text_atomic

require = attribution.require
UNKNOWN = "نامشخص"
POLICY = {
    "renderer_version": 1,
    "text_policy": "EXACT_SOURCE_CHARACTER_SLICES_OR_EXACT_PARENT_BLOCK",
    "maximum_group_gap_seconds": 0.75,
    "cross_segment_grouping": "SINGLE_SPEAKER_ONLY",
    "overlap_grouping": False,
    "speaker_change_grouping": False,
    "unknown_cross_segment_grouping": False,
    "timestamp_format": "HH:MM:SS.mmm",
    "text_normalization": False,
    "nearest_speaker_fallback": False,
}


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def label_for(segment, mapping):
    speakers = sorted(
        {
            w["attribution"]["speaker"]
            for w in segment["words"]
            if w["attribution"]["speaker"] is not None
        }
    )
    labels = [mapping[speaker] for speaker in speakers]
    if not labels or any(w["attribution"]["speaker"] is None for w in segment["words"]):
        labels.append(UNKNOWN)
    return " / ".join(labels)


def exact_spans(segment):
    """Prove word offsets without reconstructing text from tokens."""
    previous = 0
    words, text = segment["words"], segment["text"]
    for word in words:
        interval = word.get("character_interval")
        if not (
            isinstance(interval, list)
            and len(interval) == 2
            and all(type(n) is int for n in interval)
        ):
            return False
        start, end = interval
        if not (previous <= start < end <= len(text) and text[start:end] == word["word"]):
            return False
        previous = end
    return bool(words)


def timing(word, segment, duration):
    interval = word["attribution"].get("attribution_interval")
    source = word["attribution"]["timing_source"]
    if interval is None:
        interval = segment["reviewed_interval"]
        source = "REVIEWED_SEGMENT_RENDER_BOUNDS"
    require(
        isinstance(interval, list)
        and len(interval) == 2
        and attribution.valid_interval(*interval)
        and interval[1] <= duration,
        "invalid rendering time bounds",
    )
    return interval, source


def can_join(previous, following, timeline, regions, *, within_segment=False):
    same_segment = previous["segment_id"] == following["segment_id"]
    if previous["speaker"] != following["speaker"]:
        return False
    if not same_segment:
        if (
            previous["speaker"] is None
            or previous["source_status"] != "SINGLE_SPEAKER"
            or following["source_status"] != "SINGLE_SPEAKER"
        ):
            return False
    # Untimed words share reviewed segment bounds. A contiguous unknown run can
    # retain that one interval without pretending to have individual word times.
    if (
        within_segment
        and same_segment
        and previous["speaker"] is None
        and previous["timing_source"] == following["timing_source"] == "REVIEWED_SEGMENT_FALLBACK"
        and [previous["start"], previous["end"]] == [following["start"], following["end"]]
    ):
        return True
    if previous["speaker"] is None and not within_segment:
        return False
    gap = following["start"] - previous["end"]
    if not 0 <= gap <= POLICY["maximum_group_gap_seconds"]:
        return False
    extent = [previous["start"], following["end"]]
    if attribution.intersects(extent, regions):
        return False
    return not any(
        t["speaker"] != previous["speaker"] and attribution.overlap(*extent, t) > 0
        for t in timeline
    )


def segment_runs(segment, mapping, duration, timeline, regions):
    text, words = segment["text"], segment["words"]
    require(isinstance(text, str), "source text must be an exact string")
    intervals = [timing(w, segment, duration) for w in words]
    safe_timing = all(
        a[0][0] <= b[0][0] and a[0][1] <= b[0][1]
        for a, b in zip(intervals, intervals[1:], strict=False)
    )
    exact = exact_spans(segment)
    base = {
        "segment_id": segment["reviewed_segment_id"],
        "source_status": segment["attribution"]["state"],
    }
    if not exact or not safe_timing:
        start, end = segment["reviewed_interval"]
        require(
            attribution.valid_interval(start, end) and end <= duration,
            "invalid reviewed block bounds",
        )
        speaker = segment["attribution"]["speaker"]
        return [
            {
                **base,
                "speaker": speaker,
                "label": label_for(segment, mapping),
                "text": text,
                "character_interval": [0, len(text)],
                "word_indices": list(range(len(words))),
                "start": start,
                "end": end,
                "timing_source": "REVIEWED_SEGMENT_RENDER_BOUNDS",
                "render_strategy": "EXACT_PARENT_BLOCK",
                "render_warning": "WORD_CHARACTER_SPANS_UNPROVEN"
                if not exact
                else "WORD_TIMING_ORDER_UNSAFE_FOR_SPLIT",
            }
        ]
    pieces = []
    for index, (word, (interval, source)) in enumerate(zip(words, intervals, strict=True)):
        left = 0 if index == 0 else word["character_interval"][0]
        right = words[index + 1]["character_interval"][0] if index + 1 < len(words) else len(text)
        speaker = word["attribution"]["speaker"]
        piece = {
            **base,
            "speaker": speaker,
            "label": mapping.get(speaker, UNKNOWN),
            "text": text[left:right],
            "character_interval": [left, right],
            "word_indices": [index],
            "start": interval[0],
            "end": interval[1],
            "timing_source": source,
            "render_strategy": "EXACT_CHARACTER_SPEAKER_RUN",
        }
        if pieces and can_join(pieces[-1], piece, timeline, regions, within_segment=True):
            pieces[-1]["text"] += piece["text"]
            pieces[-1]["character_interval"][1] = right
            pieces[-1]["word_indices"].append(index)
            pieces[-1]["end"] = max(pieces[-1]["end"], piece["end"])
            if pieces[-1]["timing_source"] != source:
                pieces[-1]["timing_source"] = "MIXED_ORIGINAL_TIMING_SOURCES"
        else:
            pieces.append(piece)
    return pieces


def plan_turns(document, human_mapping, duration=None):
    require(
        human_mapping is not None
        and human_mapping.get("human_confirmed") is True
        and human_mapping.get("identity_inference_used") is False
        and human_mapping.get("attribution_checksum") == document["artifact_checksum"],
        "current human identity mapping required",
    )
    speaker_review.validate_mapping(human_mapping["mapping"])
    mapping = human_mapping["mapping"]
    evidence = document["diarization_evidence"]
    timeline = evidence.get("exclusive_turns") or evidence["regular_turns"]
    if duration is None:
        duration = max(
            [t["end"] for t in timeline]
            + [s["end"] for s in document["segments"]]
            + [
                w["attribution"]["attribution_interval"][1]
                for s in document["segments"]
                for w in s["words"]
                if w["attribution"].get("attribution_interval") is not None
            ]
        )
    regions = attribution.overlap_regions(evidence["regular_turns"])
    turns = []
    for segment in document["segments"]:
        for piece in segment_runs(segment, mapping, duration, timeline, regions):
            if (
                turns
                and turns[-1]["pieces"][-1]["render_strategy"] != "EXACT_PARENT_BLOCK"
                and piece["render_strategy"] != "EXACT_PARENT_BLOCK"
                and can_join(turns[-1]["pieces"][-1], piece, timeline, regions)
            ):
                turns[-1]["pieces"].append(piece)
                turns[-1]["end"] = piece["end"]
            else:
                turns.append(
                    {
                        "speaker": piece["speaker"],
                        "label": piece["label"],
                        "start": piece["start"],
                        "end": piece["end"],
                        "pieces": [piece],
                    }
                )
    return turns


def validate_coverage(segments, turns, duration):
    ids = [s["reviewed_segment_id"] for s in segments]
    require(len(ids) == len(set(ids)), "duplicate source segment identity")
    by_id = {s["reviewed_segment_id"]: s for s in segments}
    source_order = {sid: index for index, sid in enumerate(ids)}
    represented, previous_index, previous_start, previous_end = {}, -1, -1, -1
    for turn in turns:
        require(
            attribution.valid_interval(turn["start"], turn["end"])
            and turn["end"] <= duration
            and turn["start"] >= previous_start
            and turn["end"] >= previous_end,
            "rendered timestamp order/bounds invalid",
        )
        previous_start, previous_end = turn["start"], turn["end"]
        for piece in turn["pieces"]:
            sid = piece["segment_id"]
            require(
                sid in by_id and source_order[sid] >= previous_index,
                "source segments dropped or reordered",
            )
            require(
                attribution.valid_interval(piece["start"], piece["end"])
                and turn["start"] <= piece["start"] < piece["end"] <= turn["end"]
                and piece["speaker"] == turn["speaker"]
                and piece["label"] == turn["label"],
                "piece timing or speaker identity changed",
            )
            if piece["render_strategy"] == "EXACT_CHARACTER_SPEAKER_RUN":
                require(
                    all(
                        type(i) is int
                        and 0 <= i < len(by_id[sid]["words"])
                        and by_id[sid]["words"][i]["attribution"]["speaker"] == piece["speaker"]
                        for i in piece["word_indices"]
                    ),
                    "unresolved word silently assigned or anonymous identity changed",
                )
            previous_index = source_order[sid]
            represented.setdefault(sid, []).append(piece)
    require(set(represented) == set(ids), "source segment disappeared")
    reconstructed = []
    for sid in ids:
        segment, pieces = by_id[sid], represented[sid]
        cursor, indices = 0, []
        for piece in pieces:
            left, right = piece["character_interval"]
            require(
                type(left) is int
                and type(right) is int
                and left == cursor
                and left <= right <= len(segment["text"])
                and piece["text"] == segment["text"][left:right],
                "text changed, duplicated, or dropped",
            )
            cursor = right
            indices.extend(piece["word_indices"])
        require(
            cursor == len(segment["text"]) and indices == list(range(len(segment["words"]))),
            "source word/character coverage changed",
        )
        reconstructed.append({"segment_id": sid, "text": "".join(p["text"] for p in pieces)})
    source = [{"segment_id": s["reviewed_segment_id"], "text": s["text"]} for s in segments]
    require(source == reconstructed, "source text reconstruction mismatch")
    return {
        "validation_pass": True,
        "source_text_sha256": digest(source),
        "reconstructed_text_sha256": digest(reconstructed),
        "represented_segments": len(ids),
    }


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def render_txt(turns):
    chunks, spans = [], []
    position = byte_position = 0

    def append(text):
        nonlocal position, byte_position
        chunks.append(text)
        position += len(text)
        byte_position += len(text.encode("utf-8"))

    for index, turn in enumerate(turns):
        append(f"[{timestamp(turn['start'])} — {timestamp(turn['end'])}] {turn['label']}:\n")
        previous_segment = None
        for piece in turn["pieces"]:
            if previous_segment is not None and piece["segment_id"] != previous_segment:
                append("\n")  # Presentation only; original strings remain exact slices.
            left, byte_left = position, byte_position
            append(piece["text"])
            spans.append(
                {
                    "turn_index": index,
                    "segment_id": piece["segment_id"],
                    "source_character_interval": piece["character_interval"],
                    "txt_character_interval": [left, position],
                    "txt_byte_interval": [byte_left, byte_position],
                }
            )
            previous_segment = piece["segment_id"]
        append("\n\n")
    return "".join(chunks), spans


def validate_txt(segments, turns, text, spans):
    pieces = [p for t in turns for p in t["pieces"]]
    require(len(spans) == len(pieces), "TXT source slice disappeared")
    encoded = text.encode("utf-8")
    previous = previous_byte = 0
    recovered = {}
    for piece, span in zip(pieces, spans, strict=True):
        left, right = span["txt_character_interval"]
        byte_left, byte_right = span["txt_byte_interval"]
        require(
            span["segment_id"] == piece["segment_id"]
            and span["source_character_interval"] == piece["character_interval"]
            and previous <= left <= right <= len(text)
            and previous_byte <= byte_left <= byte_right <= len(encoded)
            and text[left:right] == piece["text"]
            and encoded[byte_left:byte_right] == piece["text"].encode("utf-8"),
            "TXT source text changed, duplicated or dropped",
        )
        previous, previous_byte = right, byte_right
        recovered.setdefault(piece["segment_id"], []).append(text[left:right])
    require(
        all("".join(recovered.get(s["reviewed_segment_id"], [])) == s["text"] for s in segments),
        "TXT exact source reconstruction failed",
    )
    expected_text, expected_spans = render_txt(turns)
    require(text == expected_text and spans == expected_spans, "TXT presentation changed")
    return True


def load_inputs(config, sid, attributed_fingerprint):
    apath, attributed = attribution.load_attributed(config, sid, attributed_fingerprint)
    directory, bundle, current = speaker_review.load_bundle(config, sid, apath.parent.name)
    require(current == attributed, "attribution changed during mapping validation")
    mapping = speaker_review.load_mapping(directory, bundle)
    require(mapping is not None, "persisted HUMAN_CONFIRMED identity mapping required")
    return apath, attributed, directory, bundle, mapping


def build_document(config, sid, attributed_fingerprint):
    apath, attributed, directory, bundle, mapping = load_inputs(config, sid, attributed_fingerprint)
    diarized_path = config.application.root / attributed["parents"]["diarization"]["relative_path"]
    duration = read_artifact(diarized_path)["source_duration_seconds"]
    turns = plan_turns(attributed, mapping, duration)
    preservation = validate_coverage(attributed["segments"], turns, duration)
    text, spans = render_txt(turns)
    validate_txt(attributed["segments"], turns, text, spans)
    segments = copy.deepcopy(attributed["segments"])
    word_counts, unresolved = Counter(), []
    for segment in segments:
        segment["speaker_identity"] = {
            "human_speaker": label_for(segment, mapping["mapping"]),
            "anonymous_speaker": segment["attribution"]["speaker"],
            "speaker_status": segment["attribution"]["state"],
            "mapping_source": "HUMAN_CONFIRMED",
        }
        for index, word in enumerate(segment["words"]):
            anonymous = word["attribution"]["speaker"]
            human = mapping["mapping"].get(anonymous, UNKNOWN)
            word["speaker_identity"] = {
                "anonymous_speaker": anonymous,
                "human_speaker": human,
                "mapping_source": "HUMAN_CONFIRMED"
                if anonymous
                else "UNRESOLVED_ANONYMOUS_ATTRIBUTION",
            }
            word_counts[human] += 1
            if anonymous is None:
                unresolved.append(
                    {
                        "segment_id": segment["reviewed_segment_id"],
                        "word_index": index,
                        "aligned_interval": [word["start"], word["end"]],
                        "reviewed_interval": segment["reviewed_interval"],
                        "attribution_interval": word["attribution"]["attribution_interval"],
                        "timing_source": word["attribution"]["timing_source"],
                        "human_speaker": UNKNOWN,
                    }
                )
    for original, final in zip(attributed["segments"], segments, strict=True):
        restored = {k: v for k, v in final.items() if k != "speaker_identity"}
        restored["words"] = [
            {k: v for k, v in w.items() if k != "speaker_identity"} for w in final["words"]
        ]
        require(restored == original, "source segment or word metadata changed")
    blocks = [p for t in turns for p in t["pieces"] if p["render_strategy"] == "EXACT_PARENT_BLOCK"]
    qa = {
        "total_source_segments": len(segments),
        "final_represented_segments": len(segments),
        "total_words": sum(word_counts.values()),
        "ali_words": word_counts["علی"],
        "consultant_words": word_counts["مشاور"],
        "unknown_words": word_counts[UNKNOWN],
        "segment_counts": copy.deepcopy(attributed["qa"]["segment_counts"]),
        "rendered_turn_count": len(turns),
        "text_preservation": preservation,
        "txt_text_preservation_pass": True,
        "segment_id_preservation_pass": True,
        "word_metadata_preservation_pass": True,
        "timestamp_ordering_bounds_pass": True,
        "identity_mapping_status": "HUMAN_CONFIRMED",
        "identity_mapping_checksum": mapping["artifact_checksum"],
        "unresolved_words": unresolved,
        "overlap_affected_words": attributed["qa"]["overlap_affected_words"],
        "reviewed_timing_fallback_words": attributed["qa"]["fallback_timing_words"],
        "exact_parent_block_segments": len({p["segment_id"] for p in blocks}),
        "parent_block_warnings": [
            {"segment_id": p["segment_id"], "warning": p["render_warning"]} for p in blocks
        ],
        "upstream_immutability_pass": True,
        "warnings": [
            key
            for key, present in {
                "UNRESOLVED_ANONYMOUS_ATTRIBUTION": bool(unresolved),
                "REGULAR_DIARIZATION_OVERLAP_PRESENT": bool(
                    attributed["qa"]["overlap_affected_words"]
                ),
                "EXACT_PARENT_BLOCK_FALLBACK": bool(blocks),
            }.items()
            if present
        ],
    }
    inputs = {
        "attribution": apath,
        "human_mapping": speaker_review.mapping_path(directory),
        "identity_review_bundle": directory / "bundle.json",
        "diarization": diarized_path,
        "aligned_asr": config.application.root
        / attributed["parents"]["aligned_asr"]["relative_path"],
    }
    bindings = {
        name: {
            "relative_path": path.relative_to(config.application.root).as_posix(),
            "artifact_checksum": read_artifact(path)["artifact_checksum"],
            "file_sha256": sha256_file(path),
        }
        for name, path in inputs.items()
    }
    require(
        bindings["attribution"]["artifact_checksum"] == attributed["artifact_checksum"]
        and bindings["human_mapping"]["artifact_checksum"] == mapping["artifact_checksum"]
        and bindings["identity_review_bundle"]["artifact_checksum"] == bundle["artifact_checksum"],
        "rendering inputs changed",
    )
    parents = {name: binding["artifact_checksum"] for name, binding in bindings.items()}
    fp = stage_fingerprint(
        stage="human_labelled_final_transcript",
        source_sha256=attributed["source_sha256"],
        relevant_config=POLICY,
        parent_checksums=parents,
        pipeline_version=__version__,
        stage_schema_version=1,
    )
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "final_human_labelled_transcript",
            "session_id": sid,
            "status": "SUCCEEDED_WITH_WARNINGS" if qa["warnings"] else "SUCCEEDED",
            "stage_fingerprint": fp,
            "renderer_version": POLICY["renderer_version"],
            "configuration": copy.deepcopy(POLICY),
            "pipeline_version": __version__,
            "source_sha256": attributed["source_sha256"],
            "source_duration_seconds": duration,
            "source": copy.deepcopy(attributed["source"]),
            "parent_checksums": parents,
            "input_bindings": bindings,
            "attribution_fingerprint": apath.parent.name,
            "attribution_checksum": attributed["artifact_checksum"],
            "attribution_parent_provenance": copy.deepcopy(attributed["parents"]),
            "identity_mapping": copy.deepcopy(mapping),
            "mapping_source": "HUMAN_CONFIRMED",
            "segment_count": len(segments),
            "segments": segments,
            "rendered_turns": turns,
            "txt_source_spans": spans,
            "txt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "qa": qa,
            "ml_inference_used": False,
            "cloud_llm_used": False,
            "transcript_text_rewritten": False,
        }
    )
    raw, _, _ = session_paths(config, sid)
    path = attribution.checked_path(raw.parent / "final", fp, "transcript.final.fa.json")
    return path, document, text


def recheck_inputs(config, bindings):
    for binding in bindings.values():
        path = config.application.root / binding["relative_path"]
        require(
            read_artifact(path)["artifact_checksum"] == binding["artifact_checksum"]
            and sha256_file(path) == binding["file_sha256"],
            "upstream input changed during render",
        )


def report(path, document):
    return {
        "state": document["status"],
        "final_fingerprint": path.parent.name,
        "json_path": str(path),
        "txt_path": str(path.with_suffix(".txt")),
        **document["qa"],
    }


def execute(config, sid, attributed_fingerprint):
    path, document, text = build_document(config, sid, attributed_fingerprint)
    _, work, _ = session_paths(config, sid)
    with exclusive_lock(work / "final-render" / ".write.lock"):
        recheck_inputs(config, document["input_bindings"])
        txt = path.with_suffix(".txt")
        if txt.exists():
            require(txt.read_bytes() == text.encode("utf-8"), "existing final TXT changed")
        else:
            require(not path.exists(), "committed final JSON is missing its TXT")
            write_text_atomic(txt, text)
        recheck_inputs(config, document["input_bindings"])
        if path.exists():
            require(read_artifact(path) == document, "existing final JSON changed")
        else:
            atomic_write_json(path, document)
    return report(path, document)


def status(config, sid, fingerprint=None):
    raw, _, _ = session_paths(config, sid)
    path = attribution.select_path(raw.parent / "final", fingerprint, "transcript.final.fa.json")
    document = read_artifact(path)
    expected_path, expected, text = build_document(config, sid, document["attribution_fingerprint"])
    require(
        path == expected_path and document == expected, "final artifact or human mapping changed"
    )
    require(path.with_suffix(".txt").read_bytes() == text.encode("utf-8"), "final TXT changed")
    recheck_inputs(config, document["input_bindings"])
    return report(path, document)
