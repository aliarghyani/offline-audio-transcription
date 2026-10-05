"""Offline word-boundary recovery and explicitly authorized AUTO_PREFER_B merge.

Legacy human decisions remain immutable inputs, never relabeled as human verification.
"""

from __future__ import annotations

import copy
import hashlib
import unicodedata
from collections import Counter
from importlib.resources import files
from pathlib import Path

from session_pipeline.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact,
    seal_artifact,
)
from session_pipeline.asr_quality import number
from session_pipeline.asr_retry import exclusive_lock
from session_pipeline.human_review import load_context, require, source_path
from session_pipeline.models import sha256_file, utc_timestamp
from session_pipeline.review_merge import render_reviewed, write_text_atomic

POLICY = "WORD_INTERVAL_OVERLAP_EXPAND_TO_FULL_WORD"
STATES = ("DIRECT_TARGET", "RECOVERED_FROM_CONTEXT", "NO_TARGET_OVERLAP", "CORRUPT")


def overlap(word, span):
    return min(word["end"], span[1]) > max(word["start"], span[0])


def timeline(segments, bounds):
    result = []
    previous = bounds[0]
    for si, segment in enumerate(segments):
        require(isinstance(segment.get("text"), str), "missing segment text")
        words = segment.get("words")
        require(isinstance(words, list), "missing word timestamps")
        require(words or not segment["text"].strip(), "speech without word timestamps")
        for wi, word in enumerate(words):
            a, b = word.get("start"), word.get("end")
            require(
                number(a)
                and number(b)
                and bounds[0] <= a <= b <= bounds[1]
                and a >= previous
                and isinstance(word.get("word"), str),
                "invalid or unordered word timestamps",
            )
            previous = a
            result.append((f"{si}:{wi}", word))
    return result


def recover(candidate, originals):
    """Close the shared interval over whole A/B words; never splice a partial word."""
    target = candidate["window"]["target_interval"]
    decoded = candidate["window"]["decoded_interval"]
    result = {
        "review_id": candidate["artifact_checksum"],
        "window_index": candidate["window"]["window_index"],
        "target_interval": target,
        "effective_interval": list(target),
        "decoded_interval": decoded,
        "classification": "CORRUPT",
        "replacement_segments": [],
        "retry_word_ids": [],
        "original_word_ids": [],
        "original_overlap_word_count": 0,
        "issues": [],
    }
    try:
        require(not candidate.get("_recovery_error"), candidate.get("_recovery_error", ""))
        retry = timeline(candidate["candidate_b"]["raw_segments"], decoded)
        selected = [(key, w) for key, w in retry if overlap(w, target) and w["word"].strip()]
        result["original_overlap_word_count"] = len(selected)
        if not selected:
            result["classification"] = "NO_TARGET_OVERLAP"
            return result
        # Original validation is scoped to segments touched by the expanding span.
        # Unrelated historical zero-duration words are retained as-is.
        span = list(target)
        for _ in range(len(retry) + sum(len(s.get("words") or []) for s in originals) + 2):
            before = list(span)
            selected = [(key, w) for key, w in retry if overlap(w, span)]
            require(all(w["word"].strip() for _, w in selected), "empty overlapping retry word")
            span = [
                min(span[0], *(w["start"] for _, w in selected)),
                max(span[1], *(w["end"] for _, w in selected)),
            ]
            original_words = []
            for si, s in enumerate(originals):
                if not any(
                    overlap(w, span) or span[0] <= w["start"] == w["end"] < span[1]
                    for w in s.get("words") or []
                ) and (min(s["end"], span[1]) <= max(s["start"], span[0])):
                    continue
                words = timeline([s], [0, float("inf")])
                # Partial reconstruction must preserve every original lexical token.
                require(
                    "".join(s["text"].split())
                    == "".join("".join(w["word"] for _, w in words).split()),
                    "original segment text cannot be reconstructed from its words",
                )
                for key, w in words:
                    if overlap(w, span) or span[0] <= w["start"] == w["end"] < span[1]:
                        original_words.append((f"{si}:{key.split(':')[1]}", w))
            if original_words:
                span = [
                    min(span[0], *(w["start"] for _, w in original_words)),
                    max(span[1], *(w["end"] for _, w in original_words)),
                ]
            require(
                decoded[0] <= span[0] < span[1] <= decoded[1],
                "whole-word reconciliation exceeds decoded evidence",
            )
            if span == before:
                break
        else:
            raise ValueError("word boundary closure did not converge")
        selected = [(key, w) for key, w in retry if overlap(w, span)]
        require(
            all(span[0] <= w["start"] < w["end"] <= span[1] for _, w in selected),
            "ambiguous retry word at replacement boundary",
        )
        result.update(
            effective_interval=span,
            retry_word_ids=[key for key, _ in selected],
            original_word_ids=[key for key, _ in original_words],
        )
        # Keep lexical order even when successive model word intervals overlap.
        for si, s in enumerate(candidate["candidate_b"]["raw_segments"]):
            words = [copy.deepcopy(w) for key, w in selected if key.startswith(f"{si}:")]
            if words:
                result["replacement_segments"].append(
                    {
                        **copy.deepcopy(s),
                        "start": min(w["start"] for w in words),
                        "end": max(w["end"] for w in words),
                        "words": words,
                        "text": "".join(w["word"] for w in words).strip(),
                    }
                )
        result["classification"] = (
            "DIRECT_TARGET"
            if any(s["text"].strip() for s in candidate["candidate_b"]["segments"])
            else "RECOVERED_FROM_CONTEXT"
        )
    except (ValueError, KeyError, TypeError) as error:
        result.update(
            classification="CORRUPT",
            replacement_segments=[],
            retry_word_ids=[],
            original_word_ids=[],
            issues=[str(error)],
        )
    return result


def evaluate(context):
    entries = [recover(c, context.raw["segments"]) for c in context.candidates]
    # Conflicting independently decoded spans have no unambiguous word ownership.
    # Keep both originals rather than guessing which retry owns shared speech.
    usable = [e for e in entries if e["classification"] in STATES[:2]]
    conflicts = set()
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            if min(left["effective_interval"][1], right["effective_interval"][1]) > max(
                left["effective_interval"][0], right["effective_interval"][0]
            ):
                conflicts.update((left["review_id"], right["review_id"]))
    for entry in entries:
        if entry["review_id"] in conflicts:
            entry.update(
                classification="CORRUPT",
                replacement_segments=[],
                retry_word_ids=[],
                original_word_ids=[],
                issues=["expanded retry spans conflict"],
            )
    # An accepted expansion must not consume a fallback window's original evidence.
    while True:
        changed = False
        fallback = [e for e in entries if e["classification"] not in STATES[:2]]
        for entry in entries:
            if entry["classification"] in STATES[:2] and any(
                min(entry["effective_interval"][1], other["target_interval"][1])
                > max(entry["effective_interval"][0], other["target_interval"][0])
                for other in fallback
            ):
                entry.update(
                    classification="CORRUPT",
                    replacement_segments=[],
                    retry_word_ids=[],
                    original_word_ids=[],
                    issues=["expansion consumes fallback A target"],
                )
                changed = True
        collisions = boundary_duplicates(
            splice(context.raw["segments"], entries), entries, context.raw["segments"]
        )
        for entry in entries:
            if entry["review_id"] in collisions:
                entry.update(
                    classification="CORRUPT",
                    replacement_segments=[],
                    retry_word_ids=[],
                    original_word_ids=[],
                    issues=["new repeated boundary text has ambiguous word ownership"],
                )
                changed = True
        if not changed:
            break
    counts = Counter(e["classification"] for e in entries)
    return entries, {state: counts[state] for state in STATES}


def boundary_duplicates(segments, entries, originals):
    """Flag new identical boundary phrases, retaining repeated speech already in A."""
    lookup = {e["review_id"]: e for e in entries}
    collisions = set()

    def tokens(words):
        return [unicodedata.normalize("NFC", w["word"]).strip() for w in words]

    for left, right in zip(segments, segments[1:], strict=False):
        lo, ro = left["review_origin"], right["review_origin"]
        if lo["candidate"] == ro["candidate"] and lo.get("review_id") == ro.get("review_id"):
            continue
        a, b = left.get("words") or [], right.get("words") or []
        for length in range(1, min(8, len(a), len(b)) + 1):
            if tokens(a[-length:]) != tokens(b[:length]):
                continue
            for origin in (lo, ro):
                if origin["candidate"] != "B":
                    continue
                entry = lookup[origin["review_id"]]
                original_words = []
                for key in entry["original_word_ids"]:
                    si, wi = map(int, key.split(":"))
                    original_words.append(originals[si]["words"][wi])
                old_edge = original_words[-length:] if origin is lo else original_words[:length]
                new_edge = a[-length:] if origin is lo else b[:length]
                if tokens(old_edge) != tokens(new_edge):
                    collisions.add(origin["review_id"])
    return collisions


def splice(originals, entries):
    removed = {
        key for e in entries if e["classification"] in STATES[:2] for key in e["original_word_ids"]
    }
    segments = []
    retained = set()
    for si, s in enumerate(originals):
        words = s.get("words") or []
        selected = [wi for wi in range(len(words)) if f"{si}:{wi}" in removed]
        if not selected:
            segments.append(
                {
                    **copy.deepcopy(s),
                    "review_origin": {"candidate": "A", "canonical_segment_index": si},
                }
            )
            retained.update(f"{si}:{wi}" for wi in range(len(words)))
            continue
        run = []
        for wi in range(len(words) + 1):
            if wi < len(words) and f"{si}:{wi}" not in removed:
                run.append(wi)
                retained.add(f"{si}:{wi}")
            elif run:
                kept = [copy.deepcopy(words[n]) for n in run]
                segments.append(
                    {
                        **copy.deepcopy(s),
                        "words": kept,
                        "start": min(w["start"] for w in kept),
                        "end": max(w["end"] for w in kept),
                        "text": "".join(w["word"] for w in kept).strip(),
                        "review_origin": {
                            "candidate": "A",
                            "canonical_segment_index": si,
                            "canonical_word_indices": run,
                        },
                    }
                )
                run = []
    all_original = {
        f"{si}:{wi}" for si, s in enumerate(originals) for wi in range(len(s.get("words") or []))
    }
    require(
        retained.isdisjoint(removed) and retained | removed == all_original,
        "original word partition failed",
    )
    for entry in entries:
        if entry["classification"] not in STATES[:2]:
            continue
        for s in entry["replacement_segments"]:
            segments.append(
                {
                    **copy.deepcopy(s),
                    "review_origin": {
                        "candidate": "B",
                        "review_id": entry["review_id"],
                        "candidate_source": entry["classification"],
                        "effective_interval": entry["effective_interval"],
                    },
                }
            )
    segments.sort(key=lambda s: s["start"])
    # Detect crossing segment order caused by overlapping source word timelines.
    starts = [w["start"] for s in segments for w in s.get("words") or []]
    require(starts == sorted(starts), "reviewed word timeline is unordered")
    for entry in entries:
        if entry["classification"] not in STATES[:2]:
            continue
        span = entry["effective_interval"]
        for s in segments:
            if s["review_origin"]["candidate"] == "A":
                require(
                    not any(overlap(w, span) for w in s.get("words") or []),
                    "retained original overlaps replacement boundary",
                )
    return segments


def build_recovery(config, sid, fingerprint=None):
    context = load_context(config, sid, fingerprint, recovery=True)
    entries, counts = evaluate(context)
    # Preflight the complete splice before exposing candidates or saving decisions.
    splice(context.raw["segments"], entries)
    document = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "retry_context_recovery",
            "session_id": sid,
            "parent_checksums": context.parents,
            "boundary_policy": POLICY,
            "entries": entries,
            "counts": counts,
            "human_verified": False,
        }
    )
    path = context.directory / "context-recovery.json"
    atomic_write_json(path, document)
    for name in ("index.html", "app.js", "style.css"):
        (context.directory / name).write_bytes(
            files("session_pipeline").joinpath("review_ui", name).read_bytes()
        )
    return {"TOTAL": len(entries), **counts, "recovery_path": str(path)}


def load_recovery(context):
    document = read_artifact(context.directory / "context-recovery.json")
    entries, counts = evaluate(context)
    require(
        document["parent_checksums"] == context.parents
        and document["entries"] == entries
        and document["counts"] == counts
        and document["boundary_policy"] == POLICY,
        "retry recovery evidence is stale",
    )
    return document


def recovery_ui_state(context, bundle, decisions):
    """Overlay recovered evidence without rewriting historical human manifests."""
    if not (context.directory / "context-recovery.json").exists():
        return bundle, decisions
    recovery = load_recovery(context)
    bundle = copy.deepcopy(bundle)
    for entry, recovered in zip(bundle["entries"], recovery["entries"], strict=True):
        usable = recovered["classification"] in STATES[:2]
        entry.update(
            candidate_source=recovered["classification"],
            effective_interval=recovered["effective_interval"],
            text_b="\n".join(s["text"] for s in recovered["replacement_segments"]),
            b_selectable=usable,
            b_usable=usable,
            merge_status="AUTO_TRIMMED" if usable else "UNUSABLE",
            merge_issues=recovered["issues"],
        )
        entry["candidate_b"]["segments"] = recovered["replacement_segments"]
    policy_path = context.directory / "auto-prefer-b.decisions.json"
    if policy_path.exists():
        policy = read_artifact(policy_path)
        require(
            policy["recovery_checksum"] == recovery["artifact_checksum"],
            "automatic decisions are stale",
        )
        decisions = policy
    bundle["read_only"] = True
    bundle["review_mode"] = "AUTO_PREFER_B" if policy_path.exists() else "RECOVERY_PREFLIGHT"
    return bundle, decisions


def auto_merge(config, sid, fingerprint=None):
    context = load_context(config, sid, fingerprint, recovery=True)
    source = source_path(config, context)
    paths = [context.raw_path, context.raw_path.with_suffix(".txt"), source]
    hashes = {str(p): sha256_file(p) for p in paths}
    mtimes = {str(p): p.stat().st_mtime_ns for p in paths}
    with exclusive_lock(context.directory / ".decisions.lock"):
        recovery = load_recovery(context)
        entries = recovery["entries"]
        segments = splice(context.raw["segments"], entries)
        require(
            not boundary_duplicates(segments, entries, context.raw["segments"]),
            "introduced duplicate boundary text",
        )
        # Archive human manifests byte-for-byte; policy choices use a separate manifest.
        for name in ("bundle.json", "decisions.json"):
            original = context.directory / name
            if original.exists():
                data = original.read_bytes()
                backup = context.directory / "before-auto-prefer-b" / name
                if backup.exists():
                    require(
                        backup.read_bytes() == data, "human manifest changed since policy backup"
                    )
                else:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(data)
        decisions = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "automatic_review_decisions",
                "review_mode": "AUTO_PREFER_B",
                "human_verified": False,
                "session_id": sid,
                "recovery_checksum": recovery["artifact_checksum"],
                "decisions": {
                    e["review_id"]: {
                        "choice": "USE_B" if e["classification"] in STATES[:2] else "USE_A",
                        "candidate_source": e["classification"]
                        if e["classification"] in STATES[:2]
                        else "FALLBACK_A",
                    }
                    for e in entries
                },
            }
        )
        identity = {
            "review_mode": "AUTO_PREFER_B",
            "boundary_policy": POLICY,
            "human_verified": False,
            "parent_checksums": context.parents,
            "recovery_checksum": recovery["artifact_checksum"],
            "decision_manifest_checksum": decisions["artifact_checksum"],
            "counts": recovery["counts"],
            "raw_file_sha256": hashes,
        }
        fp = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        directory = context.raw_path.parent / "reviewed" / fp
        txt = directory / "transcript.reviewed.fa.txt"
        js = directory / "transcript.reviewed.fa.json"
        text = render_reviewed(segments, [])
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        original_limits = [
            min(s["start"] for s in context.raw["segments"]),
            max(s["end"] for s in context.raw["segments"]),
        ]
        require(
            min(s["start"] for s in segments) <= original_limits[0]
            and max(s["end"] for s in segments) >= original_limits[1],
            "session start or tail lost",
        )
        qa = {
            "status": "STRUCTURAL_PASS",
            "alignment_ready": True,
            "human_verified": False,
            "speech_accuracy_verified": False,
            "unresolved_count": 0,
            "original_word_partition": "PASS",
            "boundary_word_ownership": "PASS",
            "introduced_duplicate_boundary_text": "PASS",
            "timeline_order": "PASS",
            "session_extent_preserved": "PASS",
        }
        document = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "reviewed_asr",
                "status": "SUCCEEDED",
                "session_id": sid,
                "source_sha256": context.raw["source_sha256"],
                "source": context.raw["source"],
                "probe": context.raw["probe"],
                "model": context.raw["model"],
                "stage_fingerprint": fp,
                "created_at": utc_timestamp(),
                "provenance": {
                    **identity,
                    "spans": [{**e, **decisions["decisions"][e["review_id"]]} for e in entries],
                    "direct_b_count": recovery["counts"]["DIRECT_TARGET"],
                    "recovered_from_context_count": recovery["counts"]["RECOVERED_FROM_CONTEXT"],
                    "fallback_a_count": sum(e["classification"] not in STATES[:2] for e in entries),
                },
                "segments": segments,
                "segment_count": len(segments),
                "word_count": sum(len(s.get("words") or []) for s in segments),
                "unresolved": [],
                "rendered_txt_sha256": text_sha,
                "qa_acceptance": qa,
                "cloud_llm_required": False,
            }
        )
        require(
            load_context(config, sid, context.plan["stage_fingerprint"], recovery=True).parents
            == context.parents,
            "retry lineage changed",
        )
        require(
            all(
                sha256_file(Path(p)) == h and Path(p).stat().st_mtime_ns == mtimes[p]
                for p, h in hashes.items()
            ),
            "immutable inputs changed",
        )
        if js.exists():
            old = read_artifact(js)
            require(
                old["provenance"] == document["provenance"]
                and old["segments"] == segments
                and old["rendered_txt_sha256"] == text_sha,
                "reviewed lineage changed",
            )
            document = old
        if txt.exists():
            require(sha256_file(txt) == text_sha, "reviewed TXT changed")
        else:
            write_text_atomic(txt, text)
        if not js.exists():
            atomic_write_json(js, document)
        atomic_write_json(context.directory / "auto-prefer-b.decisions.json", decisions)
        atomic_write_json(
            context.raw_path.parent / "reviewed/latest.json",
            seal_artifact(
                {
                    "schema_version": 1,
                    "artifact_type": "reviewed_asr_pointer",
                    "session_id": sid,
                    "reviewed_json": js.relative_to(context.raw_path.parent).as_posix(),
                    "artifact_checksum_of_reviewed_json": document["artifact_checksum"],
                    "review_mode": "AUTO_PREFER_B",
                    "alignment_ready": True,
                    "qa_acceptance": "STRUCTURAL_PASS",
                    "human_verified": False,
                }
            ),
        )
    return {
        "reviewed_txt": str(txt),
        "reviewed_json": str(js),
        "counts": recovery["counts"],
        "USE_B": sum(d["choice"] == "USE_B" for d in decisions["decisions"].values()),
        "USE_A": sum(d["choice"] == "USE_A" for d in decisions["decisions"].values()),
        "qa_acceptance": qa,
        "raw_hashes": hashes,
        "raw_modified": False,
    }
