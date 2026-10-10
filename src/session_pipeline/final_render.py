"""Future renderer foundation. Pure turn planning only; never writes final files."""

from __future__ import annotations

from session_pipeline.attribution import require
from session_pipeline.speaker_review import validate_mapping


def plan_turns(document, human_mapping):
    """Preserve exact character slices and word provenance, gated by human mapping.

    Mixed segments split only at original word character boundaries. Unknown words
    keep a neutral label. Group only adjacent same-speaker pieces when the exclusive
    timeline contains no intervening different speaker. Text pieces remain separate
    inside a turn, so no guessed punctuation or sentence rewriting is introduced.
    """
    require(
        human_mapping is not None
        and human_mapping.get("human_confirmed") is True
        and human_mapping.get("identity_inference_used") is False
        and human_mapping.get("attribution_checksum") == document["artifact_checksum"],
        "current human identity mapping required",
    )
    validate_mapping(human_mapping["mapping"])
    pieces = []
    for s in document["segments"]:
        words = s["words"]
        text = s["text"]
        if not words:
            pieces.append(
                {
                    "speaker": None,
                    "text": text,
                    "character_interval": [0, len(text)],
                    "segment_id": s["reviewed_segment_id"],
                    "word_indices": [],
                    "start": s["start"],
                    "end": s["end"],
                }
            )
            continue
        starts = [w["character_interval"][0] for w in words]
        require(
            starts == sorted(starts) and len(set(starts)) == len(starts),
            "cannot split unordered character intervals",
        )
        for i, w in enumerate(words):
            a, b = w["character_interval"]
            require(
                0 <= a < b <= len(text) and text[a:b] == w["word"],
                "word character provenance mismatch",
            )
            left, right = (0 if i == 0 else a), (starts[i + 1] if i + 1 < len(words) else len(text))
            interval = w["attribution"]["attribution_interval"] or s["reviewed_interval"]
            pieces.append(
                {
                    "speaker": w["attribution"]["speaker"],
                    "text": text[left:right],
                    "character_interval": [left, right],
                    "segment_id": s["reviewed_segment_id"],
                    "word_indices": [i],
                    "start": interval[0],
                    "end": interval[1],
                }
            )
    turns = []
    exclusive = document["diarization_evidence"].get("exclusive_turns")
    timeline = exclusive or document["diarization_evidence"]["regular_turns"]
    for piece in pieces:
        speaker = piece["speaker"]
        # Include both pieces' extents, not just an inter-segment gap: an aligned
        # word may cross a detected speaker change despite winning max overlap.
        change = bool(turns) and any(
            t["speaker"] != speaker
            and t["end"] > turns[-1]["pieces"][-1]["start"]
            and t["start"] < piece["end"]
            for t in timeline
        )
        if (
            turns
            and speaker is not None
            and turns[-1]["speaker"] == speaker
            and not change
            and piece["start"] >= turns[-1]["pieces"][-1]["start"]
        ):
            turns[-1]["pieces"].append(piece)
            turns[-1]["end"] = max(turns[-1]["end"], piece["end"])
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "label": human_mapping["mapping"].get(speaker, "نامشخص"),
                    "start": piece["start"],
                    "end": piece["end"],
                    "pieces": [piece],
                }
            )
    return turns
