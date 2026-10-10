# Phase 4F: human-labelled final transcript

This phase renders only sealed local data. It requires the anonymous attribution,
its aligned/diarization parents, the verified identity-review bundle, and a persisted
`HUMAN_CONFIRMED` mapping. Exactly one speaker must map to علی and one to مشاور.
The persisted mapping is authoritative; the command has no identity override.
Missing, stale, unconfirmed, duplicate, or corrupt mappings refuse rendering.
No ASR, alignment, diarization, ML runtime, LLM, historical transcript, or network
is used. Rendering does not change any upstream artifact.

From `E:\personal\transcripts`:

```powershell
uv run --no-sync session-pipeline final render s-d57224782d66f69a7446 --attributed-fingerprint 0f88ee692d361c2561f645084f2f00e0cb50e9ef68f0176e06d560f8d4450b18 --json
uv run --no-sync session-pipeline final status s-d57224782d66f69a7446 --json
```

Status also accepts `--fingerprint <final-fingerprint>` when multiple final lineages
exist. It is read-only: it validates the sealed JSON, recreates expected deterministic
rendering in memory, compares the TXT byte-for-byte, and rechecks input hashes.
The status of a historical final lineage refuses if the current human mapping has
changed; it does not silently reinterpret historical output.

## Exact text and mixed/partial content

The renderer first verifies each word's original character offsets: ordered,
nonoverlapping integer bounds and exact equality between the source substring
and the original word. It never reconstructs segment text by joining normalized
tokens. Complete contiguous source character slices include original whitespace,
punctuation, spelling, Persian characters, and newlines without normalization.

When the proof succeeds, mixed and partial segments split into contiguous anonymous
speaker runs and use the persisted human names. Unassigned words retain نامشخص.
Word timings remain original aligned bounds. Failed-alignment words retain their
shared reviewed-segment fallback interval, explicitly marked as such; contiguous
unknown fallback words can retain that one interval as a single unknown run.

When character provenance is missing/incorrect, or valid word timing order cannot
support splitting, the renderer retains the exact whole parent segment. Its original
word attribution and MIXED/PARTIAL/UNASSIGNED status remain in JSON. The TXT label
lists the represented identities and includes نامشخص for unresolved content. It
never substitutes majority assignment or invents word placement. Invalid time bounds
fail rather than being repaired or clamped.

Session 4 supports exact character slicing for all 2,810 segments, including all
164 mixed segments. No semantic inference, rewriting, punctuation addition, grammar
correction, or nearest-speaker fallback occurs.

## Conservative turn policy

The versioned configuration records a maximum grouping gap of **0.75 seconds**.
Adjacent resolved runs join only with the same anonymous/human speaker, ordered
nonoverlapping bounds, and no different exclusive speaker or regular overlap region
across their extents. Cross-segment joins require both source segments to be
`SINGLE_SPEAKER`. Mixed, partial, unresolved and whole-parent fallback blocks cannot
be crossed. Original source order is retained, including when speakers alternate
inside a segment. Unknown runs never join across segments.

TXT uses `[HH:MM:SS.mmm — HH:MM:SS.mmm] <human label>:` and UTF-8. Millisecond
presentation preserves short durations; JSON retains the original numeric timings.
Newlines between source segments and turns are presentation structure only. The
timestamp/header strings are metadata and do not become transcript wording.

## Immutable final lineage and preservation proof

```text
outputs/<session>/final/<fingerprint>/transcript.final.fa.json
outputs/<session>/final/<fingerprint>/transcript.final.fa.txt
```

The fingerprint binds renderer version/policy, pipeline version, source hash,
attribution checksum, aligned/diarization checksums, review-bundle checksum and the
complete persisted mapping checksum. Mapping revisions therefore create new lineages.
JSON preserves all original segments and word attribution, adds identity metadata,
and records exact rendering pieces, human confirmation, parent paths/hashes, QA,
TXT checksum, and source-to-TXT character **and UTF-8 byte** offsets.

Validation requires each source segment ID to be represented; each source character
interval and word index to form an ordered, exact, disjoint partition; reconstructed
text to equal the source string for every segment; and source/reconstructed text
hashes to match. TXT offsets recover those same source slices byte-for-byte, while
a complete deterministic TXT comparison also detects unexpected presentation text.
Original segment/word metadata is compared field-for-field after identity enrichment.
All unknown words must remain unknown. Rendered bounds must be positive, ordered
and within the source duration. QA lists each unresolved segment ID, word index,
aligned/reviewed bounds and timing provenance.

Publication uses the existing atomic UTF-8 text writer and sealed JSON writer under
a process lock. Input checksums and file hashes are checked immediately before and
after TXT publication. JSON is the completion marker for the pair. An interrupted
TXT-only publication can resume if its bytes equal expected output. Existing changed
files refuse reuse and are never overwritten. Identical reruns reuse both files.

Warnings preserve the known unresolved attribution and overlapping speech. Structural
PASS does not establish the linguistic accuracy of upstream ML output or human
listening acceptance of every word. Phase 5 productization is outside this phase.
