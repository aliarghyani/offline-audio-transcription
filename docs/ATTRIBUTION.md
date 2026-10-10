# Phase 4E: deterministic anonymous attribution and human identity review

This stage consumes sealed canonical aligned and diarization artifacts. It never
imports an ML runtime, loads a model, calls an LLM, reads historical transcripts,
or changes an upstream file. Run from `E:\personal\transcripts` with the existing
environment. `--no-sync` avoids dependency provisioning; no network is required.

```powershell
uv run --no-sync session-pipeline attribute s-d57224782d66f69a7446 --execute --aligned-fingerprint 4ba81b10c2e0daff7708ce89c7bcb0b504c8935702d48ee2a4337063554e96f9 --diarization-fingerprint 41031dbf581439b2afb2c842096a129c0b652e360bab710a533cc16dc8ab11eb --json
$attributionFingerprint = "0f88ee692d361c2561f645084f2f00e0cb50e9ef68f0176e06d560f8d4450b18"
uv run --no-sync session-pipeline speakers review build s-d57224782d66f69a7446 --fingerprint $attributionFingerprint --json
uv run --no-sync session-pipeline speakers review serve s-d57224782d66f69a7446 --fingerprint $attributionFingerprint --port 0 --open
```

The server prints its loopback address and opens the browser. Keep the terminal
running; Ctrl+C stops it. The default fixed port is 8766. `--port 0` chooses a free
port. The review page uses only local assets and audio, with no transcript text,
analytics, external fonts, or identity defaults. Choose one علی and one مشاور,
then explicitly confirm that you identified both voices by listening.

Read-only commands:

```powershell
uv run --no-sync session-pipeline attribute s-d57224782d66f69a7446 --status --fingerprint $attributionFingerprint --json
uv run --no-sync session-pipeline speakers status s-d57224782d66f69a7446 --fingerprint $attributionFingerprint --json
```

`attribute` without `--execute` computes a plan/report without publishing anything.
Status and speaker-review commands accept `--fingerprint` to pin an existing
attribution. If multiple lineages exist, select explicitly; no newest-lineage
guessing occurs. Execution accepts separate aligned/diarization fingerprints.

## Attribution contract

Both parents must have valid artifact checksums, expected types, matching session,
source and path fingerprints, and completed full processing. The diarization parent
must bind the aligned checksum. Its sealed plan must bind the aligned file SHA-256
and agree with the canonical diarization plan checksum. Regular turns are validated
without flattening overlaps; exclusive turns must also be nonoverlapping.

For each positive-duration aligned word, select the speaker of the exclusive turn
with the largest positive intersection. Multiple turns for the same speaker use
their **maximum individual turn overlap**, not a sum across turns. Equal maxima
from different speakers leave the word ambiguous and unassigned. A positive
exclusive tie never triggers a regular-turn override. If no exclusive turn overlaps,
apply the same rule to regular turns; if neither representation overlaps, leave
the word unassigned. No nearest-speaker assignment is applied.

Word attribution records the anonymous label, status, method, maximum overlap
duration/fraction, per-speaker maximum intersections, original interval, timing
source, regular-turn intersections and overlap-region metadata. Fractions measure
temporal intersection only; they are not speaker-confidence probabilities. Original
alignment scores remain unchanged and retain their upstream meaning.

Only untimed words in `ALIGNMENT_FAILED` segments use the original reviewed
segment interval. They retain null aligned timestamps and explicitly record
`REVIEWED_SEGMENT_FALLBACK`. Assignment requires exactly one supported speaker
and no regular overlap region; multiple supported speakers leave them unassigned.
Other invalid/zero-duration word bounds receive no guessed interval.

Segment classification uses all word assignments:

- `MIXED_SPEAKER`: both anonymous speakers have assigned words, including when
  some other words are unassigned. `is_partial` records that additional condition.
- `SINGLE_SPEAKER`: all words are assigned to the same speaker.
- `PARTIAL`: one speaker is represented and some words remain unassigned.
- `UNASSIGNED`: no words are assigned, including a segment with no words.

No majority rule flattens mixed segments. Every original segment ID, text, timing,
word and field is retained, with metadata added under `attribution`.

The old uncovered-reviewed-bound IDs are reclassified by word assignment into
fully attributable, partially attributable and unassigned. This measures whether
every available word has a deterministic attribution, not full diarization coverage
of every millisecond in the reviewed outer bounds. Unassigned word/segment diagnostics
record neighboring exclusive turns, distances, neighbor labels/equality, total gap,
regular intersections and known overlaps. Threshold buckets are cumulative and
diagnostic only. Missing neighbors stay null. No fallback is applied.
When an ambiguous item intersects exclusive speech, gap duration stays null;
the span between surrounding turns is recorded separately as a neighbor window,
so overlap ambiguity is never misreported as an uncovered speech gap.

## Artifact and review storage

```text
outputs/<session>/attributed/<fingerprint>/transcript.attributed.fa.json
work/<session>/speakers/review/<attribution-fingerprint>/
  bundle.json
  index.html
  app.js
  style.css
  SPEAKER_00-1.wav ... SPEAKER_01-3.wav
work/<session>/speakers/identity/<attribution-fingerprint>/mapping.json
```

The new fingerprint binds both parent artifact checksums and the versioned policy.
Provenance additionally preserves parent fingerprints, file hashes and paths.
Both complete turn representations are copied into the anonymous artifact; regular
overlap regions and unresolved diagnostics remain available. Publishing rechecks
parent file hashes. Repeated identical execution validates/reuses the same artifact.
Status rebuilds expected deterministic metadata to detect stale or altered inputs.

Review clips are ignored local mono 16 kHz PCM WAVs decoded with the existing
FFmpeg primitive. Selection uses exclusive turns intersected with matching regular
speaker coverage, subtracts all regular overlap and turns shorter than 0.1 seconds,
and trims a 0.15-second boundary margin. Three nonoverlapping 6–12-second samples
per speaker target the centers of the three session thirds; when safe speech is
unavailable near a third, the nearest eligible interval is used deterministically.
If six safe clips cannot be selected, build fails instead of fabricating samples.
The manifest records exact intervals, exclusive turn indices, policy and clip hashes.
The source hash is verified before and after decoding. Existing changed clips fail
closed. Playback verifies bundle lineage and clip hashes and supports byte ranges.

The human mapping is a separate atomically written sealed artifact with confirmation
time, revision, attribution/bundle bindings and previous checksum. Past confirmations
are archived when a human changes a mapping. Concurrent stale saves fail with HTTP
409. The server binds only `127.0.0.1`, accepts its exact Host, and requires same-origin,
token-bearing JSON writes. Mapping does not modify diarization or attribution and
does not trigger final rendering.

## Renderer foundation and remaining human work

`final_render.plan_turns` is a pure function gated by a matching explicitly confirmed
human mapping. It preserves character slices from each original segment and word
indices, splits mixed segments by those original boundaries, labels unresolved words
`نامشخص`, and groups adjacent same-speaker pieces only when no different exclusive
speaker occurs across their extents. Each turn retains separate text pieces, so it
cannot invent punctuation, reorder words, or rewrite Persian. Times remain metadata.
There is no final-output CLI or file publisher in this phase. A later renderer can
write `transcript.final.fa.json` and TXT using `[HH:MM:SS — HH:MM:SS] <human label>:`
after the human mapping and rendering policy are accepted.

Phase 4F now implements that later step; see `FINAL_TRANSCRIPT.md` for its persisted
mapping gate, exact-text proof, conservative grouping policy and final-render CLI.

Metadata QA and synthetic tests do not establish actual voice identity or the
linguistic correctness of earlier ML output. The remaining action is human listening
and explicit mapping. Session 4 must remain unmapped during Phase 4E implementation.
