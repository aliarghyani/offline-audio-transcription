# Phase 4B: local human A/B review and safe merge

The human workflow below describes Phase 4B/4B.1. The explicitly authorized
Phase 4B.2 automatic workflow at the end supersedes its boundary and readiness
rules when `AUTO_PREFER_B` is selected.

Review reads the existing Phase 4A plan, progress and sealed candidate checkpoints.
It does not run ASR, alignment, diarization, text generation or cloud services.
Historical reference transcripts are not inputs. The server uses only Python's
standard library, binds to `127.0.0.1`, and serves local assets without analytics.

## Start reviewing

In PowerShell from `E:\personal\transcripts`, using the existing environment:

```powershell
uv run --no-sync session-pipeline qa review build s-d57224782d66f69a7446 --json
uv run --no-sync session-pipeline qa review serve s-d57224782d66f69a7446 --port 0 --open
```

`--port 0` lets Windows allocate a free port; `--open` opens that local URL in the
default browser. The terminal prints the address. Keep the terminal running;
Ctrl+C stops the server. Repeating the command restores saved decisions even if
the allocated port changes. The browser's navigation position may change with
the origin, but the decision file is authoritative and independent of the port.
You may specify a fixed available port instead; the default is 8765.

Other commands:

```powershell
uv run --no-sync session-pipeline qa review status <session-id> --json
uv run --no-sync session-pipeline qa merge <session-id> --json
```

Every review/merge command accepts `--retry-fingerprint <64-character hash>`.
Without it, exactly one completed retry plan must exist. Multiple completed
plans require explicit selection; the pipeline does not guess the newest plan.
Incomplete, stale or tampered lineages fail closed.

## Existing artifact validation

Validation checks the canonical source identity and checksum, quality review,
integrity parent, plan fingerprint/configuration/model, completed progress, every
candidate's fingerprint/parents/window and canonical A segment equality. B's
owned segments are recomputed using Phase 4A's existing midpoint ownership rule
and must match the saved artifact. Absolute segment/word timing is checked.
Target intervals must be ordered and nonoverlapping; padded context may overlap.
These checks establish artifact identity, not the linguistic accuracy of B.

A successful retry checkpoint can contain **empty target B text**. This is shown
explicitly; it never authorizes deletion of A. The UI distinguishes this case
from missing/corrupt checkpoints and keeps listening and A/UNCLEAR choices available.

## Bundle and playback

The ignored bundle lives at:

```text
work/<session-id>/qa/human-review/<retry-plan-fingerprint>/
  bundle.json
  decisions.json
  index.html
  app.js
  style.css
```

The bundle records source/parent identities, per-window flags, target and decoded
intervals, A/B data, diagnostics, merge blockers and audio checksums. It does not
rank or auto-select candidates. The UI shows absolute timestamps, original A,
owned target B, separate padded B context, diagnostic fields and audio controls.
Candidate text is rendered as text, with Persian RTL and explicit LTR diagnostics.

Existing derived `retry-<index>.wav` clips are reused after metadata validation.
If a clip is missing, build hashes the source and uses FFmpeg to create only that
small padded derived clip. An invalid existing clip is preserved for investigation;
it is not silently overwritten. Source hash/timestamp are checked. Clips and the
entire bundle remain ignored by Git. Playback supports byte ranges for seeking;
audio checksums are verified before serving.

"Play target interval" seeks past the leading context and pauses at the target
end. Standard audio controls allow listening to all context. Timing controls are
for listening assistance, not forced alignment or proof of word boundaries.

## Persisted human decisions

Choices are `USE_A`, `USE_B`, `UNCLEAR`, or `UNREVIEWED`. Every new entry starts
`UNREVIEWED`. Shortcuts: A/B/U select; N/P navigate. Reset returns a decision to
`UNREVIEWED`. No decision triggers an automatic merge.

Each save atomically updates a sealed local `decisions.json`, recording choice,
timestamp, revision, bundle checksum and source/review identity. A process lock
and expected manifest checksum prevent one tab silently overwriting another.
Conflicting saves reload decisions and require choosing again. Rebuilding and
restarting preserve valid decisions; changed candidates/parents invalidate the
old bundle rather than silently carrying approvals onto new text.

The HTTP server accepts only its own loopback Host and same-origin, token-bearing
JSON decision requests. It serves only whitelisted assets and clip IDs; it does
not expose a general file browser. There is no CORS, external asset request,
telemetry, transcript logging or cloud model dependency.

## Human choice and merge eligibility (Phase 4B.1)

Human candidate quality and merge safety are separate. Any usable target B stays
selectable, including when timestamps require manual boundary resolution.
UNCLEAR means the reviewer is unsure about spoken content; software boundary
uncertainty never changes a USE_B decision into UNCLEAR or USE_A.

Each entry and saved decision carries a separate `merge_status`:

- `SAFE`: target data already fits cleanly; no context trimming is needed.
- `AUTO_TRIMMED`: context lies outside the target, but clean word timestamps
  isolate the target. Only fully contained words with midpoint ownership in the
  half-open target are eligible for insertion. Word IDs record this deterministic
  selection. Context is not inserted.
- `MANUAL_BOUNDARY_REQUIRED`: usable B exists, but an original/retry boundary
  crosses speech, word timing overlaps, or relevant word timing is ambiguous.
  The reviewer can save USE_B. Final merge refuses and lists the specific USE_B
  windows/IDs requiring resolution. No silent fallback to A occurs.
- `UNUSABLE`: no usable B target text. B selection remains disabled/rejected.
  Missing, corrupt or invalid-timestamp checkpoint contracts fail closed during
  validation rather than permitting an invalid review or merge.

Manual first/last-word controls are not implemented in this phase. USE_B choices
requiring such resolution persist for a later step. The warning makes this
limitation explicit while allowing the reviewer to judge candidate fidelity.

## Conservative automatic replacement boundaries

- `USE_A` retains canonical original text and timing unchanged.
- `USE_B` inserts only the existing **owned target B segments**, never padded
  `raw_segments`. Every original segment/word being replaced must fit wholly
  within the target, and retry words intersecting the target must not cross its
  boundary. A retry segment spanning context can be split only where its timed
  words are unambiguous. Without word timing, a crossing segment is blocked.
- Empty target B disables human selection. Original/retry crossing boundaries
  instead mark `MANUAL_BOUNDARY_REQUIRED`, with a visible explanation. Such cases
  need later manual resolution to apply an otherwise valid human USE_B choice.
- `UNCLEAR` preserves A and records an explicit unresolved span in JSON and an
  `[UNRESOLVED ...]` marker in the reviewed TXT. No words are guessed.
- `UNREVIEWED` causes merge to refuse before writing any output.

The numeric epsilon is only `1e-6` seconds for floating-point arithmetic. It is
not a tolerance for crossing speech. Overlapping targets and duplicate canonical
replacement regions fail closed. Diagnostics never override a human decision.
Safety validation can refuse final merge; it never substitute-selects another candidate.

## Decision schema upgrade

Build upgrades review/decision metadata from version 1 to version 2 without
changing human choices, their timestamps, revision or last human-save timestamp.
The unchanged old sealed documents remain in `migration-v1/`. A sealed
`migration-v2.json` journal lets interrupted pair writes resume. Newly saved human
choices after a completed pair write are retained if the completion marker was
interrupted. Migration does not infer decisions. Rebuilds are idempotent; stale
source/candidate identities cannot silently inherit old approvals.

## Reviewed lineage and QA acceptance

After all entries have an explicit choice, `qa merge` creates an immutable
decision-specific lineage:

```text
outputs/<session-id>/reviewed/<merge-fingerprint>/transcript.reviewed.fa.json
outputs/<session-id>/reviewed/<merge-fingerprint>/transcript.reviewed.fa.txt
outputs/<session-id>/reviewed/latest.json
```

Raw `transcript.raw.fa.json` and `.txt` are never write targets. Provenance includes
raw checksum/path, source audio hash, retry plan fingerprint, bundle and decision
checksums, all candidate parents, A/B/unclear spans, and merge version. Retained
segments record canonical indices; B segments record their reviewed window.
Output TXT uses UTF-8, NFC only, and explicit newlines; JSON retains original
strings. There is no grammar correction, character substitution or semantic rewrite.

TXT is written atomically before JSON; JSON records its SHA-256. `latest.json`
is published only after the pair is complete. Repeating the same merge validates
and reuses the same files. Changed decisions create another lineage and preserve
previous reviewed versions. Raw/source and lineage parents are rechecked before
publishing. QA acceptance remains `PENDING` and `alignment_ready=false`, even when
all choices are clear. A future explicit QA acceptance step is required before
alignment consumes the reviewed artifact.

## Phase boundary

Phase 4B builds the Session 4 bundle only. It makes no Session 4 decisions and
does not merge Session 4. Tests use synthetic metadata/audio and injected retry
outputs, including UI decision persistence. Actual retry correctness still
requires human listening; metadata validation cannot establish spoken meaning.

## Phase 4B.2: retry context recovery and AUTO_PREFER_B

```powershell
# Reconstruct all candidates and preflight the merge; does not make decisions.
uv run --no-sync session-pipeline qa recover <session-id> --json
# Explicit automatic policy: reconstruct, choose B where safe, and merge.
uv run --no-sync session-pipeline qa recover <session-id> --auto-prefer-b --json
```

Recovery reads every full retry word timeline. Positive interval overlap selects
the initial words; midpoint ownership and empty saved B are not evidence that
no retry exists. Effective boundaries expand to whole overlapping retry words
and original words until the common interval stabilizes. Additional retry words
are included only if they overlap that effective interval. Expansion outside the
decoded evidence, conflicting retry spans, consumption of fallback A targets,
unreconstructable original text and newly repeated boundary text block replacement.
Original partial segments are split at word boundaries; all retained original
word IDs and all removed word IDs form an exact disjoint partition.

`DIRECT_TARGET` means old owned B was nonempty and whole-word reconciliation
succeeded. `RECOVERED_FROM_CONTEXT` means old owned B was empty and reconstruction
succeeded. `NO_TARGET_OVERLAP` requires no usable positive-overlap retry word.
`CORRUPT` also represents genuinely impossible deterministic boundary reconciliation;
its recorded issues distinguish that from invalid artifacts or timestamps. Corrupt
checkpoints are preserved, with their file hashes recorded as fallback evidence.
Plan, source and canonical integrity errors still fail the whole operation.

Recovery evidence is saved in `context-recovery.json`; policy decisions live in
`auto-prefer-b.decisions.json`. Historical human manifests are preserved byte for
byte and backed up under `before-auto-prefer-b/`. The UI overlays reconstructed B,
shows “Recovered from retry context”, and presents the policy in a read-only view.
Padded context stays separate; playback uses the effective whole-word interval.

Automatic decisions choose B for the two usable states and A for the other two;
they never become human verification. The new reviewed lineage records
`review_mode=AUTO_PREFER_B`,
`boundary_policy=WORD_INTERVAL_OVERLAP_EXPAND_TO_FULL_WORD`, counts, decisions,
effective spans, original/retry word IDs, source/raw hashes and all retry parents.
Publishing checks word ordering, preserved session extent, original word partition,
nonoverlapping boundary ownership and newly duplicated boundary text. Identical
speech already present at an original boundary is preserved, not deduplicated.

`qa_acceptance.status=STRUCTURAL_PASS` and `alignment_ready=true` indicate a
structurally valid alignment input, not verified speech accuracy. Neither human
listening nor semantic completeness is established. Raw files remain unchanged.
No ASR, alignment, diarization, network call or language model is run.
