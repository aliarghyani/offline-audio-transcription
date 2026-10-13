# Phase 4C: local Persian forced alignment

This stage consumes a structurally accepted **reviewed** JSON/TXT pair. Raw ASR
and historical reference transcripts are never alignment text inputs. Every
reviewed string remains authoritative, including punctuation and Persian ZWNJ.
Alignment supplies timing; it does not correct, regenerate or rewrite text.

## Commands and explicit execution

Run commands directly in PowerShell from the repository. No Codex session or
cloud service is needed. Use the installed environment without installing or syncing:

```powershell
Set-Location -LiteralPath 'E:\personal\transcripts'
$session = 's-d57224782d66f69a7446'
$reviewed = '0a3e1f52e4c724fee3162e4c254447b93a1ea686d361ce13d0d0f3428433966b'

# Metadata plan only; this is also the default without an action flag.
uv run --no-sync session-pipeline align $session --plan --reviewed-fingerprint $reviewed --json

# Manual real smoke, at most the first 180 seconds, separate artifacts.
uv run --no-sync session-pipeline align $session --execute --smoke --reviewed-fingerprint $reviewed --json

# Manual full execution. Rerun exactly this command to resume.
uv run --no-sync session-pipeline align $session --execute --reviewed-fingerprint $reviewed --json

# Finalize existing results only; no model, decoding, or retries.
uv run --no-sync session-pipeline align $session --finalize --fingerprint <alignment-fingerprint> --json

# Read-only full-run status. Add --smoke for smoke status.
uv run --no-sync session-pipeline align $session --status --reviewed-fingerprint $reviewed --json
```

`--plan`, `--execute`, `--status`, and `--finalize` are mutually exclusive. Only `--execute`
loads the real model or decodes derived audio. An omitted reviewed fingerprint
selects the validated `reviewed/latest.json` pointer; the commands above pin the
accepted lineage. Status can select an exact existing plan using
`--status --fingerprint <alignment-fingerprint>`. The same exact fingerprint can
be used with `--finalize`; finalization never creates a plan or runs inference.
Progress is written to stderr; `--json` stdout stays machine-readable. Failed or
incomplete execution returns exit code 3. Process exit alone is never completion.

## Local model and resource lifecycle

The sole alignment model is:

- Repository: `jonatasgrosman/wav2vec2-large-xlsr-53-persian`
- Revision: `234714078a1398a9db88194c5a40fefe6f376dc1`
- Local directory: `models/alignment/234714078a1398a9db88194c5a40fefe6f376dc1`

The runner fully verifies pinned local files before loading. It calls WhisperX
3.8.6 `load_align_model` with the absolute local directory, `language_code=fa`,
`device=cpu`, and `model_cache_only=True`. It never resolves a remote `main` or
loads the large-v3 ASR model. Offline environment variables, removed Hub tokens,
disabled telemetry, and a socket guard apply throughout execution. There is no
network fallback. Missing/corrupt models remain failures requiring investigation;
this command does not provision anything.

One nonblocking OS lock per session excludes simultaneous full/smoke runners.
One alignment model is reused for all pending units, with four CPU threads and
sequential per-segment alignment. Completed-only resumes load no model. Units
hold roughly five minutes of mono 16 kHz PCM; inference/trellis allocation is
bounded by one reviewed segment. Segments longer than 60 seconds block execution
instead of risking an uncontrolled allocation. Session 4's maximum is below this.
The model and ephemeral derived PCM are released at completion/interruption.
The original recording is never a write target.

WhisperX normally uses an NLTK sentence tokenizer with a download fallback. The
adapter treats each reviewed segment as one sentence and patches the process-local
tokenizer lookup, rejecting NLTK downloads. This preserves segment identity and
avoids requiring a new dataset. Timestamp interpolation is `ignore`; absent
alignment remains explicitly absent. WhisperX transcript-bearing warning logs
are suppressed. Failure diagnostics contain exception types, never exception text.

## Deterministic ownership and resume

Full work units use the established 300-second session grid. A reviewed segment
belongs to exactly the half-open unit containing its **start**, including a
possible zero-duration segment at the session endpoint in the final unit.
No segment is cut or assigned twice. A unit's decoded interval extends through
the end of all its owned segments; decoded clips may overlap, segment ownership
does not. Empty units still have known status and timeline ownership.

Each unit ID hashes the reviewed checksum, indices, stable segment IDs, ownership,
and decode intervals. Each reviewed segment ID hashes its parent checksum and
its original ordered index. Plan fingerprints also include pipeline version,
alignment policy, exact model lock, integrity parent and package versions.

```text
work/<session>/alignment/<fingerprint>/plan.json
work/<session>/alignment/<fingerprint>/units/<unit-id>.json
work/<session>/alignment/<fingerprint>/progress.json
outputs/<session>/aligned/<fingerprint>/transcript.aligned.fa.json
outputs/<session>/aligned/smoke/<fingerprint>/transcript.aligned.fa.json
```

Checkpoints are atomically sealed and bind the reviewed, plan, model-lock and unit
checksums. Valid `SUCCEEDED` and recoverable `PARTIAL` units are skipped. `FAILED` and
`INTERRUPTED` units are retried only by explicit `--execute`. A caught
KeyboardInterrupt/SystemExit seals an `INTERRUPTED` checkpoint. Read-only status
reports stale `RUNNING` as `INTERRUPTED` when the OS worker lock is unheld, and
keeps a live worker `RUNNING`. Tampered or stale checkpoints are refused, not silently reused.
Attempts, cumulative unit wall time, processed audio seconds, statuses and QA
diagnostics are recorded. Status derives actual completion from checkpoint files,
not the potentially stale progress snapshot.

Smoke is a separate fingerprint and output namespace. It owns whole reviewed
segments that fit entirely within the first 180 seconds; a segment crossing the
smoke endpoint is excluded. Smoke completion never marks the full session complete
and cannot substitute for a full-run checkpoint.

## Authoritative text and QA

The adapter aligns one reviewed segment at a time and verifies an exact ordered
mapping to whitespace-delimited tokens of its unchanged text. It records character
spans and word indices. Unmappable output is an alignment failure, not permission
to replace wording. Word timestamps are converted from clip-relative to absolute
session time; reviewed segment timestamps remain unchanged.

QA tags:

- `ALIGNED`: the segment's authoritative tokens have reported timings.
- `UNALIGNED_WORD`: a token has no usable timing; start/end/score are null.
- `PARTIAL_ALIGNMENT`: only some segment tokens have timings.
- `ALIGNMENT_FAILED`: failed engine/lexical mapping or no timed tokens.
- `OUT_OF_BOUNDS`: invalid timing or timing outside the source/unit interval.
- `TIMESTAMP_ORDER_ERROR`: reversed timing or decreasing ordered word starts.
- `SUSPICIOUS_WORD_DURATION`: under 0.02 or over 3 seconds, an inspection flag.

WhisperX scores are stored as reported. They are CTC alignment scores, **not**
calibrated confidence in transcript wording. Duration thresholds are operational
flags, not scientific judgments. No nearest-word interpolation is applied.
Failed segments preserve all authoritative tokens as untimed in checkpoints.
Partial alignment can complete processing with explicit untimed words; fatal
failures or invalid bounds/order keep the unit failed and block canonical output.

The final validator requires exact segment partition, every unit `SUCCEEDED` or `PARTIAL`,
source-duration bounds, nondecreasing aligned word starts within/across units,
and beginning/tail ownership. Any global coverage/order failure blocks publishing.
The canonical output records reviewed parent checksum, source SHA-256, model and
exact revision, configuration/fingerprint, segment IDs, reviewed text/timestamps,
aligned words with score/status, unit IDs and all checkpoint checksums.

Full completion means all owned material was processed and structurally validated.
It does not mean every word aligned or that automatic reviewed wording was human
verified. Consult `unaligned_word_count` and QA flags before later attribution or
diarization. Raw/reviewed pairs are hashed and rechecked before publication, and
their files remain untouched. Identical completed reruns preserve the aligned file.

## Original Phase 4C development boundary

Phase 4C tests use generated synthetic audio, fake model interfaces, and injected
alignment results. They test the adapter call contract without loading real
weights. Session 4 receives a **plan only**. Real model/runtime accuracy, memory
and wall time must be established by the user's manual smoke. No real alignment,
ASR, diarization, LLM invocation, package install or model download is authorized
inside the development task.


## Phase 4C.1: completion with recoverable alignment warnings

`SUCCEEDED` means every owned segment and word was aligned acceptably. `PARTIAL`
means execution completed and every owned segment is represented, with recoverable
content alignment failures or missing word timings. `FAILED` remains an execution,
model, infrastructure, invalid timestamp, or coverage failure. WhisperX execution
exceptions propagate to the unit failure handler; they are not content warnings.
Suspicious durations remain visible QA warnings even when timing is available.

`completed_units` includes `SUCCEEDED` and `PARTIAL`; `succeeded_units`,
`partial_units`, `failed_units`, and `unit_status_counts` expose the distinction.
`mode_processing_complete`, `full_alignment_complete`, and
`coverage_validation_pass` require complete deterministic coverage and valid
structure, not 100% forced-aligned words. `alignment_has_warnings` explicitly
indicates non-ALIGNED QA. Canonical QA retains all existing counts and the exact
unaligned word count. Completion describes processing, not scientific acceptance.

For failed segments, unchanged reviewed text, segment ID, reviewed start/end,
review origin, and parent lineage are retained. `status` and `alignment_status`
are `ALIGNMENT_FAILED`, with machine-readable `alignment_diagnostics.failure_reason`.
Unaligned words retain text and have null start/end/score; segment bounds are
reviewed fallback timing, not invented forced-alignment word timing. Later
speaker processing may use aligned word timestamps where available and reviewed
segment bounds where alignment failed. No speaker assignment happens here.

Legacy `FAILED` checkpoints can be recovered only for the old `ReviewError`
post-validation content rejection: complete owned results, valid timestamps,
finished execution, processed audio coverage, at least one failed alignment, and
no execution exception diagnostics. Finalization refuses incomplete, corrupt,
execution-failed, or invalid-coverage results. Before reclassification it archives
the original sealed checkpoint under `recovery/<old-checksum>.json` and records
`recovered_from_checksum`, the recovery reason, and the legacy failure code.
Attempts, inference results, plan fingerprint, parents, and successful checkpoint
bytes are retained. Status classifies in memory and never performs migration.
Existing old successful checkpoints with missing word timings are reported as
`PARTIAL` without changing their bytes.

`--finalize` consumes only existing checkpoints under the session worker lock,
validates reviewed/source/plan lineage and complete coverage, archives/reclassifies
eligible legacy content failures, and publishes `transcript.aligned.fa.json` and
`transcript.aligned.fa.txt` plus
updated progress. It denies network access and never decodes audio, loads a model,
uses ASR/diarization/LLMs, or retries a failed unit. Repeating it is idempotent for
the canonical artifact. A real failed unit still blocks canonical publication.

Canonical segment timing is explicitly labeled `segment_timing_source=REVIEWED_PARENT`
with `reviewed_interval`; unavailable aligned word timings remain null. QA exposes
both segment flag counts and word flag counts, including zero counts. Session 4's
two failed segments have persisted lexical-mapping failures but no engine exception.
The exact returned token lists were not saved, so the lower-level mismatch cause
is UNKNOWN; finalization does not rerun inference to obtain it.

Phase 4C.1 finalized the existing Session 4 fingerprint
`4ba81b10c2e0daff7708ce89c7bcb0b504c8935702d48ee2a4337063554e96f9`
from persisted checkpoints, without inference. All 2810 reviewed segments are
present. Units: 22 `SUCCEEDED`, 2 `PARTIAL`, 0 `FAILED`. Segment QA remains
2808 `ALIGNED`, 2 `ALIGNMENT_FAILED`, 2 `UNALIGNED_WORD` flags, and 47
`SUSPICIOUS_WORD_DURATION` flags. Word-level counts are 4 unaligned words and
48 suspicious-duration flags; these differ from the segment-level flag counts.
Coverage, processing completeness, full alignment completion, warning visibility,
and canonical idempotence passed. Reviewed/raw/audio hashes and modification times
were unchanged; all 22 successful checkpoint files retained their original bytes.
Both reclassified checkpoint originals were archived, and their attempt counts
remain 2. `uv lock --check --offline`, full pytest (166 passed), and
`python -m ruff check .` passed. Detailed private verification evidence is in
`work/<session>/alignment/<fingerprint>/partial-finalization-verification.json`;
transcripts and checkpoints remain ignored local data.

Phase 5.1 applies the same recovery contract to every session and QUALITY. A
completed full plan with SUCCEEDED/PARTIAL units and validated complete segment
coverage publishes as SUCCEEDED_WITH_WARNINGS when QA has warnings. Legacy
SUCCEEDED canonical JSON with identical content and explicit QA warnings remains
valid and is reused byte-for-byte. QUALITY first tries checkpoint-only finalization;
no model is needed when persisted results are safely complete. Its receipt and
console expose failed/partial segment counts, unaligned words, suspicious word
durations, and partial/failed unit counts.

Structurally valid reviewed zero-duration segments cannot provide an alignment
interval. Execution now isolates them before calling the engine, retains reviewed
text/timing, and emits ZERO_DURATION_REVIEWED_SEGMENT with null word timestamps.
Remaining segments still execute, and only a completely returned/validated unit
can finish PARTIAL. Negative durations and genuine engine errors remain failures.
Failure diagnostics distinguish ENGINE_EXECUTION_FAILED from lexical mismatch.
Never promote an aborted checkpoint merely because its fallback words are null.

Ravaq-04 inspection found Unit 3's isolated lexical failure safely PARTIAL. Unit 6
instead has ReviewError, zero processed audio, no completion marker, and nine
unit-exception fallback segments (reviewed indices 359-367). Index 366 has a valid
reviewed zero-duration interval at 1610.08 seconds, which the former adapter
assertion rejects. The historical checkpoint omits the exception message and
discarded per-segment engine results; the exact sanitized-code-only exception
origin cannot exclude the adapter's PCM validation guards. The checkpoint does
not prove completed inference and is not recoverable from persisted data.
No production inference or canonical publication was performed for Ravaq-04.
Private inspection with exact segment IDs/text/timing and immutable input hashes:
`work/s-1c781318ac9e4a083210/alignment/56f75e93fa5a1ba4af9c492cc4785fa625e03aee6a471b367e85f5fb6e64bcce/phase-5.1-inspection.json`.
