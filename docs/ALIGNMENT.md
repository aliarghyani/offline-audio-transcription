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

# Read-only full-run status. Add --smoke for smoke status.
uv run --no-sync session-pipeline align $session --status --reviewed-fingerprint $reviewed --json
```

`--plan`, `--execute`, and `--status` are mutually exclusive. Only `--execute`
loads the real model or decodes derived audio. An omitted reviewed fingerprint
selects the validated `reviewed/latest.json` pointer; the commands above pin the
accepted lineage. Status can select an exact existing plan using
`--status --fingerprint <alignment-fingerprint>`.
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
checksums. A valid `SUCCEEDED` unit is skipped; `FAILED` and interrupted `RUNNING`
units are retried. Tampered or stale checkpoints are refused, not silently reused.
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

The final validator requires exact segment partition, every unit succeeded,
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

## Development boundary

Phase 4C tests use generated synthetic audio, fake model interfaces, and injected
alignment results. They test the adapter call contract without loading real
weights. Session 4 receives a **plan only**. Real model/runtime accuracy, memory
and wall time must be established by the user's manual smoke. No real alignment,
ASR, diarization, LLM invocation, package install or model download is authorized
inside the development task.
