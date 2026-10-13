# Local Persian session transcription

Transcribe a new recording on Windows, using pinned CPU models:

```powershell
uv run --offline --no-sync session-pipeline transcribe .\sessions\session5.m4a --profile full
# Equivalent PowerShell wrapper:
.\scripts\run.ps1 transcribe .\sessions\session5.m4a -Profile full
```

- **full** (default): ASR → QA → bounded retry → reviewed transcript → Persian alignment
  → whole-session two-speaker diarization → anonymous attribution → human identity gate
  → labelled final JSON/TXT. Use this for your normal two-person session.
- **quality**: the same stages through alignment, producing aligned JSON and readable TXT
  without speaker labels.
- **fast**: integrity → ASR → raw JSON/TXT. Lowest processing cost; no downstream QA/retry.

The source is hashed and fully checked with FFmpeg **before inference**. Its bytes are
never changed. Spaces, Persian filenames, external files and nested recordings work;
external/nested input gets an automatic private content-addressed copy under `sessions/`.
Supported extensions: M4A, MP4, MP3, WAV, FLAC, OGG, OPUS, AAC, WMA.

**Interrupted? Run the same command again.** Verified completed stages are reused.
Alignment units and retry windows resume; complete native diarization is saved before
validation and reused after an interruption. A stale RUNNING workflow becomes
INTERRUPTED on read-only status and is recovered on the next run. Incomplete initial
ASR may restart decoding; partial diagnostics never count as a completed transcript.

**Identify speakers:** FULL stops at `WAITING_FOR_USER: SPEAKER_IDENTITY` and prints one
exact `uv run --no-sync session-pipeline review <session-id> --open` command. Run it,
listen to local samples, explicitly confirm the Ali/consultant mapping, then stop the
review server with Ctrl+C and repeat the original `transcribe` command. Fingerprints
are resolved from the saved workflow; you do not need to remember them. Identities
are never inferred from text, names, speaking time or voice guesses.

**Find the TXT:** completion prints absolute `TXT:` and `JSON:` paths. FULL writes
`outputs/<session-id>/final/<lineage>/transcript.final.fa.txt` and `.json`.
FAST writes `outputs/<session-id>/transcript.raw.fa.txt` and `.json`;
QUALITY writes `outputs/<session-id>/aligned/<lineage>/transcript.aligned.fa.txt`
and `.json`. The profile and review policy persist in `work/<session-id>/workflow.json`.

**Cloud and internet:** no ChatGPT, cloud LLM, API key, paid service or active coding
agent is needed at runtime. Inference is local and denies network connections.
Internet is needed once to install software and explicitly provision models.
Normal transcription/status never downloads models or asks for HF_TOKEN.

## One-time setup

Use Python 3.11, uv, and FFmpeg/ffprobe available on PATH:

```powershell
uv sync --locked --dev --python 3.11
uv run --no-sync session-pipeline doctor
uv run --no-sync session-pipeline models provision --role asr
uv run --no-sync session-pipeline models provision --role alignment
```

FULL also needs the gated Community-1 files: accept its official access conditions,
provide an authorized HF_TOKEN only in the provisioning process environment, then:

```powershell
uv run --no-sync session-pipeline models provision --role diarization
# Remove the provisioning token from this PowerShell process after provisioning:
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
uv run --offline --no-sync session-pipeline models verify
```

Never paste tokens into commands, config or chat. See [model provisioning](docs/MODEL_PROVISIONING.md)
for exact locked revisions and access conditions. Local status is read-only; offline
`models verify` hashes files and refreshes local receipts. Missing/corrupt/unverified
required files produce `BLOCKED_MODEL_PROVISIONING` with the exact provisioning command.
Receipt checks inspect all profile model roles before execution; loaders additionally
verify full payload checksums before loading. Completed verified stages need no model load.

TorchCodec 0.7.0 requires FFmpeg 4–7 **shared** libraries separately from CLI FFmpeg.
The supported user-scope Windows installation (no administrator rights):

```powershell
winget install --id Gyan.FFmpeg.Shared --exact --version 7.1.1 --scope user --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
```

See [runtime details](docs/ML_RUNTIME.md). The doctor reports environment readiness;
it does not run transcription or prove future performance/RAM fit.

## Review policy, status and batch

The default **auto-prefer-retry** reuses Session 4's `AUTO_PREFER_B` policy: choose a
structurally usable bounded retry candidate, recover words from padded context where
safe, otherwise retain A. This avoids dozens of manual A/B clicks while pursuing
accuracy. It is explicitly **automatic, not human verification**; acoustic accuracy
is not guaranteed. Unselected entries beyond the 900-second decoded retry budget remain
in A with warnings; targeted retries never escalate to a full ASR rerun.

`--review-policy conservative` runs the bounded retry but retains original A text.
`--review-policy human` opens a separate resumable A/B gate when retry targets exist;
it requires resolved decisions before structural acceptance/alignment. With no retry
targets, all policies retain raw text without claiming human verification.
Alignment lexical failures remain PARTIAL with warnings and preserved words.
Safely complete partial alignment finishes QUALITY as SUCCEEDED_WITH_WARNINGS,
with full segment coverage, null unavailable word timings, and visible warning counts.
Checkpoint-only finalization publishes both aligned JSON and TXT without a model.
`--speakers 2` is the only supported count for the current product.

```powershell
uv run --offline --no-sync session-pipeline status .\sessions\session5.m4a
uv run --offline --no-sync session-pipeline status <session-id> --json
uv run --offline --no-sync session-pipeline transcribe .\sessions --profile full
uv run --offline --no-sync session-pipeline transcribe .\sessions\ravaq --profile quality --continue-on-error
uv run --offline --no-sync session-pipeline transcribe .\sessions\session5.m4a --profile full --dry-run
```

Directory batches default to stop on the first failure or gate. `--continue-on-error`
continues after FAILED sessions, while review/model gates and interruption still stop.
The final summary reports succeeded, succeeded with warnings, failed, waiting/blocking,
and skipped/reused sessions, plus unprocessed files. Reuse counts overlap completion
counts. Failed filenames/session IDs stay visible in text and JSON output; any failure
keeps a nonzero exit status even when later files succeed.

Status verifies saved files without loading models or contacting the network, and
recognizes an existing pre-productization final lineage read-only. Multiple historical
final lineages require explicit engineering selection; no arbitrary newest-file guess.
`--json` gives detailed machine-readable output. Default output reports stage runtime,
reuse, warnings, gates and output paths. Runtime displayed on reused stages is the
recorded stage runtime, not a new inference duration.

Directory batches run **sequentially** on this 16 GB machine. Discovery sorts paths,
ignores symlinks/unsupported files and deduplicates identical audio. The batch stops
at the first human gate, model block, interruption or failure and reports unprocessed
files. After resolving the gate, repeat the same batch command.

Exit codes: 0 completed/planned, 2 CLI usage, 3 input/stage failure, 4 configuration,
5 doctor environment failure, 6 model provisioning block, 7 waiting for human review,
130 interrupted. The wrapper preserves these codes and visible errors.

## Configuration, privacy and engineering

Defaults are centralized in `config/defaults.json` (Persian, CPU INT8, models, two speakers),
`config/models.lock.json` (immutable revisions/public file identities),
`config/asr-quality.json` (QA thresholds), and `product_defaults.py` (profile, review
policy, 900-second retry budget, shared decode options, 0.75-second turn grouping,
offline contract). Stage fingerprints include relevant settings/models/parents;
changing policy reuses ASR but creates a different reviewed/downstream lineage.
The frozen model/device contracts are validated rather than silently substituted.
Changed/corrupt canonical raw ASR is preserved and fails clearly; use a separate
workspace for a deliberately changed ASR model/decode contract.

`sessions/`, `outputs/`, `work/`, `models/`, `reference_transcripts/`, caches and secrets
remain ignored. Reference transcript content never conditions fresh ASR; its directory
is rejected as input. No private recording/transcript is stored in tests.

See [workflow design and acceptance](docs/WORKFLOW.md), [architecture](docs/ARCHITECTURE.md),
[artifacts](docs/ARTIFACTS.md), [privacy](docs/PRIVACY.md), [ASR QA](docs/ASR_QUALITY.md),
[alignment](docs/ALIGNMENT.md), [diarization](docs/DIARIZATION.md),
[attribution](docs/ATTRIBUTION.md), and [final rendering](docs/FINAL_TRANSCRIPT.md).

```powershell
uv lock --check --offline
uv run --offline --no-sync pytest
uv run --offline --no-sync ruff check .
```

Product acceptance uses synthetic audio/artifacts and injected inference functions;
real ML throughput, RAM fit and accuracy on future recordings remain to be measured.
