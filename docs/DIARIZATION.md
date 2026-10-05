# Phase 4D: local two-speaker diarization foundation

Development is complete using synthetic fixtures. Session 4 has a plan only;
real diarization, model loading/provisioning, ASR/alignment inference, reference
transcript use, identity mapping, and transcript attribution were not performed.

## Model and provisioning

Reuse Phase 3C's `session-pipeline models provision --role diarization`. No second
downloader, model mirror, replacement model, or lock revision was introduced.

- Repository: `pyannote/speaker-diarization-community-1`
- Revision: `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
- Directory: `models/diarization/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/`
- Required: config YAML, segmentation and embedding checkpoints, and two PLDA NPZs.
- Current state: `BLOCKED_AUTH`, zero required files provisioned. An ignored
  historical receipt exists; it does not establish model readiness or granted access.

The existing provisioner reads HF_TOKEN from the process environment only,
performs gated-access preflight, requests the exact locked revision, records file
inventory/hashes, and verifies local bytes/configuration after downloading.
Missing credentials and denied conditions have distinct fixed state codes.
Raw remote exceptions and credentials are not published. Git-blob-pinned small
files gain SHA256s after validation. Diarization fingerprints bind their stable
Git blob identities, so this checksum enrichment does not invalidate the plan.
Neither runtime nor status requires a token. All `models/` files, project HF
caches, and credentials remain ignored by Git.

The operator must accept the [official gated model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1)
in their account and create a token permitted to read this repository. Run the
following only in ordinary PowerShell; it was not executed during development.
Enter the token at the secure prompt; its value is not a command argument or file.

```powershell
Set-Location -LiteralPath 'E:\personal\transcripts'
$diarizationToken = Read-Host 'One-time Hugging Face read token' -AsSecureString
try {
    $env:HF_TOKEN = [System.Net.NetworkCredential]::new('', $diarizationToken).Password
    uv run --no-sync session-pipeline models provision --role diarization --json
    if ($LASTEXITCODE -ne 0) { throw 'Model provisioning did not pass; inspect its state.' }
} finally {
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    $diarizationToken.Dispose()
}
uv run --no-sync session-pipeline models verify --role diarization --json
if ($LASTEXITCODE -ne 0) { throw 'Offline model verification did not pass.' }
```

Provisioning may enrich tracked model-lock SHA256 values; review those ordinary
model metadata changes separately. It never follows floating `main` for downloads.
After provisioning, remove token variables and use the commands below without a
token. Do not run ASR/alignment/another heavy inference process concurrently.

## Commands

```powershell
Set-Location -LiteralPath 'E:\personal\transcripts'
$session = 's-d57224782d66f69a7446'
$aligned = '4ba81b10c2e0daff7708ce89c7bcb0b504c8935702d48ee2a4337063554e96f9'

# Metadata-only; omission of an action also means plan.
uv run --no-sync session-pipeline diarize $session --plan --aligned-fingerprint $aligned --json

# Operator-only real smoke after provisioning: first 180 seconds.
uv run --no-sync session-pipeline diarize $session --execute --smoke --aligned-fingerprint $aligned --json

# Operator-only whole-session execution. Repeat to reuse success or restart an interrupted job.
uv run --no-sync session-pipeline diarize $session --execute --aligned-fingerprint $aligned --json

# Read-only status; use --smoke for the smoke namespace.
uv run --no-sync session-pipeline diarize $session --status --aligned-fingerprint $aligned --json
uv run --no-sync session-pipeline diarize $session --status --fingerprint 41031dbf581439b2afb2c842096a129c0b652e360bab710a533cc16dc8ab11eb --json
```

Modes are mutually exclusive. `--fingerprint` selects an exact existing plan for
status; conflicting parent selection fails. Explicit execute returns exit 3 when
processing did not complete. Planning can succeed while gated model access is
blocked; this is not inference readiness. Status never creates a plan, loads a
model, decodes audio, writes a receipt, or accesses the network. It hashes local
model files and reports the pinned revision separately from a verified revision.

## Whole-session clustering and resources

Community-1 computes internal segmentations and embeddings, then clusters the
recording globally and reconstructs speaker turns. The installed pyannote.audio
4.0.7 `SpeakerDiarization.apply` calls clustering with the complete embedding
array. Its public inference API does not expose a supported, safely resumable
global clustering checkpoint. Independent 300-second jobs would produce unrelated
speaker label spaces. Therefore full mode is one recording-level inference job,
not a sequence of independently diarized alignment units. This is supported by
the [official pipeline source](https://github.com/pyannote/pyannote-audio/blob/main/src/pyannote/audio/pipelines/speaker_diarization.py).

Execution calls the verified local snapshot loader with `token=False`, CPU,
four CPU threads, segmentation batch size 4, embedding batch size 8, and exactly
`min_speakers=2`, `max_speakers=2`. The installed 4.0.7 loader resolves the pinned
config's `$model/segmentation`, `$model/embedding`, and `$model/plda` locally.
Full checksum/config verification happens before decoding/factory/inference.
Offline environment variables, disabled telemetry, project-local cache, removed
token variables, and a socket guard cover runtime. No Hub fallback or download
attempt is authorized. A missing/corrupt model yields FAILED at MODEL_VERIFICATION.

Audio comes only from the immutable source. An ephemeral, separately stored mono
16 kHz PCM derivative provides format consistency. Full mode decodes the entire
recording; smoke decodes only the first 180 seconds and clusters that excerpt.
The adapter reads validated 16-bit PCM into a float32 waveform dictionary, the
supported pyannote input API, avoiding dependence on TorchCodec DLL discovery in
the worker process. No transcript is used as model input. No ASR/alignment model is loaded alongside
diarization. Same-session diarization, alignment, and ASR retry worker locks exclude
simultaneous runners. Model/PCM references are released and PCM is deleted on
success/failure/interruption. The operator must still avoid other-session heavy
jobs. Real peak RAM, CPU wall time, and accuracy on 16 GB hardware remain NOT RUN;
the real smoke is the resource/quality gate before full execution.

## Checkpoints and interruption

```text
work/<session>/diarization/<fingerprint>/plan.json
work/<session>/diarization/<fingerprint>/checkpoint.json
work/<session>/diarization/<fingerprint>/attempts/<previous-checksum>.json
outputs/<session>/diarized/<fingerprint>/diarization.json
outputs/<session>/diarized/smoke/<fingerprint>/diarization.json
```

Checkpoint states: RUNNING, INTERRUPTED, FAILED, SUCCEEDED. Caught
KeyboardInterrupt/SystemExit seals INTERRUPTED; read-only status recognizes an
unlocked stale RUNNING checkpoint as INTERRUPTED. A live worker remains RUNNING.
Only a complete, validated successful result is reused, without model loading.
If publication was interrupted after checkpoint success, a repeat execute can
republish the stored result without inference. Corrupt/tampered checkpoints and
existing conflicting outputs are refused. Interrupted/failed retries archive the
previous sealed attempt and restart the whole recording, preserving anonymous
speaker consistency within each resulting job. Partial inference is not reused.

## Output and QA contracts

Output is separate from source/raw/reviewed/aligned lineage. It records source
SHA256/duration, aligned checksum/fingerprint, model repo/revision, plan/config
fingerprint, interval/mode, checkpoint attempt, timestamps, and model inventory.
Pre/post input hashes and modification checks protect immutable parents. Statistics
include wall seconds, audio seconds, realtime factor, speaker/turn counts, speech
union duration, and summed turn duration (which includes simultaneous speakers).

The canonical `turns` list contains absolute start, end, duration, and anonymous
speaker. It uses regular `speaker_diarization`, never the exclusive variant;
overlapping turns and explicit overlap intervals remain available. Labels are
SPEAKER_00/SPEAKER_01, not people. Labels are scoped to this recording/job; smoke
labels are not asserted to identify the same speakers as full labels.

Fatal QA separately records empty output, missing expected speaker, invalid
additional/nonanonymous label, invalid duration, out-of-bounds turn, and ordering.
Valid turns satisfy `0 <= start < end <= source_duration` and the processed
interval, ordered by `(start, end, speaker)`. No timestamp repair/clamping or
speaker-count fabrication occurs. Failed QA blocks canonical publication.

Warnings retain short turns (<0.1 seconds), uncovered intervals (>30 seconds), and
reviewed segment bounds uncovered for more than one second. These are operational
inspection thresholds, not calibrated confidence or proven silence: absent model
turns do not prove silence; reviewed bounds are only a reference proxy. Speaker
confidence is unavailable, not invented. Coverage uses the union of regular turns
so overlap does not inflate it. Complete processing does not require 100% speech
coverage. Speaker attribution, identity assignment, DER/scientific acceptance,
and transcript rendering are deferred.

All 2810 aligned segment references remain available with stable reviewed IDs,
alignment status, and unchanged reviewed intervals. The two ALIGNMENT_FAILED
segments retain explicit REVIEWED_SEGMENT_FALLBACK timing. Later attribution can
combine aligned word timings and speaker turns, falling back to those bounds.
This stage does not insert labels into text or discard failed alignments.

## Session 4 plan-only evidence

Aligned parent fingerprint:
`4ba81b10c2e0daff7708ce89c7bcb0b504c8935702d48ee2a4337063554e96f9`.
Aligned parent checksum:
`a1e3b751c1626e0558ac4e0f15fc23fe46f1fe7e07844bb68d2b19ced2fd8366`.
Source duration: 7094.421333 seconds. Full plan fingerprint:
`41031dbf581439b2afb2c842096a129c0b652e360bab710a533cc16dc8ab11eb`.

The plan contains 2810 references and both alignment fallbacks. Execution state
is PENDING, model state BLOCKED_AUTH. No diarization checkpoint or canonical
diarization output was created for Session 4. Reviewed/raw/audio/aligned hashes
and modification times, alignment checkpoints, and gated model files were
unchanged. Private plan evidence is under the plan's `development-verification.json`.

Synthetic tests cover model pinning/corruption/missing files, offline CPU/two-speaker
loader calls, overlap, turn validation, lineage, fallback references, smoke,
interruption/retry, status purity, token exclusion, and immutable inputs. Required
verification commands: `uv lock --check --offline`, `python -m pytest -q`, and
`python -m ruff check .`. No real diarization or model download is part of these tests.

Executed local verification after the waveform adapter change:

- `uv lock --check --offline`: PASS, 109 packages resolved offline.
- `.\.venv\Scripts\python.exe -m pytest -q --tb=short`: PASS, 208 tests in 42.72 seconds.
- Diarization-focused synthetic suite: PASS, 42 tests.
- `.\.venv\Scripts\python.exe -m ruff check .`: PASS.
- Ruff format check for the three changed Python files: PASS.
- `git diff --check` and model/cache/private-plan ignore checks: PASS.
- Guarded Session 4 planning/status: PASS; ML imports, execution, and network
  prohibited; input/model files and alignment checkpoint hashes/mtimes unchanged.

The graph service returned `Transport closed`; exact source and executable
verification supplied the evidence. Real model loading, smoke/full inference,
RAM fit, wall time, and diarization accuracy remain NOT RUN.
