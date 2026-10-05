# Pipeline architecture

Phase 3A implements the integrity and contract foundation. Phase 3B adds the
locked CPU software environment and offline `doctor` command. Phase 3C adds
pinned model-file provisioning/status/verification, without model loading.
Phase 4A adds [ASR quality review and separate targeted retry candidates](ASR_QUALITY.md)
from completed raw JSON. Review/status and retry planning are offline metadata
operations; explicit `qa retry --execute` launches a local pinned ASR worker.
Alignment and diarization remain future CLI stages. Codex is development-only;
local runtime has no cloud LLM dependency or automatic candidate replacement.

```text
Integrity → Prepare → ASR → Alignment ─┐
             └──────→ Diarization ─────┤→ Attribution → QA → Render
                                       └→ human speaker mapping ─┘
```

Integrity computes a streaming source hash, obtains structured ffprobe metadata,
requires an audio stream, and performs complete FFmpeg decode validation to a
null output. A failure blocks all downstream work. Prepare will create only
derived media and a timeline under ignored `work/`. ASR will use WhisperX 3.8.6
with faster-whisper/CTranslate2, Whisper large-v3, forced `fa`, CPU, and initial
`int8` compute. Alignment will use a pinned Persian WhisperX model. Diarization
will use `pyannote/speaker-diarization-community-1` with `num_speakers=2`.
These software dependencies are installed and import-validated in Phase 3B.
Whisper large-v3, Persian alignment and Community-1 diarization weights are not
downloaded or loaded. The required WhisperX wheel's bundled VAD checkpoint is an
explicitly approved package asset, installed but never loaded. External model
downloads and model loading/inference did not occur in Phase 3B; see
[ML runtime](ML_RUNTIME.md).

Phase 3C's separate [model lock and offline path contract](MODEL_PROVISIONING.md)
pins the approved external repositories. ASR and PyTorch Persian alignment files
are provisioned locally; Community-1 remains BLOCKED_AUTH. No inference model
was loaded. Future loaders must validate and use local pinned paths with offline
semantics, failing on missing files rather than fetching another revision.

ASR may use checkpointed chunks with absolute time offsets; duration, overlap,
batch size, and workers remain unset until measured on valid input. Chunking is
an execution detail, not an assumption about the current recording. Full-session
diarization is preferred so the two anonymous cluster IDs share one namespace.
It depends on a valid source/timeline but not on ASR text. Attribution requires
both alignment and diarization. `fill_nearest=false`: a word without an adequate
speaker overlap remains unassigned for review. The two-cluster constraint does
not identify either person; a human must explicitly approve cluster-to-name
mapping. QA must check completeness, suspicious gaps, tail coverage, and
uncertainty before rendering. A final Persian TXT is derived from approved
canonical JSON, never treated as the source of truth.

The target has 16 GB total RAM and no detected NVIDIA GPU. Future heavy model
stages should run sequentially, release process memory between stages, and use
small measured batches. This is a strategy, not a claim that a model fits the
currently available RAM. Model provisioning is separate from runtime; pinned
model revisions and local cache checks prevent surprise downloads.

## Validated software boundary (Phase 3B)

The exact stack is documented in [ML runtime](ML_RUNTIME.md). PyTorch 2.8.0+cpu,
torchaudio 2.8.0+cpu and torchvision 0.23.0+cpu come from an explicit official CPU
index in pyproject.toml and uv.lock. Runtime verification requires a CPU build
identifier, `torch.version.cuda is None`, CUDA unavailable, and zero CUDA devices.
CTranslate2 reports CPU `int8` support; no Whisper model is created for this check.

Integrity continues to resolve FFmpeg executables through PATH. TorchCodec 0.7.0
uses a separate trusted Gyan FFmpeg Shared 7.1.1 user installation. The doctor
checks its DLL families and registers the directory with Python's Windows DLL
search API in fresh subprocesses. Copying arbitrary DLLs into site-packages or
patching installed binaries is prohibited. A temporary synthetic PCM16 tone
validates actual audio decoding, then is removed; private source audio is unused.

The doctor imports software with HF/Transformers offline flags and pyannote
telemetry disabled, removes token variables from child environments, and inspects
WhisperX mappings/signatures/source without calling loaders. It reads no transcript
content and does not obtain HF_TOKEN values. CLI configuration loads non-secret
settings; the existing environment-only secret provider remains available to a
future explicitly authorized stage. Doctor JSON preserves import diagnostics and
distinguishes software PASS/FAIL from model-readiness WARN. It reports no models
loaded; it does not claim that gated model access or memory fit has been tested.

## Resumability and invalidation

Every expensive stage will have a deterministic fingerprint using source hash,
relevant settings, required parent artifact checksums, stage/schema and pipeline
versions, and pinned model identity/revision where relevant. Operational
timestamps and secrets are excluded. A checkpoint is reusable only if its
fingerprint matches, its artifact and all required parents have valid checksums
and supported schemas, and their statuses are `SUCCEEDED`. A filename alone
never proves validity. Failed/interrupted writes cannot publish completed
artifacts.

A diarization setting change invalidates diarization, attribution, relevant QA,
and rendering while preserving valid ASR and alignment. An ASR model or setting
change invalidates ASR, alignment, attribution, relevant QA, and rendering.
Changing the source hash invalidates all stages. A diarization retry after an
ASR success therefore reuses the valid ASR checkpoint.
