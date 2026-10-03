# Persian Session Transcription — local pipeline

This repository is the privacy-first foundation for a local Persian session
transcription pipeline. Phase 3A implements source discovery, full input integrity
validation, canonical JSON artifacts, stage fingerprints, checkpoint contracts,
and typed QA concepts. Phase 3B provisions CPU ML software and adds an offline
environment doctor. No command runs transcription or downloads model weights.
No psychological or therapy-content interpretation belongs in this repository.

## Private data boundary

Put future recordings in `sessions/` and leave each original file unchanged.
The source filename may contain spaces or Persian Unicode; the internal session
ID comes from the source SHA-256. `sessions/`, `reference_transcripts/`, `work/`,
`outputs/`, caches, and credentials are excluded from Git. Historical reference
transcripts must not influence fresh ASR. See [privacy](docs/PRIVACY.md).

The recording present during Phase 3A is structurally incomplete: its declared
`mdat` size extends beyond the physical file, `moov` is absent, and FFmpeg cannot
open it for decoding. It is retained unchanged as a local negative integrity
case. Re-copy or re-export from the original source to a separate file before
any future transcription attempt.

## Bootstrap on Windows

Use Python 3.11, `uv`, and FFmpeg/ffprobe on `PATH`:

```powershell
uv sync --locked --dev --python 3.11
uv run --no-sync session-pipeline version
```

The checked-in lock installs the Phase 3B software stack. `torch`, `torchaudio`,
and `torchvision` resolve through explicit source mappings to the official
[PyTorch CPU index](https://download.pytorch.org/whl/cpu). See the exact validated
versions and dependency decisions in [ML runtime](docs/ML_RUNTIME.md).

### Windows shared FFmpeg runtime

The CLI integrity checks use `ffmpeg.exe` and `ffprobe.exe` found on `PATH`.
TorchCodec 0.7.0 separately needs FFmpeg 4–7 shared libraries; an executable-only
FFmpeg distribution does not satisfy that requirement. This machine uses the
trusted **Gyan.FFmpeg.Shared 7.1.1** WinGet package at user scope:

```powershell
winget install --id Gyan.FFmpeg.Shared --exact --version 7.1.1 --scope user --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
```

The original yt-dlp FFmpeg distribution is preserved. The doctor discovers the
shared build's real `bin` directory and registers it with `os.add_dll_directory`
inside fresh Python child processes. It keeps the registration alive through
import and decoding. No DLL is copied into site-packages and no binary is
patched. Arbitrary DLL downloads/copies are prohibited because their provenance
and ABI compatibility are unknown. WinGet may change user PATH; the doctor
reports whichever executables the current shell actually resolves.

## Current commands

```powershell
uv run --no-sync session-pipeline version
uv run --no-sync session-pipeline discover
uv run --no-sync session-pipeline discover --json
uv run --no-sync session-pipeline config show --json
uv run --no-sync session-pipeline doctor
uv run --no-sync session-pipeline doctor --json
uv run --no-sync session-pipeline integrity "sessions\example.m4a"
& .\scripts\run.ps1 integrity "sessions\example.m4a" --json
```

`discover` only traverses `sessions/`. `integrity` hashes, probes, and fully
decodes into FFmpeg's null muxer without writing media. It exits non-zero for an
invalid input. Versioned integrity JSON and bounded diagnostics are written
under ignored `work/<session-id>/`. The source is never modified. JSON mode
returns structured results without raw FFmpeg diagnostic text.

`doctor` imports each ML component in a fresh offline process, checks CPU-only
PyTorch, CTranslate2 INT8, WhisperX mappings/signatures, and decodes a generated
temporary tone with TorchCodec. It never reads sessions or reference transcripts,
calls an inference model loader, or uses a token value. PASS/FAIL describe the
software checks; WARN explicitly leaves model access/revisions/RAM fit untested.
Model packages installed **does not mean the requested inference models are
provisioned**. No model repository or separate model-weight download was initiated.
The standard WhisperX wheel includes the explicitly approved bundled VAD
checkpoint (17,719,103 bytes), installed but never loaded in Phase 3B. External
model downloads and model loading/inference were prohibited and did not occur;
see the classification in [ML runtime](docs/ML_RUNTIME.md).

Current exit codes: `0` success (including doctor WARN), `2` CLI usage,
`3` invalid input/discovery, `4` configuration error, and `5` doctor software
failure. No ASR, alignment, diarization, speaker mapping,
rendering, or model-download command exists yet.

## Architecture

The accepted stage design and invalidation rules are in
[architecture](docs/ARCHITECTURE.md). Artifact checksums and schema rules are in
[artifacts](docs/ARTIFACTS.md). Historical workflow files live in `legacy/`.
