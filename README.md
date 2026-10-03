# Persian Session Transcription — local pipeline

This repository is the privacy-first foundation for a local Persian session
transcription pipeline. Phase 3A implements source discovery, full input integrity
validation, canonical JSON artifacts, stage fingerprints, checkpoint contracts,
and typed QA concepts. It does **not** run transcription or install ML packages.
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
uv sync --dev --python 3.11
uv run --no-sync session-pipeline version
```

`uv sync` installs only Phase 3A tooling. Do not add the future ML stack during
this phase. The frozen future dependency is `whisperx==3.8.6`; Phase 3B must
deliberately select **CPU-only PyTorch packages** and check dependency resolution
before installation, rather than accidentally pulling CUDA packages.

## Current commands

```powershell
uv run --no-sync session-pipeline version
uv run --no-sync session-pipeline discover
uv run --no-sync session-pipeline discover --json
uv run --no-sync session-pipeline config show --json
uv run --no-sync session-pipeline integrity "sessions\example.m4a"
& .\scripts\run.ps1 integrity "sessions\example.m4a" --json
```

`discover` only traverses `sessions/`. `integrity` hashes, probes, and fully
decodes into FFmpeg's null muxer without writing media. It exits non-zero for an
invalid input. Versioned integrity JSON and bounded diagnostics are written
under ignored `work/<session-id>/`. The source is never modified. JSON mode
returns structured results without raw FFmpeg diagnostic text.

Current exit codes: `0` success, `2` CLI usage, `3` invalid input/discovery,
and `4` configuration error. No ASR, alignment, diarization, speaker mapping,
rendering, or model-download command exists yet.

## Architecture

The accepted stage design and invalidation rules are in
[architecture](docs/ARCHITECTURE.md). Artifact checksums and schema rules are in
[artifacts](docs/ARTIFACTS.md). Historical workflow files live in `legacy/`.
