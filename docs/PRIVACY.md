# Privacy boundary

- Original recordings are immutable. Never modify, rename, repair, or overwrite
  a file under `sessions/`.
- Session audio, reference transcripts, generated transcript outputs, and
  credentials never belong in Git. `sessions/`, `reference_transcripts/`,
  `work/`, `outputs/`, model caches, and local environments are ignored.
- Generated JSON, diagnostics, and future TXT may contain sensitive conversation
  material. `work/` and `outputs/` remain local and should be shared only by an
  explicit, separate decision.
- Historical reference transcripts cannot influence fresh ASR, alignment,
  diarization, or speaker identity assignment. They are provenance and possible
  post-run QA material only.
- Secrets are read from the process environment when a future authorized stage
  needs them. `HF_TOKEN` is never stored in tracked settings, canonical JSON,
  CLI output, or diagnostics. `.env.example` contains only a placeholder.
- Phase 3A uses FFmpeg locally for probing and null-output decode validation.
  It makes no model request and performs no transcription.
