# Pipeline architecture

Phase 3A implements the integrity and contract foundation only. Future stages
are documented here; they are not stub modules or available CLI commands.

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
None of these ML dependencies is installed in Phase 3A.

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
