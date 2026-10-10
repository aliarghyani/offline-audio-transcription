# Pinned model provisioning (Phase 3C)

Explicit model-management commands fetch/verify files without loading models,
processing audio or reading reference transcript content. Normal `transcribe`
uses provisioned local files and never starts provisioning automatically.

## Approved identities

| Role | Repository | Immutable revision | License metadata |
| --- | --- | --- | --- |
| ASR | Systran/faster-whisper-large-v3 | edaa852ec7e145841d8ffdb056a99866b5f0a478 | MIT |
| Alignment | jonatasgrosman/wav2vec2-large-xlsr-53-persian | 234714078a1398a9db88194c5a40fefe6f376dc1 | Apache-2.0 |
| Diarization | pyannote/speaker-diarization-community-1 | 3533c8cf8e369892e6b79ff1bf80f7b0286a54ee | CC-BY-4.0 |

`config/models.lock.json` records relative destinations, required/optional files,
sizes, pinned content identities, family and license metadata. Large weight
SHA-256 identities come from official Hub LFS metadata. Regular files initially
use their pinned Git blob identity; explicit provisioning records their computed
SHA-256 in the lock after verification. Local receipts contain sizes, hashes,
revision, file metadata and a lock fingerprint; they contain no credentials.

Read live readiness with `models status`; historical Phase 3C BLOCKED_AUTH reports
do not describe current provisioning. Session 4 later completed local Community-1
inference. The lock contains public model identities; receipts, credentials and
caches stay ignored. Normal runtime requires no HF_TOKEN.

## Local storage and commands

```text
models/asr/<revision>/
models/alignment/<revision>/
models/diarization/<revision>/
models/.hf-cache/<role>/<revision>/
```

Everything under `models/` is Git-ignored. Hub local-directory metadata also lives
inside each snapshot's `.cache/huggingface/`; no unrelated global cache is changed.
The bundled WhisperX VAD from Phase 3B remains an allowed package asset in
site-packages and is separate from these external models.

```powershell
uv run --no-sync session-pipeline models status
uv run --no-sync session-pipeline models status --json
uv run --no-sync session-pipeline models provision
uv run --no-sync session-pipeline models provision --role diarization --json
uv run --no-sync session-pipeline models verify
uv run --no-sync session-pipeline models verify --role asr --json
uv run --no-sync session-pipeline models check-updates --json
```

`status` is read-only and never contacts the Hub. It checks pinned files and their
verified receipts/sizes/mtime; changed metadata requires fresh verification.
`verify` is offline, hashes every present locked file and validates configurations.
It writes a local verified receipt, never downloads or changes model weights.
`provision` explicitly contacts only approved repositories at locked commits.
It reuses valid files. Model-management exit code 6 means missing, partial,
corrupt or blocked roles; code 4 means invalid configuration or concurrent
provisioning. Doctor's exit 0 indicates software success, independently of its
`models_ready` field. Both command families report `models_loaded=false`.

Large files use bounded 16 MiB HTTP ranges from the official Hub-provided CDN URL,
after checking revision, size and etag against the lock. Partials are retained in
the ignored revision/checksum-bound cache. Only a full, checksum-verified file is
atomically published to the snapshot. This avoids the stalled unbounded transfer
observed here. Small files use `hf_hub_download` with the exact revision, local
destination and explicit token policy. A local file lock serializes provisioning.
No signed URLs, request headers or raw Hub error messages are persisted.

States: MISSING, PARTIAL, PROVISIONED, CORRUPT, BLOCKED_AUTH,
BLOCKED_ACCESS_CONDITIONS. A directory alone never proves readiness. Alignment
excludes `flax_model.msgpack`; no Flax runtime is required. Community-1's required
assets are `config.yaml`, segmentation/embedding checkpoints, and both packaged
PLDA NPZ files. Its illustrative GIF is excluded.

## Gated access and secrets

Accept the [official Community-1 conditions](https://huggingface.co/pyannote/speaker-diarization-community-1)
and provide an appropriately authorized HF_TOKEN through the process environment.
Do not paste it into chat or put its value in command arguments/project files.
No login/token-save flow is used. Public model downloads explicitly disable
implicit cached-token use; gated access receives the environment token in memory.
Missing tokens do not stop public provisioning. Access errors produce fixed
non-secret state codes; no mirrors, alternative models or access workarounds exist.

## Future offline loading contract

Future authorized loaders must call `local_model_path(root, role)` first. This
full checksum/configuration guard fails on a missing or corrupt pinned snapshot;
it never downloads. Use the returned directory instead of the configured remote
model name. Launch future inference with `local_only_environment(root)`, which
sets HF/Transformers offline flags, project-local HF cache variables, disables
telemetry and removes token variables.

ASR: pass the local directory to faster-whisper with `local_files_only=True`.
Alignment: pass the local directory to WhisperX with `model_cache_only=True`.
Community-1: pass its local directory to pyannote; the installed 4.0.7 loader
resolves `$model/segmentation`, `$model/embedding` and `$model/plda` locally.
No loader is called during provisioning. New inputs pass integrity before
transcription; earlier damaged-source readiness reports do not apply to a newly
supplied valid recording.

`check-updates` is an explicit metadata-only network operation that reports remote
main drift without changing pins. Model updates require an explicit lock change,
new snapshots and a new validation cycle; provisioning never follows floating main.
