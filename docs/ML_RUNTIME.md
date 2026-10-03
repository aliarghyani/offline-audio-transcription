# Phase 3B software environment

## Dependency decisions before installation

Use Python 3.11 and the checked-in uv lock on Windows. WhisperX 3.8.6,
torch 2.8.0, torchaudio 2.8.0, torchvision 0.23.0, and TorchCodec 0.7.0
are explicit project pins. The three PyTorch distributions resolve only from
the explicit official https://download.pytorch.org/whl/cpu index.

The resolution contains three required transitive packages with beta version
labels: opentelemetry-semantic-conventions, opentelemetry-exporter-otlp-common,
and opentelemetry-exporter-http-transport, all 0.66b0. This is a narrowly
justified exception to the stable-package policy: stable pyannote.audio 4.0.7
requires OpenTelemetry SDK/exporter >=1.34.0; the selected stable SDK/exporter
1.45.0 requires these exact matching 0.66b0 dependencies. These are software
telemetry dependencies, not inference models or a prerelease WhisperX stack.
Validation disables pyannote telemetry and Hugging Face network access.

The lock was inspected before sync: no NVIDIA, CUDA, Triton, or GPU ONNX
distribution was selected. The Windows cp311 PyTorch wheel versions are
2.8.0+cpu, 2.8.0+cpu, and 0.23.0+cpu respectively. Installed runtime checks
must independently confirm the CPU build.

Phase 3B distinguishes four categories:

1. Python/package downloads: allowed and installed through uv.
2. Bundled package model assets: the user explicitly approved the unavoidable
   WhisperX 3.8.6 VAD checkpoint, `whisperx/assets/pytorch_model.bin`
   (17,719,103 bytes), as an allowed package asset. It was installed inside the
   official distribution, not fetched separately from Hugging Face/model hub.
3. External inference model downloads: prohibited; none occurred.
4. Model loading/inference: prohibited; none occurred, including the bundled VAD.

## Provisioned versions

| Component | Validated version |
| --- | --- |
| Python | 3.11.5 |
| uv | 0.12.13 |
| session-pipeline | 0.1.0 |
| WhisperX | 3.8.6 |
| torch | 2.8.0+cpu |
| torchaudio | 2.8.0+cpu |
| torchvision | 0.23.0+cpu |
| TorchCodec | 0.7.0 |
| faster-whisper | 1.2.1 |
| CTranslate2 | 4.8.2 |
| pyannote.audio | 4.0.7 |
| transformers | 4.57.6 |
| huggingface-hub | 0.36.2 |

All transitive versions and artifact hashes are recorded in uv.lock. No pip
installation outside uv was used. The supported lock environment is Windows;
the wheel tags enforce the architecture. Keep the existing dev dependency group.

## Shared runtime and verification

The existing yt-dlp WinGet FFmpeg executables report
`N-121583-g4348bde2d2-20251031` and have no adjacent shared DLLs. TorchCodec
initially failed in a fresh process. Its installed 0.7.0 source attempts FFmpeg
7, 6, 5 and 4 only; current FFmpeg 8/9 builds are unsuitable for this pin.
Installed `Gyan.FFmpeg.Shared` version 7.1.1 at user scope with WinGet after this
failure. Its bin directory is:

```text
%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build-shared\bin
```

FFmpeg Shared is an OS-level dependency, separate from uv.lock. Discovery checks
PATH plus `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_*\*\bin`,
sorts candidates, verifies the DLL families and selects the first executable
reporting supported FFmpeg major 4–7. Thus it does not require a temporary PATH
mutation or a hard-coded username. Reproduce the user-scope WinGet installation
with the pinned command in README.md; the doctor reports the selected directory,
loaded library versions and functional decode PASS/FAIL. No administrator access,
arbitrary DLL copying or site-packages patching is required.
Closure verification also decoded the synthetic tone in a fresh process with
PATH empty, using WinGet discovery and DLL registration alone.

It provides avcodec-61, avformat-61, avutil-59, swresample-5 and swscale-8 DLLs.
Fresh probes register this directory with `os.add_dll_directory`, retaining
handles until import/decode finish. The loaded runtime reports
`7.1.1-full_build-www.gyan.dev`; libavcodec 61.19.101, libavformat 61.7.100,
libavutil 59.39.100 and libavfilter 10.4.100. The old installation was neither
overwritten nor uninstalled. Direct external Python imports must also register
the compatible DLL directory; PATH alone is insufficient proof.

Actual TorchCodec AudioDecoder decoding succeeds on a generated 0.1-second,
mono 16 kHz PCM16 tone: shape `[1, 1600]`, maximum sample error 0.0. Temporary
material is removed. No private source is used. Each required import runs in a
separate Python process. CPU runtime: CUDA version None, availability False,
device count 0. CTranslate2 CPU types: float32, int16, int8, int8_float32.

Installed WhisperX source/signatures confirm Persian `fa` maps to
`jonatasgrosman/wav2vec2-large-xlsr-53-persian`; its diarization wrapper defaults
to `pyannote/speaker-diarization-community-1`, supports exact `num_speakers`, and
speaker assignment defaults `fill_nearest=False`. None of these loaders is called.

The doctor reports software PASS and model-readiness WARN. HF_TOKEN is absent;
gated access, model revision pins, future RAM fit and throughput remain untested.
No current authorization to fetch separate models is implied by this readiness
report. Hugging Face, Whisper, Torch and project model caches are unchanged;
the pre-existing Whisper cache file (852,074,496 bytes) was preserved. The bundled
VAD checkpoint is an installed wheel asset, not a change in those model caches.

Primary dependency references:
[WhisperX 3.8.6 metadata](https://github.com/m-bain/whisperX/blob/v3.8.6/pyproject.toml),
[uv PyTorch index guidance](https://docs.astral.sh/uv/guides/integration/pytorch/).
