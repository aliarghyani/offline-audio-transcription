# Local ASR quality review and targeted retry (Phase 4A)

Codex is used for development only. The user starts and owns the local CLI process.
Transcription has no OpenAI, LLM API, cloud prompt or ChatGPT token dependency.
QA reads the completed, checksum-validated canonical raw JSON and available local
verification/chunk artifacts. TXT is not the machine source of truth. Historical
reference transcripts are never read. Review does not interpret session content.

## Commands

From the repository, using the existing environment:

```powershell
uv run --no-sync session-pipeline qa asr <session-id> --json
uv run --no-sync session-pipeline qa status <session-id> --json
uv run --no-sync session-pipeline qa retry <session-id> --json
```

`qa asr` creates or reuses `work/<session-id>/qa/asr-review.json`. It includes
each flagged interval, original text, start timestamp in absolute seconds, word
metadata, diagnostics, reasons, stable entry ID and padded suggested interval.
Console output contains counts/distributions and paths, without transcript text.
`qa status` reads the existing review and checks its canonical parent identity;
it does not load a model, contact the network or create artifacts. Status reflects
the stored thresholds; rerun `qa asr` after changing thresholds or verification.

`qa retry` is **plan only by default**. It saves a checksummed plan under
`work/<session-id>/qa/retries/<fingerprint>/plan.json`. Select individual review
entries with repeated `--entry <entry-id>` options; omission selects all entries.
The default decoded-audio budget is 900 seconds, configurable using
`--max-seconds` within (0, 1800]. Over-budget plans explain why execution is blocked.
Targets covering at least half the source or padded clips collectively covering
the entire source are blocked. No full-session fallback exists.

For a later, explicitly requested retry, the user can run:

```powershell
uv run --no-sync session-pipeline qa retry <session-id> --entry <entry-id> --execute
```

The foreground CLI launches a local Python worker, waits for it and returns its
exit code. It needs no Codex session, scheduler or cloud service. Keep the user's
terminal/process running. Repeat the same command after an interruption to reuse
completed window checkpoints. An OS file lock prevents duplicate session workers.
This phase does not execute a real Session 4 retry.

## Configurable signals

Defaults are in `config/asr-quality.json`; `qa asr --thresholds <json-path>` accepts
an alternate threshold object. Unspecified keys use defaults; unknown keys fail.
All numeric thresholds must be finite. These are triage signals, **not calibrated
confidence or error probabilities**. There is no aggregate confidence score.
Rules are independent and their counts overlap.

- `HIGH_TEMPERATURE`: temperature >= 0.6. Records stronger decoding fallback;
  it does not prove an error.
- `LOW_AVG_LOGPROB`: average log probability < -1.0. A practical review cutoff,
  not a language-specific validated certainty threshold.
- `LOW_WORD_PROBABILITY` (**experimental**): at least 2 available word probabilities
  < 0.20, comprising at least 20% of the segment's available probabilities.
  Isolated low words alone do not flag a whole segment.
- `SUSPICIOUS_REPETITION` (**experimental**): a normalized 5-word n-gram occurs
  at least 3 times within a segment, or consecutive identical normalized segment
  texts of at least 20 characters are at most 15 seconds apart. Both consecutive
  segments are marked. Repetition can be genuine speech.
- `SUSPICIOUS_SHORT_SEGMENT` (**experimental**): nonempty text with duration < 0.35
  seconds. Short interjections can be genuine.
- `SUSPICIOUS_GAP` (**experimental**): a transcript timeline gap >= 15 seconds
  containing >= 3 seconds of recorded VAD speech. Without VAD evidence, gaps >= 30
  seconds are flagged as uncertain review candidates, not asserted omissions.
  Overlapping VAD intervals are unioned to avoid counting context twice.
- `BOUNDARY_RISK` (**experimental**): a segment already has another signal and
  its span plus a 0.5-second margin intersects an internal chunk boundary.
- `POSSIBLE_HALLUCINATION` (**experimental**): nonempty text with no_speech_prob
  >= 0.8 AND avg_logprob < -1.0. No automatic text deletion follows.

Distributions report available/missing counts, minimum, linearly interpolated
p05/p95, median, maximum and mean when available. Missing diagnostics do not get
invented values. Repeated text group counts are descriptive; only the stricter
repetition rules create retry entries. Gap intervals include source head/tail.
Comparison uses Unicode NFKC/casefold/whitespace normalization only on temporary
comparison strings. Original review/candidate A strings are preserved exactly;
there is no grammar correction or transcript rewriting.

## Retry identity and candidate preservation

Target interval unions are split into at most 60-second owned windows, padded by
5 seconds per side and clamped to source duration. Configuration allows at most
90-second targets and 10-second padding. Decoded workload sums clip durations,
including duplicated context; the review also reports the smaller padded union.
Only derived mono 16 kHz PCM WAVs are decoded using FFmpeg. The original is never
a write target. Source size, modification timestamp and SHA-256 are checked, as
are canonical/parent checksums, before candidates are published.

The model is the same pinned `Systran/faster-whisper-large-v3` revision
`edaa852ec7e145841d8ffdb056a99866b5f0a478`. A matching canonical model identity and
verified local model payload are required. The worker sets offline environment
flags and disables socket connections; the loader uses `local_files_only=True`.
There is no provisioning or remote fallback from QA.

Retry uses forced `fa`, CPU, requested `int8` (actual `int8` or `int8_float32`),
8 CPU threads, 1 worker, beam/best-of 5, word timestamps, temperatures
0/0.2/0.4/0.6/0.8/1.0, previous-text conditioning, and no prompt/hotwords.
VAD is disabled for these short clips to avoid the earlier full-buffer VAD path;
context silence can still produce false positives and requires review.

Every separate `candidate-<index>.json` stores:

- A: exact original canonical segments intersecting the owned target.
- B: the full retry output with absolute timestamps, plus words selected by
  midpoint ownership in the half-open target interval.
- Diagnostics, window/model/config identity and all parent checksums.
- `review_required=true`, `canonical_replacement=false`, and an explicit empty-B
  indicator. A successful checkpoint means execution completed, not that B is better.

Window fingerprints include source hash, selected review, canonical/plan/integrity
parents, model revision, decode policy, package versions and intervals. Changed
identities do not reuse previous candidates. Atomic writes and sealed progress
preserve completed windows after failures. Corrupt checkpoints fail closed;
they are not silently accepted. Candidates are never merged into or substituted
for canonical JSON/TXT. Candidate review/acceptance is future work.

## Tests and limits

Tests use synthetic metadata, injected local inference and tiny generated audio;
they cover signal edges, deterministic manifests, interval ownership, budgets,
plan tampering, interrupted retry recovery, preservation and offline status.
Real model retry quality, speed and RAM usage have not been measured in Phase 4A.
Audio seconds are not a wall-clock estimate. Heuristics can miss errors and flag
correct speech; absence of flags is not a guarantee of transcription accuracy.
No alignment or diarization is part of this implementation.
