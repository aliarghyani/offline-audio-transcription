# Product orchestration contract

`transcribe <file|directory> --profile fast|quality|full` resumes by default.
`--dry-run` hashes/discovers inputs and reports the graph without writes,
integrity decoding, model inspection/loading, or network.

FAST: INGEST → INTEGRITY → ASR.
QUALITY adds ASR_QA → ASR_RETRY → REVIEW → ALIGNMENT.
FULL adds DIARIZATION → ATTRIBUTION → IDENTITY_MAPPING → FINAL_RENDER.

Adapters reuse stage functions. Initial ASR shares the existing local retry
engine/decode implementation. Models run sequentially; engine references are
released between stages. Whole-session global diarization and regular/exclusive
outputs remain engine-owned. No independently clustered diarization chunks.

## Receipts, resume and settings

`work/<session-id>/workflow.json` is an atomic sealed receipt with profile,
review policy, source hash, relevant settings/model/package fingerprints,
dependency file bindings, states, warnings, runtime and reuse. Each output's
SHA-256 is checked before reuse; JSON additionally needs a valid artifact checksum.
A success flag alone never proves completion. Every invocation hashes input.
Changing review policy reuses ASR but selects distinct reviewed/downstream lineages.

States: PENDING, RUNNING, SUCCEEDED, SUCCEEDED_WITH_WARNINGS, WAITING_FOR_USER,
FAILED, INTERRUPTED, SKIPPED, BLOCKED_MODEL_PROVISIONING. Failure/gate stops
descendants. Warnings propagate. Identity mapping is always rechecked so corrected
human confirmations produce a new final lineage.

The session lock is OS-released after crashes. Ctrl+C saves INTERRUPTED. Power
or terminal loss can leave RUNNING on disk; status probes the existing lock and
displays INTERRUPTED if no owner survives. Next invocation persists recovery and
delegates checkpoint resume. No age/PID reuse guess. Complete initial ASR native
output persists before validation/publication; partial diagnostics do not count
as completion and incomplete initial ASR can restart. Retry resumes per window,
alignment per unit. Diarization preserves its native-output-before-validation fix.

Changed/corrupt artifacts are never silently accepted or overwritten. Current
canonical raw ASR is immutable: deliberately changing its model/decode contract
requires a separate workspace. No destructive rebuild switch is exposed.
Defaults live in defaults.json, models.lock.json, asr-quality.json and
product_defaults.py; relevant settings enter workflow and engine fingerprints.

## Review policy and gates

auto-prefer-retry invokes existing AUTO_PREFER_B recovery/merge when retry targets
exist; provenance records automatic policy and human_verified=false. Conservative
retains A after bounded retry. Human uses existing A/B bundle/server/merge, waits
for resolved decisions, then creates a separate structurally accepted lineage
instead of modifying old PENDING merges. UNCLEAR keeps the gate pending. Unsafe
manual-boundary B choices fail clearly and must be resolved in review.
No targets means accepted raw text without a human-verification claim.

Retry considers entries in deterministic manifest order and admits each only if
the existing 900-second decoded budget, half-session and full-clip guards allow it.
Omitted entries remain A with warnings. No full-session retry escalation.
Alignment lexical failures remain PARTIAL with preserved words. QUALITY TXT uses
the aligned segment text without labels or rewriting.

Both gates print one `review <session-id> --open` command. The workflow supplies
retry/attribution fingerprints, even with multiple engineering lineages. The
loopback server preserves existing host/origin/token safeguards. Repeat the same
transcribe command after saving decisions/mapping. Speaker identity is never inferred.

## Status and batch

Status performs no model load/network/artifact write. It verifies current settings,
saved outputs and dependencies; FULL additionally validates current mapping and
exact final TXT. A unique final lineage created before this workflow is recognized
read-only from recorded ancestry; multiple historical finals are ambiguous rather
than selected by mtime. Historical runtime is reported unknown.

Batch discovery sorts recursive supported files, excludes symlinks and deduplicates
content. Sessions run sequentially, stop at the first gate/block/failure, and report
remaining inputs. External/nested audio receives an automatic private direct-child
copy to satisfy existing source contracts; supplied originals stay unchanged.

## Synthetic acceptance and private manual regression

Tests use generated zero PCM, sealed synthetic artifacts and injected inference.
They exercise actual raw publication, QA/review, alignment checkpoints, raw
diarization persistence, attribution, identity mapping and final rendering.
No real model or private recording/transcript is a fixture. Additional tests cover
planning, profiles, warnings, resume, stale RUNNING, interruptions, blocks, failures,
batch ordering, CLI/JSON, network denial, references and Git ignores.

Metadata-only Session 4 manual checklist (private files remain ignored):

- `status s-d57224782d66f69a7446` recognizes the existing unique completed FULL lineage.
- Final fingerprint: `9f424a2157989ef953fdad2f9c2a068f54990ad6ee0e27562ff03e334a1f25fc`.
- 2810 preserved segments, 17840 words; partial alignment accepted.
- Native diarization retained; two anonymous speakers, regular/exclusive turns.
- Confirmed human mapping; exact final JSON/TXT; unknown words explicit.
- SUCCEEDED_WITH_WARNINGS; 12 unresolved words, 99.9327% attribution.
- Compare every private upstream/final file hash and mtime with the baseline.
- Never run Session 4 ASR/alignment/diarization for this checklist.

New 1–2 hour throughput, peak memory and speech accuracy remain unmeasured by
synthetic acceptance. Speaker sample selection needs adequate non-overlapping
speech from each person; insufficient samples fail clearly.
