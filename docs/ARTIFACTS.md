# Artifact contracts

Canonical artifacts are UTF-8 JSON with sorted keys, deterministic compact
separators, preserved Unicode, and a terminating newline. Schema version `1`
is supported in Phase 3A. The integrity artifact contains pipeline version,
session ID, source hash/size/original filename/modified timestamp, structured
probe and decode results, failure codes, creation time, and its checksum. It
does not contain the absolute source path or raw diagnostic output.

`artifact_checksum` is SHA-256 over canonical JSON after removing only that
field. A changed field, including an operational timestamp, invalidates the
checksum. Artifacts are written through a same-directory temporary sibling,
flushed, fsynced, and atomically replaced. A prior valid artifact remains if a
replacement fails before the atomic step. Raw bounded FFmpeg diagnostics live
separately under ignored `work/` storage.

Stage fingerprints use source SHA-256, relevant semantic configuration, required
parent artifact checksums, pipeline version, stage/schema version, and future
model name/version/revision. Non-semantic creation timestamps are excluded.
Configuration changes only invalidate stages that use those settings and their
descendants. A checkpoint is reusable only when status is `SUCCEEDED`, expected
fingerprint matches, its artifact checksum verifies, the schema is supported,
and all required parent artifacts also validate and match recorded checksums.

Future final transcript JSON is canonical; Persian TXT is a derived view from
approved JSON plus acceptable QA and human-reviewed speaker mapping. Unknown
speaker identity remains unknown. Generated outputs remain outside Git.
