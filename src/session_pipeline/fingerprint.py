"""Deterministic fingerprints for future resumable processing stages."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from session_pipeline.artifacts import canonical_json_bytes

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPERATIONAL_KEYS = frozenset(
    {"created_at", "discovered_at", "started_at", "finished_at", "modified_timestamp"}
)


def _valid_hash(value: str, name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _semantic(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _semantic(item) for key, item in value.items() if key not in _OPERATIONAL_KEYS}
    if isinstance(value, (list, tuple)):
        return [_semantic(item) for item in value]
    return value


def stage_fingerprint(
    *,
    stage: str,
    source_sha256: str,
    relevant_config: Mapping[str, Any],
    parent_checksums: Mapping[str, str],
    pipeline_version: str,
    stage_schema_version: int,
    models: Mapping[str, Mapping[str, str]] | None = None,
    operational_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash semantic inputs only. Operational timestamps are deliberately ignored."""
    del operational_metadata
    if not stage or not pipeline_version or stage_schema_version < 1:
        raise ValueError("stage, pipeline version, and positive schema version are required")
    _valid_hash(source_sha256, "source_sha256")
    for name, checksum in parent_checksums.items():
        _valid_hash(checksum, f"parent {name}")
    payload = {
        "stage": stage,
        "source_sha256": source_sha256,
        "relevant_config": _semantic(relevant_config),
        "parent_checksums": dict(parent_checksums),
        "pipeline_version": pipeline_version,
        "stage_schema_version": stage_schema_version,
        "models": _semantic(models or {}),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
