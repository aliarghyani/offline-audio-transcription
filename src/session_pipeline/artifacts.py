"""Canonical UTF-8 JSON, checksums, atomic writes, and checkpoint validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from session_pipeline import __version__
from session_pipeline.models import StageStatus, utc_timestamp

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactError(ValueError):
    """An artifact is malformed, unsupported, or altered."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def artifact_checksum(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "artifact_checksum"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_artifact(document: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["artifact_checksum"] = artifact_checksum(sealed)
    return sealed


def validate_artifact(document: object) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ArtifactError("artifact must be a JSON object")
    if document.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("unsupported artifact schema version")
    if document.get("artifact_checksum") != artifact_checksum(document):
        raise ArtifactError("artifact checksum mismatch")
    return document


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    """Write a complete sibling temp file, fsync it, then replace atomically."""
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_artifact(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError("artifact cannot be read as UTF-8 JSON") from error
    return validate_artifact(document)


def integrity_artifact(result: Any, stage_fingerprint: str) -> dict[str, Any]:
    if result.source is None:
        raise ArtifactError("integrity artifact requires an identified source")
    source = result.source
    return seal_artifact(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "integrity",
            "pipeline_version": __version__,
            "stage_fingerprint": stage_fingerprint,
            "status": StageStatus.SUCCEEDED if result.success else StageStatus.FAILED,
            "session_id": source.session_id,
            "source": {
                "sha256": source.source_sha256,
                "size_bytes": source.size_bytes,
                "original_filename": source.original_filename,
                "modified_timestamp": source.modified_timestamp,
            },
            "probe": result.probe,
            "decode": result.decode,
            "failure_codes": [code.value for code in result.failure_codes],
            "created_at": utc_timestamp(),
        }
    )


@dataclass(frozen=True)
class Checkpoint:
    status: StageStatus
    stage_fingerprint: str
    artifact_checksum: str
    parent_checksums: dict[str, str]


def checkpoint_reusable(
    checkpoint: Checkpoint,
    *,
    expected_fingerprint: str,
    artifact_path: Path,
    required_parents: dict[str, Path],
) -> bool:
    if checkpoint.status != StageStatus.SUCCEEDED:
        return False
    if checkpoint.stage_fingerprint != expected_fingerprint:
        return False
    if set(checkpoint.parent_checksums) != set(required_parents):
        return False
    try:
        artifact = read_artifact(artifact_path)
        if artifact.get("status") != StageStatus.SUCCEEDED:
            return False
        if artifact.get("stage_fingerprint") != expected_fingerprint:
            return False
        if artifact["artifact_checksum"] != checkpoint.artifact_checksum:
            return False
        for name, parent_path in required_parents.items():
            parent = read_artifact(parent_path)
            if parent.get("status") != StageStatus.SUCCEEDED:
                return False
            if parent["artifact_checksum"] != checkpoint.parent_checksums[name]:
                return False
    except (ArtifactError, KeyError):
        return False
    return True
