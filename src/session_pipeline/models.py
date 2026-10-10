"""Typed source identity and checkpoint state contracts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

SOURCE_SCHEMA_VERSION = 1
SESSION_ID_HEX_LENGTH = 20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SourceChangedError(RuntimeError):
    """The source changed while it was being read."""


class StageStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"
    SUCCEEDED_WITH_WARNINGS = "SUCCEEDED_WITH_WARNINGS"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    INTERRUPTED = "INTERRUPTED"
    SKIPPED = "SKIPPED"
    BLOCKED_MODEL_PROVISIONING = "BLOCKED_MODEL_PROVISIONING"


def utc_timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash with bounded memory and without changing the source file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_id_from_sha256(source_sha256: str) -> str:
    if not _SHA256.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be lowercase SHA-256 hex")
    return f"s-{source_sha256[:SESSION_ID_HEX_LENGTH]}"


@dataclass(frozen=True)
class SessionSource:
    schema_version: int
    session_id: str
    original_filename: str
    source_path: Path
    source_sha256: str
    size_bytes: int
    modified_timestamp: str
    discovered_at: str

    def to_dict(self, *, include_path: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "original_filename": self.original_filename,
            "source_sha256": self.source_sha256,
            "size_bytes": self.size_bytes,
            "modified_timestamp": self.modified_timestamp,
            "discovered_at": self.discovered_at,
        }
        if include_path:
            result["source_path"] = str(self.source_path)
        return result


def build_session_source(path: Path) -> SessionSource:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("source is not a file")
    before = resolved.stat()
    source_sha256 = sha256_file(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SourceChangedError("source changed while hashing")
    modified = datetime.fromtimestamp(after.st_mtime, UTC)
    return SessionSource(
        schema_version=SOURCE_SCHEMA_VERSION,
        session_id=session_id_from_sha256(source_sha256),
        original_filename=path.name,
        source_path=resolved,
        source_sha256=source_sha256,
        size_bytes=after.st_size,
        modified_timestamp=utc_timestamp(modified),
        discovered_at=utc_timestamp(),
    )
