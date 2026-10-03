"""Discover audio candidates only beneath the private sessions directory."""

from __future__ import annotations

from pathlib import Path

from session_pipeline.models import SessionSource, build_session_source

SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".m4a", ".mp4", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}
)


def discover_sessions(sessions_dir: Path) -> list[SessionSource]:
    if not sessions_dir.is_dir():
        return []
    boundary = sessions_dir.resolve(strict=True)
    sources: list[SessionSource] = []
    for candidate in sorted(boundary.rglob("*"), key=lambda path: str(path).casefold()):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if not candidate.resolve(strict=True).is_relative_to(boundary):
            continue
        sources.append(build_session_source(candidate))
    return sources
