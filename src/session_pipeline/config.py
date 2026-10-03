"""Typed, validated non-secret configuration and environment-only secrets."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """The configuration does not satisfy the Phase 3A contract."""


def _keys(data: object, expected: set[str], section: str) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != expected:
        raise ConfigurationError(f"{section} must contain exactly: {', '.join(sorted(expected))}")
    return data


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigurationError(f"{name} must be a positive number or null")
    return float(value)


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer or null")
    return value


@dataclass(frozen=True)
class PathSettings:
    root: Path
    sessions_dir: str
    work_dir: str
    outputs_dir: str

    def __post_init__(self) -> None:
        for name in (self.sessions_dir, self.work_dir, self.outputs_dir):
            if not isinstance(name, str):
                raise ConfigurationError("application paths must be strings")
            candidate = Path(name)
            if (
                candidate.is_absolute()
                or candidate.drive
                or len(candidate.parts) != 1
                or name in {".", ".."}
            ):
                raise ConfigurationError(
                    "application paths must be single relative directory names"
                )

    @property
    def sessions_path(self) -> Path:
        return self.root / self.sessions_dir

    @property
    def work_path(self) -> Path:
        return self.root / self.work_dir

    def public_dict(self) -> dict[str, str]:
        return {
            "sessions_dir": self.sessions_dir,
            "work_dir": self.work_dir,
            "outputs_dir": self.outputs_dir,
        }


@dataclass(frozen=True)
class PipelineSettings:
    chunk_duration_seconds: float | None
    overlap_seconds: float | None
    batch_size: int | None
    worker_count: int | None

    def __post_init__(self) -> None:
        if self.overlap_seconds is not None and self.overlap_seconds < 0:
            raise ConfigurationError("overlap_seconds must be non-negative or null")
        if self.chunk_duration_seconds is not None and self.overlap_seconds is not None:
            if self.overlap_seconds >= self.chunk_duration_seconds:
                raise ConfigurationError("overlap must be shorter than chunk duration")


@dataclass(frozen=True)
class MLSettings:
    whisperx_version: str
    language: str
    asr_model: str
    device: str
    compute_type: str
    alignment_model: str
    diarization_model: str
    num_speakers: int
    fill_nearest: bool

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str)
            for value in (
                self.whisperx_version, self.language, self.asr_model, self.device,
                self.compute_type, self.alignment_model, self.diarization_model,
            )
        ):
            raise ConfigurationError("ML model settings must be strings")
        if type(self.num_speakers) is not int or type(self.fill_nearest) is not bool:
            raise ConfigurationError("speaker count and fill_nearest must have exact types")
        required = ("3.8.6", "fa", "large-v3", "cpu", "int8")
        actual = (
            self.whisperx_version,
            self.language,
            self.asr_model,
            self.device,
            self.compute_type,
        )
        if actual != required or self.num_speakers != 2 or self.fill_nearest is not False:
            raise ConfigurationError("ML settings must match the frozen Phase 3A architecture")
        if self.alignment_model != "jonatasgrosman/wav2vec2-large-xlsr-53-persian":
            raise ConfigurationError("unexpected Persian alignment model")
        if self.diarization_model != "pyannote/speaker-diarization-community-1":
            raise ConfigurationError("unexpected diarization model")


@dataclass(frozen=True)
class QASettings:
    alignment_confidence_threshold: float | None
    coverage_gap_seconds: float | None
    tail_gap_seconds: float | None


@dataclass(frozen=True, repr=False)
class SecretSettings:
    hf_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> SecretSettings:
        return cls(hf_token=os.environ.get("HF_TOKEN") or None)


@dataclass(frozen=True)
class AppConfig:
    application: PathSettings
    pipeline: PipelineSettings
    ml: MLSettings
    qa: QASettings
    secrets: SecretSettings = field(default_factory=SecretSettings, repr=False)

    def public_dict(self) -> dict[str, object]:
        return {
            "application": self.application.public_dict(),
            "pipeline": vars(self.pipeline),
            "ml": vars(self.ml),
            "qa": vars(self.qa),
        }


def load_config(root: Path | None = None) -> AppConfig:
    project_root = (root or Path(__file__).resolve().parents[2]).resolve()
    config_path = project_root / "config" / "defaults.json"
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot load configuration: {config_path}") from error
    data = _keys(document, {"application", "pipeline", "ml", "qa"}, "config")
    app = _keys(data["application"], {"sessions_dir", "work_dir", "outputs_dir"}, "application")
    pipe = _keys(
        data["pipeline"],
        {"chunk_duration_seconds", "overlap_seconds", "batch_size", "worker_count"},
        "pipeline",
    )
    ml = _keys(
        data["ml"],
        {
            "whisperx_version", "language", "asr_model", "device", "compute_type",
            "alignment_model", "diarization_model", "num_speakers", "fill_nearest",
        },
        "ml",
    )
    qa = _keys(
        data["qa"],
        {"alignment_confidence_threshold", "coverage_gap_seconds", "tail_gap_seconds"},
        "qa",
    )
    overlap = pipe["overlap_seconds"]
    if overlap is not None:
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, (int, float))
            or not math.isfinite(overlap)
            or overlap < 0
        ):
            raise ConfigurationError("overlap_seconds must be non-negative or null")
        overlap = float(overlap)
    return AppConfig(
        application=PathSettings(root=project_root, **app),
        pipeline=PipelineSettings(
            chunk_duration_seconds=_optional_number(
                pipe["chunk_duration_seconds"], "chunk duration"
            ),
            overlap_seconds=overlap,
            batch_size=_optional_int(pipe["batch_size"], "batch size"),
            worker_count=_optional_int(pipe["worker_count"], "worker count"),
        ),
        ml=MLSettings(**ml),
        qa=QASettings(
            alignment_confidence_threshold=_optional_number(
                qa["alignment_confidence_threshold"], "alignment confidence threshold"
            ),
            coverage_gap_seconds=_optional_number(qa["coverage_gap_seconds"], "coverage gap"),
            tail_gap_seconds=_optional_number(qa["tail_gap_seconds"], "tail gap"),
        ),
        secrets=SecretSettings.from_environment(),
    )
