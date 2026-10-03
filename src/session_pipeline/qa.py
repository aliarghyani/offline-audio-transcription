"""Typed QA observations, separate from human review and resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class QAIssueCategory(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    DECODE_FAILED = "DECODE_FAILED"
    MISSING_AUDIO_STREAM = "MISSING_AUDIO_STREAM"
    ASR_INCOMPLETE = "ASR_INCOMPLETE"
    POSSIBLE_TRUNCATION = "POSSIBLE_TRUNCATION"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    ALIGNMENT_LOW_CONFIDENCE = "ALIGNMENT_LOW_CONFIDENCE"
    SPEAKER_UNASSIGNED = "SPEAKER_UNASSIGNED"
    SPEAKER_AMBIGUOUS = "SPEAKER_AMBIGUOUS"
    OVERLAPPING_SPEECH = "OVERLAPPING_SPEECH"
    SUSPICIOUS_GAP = "SUSPICIOUS_GAP"
    TAIL_COVERAGE_GAP = "TAIL_COVERAGE_GAP"


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKER = "BLOCKER"


class AutomaticStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class HumanReviewStatus(StrEnum):
    NOT_REVIEWED = "NOT_REVIEWED"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"
    NEEDS_FOLLOW_UP = "NEEDS_FOLLOW_UP"


class ResolutionStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    ACCEPTED_WITH_NOTE = "ACCEPTED_WITH_NOTE"


@dataclass(frozen=True)
class QAIssue:
    category: QAIssueCategory
    severity: Severity
    automatic_status: AutomaticStatus
    human_review_status: HumanReviewStatus = HumanReviewStatus.NOT_REVIEWED
    resolution_status: ResolutionStatus = ResolutionStatus.OPEN
    start_seconds: float | None = None
    end_seconds: float | None = None
    related_segment_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
