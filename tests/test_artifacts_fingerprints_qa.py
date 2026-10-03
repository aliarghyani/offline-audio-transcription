from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from session_pipeline.artifacts import (
    ArtifactError,
    Checkpoint,
    atomic_write_json,
    canonical_json_bytes,
    checkpoint_reusable,
    read_artifact,
    seal_artifact,
)
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.models import StageStatus
from session_pipeline.qa import (
    AutomaticStatus,
    HumanReviewStatus,
    QAIssue,
    QAIssueCategory,
    ResolutionStatus,
    Severity,
)

SOURCE_HASH = "a" * 64
OTHER_HASH = "b" * 64


def test_canonical_json_unicode_and_order() -> None:
    assert canonical_json_bytes({"z": "فارسی", "a": 1}) == (
        '{"a":1,"z":"فارسی"}\n'.encode()
    )


def test_atomic_write_checksum_corruption_and_schema(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    sealed = seal_artifact({"schema_version": 1, "status": StageStatus.SUCCEEDED, "text": "نمونه"})
    atomic_write_json(target, sealed)
    assert read_artifact(target) == sealed
    assert list(tmp_path.glob("*.tmp")) == []
    target.write_bytes(target.read_bytes().replace("نمونه".encode(), "خراب".encode()))
    with pytest.raises(ArtifactError, match="checksum"):
        read_artifact(target)
    unsupported = seal_artifact({"schema_version": 99, "status": StageStatus.SUCCEEDED})
    atomic_write_json(target, unsupported)
    with pytest.raises(ArtifactError, match="schema"):
        read_artifact(target)


def test_atomic_write_failure_preserves_previous_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from session_pipeline import artifacts

    target = tmp_path / "artifact.json"
    previous = seal_artifact({"schema_version": 1, "status": StageStatus.SUCCEEDED, "n": 1})
    atomic_write_json(target, previous)
    prior_bytes = target.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_json(target, seal_artifact({"schema_version": 1, "n": 2}))
    assert target.read_bytes() == prior_bytes
    assert list(tmp_path.glob("*.tmp")) == []


def _fingerprint(**overrides) -> str:
    inputs = {
        "stage": "asr",
        "source_sha256": SOURCE_HASH,
        "relevant_config": {"language": "fa", "model": "large-v3"},
        "parent_checksums": {"prepare": OTHER_HASH},
        "pipeline_version": "0.1.0",
        "stage_schema_version": 1,
        "models": {"asr": {"name": "large-v3", "version": "1", "revision": "fixed"}},
        "operational_metadata": {"created_at": "first"},
    }
    inputs.update(overrides)
    return stage_fingerprint(**inputs)


def test_fingerprint_stability_and_invalidation() -> None:
    baseline = _fingerprint()
    assert baseline == _fingerprint(operational_metadata={"created_at": "later"})
    assert baseline == _fingerprint(
        relevant_config={"language": "fa", "model": "large-v3", "created_at": "later"}
    )
    assert baseline == _fingerprint(relevant_config={"model": "large-v3", "language": "fa"})
    assert baseline != _fingerprint(source_sha256=OTHER_HASH)
    assert baseline != _fingerprint(relevant_config={"language": "fa", "model": "medium"})
    assert baseline != _fingerprint(parent_checksums={"prepare": SOURCE_HASH})
    assert baseline != _fingerprint(models={"asr": {"name": "large-v3", "revision": "new"}})
    assert baseline == hashlib.sha256(
        canonical_json_bytes(
            {
                "stage": "asr",
                "source_sha256": SOURCE_HASH,
                "relevant_config": {"language": "fa", "model": "large-v3"},
                "parent_checksums": {"prepare": OTHER_HASH},
                "pipeline_version": "0.1.0",
                "stage_schema_version": 1,
                "models": {"asr": {"name": "large-v3", "version": "1", "revision": "fixed"}},
            }
        )
    ).hexdigest()


def test_checkpoint_reuse_requires_valid_child_and_parents(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.json"
    child_path = tmp_path / "child.json"
    parent = seal_artifact({"schema_version": 1, "status": StageStatus.SUCCEEDED})
    child = seal_artifact(
        {"schema_version": 1, "status": StageStatus.SUCCEEDED, "stage_fingerprint": "key"}
    )
    atomic_write_json(parent_path, parent)
    atomic_write_json(child_path, child)
    checkpoint = Checkpoint(
        status=StageStatus.SUCCEEDED,
        stage_fingerprint="key",
        artifact_checksum=child["artifact_checksum"],
        parent_checksums={"prepare": parent["artifact_checksum"]},
    )
    args = {
        "expected_fingerprint": "key",
        "artifact_path": child_path,
        "required_parents": {"prepare": parent_path},
    }
    assert checkpoint_reusable(checkpoint, **args)
    assert not checkpoint_reusable(checkpoint, **(args | {"expected_fingerprint": "changed"}))
    assert not checkpoint_reusable(
        Checkpoint(
            StageStatus.FAILED, "key", child["artifact_checksum"], checkpoint.parent_checksums
        ),
        **args,
    )
    changed_parent = seal_artifact({"schema_version": 1, "status": StageStatus.SUCCEEDED, "n": 2})
    atomic_write_json(parent_path, changed_parent)
    assert not checkpoint_reusable(checkpoint, **args)
    atomic_write_json(parent_path, parent)
    child_path.write_text("{}", encoding="utf-8")
    assert not checkpoint_reusable(checkpoint, **args)


def test_qa_serialization_keeps_auto_and_human_separate() -> None:
    issue = QAIssue(
        category=QAIssueCategory.SPEAKER_UNASSIGNED,
        severity=Severity.WARNING,
        automatic_status=AutomaticStatus.DETECTED,
    )
    value = issue.to_dict()
    assert value["category"] == "SPEAKER_UNASSIGNED"
    assert value["human_review_status"] == HumanReviewStatus.NOT_REVIEWED
    assert value["resolution_status"] == ResolutionStatus.OPEN
