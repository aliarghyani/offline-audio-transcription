"""Artifact adapter around the existing pinned local faster-whisper engine.

The engine/decode implementation is shared with targeted retry. Complete native
output is committed before canonical validation; partial progress is diagnostic
only and never represented as completed ASR.
"""

from __future__ import annotations

import gc

from session_pipeline import __version__
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_quality import load_canonical, session_paths
from session_pipeline.asr_retry import (
    DECODE,
    MODEL,
    deny_network,
    exclusive_lock,
    local_engine,
    package_identity,
    transcribe_clip,
)
from session_pipeline.fingerprint import stage_fingerprint
from session_pipeline.model_store import load_model_lock, local_only_environment
from session_pipeline.models import sha256_file
from session_pipeline.review_merge import write_text_atomic


def identity(config, integrity):
    spec = next(m for m in load_model_lock(config.application.root)["models"] if m["role"] == "asr")
    if {"repo_id": spec["repo_id"], "revision": spec["revision"]} != MODEL["asr"]:
        raise ValueError("ASR lock differs from approved local engine")
    settings = {"decode": DECODE, "packages": package_identity(), "model_lock": spec}
    return stage_fingerprint(
        stage="product_raw_asr",
        source_sha256=integrity["source"]["sha256"],
        relevant_config=settings,
        parent_checksums={"integrity": integrity["artifact_checksum"]},
        pipeline_version=__version__,
        stage_schema_version=1,
        models=MODEL,
    )


def execute(config, sid, *, engine_factory=local_engine, inference=transcribe_clip):
    path, work, _ = session_paths(config, sid)
    integrity = read_artifact(work / "integrity.json")
    if integrity["status"] != "SUCCEEDED" or not integrity["decode"]["success"]:
        raise ValueError("successful integrity required before ASR")
    source = config.application.sessions_path / integrity["source"]["original_filename"]
    if source.resolve().parent != config.application.sessions_path.resolve():
        raise ValueError("ASR source escapes private sessions")
    sha = integrity["source"]["sha256"]
    fp = identity(config, integrity)
    txt = path.with_suffix(".txt")
    with exclusive_lock(work / "asr/.write.lock"), deny_network():
        if sha256_file(source) != sha:
            raise ValueError("source changed before ASR")
        if path.exists():
            old = load_canonical(path, sid)
            if old.get("stage_fingerprint") != fp:
                raise ValueError(
                    "existing canonical ASR has different settings; use a new workspace"
                )
            if not txt.exists() or sha256_file(txt) != old["rendered_txt_sha256"]:
                raise ValueError("canonical ASR TXT changed")
            return {"state": "SUCCEEDED", "json_path": str(path), "txt_path": str(txt)}
        raw_path = work / "asr" / fp / "raw-inference.json"
        raw = read_artifact(raw_path) if raw_path.exists() else None
        if raw is not None and (
            raw["stage_fingerprint"] != fp
            or raw["source_sha256"] != sha
            or raw["parent_checksums"] != {"integrity": integrity["artifact_checksum"]}
        ):
            raise ValueError("native ASR lineage changed")
        if raw is None:

            def checkpoint(segments):
                if len(segments) % 25 == 0:
                    atomic_write_json(
                        raw_path.with_name("partial.json"),
                        seal_artifact(
                            {
                                "schema_version": 1,
                                "artifact_type": "asr_partial",
                                "status": "RUNNING",
                                "stage_fingerprint": fp,
                                "source_sha256": sha,
                                "segments": segments,
                            }
                        ),
                    )

            engine = None
            try:
                # Removes credentials, disables telemetry and forces local-only loaders.
                from unittest.mock import patch

                with patch.dict(
                    "os.environ", local_only_environment(config.application.root), clear=True
                ):
                    engine = engine_factory(config.application.root)
                    segments, diagnostics = inference(engine, source, 0.0, checkpoint=checkpoint)
                    raw = seal_artifact(
                        {
                            "schema_version": 1,
                            "artifact_type": "asr_native_output",
                            "status": "INFERENCE_COMPLETED",
                            "stage_fingerprint": fp,
                            "source_sha256": sha,
                            "parent_checksums": {"integrity": integrity["artifact_checksum"]},
                            "segments": segments,
                            "diagnostics": diagnostics,
                        }
                    )
                    atomic_write_json(raw_path, raw)
            finally:
                del engine
                gc.collect()
        if sha256_file(source) != sha:
            raise ValueError("source changed during ASR")
        text = "\n".join(s["text"] for s in raw["segments"]) + "\n"
        import hashlib

        document = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "asr",
                "status": "SUCCEEDED",
                "session_id": sid,
                "source_sha256": sha,
                "source": integrity["source"],
                "probe": integrity["probe"],
                "model": MODEL,
                "configuration": DECODE,
                "stage_fingerprint": fp,
                "parent_checksums": {"integrity": integrity["artifact_checksum"]},
                "segments": raw["segments"],
                "segment_count": len(raw["segments"]),
                "rendered_txt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "human_verified": False,
                "cloud_llm_required": False,
            }
        )
        # Validate using the existing canonical reader before publishing the output pair.
        validation_path = raw_path.with_name("validated.json")
        atomic_write_json(validation_path, document)
        load_canonical(validation_path, sid)
        write_text_atomic(txt, text)
        atomic_write_json(path, document)
        return {"state": "SUCCEEDED", "json_path": str(path), "txt_path": str(txt)}
