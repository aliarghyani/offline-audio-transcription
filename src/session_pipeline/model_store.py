"""Pinned model file management. No model loading, audio access, or implicit downloads."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

APPROVED = {
    "asr": "Systran/faster-whisper-large-v3",
    "alignment": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "diarization": "pyannote/speaker-diarization-community-1",
}
REQUIRED = {
    "asr": {"model.bin", "config.json", "preprocessor_config.json", "tokenizer.json",
            "vocabulary.json"},
    "alignment": {"config.json", "pytorch_model.bin", "preprocessor_config.json",
                  "special_tokens_map.json", "vocab.json"},
    "diarization": {"config.yaml", "embedding/pytorch_model.bin",
                    "segmentation/pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"},
}
OPTIONAL = {"README.md", "embedding/README.md", "plda/README.md",
            "tokenizer_config.json", "added_tokens.json"}
STATES = {"MISSING", "PARTIAL", "PROVISIONED", "CORRUPT", "BLOCKED_AUTH",
          "BLOCKED_ACCESS_CONDITIONS"}


class ModelLockError(ValueError):
    """Non-secret model contract error."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def load_model_lock(root: Path) -> dict[str, Any]:
    try:
        lock = json.loads((root / "config/models.lock.json").read_text(encoding="utf-8"))
        if (set(lock) != {"schema_version", "states", "models"}
                or type(lock["schema_version"]) is not int or lock["schema_version"] != 1):
            raise ModelLockError("unsupported model lock schema")
        if set(lock["states"]) != STATES or len(lock["models"]) != len(APPROVED):
            raise ModelLockError("invalid model roles or states")
        roles = set()
        for spec in lock["models"]:
            if set(spec) != {"role", "repo_id", "revision", "expected_family", "license",
                             "destination", "gated", "tracking_ref", "files", "excluded_files"}:
                raise ModelLockError("unexpected model lock fields")
            role = spec["role"]
            if role not in APPROVED or role in roles or spec["repo_id"] != APPROVED[role]:
                raise ModelLockError("unapproved or duplicate model role")
            roles.add(role)
            if not _digest(spec["revision"], 40):
                raise ModelLockError("model revision must be an immutable 40-character commit SHA")
            if spec["destination"] != f"models/{role}/{spec['revision']}":
                raise ModelLockError("invalid logical model destination")
            if spec["gated"] is not (role == "diarization") or spec["tracking_ref"] != "main":
                raise ModelLockError("invalid access or update policy")
            if not all(isinstance(spec[key], str) and spec[key] for key in
                       ("license", "expected_family")):
                raise ModelLockError("missing model metadata")
            names = set()
            required = set()
            for file in spec["files"]:
                if set(file) != {"path", "required", "size_bytes", "sha256", "git_blob_sha1"}:
                    raise ModelLockError("unexpected file policy fields")
                name = file["path"]
                if name not in REQUIRED[role] | OPTIONAL or name in names:
                    raise ModelLockError("unapproved or duplicate model file")
                if (type(file["required"]) is not bool
                        or file["required"] != (name in REQUIRED[role])):
                    raise ModelLockError("invalid required file policy")
                if type(file["size_bytes"]) is not int or file["size_bytes"] <= 0:
                    raise ModelLockError("invalid expected file size")
                if file["sha256"] is not None and not _digest(file["sha256"], 64):
                    raise ModelLockError("invalid SHA-256")
                if file["git_blob_sha1"] is not None and not _digest(file["git_blob_sha1"], 40):
                    raise ModelLockError("invalid Git blob identity")
                if file["sha256"] is None and file["git_blob_sha1"] is None:
                    raise ModelLockError("file lacks a pinned content identity")
                names.add(name)
                if file["required"]:
                    required.add(name)
            if required != REQUIRED[role] or names & set(spec["excluded_files"]):
                raise ModelLockError("incomplete or conflicting file policy")
        return lock
    except ModelLockError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ModelLockError("cannot read valid model lock") from error


def model_path(root: Path, spec: dict[str, Any]) -> Path:
    root = root.resolve()
    destination = root.joinpath(*PurePosixPath(spec["destination"]).parts).resolve()
    if not destination.is_relative_to(root / "models"):
        raise ModelLockError("model destination escapes local storage")
    return destination


def _file_path(destination: Path, name: str) -> Path:
    path = destination.joinpath(*PurePosixPath(name).parts).resolve()
    if not path.is_relative_to(destination):
        raise ModelLockError("model file escapes its snapshot")
    return path


def _fingerprint(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _read_receipt(destination: Path) -> dict[str, Any]:
    try:
        result = json.loads((destination / ".provision.json").read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or not isinstance(result.get("files"), list)
                or any(not isinstance(file, dict) or not isinstance(file.get("path"), str)
                       for file in result["files"])):
            return {}
        return result
    except (OSError, ValueError):
        return {}


def check_file(path: Path, file: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    result = {"path": file["path"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size != file["size_bytes"]:
        return {**result, "valid": False, "reason": "SIZE_MISMATCH"}
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {stat.st_size}\0".encode())
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
            blob.update(data)
    result["sha256"] = digest.hexdigest()
    valid = (result["sha256"] == file["sha256"] if file["sha256"] else
             blob.hexdigest() == file["git_blob_sha1"])
    unchanged = path.stat().st_mtime_ns == stat.st_mtime_ns and path.stat().st_size == stat.st_size
    return {**result, "valid": valid and unchanged,
            "reason": "VERIFIED" if valid and unchanged else "CHECKSUM_MISMATCH"}


def validate_configs(destination: Path, spec: dict[str, Any]) -> list[str]:
    """Parse small config files only, without deserializing model checkpoints."""
    try:
        for file in spec["files"]:
            if file["path"].endswith(".json"):
                path = _file_path(destination, file["path"])
                if path.is_file() and not isinstance(
                    json.loads(path.read_text(encoding="utf-8")), (dict, list)
                ):
                    return ["INVALID_JSON_CONFIG"]
        if spec["role"] == "alignment":
            config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "wav2vec2":
                return ["UNEXPECTED_ALIGNMENT_MODEL_TYPE"]
        if spec["role"] == "diarization":
            import yaml

            try:
                config = yaml.safe_load((destination / "config.yaml").read_text(encoding="utf-8"))
            except yaml.YAMLError:
                return ["INVALID_MODEL_CONFIG"]
            if config["pipeline"]["name"] != "pyannote.audio.pipelines.SpeakerDiarization":
                return ["UNEXPECTED_DIARIZATION_PIPELINE"]
            parameters = config["pipeline"]["params"]
            if any(parameters.get(name) != f"$model/{name}" for name in
                   ("segmentation", "embedding", "plda")):
                return ["NONLOCAL_DIARIZATION_SUBORDINATE"]
        return []
    except (OSError, ValueError, KeyError, TypeError):
        return ["INVALID_MODEL_CONFIG"]


def inspect_model(root: Path, spec: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    destination = model_path(root, spec)
    receipt = _read_receipt(destination)
    recorded = {file["path"]: file for file in receipt.get("files", [])}
    files = []
    missing = []
    corrupt = []
    verified = True
    for file in spec["files"]:
        path = _file_path(destination, file["path"])
        if not path.is_file():
            if file["required"]:
                missing.append(file["path"])
            continue
        if full:
            info = check_file(path, file)
            if not info["valid"]:
                corrupt.append(file["path"])
        else:
            stat = path.stat()
            info = {"path": file["path"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            prior = recorded.get(file["path"], {})
            if stat.st_size != file["size_bytes"]:
                corrupt.append(file["path"])
            if (prior.get("bytes") != stat.st_size or prior.get("mtime_ns") != stat.st_mtime_ns
                    or prior.get("sha256") != file["sha256"]):
                verified = False
            info["sha256"] = prior.get("sha256")
        files.append(info)
    config_errors = (
        validate_configs(destination, spec) if full and not missing and not corrupt else []
    )
    if corrupt or config_errors:
        state = "CORRUPT"
    elif not missing and (full or (verified and receipt.get("lock_fingerprint") ==
                                  _fingerprint(spec) and receipt.get("state") == "PROVISIONED")):
        state = "PROVISIONED"
    elif missing and spec["gated"] and "HF_TOKEN" not in os.environ:
        state = "BLOCKED_AUTH"
    elif missing and receipt.get("state") in {"BLOCKED_AUTH", "BLOCKED_ACCESS_CONDITIONS"}:
        state = receipt["state"]
    else:
        state = "PARTIAL" if destination.exists() else "MISSING"
    return {"role": spec["role"], "repo_id": spec["repo_id"], "revision": spec["revision"],
            "local_relative_path": spec["destination"], "license": spec["license"],
            "status": state, "verification": "SHA256" if full else "RECEIPT_AND_FILE_METADATA",
            "access_readiness": ("local_files_verified" if state == "PROVISIONED" else
                                 "token_present_access_untested" if "HF_TOKEN" in os.environ else
                                 "token_absent") if spec["gated"] else "public",
            "provisioned_bytes": sum(f["bytes"] for f in files), "files": files,
            "missing_required": missing, "corrupt_files": corrupt, "config_errors": config_errors}


def _save_receipt(root: Path, spec: dict[str, Any], result: dict[str, Any]) -> None:
    write_json(model_path(root, spec) / ".provision.json", {
        "schema_version": 1, "repo_id": spec["repo_id"], "revision": spec["revision"],
        "lock_fingerprint": _fingerprint(spec), "state": result["status"], "files": result["files"],
    })


def safe_error_state(error: Exception) -> str:
    code = getattr(getattr(error, "response", None), "status_code", None)
    if type(error).__name__ == "GatedRepoError" or code == 403:
        return "BLOCKED_ACCESS_CONDITIONS"
    if code == 401:
        return "BLOCKED_AUTH"
    return "PARTIAL"


def _download(spec: dict[str, Any], file: dict[str, Any], root: Path,
              token: str | bool, *, force: bool) -> None:
    if file["size_bytes"] > 64 * 1024 * 1024:
        _download_ranges(spec, file, root, token)
        return
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=spec["repo_id"], filename=file["path"], revision=spec["revision"],
        local_dir=model_path(root, spec), cache_dir=root / "models/.hf-cache",
        token=token, force_download=force,
    )


def _download_ranges(spec: dict[str, Any], file: dict[str, Any], root: Path,
                     token: str | bool) -> None:
    """Bounded official Hub CDN transfers, with an ignored, revision-bound partial file."""
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(spec["repo_id"], file["path"], revision=spec["revision"])
    metadata = get_hf_file_metadata(url, token=token, timeout=30)
    if (metadata.commit_hash != spec["revision"] or metadata.size != file["size_bytes"]
            or metadata.etag != file["sha256"] or not metadata.location.startswith("https://")):
        raise ModelLockError("Hub artifact identity does not match the pinned lock")
    cache = (root / "models/.hf-cache" / spec["role"] / spec["revision"]).resolve()
    if not cache.is_relative_to(root.resolve() / "models"):
        raise ModelLockError("model cache escapes local storage")
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / f"{Path(file['path']).name}.{file['sha256']}.incomplete"
    total = file["size_bytes"]
    if partial.exists() and partial.stat().st_size >= total:
        if partial.stat().st_size == total and check_file(partial, file)["valid"]:
            destination = _file_path(model_path(root, spec), file["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, destination)
            return
        partial.write_bytes(b"")  # Only a corrupt ignored model partial, never source audio.
    offset = partial.stat().st_size if partial.exists() else 0
    while offset < total:
        end = min(offset + 16 * 1024 * 1024, total) - 1
        with requests.get(metadata.location, headers={"Range": f"bytes={offset}-{end}"},
                          stream=True, timeout=(30, 90)) as response:
            if (response.status_code != 206 or response.headers.get("Content-Range") !=
                    f"bytes {offset}-{end}/{total}"):
                raise ModelLockError("official Hub CDN did not honor the bounded range")
            received = 0
            with partial.open("ab") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if received + len(chunk) > end - offset + 1:
                        raise ModelLockError("unexpected range response size")
                    stream.write(chunk)
                    stream.flush()
                    received += len(chunk)
            if received != end - offset + 1:
                raise ModelLockError("incomplete range response; rerun to resume")
        offset = partial.stat().st_size
    if not check_file(partial, file)["valid"]:
        raise ModelLockError("downloaded model checksum mismatch")
    destination = _file_path(model_path(root, spec), file["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)


def _access_preflight(spec: dict[str, Any], token: str) -> None:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    get_hf_file_metadata(hf_hub_url(spec["repo_id"], "config.yaml", revision=spec["revision"]),
                         token=token)


def _pin_checksums(root: Path, role: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    lock = load_model_lock(root)
    spec = next(model for model in lock["models"] if model["role"] == role)
    hashes = {file["path"]: file["sha256"] for file in files if file.get("valid")}
    for file in spec["files"]:
        if file["sha256"] is None and file["path"] in hashes:
            file["sha256"] = hashes[file["path"]]
    write_json(root / "config/models.lock.json", lock)
    return spec


def provision_model(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or False
    if spec["gated"] and not token:
        result = inspect_model(root, spec, full=True)
        if result["status"] != "PROVISIONED":
            result["status"] = "BLOCKED_AUTH"
        _save_receipt(root, spec, result)
        return result
    try:
        if spec["gated"]:
            _access_preflight(spec, token)
        for file in spec["files"]:
            path = _file_path(model_path(root, spec), file["path"])
            exists = path.is_file()
            if exists and check_file(path, file)["valid"]:
                continue
            _download(spec, file, root, token if spec["gated"] else False, force=exists)
            if not check_file(path, file)["valid"]:
                result = inspect_model(root, spec, full=True)
                result["status"] = "CORRUPT"
                _save_receipt(root, spec, result)
                return result
        result = inspect_model(root, spec, full=True)
        if result["status"] == "PROVISIONED":
            spec = _pin_checksums(root, spec["role"], result["files"])
        _save_receipt(root, spec, result)
        return result
    except Exception as error:
        # Never publish raw Hub exception text, URLs, headers, or credentials.
        result = inspect_model(root, spec, full=True)
        result["status"] = safe_error_state(error)
        result["error_code"] = result["status"]
        _save_receipt(root, spec, result)
        return result


def models_report(root: Path, *, action: str = "status", role: str | None = None) -> dict[str, Any]:
    if action not in {"status", "verify", "provision"}:
        raise ModelLockError("unknown model operation")
    if action == "provision":
        from filelock import FileLock, Timeout

        storage = root / "models"
        storage.mkdir(parents=True, exist_ok=True)
        (storage / ".hf-cache").mkdir(exist_ok=True)
        # Serialize downloads and tracked checksum updates; partial Hub files are retained.
        try:
            with FileLock(storage / ".provision.lock", timeout=0):
                return _models_report(root, action=action, role=role)
        except Timeout as error:
            raise ModelLockError(
                "model provisioning already active; retry after it finishes"
            ) from error
    return _models_report(root, action=action, role=role)


def _models_report(root: Path, *, action: str, role: str | None) -> dict[str, Any]:
    lock = load_model_lock(root)
    selected = [model for model in lock["models"] if role is None or model["role"] == role]
    if not selected:
        raise ModelLockError("unknown model role")
    results = []
    for spec in selected:
        if action == "provision":
            result = provision_model(root, spec)
        else:
            result = inspect_model(root, spec, full=action == "verify")
            if action == "verify" and result["status"] == "PROVISIONED":
                _save_receipt(root, spec, result)
        results.append(result)
    return {"schema_version": 1, "operation": action,
            "models_ready": all(result["status"] == "PROVISIONED" for result in results),
            "hf_token_present": "HF_TOKEN" in os.environ, "models_loaded": False, "models": results}


def local_model_path(root: Path, role: str) -> Path:
    """Future loaders must call this full offline guard and use local-files-only options."""
    lock = load_model_lock(root)
    spec = next((model for model in lock["models"] if model["role"] == role), None)
    if spec is None or inspect_model(root, spec, full=True)["status"] != "PROVISIONED":
        raise ModelLockError("pinned local model unavailable; explicit provisioning required")
    return model_path(root, spec)


def local_only_environment(root: Path) -> dict[str, str]:
    """Future inference processes use this environment plus guarded local paths."""
    from session_pipeline.runtime import offline_environment

    environment = offline_environment()
    cache = root.resolve() / "models/.hf-cache"
    environment.update({"HF_HOME": str(cache), "HF_HUB_CACHE": str(cache / "hub")})
    return environment


def check_remote_revisions(root: Path) -> dict[str, Any]:
    """Explicit metadata-only network check; never changes the lock or fetches files."""
    from huggingface_hub import HfApi

    models = []
    for spec in load_model_lock(root)["models"]:
        try:
            info = HfApi(token=False).model_info(spec["repo_id"], revision="main", token=False)
            models.append({"role": spec["role"], "pinned_revision": spec["revision"],
                           "remote_revision": info.sha, "changed": info.sha != spec["revision"]})
        except Exception:
            models.append({"role": spec["role"], "pinned_revision": spec["revision"],
                           "remote_revision": "UNKNOWN", "changed": None})
    return {"models": models, "lock_updated": False, "model_files_downloaded": False}


def human_models_report(report: dict[str, Any]) -> str:
    lines = [f"Models ready: {report['models_ready']}; models loaded: False"]
    for model in report["models"]:
        lines.append(f"{model['role']}: {model['status']} — {model['repo_id']}@{model['revision']}")
        lines.append(f"  {model['local_relative_path']} ({model['provisioned_bytes']} bytes)")
        if model["missing_required"]:
            lines.append("  Missing: " + ", ".join(model["missing_required"]))
    return "\n".join(lines)
