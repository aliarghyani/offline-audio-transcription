"""Local audio-only identity review; no text-based identity inference or defaults."""

from __future__ import annotations

import copy
import json
import re
import secrets
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from session_pipeline import attribution
from session_pipeline.artifacts import atomic_write_json, read_artifact, seal_artifact
from session_pipeline.asr_quality import session_paths, union
from session_pipeline.asr_retry import exclusive_lock, extract_clip
from session_pipeline.models import sha256_file, utc_timestamp

IDENTITIES = {"علی", "مشاور"}
SAMPLE_POLICY = {
    "version": 1,
    "minimum_seconds": 6,
    "maximum_seconds": 12,
    "boundary_margin_seconds": 0.15,
    "samples_per_speaker": 3,
    "selection": "NEAREST_SESSION_THIRD_CENTERS_AFTER_EXCLUSIONS",
}
require = attribution.require


def subtract(interval, exclusions):
    pieces = [list(interval)]
    for a, b in exclusions:
        updated = []
        for start, end in pieces:
            if end <= a or start >= b:
                updated.append([start, end])
            else:
                if start < a:
                    updated.append([start, a])
                if end > b:
                    updated.append([b, end])
        pieces = updated
    return pieces


def select_samples(diarization_evidence, duration):
    exclusive = diarization_evidence.get("exclusive_turns")
    require(bool(exclusive), "exclusive turns required for safe identity samples")
    regular = diarization_evidence["regular_turns"]
    overlaps = attribution.overlap_regions(regular)
    short = [[t["start"], t["end"]] for t in regular + exclusive if t["end"] - t["start"] < 0.1]
    excluded = union(overlaps + short)
    samples = []
    for speaker in attribution.SPEAKERS:
        candidates = []
        coverage = union([[t["start"], t["end"]] for t in regular if t["speaker"] == speaker])
        for index, t in enumerate(exclusive):
            if t["speaker"] != speaker:
                continue
            # Require matching regular evidence throughout the clip, as well as exclusivity.
            for a, b in coverage:
                interval = [max(a, t["start"]), min(b, t["end"])]
                if interval[1] <= interval[0]:
                    continue
                for start, end in subtract(interval, excluded):
                    start, end = start + 0.15, end - 0.15
                    if end - start >= 6:
                        candidates.append((start, end, index))
        used = []
        for third in range(3):
            target = duration * (third + 0.5) / 3
            possibilities = []
            for start, end, index in candidates:
                for a, b in subtract([start, end], used):
                    length = min(12, b - a)
                    if length < 6:
                        continue
                    left = min(max(target - length / 2, a), b - length)
                    possibilities.append(
                        (abs(left + length / 2 - target), left, left + length, index)
                    )
            require(bool(possibilities), f"three safe samples unavailable for {speaker}")
            _, start, end, index = min(possibilities)
            used.append([start, end])
            samples.append(
                {
                    "sample_id": f"{speaker}-{third + 1}",
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration_seconds": end - start,
                    "exclusive_turn_index": index,
                    "session_third_target": third + 1,
                    "overlap_free": True,
                    "short_turn_free": True,
                    "regular_same_speaker_coverage": True,
                }
            )
    return samples


def bundle_directory(config, sid, fingerprint):
    _, work, _ = session_paths(config, sid)
    return attribution.checked_path(work / "speakers" / "review", fingerprint, "bundle.json").parent


def clip_valid(path, sample):
    try:
        with wave.open(str(path), "rb") as stream:
            return (
                stream.getnchannels() == 1
                and stream.getsampwidth() == 2
                and stream.getframerate() == 16000
                and abs(stream.getnframes() / 16000 - sample["duration_seconds"]) <= 0.05
            )
    except (OSError, wave.Error):
        return False


def source_path(config, document):
    name = document["source"]["original_filename"]
    require(
        isinstance(name, str) and Path(name).name == name and "/" not in name and "\\" not in name,
        "invalid source filename",
    )
    path = (config.application.sessions_path / name).resolve()
    require(
        path.is_relative_to(config.application.sessions_path.resolve()), "source escapes sessions"
    )
    require(
        path.stat().st_size == document["source"]["size_bytes"]
        and sha256_file(path) == document["source_sha256"],
        "source audio changed",
    )
    return path


def build_bundle(config, sid, fingerprint=None):
    path, document = attribution.load_attributed(config, sid, fingerprint)
    directory = bundle_directory(config, sid, path.parent.name)
    diarized = read_artifact(
        config.application.root / document["parents"]["diarization"]["relative_path"]
    )
    samples = select_samples(document["diarization_evidence"], diarized["source_duration_seconds"])
    source = source_path(config, document)
    with exclusive_lock(directory / ".build.lock"):
        existing = (
            read_artifact(directory / "bundle.json")
            if (directory / "bundle.json").exists()
            else None
        )
        for sample in samples:
            clip = directory / (sample["sample_id"] + ".wav")
            if clip.exists():
                require(clip_valid(clip, sample), "invalid existing identity clip")
                if existing:
                    old = next(
                        s for s in existing["samples"] if s["sample_id"] == sample["sample_id"]
                    )
                    require(sha256_file(clip) == old["file_sha256"], "identity clip changed")
            else:
                extract_clip(source, clip, [sample["start"], sample["end"]])
            require(clip_valid(clip, sample), "identity clip duration/format invalid")
            sample.update(filename=clip.name, file_sha256=sha256_file(clip))
        require(
            sha256_file(source) == document["source_sha256"], "source changed during clip decode"
        )
        attribution.load_attributed(config, sid, path.parent.name)
        bundle = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "speaker_identity_review_bundle",
                "session_id": sid,
                "attribution_fingerprint": path.parent.name,
                "attribution_checksum": document["artifact_checksum"],
                "source_sha256": document["source_sha256"],
                "sample_policy": SAMPLE_POLICY,
                "samples": samples,
                "default_mapping": None,
                "human_choice_required": True,
            }
        )
        if existing:
            require(existing == bundle, "identity review lineage changed")
        else:
            atomic_write_json(directory / "bundle.json", bundle)
        assets = Path(__file__).parent / "speaker_review_assets"
        for name in ("index.html", "app.js", "style.css"):
            (directory / name).write_bytes((assets / name).read_bytes())
    return status(config, sid, path.parent.name)


def load_bundle(config, sid, fingerprint=None):
    path, document = attribution.load_attributed(config, sid, fingerprint)
    directory = bundle_directory(config, sid, path.parent.name)
    bundle = read_artifact(directory / "bundle.json")
    diarized = read_artifact(
        config.application.root / document["parents"]["diarization"]["relative_path"]
    )
    expected = select_samples(document["diarization_evidence"], diarized["source_duration_seconds"])
    require(
        bundle["artifact_type"] == "speaker_identity_review_bundle"
        and bundle["session_id"] == sid
        and bundle["attribution_fingerprint"] == path.parent.name
        and bundle["attribution_checksum"] == document["artifact_checksum"]
        and bundle["source_sha256"] == document["source_sha256"]
        and bundle["sample_policy"] == SAMPLE_POLICY
        and bundle["default_mapping"] is None
        and bundle["human_choice_required"] is True
        and len(bundle["samples"]) == len(expected),
        "stale identity bundle",
    )
    for sample, selected in zip(bundle["samples"], expected, strict=True):
        require(
            {k: sample[k] for k in selected} == selected
            and sample["filename"] == sample["sample_id"] + ".wav",
            "sample lineage changed",
        )
        clip = directory / sample["filename"]
        require(
            clip_valid(clip, sample) and sha256_file(clip) == sample["file_sha256"],
            "identity clip changed or missing",
        )
    return directory, bundle, document


def validate_mapping(mapping):
    require(
        isinstance(mapping, dict)
        and set(mapping) == set(attribution.SPEAKERS)
        and all(isinstance(v, str) for v in mapping.values())
        and set(mapping.values()) == IDENTITIES,
        "explicit bijection required: exactly one علی and one مشاور",
    )


def mapping_path(directory):
    return directory.parent.parent / "identity" / directory.name / "mapping.json"


def load_mapping(directory, bundle):
    path = mapping_path(directory)
    if not path.exists():
        return None
    mapping = read_artifact(path)
    require(
        mapping["artifact_type"] == "human_speaker_identity_mapping"
        and mapping["human_confirmed"] is True
        and mapping["identity_inference_used"] is False
        and mapping["session_id"] == bundle["session_id"]
        and mapping["bundle_checksum"] == bundle["artifact_checksum"]
        and mapping["attribution_checksum"] == bundle["attribution_checksum"]
        and mapping["attribution_fingerprint"] == bundle["attribution_fingerprint"],
        "identity mapping lineage changed",
    )
    validate_mapping(mapping["mapping"])
    return mapping


class MappingConflict(ValueError):
    """A concurrent human save must be reviewed before trying again."""


def save_mapping(config, sid, fingerprint, mapping, expected_checksum, *, human_confirmed):
    require(human_confirmed is True, "human confirmation required")
    validate_mapping(mapping)
    directory, bundle, _ = load_bundle(config, sid, fingerprint)
    path = mapping_path(directory)
    with exclusive_lock(path.parent / ".write.lock"):
        current = load_mapping(directory, bundle)
        if (current["artifact_checksum"] if current else None) != expected_checksum:
            raise MappingConflict("mapping changed; reload and confirm again")
        result = seal_artifact(
            {
                "schema_version": 1,
                "artifact_type": "human_speaker_identity_mapping",
                "session_id": sid,
                "attribution_fingerprint": bundle["attribution_fingerprint"],
                "attribution_checksum": bundle["attribution_checksum"],
                "bundle_checksum": bundle["artifact_checksum"],
                "mapping": copy.deepcopy(mapping),
                "human_confirmed": True,
                "identity_inference_used": False,
                "confirmed_at": utc_timestamp(),
                "revision": current["revision"] + 1 if current else 1,
                "previous_checksum": current["artifact_checksum"] if current else None,
            }
        )
        # Preserve past confirmations, including when a human corrects their mapping.
        if current:
            atomic_write_json(
                path.parent / "history" / (current["artifact_checksum"] + ".json"), current
            )
        atomic_write_json(path, result)
    return result


def status(config, sid, fingerprint=None):
    path, document = attribution.load_attributed(config, sid, fingerprint)
    directory = bundle_directory(config, sid, path.parent.name)
    built = (directory / "bundle.json").exists()
    mapping = None
    if built:
        _, bundle, _ = load_bundle(config, sid, path.parent.name)
        mapping = load_mapping(directory, bundle)
    return {
        "session_id": sid,
        "fingerprint": path.parent.name,
        "bundle_path": str(directory),
        "bundle_ready": built,
        "sample_count": 6 if built else 0,
        "identity_mapping_status": "HUMAN_CONFIRMED" if mapping else "HUMAN_MAPPING_REQUIRED",
        "mapping_path": str(mapping_path(directory)) if mapping else None,
        "qa": {
            **document["qa"],
            "identity_mapping_status": "HUMAN_CONFIRMED" if mapping else "HUMAN_MAPPING_REQUIRED",
        },
        "ml_inference_used": False,
        "final_transcript_created": False,
    }


def create_server(config, sid, fingerprint=None, port=8766):
    require(type(port) is int and 0 <= port <= 65535, "invalid port")
    directory, bundle, _ = load_bundle(config, sid, fingerprint)
    load_mapping(directory, bundle)
    fp = bundle["attribution_fingerprint"]
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass

        def host_valid(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def response_headers(self, code, kind, size):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; "
                "style-src 'self'; media-src 'self'; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )

        def respond(self, code, value):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.response_headers(code, "application/json; charset=utf-8", len(data))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.host_valid():
                self.respond(403, {"error": "loopback host required"})
                return
            try:
                if self.path == "/favicon.ico":
                    self.response_headers(204, "image/x-icon", 0)
                    self.end_headers()
                elif self.path in {"/", "/index.html", "/app.js", "/style.css"}:
                    name = "index.html" if self.path == "/" else self.path[1:]
                    data = (directory / name).read_bytes()
                    kind = {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "style.css": "text/css",
                    }[name]
                    self.response_headers(200, kind + "; charset=utf-8", len(data))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/api/state":
                    _, current, _ = load_bundle(config, sid, fp)
                    self.respond(
                        200,
                        {
                            "bundle": current,
                            "mapping": load_mapping(directory, current),
                            "save_token": token,
                        },
                    )
                elif re.fullmatch(r"/audio/SPEAKER_0[01]-[123]", self.path):
                    _, current, _ = load_bundle(config, sid, fp)
                    sample = next(
                        s
                        for s in current["samples"]
                        if s["sample_id"] == self.path.rsplit("/", 1)[1]
                    )
                    self.audio(directory / sample["filename"])
                else:
                    self.respond(404, {"error": "not found"})
            except (BrokenPipeError, ConnectionError):
                return  # Navigating away can close an in-flight local request.
            except (OSError, ValueError, KeyError, TypeError):
                self.respond(400, {"error": "local review validation failed"})

        def audio(self, path):
            size = path.stat().st_size
            start, end, code = 0, size - 1, 200
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if match and (match[1] or match[2]):
                    if match[1]:
                        start = int(match[1])
                        end = min(size - 1, int(match[2])) if match[2] else size - 1
                    else:
                        start = max(0, size - int(match[2]))
                    code = 206
                else:
                    start = size
                if not 0 <= start <= end < size:
                    self.response_headers(416, "audio/wav", 0)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.response_headers(code, "audio/wav", end - start + 1)
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                self.wfile.write(stream.read(end - start + 1))

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (
                not self.host_valid()
                or self.path != "/api/mapping"
                or self.headers.get("Origin") != origin
                or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), token)
            ):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if 0 < length <= 8192:
                        self.rfile.read(length)
                except (ValueError, OSError):
                    pass
                self.respond(403, {"error": "same-origin token-bearing request required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(
                    0 < length <= 8192 and self.headers.get("Content-Type") == "application/json",
                    "invalid mapping request",
                )
                body = json.loads(self.rfile.read(length))
                result = save_mapping(
                    config,
                    sid,
                    fp,
                    body["mapping"],
                    body["expected_checksum"],
                    human_confirmed=body["human_confirmed"],
                )
                self.respond(200, {"mapping": result})
            except MappingConflict:
                self.respond(409, {"error": "mapping changed; reload and confirm again"})
            except (BrokenPipeError, ConnectionError):
                return
            except (OSError, ValueError, KeyError, TypeError):
                self.respond(400, {"error": "explicit human bijection required; mapping not saved"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve_review(config, sid, fingerprint=None, port=8766, open_browser=False):
    server = create_server(config, sid, fingerprint, port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Local speaker review: {url}\nKeep this terminal running. Ctrl+C stops it.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
