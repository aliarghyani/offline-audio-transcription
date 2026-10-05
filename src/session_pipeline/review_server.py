"""Loopback-only, dependency-free human review UI. No external assets or analytics."""

from __future__ import annotations

import json
import re
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from session_pipeline.config import AppConfig
from session_pipeline.human_review import (
    DecisionConflict,
    RetryContext,
    clip_path,
    clip_valid,
    load_bundle,
    load_context,
    load_decisions,
    require,
    save_decision,
)
from session_pipeline.models import sha256_file


def create_server(
    config: AppConfig, sid: str, fingerprint: str | None = None, port: int = 8765
) -> ThreadingHTTPServer:
    require(type(port) is int and 0 <= port <= 65535, "port must be within [0, 65535]")
    context = load_context(config, sid, fingerprint)
    load_decisions(context, load_bundle(context))
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # Private paths and content are never written to HTTP access logs.

        def valid_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def response_headers(self, status: int, kind: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; "
                "script-src 'self'; style-src 'self'; media-src 'self'; img-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )

        def respond(self, status: int, value: Any) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.response_headers(status, "application/json; charset=utf-8", len(data))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if not self.valid_host():
                self.respond(403, {"error": "loopback host required"})
                return
            try:
                if self.path in {"/", "/index.html", "/app.js", "/style.css"}:
                    name = "index.html" if self.path == "/" else self.path[1:]
                    data = (context.directory / name).read_bytes()
                    kind = {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "style.css": "text/css",
                    }[name]
                    self.response_headers(200, kind + "; charset=utf-8", len(data))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/api/state":
                    current = load_context(config, sid, context.plan["stage_fingerprint"])
                    bundle = load_bundle(current)
                    from session_pipeline.retry_recovery import recovery_ui_state

                    bundle, decisions = recovery_ui_state(
                        current, bundle, load_decisions(current, bundle)
                    )
                    self.respond(
                        200,
                        {
                            "bundle": bundle,
                            "decisions": decisions,
                            "save_token": token,
                        },
                    )
                elif re.fullmatch(r"/audio/[0-9]+", self.path):
                    current = load_context(config, sid, context.plan["stage_fingerprint"])
                    self.audio(current, int(self.path.rsplit("/", 1)[1]))
                else:
                    self.respond(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.respond(
                    400,
                    {
                        "error": str(error)
                        if isinstance(error, ValueError)
                        else type(error).__name__
                    },
                )

        def audio(self, current: RetryContext, index: int) -> None:
            path = clip_path(current, index)
            bundle = read_audio_bundle(current)
            require(
                sha256_file(path) == bundle["audio_checksums"][str(index)], "audio clip changed"
            )
            require(clip_valid(path, current.candidates[index]["window"]), "audio clip unavailable")
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if match and (match[1] or match[2]):
                    if match[1]:
                        start = int(match[1])
                        end = min(size - 1, int(match[2])) if match[2] else size - 1
                    else:
                        start = max(0, size - int(match[2]))
                    status = 206
                else:
                    start = size
                if not 0 <= start <= end < size:
                    self.response_headers(416, "audio/wav", 0)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            with path.open("rb") as stream:
                self.response_headers(status, "audio/wav", end - start + 1)
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_POST(self) -> None:
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (
                not self.valid_host()
                or self.path != "/api/decision"
                or self.headers.get("Origin") != origin
                or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), token)
            ):
                # Drain small rejected bodies so Windows does not reset the socket
                # before the 403 response reaches the client. Never accept a decision.
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if 0 < length <= 8192:
                        self.rfile.read(length)
                except (ValueError, OSError):
                    pass
                self.respond(403, {"error": "same-origin review request required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                require(
                    0 < length <= 8192 and self.headers.get("Content-Type") == "application/json",
                    "invalid decision request",
                )
                body = json.loads(self.rfile.read(length))
                require(
                    not (context.directory / "context-recovery.json").exists(),
                    "recovered policy view is read-only; historical decisions are preserved",
                )
                result = save_decision(
                    config,
                    sid,
                    context.plan["stage_fingerprint"],
                    body["review_id"],
                    body["choice"],
                    body["expected_checksum"],
                )
                self.respond(200, {"decisions": result})
            except DecisionConflict as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.respond(
                    400,
                    {
                        "error": str(error)
                        if isinstance(error, ValueError)
                        else type(error).__name__
                    },
                )

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def read_audio_bundle(context: RetryContext) -> dict[str, Any]:
    # Validate current lineage as well as the clip hash before playback.
    return load_bundle(context)


def serve_review(
    config: AppConfig,
    sid: str,
    fingerprint: str | None = None,
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    server = create_server(config, sid, fingerprint, port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Local review: {url}\nKeep this terminal running. Ctrl+C stops the server.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
