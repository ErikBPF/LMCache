# SPDX-License-Identifier: Apache-2.0
"""Single-compute session switching for llama.cpp checkpoints."""

# Standard
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Any, cast
from urllib.request import Request, urlopen
import json
import os
import sqlite3

# Local
from .checkpoint_client import CheckpointClientError
from .checkpoint_client import request as checkpoint_request

_MAX_UPSTREAM_RESPONSE_BYTES = 32 << 20


def _cache_salt(principal: str, session_id: str) -> str:
    identity = f"{len(principal)}:{principal}{len(session_id)}:{session_id}"
    return sha256(identity.encode()).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, b'{"status":"ok"}')
        else:
            self._error(404, "not_found")

    def do_POST(self) -> None:
        server = cast(SessionBrokerHTTPServer, self.server)
        if self.path != "/v1/chat/completions":
            self._error(404, "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > server.max_request_bytes:
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            revision = int(self.headers.get("X-LMCache-Revision", ""))
            body, headers = server.broker.complete(
                self.headers.get("X-LMCache-Principal", ""),
                self.headers.get("X-LMCache-Session", ""),
                revision,
                payload,
            )
        except (json.JSONDecodeError, TypeError):
            self._error(400, "invalid_request")
            return
        except ValueError as exc:
            if str(exc) == "revision conflict":
                self._error(409, "revision_conflict")
            else:
                self._error(400, "invalid_request")
            return
        except OverflowError:
            self._error(429, "queue_full")
            return
        except OSError:
            self._error(502, "upstream")
            return
        self._respond(200, body, headers)

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _error(self, status: int, code: str) -> None:
        self._respond(status, json.dumps({"error": code}).encode())

    def _respond(
        self,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


class SessionBrokerHTTPServer(ThreadingHTTPServer):
    """HTTP transport for one ``SessionBroker`` instance."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        broker: "SessionBroker",
        max_request_bytes: int,
    ) -> None:
        if max_request_bytes < 2:
            raise ValueError("max_request_bytes must be at least 2")
        self.broker = broker
        self.max_request_bytes = max_request_bytes
        super().__init__(address, _Handler)


class SessionBroker:
    """Serialize session-aware chat requests through one llama.cpp slot.

    Args:
        database_path: Private SQLite file holding committed session revisions.
        checkpoint_socket: Private checkpoint-service Unix socket.
        llama_url: Trusted llama-server base URL.
        compatibility: Immutable model and engine checkpoint identity.
        timeout: Backend request timeout in seconds.
        max_queue_depth: Maximum number of waiting requests.

    Raises:
        ValueError: If configuration is invalid.
    """

    def __init__(
        self,
        database_path: Path,
        checkpoint_socket: Path,
        llama_url: str,
        compatibility: dict[str, Any],
        timeout: float,
        max_queue_depth: int = 4,
    ) -> None:
        if not llama_url.startswith(("http://", "https://")):
            raise ValueError("llama_url must use HTTP or HTTPS")
        if timeout <= 0 or max_queue_depth < 0 or not compatibility:
            raise ValueError("timeout, queue depth, and compatibility are invalid")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(database_path.parent, 0o700)
        self._database_path = database_path
        self._checkpoint_socket = checkpoint_socket
        self._llama_url = llama_url.rstrip("/")
        self._compatibility = deepcopy(compatibility)
        self._timeout = timeout
        self._admission = BoundedSemaphore(max_queue_depth + 1)
        self._compute = Lock()
        self._active: tuple[str, int] | None = None
        with self._connect() as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS session_heads ("
                "cache_salt TEXT PRIMARY KEY, revision INTEGER NOT NULL)"
            )
        os.chmod(database_path, 0o600)

    def complete(
        self,
        principal: str,
        session_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> tuple[bytes, dict[str, str]]:
        """Run one non-streaming completion and commit its next checkpoint.

        Args:
            principal: Authenticated caller identity supplied by a trusted proxy.
            session_id: Opaque caller-scoped session identifier.
            expected_revision: Last revision committed by the caller.
            payload: Canonical full-history OpenAI chat completion request.

        Returns:
            Upstream JSON bytes and LMCache timing/revision response headers.

        Raises:
            ValueError: If identity, revision, or payload validation fails.
            OverflowError: If the bounded queue is full.
            OSError: If llama-server cannot complete the request.
        """
        self._validate_request(principal, session_id, expected_revision, payload)
        if not self._admission.acquire(blocking=False):
            raise OverflowError("session queue is full")
        queued_at = monotonic()
        try:
            with self._compute:
                queue_wait_ms = (monotonic() - queued_at) * 1000
                return self._complete_locked(
                    principal,
                    session_id,
                    expected_revision,
                    payload,
                    queue_wait_ms,
                )
        finally:
            self._admission.release()

    def _complete_locked(
        self,
        principal: str,
        session_id: str,
        expected_revision: int,
        payload: dict[str, Any],
        queue_wait_ms: float,
    ) -> tuple[bytes, dict[str, str]]:
        cache_salt = _cache_salt(principal, session_id)
        with self._connect() as database:
            row = database.execute(
                "SELECT revision FROM session_heads WHERE cache_salt = ?",
                (cache_salt,),
            ).fetchone()
        current_revision = 0 if row is None else row[0]
        if current_revision != expected_revision:
            raise ValueError("revision conflict")

        switch_started = monotonic()
        cache_status = "cold"
        if expected_revision > 0:
            if self._active == (cache_salt, expected_revision):
                cache_status = "hit"
            else:
                cache_status = self._restore(cache_salt, expected_revision)
        switch_ms = (monotonic() - switch_started) * 1000

        upstream_payload = deepcopy(payload)
        upstream_payload["cache_prompt"] = cache_status == "hit"
        encoded = json.dumps(upstream_payload, separators=(",", ":")).encode()
        request = Request(
            f"{self._llama_url}/v1/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout) as response:
            body = response.read(_MAX_UPSTREAM_RESPONSE_BYTES + 1)
        if len(body) > _MAX_UPSTREAM_RESPONSE_BYTES:
            self._active = None
            raise OSError("llama-server response is too large")

        next_revision = expected_revision + 1
        store_status = self._store(cache_salt, next_revision)
        with self._connect() as database:
            database.execute(
                "INSERT INTO session_heads(cache_salt, revision) VALUES(?, ?) "
                "ON CONFLICT(cache_salt) DO UPDATE SET revision=excluded.revision",
                (cache_salt, next_revision),
            )
        self._active = (cache_salt, next_revision)
        if store_status != "stored":
            cache_status = f"{cache_status};store-failed"
        return body, {
            "X-LMCache-Cache": cache_status,
            "X-LMCache-Queue-Wait-Ms": f"{queue_wait_ms:.3f}",
            "X-LMCache-Switch-Ms": f"{switch_ms:.3f}",
            "X-LMCache-Revision": str(next_revision),
        }

    def _restore(self, cache_salt: str, revision: int) -> str:
        try:
            response = checkpoint_request(
                self._checkpoint_socket,
                {
                    "action": "restore",
                    "cache_salt": cache_salt,
                    "revision": revision,
                    "compatibility": self._compatibility,
                },
                timeout=self._timeout,
            )
        except (CheckpointClientError, OSError, TimeoutError):
            return "unavailable"
        return (
            "hit" if response.get("ok") is True and response.get("result") else "miss"
        )

    def _store(self, cache_salt: str, revision: int) -> str:
        try:
            response = checkpoint_request(
                self._checkpoint_socket,
                {
                    "action": "store",
                    "cache_salt": cache_salt,
                    "revision": revision,
                    "compatibility": self._compatibility,
                },
                timeout=self._timeout,
            )
        except (CheckpointClientError, OSError, TimeoutError):
            return "failed"
        return "stored" if response.get("ok") is True else "failed"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    @staticmethod
    def _validate_request(
        principal: str,
        session_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> None:
        if (
            not isinstance(principal, str)
            or not 0 < len(principal.encode()) <= 256
            or not isinstance(session_id, str)
            or not 0 < len(session_id.encode()) <= 256
            or not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 0
            or not isinstance(payload, dict)
            or not isinstance(payload.get("messages"), list)
            or payload.get("stream") is True
        ):
            raise ValueError("invalid session completion request")


def create_server(
    broker: SessionBroker,
    host: str,
    port: int,
    max_request_bytes: int,
) -> SessionBrokerHTTPServer:
    """Create the bounded HTTP transport for a session broker.

    Args:
        broker: Serialized session broker.
        host: Listen address.
        port: Listen port, or zero for an ephemeral test port.
        max_request_bytes: Maximum completion request body size.

    Returns:
        Bound HTTP server, not yet serving.

    Raises:
        ValueError: If the address or request limit is invalid.
    """
    if not host or not 0 <= port <= 65535:
        raise ValueError("host or port is invalid")
    return SessionBrokerHTTPServer((host, port), broker, max_request_bytes)


def main(argv: list[str] | None = None) -> int:
    """Run a non-streaming session broker until interrupted.

    Args:
        argv: Optional command-line arguments for tests and embedding.

    Returns:
        Process exit code.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--checkpoint-socket", type=Path, required=True)
    parser.add_argument("--llama-url", required=True)
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-queue-depth", type=int, default=4)
    parser.add_argument("--max-request-bytes", type=int, default=4 << 20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args(argv)
    try:
        compatibility = json.loads(args.compatibility)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid compatibility JSON: {exc.msg}")
    if not isinstance(compatibility, dict):
        parser.error("compatibility must be a JSON object")
    broker = SessionBroker(
        database_path=args.database,
        checkpoint_socket=args.checkpoint_socket,
        llama_url=args.llama_url,
        compatibility=compatibility,
        timeout=args.timeout,
        max_queue_depth=args.max_queue_depth,
    )
    server = create_server(
        broker,
        args.host,
        args.port,
        max_request_bytes=args.max_request_bytes,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
