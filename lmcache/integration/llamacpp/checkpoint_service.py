# SPDX-License-Identifier: Apache-2.0
"""Private Unix-socket control service for llama.cpp checkpoints."""

# Standard
from pathlib import Path
from socketserver import StreamRequestHandler, UnixStreamServer
from typing import Any, cast
import json
import os
import stat

# Local
from .checkpoint_bridge import CheckpointBridge, CheckpointBridgeError
from .checkpoint_store import CheckpointCorruptError


class _RequestError(ValueError):
    pass


class _Handler(StreamRequestHandler):
    def handle(self) -> None:
        service = cast(CheckpointService, self.server)
        request = self.rfile.readline(service.max_request_bytes + 1)
        if len(request) > service.max_request_bytes or not request.endswith(b"\n"):
            service.write_response(self.wfile, service.error("invalid_request"))
            return
        service.write_response(self.wfile, service.dispatch(request))


class CheckpointService(UnixStreamServer):
    """Serve serialized checkpoint operations on a private Unix socket.

    Args:
        bridge: Checkpoint bridge used for store and restore operations.
        socket_path: New Unix socket path in a trusted local directory.
        max_request_bytes: Maximum JSON request frame size including newline.

    Raises:
        ValueError: If the request limit is invalid or the socket path exists.
    """

    def __init__(
        self,
        bridge: CheckpointBridge,
        socket_path: Path,
        max_request_bytes: int,
    ) -> None:
        if max_request_bytes < 2:
            raise ValueError("max_request_bytes must be at least 2")
        if socket_path.exists() or socket_path.is_symlink():
            raise ValueError("socket_path must not exist")
        self.bridge = bridge
        self.socket_path = socket_path
        self.max_request_bytes = max_request_bytes
        self.stats = {"errors": 0, "misses": 0, "restores": 0, "stores": 0}
        super().__init__(str(socket_path), _Handler)
        os.chmod(socket_path, 0o600)

    def dispatch(self, request: bytes) -> dict[str, Any]:
        """Validate and execute one JSON request.

        Args:
            request: One newline-terminated JSON object.

        Returns:
            JSON-compatible success or stable error response.
        """
        try:
            payload = json.loads(request)
            action = self._validate(payload)
            if action == "stats":
                return {"ok": True, "result": dict(self.stats)}
            result: dict[str, Any] | None
            if action == "store":
                result = self.bridge.store(
                    payload["cache_salt"],
                    payload["revision"],
                    payload["compatibility"],
                )
                self.stats["stores"] += 1
            else:
                result = self.bridge.restore(
                    payload["cache_salt"],
                    payload["revision"],
                    payload["compatibility"],
                )
                self.stats["restores"] += 1
                if result is None:
                    self.stats["misses"] += 1
            return {"ok": True, "result": result}
        except (json.JSONDecodeError, UnicodeDecodeError, _RequestError):
            return self.error("invalid_request")
        except CheckpointCorruptError:
            return self.error("corrupt")
        except TimeoutError:
            return self.error("timeout")
        except CheckpointBridgeError:
            return self.error("backend")
        except ValueError:
            return self.error("conflict")

    def error(self, code: str) -> dict[str, Any]:
        """Return a content-free error response and increment its counter.

        Args:
            code: Stable machine-readable error code.

        Returns:
            JSON-compatible error response.
        """
        self.stats["errors"] += 1
        return {"ok": False, "error": code}

    def write_response(self, stream: Any, response: dict[str, Any]) -> None:
        """Write one bounded JSON response.

        Args:
            stream: Socket file used by ``StreamRequestHandler``.
            response: JSON-compatible response object.
        """
        encoded = json.dumps(response, separators=(",", ":"), sort_keys=True).encode()
        stream.write(encoded + b"\n")

    def server_close(self) -> None:
        """Close the server and remove only its Unix socket."""
        super().server_close()
        try:
            mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(mode):
            self.socket_path.unlink()

    @staticmethod
    def _validate(payload: Any) -> str:
        if not isinstance(payload, dict):
            raise _RequestError
        action = payload.get("action")
        if action == "stats":
            if set(payload) != {"action"}:
                raise _RequestError
            return action
        if action not in {"store", "restore"} or set(payload) != {
            "action",
            "cache_salt",
            "revision",
            "compatibility",
        }:
            raise _RequestError
        cache_salt = payload["cache_salt"]
        revision = payload["revision"]
        if (
            not isinstance(cache_salt, str)
            or len(cache_salt) != 64
            or cache_salt.lower() != cache_salt
            or any(char not in "0123456789abcdef" for char in cache_salt)
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
            or not isinstance(payload["compatibility"], dict)
        ):
            raise _RequestError
        return action
