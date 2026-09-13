# SPDX-License-Identifier: Apache-2.0
"""Private Unix-socket control service for llama.cpp checkpoints."""

# Standard
from pathlib import Path
from socketserver import StreamRequestHandler, UnixStreamServer
from typing import Any, cast
import array
import json
import mmap
import os
import socket
import stat

# First Party
from lmcache.v1.multiprocess.posix_shm import shm_open_readonly_fd

# Local
from .checkpoint_store import CheckpointCorruptError, CheckpointStore


class _RequestError(ValueError):
    pass


class _Handler(StreamRequestHandler):
    def handle(self) -> None:
        service = cast(CheckpointService, self.server)
        self.request.settimeout(service.io_timeout)
        try:
            request = self.rfile.readline(service.max_request_bytes + 1)
        except TimeoutError:
            service.write_response(self.wfile, service.error("timeout"))
            return
        if len(request) > service.max_request_bytes or not request.endswith(b"\n"):
            service.write_response(self.wfile, service.error("invalid_request"))
            return
        service.serve_request(request, self.rfile, self.wfile, self.request)


class CheckpointService(UnixStreamServer):
    """Serve serialized checkpoint operations on a private Unix socket.

    Args:
        checkpoint_store: Opaque checkpoint store used for store and restore operations.
        socket_path: New Unix socket path in a trusted local directory.
        max_request_bytes: Maximum JSON request frame size including newline.
        max_checkpoint_bytes: Maximum binary checkpoint frame size.
        l1_shm_name: POSIX shared-memory name backing LMCache L1, if enabled.
        io_timeout: Maximum seconds spent on one socket transfer stage.
        owner_fd: Optional runtime lock descriptor owned by this service.

    Raises:
        ValueError: If the request limit is invalid or the socket path exists.
    """

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        socket_path: Path,
        max_request_bytes: int,
        max_checkpoint_bytes: int = 64 << 30,
        l1_shm_name: str = "",
        io_timeout: float = 30.0,
        owner_fd: int | None = None,
    ) -> None:
        if max_request_bytes < 2 or max_checkpoint_bytes < 0 or io_timeout <= 0:
            raise ValueError("request limits are invalid")
        if socket_path.exists() or socket_path.is_symlink():
            raise ValueError("socket_path must not exist")
        self.checkpoint_store = checkpoint_store
        self.socket_path = socket_path
        self.max_request_bytes = max_request_bytes
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.l1_shm_name = l1_shm_name
        self.io_timeout = io_timeout
        self._owner_fd = owner_fd
        self.stats = {"errors": 0, "misses": 0, "restores": 0, "stores": 0}
        super().__init__(str(socket_path), _Handler)
        os.chmod(socket_path, 0o600)

    def serve_request(
        self,
        request: bytes,
        input_stream: Any,
        output_stream: Any,
        connection: socket.socket | None = None,
    ) -> None:
        """Validate, execute, and write one JSON request.

        Args:
            request: One newline-terminated JSON object.
            input_stream: Socket stream containing an optional checkpoint frame.
            output_stream: Socket stream receiving the response frame.
            connection: Raw socket used only for descriptor passing.
        """
        try:
            payload = json.loads(request)
            action = self._validate(payload)
            if action == "stats":
                response = {"ok": True, "result": dict(self.stats)}
            elif action == "store":
                timings: dict[str, float] = {}
                if payload.get("transport") == "fd":
                    if connection is None:
                        raise _RequestError
                    self.write_response(output_stream, {"ok": True, "ready": True})
                    output_stream.flush()
                    self._store_from_fd(payload, connection, timings)
                else:
                    self.checkpoint_store.store_from_reader(
                        payload["cache_salt"],
                        payload["revision"],
                        payload["length"],
                        input_stream.read,
                        payload["compatibility"],
                        timings,
                    )
                self.stats["stores"] += 1
                response = {"ok": True, "timings_ms": timings}
            else:
                timings = {}
                read_checkpoint = (
                    self.checkpoint_store.read_checkpoint_regions
                    if payload.get("transport") == "fd" and self.l1_shm_name
                    else self.checkpoint_store.read_checkpoint
                )
                with read_checkpoint(
                    payload["cache_salt"],
                    payload["revision"],
                    payload["compatibility"],
                    timings,
                ) as restored:
                    self.stats["restores"] += 1
                    if restored is None:
                        self.stats["misses"] += 1
                        response = {"found": False, "ok": True}
                    else:
                        length, chunks = restored[:2]
                        regions = restored[2] if len(restored) == 3 else None
                        if regions is not None:
                            if connection is None:
                                raise _RequestError
                            self._restore_from_fd(
                                connection,
                                output_stream,
                                length,
                                regions,
                                timings,
                            )
                            return
                        self.write_response(
                            output_stream,
                            {
                                "found": True,
                                "length": length,
                                "ok": True,
                                "timings_ms": timings,
                            },
                        )
                        for chunk in chunks:
                            output_stream.write(chunk)
                        return
        except (EOFError, json.JSONDecodeError, UnicodeDecodeError, _RequestError):
            response = self.error("invalid_request")
        except CheckpointCorruptError:
            response = self.error("corrupt")
        except TimeoutError:
            response = self.error("timeout")
        except ValueError:
            response = self.error("conflict")
        self.write_response(output_stream, response)

    def _store_from_fd(
        self,
        payload: dict[str, Any],
        connection: socket.socket,
        timings: dict[str, float],
    ) -> None:
        checkpoint_fd = self._receive_fd(connection)
        try:
            length = payload["length"]
            if os.fstat(checkpoint_fd).st_size < length:
                raise _RequestError
            if length == 0:
                reader = lambda _size: b""
                self.checkpoint_store.store_from_reader(
                    payload["cache_salt"],
                    payload["revision"],
                    length,
                    reader,
                    payload["compatibility"],
                    timings,
                )
                return
            with mmap.mmap(checkpoint_fd, length, access=mmap.ACCESS_COPY) as mapped:
                view = memoryview(mapped)
                offset = 0

                def read(size: int) -> memoryview:
                    nonlocal offset
                    result = view[offset : offset + size]
                    offset += len(result)
                    return result

                try:
                    self.checkpoint_store.store_from_reader(
                        payload["cache_salt"],
                        payload["revision"],
                        length,
                        read,
                        payload["compatibility"],
                        timings,
                    )
                finally:
                    view.release()
        except OSError as exc:
            raise _RequestError from exc
        finally:
            os.close(checkpoint_fd)

    def _restore_from_fd(
        self,
        connection: socket.socket,
        output_stream: Any,
        length: int,
        regions: tuple[tuple[int, int], ...],
        timings: dict[str, float],
    ) -> None:
        checkpoint_fd = shm_open_readonly_fd(self.l1_shm_name)
        try:
            self.write_response(
                output_stream,
                {
                    "found": True,
                    "length": length,
                    "ok": True,
                    "regions": regions,
                    "timings_ms": timings,
                    "transport": "fd",
                },
            )
            output_stream.flush()
            sent = connection.sendmsg(
                [b"\0"],
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        array.array("i", [checkpoint_fd]),
                    )
                ],
            )
            if sent != 1 or connection.recv(1) != b"\0":
                raise _RequestError
        except OSError as exc:
            raise _RequestError from exc
        finally:
            os.close(checkpoint_fd)

    @staticmethod
    def _receive_fd(connection: socket.socket) -> int:
        descriptors = array.array("i")
        try:
            marker, ancillary, flags, _address = connection.recvmsg(
                1,
                socket.CMSG_SPACE(descriptors.itemsize),
            )
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors.frombytes(
                        data[: len(data) - len(data) % descriptors.itemsize]
                    )
            if marker != b"\0" or flags & socket.MSG_CTRUNC or len(descriptors) != 1:
                raise _RequestError
            return descriptors.pop()
        except OSError as exc:
            raise _RequestError from exc
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def error(self, code: str) -> dict[str, Any]:
        """Return a content-free error response and increment its counter.

        Args:
            code: Stable machine-readable error code.

        Returns:
            JSON-compatible error response.
        """
        self.stats["errors"] += 1
        return {"ok": False, "error": code}

    def write_response(
        self,
        stream: Any,
        response: dict[str, Any],
        checkpoint: bytes = b"",
    ) -> None:
        """Write one bounded JSON response.

        Args:
            stream: Socket file used by ``StreamRequestHandler``.
            response: JSON-compatible response object.
            checkpoint: Optional opaque checkpoint response frame.
        """
        encoded = json.dumps(response, separators=(",", ":"), sort_keys=True).encode()
        stream.write(encoded + b"\n")
        if checkpoint:
            stream.write(checkpoint)

    def server_close(self) -> None:
        """Close the server and remove only its Unix socket."""
        super().server_close()
        try:
            mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISSOCK(mode):
                self.socket_path.unlink()
        if self._owner_fd is not None:
            os.close(self._owner_fd)
            self._owner_fd = None

    def _validate(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            raise _RequestError
        action = payload.get("action")
        if action == "stats":
            if set(payload) != {"action"}:
                raise _RequestError
            return action
        expected = {
            "action",
            "cache_salt",
            "revision",
            "compatibility",
        }
        if action in {"store", "restore"}:
            if "transport" in payload:
                expected.add("transport")
        if action == "store":
            expected.add("length")
        if action not in {"store", "restore"} or set(payload) != expected:
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
        if action == "store" and (
            not isinstance(payload["length"], int)
            or isinstance(payload["length"], bool)
            or payload["length"] < 0
            or payload["length"] > self.max_checkpoint_bytes
        ):
            raise _RequestError
        if payload.get("transport") not in {None, "fd"}:
            raise _RequestError
        return action
