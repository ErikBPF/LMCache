# SPDX-License-Identifier: Apache-2.0
"""llama.cpp checkpoint client tests."""

# Standard
from pathlib import Path
from socketserver import StreamRequestHandler, UnixStreamServer
from threading import Thread
import json

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp import checkpoint_client
from lmcache.integration.llamacpp.checkpoint_client import (
    CheckpointClientError,
    main,
    request,
)


class _Handler(StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        server.request_body = json.loads(self.rfile.readline())  # type: ignore[attr-defined]
        length = server.request_body.get("length", 0)  # type: ignore[attr-defined]
        server.request_checkpoint = self.rfile.read(length)  # type: ignore[attr-defined]
        self.wfile.write(server.response_body)  # type: ignore[attr-defined]


@pytest.fixture
def control_socket(tmp_path: Path):
    socket_path = tmp_path / "bridge.sock"
    server = UnixStreamServer(str(socket_path), _Handler)
    server.request_body = None  # type: ignore[attr-defined]
    server.request_checkpoint = b""  # type: ignore[attr-defined]
    server.response_body = b'{"ok":true,"result":{"stores":1}}\n'  # type: ignore[attr-defined]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, socket_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_request_round_trips_one_bounded_json_frame(control_socket) -> None:
    server, socket_path = control_socket

    response = request(
        socket_path, {"action": "stats"}, timeout=1.0, max_response_bytes=64
    )

    assert server.request_body == {"action": "stats"}  # type: ignore[attr-defined]
    assert response == {"ok": True, "result": {"stores": 1}}


def test_request_rejects_oversized_response(control_socket) -> None:
    server, socket_path = control_socket
    server.response_body = (  # type: ignore[attr-defined]
        b'{"ok":true,"result":"' + b"x" * 64 + b'"}\n'
    )

    with pytest.raises(CheckpointClientError, match="too large"):
        request(socket_path, {"action": "stats"}, timeout=1.0, max_response_bytes=32)


def test_checkpoint_bytes_use_length_prefixed_frames(tmp_path: Path) -> None:
    checkpoint = b"opaque-llama-state\0" * 100

    class CheckpointHandler(StreamRequestHandler):
        def handle(self) -> None:
            server = self.server
            request_body = json.loads(self.rfile.readline())
            server.requests.append(request_body)  # type: ignore[attr-defined]
            if request_body["action"] == "store":
                server.stored = self.rfile.read(request_body["length"])  # type: ignore[attr-defined]
                self.wfile.write(b'{"ok":true}\n')
            else:
                body = server.stored  # type: ignore[attr-defined]
                self.wfile.write(
                    json.dumps(
                        {"found": True, "length": len(body), "ok": True},
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                    + body
                )

    socket_path = tmp_path / "checkpoint.sock"
    server = UnixStreamServer(str(socket_path), CheckpointHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.stored = b""  # type: ignore[attr-defined]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cache_salt = "b" * 64
        revision = 8
        compatibility = {"model": "qwen"}
        identity = {
            "cache_salt": cache_salt,
            "revision": revision,
            "compatibility": compatibility,
        }

        checkpoint_client.store_checkpoint(
            socket_path,
            checkpoint,
            cache_salt,
            revision,
            compatibility,
        )
        restored = checkpoint_client.restore_checkpoint(
            socket_path,
            cache_salt,
            revision,
            compatibility,
        )

        assert restored == checkpoint
        assert server.requests == [  # type: ignore[attr-defined]
            {"action": "store", "length": len(checkpoint), **identity},
            {"action": "restore", **identity},
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_main_store_sends_checkpoint_bytes(
    control_socket, capsys, tmp_path: Path
) -> None:
    server, socket_path = control_socket
    checkpoint_path = tmp_path / "checkpoint.bin"
    checkpoint_path.write_bytes(b"opaque-state")

    exit_code = main(
        [
            "--socket",
            str(socket_path),
            "store",
            "--cache-salt",
            "a" * 64,
            "--revision",
            "7",
            "--compatibility",
            '{"model":"qwen"}',
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    assert exit_code == 0
    assert server.request_body == {  # type: ignore[attr-defined]
        "action": "store",
        "cache_salt": "a" * 64,
        "length": 12,
        "revision": 7,
        "compatibility": {"model": "qwen"},
    }
    assert server.request_checkpoint == b"opaque-state"  # type: ignore[attr-defined]
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_main_restore_writes_checkpoint_bytes(
    control_socket, capsys, tmp_path: Path
) -> None:
    server, socket_path = control_socket
    checkpoint_path = tmp_path / "checkpoint.bin"
    server.response_body = (  # type: ignore[attr-defined]
        b'{"found":true,"length":12,"ok":true}\nopaque-state'
    )

    exit_code = main(
        [
            "--socket",
            str(socket_path),
            "restore",
            "--cache-salt",
            "a" * 64,
            "--revision",
            "7",
            "--compatibility",
            '{"model":"qwen"}',
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    assert exit_code == 0
    assert checkpoint_path.read_bytes() == b"opaque-state"
    assert json.loads(capsys.readouterr().out)["found"] is True


def test_main_returns_failure_for_service_error(control_socket, capsys) -> None:
    server, socket_path = control_socket
    server.response_body = b'{"error":"backend","ok":false}\n'  # type: ignore[attr-defined]

    exit_code = main(["--socket", str(socket_path), "stats"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "backend",
        "ok": False,
    }
