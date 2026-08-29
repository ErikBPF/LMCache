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
from lmcache.integration.llamacpp.checkpoint_client import (
    CheckpointClientError,
    main,
    request,
)


class _Handler(StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        server.request_body = json.loads(self.rfile.readline())  # type: ignore[attr-defined]
        self.wfile.write(server.response_body)  # type: ignore[attr-defined]


@pytest.fixture
def control_socket(tmp_path: Path):
    socket_path = tmp_path / "bridge.sock"
    server = UnixStreamServer(str(socket_path), _Handler)
    server.request_body = None  # type: ignore[attr-defined]
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


def test_main_builds_store_request_and_prints_response(control_socket, capsys) -> None:
    server, socket_path = control_socket

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
        ]
    )

    assert exit_code == 0
    assert server.request_body == {  # type: ignore[attr-defined]
        "action": "store",
        "cache_salt": "a" * 64,
        "revision": 7,
        "compatibility": {"model": "qwen"},
    }
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_main_returns_failure_for_service_error(control_socket, capsys) -> None:
    server, socket_path = control_socket
    server.response_body = b'{"error":"backend","ok":false}\n'  # type: ignore[attr-defined]

    exit_code = main(["--socket", str(socket_path), "stats"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "backend",
        "ok": False,
    }
