# SPDX-License-Identifier: Apache-2.0
"""Serialized llama.cpp session broker tests."""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import StreamRequestHandler, UnixStreamServer
from threading import Thread
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp.session_broker import SessionBroker, create_server


class _CheckpointHandler(StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        payload = json.loads(self.rfile.readline())
        server.requests.append(payload)  # type: ignore[attr-defined]
        result: dict[str, Any] | None = {"n_read": 4, "n_written": 4}
        if payload["action"] == "restore" and payload["cache_salt"] in server.misses:  # type: ignore[attr-defined]
            result = None
        self.wfile.write(json.dumps({"ok": True, "result": result}).encode() + b"\n")


class _LlamaHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)  # type: ignore[attr-defined]
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def backends(tmp_path: Path):
    socket_path = tmp_path / "bridge.sock"
    checkpoint = UnixStreamServer(str(socket_path), _CheckpointHandler)
    checkpoint.requests = []  # type: ignore[attr-defined]
    checkpoint.misses = set()  # type: ignore[attr-defined]
    llama = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama.requests = []  # type: ignore[attr-defined]
    threads = [
        Thread(target=checkpoint.serve_forever, daemon=True),
        Thread(target=llama.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        yield checkpoint, llama, socket_path
    finally:
        checkpoint.shutdown()
        checkpoint.server_close()
        llama.shutdown()
        llama.server_close()
        for thread in threads:
            thread.join()


def test_four_sessions_restore_only_the_requested_revision(
    tmp_path: Path, backends
) -> None:
    checkpoint, llama, socket_path = backends
    broker = SessionBroker(
        database_path=tmp_path / "sessions.sqlite3",
        checkpoint_socket=socket_path,
        llama_url=f"http://127.0.0.1:{llama.server_port}",
        compatibility={"model": "qwen"},
        timeout=1.0,
    )
    payload = {"messages": [{"role": "user", "content": "canonical"}]}

    revisions = {}
    for session in ("A", "B", "C", "D"):
        response, headers = broker.complete("alice", session, 0, payload)
        revisions[session] = int(headers["X-LMCache-Revision"])
        assert json.loads(response)["choices"][0]["message"]["content"] == "ok"
    _, headers = broker.complete("alice", "A", revisions["A"], payload)

    actions = [request["action"] for request in checkpoint.requests]
    assert actions == ["store", "store", "store", "store", "restore", "store"]
    assert llama.requests[-1]["cache_prompt"] is True
    assert headers["X-LMCache-Revision"] == "2"
    assert float(headers["X-LMCache-Queue-Wait-Ms"]) >= 0
    assert float(headers["X-LMCache-Switch-Ms"]) >= 0

    with pytest.raises(ValueError, match="revision conflict"):
        broker.complete("alice", "A", 1, payload)


def test_restore_miss_recomputes_from_canonical_history(
    tmp_path: Path, backends
) -> None:
    checkpoint, llama, socket_path = backends
    broker = SessionBroker(
        database_path=tmp_path / "sessions.sqlite3",
        checkpoint_socket=socket_path,
        llama_url=f"http://127.0.0.1:{llama.server_port}",
        compatibility={"model": "qwen"},
        timeout=1.0,
    )
    payload = {"messages": [{"role": "user", "content": "canonical"}]}
    _, first = broker.complete("alice", "A", 0, payload)
    broker.complete("alice", "B", 0, payload)
    salt = checkpoint.requests[0]["cache_salt"]
    checkpoint.misses.add(salt)

    _, resumed = broker.complete(
        "alice", "A", int(first["X-LMCache-Revision"]), payload
    )

    assert llama.requests[-1]["cache_prompt"] is False
    assert resumed["X-LMCache-Cache"] == "miss"


def test_http_server_requires_identity_and_returns_committed_revision(
    tmp_path: Path, backends
) -> None:
    _, llama, socket_path = backends
    broker = SessionBroker(
        database_path=tmp_path / "sessions.sqlite3",
        checkpoint_socket=socket_path,
        llama_url=f"http://127.0.0.1:{llama.server_port}",
        compatibility={"model": "qwen"},
        timeout=1.0,
    )
    server = create_server(broker, "127.0.0.1", 0, max_request_bytes=4096)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-LMCache-Principal": "alice",
                "X-LMCache-Session": "A",
                "X-LMCache-Revision": "0",
            },
            method="POST",
        )
        with urlopen(request, timeout=1) as response:
            assert response.status == 200
            assert response.headers["X-LMCache-Revision"] == "1"

        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=1,
            )
        assert error.value.code == 400
        error.value.close()

        invalid = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=b"\xff",
            headers={
                "Content-Type": "application/json",
                "X-LMCache-Principal": "alice",
                "X-LMCache-Session": "A",
                "X-LMCache-Revision": "1",
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(invalid, timeout=1)
        assert error.value.code == 400
        error.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
