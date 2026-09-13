# SPDX-License-Identifier: Apache-2.0
"""Single-compute llama.cpp session broker tests."""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit
import json
import sqlite3
import stat

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp.session_broker import SessionBroker


class _LlamaHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = self.server
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        url = urlsplit(self.path)
        if url.path.startswith("/slots/"):
            action = parse_qs(url.query)["action"][0]
            key = (payload["cache_salt"], payload["revision"])
            server.slot_actions.append((action, *key))  # type: ignore[attr-defined]
            if action == "restore" and key not in server.states:  # type: ignore[attr-defined]
                self._respond(400, {"error": "missing"})
                return
            if action == "save":
                server.states.add(key)  # type: ignore[attr-defined]
            self._respond(200, {"ok": True})
            return
        if getattr(server, "fail_next_completion", False):
            server.fail_next_completion = False
            self._respond(500, {"error": "synthetic completion failure"})
            return
        server.completions.append(payload)  # type: ignore[attr-defined]
        if server.block_completions:  # type: ignore[attr-defined]
            server.completion_started.set()  # type: ignore[attr-defined]
            server.completion_release.wait()  # type: ignore[attr-defined]
        self._respond(200, {"choices": [{"message": {"content": "ok"}}]})

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_four_sessions_share_one_compute_slot(tmp_path: Path) -> None:
    class Payload(dict):
        def __deepcopy__(self, memo: dict) -> dict:
            raise AssertionError("full history was deep-copied")

    llama = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama.states = set()  # type: ignore[attr-defined]
    llama.slot_actions = []  # type: ignore[attr-defined]
    llama.completions = []  # type: ignore[attr-defined]
    llama.block_completions = False  # type: ignore[attr-defined]
    thread = Thread(target=llama.serve_forever, daemon=True)
    thread.start()
    try:
        broker = SessionBroker(
            database_path=tmp_path / "sessions.sqlite3",
            llama_url=f"http://127.0.0.1:{llama.server_port}",
            compatibility={"model": "tinyllama-unpinned"},
            timeout=1.0,
        )
        payload = Payload(messages=[{"role": "user", "content": "full history"}])

        revisions = {}
        for session_id in ("A", "B", "C", "D"):
            _body, headers = broker.complete("alice", session_id, 0, payload)
            revisions[session_id] = int(headers["X-LMCache-Revision"])
        _body, headers = broker.complete("alice", "A", revisions["A"], payload)

        assert [action[0] for action in llama.slot_actions] == [  # type: ignore[attr-defined]
            "save",
            "save",
            "save",
            "save",
            "restore",
            "save",
        ]
        assert [item["cache_prompt"] for item in llama.completions] == [  # type: ignore[attr-defined]
            False,
            False,
            False,
            False,
            True,
        ]
        assert headers["X-LMCache-Revision"] == "2"
    finally:
        llama.shutdown()
        llama.server_close()
        thread.join()


def test_rolling_sessions_save_only_when_switching_away(tmp_path: Path) -> None:
    llama = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama.states = set()  # type: ignore[attr-defined]
    llama.slot_actions = []  # type: ignore[attr-defined]
    llama.completions = []  # type: ignore[attr-defined]
    llama.block_completions = False  # type: ignore[attr-defined]
    thread = Thread(target=llama.serve_forever, daemon=True)
    thread.start()
    try:
        broker = SessionBroker(
            database_path=tmp_path / "sessions.sqlite3",
            llama_url=f"http://127.0.0.1:{llama.server_port}",
            compatibility={"model": "tinyllama-unpinned"},
            timeout=1.0,
            save_policy="switch",
        )
        payload = {"messages": [{"role": "user", "content": "full history"}]}

        _body, headers = broker.complete("alice", "A", 0, payload)
        _body, headers = broker.complete(
            "alice", "A", int(headers["X-LMCache-Revision"]), payload
        )
        revision_a = int(headers["X-LMCache-Revision"])
        _body, headers = broker.complete("alice", "B", 0, payload)
        revision_b = int(headers["X-LMCache-Revision"])
        broker.complete("alice", "A", revision_a, payload)

        assert [
            (action, revision)
            for action, _cache_salt, revision in llama.slot_actions  # type: ignore[attr-defined]
        ] == [
            ("save", revision_a),
            ("save", revision_b),
            ("restore", revision_a),
        ]
        assert [
            item["cache_prompt"]
            for item in llama.completions  # type: ignore[attr-defined]
        ] == [False, True, False, True]
    finally:
        llama.shutdown()
        llama.server_close()
        thread.join()


def test_broker_restores_committed_session_after_restart(tmp_path: Path) -> None:
    llama = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama.states = set()  # type: ignore[attr-defined]
    llama.slot_actions = []  # type: ignore[attr-defined]
    llama.completions = []  # type: ignore[attr-defined]
    llama.block_completions = False  # type: ignore[attr-defined]
    thread = Thread(target=llama.serve_forever, daemon=True)
    thread.start()
    try:
        database_path = tmp_path / "sessions.sqlite3"
        options = {
            "database_path": database_path,
            "llama_url": f"http://127.0.0.1:{llama.server_port}",
            "compatibility": {"model": "tinyllama-unpinned"},
            "timeout": 1.0,
        }
        payload = {"messages": [{"role": "user", "content": "full history"}]}
        _body, headers = SessionBroker(**options).complete("alice", "A", 0, payload)
        _body, headers = SessionBroker(**options).complete(
            "alice", "A", int(headers["X-LMCache-Revision"]), payload
        )

        assert [action[0] for action in llama.slot_actions] == [  # type: ignore[attr-defined]
            "save",
            "restore",
            "save",
        ]
        assert headers["X-LMCache-Cache"] == "hit"
        assert headers["X-LMCache-Revision"] == "2"
    finally:
        llama.shutdown()
        llama.server_close()
        thread.join()


def test_broker_does_not_repermission_shared_database_directory(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)

    with pytest.raises(ValueError, match="private"):
        SessionBroker(
            database_path=shared / "sessions.sqlite3",
            llama_url="http://127.0.0.1:8080",
            compatibility={"model": "tinyllama-unpinned"},
            timeout=1.0,
        )

    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_broker_rejects_unknown_save_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configuration"):
        SessionBroker(
            database_path=tmp_path / "sessions.sqlite3",
            llama_url="http://127.0.0.1:8080",
            compatibility={"model": "tinyllama-unpinned"},
            timeout=1.0,
            save_policy="sometimes",  # type: ignore[arg-type]
        )


def test_broker_rejects_requests_beyond_queue_capacity(tmp_path: Path) -> None:
    llama = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama.states = set()  # type: ignore[attr-defined]
    llama.slot_actions = []  # type: ignore[attr-defined]
    llama.completions = []  # type: ignore[attr-defined]
    llama.block_completions = True  # type: ignore[attr-defined]
    llama.completion_started = Event()  # type: ignore[attr-defined]
    llama.completion_release = Event()  # type: ignore[attr-defined]
    server_thread = Thread(target=llama.serve_forever, daemon=True)
    server_thread.start()
    errors: list[Exception] = []
    try:
        broker = SessionBroker(
            database_path=tmp_path / "sessions.sqlite3",
            llama_url=f"http://127.0.0.1:{llama.server_port}",
            compatibility={"model": "tinyllama-unpinned"},
            timeout=1.0,
            max_queue_depth=0,
        )
        payload = {"messages": [{"role": "user", "content": "full history"}]}
        request_thread = Thread(
            target=lambda: _complete_and_capture(errors, broker, "A", payload)
        )
        request_thread.start()
        assert llama.completion_started.wait(timeout=1)  # type: ignore[attr-defined]

        with pytest.raises(OverflowError, match="queue"):
            broker.complete("alice", "B", 0, payload)

        llama.completion_release.set()  # type: ignore[attr-defined]
        request_thread.join(timeout=2)
        assert not request_thread.is_alive()
        assert errors == []
    finally:
        llama.completion_release.set()  # type: ignore[attr-defined]
        llama.shutdown()
        llama.server_close()
        server_thread.join()


def _complete_and_capture(
    errors: list[Exception],
    broker: SessionBroker,
    session_id: str,
    payload: dict,
) -> None:
    try:
        broker.complete("alice", session_id, 0, payload)
    except Exception as exc:
        errors.append(exc)


@pytest.mark.parametrize("save_policy", ["always", "switch"])
def test_failed_switch_requires_restoring_previous_session(
    tmp_path: Path, save_policy: str
) -> None:
    llama = ThreadingHTTPServer(("127.0.0.1", 0), _LlamaHandler)
    llama.states = set()
    llama.slot_actions = []
    llama.completions = []
    llama.block_completions = False
    thread = Thread(target=llama.serve_forever, daemon=True)
    thread.start()
    try:
        broker = SessionBroker(
            tmp_path / "sessions.sqlite3",
            f"http://127.0.0.1:{llama.server_port}",
            {"model": "dummy"},
            1.0,
            save_policy=save_policy,
        )
        payload = {"messages": [{"role": "user", "content": "full history"}]}
        broker.complete("alice", "B", 0, payload)
        broker.complete("alice", "A", 0, payload)
        llama.fail_next_completion = True
        with pytest.raises(OSError):
            broker.complete("alice", "B", 1, payload)
        llama.slot_actions.clear()
        broker.complete("alice", "A", 1, payload)
        assert [action[0] for action in llama.slot_actions][:1] == ["restore"]
    finally:
        llama.shutdown()
        llama.server_close()
        thread.join()


def test_database_connections_close_after_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connections = []
    connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=TrackedConnection)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    SessionBroker(
        tmp_path / "sessions.sqlite3", "http://localhost:1", {"model": "dummy"}, 1.0
    )
    assert connections and all(connection.closed for connection in connections)
