# SPDX-License-Identifier: Apache-2.0
"""Fixed-path llama.cpp checkpoint bridge integration tests."""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse
import array
import json
import mmap
import os
import socket
import stat
import struct
import tempfile

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp.checkpoint_bridge import CheckpointBridge
from lmcache.integration.llamacpp.checkpoint_client import (
    restore_checkpoint,
    store_checkpoint,
)
from lmcache.integration.llamacpp.checkpoint_service import CheckpointService
from lmcache.integration.llamacpp.checkpoint_store import CheckpointStore
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager


def _storage_manager(cache_dir: Path, shm_name: str = "") -> StorageManager:
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 20,
                    use_lazy=False,
                    init_size_in_bytes=4 << 20,
                    align_bytes=4096,
                    shm_name=shm_name,
                )
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[FSL2AdapterConfig(base_path=str(cache_dir))]
            ),
        )
    )


@pytest.fixture
def llama_server(tmp_path: Path):
    transfer_dir = tmp_path / "transfer"
    transfer_dir.mkdir()
    state: dict[str, Any] = {
        "filenames": [],
        "payload": b"slot-state" * 1000,
        "restored": None,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = urlparse(self.path)
            action = parse_qs(request.query)["action"][0]
            length = int(self.headers["Content-Length"])
            filename = json.loads(self.rfile.read(length))["filename"]
            state["filenames"].append(filename)
            path = transfer_dir / filename
            if action == "save":
                path.write_bytes(state["payload"])
                response = {"filename": filename, "n_written": path.stat().st_size}
            else:
                state["restored"] = path.read_bytes()
                response = {"filename": filename, "n_read": path.stat().st_size}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", transfer_dir, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_store_restore_uses_generated_files_and_cleans_transfer_dir(
    tmp_path: Path,
    llama_server,
) -> None:
    llama_url, transfer_dir, state = llama_server
    manager = _storage_manager(tmp_path / "cache")
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    bridge = CheckpointBridge(
        store,
        transfer_dir=transfer_dir,
        llama_url=llama_url,
        slot_id=0,
        timeout=5.0,
    )
    compatibility = {"model": "qwen", "llama_cpp_commit": "86632248"}

    try:
        bridge.store("e" * 64, 4, compatibility)
        manager.clear()

        result = bridge.restore("e" * 64, 4, compatibility)

        assert result is not None
        assert state["restored"] == state["payload"]
        assert all(Path(name).name == name for name in state["filenames"])
        assert list(transfer_dir.iterdir()) == []
    finally:
        manager.close()


def test_duplicate_checkpoint_commit_is_idempotent(
    tmp_path: Path,
    llama_server,
) -> None:
    llama_url, transfer_dir, _state = llama_server
    manager = _storage_manager(tmp_path / "cache")
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    bridge = CheckpointBridge(
        store,
        transfer_dir=transfer_dir,
        llama_url=llama_url,
        slot_id=0,
        timeout=5.0,
    )

    try:
        bridge.store("f" * 64, 5, {"model": "qwen"})

        bridge.store("f" * 64, 5, {"model": "qwen"})

        assert list(transfer_dir.iterdir()) == []
    finally:
        manager.close()


def test_reused_revision_with_different_state_is_rejected(
    tmp_path: Path,
    llama_server,
) -> None:
    llama_url, transfer_dir, state = llama_server
    manager = _storage_manager(tmp_path / "cache")
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    bridge = CheckpointBridge(
        store,
        transfer_dir=transfer_dir,
        llama_url=llama_url,
        slot_id=0,
        timeout=5.0,
    )

    try:
        bridge.store("1" * 64, 5, {"model": "qwen"})
        state["payload"] = b"different-slot-state"

        with pytest.raises(ValueError, match="conflicts"):
            bridge.store("1" * 64, 5, {"model": "qwen"})
    finally:
        manager.close()


def _service_request(
    socket_path: Path,
    request: dict[str, Any],
    body: bytes = b"",
) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(json.dumps(request).encode() + b"\n" + body)
        response = client.makefile("rb").readline()
    return json.loads(response)


def _start_service(
    tmp_path: Path,
    max_request_bytes: int = 4096,
    max_checkpoint_bytes: int = 64 << 30,
    shm_name: str = "",
) -> tuple[CheckpointService, Thread, StorageManager, Path]:
    manager = _storage_manager(tmp_path / "cache", shm_name)
    socket_path = tmp_path / "bridge.sock"
    service = CheckpointService(
        CheckpointStore(manager, chunk_size=4096, timeout=5.0),
        socket_path,
        max_request_bytes,
        max_checkpoint_bytes,
        shm_name,
    )
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    return service, thread, manager, socket_path


def _stop_service(
    service: CheckpointService,
    thread: Thread,
    manager: StorageManager,
) -> None:
    service.shutdown()
    service.server_close()
    thread.join()
    manager.close()


def test_service_round_trips_checkpoint_over_private_unix_socket(
    tmp_path: Path,
) -> None:
    service, thread, manager, socket_path = _start_service(tmp_path)
    checkpoint = b"opaque-llama-state\0" * 1000
    cache_salt = "2" * 64
    revision = 6
    compatibility = {"model": "qwen"}

    try:
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        store_checkpoint(
            socket_path,
            checkpoint,
            cache_salt,
            revision,
            compatibility,
        )
        manager.clear()

        restored = restore_checkpoint(
            socket_path,
            cache_salt,
            revision,
            compatibility,
        )
        stats = _service_request(socket_path, {"action": "stats"})

        assert restored == checkpoint
        assert stats == {
            "ok": True,
            "result": {"errors": 0, "misses": 0, "restores": 1, "stores": 1},
        }
    finally:
        _stop_service(service, thread, manager)

    assert not socket_path.exists()


def test_service_stores_checkpoint_from_passed_file_descriptor(
    tmp_path: Path,
) -> None:
    service, thread, manager, socket_path = _start_service(tmp_path)
    checkpoint = b"fd-backed-llama-state\0" * 1000
    identity = {
        "cache_salt": "8" * 64,
        "revision": 1,
        "compatibility": {"model": "qwen"},
    }

    try:
        with tempfile.TemporaryFile() as checkpoint_file:
            checkpoint_file.write(checkpoint)
            checkpoint_file.flush()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(socket_path))
                request = {
                    "action": "store",
                    "length": len(checkpoint),
                    "transport": "fd",
                    **identity,
                }
                client.sendall(json.dumps(request).encode() + b"\n")
                with client.makefile("rb") as response_stream:
                    assert json.loads(response_stream.readline()) == {
                        "ok": True,
                        "ready": True,
                    }
                    client.sendmsg(
                        [b"\0"],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                struct.pack("i", checkpoint_file.fileno()),
                            )
                        ],
                    )
                    assert json.loads(response_stream.readline())["ok"] is True

        manager.clear()
        assert restore_checkpoint(socket_path, **identity) == checkpoint
    finally:
        _stop_service(service, thread, manager)


def test_service_restores_checkpoint_from_l1_shared_memory_fd(
    tmp_path: Path,
) -> None:
    shm_name = f"lmcache_l1_pool_llamacpp_test_{os.getpid()}"
    service, thread, manager, socket_path = _start_service(
        tmp_path,
        shm_name=shm_name,
    )
    checkpoint = b"a" * 4096 + b"b" * 4096 + b"tail"
    identity = {
        "cache_salt": "9" * 64,
        "revision": 1,
        "compatibility": {"model": "qwen"},
    }

    try:
        store_checkpoint(socket_path, checkpoint, **identity)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(socket_path))
            client.sendall(
                json.dumps(
                    {"action": "restore", "transport": "fd", **identity}
                ).encode()
                + b"\n"
            )
            with client.makefile("rb") as response_stream:
                response = json.loads(response_stream.readline())
                assert response["transport"] == "fd"
                assert response["length"] == len(checkpoint)
                descriptors = array.array("i")
                marker, ancillary, flags, _address = client.recvmsg(
                    1,
                    socket.CMSG_SPACE(descriptors.itemsize),
                )
                for level, kind, data in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        descriptors.frombytes(data[: descriptors.itemsize])
                assert marker == b"\0"
                assert not flags & socket.MSG_CTRUNC
                assert len(descriptors) == 1
                fd = descriptors.pop()
                try:
                    with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as pool:
                        restored = b"".join(
                            pool[offset : offset + length]
                            for offset, length in response["regions"]
                        )
                finally:
                    os.close(fd)
                client.sendall(b"\0")

        assert restored == checkpoint
    finally:
        _stop_service(service, thread, manager)


def test_service_reports_store_stage_timings(tmp_path: Path) -> None:
    service, thread, manager, socket_path = _start_service(tmp_path)
    request = {
        "action": "store",
        "cache_salt": "5" * 64,
        "revision": 1,
        "compatibility": {"model": "qwen"},
        "length": 4096,
    }

    try:
        response = _service_request(socket_path, request, b"x" * 4096)

        assert response["ok"] is True
        assert set(response["timings_ms"]) == {
            "chunk_write",
            "hash",
            "hash_wait",
            "manifest",
            "receive",
            "wait",
        }
        assert all(value >= 0 for value in response["timings_ms"].values())
    finally:
        _stop_service(service, thread, manager)


def test_service_streams_incoming_checkpoint_to_store() -> None:
    class RecordingStore:
        def __init__(self) -> None:
            self.aggregate_called = False
            self.streamed = b""

        def store(self, *args: Any, **kwargs: Any) -> None:
            self.aggregate_called = True

        def store_from_reader(
            self,
            cache_salt: str,
            revision: int,
            checkpoint_length: int,
            reader: Any,
            compatibility: dict[str, Any],
            timings: dict[str, float],
        ) -> None:
            self.streamed = reader(checkpoint_length)
            timings.update(
                {
                    "chunk_write": 0.0,
                    "hash": 0.0,
                    "hash_wait": 0.0,
                    "receive": 0.0,
                    "wait": 0.0,
                    "manifest": 0.0,
                }
            )

    store = RecordingStore()
    service = object.__new__(CheckpointService)
    service.checkpoint_store = store
    service.max_checkpoint_bytes = 64 << 30
    service.stats = {"errors": 0, "misses": 0, "restores": 0, "stores": 0}
    request = json.dumps(
        {
            "action": "store",
            "cache_salt": "6" * 64,
            "revision": 1,
            "compatibility": {"model": "qwen"},
            "length": 8,
        }
    ).encode()
    output = BytesIO()

    service.serve_request(request, BytesIO(b"abcdefgh"), output)

    assert store.aggregate_called is False
    assert store.streamed == b"abcdefgh"
    assert json.loads(output.getvalue())["ok"] is True


def test_service_writes_checkpoint_without_combining_frame() -> None:
    class RecordingStream:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

    stream = RecordingStream()
    checkpoint = b"opaque-state"
    service = object.__new__(CheckpointService)

    service.write_response(
        stream,
        {"found": True, "length": len(checkpoint), "ok": True},
        checkpoint,
    )

    assert stream.writes == [
        b'{"found":true,"length":12,"ok":true}\n',
        checkpoint,
    ]


def test_service_streams_checkpoint_views(tmp_path: Path) -> None:
    class RecordingStream:
        def __init__(self) -> None:
            self.writes: list[bytes | memoryview] = []

        def write(self, data: bytes | memoryview) -> int:
            self.writes.append(data)
            return len(data)

    service, thread, manager, _socket_path = _start_service(tmp_path)
    payload = b"a" * 4096 + b"b" * 4096 + b"tail"
    cache_salt = "7" * 64
    compatibility = {"model": "qwen"}
    request = json.dumps(
        {
            "action": "restore",
            "cache_salt": cache_salt,
            "revision": 1,
            "compatibility": compatibility,
            "transport": "fd",
        }
    ).encode()
    stream = RecordingStream()

    try:
        service.checkpoint_store.store(cache_salt, 1, payload, compatibility)

        service.serve_request(request, BytesIO(), stream)

        header = json.loads(stream.writes[0])
        assert header["found"] is True
        assert header["length"] == len(payload)
        assert header["ok"] is True
        assert "transport" not in header
        assert set(header["timings_ms"]) == {
            "chunk_read",
            "manifest_read",
            "validate",
            "validate_skipped",
        }
        assert all(value >= 0 for value in header["timings_ms"].values())
        assert all(isinstance(chunk, memoryview) for chunk in stream.writes[1:])
        assert b"".join(stream.writes[1:]) == payload
    finally:
        _stop_service(service, thread, manager)


def test_service_rejects_caller_selected_paths(
    tmp_path: Path,
) -> None:
    service, thread, manager, socket_path = _start_service(tmp_path)
    request = {
        "action": "store",
        "cache_salt": "3" * 64,
        "revision": 1,
        "compatibility": {"model": "qwen"},
        "filename": "outside.bin",
    }

    try:
        assert _service_request(socket_path, request) == {
            "ok": False,
            "error": "invalid_request",
        }
    finally:
        _stop_service(service, thread, manager)


def test_service_rejects_oversized_request(
    tmp_path: Path,
) -> None:
    service, thread, manager, socket_path = _start_service(
        tmp_path, max_request_bytes=256
    )
    request = {
        "action": "store",
        "cache_salt": "4" * 64,
        "revision": 1,
        "compatibility": {"padding": "x" * 512},
    }

    try:
        assert _service_request(socket_path, request) == {
            "ok": False,
            "error": "invalid_request",
        }
    finally:
        _stop_service(service, thread, manager)


def test_service_rejects_oversized_checkpoint(tmp_path: Path) -> None:
    service, thread, manager, socket_path = _start_service(
        tmp_path,
        max_checkpoint_bytes=8,
    )
    request = {
        "action": "store",
        "cache_salt": "6" * 64,
        "revision": 1,
        "compatibility": {"model": "qwen"},
        "length": 9,
    }

    try:
        assert _service_request(socket_path, request, b"123456789") == {
            "ok": False,
            "error": "invalid_request",
        }
    finally:
        _stop_service(service, thread, manager)
