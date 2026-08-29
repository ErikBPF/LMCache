# SPDX-License-Identifier: Apache-2.0
"""Fixed-path llama.cpp checkpoint bridge integration tests."""

# Standard
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse
import json

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp.checkpoint_bridge import CheckpointBridge
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


def _storage_manager(cache_dir: Path) -> StorageManager:
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 20,
                    use_lazy=False,
                    init_size_in_bytes=4 << 20,
                    align_bytes=4096,
                    shm_name="",
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
