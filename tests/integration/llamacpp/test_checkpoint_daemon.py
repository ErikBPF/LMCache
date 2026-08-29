# SPDX-License-Identifier: Apache-2.0
"""Executable llama.cpp checkpoint service integration tests."""

# Standard
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp.checkpoint_daemon import create_checkpoint_service


def test_create_service_uses_private_bounded_lmcache_storage(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"

    service, manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=cache_dir,
        llama_url="http://127.0.0.1:8080",
        slot_id=0,
        l1_size_bytes=4 << 20,
        l2_size_gb=0.01,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
    )

    try:
        _descriptor, adapter = manager.l2_adapters()[0]
        usage = adapter.get_usage()
        assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(service.socket_path.stat().st_mode) == 0o600
        assert adapter.supports_global_eviction
        assert usage.total_capacity_bytes == int(0.01 * (1024**3))
    finally:
        service.server_close()
        manager.close()


def test_daemon_removes_socket_after_sigterm(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    socket_path = runtime_dir / "bridge.sock"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.integration.llamacpp.checkpoint_daemon",
            "--runtime-dir",
            str(runtime_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--llama-url",
            "http://127.0.0.1:8080",
            "--l1-size-bytes",
            str(4 << 20),
            "--l2-size-gb",
            "0.01",
            "--chunk-size",
            "4096",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10

    try:
        while not socket_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("checkpoint daemon did not create its socket")
            time.sleep(0.05)
        assert process.poll() is None

        process.send_signal(signal.SIGTERM)

        assert process.wait(timeout=10) == 0
        assert not socket_path.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_create_service_replaces_stale_unix_socket(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir(mode=0o700)
    socket_path = runtime_dir / "bridge.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale_socket:
        stale_socket.bind(str(socket_path))

    service, manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=tmp_path / "cache",
        llama_url="http://127.0.0.1:8080",
        slot_id=0,
        l1_size_bytes=4 << 20,
        l2_size_gb=0.01,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
    )

    try:
        assert stat.S_ISSOCK(socket_path.stat().st_mode)
    finally:
        service.server_close()
        manager.close()


def test_create_service_preserves_non_socket_path(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir(mode=0o700)
    socket_path = runtime_dir / "bridge.sock"
    socket_path.write_text("operator-data")
    service = None
    manager = None

    try:
        with pytest.raises(ValueError, match="refusing to replace"):
            service, manager = create_checkpoint_service(
                runtime_dir=runtime_dir,
                cache_dir=tmp_path / "cache",
                llama_url="http://127.0.0.1:8080",
                slot_id=0,
                l1_size_bytes=4 << 20,
                l2_size_gb=0.01,
                chunk_size=4096,
                timeout=5.0,
                max_request_bytes=4096,
            )
    finally:
        if service is not None:
            service.server_close()
        if manager is not None:
            manager.close()

    assert socket_path.read_text() == "operator-data"
