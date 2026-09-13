# SPDX-License-Identifier: Apache-2.0
"""Executable llama.cpp checkpoint service integration tests."""

# Standard
from hashlib import sha256 as real_sha256
from pathlib import Path
from threading import Thread
import signal
import socket
import stat
import subprocess
import sys
import time

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp import checkpoint_store
from lmcache.integration.llamacpp.checkpoint_client import (
    restore_checkpoint,
    store_checkpoint,
)
from lmcache.integration.llamacpp.checkpoint_daemon import create_checkpoint_service

# Cold Torch imports take about nine seconds before the daemon binds its socket.
DAEMON_STARTUP_TIMEOUT = 30


def test_create_service_uses_private_bounded_lmcache_storage(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"

    service, manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=cache_dir,
        l1_size_bytes=4 << 20,
        l2_size_gb=0.01,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
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


def test_create_service_can_validate_checkpoints_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, manager = create_checkpoint_service(
        runtime_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
        validate_once=True,
    )

    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    checkpoint = b"opaque-state" * 1000
    compatibility = {"model": "qwen"}

    try:
        store_checkpoint(
            service.socket_path,
            checkpoint,
            "9" * 64,
            1,
            compatibility,
        )
        calls = 0

        def tracked_sha256(data=b""):
            nonlocal calls
            calls += 1
            return real_sha256(data)

        monkeypatch.setattr(checkpoint_store, "sha256", tracked_sha256)

        assert (
            restore_checkpoint(
                service.socket_path,
                "9" * 64,
                1,
                compatibility,
            )
            == checkpoint
        )
        assert calls == 1
    finally:
        service.shutdown()
        service.server_close()
        thread.join()
        manager.close()


def test_create_service_promotes_l2_restore_into_l1(tmp_path: Path) -> None:
    service, manager = create_checkpoint_service(
        runtime_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0.01,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
    )
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    checkpoint = b"opaque-state" * 1000
    compatibility = {"model": "qwen"}

    try:
        store_checkpoint(
            service.socket_path,
            checkpoint,
            "a" * 64,
            1,
            compatibility,
        )
        manager.clear()
        assert manager.get_l1_usage()[0] == 0

        assert (
            restore_checkpoint(
                service.socket_path,
                "a" * 64,
                1,
                compatibility,
            )
            == checkpoint
        )
        assert manager.get_l1_usage()[0] > 0
    finally:
        service.shutdown()
        service.server_close()
        thread.join()
        manager.close()


def test_create_service_allows_ram_only_storage(tmp_path: Path) -> None:
    service, manager = create_checkpoint_service(
        runtime_dir=tmp_path / "run",
        cache_dir=tmp_path / "unused-cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
    )

    try:
        assert manager.l2_adapters() == []
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
    deadline = time.monotonic() + DAEMON_STARTUP_TIMEOUT

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


def test_daemon_times_out_stalled_control_request(tmp_path: Path) -> None:
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
            "--l1-size-bytes",
            str(4 << 20),
            "--l2-size-gb",
            "0",
            "--chunk-size",
            "4096",
            "--timeout",
            "0.05",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + DAEMON_STARTUP_TIMEOUT

    try:
        while not socket_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("checkpoint daemon did not create its socket")
            time.sleep(0.05)
        assert process.poll() is None

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            client.connect(str(socket_path))
            client.sendall(b'{"action":')
            response = client.makefile("rb").readline()

        assert response
        assert b'"error":"timeout"' in response
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_create_service_refuses_live_runtime_owner(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    service, manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=tmp_path / "cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
    )
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()

    try:
        with pytest.raises(ValueError, match="already active"):
            create_checkpoint_service(
                runtime_dir=runtime_dir,
                cache_dir=tmp_path / "other-cache",
                l1_size_bytes=4 << 20,
                l2_size_gb=0,
                chunk_size=4096,
                timeout=5.0,
                max_request_bytes=4096,
                max_checkpoint_bytes=1 << 20,
            )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(runtime_dir / "bridge.sock"))
    finally:
        service.shutdown()
        service.server_close()
        thread.join()
        manager.close()


def test_create_services_use_collision_safe_shared_memory(tmp_path: Path) -> None:
    services = []
    managers = []
    try:
        for name in ("a", "b"):
            service, manager = create_checkpoint_service(
                runtime_dir=tmp_path / f"run-{name}",
                cache_dir=tmp_path / f"cache-{name}",
                l1_size_bytes=4 << 20,
                l2_size_gb=0,
                chunk_size=4096,
                timeout=5.0,
                max_request_bytes=4096,
                max_checkpoint_bytes=1 << 20,
            )
            services.append(service)
            managers.append(manager)
    finally:
        for service in services:
            service.server_close()
        for manager in managers:
            manager.close()


def test_service_releases_runtime_lock_when_socket_is_missing(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    service, manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=tmp_path / "cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
    )
    (runtime_dir / "bridge.sock").unlink()
    service.server_close()
    manager.close()

    replacement, replacement_manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=tmp_path / "replacement-cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
    )
    replacement.server_close()
    replacement_manager.close()


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="POSIX SHM is unavailable")
def test_daemon_restart_cleans_its_sigkill_shared_memory(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    socket_path = runtime_dir / "bridge.sock"
    baseline = set(Path("/dev/shm").glob("lmcache_l1_pool_llamacpp_*"))

    def start(previous_socket_inode: int | None = None) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lmcache.integration.llamacpp.checkpoint_daemon",
                "--runtime-dir",
                str(runtime_dir),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--l1-size-bytes",
                str(4 << 20),
                "--l2-size-gb",
                "0",
                "--chunk-size",
                "4096",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + DAEMON_STARTUP_TIMEOUT
        try:
            while process.poll() is None:
                current_inode = (
                    socket_path.stat().st_ino if socket_path.exists() else None
                )
                if current_inode is not None and current_inode != previous_socket_inode:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError("checkpoint daemon did not create its socket")
                time.sleep(0.05)
            assert process.poll() is None
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        return process

    first = start()
    first_socket_inode = socket_path.stat().st_ino
    first.kill()
    assert first.wait(timeout=10) < 0
    second = start(first_socket_inode)
    try:
        second.send_signal(signal.SIGTERM)
        assert second.wait(timeout=10) == 0
    finally:
        if second.poll() is None:
            second.kill()
            second.wait()

    assert set(Path("/dev/shm").glob("lmcache_l1_pool_llamacpp_*")) == baseline


def test_create_service_replaces_stale_unix_socket(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir(mode=0o700)
    socket_path = runtime_dir / "bridge.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale_socket:
        stale_socket.bind(str(socket_path))

    service, manager = create_checkpoint_service(
        runtime_dir=runtime_dir,
        cache_dir=tmp_path / "cache",
        l1_size_bytes=4 << 20,
        l2_size_gb=0.01,
        chunk_size=4096,
        timeout=5.0,
        max_request_bytes=4096,
        max_checkpoint_bytes=1 << 20,
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
                l1_size_bytes=4 << 20,
                l2_size_gb=0.01,
                chunk_size=4096,
                timeout=5.0,
                max_request_bytes=4096,
                max_checkpoint_bytes=1 << 20,
            )
    finally:
        if service is not None:
            service.server_close()
        if manager is not None:
            manager.close()

    assert socket_path.read_text() == "operator-data"
