# SPDX-License-Identifier: Apache-2.0
"""Construct the bounded local llama.cpp checkpoint service."""

# Standard
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from types import FrameType
from typing import NoReturn
import fcntl
import os
import secrets
import signal
import stat

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.posix_shm import shm_unlink

# Local
from .checkpoint_service import CheckpointService
from .checkpoint_store import CheckpointStore


class _ExitRequested(Exception):
    def __init__(self, exit_code: int) -> None:
        super().__init__()
        self.exit_code = exit_code


def create_checkpoint_service(
    runtime_dir: Path,
    cache_dir: Path,
    l1_size_bytes: int,
    l2_size_gb: float,
    chunk_size: int,
    timeout: float,
    max_request_bytes: int,
    max_checkpoint_bytes: int,
    validate_once: bool = False,
) -> tuple[CheckpointService, StorageManager]:
    """Create a private service with bounded LMCache RAM and filesystem tiers.

    Args:
        runtime_dir: Private directory for the Unix socket.
        cache_dir: Private local filesystem L2 directory when L2 is enabled.
        l1_size_bytes: LMCache RAM capacity in bytes.
        l2_size_gb: LMCache filesystem capacity in GiB; zero disables L2.
        chunk_size: Opaque checkpoint chunk size in bytes.
        timeout: Storage timeout in seconds.
        max_request_bytes: Maximum Unix control request size.
        max_checkpoint_bytes: Maximum binary checkpoint frame size.
        validate_once: Reuse successful integrity validation for this process.

    Returns:
        Bound checkpoint service and its storage manager.

    Raises:
        ValueError: If a capacity is invalid or a private directory is unsafe.
    """
    if l1_size_bytes <= 0 or l2_size_gb < 0:
        raise ValueError("LMCache RAM must be positive and disk non-negative")
    _private_directory(runtime_dir)
    owner_fd = _acquire_runtime_lock(runtime_dir / "checkpoint.lock")
    socket_path = runtime_dir / "bridge.sock"
    manager = None
    try:
        _remove_stale_socket(socket_path)
        shm_prefix = _shared_memory_prefix(runtime_dir)
        _remove_stale_shared_memory(shm_prefix)

        l2_adapters: list[L2AdapterConfigBase] = []
        if l2_size_gb > 0:
            _private_directory(cache_dir)
            l2_config = FSNativeL2AdapterConfig(
                base_path=str(cache_dir),
                max_capacity_gb=l2_size_gb,
            )
            l2_config.eviction_config = EvictionConfig(
                eviction_policy="LRU",
                trigger_watermark=0.8,
                eviction_ratio=0.2,
            )
            l2_adapters.append(l2_config)
        l1_shm_name = f"{shm_prefix}{secrets.token_hex(8)}"
        manager = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=l1_size_bytes,
                        use_lazy=False,
                        init_size_in_bytes=l1_size_bytes,
                        shm_name=l1_shm_name,
                    )
                ),
                eviction_config=EvictionConfig(
                    eviction_policy="LRU",
                    trigger_watermark=0.8,
                    eviction_ratio=0.2,
                ),
                l2_adapter_config=L2AdaptersConfig(adapters=l2_adapters),
                prefetch_policy="retain",
            )
        )
        store = CheckpointStore(
            manager,
            chunk_size=chunk_size,
            timeout=timeout,
            validate_once=validate_once,
        )
        service = CheckpointService(
            store,
            socket_path,
            max_request_bytes=max_request_bytes,
            max_checkpoint_bytes=max_checkpoint_bytes,
            l1_shm_name=l1_shm_name,
            io_timeout=timeout,
            owner_fd=owner_fd,
        )
        owner_fd = -1
    except Exception:
        if manager is not None:
            manager.close()
        if owner_fd >= 0:
            os.close(owner_fd)
        raise
    return service, manager


def main(argv: list[str] | None = None) -> int:
    """Run the bounded local checkpoint service until interrupted.

    Args:
        argv: Optional command-line arguments for tests and embedding.

    Returns:
        Process exit code.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--l1-size-bytes", type=int, required=True)
    parser.add_argument("--l2-size-gb", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-request-bytes", type=int, default=64 << 10)
    parser.add_argument("--max-checkpoint-bytes", type=int, default=64 << 30)
    parser.add_argument("--validate-once", action="store_true")
    args = parser.parse_args(argv)

    service, manager = create_checkpoint_service(
        runtime_dir=args.runtime_dir,
        cache_dir=args.cache_dir,
        l1_size_bytes=args.l1_size_bytes,
        l2_size_gb=args.l2_size_gb,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        max_request_bytes=args.max_request_bytes,
        max_checkpoint_bytes=args.max_checkpoint_bytes,
        validate_once=args.validate_once,
    )
    previous_sigterm = signal.signal(signal.SIGTERM, _request_exit)
    previous_sigint = signal.signal(signal.SIGINT, _request_exit)
    try:
        service.serve_forever(poll_interval=0.1)
    except _ExitRequested as exc:
        return exc.exit_code
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        manager.close()
        service.server_close()
    return 0


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.geteuid():
        raise ValueError("LMCache directories must be owned local directories")
    os.chmod(path, 0o700)


def _remove_stale_socket(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(status.st_mode) or status.st_uid != os.geteuid():
        raise ValueError("refusing to replace a non-socket or unowned path")
    path.unlink()


def _acquire_runtime_lock(path: Path) -> int:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.chmod(path, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise ValueError("checkpoint runtime is already active") from exc
    except Exception:
        os.close(fd)
        raise
    return fd


def _shared_memory_prefix(runtime_dir: Path) -> str:
    status = runtime_dir.stat()
    identity = sha256(f"{status.st_dev}:{status.st_ino}".encode()).hexdigest()[:16]
    return f"lmcache_l1_pool_llamacpp_{identity}_"


def _remove_stale_shared_memory(prefix: str) -> None:
    shared_memory_dir = Path("/dev/shm")
    if not shared_memory_dir.is_dir():
        return
    for path in shared_memory_dir.glob(f"{prefix}*"):
        try:
            status = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(status.st_mode) and status.st_uid == os.geteuid():
            shm_unlink(path.name)


def _request_exit(signum: int, _frame: FrameType | None) -> NoReturn:
    raise _ExitRequested(130 if signum == signal.SIGINT else 0)


if __name__ == "__main__":
    raise SystemExit(main())
