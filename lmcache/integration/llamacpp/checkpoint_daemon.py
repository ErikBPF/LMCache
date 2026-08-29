# SPDX-License-Identifier: Apache-2.0
"""Construct the bounded local llama.cpp checkpoint service."""

# Standard
from argparse import ArgumentParser
from pathlib import Path
from types import FrameType
from typing import NoReturn
import os
import signal
import stat

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager

# Local
from .checkpoint_bridge import CheckpointBridge
from .checkpoint_service import CheckpointService
from .checkpoint_store import CheckpointStore


class _ExitRequested(Exception):
    def __init__(self, exit_code: int) -> None:
        super().__init__()
        self.exit_code = exit_code


def create_checkpoint_service(
    runtime_dir: Path,
    cache_dir: Path,
    llama_url: str,
    slot_id: int,
    l1_size_bytes: int,
    l2_size_gb: float,
    chunk_size: int,
    timeout: float,
    max_request_bytes: int,
) -> tuple[CheckpointService, StorageManager]:
    """Create a private service with bounded LMCache RAM and filesystem tiers.

    Args:
        runtime_dir: Private directory for the socket and llama transfer files.
        cache_dir: Private local filesystem L2 directory.
        llama_url: Trusted llama-server base URL.
        slot_id: Serialized active llama-server slot.
        l1_size_bytes: LMCache RAM capacity in bytes.
        l2_size_gb: LMCache filesystem capacity in GiB.
        chunk_size: Opaque checkpoint chunk size in bytes.
        timeout: Storage and llama-server timeout in seconds.
        max_request_bytes: Maximum Unix control request size.

    Returns:
        Bound checkpoint service and its storage manager.

    Raises:
        ValueError: If a capacity is not positive or a private directory is unsafe.
    """
    if l1_size_bytes <= 0 or l2_size_gb <= 0:
        raise ValueError("LMCache RAM and disk capacities must be positive")
    _private_directory(runtime_dir)
    _private_directory(cache_dir)
    transfer_dir = runtime_dir / "transfer"
    _private_directory(transfer_dir)
    socket_path = runtime_dir / "bridge.sock"
    _remove_stale_socket(socket_path)

    l2_config = FSNativeL2AdapterConfig(
        base_path=str(cache_dir),
        max_capacity_gb=l2_size_gb,
    )
    l2_config.eviction_config = EvictionConfig(
        eviction_policy="LRU",
        trigger_watermark=0.8,
        eviction_ratio=0.2,
    )
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=l1_size_bytes,
                    use_lazy=False,
                    init_size_in_bytes=l1_size_bytes,
                    shm_name="",
                )
            ),
            eviction_config=EvictionConfig(
                eviction_policy="LRU",
                trigger_watermark=0.8,
                eviction_ratio=0.2,
            ),
            l2_adapter_config=L2AdaptersConfig(adapters=[l2_config]),
        )
    )
    try:
        store = CheckpointStore(manager, chunk_size=chunk_size, timeout=timeout)
        bridge = CheckpointBridge(
            store,
            transfer_dir=transfer_dir,
            llama_url=llama_url,
            slot_id=slot_id,
            timeout=timeout,
        )
        service = CheckpointService(
            bridge,
            socket_path,
            max_request_bytes=max_request_bytes,
        )
    except Exception:
        manager.close()
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
    parser.add_argument("--llama-url", required=True)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--l1-size-bytes", type=int, required=True)
    parser.add_argument("--l2-size-gb", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-request-bytes", type=int, default=64 << 10)
    args = parser.parse_args(argv)

    service, manager = create_checkpoint_service(
        runtime_dir=args.runtime_dir,
        cache_dir=args.cache_dir,
        llama_url=args.llama_url,
        slot_id=args.slot_id,
        l1_size_bytes=args.l1_size_bytes,
        l2_size_gb=args.l2_size_gb,
        chunk_size=args.chunk_size,
        timeout=args.timeout,
        max_request_bytes=args.max_request_bytes,
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
        service.server_close()
        manager.close()
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


def _request_exit(signum: int, _frame: FrameType | None) -> NoReturn:
    raise _ExitRequested(130 if signum == signal.SIGINT else 0)


if __name__ == "__main__":
    raise SystemExit(main())
