# SPDX-License-Identifier: Apache-2.0
"""Opaque llama.cpp checkpoint storage integration tests."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp.checkpoint_store import (
    CheckpointCorruptError,
    CheckpointStore,
)
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


def test_multichunk_checkpoint_restores_from_l2_after_l1_removal(
    tmp_path: Path,
) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = bytes(range(251)) * 40
    compatibility = {
        "context_size": 65536,
        "llama_cpp_commit": "86632248188c106d749fad34a1dcd237c95863d4",
        "model": "Qwen3.8-27B-UD-IQ3_XXS-v3",
    }

    try:
        store.store("a" * 64, 7, payload, compatibility)
        manager.clear()

        assert manager.get_l1_usage()[0] == 0

        restored = store.load("a" * 64, 7, compatibility)

        assert restored == payload
        assert manager.get_l2_usages()[0][0] > len(payload)
    finally:
        manager.close()


def test_same_length_chunk_corruption_is_rejected(tmp_path: Path) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    compatibility = {"model": "Qwen3.8-27B-UD-IQ3_XXS-v3"}

    try:
        store.store("b" * 64, 3, b"checkpoint" * 1000, compatibility)
        manager.clear()
        chunk_path = next(tmp_path.glob("llamacpp-checkpoint-chunk-v1@*.data"))
        corrupted = bytearray(chunk_path.read_bytes())
        corrupted[0] ^= 1
        chunk_path.write_bytes(corrupted)

        with pytest.raises(CheckpointCorruptError, match="checksum"):
            store.load("b" * 64, 3, compatibility)
    finally:
        manager.close()


def test_incompatible_checkpoint_is_a_cache_miss(tmp_path: Path) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)

    try:
        store.store("c" * 64, 1, b"checkpoint", {"model": "qwen"})

        restored = store.load("c" * 64, 1, {"model": "different"})

        assert restored is None
    finally:
        manager.close()


def test_repeated_chunks_preserve_checkpoint_positions(tmp_path: Path) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = b"x" * 8192

    try:
        store.store("d" * 64, 2, payload, {"model": "qwen"})
        manager.clear()

        assert store.load("d" * 64, 2, {"model": "qwen"}) == payload
    finally:
        manager.close()
