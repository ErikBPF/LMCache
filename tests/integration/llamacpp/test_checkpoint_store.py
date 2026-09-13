# SPDX-License-Identifier: Apache-2.0
"""Opaque llama.cpp checkpoint storage integration tests."""

# Standard
from contextlib import contextmanager
from hashlib import sha256 as real_sha256
from pathlib import Path
from threading import Event, Lock
from time import sleep
import ctypes
import json
import struct

# Third Party
import pytest

# First Party
from lmcache.integration.llamacpp import checkpoint_store
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


def _storage_manager(
    cache_dir: Path,
    l1_size_bytes: int = 4 << 20,
) -> StorageManager:
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=l1_size_bytes,
                    use_lazy=False,
                    init_size_in_bytes=l1_size_bytes,
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


def test_read_checkpoint_exposes_validated_chunk_views(tmp_path: Path) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = b"a" * 4096 + b"b" * 4096 + b"tail"
    compatibility = {"model": "qwen"}

    try:
        store.store("7" * 64, 1, payload, compatibility)
        manager.clear()

        with store.read_checkpoint("7" * 64, 1, compatibility) as restored:
            assert restored is not None
            length, chunks = restored
            assert length == len(payload)
            assert len(chunks) == 3
            assert all(isinstance(chunk, memoryview) for chunk in chunks)
            assert b"".join(chunks) == payload
    finally:
        manager.close()


def test_checkpoint_larger_than_l1_streams_through_l2(tmp_path: Path) -> None:
    chunk_size = 64 << 10
    manager = _storage_manager(tmp_path, l1_size_bytes=2 * chunk_size)
    store = CheckpointStore(manager, chunk_size=chunk_size, timeout=5.0)
    payload = b"a" * chunk_size + b"b" * chunk_size + b"c" * chunk_size
    compatibility = {"model": "qwen"}

    try:
        store.store("e" * 64, 1, payload, compatibility)
        manager.clear()

        assert store.load("e" * 64, 1, compatibility) == payload
    finally:
        manager.close()


def test_restore_retries_when_l1_key_disappears_after_prefetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = b"checkpoint" * 1000
    compatibility = {"model": "qwen"}
    original_read = manager.read_prefetched_results
    evict = True

    @contextmanager
    def evict_before_first_read(keys):
        nonlocal evict
        if evict:
            evict = False
            manager._l1_manager.delete(keys, force=True)
        with original_read(keys) as objects:
            yield objects

    try:
        store.store("1" * 64, 1, payload, compatibility)
        monkeypatch.setattr(manager, "read_prefetched_results", evict_before_first_read)

        assert store.load("1" * 64, 1, compatibility) == payload
    finally:
        manager.close()


def test_stream_restore_clears_l1_headroom_for_one_l2_chunk(tmp_path: Path) -> None:
    chunk_size = 64 << 10
    manager = _storage_manager(tmp_path, l1_size_bytes=4 * chunk_size)
    store = CheckpointStore(
        manager,
        chunk_size=chunk_size,
        manifest_size=4096,
        timeout=5.0,
    )
    compatibility = {"model": "qwen"}
    payloads = {
        salt: b"".join(bytes([value + offset]) * chunk_size for offset in range(3))
        for salt, value in zip(("1", "2", "3", "4"), range(1, 5), strict=True)
    }

    try:
        for salt, payload in payloads.items():
            store.store(salt * 64, 1, payload, compatibility)

        assert store.load("2" * 64, 1, compatibility) == payloads["2"]
    finally:
        manager.close()


def test_same_length_chunk_corruption_is_rejected(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        assert "finish read on non-existing key" not in caplog.text
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


def test_store_does_not_slice_checkpoint_bytes(tmp_path: Path) -> None:
    class UnsliceableBytes(bytes):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise AssertionError("checkpoint bytes were copied by slicing")
            return super().__getitem__(key)

    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = UnsliceableBytes(b"x" * 8192)

    try:
        store.store("9" * 64, 1, payload, {"model": "qwen"})
        manager.clear()

        assert store.load("9" * 64, 1, {"model": "qwen"}) == payload
    finally:
        manager.close()


def test_store_hashes_large_checkpoint_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0
    lock = Lock()

    def tracked_sha256(data=b""):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            sleep(0.01)
            return real_sha256(data)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(checkpoint_store, "sha256", tracked_sha256)
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=512 << 10, timeout=5.0)

    try:
        store.store("8" * 64, 1, b"x" * (2 << 20), {"model": "qwen"})

        assert peak > 1
    finally:
        manager.close()


def test_store_from_reader_hashes_while_receiving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    first_hash_started = Event()
    payload = b"a" * 4096 + b"b" * 4096
    offset = 0

    def tracked_sha256(data=b""):
        if data == payload[:4096]:
            first_hash_started.set()
        return real_sha256(data)

    def read(size: int) -> bytes:
        nonlocal offset
        if offset == 4096:
            assert first_hash_started.wait(1)
        chunk = payload[offset : offset + size]
        offset += len(chunk)
        return chunk

    monkeypatch.setattr(checkpoint_store, "sha256", tracked_sha256)
    try:
        store.store_from_reader(
            "8" * 64,
            2,
            len(payload),
            read,
            {"model": "qwen"},
        )

        assert store.load("8" * 64, 2, {"model": "qwen"}) == payload
    finally:
        manager.close()


def test_store_from_reader_copies_writable_chunks_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _storage_manager(tmp_path, l1_size_bytes=4 << 20)
    store = CheckpointStore(manager, chunk_size=512 << 10, timeout=5.0)
    payload = bytearray(b"".join(bytes([value]) * (512 << 10) for value in range(4)))
    view = memoryview(payload)
    offset = 0
    active = 0
    peak = 0
    lock = Lock()
    real_memmove = ctypes.memmove

    def tracked_memmove(destination, source, count):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            sleep(0.01)
            return real_memmove(destination, source, count)
        finally:
            with lock:
                active -= 1

    def read(size: int) -> memoryview:
        nonlocal offset
        chunk = view[offset : offset + size]
        offset += len(chunk)
        return chunk

    monkeypatch.setattr(ctypes, "memmove", tracked_memmove)
    try:
        store.store_from_reader(
            "b" * 64,
            1,
            len(payload),
            read,
            {"model": "qwen"},
        )

        assert peak > 1
        assert store.load("b" * 64, 1, {"model": "qwen"}) == payload
    finally:
        view.release()
        manager.close()


def test_load_hashes_each_checkpoint_chunk_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = b"a" * 4096 + b"b" * 4096
    compatibility = {"model": "qwen"}

    try:
        store.store("6" * 64, 1, payload, compatibility)
        calls = 0

        def tracked_sha256(data=b""):
            nonlocal calls
            calls += 1
            return real_sha256(data)

        monkeypatch.setattr(checkpoint_store, "sha256", tracked_sha256)

        assert store.load("6" * 64, 1, compatibility) == payload
        assert calls == 4  # Manifest key, two chunks, and chunk-digest checksum.
    finally:
        manager.close()


def test_validate_once_skips_rehash_after_verified_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(
        manager,
        chunk_size=4096,
        timeout=5.0,
        validate_once=True,
    )
    payload = b"a" * 4096 + b"b" * 4096
    compatibility = {"model": "qwen"}

    try:
        store.store("9" * 64, 1, payload, compatibility)
        calls = 0

        def tracked_sha256(data=b""):
            nonlocal calls
            calls += 1
            return real_sha256(data)

        monkeypatch.setattr(checkpoint_store, "sha256", tracked_sha256)

        assert store.load("9" * 64, 1, compatibility) == payload
        assert calls == 1  # Manifest lookup only; stored checkpoint is trusted.
    finally:
        manager.close()


def test_load_hashes_large_checkpoint_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0
    lock = Lock()
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=512 << 10, timeout=5.0)
    payload = b"x" * (2 << 20)
    compatibility = {"model": "qwen"}

    try:
        store.store("6" * 64, 1, payload, compatibility)

        def tracked_sha256(data=b""):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                sleep(0.01)
                return real_sha256(data)
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr(checkpoint_store, "sha256", tracked_sha256)

        assert store.load("6" * 64, 1, compatibility) == payload
        assert peak > 1
    finally:
        manager.close()


def test_load_accepts_legacy_whole_checkpoint_checksum(tmp_path: Path) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    payload = b"legacy-checkpoint" * 1000
    compatibility = {"model": "qwen"}

    try:
        store.store("5" * 64, 1, payload, compatibility)
        manager.clear()
        path = next(tmp_path.glob("llamacpp-checkpoint-manifest-v1@*.data"))
        blob = bytearray(path.read_bytes())
        length = struct.unpack(">I", blob[:4])[0]
        manifest = json.loads(blob[4 : 4 + length])
        manifest.pop("checkpoint_hash_scheme")
        manifest["checkpoint_sha256"] = real_sha256(payload).hexdigest()
        encoded = json.dumps(
            manifest,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        blob[:] = (
            struct.pack(">I", len(encoded))
            + encoded
            + bytes(len(blob) - len(encoded) - 4)
        )
        path.write_bytes(blob)

        assert store.load("5" * 64, 1, compatibility) == payload
    finally:
        manager.close()


def test_delete_revision_keeps_chunks_shared_with_retained_head(
    tmp_path: Path,
) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    cache_salt = "f" * 64
    compatibility = {"model": "qwen"}
    shared = b"s" * 4096
    retained = shared + b"n" * 4096

    try:
        store.store(cache_salt, 1, shared + b"o" * 4096, compatibility)
        store.store(cache_salt, 2, retained, compatibility)

        store.delete_revision(cache_salt, 1, retained_revision=2)
        store.delete_revision(cache_salt, 1, retained_revision=2)
        manager.clear()

        assert store.load(cache_salt, 1, compatibility) is None
        assert store.load(cache_salt, 2, compatibility) == retained
        assert len(list(tmp_path.glob("llamacpp-checkpoint-chunk-v1@*.data"))) == 2
        assert len(list(tmp_path.glob("llamacpp-checkpoint-manifest-v1@*.data"))) == 1
    finally:
        manager.close()


def test_delete_revision_resumes_after_missing_old_chunk(tmp_path: Path) -> None:
    manager = _storage_manager(tmp_path)
    store = CheckpointStore(manager, chunk_size=4096, timeout=5.0)
    cache_salt = "0" * 64
    compatibility = {"model": "qwen"}
    shared = b"s" * 4096
    retained = shared + b"n" * 4096

    try:
        store.store(cache_salt, 1, shared + b"o" * 4096, compatibility)
        store.store(cache_salt, 2, retained, compatibility)
        manager.clear()
        old_chunk = next(
            path
            for path in tmp_path.glob("llamacpp-checkpoint-chunk-v1@*.data")
            if path.read_bytes().startswith(b"o" * 4096)
        )
        old_chunk.unlink()

        store.delete_revision(cache_salt, 1, retained_revision=2)
        manager.clear()

        assert store.load(cache_salt, 1, compatibility) is None
        assert store.load(cache_salt, 2, compatibility) == retained
    finally:
        manager.close()
