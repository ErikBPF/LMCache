# SPDX-License-Identifier: Apache-2.0
"""Opaque llama.cpp checkpoint storage over LMCache L1 and L2 tiers."""

# Standard
from collections.abc import Mapping
from hashlib import sha256
from typing import Any
import json
import struct
import threading
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

_MANIFEST_MODEL = "llamacpp-checkpoint-manifest-v1"
_CHUNK_MODEL = "llamacpp-checkpoint-chunk-v1"
_MANIFEST_VERSION = 1
_STANDALONE = AttnWindowDesc(
    num_chunks_in_sw=[-1],
    group_kinds=("standalone",),
)


class CheckpointCorruptError(ValueError):
    """Raised when a committed checkpoint fails integrity validation."""


class _StoreListener(L2AdapterListener):
    def __init__(self) -> None:
        self._stored: set[ObjectKey] = set()
        self._condition = threading.Condition()

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        with self._condition:
            self._stored.update(keys)
            self._condition.notify_all()

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        with self._condition:
            self._stored.difference_update(keys)

    def wait(self, keys: list[ObjectKey], timeout: float) -> bool:
        wanted = set(keys)
        deadline = time.monotonic() + timeout
        with self._condition:
            return self._condition.wait_for(
                lambda: wanted <= self._stored,
                timeout=max(0.0, deadline - time.monotonic()),
            )


class CheckpointStore:
    """Store immutable opaque checkpoints in an LMCache storage manager.

    Args:
        storage_manager: Configured LMCache L1 manager with zero or one L2 adapter.
        chunk_size: Bytes in each fixed-size data object.
        timeout: Seconds allowed for asynchronous L2 store and load operations.
        manifest_size: Bytes reserved for the fixed-size manifest object.

    Raises:
        ValueError: If limits are invalid or more than one L2 adapter is configured.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        chunk_size: int,
        timeout: float,
        manifest_size: int = 64 << 10,
    ) -> None:
        if chunk_size < 1 or manifest_size < 5 or timeout <= 0:
            raise ValueError("chunk_size, manifest_size, and timeout must be positive")
        if len(storage_manager.l2_adapters()) > 1:
            raise ValueError("CheckpointStore supports at most one L2 adapter")
        self._storage_manager = storage_manager
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._manifest_size = manifest_size
        self._listener = _StoreListener()
        storage_manager.register_l2_listener(self._listener)

    def store(
        self,
        cache_salt: str,
        revision: int,
        checkpoint: bytes,
        compatibility: Mapping[str, Any],
    ) -> None:
        """Store one immutable checkpoint and commit its manifest last.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Non-negative immutable session revision.
            checkpoint: Opaque llama.cpp slot-state bytes.
            compatibility: JSON-compatible model and engine identity fields.

        Raises:
            ValueError: If inputs are invalid, the revision already exists, or L1
                cannot reserve enough space.
            TimeoutError: If L2 does not persist an object before ``timeout``.
        """
        self._validate_identity(cache_salt, revision)
        compatibility_dict = dict(compatibility)
        chunks = [
            checkpoint[offset : offset + self._chunk_size]
            for offset in range(0, len(checkpoint), self._chunk_size)
        ]
        chunk_hashes = [sha256(chunk).digest() for chunk in chunks]
        chunk_keys = [self._chunk_key(cache_salt, digest) for digest in chunk_hashes]

        unique_chunks = dict(zip(chunk_keys, chunks, strict=True))
        new_chunk_keys = self._write_objects(
            unique_chunks,
            self._chunk_size,
        )
        self._wait_for_l2(new_chunk_keys)

        manifest = {
            "checkpoint_sha256": sha256(checkpoint).hexdigest(),
            "chunk_size": self._chunk_size,
            "chunks": [
                {"length": len(chunk), "sha256": digest.hex()}
                for chunk, digest in zip(chunks, chunk_hashes, strict=True)
            ],
            "compatibility": compatibility_dict,
            "length": len(checkpoint),
            "revision": revision,
            "version": _MANIFEST_VERSION,
        }
        manifest_data = self._encode_manifest(manifest)
        manifest_key = self._manifest_key(cache_salt, revision)
        new_manifest_keys = self._write_objects(
            {manifest_key: manifest_data},
            self._manifest_size,
        )
        if not new_manifest_keys:
            if self.load(cache_salt, revision, compatibility_dict) == checkpoint:
                return
            raise ValueError("checkpoint revision conflicts with existing state")
        self._wait_for_l2(new_manifest_keys)

    def load(
        self,
        cache_salt: str,
        revision: int,
        compatibility: Mapping[str, Any],
    ) -> bytes | None:
        """Load and verify a checkpoint, returning ``None`` for a cache miss.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Session revision to retrieve.
            compatibility: Required model and engine identity fields.

        Returns:
            Exact checkpoint bytes, or ``None`` when absent or incompatible.

        Raises:
            ValueError: If identity inputs are invalid.
            CheckpointCorruptError: If committed bytes fail manifest validation.
            TimeoutError: If an LMCache prefetch does not finish before ``timeout``.
        """
        self._validate_identity(cache_salt, revision)
        manifest_blob = self._read_objects(
            [self._manifest_key(cache_salt, revision)],
            self._manifest_size,
        )
        if manifest_blob is None:
            return None
        manifest = self._decode_manifest(manifest_blob[0])
        if manifest.get("revision") != revision or manifest.get(
            "compatibility"
        ) != dict(compatibility):
            return None

        try:
            chunk_records = manifest["chunks"]
            chunk_size = manifest["chunk_size"]
            checkpoint_length = manifest["length"]
            checkpoint_hash = manifest["checkpoint_sha256"]
            chunk_keys = [
                self._chunk_key(cache_salt, bytes.fromhex(record["sha256"]))
                for record in chunk_records
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptError("invalid checkpoint manifest") from exc
        if chunk_size != self._chunk_size or not isinstance(checkpoint_length, int):
            return None

        unique_chunk_keys = list(dict.fromkeys(chunk_keys))
        unique_chunks = self._read_objects(unique_chunk_keys, self._chunk_size)
        if unique_chunks is None:
            return None
        chunks_by_key = dict(zip(unique_chunk_keys, unique_chunks, strict=True))
        chunks = [chunks_by_key[key] for key in chunk_keys]
        try:
            payload = b"".join(
                chunk[: record["length"]]
                for chunk, record in zip(chunks, chunk_records, strict=True)
            )
            valid_chunks = all(
                sha256(chunk[: record["length"]]).hexdigest() == record["sha256"]
                for chunk, record in zip(chunks, chunk_records, strict=True)
            )
        except (KeyError, TypeError) as exc:
            raise CheckpointCorruptError("invalid checkpoint chunk metadata") from exc
        if (
            not valid_chunks
            or len(payload) != checkpoint_length
            or sha256(payload).hexdigest() != checkpoint_hash
        ):
            raise CheckpointCorruptError("checkpoint checksum mismatch")
        return payload

    def _write_objects(
        self,
        objects: Mapping[ObjectKey, bytes],
        object_size: int,
    ) -> list[ObjectKey]:
        if not objects:
            return []
        if self._storage_manager.l2_adapters() and not self._fits_in_available_l1(
            len(objects), object_size
        ):
            # ponytail: one-object staging; batch only if profiling proves need.
            new_keys: list[ObjectKey] = []
            for key, data in objects.items():
                written = self._write_object_batch({key: data}, object_size)
                self._wait_for_l2(written)
                _deleted, skipped = self._storage_manager.delete_l1_keys(written)
                if skipped:
                    raise RuntimeError("persisted checkpoint object remained in L1")
                new_keys.extend(written)
            return new_keys
        return self._write_object_batch(objects, object_size)

    def _write_object_batch(
        self,
        objects: Mapping[ObjectKey, bytes],
        object_size: int,
    ) -> list[ObjectKey]:
        layout = self._layout(object_size)
        reserved = self._storage_manager.reserve_write(
            list(objects), layout, mode="new"
        )
        for key, memory_obj in reserved.items():
            data = objects[key]
            target = memoryview(memory_obj.byte_array).cast("B")
            target[: len(data)] = data
            if len(data) < object_size:
                target[len(data) :] = b"\0" * (object_size - len(data))
        new_keys = list(reserved)
        self._storage_manager.finish_write(new_keys)
        return new_keys

    def _read_objects(
        self, keys: list[ObjectKey], object_size: int
    ) -> list[bytes] | None:
        if not keys:
            return []
        if self._storage_manager.l2_adapters() and not self._fits_in_available_l1(
            len(keys), object_size
        ):
            # ponytail: one-object staging; batch only if profiling proves need.
            result: list[bytes] = []
            for key in keys:
                item = self._read_object_batch([key], object_size)
                if item is None:
                    return None
                result.extend(item)
                _deleted, skipped = self._storage_manager.delete_l1_keys([key])
                if skipped:
                    raise RuntimeError("restored checkpoint object remained in L1")
            return result
        return self._read_object_batch(keys, object_size)

    def _read_object_batch(
        self, keys: list[ObjectKey], object_size: int
    ) -> list[bytes] | None:
        handle = self._storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=keys,
                group_layout_descs={0: self._layout(object_size)},
                policy=TrimPolicy.SPARSE,
                attn_desc=_STANDALONE,
            )
        )
        if not self._storage_manager.wait_prefetch_status(handle, self._timeout):
            raise LMCacheTimeoutError("checkpoint load timed out")
        found = self._storage_manager.query_prefetch_status(handle)
        if found is None:
            raise LMCacheTimeoutError("checkpoint load result unavailable")
        with self._storage_manager.read_prefetched_results(keys) as memory_objs:
            if found.popcount() != len(keys) or memory_objs is None:
                return None
            result = [bytes(memory_obj.byte_array) for memory_obj in memory_objs]
        self._storage_manager.finish_read_prefetched(keys)
        return result

    def _fits_in_available_l1(self, count: int, object_size: int) -> bool:
        used, total = self._storage_manager.get_l1_usage()
        return count * object_size <= total - used

    def _wait_for_l2(self, keys: list[ObjectKey]) -> None:
        if not keys or not self._storage_manager.l2_adapters():
            return
        deadline = time.monotonic() + self._timeout
        if not self._listener.wait(keys, self._timeout):
            raise LMCacheTimeoutError("checkpoint L2 store timed out")
        while time.monotonic() < deadline:
            status = self._storage_manager.report_status()["store_controller"]
            if (
                status["pending_keys_count"] == 0
                and status["in_flight_task_count"] == 0
            ):
                return
            time.sleep(0.001)
        raise LMCacheTimeoutError("checkpoint L2 completion timed out")

    def _encode_manifest(self, manifest: Mapping[str, Any]) -> bytes:
        encoded = json.dumps(
            manifest,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if len(encoded) + 4 > self._manifest_size:
            raise ValueError("checkpoint manifest exceeds manifest_size")
        return struct.pack(">I", len(encoded)) + encoded

    @staticmethod
    def _decode_manifest(blob: bytes) -> dict[str, Any]:
        try:
            length = struct.unpack(">I", blob[:4])[0]
            manifest = json.loads(blob[4 : 4 + length])
        except (json.JSONDecodeError, struct.error, UnicodeDecodeError) as exc:
            raise CheckpointCorruptError("invalid checkpoint manifest") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != _MANIFEST_VERSION
        ):
            raise CheckpointCorruptError("unsupported checkpoint manifest")
        return manifest

    @staticmethod
    def _layout(size: int) -> MemoryLayoutDesc:
        return MemoryLayoutDesc(shapes=[torch.Size([size])], dtypes=[torch.uint8])

    @staticmethod
    def _manifest_key(cache_salt: str, revision: int) -> ObjectKey:
        digest = sha256(f"manifest\0{revision}".encode()).digest()
        return ObjectKey(digest, _MANIFEST_MODEL, kv_rank=0, cache_salt=cache_salt)

    @staticmethod
    def _chunk_key(cache_salt: str, digest: bytes) -> ObjectKey:
        return ObjectKey(digest, _CHUNK_MODEL, kv_rank=0, cache_salt=cache_salt)

    @staticmethod
    def _validate_identity(cache_salt: str, revision: int) -> None:
        if (
            len(cache_salt) != 64
            or cache_salt.lower() != cache_salt
            or any(char not in "0123456789abcdef" for char in cache_salt)
        ):
            raise ValueError("cache_salt must be a lowercase 64-character hex digest")
        if revision < 0:
            raise ValueError("revision must be non-negative")
