# SPDX-License-Identifier: Apache-2.0
"""Opaque llama.cpp checkpoint storage over LMCache L1 and L2 tiers."""

# Standard
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from typing import Any, Iterator
import ctypes
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
_CHECKPOINT_HASH_SCHEME = "chunk-sha256-v1"
_STANDALONE = AttnWindowDesc(
    num_chunks_in_sw=[-1],
    group_kinds=("standalone",),
)


def _sha256_digest(data: bytes | memoryview) -> bytes:
    return sha256(data).digest()


def _checkpoint_hashes(
    chunks: list[bytes | memoryview],
) -> tuple[list[bytes], str]:
    if sum(map(len, chunks)) >= 1 << 20 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 7)) as executor:
            chunk_hashes = list(executor.map(_sha256_digest, chunks))
    else:
        chunk_hashes = [_sha256_digest(chunk) for chunk in chunks]
    checkpoint_hash = sha256()
    for digest in chunk_hashes:
        checkpoint_hash.update(digest)
    return chunk_hashes, checkpoint_hash.hexdigest()


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
        validate_once: Reuse successful integrity validation for this process.

    Raises:
        ValueError: If limits are invalid or more than one L2 adapter is configured.
    """

    def __init__(
        self,
        storage_manager: StorageManager,
        chunk_size: int,
        timeout: float,
        manifest_size: int = 64 << 10,
        validate_once: bool = False,
    ) -> None:
        if chunk_size < 1 or manifest_size < 5 or timeout <= 0:
            raise ValueError("chunk_size, manifest_size, and timeout must be positive")
        if len(storage_manager.l2_adapters()) > 1:
            raise ValueError("CheckpointStore supports at most one L2 adapter")
        self._storage_manager = storage_manager
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._manifest_size = manifest_size
        self._validate_once = validate_once
        self._validated: set[tuple[str, int, str]] = set()
        self._listener = _StoreListener()
        storage_manager.register_l2_listener(self._listener)

    def store(
        self,
        cache_salt: str,
        revision: int,
        checkpoint: bytes,
        compatibility: Mapping[str, Any],
        timings: dict[str, float] | None = None,
    ) -> None:
        """Store one immutable checkpoint and commit its manifest last.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Non-negative immutable session revision.
            checkpoint: Opaque llama.cpp slot-state bytes.
            compatibility: JSON-compatible model and engine identity fields.
            timings: Optional mutable mapping populated with stage milliseconds.

        Raises:
            ValueError: If inputs are invalid, the revision already exists, or L1
                cannot reserve enough space.
            TimeoutError: If L2 does not persist an object before ``timeout``.
        """
        self._validate_identity(cache_salt, revision)
        compatibility_dict = dict(compatibility)
        checkpoint_view = memoryview(checkpoint)
        chunks = [
            checkpoint_view[offset : offset + self._chunk_size]
            for offset in range(0, len(checkpoint), self._chunk_size)
        ]
        stage_start = time.perf_counter()
        chunk_hashes, checkpoint_hash = _checkpoint_hashes(chunks)
        if timings is not None:
            timings["hash"] = (time.perf_counter() - stage_start) * 1000
        self._commit(
            cache_salt,
            revision,
            chunks,
            chunk_hashes,
            checkpoint_hash,
            compatibility_dict,
            len(checkpoint),
            timings,
        )

    def store_from_reader(
        self,
        cache_salt: str,
        revision: int,
        checkpoint_length: int,
        reader: Callable[[int], bytes | memoryview],
        compatibility: Mapping[str, Any],
        timings: dict[str, float] | None = None,
    ) -> None:
        """Receive and store a checkpoint while hashing incoming chunks.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Non-negative immutable session revision.
            checkpoint_length: Exact number of checkpoint bytes to read.
            reader: Callable returning up to the requested number of bytes.
            compatibility: JSON-compatible model and engine identity fields.
            timings: Optional mutable mapping populated with stage milliseconds.

        Raises:
            ValueError: If inputs are invalid, the revision exists with different
                bytes, or L1 cannot reserve enough space.
            EOFError: If the reader ends before ``checkpoint_length`` bytes.
            TimeoutError: If L2 does not persist an object before ``timeout``.
        """
        self._validate_identity(cache_salt, revision)
        if checkpoint_length < 0:
            raise ValueError("checkpoint_length must be non-negative")

        receive_ms = 0.0
        chunks: list[bytes | memoryview] = []
        futures = []
        hash_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=7) as executor:
            remaining = checkpoint_length
            while remaining:
                wanted = min(remaining, self._chunk_size)
                parts: list[bytes | memoryview] = []
                received = 0
                while received < wanted:
                    stage_start = time.perf_counter()
                    part = reader(wanted - received)
                    receive_ms += (time.perf_counter() - stage_start) * 1000
                    if not part:
                        raise EOFError("checkpoint reader ended early")
                    if len(part) > wanted - received:
                        raise ValueError("checkpoint reader returned too many bytes")
                    parts.append(part)
                    received += len(part)
                chunk = parts[0] if len(parts) == 1 else b"".join(parts)
                chunks.append(chunk)
                futures.append(executor.submit(_sha256_digest, chunk))
                remaining -= len(chunk)

            hash_wait_start = time.perf_counter()
            chunk_hashes = [future.result() for future in futures]
            hash_wait_ms = (time.perf_counter() - hash_wait_start) * 1000

        checkpoint_hash = sha256()
        for digest in chunk_hashes:
            checkpoint_hash.update(digest)
        if timings is not None:
            timings["receive"] = receive_ms
            timings["hash"] = (time.perf_counter() - hash_start) * 1000
            timings["hash_wait"] = hash_wait_ms
        self._commit(
            cache_salt,
            revision,
            chunks,
            chunk_hashes,
            checkpoint_hash.hexdigest(),
            dict(compatibility),
            checkpoint_length,
            timings,
        )

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
        with self.read_checkpoint(cache_salt, revision, compatibility) as restored:
            if restored is None:
                return None
            _length, chunks = restored
            return b"".join(chunks)

    @contextmanager
    def read_checkpoint_regions(
        self,
        cache_salt: str,
        revision: int,
        compatibility: Mapping[str, Any],
        timings: dict[str, float] | None = None,
    ) -> Iterator[
        tuple[
            int,
            tuple[memoryview, ...],
            tuple[tuple[int, int], ...] | None,
        ]
        | None
    ]:
        """Open a checkpoint and locate chunks inside the contiguous L1 arena.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Session revision to retrieve.
            compatibility: Required model and engine identity fields.
            timings: Optional mutable mapping populated with stage milliseconds.

        Yields:
            Length, validated chunk views, and ``(offset, length)`` L1 regions.
            Regions are ``None`` when this read is not backed by the L1 arena.
        """
        with self.read_checkpoint(
            cache_salt,
            revision,
            compatibility,
            timings,
        ) as restored:
            if restored is None:
                yield None
                return
            length, chunks = restored
            desc = self._storage_manager.get_l1_memory_desc()
            regions: list[tuple[int, int]] = []
            if desc is not None:
                for chunk in chunks:
                    try:
                        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(chunk))
                    except (TypeError, ValueError):
                        break
                    offset = address - desc.ptr
                    if offset < 0 or offset + len(chunk) > desc.size:
                        break
                    regions.append((offset, len(chunk)))
            yield (
                length,
                chunks,
                (tuple(regions) if len(regions) == len(chunks) else None),
            )

    @contextmanager
    def read_checkpoint(
        self,
        cache_salt: str,
        revision: int,
        compatibility: Mapping[str, Any],
        timings: dict[str, float] | None = None,
    ) -> Iterator[tuple[int, tuple[memoryview, ...]] | None]:
        """Open one validated checkpoint as locked LMCache chunk views.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Session revision to retrieve.
            compatibility: Required model and engine identity fields.
            timings: Optional mutable mapping populated with stage milliseconds.

        Yields:
            Checkpoint length and views valid until context exit, or ``None`` for
            a cache miss or incompatible checkpoint.

        Raises:
            ValueError: If identity inputs are invalid.
            CheckpointCorruptError: If committed bytes fail integrity validation.
            TimeoutError: If an LMCache prefetch does not finish before ``timeout``.
        """
        self._validate_identity(cache_salt, revision)
        stage_start = time.perf_counter()
        manifest_blob = self._read_objects(
            [self._manifest_key(cache_salt, revision)],
            self._manifest_size,
        )
        if manifest_blob is None:
            yield None
            return
        manifest = self._decode_manifest(manifest_blob[0])
        if timings is not None:
            timings["manifest_read"] = (time.perf_counter() - stage_start) * 1000
        if manifest.get("revision") != revision or manifest.get(
            "compatibility"
        ) != dict(compatibility):
            yield None
            return

        try:
            chunk_records = manifest["chunks"]
            chunk_size = manifest["chunk_size"]
            checkpoint_length = manifest["length"]
            expected_checkpoint_hash = manifest["checkpoint_sha256"]
            chunk_keys = [
                self._chunk_key(cache_salt, bytes.fromhex(record["sha256"]))
                for record in chunk_records
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptError("invalid checkpoint manifest") from exc
        if chunk_size != self._chunk_size or not isinstance(checkpoint_length, int):
            yield None
            return
        validation_key = (cache_salt, revision, expected_checkpoint_hash)
        verify_hashes = not (self._validate_once and validation_key in self._validated)

        unique_chunk_keys = list(dict.fromkeys(chunk_keys))
        if self._storage_manager.l2_adapters() and not self._fits_in_available_l1(
            len(unique_chunk_keys), self._chunk_size
        ):
            stage_start = time.perf_counter()
            unique_chunks = self._read_objects(unique_chunk_keys, self._chunk_size)
            if timings is not None:
                timings["chunk_read"] = (time.perf_counter() - stage_start) * 1000
            if unique_chunks is None:
                yield None
                return
            unique_views = [memoryview(chunk) for chunk in unique_chunks]
            chunks_by_key = dict(zip(unique_chunk_keys, unique_views, strict=True))
            stage_start = time.perf_counter()
            chunks = self._validated_chunks(
                [chunks_by_key[key] for key in chunk_keys],
                chunk_records,
                checkpoint_length,
                expected_checkpoint_hash,
                manifest.get("checkpoint_hash_scheme"),
                verify_hashes,
            )
            if self._validate_once:
                self._validated.add(validation_key)
            if timings is not None:
                timings["validate"] = (time.perf_counter() - stage_start) * 1000
                timings["validate_skipped"] = float(not verify_hashes)
            yield checkpoint_length, chunks
            return

        stage_start = time.perf_counter()
        with self._read_object_views(
            unique_chunk_keys,
            self._chunk_size,
        ) as unique_views:
            if timings is not None:
                timings["chunk_read"] = (time.perf_counter() - stage_start) * 1000
            if unique_views is None:
                yield None
                return
            chunks_by_key = dict(zip(unique_chunk_keys, unique_views, strict=True))
            corruption = None
            try:
                stage_start = time.perf_counter()
                chunks = self._validated_chunks(
                    [chunks_by_key[key] for key in chunk_keys],
                    chunk_records,
                    checkpoint_length,
                    expected_checkpoint_hash,
                    manifest.get("checkpoint_hash_scheme"),
                    verify_hashes,
                )
                if self._validate_once:
                    self._validated.add(validation_key)
                if timings is not None:
                    timings["validate"] = (time.perf_counter() - stage_start) * 1000
                    timings["validate_skipped"] = float(not verify_hashes)
            except CheckpointCorruptError as exc:
                corruption = exc
            if corruption is None:
                yield checkpoint_length, chunks
        if corruption is not None:
            raise corruption

    def delete_revision(
        self,
        cache_salt: str,
        revision: int,
        retained_revision: int,
    ) -> None:
        """Delete one obsolete revision without deleting shared checkpoint chunks.

        Args:
            cache_salt: Lowercase 64-character digest isolating one session.
            revision: Obsolete session revision to delete.
            retained_revision: Revision whose chunks must remain available.

        Raises:
            ValueError: If identities are invalid or the retained revision is absent.
            CheckpointCorruptError: If either manifest has invalid chunk metadata.
            RuntimeError: If an L1 object is locked and cannot be deleted.
        """
        self._validate_identity(cache_salt, revision)
        self._validate_identity(cache_salt, retained_revision)
        if revision == retained_revision:
            raise ValueError("revision and retained_revision must differ")

        manifest_key = self._manifest_key(cache_salt, revision)
        manifest_blob = self._read_objects([manifest_key], self._manifest_size)
        if manifest_blob is None:
            return
        retained_blob = self._read_objects(
            [self._manifest_key(cache_salt, retained_revision)],
            self._manifest_size,
        )
        if retained_blob is None:
            raise ValueError("retained checkpoint revision does not exist")

        manifest = self._decode_manifest(manifest_blob[0])
        retained_manifest = self._decode_manifest(retained_blob[0])
        if (
            manifest.get("revision") != revision
            or retained_manifest.get("revision") != retained_revision
        ):
            raise CheckpointCorruptError("checkpoint manifest revision mismatch")
        retained_chunks = set(self._manifest_chunk_keys(cache_salt, retained_manifest))
        obsolete_chunks = [
            key
            for key in dict.fromkeys(self._manifest_chunk_keys(cache_salt, manifest))
            if key not in retained_chunks
        ]

        # Chunks first keeps the manifest available to resume an interrupted delete.
        self._delete_objects(obsolete_chunks)
        self._delete_objects([manifest_key])
        self._validated = {
            key for key in self._validated if key[:2] != (cache_salt, revision)
        }

    def _commit(
        self,
        cache_salt: str,
        revision: int,
        chunks: list[bytes | memoryview],
        chunk_hashes: list[bytes],
        checkpoint_hash: str,
        compatibility: dict[str, Any],
        checkpoint_length: int,
        timings: dict[str, float] | None,
    ) -> None:
        chunk_keys = [self._chunk_key(cache_salt, digest) for digest in chunk_hashes]
        unique_chunks = dict(zip(chunk_keys, chunks, strict=True))
        stage_start = time.perf_counter()
        new_chunk_keys = self._write_objects(unique_chunks, self._chunk_size)
        if timings is not None:
            timings["chunk_write"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        self._wait_for_l2(new_chunk_keys)
        if timings is not None:
            timings["wait"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        manifest = {
            "checkpoint_hash_scheme": _CHECKPOINT_HASH_SCHEME,
            "checkpoint_sha256": checkpoint_hash,
            "chunk_size": self._chunk_size,
            "chunks": [
                {"length": len(chunk), "sha256": digest.hex()}
                for chunk, digest in zip(chunks, chunk_hashes, strict=True)
            ],
            "compatibility": compatibility,
            "length": checkpoint_length,
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
            if self.load(cache_salt, revision, compatibility) == b"".join(chunks):
                if timings is not None:
                    timings["manifest"] = (time.perf_counter() - stage_start) * 1000
                return
            raise ValueError("checkpoint revision conflicts with existing state")
        self._wait_for_l2(new_manifest_keys)
        if self._validate_once:
            self._validated.add((cache_salt, revision, checkpoint_hash))
        if timings is not None:
            timings["manifest"] = (time.perf_counter() - stage_start) * 1000

    def _write_objects(
        self,
        objects: Mapping[ObjectKey, bytes | memoryview],
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
                if not self._fits_in_available_l1(1, object_size):
                    self._storage_manager.clear()
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
        objects: Mapping[ObjectKey, bytes | memoryview],
        object_size: int,
    ) -> list[ObjectKey]:
        layout = self._layout(object_size)
        reserved = self._storage_manager.reserve_write(
            list(objects), layout, mode="new"
        )

        def copy_object(item: tuple[ObjectKey, Any]) -> None:
            key, memory_obj = item
            data = objects[key]
            if isinstance(data, memoryview) and not data.readonly and data.c_contiguous:
                source = ctypes.addressof(ctypes.c_ubyte.from_buffer(data))
                ctypes.memmove(memory_obj.data_ptr, source, len(data))
                if len(data) < object_size:
                    ctypes.memset(
                        memory_obj.data_ptr + len(data),
                        0,
                        object_size - len(data),
                    )
                return
            target = memoryview(memory_obj.byte_array).cast("B")
            target[: len(data)] = data
            if len(data) < object_size:
                target[len(data) :] = b"\0" * (object_size - len(data))

        items = list(reserved.items())
        if len(items) > 1 and sum(map(len, objects.values())) >= 1 << 20:
            with ThreadPoolExecutor(max_workers=min(len(items), 7)) as executor:
                list(executor.map(copy_object, items))
        else:
            for item in items:
                copy_object(item)
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
                if not self._fits_in_available_l1(1, object_size):
                    self._storage_manager.clear()
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
        with self._read_object_views(keys, object_size) as views:
            if views is None:
                return None
            return [bytes(view) for view in views]

    @contextmanager
    def _read_object_views(
        self,
        keys: list[ObjectKey],
        object_size: int,
    ) -> Iterator[list[memoryview] | None]:
        if not keys:
            yield []
            return
        for attempt in range(2):
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
            complete = found.popcount() == len(keys)
            with self._storage_manager.read_prefetched_results(keys) as memory_objs:
                if memory_objs is None:
                    if complete and attempt == 0:
                        continue
                    yield None
                    return
                if not complete:
                    yield None
                else:
                    yield [
                        memoryview(memory_obj.byte_array).cast("B")
                        for memory_obj in memory_objs
                    ]
            self._storage_manager.finish_read_prefetched(keys)
            return

    @staticmethod
    def _validated_chunks(
        chunks: list[memoryview],
        records: list[dict[str, Any]],
        checkpoint_length: int,
        expected_checkpoint_hash: str,
        checkpoint_hash_scheme: str | None,
        verify_hashes: bool = True,
    ) -> tuple[memoryview, ...]:
        try:
            views = tuple(
                chunk[: record["length"]]
                for chunk, record in zip(chunks, records, strict=True)
            )
        except (KeyError, TypeError) as exc:
            raise CheckpointCorruptError("invalid checkpoint chunk metadata") from exc
        if sum(map(len, views)) != checkpoint_length:
            raise CheckpointCorruptError("checkpoint checksum mismatch")
        if checkpoint_hash_scheme is None:
            if not verify_hashes:
                return views
            checkpoint_hash = sha256()
            for chunk in views:
                checkpoint_hash.update(chunk)
            if checkpoint_hash.hexdigest() != expected_checkpoint_hash:
                raise CheckpointCorruptError("checkpoint checksum mismatch")
            return views
        if checkpoint_hash_scheme != _CHECKPOINT_HASH_SCHEME:
            raise CheckpointCorruptError("unsupported checkpoint hash scheme")
        try:
            expected_chunk_hashes = [
                bytes.fromhex(record["sha256"]) for record in records
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptError("invalid checkpoint chunk metadata") from exc
        if not verify_hashes:
            return views
        chunk_hashes, checkpoint_hash = _checkpoint_hashes(list(views))
        if (
            chunk_hashes != expected_chunk_hashes
            or checkpoint_hash != expected_checkpoint_hash
        ):
            raise CheckpointCorruptError("checkpoint checksum mismatch")
        return views

    def _delete_objects(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return
        adapters = self._storage_manager.l2_adapters()
        if adapters:
            adapters[0][1].delete(keys)
        _deleted, skipped = self._storage_manager.delete_l1_keys(keys)
        if skipped:
            raise RuntimeError("checkpoint object remained locked in L1")

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

    def _manifest_chunk_keys(
        self,
        cache_salt: str,
        manifest: Mapping[str, Any],
    ) -> list[ObjectKey]:
        try:
            return [
                self._chunk_key(cache_salt, bytes.fromhex(record["sha256"]))
                for record in manifest["chunks"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptError("invalid checkpoint manifest") from exc

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
