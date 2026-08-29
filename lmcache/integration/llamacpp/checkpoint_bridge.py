# SPDX-License-Identifier: Apache-2.0
"""Fixed-path bridge between llama-server slot files and LMCache."""

# Standard
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import json
import os

# Local
from .checkpoint_store import CheckpointStore

_MAX_RESPONSE_BYTES = 1 << 20


class CheckpointBridgeError(RuntimeError):
    """Raised when llama-server or the fixed transfer path violates its contract."""


class CheckpointBridge:
    """Move slot-state files between llama-server and a ``CheckpointStore``.

    Args:
        checkpoint_store: Opaque LMCache checkpoint store.
        transfer_dir: Shared directory configured as llama-server's slot-save path.
        llama_url: Base URL for the trusted llama-server instance.
        slot_id: Serialized llama-server slot to save and restore.
        timeout: HTTP timeout in seconds.

    Raises:
        ValueError: If configuration is invalid.
    """

    def __init__(
        self,
        checkpoint_store: CheckpointStore,
        transfer_dir: Path,
        llama_url: str,
        slot_id: int,
        timeout: float,
    ) -> None:
        if not transfer_dir.is_dir():
            raise ValueError("transfer_dir must be an existing directory")
        if not llama_url.startswith(("http://", "https://")):
            raise ValueError("llama_url must use HTTP or HTTPS")
        if slot_id < 0 or timeout <= 0:
            raise ValueError("slot_id must be non-negative and timeout positive")
        self._checkpoint_store = checkpoint_store
        self._transfer_dir = transfer_dir.resolve()
        self._llama_url = llama_url.rstrip("/")
        self._slot_id = slot_id
        self._timeout = timeout

    def store(
        self,
        cache_salt: str,
        revision: int,
        compatibility: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Save the active slot and commit it to LMCache.

        Args:
            cache_salt: Lowercase session-isolation digest.
            revision: Immutable session revision to commit.
            compatibility: Required model and engine identity fields.

        Returns:
            Validated llama-server slot-save response.

        Raises:
            CheckpointBridgeError: If save output is absent or inconsistent.
            CheckpointStore errors: If LMCache rejects or cannot store the state.
        """
        filename = self._new_filename()
        transfer_path = self._transfer_dir / filename
        try:
            response = self._slot_action("save", filename)
            if not transfer_path.is_file() or transfer_path.is_symlink():
                raise CheckpointBridgeError("llama-server did not create a slot file")
            checkpoint = transfer_path.read_bytes()
            if response.get("n_written") != len(checkpoint):
                raise CheckpointBridgeError("slot-save byte count mismatch")
            self._checkpoint_store.store(
                cache_salt,
                revision,
                checkpoint,
                compatibility,
            )
            return response
        finally:
            transfer_path.unlink(missing_ok=True)

    def restore(
        self,
        cache_salt: str,
        revision: int,
        compatibility: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Restore one LMCache checkpoint into the active llama-server slot.

        Args:
            cache_salt: Lowercase session-isolation digest.
            revision: Session revision to restore.
            compatibility: Required model and engine identity fields.

        Returns:
            Validated llama-server response, or ``None`` for a cache miss.

        Raises:
            CheckpointBridgeError: If llama-server reads an unexpected byte count.
            CheckpointStore errors: If retrieval fails integrity validation.
        """
        checkpoint = self._checkpoint_store.load(
            cache_salt,
            revision,
            compatibility,
        )
        if checkpoint is None:
            return None

        transfer_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self._transfer_dir,
                prefix="llamacpp-",
                suffix=".bin",
                delete=False,
            ) as transfer_file:
                transfer_file.write(checkpoint)
                transfer_file.flush()
                os.fsync(transfer_file.fileno())
                transfer_path = Path(transfer_file.name)
            response = self._slot_action("restore", transfer_path.name)
            if response.get("n_read") != len(checkpoint):
                raise CheckpointBridgeError("slot-restore byte count mismatch")
            return response
        finally:
            if transfer_path is not None:
                transfer_path.unlink(missing_ok=True)

    def _slot_action(self, action: str, filename: str) -> dict[str, Any]:
        body = json.dumps({"filename": filename}).encode()
        request = Request(
            f"{self._llama_url}/slots/{self._slot_id}?action={action}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                response_body = response.read(_MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise CheckpointBridgeError(f"llama-server slot {action} failed") from exc
        if len(response_body) > _MAX_RESPONSE_BYTES:
            raise CheckpointBridgeError("llama-server response is too large")
        try:
            result = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CheckpointBridgeError("llama-server returned invalid JSON") from exc
        if not isinstance(result, dict) or result.get("filename") != filename:
            raise CheckpointBridgeError("llama-server returned an invalid filename")
        return result

    @staticmethod
    def _new_filename() -> str:
        return f"llamacpp-{uuid4().hex}.bin"
