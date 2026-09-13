# SPDX-License-Identifier: Apache-2.0
"""Single-compute session switching through llama.cpp slot checkpoints."""

# Standard
from contextlib import closing
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
import os
import sqlite3
import stat

_MAX_RESPONSE_BYTES = 32 << 20


def _cache_salt(principal: str, session_id: str) -> str:
    identity = f"{len(principal)}:{principal}{len(session_id)}:{session_id}"
    return sha256(identity.encode()).hexdigest()


class SessionBroker:
    """Serialize session-aware requests through one llama.cpp slot.

    Args:
        database_path: SQLite file holding committed session revisions.
        llama_url: Trusted llama-server base URL.
        compatibility: Model and engine checkpoint identity.
        timeout: Upstream request timeout in seconds.
        max_queue_depth: Maximum number of requests waiting for compute.
        save_policy: Save after every completion or only before switching sessions.

    Raises:
        ValueError: If configuration is invalid.
    """

    def __init__(
        self,
        database_path: Path,
        llama_url: str,
        compatibility: dict[str, Any],
        timeout: float,
        max_queue_depth: int = 4,
        save_policy: Literal["always", "switch"] = "always",
    ) -> None:
        """Configure durable revision heads and the serialized compute lock.

        Args:
            database_path: SQLite file holding committed session revisions.
            llama_url: Trusted llama-server base URL.
            compatibility: Model and engine checkpoint identity.
            timeout: Upstream request timeout in seconds.
            max_queue_depth: Maximum number of requests waiting for compute.
            save_policy: Save after every completion or only before switching sessions.

        Raises:
            ValueError: If configuration or database path safety is invalid.
        """
        if (
            not llama_url.startswith(("http://", "https://"))
            or not compatibility
            or timeout <= 0
            or not isinstance(max_queue_depth, int)
            or isinstance(max_queue_depth, bool)
            or max_queue_depth < 0
            or save_policy not in ("always", "switch")
        ):
            raise ValueError("invalid session broker configuration")
        database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_status = database_path.parent.stat()
        if (
            database_path.parent.is_symlink()
            or not stat.S_ISDIR(parent_status.st_mode)
            or parent_status.st_uid != os.geteuid()
            or stat.S_IMODE(parent_status.st_mode) & 0o077
            or database_path.is_symlink()
        ):
            raise ValueError("session database directory must be private and owned")
        if database_path.exists():
            database_status = database_path.stat()
            if (
                not stat.S_ISREG(database_status.st_mode)
                or database_status.st_uid != os.geteuid()
            ):
                raise ValueError("session database must be an owned regular file")
        self._database_path = database_path
        self._llama_url = llama_url.rstrip("/")
        self._compatibility = deepcopy(compatibility)
        self._timeout = timeout
        self._save_policy = save_policy
        self._admission = BoundedSemaphore(max_queue_depth + 1)
        self._compute = Lock()
        self._active: tuple[str, int] | None = None
        with closing(self._connect()) as database, database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS session_heads ("
                "cache_salt TEXT PRIMARY KEY, revision INTEGER NOT NULL)"
            )
        os.chmod(database_path, 0o600)

    def complete(
        self,
        principal: str,
        session_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> tuple[bytes, dict[str, str]]:
        """Run one completion and commit its next checkpoint revision.

        Args:
            principal: Authenticated caller identity.
            session_id: Caller-scoped session identifier.
            expected_revision: Last revision observed by the caller.
            payload: Canonical full-history chat completion request.

        Returns:
            Upstream response bytes and session metadata headers.

        Raises:
            ValueError: If the request or expected revision is invalid.
            OSError: If llama-server fails the completion or checkpoint action.
        """
        self._validate_request(principal, session_id, expected_revision, payload)
        if not self._admission.acquire(blocking=False):
            raise OverflowError("session queue is full")
        queued_at = monotonic()
        try:
            with self._compute:
                queue_wait_ms = (monotonic() - queued_at) * 1000
                return self._complete_locked(
                    principal,
                    session_id,
                    expected_revision,
                    payload,
                    queue_wait_ms,
                )
        finally:
            self._admission.release()

    def _complete_locked(
        self,
        principal: str,
        session_id: str,
        expected_revision: int,
        payload: dict[str, Any],
        queue_wait_ms: float,
    ) -> tuple[bytes, dict[str, str]]:
        cache_salt = _cache_salt(principal, session_id)
        with closing(self._connect()) as database, database:
            row = database.execute(
                "SELECT revision FROM session_heads WHERE cache_salt = ?",
                (cache_salt,),
            ).fetchone()
        current_revision = 0 if row is None else row[0]
        if current_revision != expected_revision:
            raise ValueError("revision conflict")

        self._save_active_before_switch(cache_salt)

        switch_started = monotonic()
        cache_status = "cold"
        if expected_revision > 0:
            if self._active == (cache_salt, expected_revision):
                cache_status = "hit"
            else:
                self._active = None
                try:
                    self._slot_action("restore", cache_salt, expected_revision)
                    cache_status = "hit"
                except HTTPError:
                    cache_status = "miss"
                    self._active = None
        switch_ms = (monotonic() - switch_started) * 1000

        # A failed completion may still have changed the sole upstream slot.
        self._active = None
        upstream_payload = dict(payload)
        upstream_payload["cache_prompt"] = cache_status == "hit"
        body = self._post_json("/v1/chat/completions", upstream_payload)

        next_revision = expected_revision + 1
        if self._save_policy == "always":
            try:
                self._slot_action("save", cache_salt, next_revision)
            except OSError:
                self._active = None
                raise
        with closing(self._connect()) as database, database:
            database.execute(
                "INSERT INTO session_heads(cache_salt, revision) VALUES(?, ?) "
                "ON CONFLICT(cache_salt) DO UPDATE SET revision=excluded.revision",
                (cache_salt, next_revision),
            )
        self._active = (cache_salt, next_revision)
        return body, {
            "X-LMCache-Cache": cache_status,
            "X-LMCache-Queue-Wait-Ms": f"{queue_wait_ms:.3f}",
            "X-LMCache-Switch-Ms": f"{switch_ms:.3f}",
            "X-LMCache-Revision": str(next_revision),
        }

    def _save_active_before_switch(self, next_cache_salt: str) -> None:
        # ponytail: switch mode trades crash durability for fewer saves; canonical
        # full history rebuilds a missing checkpoint after restart.
        if (
            self._save_policy != "switch"
            or self._active is None
            or self._active[0] == next_cache_salt
        ):
            return
        cache_salt, revision = self._active
        try:
            self._slot_action("save", cache_salt, revision)
        except OSError:
            self._active = None
            raise

    def _slot_action(self, action: str, cache_salt: str, revision: int) -> None:
        self._post_json(
            f"/slots/0?action={action}",
            {
                "cache_salt": cache_salt,
                "compatibility": self._compatibility,
                "revision": revision,
            },
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> bytes:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        request = Request(
            self._llama_url + path,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise OSError("llama-server response is too large")
        return body

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    @staticmethod
    def _validate_request(
        principal: str,
        session_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> None:
        if (
            not isinstance(principal, str)
            or not 0 < len(principal.encode()) <= 256
            or not isinstance(session_id, str)
            or not 0 < len(session_id.encode()) <= 256
            or not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or not 0 <= expected_revision < 2**63 - 1
            or not isinstance(payload, dict)
            or not isinstance(payload.get("messages"), list)
            or payload.get("stream") is True
        ):
            raise ValueError("invalid session completion request")
