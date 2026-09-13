# SPDX-License-Identifier: Apache-2.0
"""Local client for the llama.cpp checkpoint service."""

# Standard
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json
import socket


class CheckpointClientError(RuntimeError):
    """Raised when the local checkpoint service violates its protocol."""


def request(
    socket_path: Path,
    payload: dict[str, Any],
    timeout: float = 30.0,
    max_response_bytes: int = 1 << 20,
) -> dict[str, Any]:
    """Send one bounded request to the local checkpoint service."""
    response, _body = _exchange(
        socket_path,
        payload,
        timeout=timeout,
        max_response_bytes=max_response_bytes,
    )
    return response


def store_checkpoint(
    socket_path: Path,
    checkpoint: bytes,
    cache_salt: str,
    revision: int,
    compatibility: dict[str, Any],
    timeout: float = 30.0,
) -> None:
    """Store opaque checkpoint bytes through the local service.

    Args:
        socket_path: Private checkpoint service socket.
        checkpoint: Opaque serialized llama.cpp state.
        cache_salt: Session-isolation digest.
        revision: Immutable session revision.
        compatibility: Model and engine identity fields.
        timeout: Socket timeout in seconds.

    Raises:
        CheckpointClientError: If the service rejects the checkpoint.
        ValueError: If ``timeout`` is not positive.
    """
    response, _body = _exchange(
        socket_path,
        {
            "action": "store",
            "cache_salt": cache_salt,
            "compatibility": compatibility,
            "length": len(checkpoint),
            "revision": revision,
        },
        body=checkpoint,
        timeout=timeout,
    )
    if response.get("ok") is not True:
        error = response.get("error", "checkpoint store failed")
        raise CheckpointClientError(str(error))


def restore_checkpoint(
    socket_path: Path,
    cache_salt: str,
    revision: int,
    compatibility: dict[str, Any],
    timeout: float = 30.0,
    max_checkpoint_bytes: int = 64 << 30,
) -> bytes | None:
    """Restore opaque checkpoint bytes from the local service.

    Args:
        socket_path: Private checkpoint service socket.
        cache_salt: Session-isolation digest.
        revision: Immutable session revision.
        compatibility: Model and engine identity fields.
        timeout: Socket timeout in seconds.
        max_checkpoint_bytes: Largest accepted checkpoint response.

    Returns:
        Exact checkpoint bytes, or ``None`` for a cache miss.

    Raises:
        CheckpointClientError: If the service rejects or truncates the response.
        ValueError: If a limit is invalid.
    """
    response, body = _exchange(
        socket_path,
        {
            "action": "restore",
            "cache_salt": cache_salt,
            "compatibility": compatibility,
            "revision": revision,
        },
        timeout=timeout,
        max_body_bytes=max_checkpoint_bytes,
    )
    if response.get("ok") is not True:
        error = response.get("error", "checkpoint restore failed")
        raise CheckpointClientError(str(error))
    if response.get("found") is False:
        return None
    if response.get("found") is not True:
        raise CheckpointClientError("checkpoint response is invalid")
    return body


def _exchange(
    socket_path: Path,
    payload: dict[str, Any],
    timeout: float,
    max_response_bytes: int = 1 << 20,
    body: bytes = b"",
    max_body_bytes: int = 64 << 30,
) -> tuple[dict[str, Any], bytes]:
    if timeout <= 0 or max_response_bytes < 2:
        raise ValueError("timeout and max_response_bytes must be positive")
    if max_body_bytes < 0:
        raise ValueError("max_body_bytes must be non-negative")
    frame = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall(frame + body)
        with client.makefile("rb") as stream:
            response = stream.readline(max_response_bytes + 1)
            if len(response) > max_response_bytes:
                raise CheckpointClientError("checkpoint response is too large")
            if not response.endswith(b"\n"):
                raise CheckpointClientError("checkpoint response is incomplete")
            try:
                result = json.loads(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise CheckpointClientError("checkpoint response is invalid") from exc
            if not isinstance(result, dict):
                raise CheckpointClientError("checkpoint response is invalid")
            length = result.get("length", 0)
            if (
                not isinstance(length, int)
                or isinstance(length, bool)
                or length < 0
                or length > max_body_bytes
            ):
                raise CheckpointClientError("checkpoint response length is invalid")
            response_body = stream.read(length)
    if len(response_body) != length:
        raise CheckpointClientError("checkpoint response body is incomplete")
    return result, response_body


def main(argv: list[str] | None = None) -> int:
    """Store or restore a checkpoint file, or print service statistics.

    Args:
        argv: Optional command-line arguments for tests and embedding.

    Returns:
        Zero for a successful service response, otherwise one.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("store", "restore"):
        command = actions.add_parser(action)
        command.add_argument("--cache-salt", required=True)
        command.add_argument("--revision", type=int, required=True)
        command.add_argument("--compatibility", required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
    actions.add_parser("stats")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {"action": args.action}
    if args.action != "stats":
        try:
            compatibility = json.loads(args.compatibility)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid compatibility JSON: {exc.msg}")
        if not isinstance(compatibility, dict):
            parser.error("compatibility must be a JSON object")
        payload.update(
            cache_salt=args.cache_salt,
            revision=args.revision,
            compatibility=compatibility,
        )
    body = b""
    if args.action == "store":
        try:
            body = args.checkpoint.read_bytes()
        except OSError as exc:
            parser.error(f"cannot read checkpoint: {exc}")
        payload["length"] = len(body)
    response, response_body = _exchange(
        args.socket,
        payload,
        body=body,
        timeout=args.timeout,
    )
    if (
        args.action == "restore"
        and response.get("ok") is True
        and response.get("found") is True
    ):
        try:
            args.checkpoint.write_bytes(response_body)
        except OSError as exc:
            parser.error(f"cannot write checkpoint: {exc}")
    print(json.dumps(response, separators=(",", ":"), sort_keys=True))
    return 0 if response.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
