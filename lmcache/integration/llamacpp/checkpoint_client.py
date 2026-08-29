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
    if timeout <= 0 or max_response_bytes < 2:
        raise ValueError("timeout and max_response_bytes must be positive")
    frame = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(socket_path))
        client.sendall(frame)
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
    return result


def main(argv: list[str] | None = None) -> int:
    """Send a store, restore, or stats request and print its JSON response."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("store", "restore"):
        command = actions.add_parser(action)
        command.add_argument("--cache-salt", required=True)
        command.add_argument("--revision", type=int, required=True)
        command.add_argument("--compatibility", required=True)
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
    response = request(args.socket, payload, timeout=args.timeout)
    print(json.dumps(response, separators=(",", ":"), sort_keys=True))
    return 0 if response.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
