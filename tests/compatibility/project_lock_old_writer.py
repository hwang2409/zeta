"""Drive an archived zeta release's public project writers for p06-4.

The candidate test runs this file inside the archived release's own environment,
so `zeta` here is the archived source. With `--gate`, the archived
`ProjectRegistry._acquire_lock` runs unchanged; only after its real exclusive
flock on `<projects>/.lock` returns does this process print `held` and wait for a
line on stdin before the public operation continues.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import zeta
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry


def _gate_root_lock(root_lock: Path) -> None:
    original = ProjectRegistry._acquire_lock
    gated = False

    def acquire(lock_fd: int, mode: int) -> None:
        nonlocal gated
        original(lock_fd, mode)
        info = os.fstat(lock_fd)
        expected = os.stat(root_lock)
        if (
            not gated
            and mode & fcntl.LOCK_EX
            and (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino)
        ):
            gated = True
            print("held", flush=True)
            sys.stdin.readline()

    ProjectRegistry._acquire_lock = staticmethod(acquire)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--operation", choices=("setup", "send"), required=True)
    parser.add_argument("--sender")
    parser.add_argument("--target")
    parser.add_argument("--message-id")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    registry = ProjectRegistry(args.home / "projects")
    result: dict[str, object] = {"zeta_file": zeta.__file__}
    try:
        if args.operation == "setup":
            result["sender"] = registry.create_project("p06-sender", "old").project_id
            result["target"] = registry.create_project("p06-target", "old").project_id
        else:
            if args.gate:
                _gate_root_lock(args.home / "projects" / ".lock")
            ProjectInbox(registry, sessions_root=args.home / "sessions").send(
                from_project=args.sender,
                from_session="a" * 32,
                to_project=args.target,
                kind="question",
                title="old-writer",
                body="old-writer-body",
                message_id=args.message_id,
            )
        result["outcome"] = "done"
    except Exception as exc:  # noqa: BLE001 - report the typed refusal to the harness
        result.update(outcome="refused", error_type=type(exc).__name__, error=str(exc))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
