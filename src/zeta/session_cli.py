"""``zeta session`` subcommands: list, delete, export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import IO

from .core.session import (
    SessionError,
    SessionManager,
    env_home,
    format_relative_age,
)


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    """Register ``zeta session <verb>`` on the shared subparser action."""

    parser = commands.add_parser("session", help="manage stored zeta sessions")
    verbs = parser.add_subparsers(dest="session_verb", required=True)
    verbs.add_parser("list", help="list stored sessions (id, name, age, preview)")
    delete = verbs.add_parser("delete", help="delete a stored session by id")
    delete.add_argument("session_id", help="session id to delete")
    delete.add_argument(
        "--force",
        action="store_true",
        help="skip the confirmation prompt",
    )
    export = verbs.add_parser("export", help="export a session as portable JSONL")
    export.add_argument("session_id", help="session id to export")
    export.add_argument(
        "--out",
        default=None,
        help="write to a file instead of stdout",
    )


def run(
    args: argparse.Namespace,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    input_reader=None,
) -> int:
    """Dispatch to the verb handler and return its exit code."""

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    reader = input_reader if input_reader is not None else input
    manager = SessionManager(env_home())
    verb = args.session_verb
    if verb == "list":
        return _run_list(manager, out)
    if verb == "delete":
        return _run_delete(manager, args, out, err, reader)
    if verb == "export":
        return _run_export(manager, args, out, err)
    print(f"zeta: unknown session verb: {verb}", file=err)
    return 2


def _run_list(manager: SessionManager, out: IO[str]) -> int:
    previews = manager.list_session_previews(limit=1000)
    if not previews:
        print("no stored zeta sessions", file=out)
        return 0
    header = f"{'ID':10}  {'AGE':>10}  {'NAME':20}  PREVIEW"
    print(header, file=out)
    for preview in previews:
        age = format_relative_age(preview.updated_at)
        name = (preview.name or "-")[:20]
        print(
            f"{preview.session_id[:8]:10}  {age:>10}  {name:20}  {preview.preview}",
            file=out,
        )
    return 0


def _run_delete(
    manager: SessionManager,
    args: argparse.Namespace,
    out: IO[str],
    err: IO[str],
    input_reader,
) -> int:
    session_id = args.session_id
    try:
        metadata = manager.open(session_id).metadata
    except SessionError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    if not args.force:
        label = f" [{metadata.name}]" if metadata.name else ""
        prompt = (
            f"delete session {session_id[:8]}{label}? [y/N] "
        )
        try:
            answer = input_reader(prompt)
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted", file=out)
            return 1
    try:
        manager.delete(session_id)
    except SessionError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(f"deleted {session_id}", file=out)
    return 0


def _run_export(
    manager: SessionManager,
    args: argparse.Namespace,
    out: IO[str],
    err: IO[str],
) -> int:
    try:
        payload = manager.export(args.session_id)
    except SessionError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    if args.out is None:
        out.write(payload)
        out.flush()
        return 0
    Path(args.out).write_text(payload, encoding="utf-8")
    print(f"wrote {args.out}", file=out)
    return 0


__all__ = ["add_subcommand", "run"]
