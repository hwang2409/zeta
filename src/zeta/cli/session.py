"""``zeta session`` subcommands: list, rename, delete, export, stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO

from ..core.session import (
    SessionError,
    SessionManager,
    env_home,
    format_relative_age,
)
from .compaction_stats import (
    DEFAULT_SINCE,
    ReportError,
    compaction_report,
    render_report,
)


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    """Register ``zeta session <verb>`` on the shared subparser action."""

    parser = commands.add_parser("session", help="manage stored zeta sessions")
    verbs = parser.add_subparsers(dest="session_verb", required=True)
    verbs.add_parser("list", help="list stored sessions (id, name, age, preview)")
    rename = verbs.add_parser("rename", help="set a session name (empty clears it)")
    rename.add_argument("session_id", help="session id or unique prefix")
    rename.add_argument("name", help="display name; pass an empty string to clear")
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
    stats = verbs.add_parser(
        "stats",
        help="read-only report over stored session logs",
        description=(
            "Scan stored session logs, including child agents, without writing "
            "or locking anything."
        ),
    )
    stats.add_argument(
        "--compaction",
        action="store_true",
        required=True,
        help="report compaction, eviction, recall_history, and budget activity",
    )
    stats.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="sessions updated within 7d/12h/30m/2w, since an ISO date, or all "
        f"(default: {DEFAULT_SINCE}; ignored with --session)",
    )
    stats.add_argument("--session", default=None, help="one session id or unique prefix")
    stats.add_argument("--top", type=int, default=10, help="top sessions to list (default: 10)")
    stats.add_argument("--json", action="store_true", help="print the report as JSON")
    push = verbs.add_parser("push", help="upload a session through SSH")
    push.add_argument("host", help="configured remote alias or explicit SSH host")
    push.add_argument("session_id", nargs="?", help="session id (default: most recent)")
    push.add_argument("--force", action="store_true", help="replace divergent remote state")
    push.add_argument("--remote-home", help="remote ZETA_HOME (default: ~/.zeta)")
    pull = verbs.add_parser("pull", help="download a session through SSH")
    pull.add_argument("host", help="configured remote alias or explicit SSH host")
    pull.add_argument("session_id", help="session id")
    pull.add_argument("--cwd", help="existing or new cwd for the imported session")
    pull.add_argument("--force", action="store_true", help="replace divergent local state")
    pull.add_argument("--remote-home", help="remote ZETA_HOME (default: ~/.zeta)")


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
    verb = args.session_verb
    if verb == "stats":
        return _run_stats(args, out, err)
    if verb in {"push", "pull"}:
        return _run_transfer(args, out, err)
    manager = SessionManager(env_home())
    if verb == "list":
        return _run_list(manager, out)
    if verb == "rename":
        try:
            metadata = manager.rename(args.session_id, args.name)
        except SessionError as exc:
            print(f"zeta: {exc}", file=err)
            return 1
        print(f"renamed {metadata.session_id}", file=out)
        return 0
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
        # Server preview is now the literal first user message with no
        # placeholder text — mirror the TUI/frontend client empty-state label here so
        # a session that has not sent its first turn still renders as
        # something readable, not a blank column (ZETA-134 review r2).
        text = preview.preview or "(no user message)"
        print(
            f"{preview.session_id[:8]:10}  {age:>10}  {name:20}  {text}",
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
    try:
        full_id = manager.resolve_id(args.session_id)
    except SessionError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    label = ""
    try:
        metadata = manager._read(full_id)
    except SessionError:
        metadata = None
    if metadata is not None and metadata.name:
        label = f" [{metadata.name}]"
    if not args.force:
        prompt = f"delete session {full_id[:8]}{label}? [y/N] "
        try:
            answer = input_reader(prompt)
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("aborted", file=out)
            return 1
    try:
        manager.delete(full_id)
    except SessionError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(f"deleted {full_id}", file=out)
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


def _run_transfer(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from ..remote_sync import (
        RemoteSyncError,
        pull_session,
        push_session,
        resolve_transport,
    )

    home = env_home()
    try:
        transport = resolve_transport(
            home, args.host, remote_home=args.remote_home
        )
        result = (
            push_session(
                home,
                transport,
                session_id=args.session_id,
                force=args.force,
            )
            if args.session_verb == "push"
            else pull_session(
                home,
                transport,
                session_id=args.session_id,
                cwd=args.cwd,
                force=args.force,
            )
        )
    except (RemoteSyncError, OSError, ValueError) as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(json.dumps({
        "session_id": result.session_id,
        "last_seq": result.last_seq,
        "digest": result.digest,
        "resume_notice": result.resume_notice,
    }, indent=2, sort_keys=True), file=out)
    return 0


def _run_stats(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    try:
        report = compaction_report(
            env_home(), since=args.since, session=args.session, top=args.top
        )
    except ReportError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    if args.json:
        json.dump(report, out, indent=2)
        out.write("\n")
    else:
        out.write(render_report(report))
    out.flush()
    return 0


__all__ = ["add_subcommand", "run"]
