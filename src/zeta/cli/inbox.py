"""CLI inspection for project inboxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO

from ..core.session import env_home
from ..project_inbox import InboxError, ProjectInbox
from ..project_registry import ProjectRegistry, ProjectRegistryError


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("inbox", help="list a project's inbox")
    parser.add_argument("--project", help="project name or ID (default: current directory)")


def run(
    args: argparse.Namespace,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    home = env_home()
    registry = ProjectRegistry(home / "projects")
    try:
        if args.project:
            try:
                project = registry.show_project(args.project)
            except ProjectRegistryError:
                project = registry.show_project(name=args.project)
        else:
            project = registry.find_for_directory(Path.cwd())
            if project is None:
                raise ProjectRegistryError(
                    "current directory is not associated with a project; use --project"
                )
        state = ProjectInbox(
            registry, sessions_root=home / "sessions"
        ).list(project.project_id)
    except (InboxError, ProjectRegistryError, OSError, UnicodeError) as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(
        json.dumps({"project": project.to_dict(), **state}, indent=2, sort_keys=True),
        file=out,
    )
    return 0


__all__ = ["add_subcommand", "run"]
