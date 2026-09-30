"""CLI for the identity-only project registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import IO

from ..core.session import env_home
from ..project_registry import ProjectRegistry, ProjectRegistryError

_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("project", help="manage projects and bounded memory")
    verbs = parser.add_subparsers(dest="project_verb", required=True)
    create = verbs.add_parser("create", help="create a project")
    create.add_argument("name")
    create.add_argument("--scope", required=True)
    create.add_argument("--canonical-integration-root")
    init = verbs.add_parser("init", help="create or discover a project for a directory")
    init.add_argument("directory", nargs="?", default=".")
    init.add_argument("--name")
    init.add_argument("--scope", default="")
    verbs.add_parser("list", help="list projects")
    discover = verbs.add_parser(
        "discover", help="find the project associated with a directory"
    )
    discover.add_argument("directory", nargs="?", default=".")
    memory = verbs.add_parser("memory", help="print or update bounded project memory")
    memory.add_argument("project")
    memory.add_argument(
        "--set", nargs=2, metavar=("FILE", "CONTENT"), action="append", default=[]
    )
    memory.add_argument(
        "--from-file", nargs=2, metavar=("FILE", "PATH"), action="append", default=[]
    )
    show = verbs.add_parser("show", help="show a project")
    show.add_argument("project", help="project ID or exact name")


def run(
    args: argparse.Namespace,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    registry = ProjectRegistry(env_home() / "projects")
    try:
        if args.project_verb == "create":
            project = registry.create_project(
                args.name, args.scope, args.canonical_integration_root
            )
            value = project.to_dict()
        elif args.project_verb == "init":
            directory = Path(args.directory).expanduser().resolve()
            project = registry.find_for_directory(directory)
            if project is None:
                project = registry.create_project(
                    args.name or directory.name,
                    args.scope or directory.name,
                    str(directory),
                )
            value = project.to_dict()
        elif args.project_verb == "discover":
            project = registry.find_for_directory(args.directory)
            if project is None:
                raise ProjectRegistryError("no project associated with directory")
            value = project.to_dict()
        elif args.project_verb == "memory":
            project_id = (
                args.project
                if _PROJECT_ID.fullmatch(args.project)
                else registry.show_project(name=args.project).project_id
            )
            supplied = list(args.set) + list(args.from_file)
            if not supplied:
                value = {
                    name: content for name, content in registry.load_memory(project_id)
                }
            else:
                if len(supplied) != 1:
                    raise ProjectRegistryError(
                        "memory accepts zero flags or exactly one --set/--from-file"
                    )
                name, value_or_path = supplied[0]
                content = (
                    value_or_path
                    if args.set
                    else Path(value_or_path).expanduser().read_text(encoding="utf-8")
                )
                registry.update_memory(project_id, {name: content})
                value = {
                    name: content for name, content in registry.load_memory(project_id)
                }
        elif args.project_verb == "list":
            value = [
                project.to_dict()
                for project in registry.list_projects()
            ]
        elif args.project_verb == "show":
            value = registry.show_project(
                args.project if _PROJECT_ID.fullmatch(args.project) else None,
                name=args.project if not _PROJECT_ID.fullmatch(args.project) else None,
            ).to_dict()
        else:
            print(f"zeta: unknown project verb: {args.project_verb}", file=err)
            return 2
    except (ProjectRegistryError, OSError, UnicodeError) as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True), file=out)
    return 0


__all__ = ["add_subcommand", "run"]
