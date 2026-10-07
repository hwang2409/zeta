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
from .user_action import confirm_memory_accept

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
    memory = verbs.add_parser(
        "memory", help="print, update, push, or pull bounded project memory"
    )
    memory.add_argument("project", help="project, or push/pull/resolve for remote sync")
    memory.add_argument("remote", nargs="?", help="remote alias or explicit SSH host")
    memory.add_argument("action", nargs="?", help="memory action file")
    memory.add_argument("--project", dest="sync_project", help="project ID or exact name")
    memory.add_argument("--remote-home", help="remote ZETA_HOME (default: ~/.zeta)")
    memory.add_argument(
        "--accept",
        choices=("local", "remote"),
        help="side to accept when resolving memory conflicts",
    )
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
            if args.project == "accept":
                if args.remote is None or args.action is not None:
                    raise ProjectRegistryError("memory accept requires exactly one file")
                project = registry.find_for_directory(Path.cwd())
                if project is None:
                    raise ProjectRegistryError("no project associated with the current directory")
                project_id = project.project_id
                confirm_memory_accept(
                    registry, project_id, args.remote, stdin=sys.stdin, stdout=out
                )
                registry.accept_memory(project_id, args.remote)
                value = {name: content for name, content in registry.load_memory(project_id)}
                print(json.dumps(value, indent=2, sort_keys=True), file=out)
                return 0
            if args.remote == "accept":
                if args.action is None:
                    raise ProjectRegistryError("memory accept requires exactly one file")
                project_id = (
                    args.project
                    if _PROJECT_ID.fullmatch(args.project)
                    else registry.show_project(name=args.project).project_id
                )
                confirm_memory_accept(
                    registry, project_id, args.action, stdin=sys.stdin, stdout=out
                )
                registry.accept_memory(project_id, args.action)
                value = {name: content for name, content in registry.load_memory(project_id)}
                print(json.dumps(value, indent=2, sort_keys=True), file=out)
                return 0
            if args.project in {"push", "pull", "resolve"}:
                return _run_memory_sync(args, registry, out, err)
            if (
                args.remote is not None
                or args.sync_project is not None
                or args.remote_home is not None
                or args.accept is not None
                or args.action is not None
            ):
                raise ProjectRegistryError(
                    "remote arguments require: project memory push|pull|resolve HOST"
                )
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


def _run_memory_sync(
    args: argparse.Namespace,
    registry: ProjectRegistry,
    out: IO[str],
    err: IO[str],
) -> int:
    from ..remote_sync import (
        RemoteSyncError,
        pull_project_memory,
        push_project_memory,
        resolve_project_memory,
        resolve_transport,
    )

    if args.remote is None:
        print("zeta: project memory push|pull|resolve requires HOST", file=err)
        return 2
    if args.set or args.from_file:
        print("zeta: memory sync does not accept --set or --from-file", file=err)
        return 2
    if args.project != "resolve" and args.accept is not None:
        print("zeta: --accept is only valid for project memory resolve", file=err)
        return 2
    try:
        if args.sync_project:
            project = (
                registry.show_project(args.sync_project)
                if _PROJECT_ID.fullmatch(args.sync_project)
                else registry.show_project(name=args.sync_project)
            )
        else:
            project = registry.find_for_directory(Path.cwd())
            if project is None:
                raise ProjectRegistryError(
                    "no project associated with the current directory; use --project"
                )
        transport = resolve_transport(
            env_home(), args.remote, remote_home=args.remote_home
        )
        if args.project == "resolve":
            if args.accept is None:
                print("zeta: project memory resolve requires --accept local|remote", file=err)
                return 2
            result = resolve_project_memory(
                env_home(),
                transport,
                project_id=project.project_id,
                accept=args.accept,
            )
        else:
            function = (
                push_project_memory if args.project == "push" else pull_project_memory
            )
            result = function(
                env_home(), transport, project_id=project.project_id
            )
    except (ProjectRegistryError, RemoteSyncError, OSError, ValueError) as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(json.dumps({
        "project_id": result.project_id,
        "updated": result.updated,
        "conflicts": result.conflicts,
    }, indent=2, sort_keys=True), file=out)
    return 1 if result.conflicts else 0


__all__ = ["add_subcommand", "run"]
