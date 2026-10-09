"""CLI for the identity-only project registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import IO, Any

from ..core.session import env_home
from ..memory.user_authorization import (
    MemoryMutationAuthorization,
    memory_accept_preview,
)
from ..project_registry import ProjectRegistry, ProjectRegistryError

_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("project", help="manage projects and bounded memory")
    verbs = parser.add_subparsers(dest="project_verb", required=True)
    create = verbs.add_parser("create", help="create a project")
    create.add_argument("name")
    create.add_argument("--scope", required=True)
    create.add_argument("--canonical-integration-root")
    create.add_argument("--memory-profile", action="append", choices=("zeta", "messaging"))
    init = verbs.add_parser("init", help="create or discover a project for a directory")
    init.add_argument("directory", nargs="?", default=".")
    init.add_argument("--name")
    init.add_argument("--scope", default="")
    init.add_argument("--memory-profile", action="append", choices=("zeta", "messaging"))
    verbs.add_parser("list", help="list projects")
    discover = verbs.add_parser(
        "discover", help="find the project associated with a directory"
    )
    discover.add_argument("directory", nargs="?", default=".")
    index = verbs.add_parser("index", help="inspect or rebuild transcript search")
    index.add_argument("project", help="project ID or exact name")
    index_actions = index.add_subparsers(dest="index_action", required=True)
    index_actions.add_parser("status", help="show index generation and cursors")
    index_actions.add_parser("rebuild", help="rebuild from linked parent sessions")
    search = index_actions.add_parser("search", help="search sanitized transcript units")
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=10)
    memory = verbs.add_parser(
        "memory", help="print, update, push, or pull bounded project memory"
    )
    memory.add_argument("project", help="project, or push/pull/resolve for remote sync")
    memory.add_argument("remote", nargs="?", help="remote alias or explicit SSH host")
    memory.add_argument("action", nargs="?", help="memory action file")
    memory.add_argument("detail", nargs="?", help="memory action value")
    memory.add_argument("--project", dest="sync_project", help="project ID or exact name")
    memory.add_argument("--remote-home", help="remote ZETA_HOME (default: ~/.zeta)")
    memory.add_argument(
        "--json",
        action="store_true",
        help="print structured memory state",
    )
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
    memory.add_argument("--map-kind", action="append", default=[], metavar="OLD=NEW")
    memory.add_argument("--resolve-kind", action="append", default=[], metavar="KIND")
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
            profiles = getattr(args, "memory_profile", None) or ["zeta"]
            if len(profiles) != 1:
                raise ProjectRegistryError("--memory-profile may be specified only once")
            project = registry.create_project(
                args.name,
                args.scope,
                args.canonical_integration_root,
                memory_profile=profiles[0],
            )
            value = project.to_dict()
        elif args.project_verb == "init":
            profiles = getattr(args, "memory_profile", None) or ["zeta"]
            if len(profiles) != 1:
                raise ProjectRegistryError("--memory-profile may be specified only once")
            directory = Path(args.directory).expanduser().resolve()
            project = registry.find_for_directory(directory)
            if project is None:
                project = registry.create_project(
                    args.name or directory.name,
                    args.scope or directory.name,
                    str(directory),
                    memory_profile=profiles[0],
                )
            registry.ensure_memory_supported(project.project_id)
            value = project.to_dict()
        elif args.project_verb == "discover":
            project = registry.find_for_directory(args.directory)
            if project is None:
                raise ProjectRegistryError("no project associated with directory")
            value = project.to_dict()
        elif args.project_verb == "index":
            value = _run_index(args, registry)
        elif args.project_verb == "memory":
            if args.project == "accept":
                if args.remote is None or args.action is not None:
                    raise ProjectRegistryError("memory accept requires exactly one file")
                project = registry.find_for_directory(Path.cwd())
                if project is None:
                    raise ProjectRegistryError("no project associated with the current directory")
                project_id = project.project_id
                noun, preview = memory_accept_preview(registry, project_id, args.remote)
                MemoryMutationAuthorization.terminal(
                    stdin=sys.stdin, stdout=out
                ).authorize(
                    action="accept", target=args.remote, preview=preview, noun=noun
                )
                if registry.memory_format(project_id) == 2:
                    registry._accept_memory_entry(project_id, args.remote)
                    value = registry._entry_memory_view(project_id)
                else:
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
                noun, preview = memory_accept_preview(registry, project_id, args.action)
                MemoryMutationAuthorization.terminal(
                    stdin=sys.stdin, stdout=out
                ).authorize(
                    action="accept", target=args.action, preview=preview, noun=noun
                )
                if registry.memory_format(project_id) == 2:
                    registry._accept_memory_entry(project_id, args.action)
                    value = registry._entry_memory_view(project_id)
                else:
                    registry.accept_memory(project_id, args.action)
                    value = {name: content for name, content in registry.load_memory(project_id)}
                print(json.dumps(value, indent=2, sort_keys=True), file=out)
                return 0
            if args.project in {"push", "pull", "resolve"}:
                return _run_memory_sync(args, registry, out, err)
            project_id = (
                args.project
                if _PROJECT_ID.fullmatch(args.project)
                else registry.show_project(name=args.project).project_id
            )
            if args.remote in {"migrate", "rollback", "finalize"}:
                if args.action is not None or getattr(args, "detail", None) is not None:
                    raise ProjectRegistryError(f"memory {args.remote} takes no arguments")
                print(
                    "warning: memory history is bounded to 128 versions; transcripts "
                    "and external backups can retain older content",
                    file=err,
                )
                if args.remote == "migrate":
                    registry.migrate_memory(project_id)
                elif args.remote == "rollback":
                    registry.rollback_memory_migration(project_id)
                else:
                    print(
                        "warning: finalize removes the protected format-1 rollback target",
                        file=err,
                    )
                    registry.finalize_memory_migration(project_id)
                value = (
                    registry._entry_memory_view(project_id)
                    if registry.memory_format(project_id) == 2
                    else {name: content for name, content in registry.load_memory(project_id)}
                )
                print(json.dumps(value, indent=2, sort_keys=True), file=out)
                return 0
            if args.remote == "schema":
                if args.action != "set-profile" or not getattr(args, "detail", None):
                    raise ProjectRegistryError(
                        "memory schema requires: set-profile PROFILE"
                    )
                mappings: dict[str, str] = {}
                for item in getattr(args, "map_kind", []):
                    old, separator, new = item.partition("=")
                    if not separator or not old or not new or old in mappings:
                        raise ProjectRegistryError("--map-kind requires unique OLD=NEW values")
                    mappings[old] = new
                registry.set_memory_profile(
                    project_id,
                    args.detail,
                    kind_mappings=mappings,
                    resolve_kinds=frozenset(getattr(args, "resolve_kind", [])),
                )
                value = registry._entry_memory_view(project_id)
                print(json.dumps(value, indent=2, sort_keys=True), file=out)
                return 0
            if (
                args.remote is not None
                or args.sync_project is not None
                or args.remote_home is not None
                or args.accept is not None
                or args.action is not None
                or getattr(args, "detail", None) is not None
                or getattr(args, "map_kind", [])
                or getattr(args, "resolve_kind", [])
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
            memory_format = registry.memory_format(project_id)
            if not supplied:
                if memory_format == 2:
                    structured = registry._entry_memory_view(project_id)
                    value = (
                        structured
                        if getattr(args, "json", False)
                        else {
                            f"{kind}.md": rendered
                            for kind, rendered in registry._entry_memory_mirrors(
                                project_id
                            ).items()
                        }
                    )
                else:
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
                action = "set" if args.set else "import"
                MemoryMutationAuthorization.terminal(
                    stdin=sys.stdin, stdout=out
                ).authorize(
                    action=action,
                    target=name,
                    preview=content,
                    noun="kind" if memory_format == 2 else "file",
                )
                if memory_format == 2:
                    registry._replace_entry_kind(project_id, name, content)
                    value = registry._entry_memory_view(project_id)
                else:
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
            project = registry.show_project(
                args.project if _PROJECT_ID.fullmatch(args.project) else None,
                name=args.project if not _PROJECT_ID.fullmatch(args.project) else None,
            )
            registry.ensure_memory_supported(project.project_id)
            value = project.to_dict()
        else:
            print(f"zeta: unknown project verb: {args.project_verb}", file=err)
            return 2
    except (
        ProjectRegistryError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True), file=out)
    return 0


def _run_index(args: argparse.Namespace, registry: ProjectRegistry) -> object:
    from ..transcript_search.index import (
        TranscriptIndex,
        TranscriptSource,
        is_indexable_top_level_project_session,
    )

    project = (
        registry.show_project(args.project)
        if _PROJECT_ID.fullmatch(args.project)
        else registry.show_project(name=args.project)
    )
    registry.ensure_memory_supported(project.project_id)
    index = TranscriptIndex(registry.root / project.project_id, project.project_id)
    if args.index_action == "rebuild":
        sources = []
        for link in registry.list_session_links(project.project_id, limit=10_000):
            session_id = link.get("session_id")
            transcript_path = link.get("transcript_path")
            if (
                isinstance(session_id, str)
                and isinstance(transcript_path, str)
                and is_indexable_top_level_project_session(
                    agent_depth=0,
                    project_id=project.project_id,
                    parent_session_id=link.get("parent_session_id"),
                    session_dir=Path(transcript_path),
                )
            ):
                sources.append(
                    TranscriptSource(
                        session_id, Path(transcript_path), project.project_id
                    )
                )
        status = index.rebuild(sources)
        return _status_value(status)
    if args.index_action == "search":
        return [
            {
                "unit_id": hit.unit_id,
                "session_id": hit.unit.session_id,
                "seq_start": hit.unit.seq_start,
                "seq_end": hit.unit.seq_end,
                "kind": hit.unit.kind,
                "match": hit.match,
                "score": hit.score,
                "text": hit.unit.text,
            }
            for hit in index.search(" ".join(args.query), limit=args.limit)
        ]
    return _status_value(index.status())


def _status_value(status: Any) -> dict[str, object]:
    return {
        "project_id": status.project_id,
        "schema_version": status.schema_version,
        "sanitizer_version": status.sanitizer_version,
        "ready": status.ready,
        "detail": status.detail,
        "generation": status.generation,
        "unit_count": status.unit_count,
        "session_count": status.session_count,
        "cursors": status.cursors,
        "path": str(status.path),
        "size_bytes": status.size_bytes,
    }


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
