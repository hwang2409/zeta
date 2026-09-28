"""CLI for the identity-only project/lane registry."""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO

from ..core.session import env_home
from ..project_registry import ProjectRegistry, ProjectRegistryError


def add_subcommand(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("project", help="manage projects and lanes")
    verbs = parser.add_subparsers(dest="project_verb", required=True)
    create = verbs.add_parser("create", help="create a project")
    create.add_argument("name")
    create.add_argument("--scope", required=True)
    create.add_argument("--canonical-integration-root")
    verbs.add_parser("list", help="list projects")
    show = verbs.add_parser("show", help="show a project and its lanes")
    show.add_argument("project", help="project ID or exact name")
    lane = verbs.add_parser("add-lane", help="add an identity-only lane")
    lane.add_argument("project_id")
    lane.add_argument("name")
    lane.add_argument("--scope", required=True)
    lanes = verbs.add_parser("list-lanes", help="list a project's lanes")
    lanes.add_argument("project_id")
    lane_show = verbs.add_parser("show-lane", help="show a lane")
    lane_show.add_argument("project_id")
    lane_show.add_argument("lane_id")


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
            value = registry.create_project(
                args.name, args.scope, args.canonical_integration_root
            ).to_dict(include_lanes=False)
        elif args.project_verb == "list":
            value = [
                project.to_dict(include_lanes=False)
                for project in registry.list_projects()
            ]
        elif args.project_verb == "show":
            value = registry.show_project(
                args.project if args.project.startswith("p_") else None,
                name=None if args.project.startswith("p_") else args.project,
            ).to_dict()
        elif args.project_verb == "add-lane":
            value = registry.add_lane(args.project_id, args.name, args.scope).to_dict()
        elif args.project_verb == "list-lanes":
            value = [lane.to_dict() for lane in registry.list_lanes(args.project_id)]
        elif args.project_verb == "show-lane":
            value = registry.show_lane(args.project_id, args.lane_id).to_dict()
        else:
            print(f"zeta: unknown project verb: {args.project_verb}", file=err)
            return 2
    except ProjectRegistryError as exc:
        print(f"zeta: {exc}", file=err)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True), file=out)
    return 0


__all__ = ["add_subcommand", "run"]
