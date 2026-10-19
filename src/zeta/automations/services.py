"""Select trusted MCP services and validate exact job grants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..core.approval import parse_approval_rule
from ..core.session import SessionManager
from ..mcp.config import MCPConfig, home_config_path, load_mcp_config, tool_prefix
from ..mcp.mount import MCPMount, mount_mcp_servers
from ..project_errors import ProjectNotFoundError
from ..tools import ToolRegistry
from ..tools import inbox as inbox_tool
from .models import Job


@dataclass(frozen=True)
class InboxGrants:
    source_id: str
    target_ids: tuple[str, ...]


def resolve_inbox_grants(job: Job, project_registry) -> InboxGrants | None:
    rules = [parse_approval_rule(text) for text in job.allow]
    inbox_rules = [rule for rule in rules if rule.tool == "inbox"]
    if not inbox_rules:
        return None
    source = project_registry.find_for_directory(job.cwd)
    if source is None:
        raise ValueError("inbox automation cwd must be inside a registered project")
    targets = []
    for rule in inbox_rules:
        scope = rule.subject_pattern or ""
        action, _, subject = scope.partition(" ")
        if scope and not subject:
            subject = scope if action != "send" else None
        if subject and any(c in subject for c in "*?["):
            raise ValueError(
                "inbox automation target must be a project name or id, not a wildcard; use inbox(send <project>)"
            )
        if action != "send" or not subject:
            raise ValueError("inbox automation rules must use scoped inbox(send <project>)")
        try:
            target = project_registry.show_project(project_id=subject)
        except ProjectNotFoundError:
            try:
                target = project_registry.show_project(name=subject)
            except ProjectNotFoundError as exc:
                raise ValueError(
                    f"unknown inbox target project {subject!r}; use inbox(send <project>)"
                ) from exc
        targets.append(target.project_id)
    return InboxGrants(source.project_id, tuple(dict.fromkeys(targets)))


def configure_inbox(
    job: Job, registry: ToolRegistry, home: Path, grants: InboxGrants | None = None
) -> None:
    if grants is None:
        project_registry = registry.project_registry or SessionManager(home).project_registry
        grants = resolve_inbox_grants(job, project_registry)
    if grants is None:
        return
    registry.project_registry = registry.project_registry or SessionManager(home).project_registry
    registry.project_id = grants.source_id
    registry.inbox_target_ids = frozenset(grants.target_ids)
    registry.unregister("inbox")
    inbox_tool.register_send_only(registry)


async def mount_services(job: Job, registry: ToolRegistry, home: Path, grants=None) -> MCPMount:
    configure_inbox(job, registry, home, grants)
    config = load_mcp_config(os.environ.get("ZETA_MCP_CONFIG", str(home_config_path(home))))
    selected = set(job.servers) | {"slack"}
    missing = selected - config.servers.keys()
    if missing:
        raise ValueError(f"unavailable MCP configuration: {', '.join(sorted(missing))}")
    slack = config.servers["slack"]
    if slack.url != "https://mcp.slack.com/mcp" or slack.auth_type != "oauth":
        raise ValueError("slack must use the official https://mcp.slack.com/mcp endpoint with OAuth")
    subset = MCPConfig(config.path, {name: config.servers[name] for name in sorted(selected)}, sources={name: config.sources[name] for name in selected if name in config.sources})
    mount = await mount_mcp_servers(registry, subset, home=str(home))
    try:
        failed = [name for name, status in mount.statuses.items() if status.state != "mounted"]
        if failed:
            raise ValueError(f"MCP servers failed to mount: {', '.join(failed)}; check /mcp status and authentication")
        requested = [parse_approval_rule(text).tool for text in job.allow]
        requested.append("slack__slack_send_message")
        catalog_prefixes = tuple(tool_prefix(name) for name in selected)
        catalog_requested = [name for name in requested if name.startswith(catalog_prefixes)]
        _activated, rejected = mount.activate_tools(registry, catalog_requested)
        if rejected:
            raise ValueError("MCP tool activation failed: " + "; ".join(rejected))
        validate_permissions(job, registry)
        sender = registry.definitions_by_name.get("slack__slack_send_message")
        properties = sender.parameters.get("properties", {}) if sender is not None else {}
        if not {"channel_id", "message"} <= properties.keys():
            raise ValueError("Slack delivery tool is unavailable or has an incompatible schema; check chat:write scope")
        return mount
    except BaseException:
        await mount.close()
        raise


def validate_permissions(job: Job, registry: ToolRegistry) -> None:
    definitions = registry.definitions_by_name
    for text in job.allow:
        parsed = parse_approval_rule(text)
        if parsed.tool not in definitions:
            raise ValueError(f"unknown allowed tool: {parsed.tool}; use mounted server__tool names")
        registry.resolve_approval_rule(parsed)
    policy = registry.approval_policy
    if policy is not None and policy.notices:
        raise ValueError("; ".join(policy.notices))
