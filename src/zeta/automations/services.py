"""Select trusted MCP services and validate exact job grants."""

from __future__ import annotations

from pathlib import Path

from ..core.approval import parse_approval_rule
from ..core.session import SessionManager
from ..mcp.config import MCPConfig, home_config_path, load_mcp_config, tool_prefix
from ..mcp.mount import MCPMount, mount_mcp_servers
from ..project_errors import ProjectNotFoundError
from ..tools import ToolRegistry
from ..tools import inbox as inbox_tool
from .models import Job


def configure_inbox(job: Job, registry: ToolRegistry, home: Path) -> None:
    rules = [parse_approval_rule(text) for text in job.allow]
    inbox_rules = [rule for rule in rules if rule.tool == "inbox"]
    if not inbox_rules:
        return
    project_registry = registry.project_registry or SessionManager(home).project_registry
    project = project_registry.find_for_directory(job.cwd)
    if project is None:
        raise ValueError(
            "inbox automation cwd must be inside a registered project"
        )
    registry.project_registry = project_registry
    registry.project_id = project.project_id
    registry.unregister("inbox")
    inbox_tool.register_send_only(registry)
    for rule in inbox_rules:
        if rule.subject_pattern and any(
            character in rule.subject_pattern for character in "*?["
        ):
            raise ValueError(
                "inbox automation target must be a project name or id, not a wildcard; "
                "use inbox(send <project>)"
            )
        resolved = registry.resolve_approval_rule(rule)
        if resolved.action != "send" or resolved.subject_pattern is None:
            raise ValueError(
                "inbox automation rules must use scoped inbox(send <project>)"
            )
        if any(character in resolved.subject_pattern for character in "*?["):
            raise ValueError(
                "inbox automation target must be a project name or id, not a wildcard; "
                "use inbox(send <project>)"
            )
        try:
            project_registry.show_project(project_id=resolved.subject_pattern)
        except ProjectNotFoundError:
            try:
                project_registry.show_project(name=resolved.subject_pattern)
            except ProjectNotFoundError as exc:
                raise ValueError(
                    f"unknown inbox target project {resolved.subject_pattern!r}; "
                    "use inbox(send <project>)"
                ) from exc


async def mount_services(job: Job, registry: ToolRegistry, home: Path) -> MCPMount:
    import os

    configure_inbox(job, registry, home)
    config = load_mcp_config(
        os.environ.get("ZETA_MCP_CONFIG", str(home_config_path(home)))
    )
    # Slack is independently selected by the fixed delivery target.
    selected = set(job.servers) | {"slack"}
    missing = selected - config.servers.keys()
    if missing:
        raise ValueError(f"unavailable MCP configuration: {', '.join(sorted(missing))}")
    slack = config.servers["slack"]
    if slack.url != "https://mcp.slack.com/mcp" or slack.auth_type != "oauth":
        raise ValueError(
            "slack must use the official https://mcp.slack.com/mcp endpoint with OAuth"
        )
    subset = MCPConfig(
        config.path,
        {name: config.servers[name] for name in sorted(selected)},
        sources={
            name: config.sources[name] for name in selected if name in config.sources
        },
    )
    mount = await mount_mcp_servers(registry, subset, home=str(home))
    try:
        failed = [
            name for name, status in mount.statuses.items() if status.state != "mounted"
        ]
        if failed:
            raise ValueError(
                f"MCP servers failed to mount: {', '.join(failed)}; check /mcp status and authentication"
            )
        requested = [parse_approval_rule(text).tool for text in job.allow]
        requested.append("slack__slack_send_message")
        # Only names owned by a selected MCP catalog (matched by exact server
        # prefix) go through activation; built-ins like ``read`` are already
        # registered and are validated the normal way below. MCP-prefixed
        # names that collide with foreign definitions still fail activation.
        catalog_prefixes = tuple(tool_prefix(name) for name in selected)
        catalog_requested = [
            name for name in requested if name.startswith(catalog_prefixes)
        ]
        _activated, rejected = mount.activate_tools(registry, catalog_requested)
        if rejected:
            raise ValueError("MCP tool activation failed: " + "; ".join(rejected))
        validate_permissions(job, registry)
        sender = registry.definitions_by_name.get("slack__slack_send_message")
        properties = (
            sender.parameters.get("properties", {}) if sender is not None else {}
        )
        if not {"channel_id", "message"} <= properties.keys():
            raise ValueError(
                "Slack delivery tool is unavailable or has an incompatible schema; check chat:write scope"
            )
        return mount
    except BaseException:
        await mount.close()
        raise


def validate_permissions(job: Job, registry: ToolRegistry) -> None:
    definitions = registry.definitions_by_name
    for text in job.allow:
        parsed = parse_approval_rule(text)
        if parsed.tool not in definitions:
            raise ValueError(
                f"unknown allowed tool: {parsed.tool}; use mounted server__tool names"
            )
        registry.resolve_approval_rule(parsed)
    policy = registry.approval_policy
    if policy is not None and policy.notices:
        raise ValueError("; ".join(policy.notices))
