"""Select trusted MCP services and validate exact job grants."""

from __future__ import annotations

from pathlib import Path

from ..core.approval import parse_approval_rule
from ..mcp.config import MCPConfig, home_config_path, load_mcp_config, tool_prefix
from ..mcp.mount import MCPMount, mount_mcp_servers
from ..tools import ToolRegistry
from .models import Job


async def mount_services(job: Job, registry: ToolRegistry, home: Path) -> MCPMount:
    import os

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
        rule = parse_approval_rule(text)
        definition = definitions.get(rule.tool)
        if definition is None:
            raise ValueError(
                f"unknown allowed tool: {rule.tool}; use mounted server__tool names"
            )
        subject_pattern = rule.pattern
        if definition.actions is not None:
            if subject_pattern is None:
                capabilities = [
                    registry.resolve_call(rule.tool, {"action": action})
                    for action in definition.actions
                ]
                if not capabilities:
                    raise ValueError(f"tool {rule.tool} declares no actions")
                continue
            action, separator, remainder = subject_pattern.partition(" ")
            try:
                capability = registry.resolve_call(rule.tool, {"action": action})
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            subject_pattern = remainder if separator else None
        else:
            capability = registry.resolve_call(rule.tool, {})
        if subject_pattern is not None and capability.subject_field is None:
            raise ValueError(
                f"tool {rule.tool} declares no approval subject; configure approval_subjects in home mcp.json"
            )
    policy = registry.approval_policy
    if policy is not None and policy.notices:
        raise ValueError("; ".join(policy.notices))
