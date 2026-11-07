"""Slash-command helpers for /mcp add, remove, auth, and resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..model_input import ModelInputEnvelope
from ..protocol.types import MessageOrigin
from .config import MCPServerConfig
from .management import ManagedServer
from .mount import MCPMount
from .resources import (
    MCPResourceError,
    fetch_resource,
    format_resource_list,
    list_resources,
)

MCP_USAGE = (
    "mcp usage: /mcp [status] | /mcp reconnect <server> | /mcp add <name> --stdio <cmd...> "
    "| /mcp add <name> --http <url> [--oauth] "
    "| /mcp remove <name> | /mcp enable|disable <name> "
    "| /mcp trust|untrust <name> | /mcp auth <server> | /mcp logout <server> "
    "| /mcp resources <server> [<uri>]"
)


class MCPCommandError(ValueError):
    """User-facing error for /mcp subcommands."""


def parse_add_command(tokens: list[str]) -> MCPServerConfig:
    """Turn `add <name> --stdio <cmd...>` / `--http <url>` into a server config."""

    if len(tokens) < 3:
        raise MCPCommandError("add requires <name> --stdio <cmd...> or --http <url>")
    name = tokens[0]
    if not name or name.startswith("-") or any(c.isspace() for c in name):
        raise MCPCommandError(f"invalid MCP server name: {name!r}")
    flag = tokens[1]
    rest = tokens[2:]
    if flag == "--stdio":
        return MCPServerConfig(
            name=name,
            transport="stdio",
            command=rest[0],
            args=tuple(rest[1:]),
        )
    if flag == "--http":
        auth_type = "none"
        url_tokens = list(rest)
        if url_tokens and url_tokens[-1] == "--oauth":
            auth_type = "oauth"
            url_tokens.pop()
        if len(url_tokens) != 1:
            raise MCPCommandError("--http takes exactly one url argument")
        return MCPServerConfig(
            name=name,
            transport="streamable-http",
            url=url_tokens[0],
            auth_type=auth_type,
        )
    raise MCPCommandError(f"unknown add flag: {flag}")


def render_mcp_status(
    mount: MCPMount,
    *,
    credential_status: Mapping[str, str] | None = None,
    configured_entries: Sequence[ManagedServer] = (),
) -> str:
    """Render `/mcp` output from resident, non-secret credential state."""

    statuses = credential_status or {}
    base = mount.render()
    output: list[str] = []
    for line in base.split("\n"):
        output.append(line)
        for name, config in mount.configs.items():
            if line.startswith(f"{name}: "):
                output.append(_render_auth_suffix(name, config, statuses))
                break
    mounted = set(mount.configs)
    effective = {item.name: item for item in configured_entries}
    for name, item in effective.items():
        if name not in mounted:
            output.append(f"{name}: {item.status}")
    return "\n".join(output)


def _render_auth_suffix(
    name: str, config: MCPServerConfig, statuses: Mapping[str, str]
) -> str:
    if config.auth_type == "oauth":
        return f"  auth: oauth ({statuses.get(name, 'unauthorized')})"
    if config.auth_type == "bearer":
        return "  auth: bearer"
    return "  auth: not-required"


async def run_mcp_resources_list(mount: MCPMount, server: str) -> str:
    """List one connected server's resources."""

    client = mount.client_for(server)
    if client is None:
        return (
            f"mcp error: {server} is not connected; run /mcp reconnect {server} first"
        )
    try:
        resources = await list_resources(client, server=server)
    except MCPResourceError as exc:
        return f"mcp error: {exc}"
    return format_resource_list(server, resources)


async def run_mcp_resource_attach(
    mount: MCPMount, server: str, uri: str
) -> str | ModelInputEnvelope:
    """Fetch one resource and hand it back for the next user turn."""

    client = mount.client_for(server)
    if client is None:
        return (
            f"mcp error: {server} is not connected; run /mcp reconnect {server} first"
        )
    try:
        attachment = await fetch_resource(
            client,
            server=server,
            uri=uri,
            spill_store=mount.registry.spills if mount.registry is not None else None,
        )
    except MCPResourceError as exc:
        return f"mcp error: {exc}"
    return ModelInputEnvelope(
        attachment.labeled_text, "", MessageOrigin.SLASH_EXPANSION
    )


__all__ = [
    "MCP_USAGE",
    "MCPCommandError",
    "parse_add_command",
    "render_mcp_status",
    "run_mcp_resource_attach",
    "run_mcp_resources_list",
]
