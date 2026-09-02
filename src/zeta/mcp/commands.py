"""Slash-command helpers for /mcp add and /mcp remove."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import (
    MCPConfigError,
    MCPServerConfig,
    read_mcp_config_file,
    server_to_json,
    write_mcp_config,
)
from .mount import MCPMount, NoticeSink

MCP_USAGE = (
    "mcp usage: /mcp | /mcp reconnect <server> | /mcp add <name> --stdio <cmd...> "
    "| /mcp add <name> --http <url> | /mcp remove <name>"
)


class MCPCommandError(ValueError):
    """User-facing error for /mcp subcommands."""


def parse_add_command(tokens: list[str]) -> MCPServerConfig:
    """Turn `add <name> --stdio <cmd...>` / `--http <url>` into a server config."""

    if len(tokens) < 3:
        raise MCPCommandError(
            "add requires <name> --stdio <cmd...> or --http <url>"
        )
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
        if len(rest) != 1:
            raise MCPCommandError("--http takes exactly one url argument")
        return MCPServerConfig(name=name, transport="streamable-http", url=rest[0])
    raise MCPCommandError(f"unknown add flag: {flag}")


def rewrite_mcp_file(
    path: Path,
    edit: Callable[[dict[str, dict[str, object]]], dict[str, dict[str, object]]],
) -> None:
    """Read one config file, apply edit, and atomically write it back."""

    try:
        current = read_mcp_config_file(path)
    except MCPConfigError as exc:
        raise MCPCommandError(str(exc)) from exc
    write_mcp_config(path, edit(current))


async def add_and_mount(
    mount: MCPMount,
    server_config: MCPServerConfig,
    *,
    target: Path,
    notice_sink: NoticeSink | None,
) -> None:
    """Write a new entry to the target file and live-mount it, rolling back on failure."""

    if server_config.name in mount.configs:
        raise MCPCommandError(f"MCP server already configured: {server_config.name}")
    rewrite_mcp_file(
        target,
        lambda servers: {
            **servers,
            server_config.name: server_to_json(server_config),
        },
    )
    try:
        await mount.add_server(
            server_config, source=target, notice_sink=notice_sink
        )
    except BaseException:
        rewrite_mcp_file(
            target,
            lambda servers: {
                name: entry
                for name, entry in servers.items()
                if name != server_config.name
            },
        )
        raise


async def remove_and_unshadow(
    mount: MCPMount,
    name: str,
    *,
    home_path: Path | None,
    load_home: Callable[[], dict[str, MCPServerConfig]],
    notice_sink: NoticeSink | None,
) -> None:
    """Remove one entry from its owning file, unmount it, remount any home shadow."""

    source = mount.sources.get(name)
    if source is None:
        raise MCPCommandError(f"unknown MCP server: {name}")
    rewrite_mcp_file(
        source,
        lambda servers: {
            key: entry for key, entry in servers.items() if key != name
        },
    )
    await mount.remove_server(name)
    if home_path is None or home_path.resolve() == source.resolve():
        return
    entry = load_home().get(name)
    if entry is None:
        return
    await mount.add_server(entry, source=home_path, notice_sink=notice_sink)


__all__ = [
    "MCP_USAGE",
    "MCPCommandError",
    "add_and_mount",
    "parse_add_command",
    "remove_and_unshadow",
    "rewrite_mcp_file",
]
