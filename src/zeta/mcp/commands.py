"""Slash-command helpers for /mcp add and /mcp remove."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import (
    MCPConfigError,
    MCPServerConfig,
    read_mcp_config_file,
    resolve_server_config,
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

    name = server_config.name

    def prepare() -> None:
        def add_entry(
            servers: dict[str, dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            if name in servers:
                raise MCPCommandError(f"MCP server already configured: {name}")
            return {**servers, name: server_to_json(server_config)}

        rewrite_mcp_file(target, add_entry)

    def rollback() -> None:
        rewrite_mcp_file(
            target,
            lambda servers: {
                key: entry for key, entry in servers.items() if key != name
            },
        )

    await mount.add_server(
        resolve_server_config(server_config),
        source=target,
        notice_sink=notice_sink,
        prepare=prepare,
        rollback=rollback,
    )


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

    def persist() -> None:
        rewrite_mcp_file(
            source,
            lambda servers: {
                key: entry for key, entry in servers.items() if key != name
            },
        )

    def replacement() -> tuple[MCPServerConfig, Path] | None:
        if home_path is None or home_path.resolve() == source.resolve():
            return None
        entry = load_home().get(name)
        if entry is None:
            return None
        return resolve_server_config(entry), home_path

    await mount.replace_server(
        name,
        persist=persist,
        replacement=replacement,
        notice_sink=notice_sink,
    )


__all__ = [
    "MCP_USAGE",
    "MCPCommandError",
    "add_and_mount",
    "parse_add_command",
    "remove_and_unshadow",
    "rewrite_mcp_file",
]
