"""Slash-command helpers for /mcp add and /mcp remove."""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    """Read, edit, validate, and atomically write one config file."""

    target = Path(path).expanduser()
    with _config_transaction_lock(target):
        try:
            current = read_mcp_config_file(target)
        except MCPConfigError as exc:
            raise MCPCommandError(str(exc)) from exc
        write_mcp_config(target, edit(current))


@contextmanager
def _config_transaction_lock(path: Path) -> Iterator[None]:
    """Hold a stable cross-process lock for a complete config transaction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


async def add_and_mount(
    mount: MCPMount,
    server_config: MCPServerConfig,
    *,
    target: Path,
    notice_sink: NoticeSink | None,
) -> None:
    """Write a new entry to the target file and live-mount it, rolling back on failure."""

    name = server_config.name
    entry = server_to_json(server_config)

    def prepare() -> None:
        def add_entry(
            servers: dict[str, dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            if name in servers:
                raise MCPCommandError(f"MCP server already configured: {name}")
            return {**servers, name: entry}

        rewrite_mcp_file(target, add_entry)

    def rollback() -> None:
        def remove_created_entry(
            servers: dict[str, dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            if servers.get(name) != entry:
                return servers
            return {key: value for key, value in servers.items() if key != name}

        rewrite_mcp_file(target, remove_created_entry)

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

    def persist(locked_source: Path) -> None:
        rewrite_mcp_file(
            locked_source,
            lambda servers: {
                key: entry for key, entry in servers.items() if key != name
            },
        )

    def replacement(locked_source: Path) -> tuple[MCPServerConfig, Path] | None:
        if home_path is None or home_path.resolve() == locked_source.resolve():
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
