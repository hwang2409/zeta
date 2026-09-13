"""Slash-command helpers for /mcp add, remove, auth, and resources."""

from __future__ import annotations

import fcntl
import time
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
from .oauth import MCPOAuthError, authorize
from .oauth_store import load_token, token_state
from .prompt_commands import SlashModelInput
from .resources import (
    MCPResourceError,
    fetch_resource,
    format_resource_list,
    list_resources,
)

MCP_USAGE = (
    "mcp usage: /mcp | /mcp reconnect <server> | /mcp add <name> --stdio <cmd...> "
    "| /mcp add <name> --http <url> [--oauth] "
    "| /mcp remove <name> | /mcp auth <server> "
    "| /mcp resources <server> [<uri>]"
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


def render_mcp_status(mount: MCPMount, *, home: str | None) -> str:
    """Render `/mcp` output including per-server auth state."""

    base = mount.render()
    output: list[str] = []
    for line in base.split("\n"):
        output.append(line)
        for name, config in mount.configs.items():
            prefix = f"{name}: "
            if not line.startswith(prefix):
                continue
            output.append(_render_auth_suffix(name, config, home=home))
            break
    return "\n".join(output)


def _render_auth_suffix(
    name: str, config: MCPServerConfig, *, home: str | None
) -> str:
    if config.auth_type == "oauth":
        token = load_token(name, home=home)
        state = token_state(token)
        suffix = f"  auth: oauth ({state})"
        if token is not None and token.expires_at is not None:
            remaining = int(token.expires_at - time.time())
            suffix += f" | expires_in: {remaining}s"
        if token is not None and token.refresh_error:
            suffix += f" | refresh_error: {token.refresh_error}"
        return suffix
    if config.auth_type == "bearer":
        return "  auth: bearer"
    return "  auth: not-required"


async def run_mcp_auth(
    mount: MCPMount,
    name: str,
    *,
    home: str | None,
    notice_sink: NoticeSink | None,
) -> str:
    """Run the browser OAuth flow for one server, then reconnect it."""

    config = mount.configs.get(name)
    if config is None:
        return f"mcp error: unknown MCP server: {name}"
    if config.transport != "streamable-http" or config.url is None:
        return (
            f"mcp error: /mcp auth requires an HTTP MCP server "
            f"(got transport {config.transport})"
        )
    try:
        await authorize(
            server_name=name,
            server_url=config.url,
            home=home,
            **({"client_id": config.client_id, "client_secret": config.client_secret,
                "callback_port": config.callback_port, "scopes": config.scopes}
               if config.client_id is not None or config.callback_port or config.scopes is not None else {}),
        )
    except MCPOAuthError as exc:
        return f"mcp error: {exc}"
    await mount.reconnect(name, notice_sink=notice_sink)
    return render_mcp_status(mount, home=home)


async def run_mcp_resources_list(mount: MCPMount, server: str) -> str:
    """List one connected server's resources."""

    client = mount.client_for(server)
    if client is None:
        return (
            f"mcp error: {server} is not connected; "
            f"run /mcp reconnect {server} first"
        )
    try:
        resources = await list_resources(client, server=server)
    except MCPResourceError as exc:
        return f"mcp error: {exc}"
    return format_resource_list(server, resources)


async def run_mcp_resource_attach(
    mount: MCPMount, server: str, uri: str
) -> str | SlashModelInput:
    """Fetch one resource and hand it back for the next user turn."""

    client = mount.client_for(server)
    if client is None:
        return (
            f"mcp error: {server} is not connected; "
            f"run /mcp reconnect {server} first"
        )
    try:
        attachment = await fetch_resource(client, server=server, uri=uri)
    except MCPResourceError as exc:
        return f"mcp error: {exc}"
    return SlashModelInput(attachment.labeled_text)


__all__ = [
    "MCP_USAGE",
    "MCPCommandError",
    "add_and_mount",
    "parse_add_command",
    "remove_and_unshadow",
    "render_mcp_status",
    "rewrite_mcp_file",
    "run_mcp_auth",
    "run_mcp_resource_attach",
    "run_mcp_resources_list",
]
