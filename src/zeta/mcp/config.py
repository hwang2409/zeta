"""Configuration for external MCP servers."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MCPConfigError(ValueError):
    """Raised when an MCP config file is not valid JSON or has a bad shape."""


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    transport: Literal["stdio", "streamable-http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    auth_type: Literal["none", "bearer"] = "none"
    auth_token: str | None = None


@dataclass(frozen=True, slots=True)
class MCPConfig:
    path: Path
    servers: dict[str, MCPServerConfig] = field(default_factory=dict)


def default_config_path() -> Path:
    configured_home = os.environ.get("ZETA_HOME")
    home = Path(configured_home).expanduser() if configured_home else Path.home() / ".zeta"
    return home / "mcp.json"


def load_mcp_config(path: str | Path | None = None) -> MCPConfig:
    """Load MCP config, skipping missing environment-backed servers."""

    selected_path = Path(
        path
        if path is not None
        else os.environ.get("ZETA_MCP_CONFIG", default_config_path())
    ).expanduser()
    if not selected_path.exists():
        logger.info("MCP config not found; continuing without MCP servers: %s", selected_path)
        return MCPConfig(selected_path)
    try:
        with selected_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(
            f"could not parse MCP config {selected_path}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise MCPConfigError(f"could not read MCP config {selected_path}: {exc}") from exc
    if type(value) is not dict:
        raise MCPConfigError(f"MCP config must be an object: {selected_path}")
    raw_servers = value.get("servers", {})
    if type(raw_servers) is not dict:
        raise MCPConfigError(f"MCP config servers must be an object: {selected_path}")

    servers: dict[str, MCPServerConfig] = {}
    for name, raw_server in raw_servers.items():
        if type(name) is not str or not name:
            raise MCPConfigError(f"MCP server names must be nonempty strings: {selected_path}")
        try:
            resolved, missing = _interpolate(raw_server, set())
        except ValueError as exc:
            raise MCPConfigError(f"invalid MCP server {name!r}: {exc}") from exc
        if missing:
            names = ", ".join(sorted(missing))
            logger.warning("skipping MCP server %s; missing environment variables: %s", name, names)
            continue
        try:
            servers[name] = _parse_server(name, resolved)
        except ValueError as exc:
            raise MCPConfigError(f"invalid MCP server {name!r} in {selected_path}: {exc}") from exc
    return MCPConfig(selected_path, servers)


def _interpolate(value: object, missing: set[str]) -> tuple[object, set[str]]:
    if type(value) is str:
        def replace(match: re.Match[str]) -> str:
            variable = match.group(1)
            resolved = os.environ.get(variable)
            if resolved is None:
                missing.add(variable)
                return match.group(0)
            return resolved

        return _ENV_PATTERN.sub(replace, value), missing
    if type(value) is list:
        values = []
        for item in value:
            resolved, _ = _interpolate(item, missing)
            values.append(resolved)
        return values, missing
    if type(value) is dict:
        values: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("object keys must be strings")
            resolved, _ = _interpolate(item, missing)
            values[key] = resolved
        return values, missing
    return value, missing


def _parse_server(name: str, value: object) -> MCPServerConfig:
    if type(value) is not dict:
        raise ValueError("server definition must be an object")
    transport = value.get("transport")
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("transport must be 'stdio' or 'streamable-http'")
    raw_args = value.get("args", [])
    if type(raw_args) is not list or any(type(arg) is not str for arg in raw_args):
        raise ValueError("args must be an array of strings")
    raw_env = value.get("env", {})
    if type(raw_env) is not dict or any(
        type(key) is not str or type(item) is not str for key, item in raw_env.items()
    ):
        raise ValueError("env must be an object of strings")
    raw_auth = value.get("auth", {})
    if raw_auth is None:
        raw_auth = {}
    if type(raw_auth) is not dict:
        raise ValueError("auth must be an object")
    auth_type = raw_auth.get("type", "none")
    if auth_type not in {"none", "bearer"}:
        raise ValueError("auth.type must be 'none' or 'bearer'")
    token = raw_auth.get("token")
    if auth_type == "bearer" and (type(token) is not str or not token):
        raise ValueError("bearer auth requires a token")
    if transport == "stdio":
        command = value.get("command")
        if type(command) is not str or not command:
            raise ValueError("stdio transport requires command")
        if "url" in value:
            raise ValueError("stdio transport does not use url")
    else:
        command = None
        url = value.get("url")
        if type(url) is not str or not url:
            raise ValueError("streamable-http transport requires url")
    url_value = value.get("url") if transport == "streamable-http" else None
    return MCPServerConfig(
        name=name,
        transport=transport,
        command=command,
        args=tuple(raw_args),
        env=dict(raw_env),
        url=url_value if type(url_value) is str else None,
        auth_type=auth_type,
        auth_token=token if type(token) is str else None,
    )


__all__ = ["MCPConfig", "MCPConfigError", "MCPServerConfig", "default_config_path", "load_mcp_config"]
