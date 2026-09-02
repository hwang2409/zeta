"""Configuration for external MCP servers."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from ..core.session import env_home

logger = logging.getLogger(__name__)
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


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
    missing_env: tuple[str, ...] = ()
    malformed_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MCPConfig:
    path: Path
    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    skipped_servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    malformed_servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    sources: dict[str, Path] = field(default_factory=dict)

    @property
    def configured_servers(self) -> dict[str, MCPServerConfig]:
        """Return every declared server, valid or reportable."""

        return {**self.servers, **self.skipped_servers, **self.malformed_servers}


def home_config_path(home: str | Path | None = None) -> Path:
    """Return the home-level MCP config path for a given zeta home."""

    root = Path(home).expanduser() if home is not None else env_home()
    return root / "mcp.json"


def project_config_path(project_dir: str | Path) -> Path:
    """Return the per-project overlay MCP config path."""

    return Path(project_dir).expanduser() / ".zeta" / "mcp.json"


def default_config_path() -> Path:
    return home_config_path()


def mcp_log_path(name: str) -> Path:
    """Return the stderr log path used by MCP transports."""

    runtime_root = Path(
        os.environ.get("WIKI_AGENT_RUNTIME_DIR", env_home())
    ).expanduser()
    safe_name = _SAFE_NAME.sub("_", name) or "server"
    return runtime_root / "mcp-logs" / f"{safe_name}.log"


def load_mcp_config(path: str | Path | None = None) -> MCPConfig:
    """Load MCP config from one file, tolerating missing-env and bad entries."""

    selected_path = Path(
        path
        if path is not None
        else os.environ.get("ZETA_MCP_CONFIG", default_config_path())
    ).expanduser()
    return _load_single(selected_path)


def load_mcp_config_overlay(
    *,
    home: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> MCPConfig:
    """Load home config with a per-project overlay merged on top."""

    override = os.environ.get("ZETA_MCP_CONFIG")
    home_path = (
        Path(override).expanduser() if override else home_config_path(home)
    )
    home_config = _load_single(home_path)
    project_path = (
        project_config_path(project_dir) if project_dir is not None else None
    )
    if project_path is None or project_path.resolve() == home_path.resolve():
        return home_config
    project_config = _load_single(project_path)
    return _overlay(home_config, project_config)


def write_mcp_config(path: str | Path, servers: dict[str, dict[str, object]]) -> None:
    """Write a JSON MCP config to path atomically (tmp + rename)."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"servers": servers}, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".mcp.", suffix=".json.tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def read_mcp_config_file(path: str | Path) -> dict[str, dict[str, object]]:
    """Read the raw servers mapping from one config file, or {} if absent."""

    target = Path(path).expanduser()
    if not target.exists():
        return {}
    try:
        with target.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise MCPConfigError(f"could not read MCP config {target}: {exc}") from exc
    if type(value) is not dict:
        raise MCPConfigError(f"MCP config must be an object: {target}")
    raw_servers = value.get("servers", {})
    if type(raw_servers) is not dict:
        raise MCPConfigError(f"MCP config servers must be an object: {target}")
    for name in raw_servers:
        if type(name) is not str or not name:
            raise MCPConfigError(
                f"MCP server names must be nonempty strings: {target}"
            )
    return dict(raw_servers)


def _load_single(selected_path: Path) -> MCPConfig:
    if not selected_path.exists():
        logger.info(
            "MCP config not found; continuing without MCP servers: %s", selected_path
        )
        return MCPConfig(selected_path)
    raw_servers = read_mcp_config_file(selected_path)
    servers: dict[str, MCPServerConfig] = {}
    skipped: dict[str, MCPServerConfig] = {}
    malformed: dict[str, MCPServerConfig] = {}
    sources: dict[str, Path] = {}
    for name, raw_server in raw_servers.items():
        try:
            resolved, missing = _interpolate(raw_server, set())
        except ValueError as exc:
            malformed[name] = MCPServerConfig(
                name, "stdio", malformed_reason=str(exc)
            )
            sources[name] = selected_path
            continue
        if missing:
            names = tuple(sorted(missing))
            logger.warning(
                "skipping MCP server %s; missing environment variables: %s",
                name,
                ", ".join(names),
            )
            skipped[name] = replace(
                _server_config_for_missing_env(name, resolved),
                missing_env=names,
            )
            sources[name] = selected_path
            continue
        try:
            server = _parse_server(name, resolved)
        except ValueError as exc:
            malformed[name] = replace(
                _server_config_for_missing_env(name, resolved),
                malformed_reason=str(exc),
            )
            sources[name] = selected_path
            continue
        servers[name] = server
        sources[name] = selected_path
    return MCPConfig(selected_path, servers, skipped, malformed, sources)


def _overlay(base: MCPConfig, top: MCPConfig) -> MCPConfig:
    """Merge two configs so that top's entries win by name."""

    servers = dict(base.servers)
    skipped = dict(base.skipped_servers)
    malformed = dict(base.malformed_servers)
    for name in {*top.servers, *top.skipped_servers, *top.malformed_servers}:
        for bucket in (servers, skipped, malformed):
            bucket.pop(name, None)
    servers.update(top.servers)
    skipped.update(top.skipped_servers)
    malformed.update(top.malformed_servers)
    return MCPConfig(
        top.path, servers, skipped, malformed, {**base.sources, **top.sources}
    )


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


def _server_config_for_missing_env(name: str, value: object) -> MCPServerConfig:
    """Keep enough shape to report a server skipped or malformed."""

    if type(value) is not dict:
        return MCPServerConfig(name, "stdio")
    transport = value.get("transport")
    if transport not in {"stdio", "streamable-http"}:
        transport = "stdio"
    raw_args = value.get("args", [])
    args = (
        tuple(item for item in raw_args if type(item) is str)
        if type(raw_args) is list
        else ()
    )
    raw_env = value.get("env", {})
    env = (
        {
            key: item
            for key, item in raw_env.items()
            if type(key) is str and type(item) is str
        }
        if type(raw_env) is dict
        else {}
    )
    raw_command = value.get("command")
    command = raw_command if type(raw_command) is str else None
    raw_url = value.get("url")
    url = raw_url if type(raw_url) is str else None
    return MCPServerConfig(name, transport, command=command, args=args, env=env, url=url)


def server_to_json(config: MCPServerConfig) -> dict[str, object]:
    """Serialize one server config to a JSON-safe object, redacted where safe."""

    payload: dict[str, object] = {"transport": config.transport}
    if config.transport == "stdio":
        if config.command is not None:
            payload["command"] = config.command
        if config.args:
            payload["args"] = list(config.args)
    else:
        if config.url is not None:
            payload["url"] = config.url
    if config.env:
        payload["env"] = dict(config.env)
    if config.auth_type == "bearer" and config.auth_token is not None:
        payload["auth"] = {"type": "bearer", "token": config.auth_token}
    return payload


__all__ = [
    "MCPConfig",
    "MCPConfigError",
    "MCPServerConfig",
    "default_config_path",
    "home_config_path",
    "load_mcp_config",
    "load_mcp_config_overlay",
    "mcp_log_path",
    "project_config_path",
    "read_mcp_config_file",
    "server_to_json",
    "write_mcp_config",
]
