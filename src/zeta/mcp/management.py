"""Offline-first persistence and live lifecycle management for MCP servers."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlsplit, urlunsplit

from . import connection as mcp_connection
from .config import (
    MCPConfig,
    home_config_path,
    load_mcp_config,
    load_mcp_config_overlay,
    project_config_path,
    read_mcp_config_file,
)
from .oauth_store import load_token, token_state

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

Scope = Literal["user", "project", "effective"]


class MCPManagementError(RuntimeError):
    """A user-correctable MCP management error."""


@dataclass(frozen=True, slots=True)
class ManagedServer:
    name: str
    scope: Literal["user", "project"]
    config: dict[str, object]
    enabled: bool
    trusted: bool
    status: str = "configured"
    auth: str = "none"
    tool_count: int = 0

    def as_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope,
            "enabled": self.enabled,
            "trusted": self.trusted,
            "status": self.status,
            "auth": self.auth,
            "tool_count": self.tool_count,
            "config": self.config,
        }


class _LockedFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_name(self.path.name + ".lock").open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


class MCPManagementService:
    """Shared persistence, trust, redaction, and lifecycle policy for CLI/TUI."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        project_dir: str | Path | None = None,
        mount: object | None = None,
    ) -> None:
        self.home = Path(home).expanduser() if home is not None else None
        self.project_dir = (
            Path(project_dir).expanduser().resolve() if project_dir is not None else None
        )
        self.mount = mount

    def path(self, scope: Literal["user", "project"]) -> Path:
        if scope == "project":
            if self.project_dir is None:
                raise MCPManagementError("project scope requires a repository")
            return project_config_path(self.project_dir)
        override = os.environ.get("ZETA_MCP_CONFIG")
        return Path(override).expanduser() if override else home_config_path(self.home)

    def _raw(self, scope: Literal["user", "project"]) -> dict[str, object]:
        return read_mcp_config_file(self.path(scope))

    @staticmethod
    def _redacted_url(value: str) -> str:
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            port = parsed.port
            if parsed.scheme not in {"http", "https"} or not host:
                return "<redacted-url>"
            if port is not None:
                host += f":{port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except ValueError:
            return "<redacted-url>"

    @staticmethod
    def _validate_url(value: str) -> None:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            _port = parsed.port
        except ValueError as exc:
            raise MCPManagementError("invalid MCP server URL") from exc
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise MCPManagementError("invalid MCP server URL")

    @classmethod
    def _safe(cls, value: object, *, key: str | None = None) -> object:
        lowered = (key or "").casefold()
        if isinstance(value, dict):
            if lowered == "headers":
                return {str(k): "<redacted>" for k in value}
            return {str(k): cls._safe(v, key=str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe(item, key=key) for item in value]
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                return value
            if lowered == "url":
                return cls._redacted_url(value)
            if lowered in {"token", "client_secret", "authorization", "proxy-authorization"}:
                return "<redacted>"
            if key is not None and value.startswith("${") and value.endswith("}"):
                return value
            if lowered not in {"transport", "command", "args", "type", "client_id", "scopes"} and key is not None:
                return "<redacted>"
        return value

    @classmethod
    def redact(cls, value: dict[str, object]) -> dict[str, object]:
        return cls._safe(copy.deepcopy(value))  # type: ignore[return-value]

    def _trust_path(self) -> Path:
        return home_config_path(self.home).parent / "mcp-trust.json"

    def _trust_key(self, name: str) -> str:
        if self.project_dir is None:
            raise MCPManagementError("project scope requires a repository")
        identity = f"{self.project_dir}\0{name}".encode()
        return hashlib.sha256(identity).hexdigest()

    @staticmethod
    def trust_fingerprint(raw: dict[str, object]) -> str:
        security = {
            key: raw.get(key)
            for key in (
                "transport", "command", "args", "env", "url", "headers", "auth", "client"
            )
        }
        payload = json.dumps(security, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def is_trusted(self, name: str, raw: dict[str, object], scope: str) -> bool:
        if scope != "project" or raw.get("transport") != "stdio":
            return True
        try:
            data = json.loads(self._trust_path().read_text())
            return data.get(self._trust_key(name)) == self.trust_fingerprint(raw)
        except (OSError, ValueError, AttributeError):
            return False

    def trust(self, name: str) -> ManagedServer:
        raw = self._raw("project").get(name)
        if raw is None:
            raise MCPManagementError(f"unknown project MCP server: {name}")
        path = self._trust_path()
        with _LockedFile(path):
            data = self._read_object(path)
            data[self._trust_key(name)] = self.trust_fingerprint(raw)
            self._atomic_json(path, data, mode=0o600)
        return self.show(name, scope="project")

    def untrust(self, name: str) -> ManagedServer:
        raw = self._raw("project").get(name)
        if raw is None:
            raise MCPManagementError(f"unknown project MCP server: {name}")
        path = self._trust_path()
        with _LockedFile(path):
            data = self._read_object(path)
            data.pop(self._trust_key(name), None)
            self._atomic_json(path, data, mode=0o600)
        return self.show(name, scope="project")

    @staticmethod
    def _read_object(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise MCPManagementError(f"invalid JSON file: {path}") from exc
        if type(value) is not dict:
            raise MCPManagementError(f"JSON document must be an object: {path}")
        return value

    @staticmethod
    def _atomic_json(path: Path, data: object, *, mode: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".mcp-management.", dir=str(path.parent))
        try:
            if mode is not None:
                os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(data, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _mutate(
        self,
        scope: Literal["user", "project"],
        change: Callable[[dict[str, dict[str, object]]], None],
    ) -> None:
        path = self.path(scope)
        with _LockedFile(path):
            document = self._read_object(path)
            raw_servers = document.get("servers", {})
            if type(raw_servers) is not dict:
                raise MCPManagementError(f"MCP config servers must be an object: {path}")
            servers = copy.deepcopy(raw_servers)
            change(servers)  # type: ignore[arg-type]
            document["servers"] = servers
            self._atomic_json(path, document)

    def add(
        self,
        name: str,
        *,
        scope: Literal["user", "project"],
        command: str | None = None,
        args: tuple[str, ...] = (),
        url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        oauth: bool = False,
        enabled: bool = True,
    ) -> ManagedServer:
        if not name:
            raise MCPManagementError("server name must not be empty")
        if (command is None) == (url is None):
            raise MCPManagementError("specify exactly one of command or url")
        raw: dict[str, object] = {
            "transport": "stdio" if command is not None else "streamable-http",
            "enabled": enabled,
        }
        if command is not None:
            raw.update(command=command, args=list(args))
        else:
            assert url is not None
            self._validate_url(url)
            raw["url"] = url
        if env:
            raw["env"] = dict(env)
        if headers:
            raw["headers"] = dict(headers)
        if oauth:
            if command is not None:
                raise MCPManagementError("OAuth requires HTTP")
            raw["auth"] = {"type": "oauth"}

        def change(servers: dict[str, dict[str, object]]) -> None:
            servers[name] = raw

        self._mutate(scope, change)
        return self.show(name, scope=scope)

    def remove(self, name: str, *, scope: Literal["user", "project"]) -> None:
        def change(servers: dict[str, dict[str, object]]) -> None:
            if name not in servers:
                raise MCPManagementError(f"unknown MCP server: {name}")
            del servers[name]

        self._mutate(scope, change)

    def set_enabled(
        self, name: str, *, scope: Literal["user", "project"], enabled: bool
    ) -> ManagedServer:
        def change(servers: dict[str, dict[str, object]]) -> None:
            if name not in servers:
                raise MCPManagementError(f"unknown MCP server: {name}")
            servers[name]["enabled"] = enabled

        self._mutate(scope, change)
        return self.show(name, scope=scope)

    def _runtime_status(self, name: str) -> tuple[str, int]:
        statuses = getattr(self.mount, "statuses", {}) if self.mount is not None else {}
        status = statuses.get(name) if isinstance(statuses, dict) else None
        state = getattr(status, "state", None)
        actors = getattr(self.mount, "_actors", {}) if self.mount is not None else {}
        actor = actors.get(name) if isinstance(actors, dict) else None
        tools = getattr(actor, "_tools", ()) if actor is not None else ()
        return (str(state), len(tools)) if state else ("configured", 0)

    def _managed(
        self, name: str, scope: Literal["user", "project"], raw: object
    ) -> ManagedServer:
        if type(raw) is not dict:
            return ManagedServer(
                name,
                scope,
                {"malformed_reason": "server definition must be an object"},
                False,
                False,
                "malformed",
            )
        enabled = raw.get("enabled", True) is not False
        trusted = self.is_trusted(name, raw, scope)
        auth_type = "none"
        auth = raw.get("auth")
        if isinstance(auth, dict):
            auth_type = str(auth.get("type", "none"))
        auth_status = token_state(load_token(name, home=self.home)) if auth_type == "oauth" else auth_type
        runtime_status, tool_count = self._runtime_status(name)
        if not enabled:
            status = "disabled"
        elif not trusted:
            status = "pending trust"
        else:
            status = runtime_status
        return ManagedServer(
            name, scope, self.redact(raw), enabled, trusted, status, auth_status, tool_count
        )

    def list(self, *, scope: Scope = "effective") -> list[ManagedServer]:
        if scope == "effective":
            result: dict[str, ManagedServer] = {
                name: self._managed(name, "user", raw)
                for name, raw in self._raw("user").items()
            }
            if self.project_dir is not None:
                result.update(
                    {
                        name: self._managed(name, "project", raw)
                        for name, raw in self._raw("project").items()
                    }
                )
            return list(result.values())
        return [self._managed(name, scope, raw) for name, raw in self._raw(scope).items()]

    def show(self, name: str, *, scope: Scope = "effective") -> ManagedServer:
        item = next((entry for entry in self.list(scope=scope) if entry.name == name), None)
        if item is None:
            raise MCPManagementError(f"unknown MCP server: {name}")
        return item

    def runtime_config(self) -> MCPConfig:
        """Return effective config with disabled and untrusted project stdio removed."""
        config = load_mcp_config_overlay(home=self.home, project_dir=self.project_dir)
        allowed_names: set[str] = set()
        sources: dict[str, Path] = {}
        project_path = self.path("project").resolve() if self.project_dir is not None else None
        project_raw = self._raw("project") if self.project_dir is not None else {}
        user_raw = self._raw("user")
        for name, server in config.configured_servers.items():
            source = config.sources.get(name, config.path)
            is_project = project_path is not None and source.resolve() == project_path
            raw = project_raw.get(name, {}) if is_project else user_raw.get(name, {})
            if type(raw) is not dict:
                continue
            enabled = raw.get("enabled", server.enabled) is not False
            if not enabled or (is_project and not self.is_trusted(name, raw, "project")):
                continue
            allowed_names.add(name)
            sources[name] = source
        return replace(
            config,
            servers={name: value for name, value in config.servers.items() if name in allowed_names},
            skipped_servers={
                name: value
                for name, value in config.skipped_servers.items()
                if name in allowed_names
            },
            malformed_servers={
                name: value
                for name, value in config.malformed_servers.items()
                if name in allowed_names
            },
            sources=sources,
        )

    async def sync_runtime(self) -> None:
        """Reconcile the attached running mount with persisted effective config."""
        if self.mount is None:
            return
        desired = self.runtime_config()
        current = dict(getattr(self.mount, "configs", {}))
        for name in tuple(current):
            if name not in desired.servers or current[name] != desired.servers[name]:
                await self.mount.remove_server(name)
        current_names = set(getattr(self.mount, "configs", {}))
        for name, config in desired.servers.items():
            if name not in current_names:
                await self.mount.add_server(
                    config,
                    source=desired.sources.get(name, desired.path),
                )

    def _policy_registry(self) -> ToolRegistry | None:
        return getattr(self.mount, "registry", None)

    def _server_allowed(self, name: str) -> bool:
        return mcp_connection.mcp_server_allowed(self._policy_registry(), name)

    async def test(self, name: str, *, scope: Scope = "effective") -> dict[str, object]:
        item = self.show(name, scope=scope)
        if not self._server_allowed(name):
            return {
                "name": name,
                "tools": [],
                "status": "skipped-policy",
                "detail": "skipped by tool policy",
            }
        path = self.path(item.scope)
        config = load_mcp_config(path).configured_servers.get(name)
        if config is None:
            raise MCPManagementError(f"server cannot be tested: {name}")
        client = mcp_connection.build_mcp_client(
            config,
            registry=self._policy_registry(),
            home=str(self.home) if self.home else None,
        )
        try:
            await client.connect()
            tools = await client.list_tools()
            summaries = []
            for tool in sorted(tools, key=lambda item: item.name):
                description = tool.description
                if len(description) > 120:
                    description = description[:117] + "..."
                summaries.append({"name": tool.name, "description": description})
            return {"name": name, "tools": summaries, "status": "ok"}
        finally:
            await client.close()

    async def login(self, name: str, *, scope: Scope = "effective") -> None:
        from .oauth import authorize

        item = self.show(name, scope=scope)
        if not self._server_allowed(name):
            raise MCPManagementError(f"{name}: skipped by tool policy")
        if item.config.get("transport") != "streamable-http":
            raise MCPManagementError("OAuth requires HTTP")
        config = load_mcp_config(self.path(item.scope)).configured_servers.get(name)
        if config is None:
            raise MCPManagementError(f"server cannot be used for OAuth: {name}")
        await authorize(
            server_name=name,
            server_url=config.url or "",
            home=str(self.home) if self.home else None,
            client_id=config.client_id,
            client_secret=config.client_secret,
            callback_port=config.callback_port,
            scopes=config.scopes,
        )

    def logout(self, name: str, *, scope: Scope = "effective") -> None:
        from .oauth_store import delete_token

        # OAuth tokens are name-global, but the requested definition must exist
        # in the selected scope before its token is removed.
        self.show(name, scope=scope)
        delete_token(name, home=self.home)


__all__ = ["MCPManagementError", "MCPManagementService", "ManagedServer", "Scope"]
