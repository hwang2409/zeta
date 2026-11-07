"""Off-loop persistence and live lifecycle management for MCP servers."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit, urlunsplit

from . import connection as mcp_connection
from .config import (
    MCPConfig,
    home_config_path,
    load_mcp_config,
    load_mcp_config_overlay,
    project_config_path,
)
from .oauth_store import (
    MCPCredentialStore,
    MCPCredentialStoreError,
    PublicCredentialStoreError,
    public_credential_store_error,
    token_state,
)

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

Scope = Literal["user", "project", "effective"]
_ConfigScope = Literal["user", "project"]
_LOCK_TIMEOUT_SECONDS = 1.0


class MCPManagementError(RuntimeError):
    """A user-correctable MCP management error."""


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ManagedServer:
    name: str
    scope: _ConfigScope
    config: Mapping[str, object]
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
            "config": _thaw(self.config),
        }


@dataclass(frozen=True, slots=True)
class MCPManagerSnapshot:
    entries: tuple[ManagedServer, ...]
    generation: int


@dataclass(frozen=True, slots=True)
class _ManagementCandidate:
    entries: tuple[ManagedServer, ...]
    runtime_config: MCPConfig
    home_path: Path
    project_path: Path | None


class MCPManagementService:
    """Own bounded config/trust transactions and one immutable display snapshot."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        project_dir: str | Path | None = None,
        mount: object | None = None,
        lock_timeout: float = _LOCK_TIMEOUT_SECONDS,
    ) -> None:
        # These values are deliberately lexical. Canonicalization belongs to the
        # first worker transaction because Path.resolve() can access storage.
        self.home = Path(home) if home is not None else None
        self.project_dir = Path(project_dir) if project_dir is not None else None
        self.mount = mount
        self._lock_timeout = lock_timeout
        self._transaction_gate = asyncio.Lock()
        self._snapshot = MCPManagerSnapshot((), 0)
        self._runtime: MCPConfig | None = None
        self._home_path: Path | None = None
        self._project_path: Path | None = None
        self._paths_ready = False
        self._credential_store = MCPCredentialStore(home=self.home)
        self._credential_status: dict[str, str] = {}
        self.credential_store_error: PublicCredentialStoreError | None = None

    @property
    def snapshot(self) -> MCPManagerSnapshot:
        """Return the current immutable, storage-free display state."""

        return self._snapshot

    def path(self, scope: _ConfigScope) -> Path:
        if scope == "project":
            if self._project_path is not None:
                return self._project_path
            if self.project_dir is None:
                raise MCPManagementError("project scope requires a repository")
            return project_config_path(self.project_dir)
        if self._home_path is not None:
            return self._home_path
        override = os.environ.get("ZETA_MCP_CONFIG")
        return Path(override).expanduser() if override else home_config_path(self.home)

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
            if lowered in {
                "token",
                "client_secret",
                "authorization",
                "proxy-authorization",
            }:
                return "<redacted>"
            if (
                lowered
                not in {"transport", "command", "args", "type", "client_id", "scopes"}
                and key is not None
            ):
                return "<redacted>"
        return value

    @classmethod
    def redact(cls, value: dict[str, object]) -> dict[str, object]:
        return cls._safe(copy.deepcopy(value))  # type: ignore[return-value]

    @staticmethod
    def trust_fingerprint(raw: dict[str, object]) -> str:
        security = {
            key: raw.get(key)
            for key in (
                "transport",
                "command",
                "args",
                "env",
                "url",
                "headers",
                "auth",
                "client",
            )
        }
        payload = json.dumps(security, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

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
        fd, temporary = tempfile.mkstemp(
            prefix=".mcp-management.", dir=str(path.parent)
        )
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

    def _canonical_paths(self) -> tuple[Path, Path | None, Path]:
        if self._paths_ready:
            assert self._home_path is not None
            return (
                self._home_path,
                self._project_path,
                self._home_path.parent / "mcp-trust.json",
            )
        override = os.environ.get("ZETA_MCP_CONFIG")
        home_path = (
            Path(override).expanduser() if override else home_config_path(self.home)
        ).resolve()
        project_dir = (
            self.project_dir.expanduser().resolve()
            if self.project_dir is not None
            else None
        )
        project_path = (
            project_config_path(project_dir).resolve()
            if project_dir is not None
            else None
        )
        trust_path = home_path.parent / "mcp-trust.json"
        return home_path, project_path, trust_path

    def _acquire_locks(self, paths: tuple[Path, ...]) -> list[object]:
        handles: list[object] = []
        deadline = time.monotonic() + self._lock_timeout
        try:
            for path in sorted(set(paths), key=str):
                path.parent.mkdir(parents=True, exist_ok=True)
                lock_path = path.with_name(f".{path.name}.lock")
                handle = lock_path.open("a+")
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            handle.close()
                            raise MCPManagementError(
                                "mcp busy: configuration is being updated; retry"
                            ) from None
                        time.sleep(0.01)
                handles.append(handle)
            return handles
        except BaseException:
            self._release_locks(handles)
            raise

    @staticmethod
    def _release_locks(handles: list[object]) -> None:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            handle.close()  # type: ignore[attr-defined]

    @staticmethod
    def _servers(document: dict[str, object], path: Path) -> dict[str, object]:
        raw = document.get("servers", {})
        if type(raw) is not dict:
            raise MCPManagementError(f"MCP config servers must be an object: {path}")
        return raw

    def _trust_key(self, name: str, project_path: Path) -> str:
        identity = f"{project_path.parent.parent}\0{name}".encode()
        return hashlib.sha256(identity).hexdigest()

    def _is_trusted(
        self,
        name: str,
        raw: dict[str, object],
        scope: _ConfigScope,
        trust: dict[str, object],
        project_path: Path | None,
    ) -> bool:
        if scope != "project" or raw.get("transport") != "stdio":
            return True
        assert project_path is not None
        return trust.get(self._trust_key(name, project_path)) == self.trust_fingerprint(
            raw
        )

    def _managed(
        self,
        name: str,
        scope: _ConfigScope,
        raw: object,
        trust: dict[str, object],
        project_path: Path | None,
    ) -> ManagedServer:
        if type(raw) is not dict:
            return ManagedServer(
                name,
                scope,
                MappingProxyType(
                    {"malformed_reason": "server definition must be an object"}
                ),
                False,
                False,
                "malformed",
            )
        enabled = raw.get("enabled", True) is not False
        trusted = self._is_trusted(name, raw, scope, trust, project_path)
        auth = raw.get("auth")
        auth_type = str(auth.get("type", "none")) if isinstance(auth, dict) else "none"
        auth_status = (
            self._credential_status.get(name, "unauthorized")
            if auth_type == "oauth"
            else auth_type
        )
        status = (
            "disabled"
            if not enabled
            else "pending trust"
            if not trusted
            else "configured"
        )
        return ManagedServer(
            name,
            scope,
            _freeze(self.redact(raw)),  # type: ignore[arg-type]
            enabled,
            trusted,
            status,
            auth_status,
        )

    def _candidate_locked(
        self, home_path: Path, project_path: Path | None, trust_path: Path
    ) -> _ManagementCandidate:
        user_document = self._read_object(home_path)
        project_document = self._read_object(project_path) if project_path else {}
        trust = self._read_object(trust_path)
        user_entries = tuple(
            self._managed(name, "user", raw, trust, project_path)
            for name, raw in self._servers(user_document, home_path).items()
        )
        project_entries = (
            tuple(
                self._managed(name, "project", raw, trust, project_path)
                for name, raw in self._servers(project_document, project_path).items()
            )
            if project_path is not None
            else ()
        )
        config = load_mcp_config_overlay(
            home=home_path.parent,
            project_dir=project_path.parent.parent if project_path else None,
        )
        allowed: set[str] = set()
        sources: dict[str, Path] = {}
        project_servers = (
            self._servers(project_document, project_path) if project_path else {}
        )
        user_servers = self._servers(user_document, home_path)
        for name, server in config.configured_servers.items():
            source = config.sources.get(name, config.path).resolve()
            is_project = project_path is not None and source == project_path
            raw = (
                project_servers.get(name, {})
                if is_project
                else user_servers.get(name, {})
            )
            if type(raw) is not dict:
                continue
            if raw.get("enabled", server.enabled) is False:
                continue
            if is_project and not self._is_trusted(
                name, raw, "project", trust, project_path
            ):
                continue
            allowed.add(name)
            sources[name] = source
        runtime = replace(
            config,
            path=home_path,
            servers={
                name: value for name, value in config.servers.items() if name in allowed
            },
            skipped_servers={
                name: value
                for name, value in config.skipped_servers.items()
                if name in allowed
            },
            malformed_servers={
                name: value
                for name, value in config.malformed_servers.items()
                if name in allowed
            },
            sources=sources,
        )
        entries = tuple(
            replace(
                item,
                status=(
                    "skipped-missing-env"
                    if item.name in runtime.skipped_servers
                    else "malformed"
                    if item.name in runtime.malformed_servers
                    else item.status
                ),
            )
            for item in user_entries + project_entries
        )
        return _ManagementCandidate(
            entries,
            runtime,
            home_path,
            project_path,
        )

    def _transaction(
        self,
        mutation: Callable[[Path, Path | None, Path], None] | None = None,
    ) -> _ManagementCandidate:
        home_path, project_path, trust_path = self._canonical_paths()
        paths = tuple(
            path for path in (home_path, project_path, trust_path) if path is not None
        )
        handles = self._acquire_locks(paths)
        try:
            if mutation is not None:
                mutation(home_path, project_path, trust_path)
            return self._candidate_locked(home_path, project_path, trust_path)
        finally:
            self._release_locks(handles)

    def _with_runtime(
        self, entries: tuple[ManagedServer, ...]
    ) -> tuple[ManagedServer, ...]:
        statuses = getattr(self.mount, "statuses", {}) if self.mount is not None else {}
        actors = getattr(self.mount, "_actors", {}) if self.mount is not None else {}
        updated: list[ManagedServer] = []
        for item in entries:
            if item.status == "malformed":
                updated.append(item)
                continue
            status = statuses.get(item.name) if isinstance(statuses, dict) else None
            state = getattr(status, "state", None)
            actor = actors.get(item.name) if isinstance(actors, dict) else None
            tools = getattr(actor, "_tools", ()) if actor is not None else ()
            runtime_status = str(state) if state else item.status
            if not item.enabled:
                runtime_status = "disabled"
            elif not item.trusted:
                runtime_status = "pending trust"
            updated.append(replace(item, status=runtime_status, tool_count=len(tools)))
        return tuple(updated)

    def _publish(self, candidate: _ManagementCandidate) -> MCPManagerSnapshot:
        assert self._transaction_gate.locked(), "publication requires transaction gate"
        self._home_path = candidate.home_path
        self._project_path = candidate.project_path
        self._paths_ready = True
        self._runtime = candidate.runtime_config
        self._snapshot = MCPManagerSnapshot(
            self._with_runtime(candidate.entries), self._snapshot.generation + 1
        )
        return self._snapshot

    async def _transact_reconcile_publish(
        self,
        mutation: Callable[[Path, Path | None, Path], None] | None = None,
    ) -> MCPManagerSnapshot:
        async with self._transaction_gate:
            candidate = await asyncio.to_thread(self._transaction, mutation)
            await self._reconcile(candidate.runtime_config)
            return self._publish(candidate)

    async def refresh(self) -> MCPManagerSnapshot:
        return await self._transact_reconcile_publish()

    def _config_mutation(
        self,
        scope: _ConfigScope,
        change: Callable[[dict[str, object]], None],
    ) -> Callable[[Path, Path | None, Path], None]:
        def mutate(
            home_path: Path, project_path: Path | None, _trust_path: Path
        ) -> None:
            path = project_path if scope == "project" else home_path
            if path is None:
                raise MCPManagementError("project scope requires a repository")
            document = self._read_object(path)
            servers = copy.deepcopy(self._servers(document, path))
            change(servers)
            document["servers"] = servers
            self._atomic_json(path, document)

        return mutate

    async def _commit(
        self, mutation: Callable[[Path, Path | None, Path], None]
    ) -> MCPManagerSnapshot:
        return await self._transact_reconcile_publish(mutation)

    async def add(
        self,
        name: str,
        *,
        scope: _ConfigScope,
        command: str | None = None,
        args: tuple[str, ...] = (),
        url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        oauth: bool = False,
        enabled: bool = True,
    ) -> MCPManagerSnapshot:
        if not name:
            raise MCPManagementError("server name must not be empty")
        if (command is None) == (url is None):
            raise MCPManagementError("specify exactly one of command or url")
        raw: dict[str, object] = {
            "transport": "stdio" if command is not None else "streamable-http",
        }
        if not enabled:
            raw["enabled"] = False
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

        def change(servers: dict[str, object]) -> None:
            if name in servers:
                raise MCPManagementError(f"MCP server already configured: {name}")
            servers[name] = raw

        return await self._commit(self._config_mutation(scope, change))

    async def remove(self, name: str, *, scope: _ConfigScope) -> MCPManagerSnapshot:
        def change(servers: dict[str, object]) -> None:
            if name not in servers:
                raise MCPManagementError(f"unknown MCP server: {name}")
            del servers[name]

        return await self._commit(self._config_mutation(scope, change))

    async def set_enabled(
        self, name: str, *, scope: _ConfigScope, enabled: bool
    ) -> MCPManagerSnapshot:
        def change(servers: dict[str, object]) -> None:
            raw = servers.get(name)
            if type(raw) is not dict:
                raise MCPManagementError(f"unknown MCP server: {name}")
            raw["enabled"] = enabled

        return await self._commit(self._config_mutation(scope, change))

    async def set_trusted(self, name: str, *, trusted: bool) -> MCPManagerSnapshot:
        def mutate(_home: Path, project_path: Path | None, trust_path: Path) -> None:
            if project_path is None:
                raise MCPManagementError("project scope requires a repository")
            project = self._servers(self._read_object(project_path), project_path)
            raw = project.get(name)
            if type(raw) is not dict:
                raise MCPManagementError(f"unknown project MCP server: {name}")
            data = self._read_object(trust_path)
            key = self._trust_key(name, project_path)
            if trusted:
                data[key] = self.trust_fingerprint(raw)
            else:
                data.pop(key, None)
            self._atomic_json(trust_path, data, mode=0o600)

        return await self._commit(mutate)

    def show(self, name: str, *, scope: Scope = "effective") -> ManagedServer:
        item = next(
            (
                entry
                for entry in reversed(self._snapshot.entries)
                if entry.name == name and (scope == "effective" or entry.scope == scope)
            ),
            None,
        )
        if item is None:
            raise MCPManagementError(f"unknown MCP server: {name}")
        return item

    async def refresh_credentials(self) -> tuple[ManagedServer, ...]:
        await self.refresh()
        self.credential_store_error = None
        for item in self._snapshot.entries:
            auth = item.config.get("auth")
            if not isinstance(auth, Mapping) or auth.get("type") != "oauth":
                continue
            try:
                credential = await self._credential_store.load(item.name)
            except MCPCredentialStoreError as exc:
                self.credential_store_error = public_credential_store_error(exc)
                raise
            self._credential_status[item.name] = token_state(credential.token)
        return (await self._transact_reconcile_publish()).entries

    async def runtime_config(self) -> MCPConfig:
        if self._runtime is None:
            await self.refresh()
        assert self._runtime is not None
        return self._runtime

    async def _reconcile(self, desired: MCPConfig) -> None:
        assert self._transaction_gate.locked(), "reconciliation requires transaction gate"
        if self.mount is None or getattr(self.mount, "pinned", False):
            return
        configured = desired.configured_servers
        current = dict(getattr(self.mount, "configs", {}))
        for name in tuple(current):
            if name not in configured or current[name] != configured[name]:
                await self.mount.remove_server(name)
        current_names = set(getattr(self.mount, "configs", {}))
        for name, config in configured.items():
            if name not in current_names:
                await self.mount.add_server(
                    config, source=desired.sources.get(name, desired.path)
                )

    async def sync_runtime(self) -> MCPManagerSnapshot:
        return await self._transact_reconcile_publish()

    def _policy_registry(self) -> ToolRegistry | None:
        return getattr(self.mount, "registry", None)

    def _server_allowed(self, name: str) -> bool:
        return mcp_connection.mcp_server_allowed(self._policy_registry(), name)

    async def test(self, name: str, *, scope: Scope = "effective") -> dict[str, object]:
        if self._snapshot.generation == 0:
            await self.refresh()
        item = self.show(name, scope=scope)
        if not self._server_allowed(name):
            return {
                "name": name,
                "tools": [],
                "status": "skipped-policy",
                "detail": "skipped by tool policy",
            }
        if not item.trusted:
            return {
                "name": name,
                "tools": [],
                "status": "untrusted",
                "detail": f"trust the server definition first (zeta mcp trust {name})",
            }
        config = (
            await asyncio.to_thread(load_mcp_config, self.path(item.scope))
        ).configured_servers.get(name)
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
            for tool in sorted(tools, key=lambda value: value.name):
                description = tool.description
                if len(description) > 120:
                    description = description[:117] + "..."
                summaries.append({"name": tool.name, "description": description})
            return {"name": name, "tools": summaries, "status": "ok"}
        finally:
            await client.close()

    async def login(
        self, name: str, *, scope: Scope = "effective", no_browser: bool = False
    ) -> None:
        from .oauth import authorize

        if self._snapshot.generation == 0:
            await self.refresh()
        item = self.show(name, scope=scope)
        if not self._server_allowed(name):
            raise MCPManagementError(f"{name}: skipped by tool policy")
        if item.config.get("transport") != "streamable-http":
            raise MCPManagementError(
                "OAuth requires an HTTP MCP server "
                f"(got transport {item.config.get('transport')})"
            )
        config = (
            await asyncio.to_thread(load_mcp_config, self.path(item.scope))
        ).configured_servers.get(name)
        if config is None:
            raise MCPManagementError(f"server cannot be used for OAuth: {name}")
        await self._credential_store.load(name)
        candidate = await authorize(
            server_name=name,
            server_url=config.url or "",
            client_id=config.client_id,
            client_secret=config.client_secret,
            callback_port=config.callback_port,
            scopes=config.scopes,
            authorization_server_url=config.authorization_server_url,
            resource_metadata_url=config.resource_metadata_url,
            authorization_params=config.authorization_params,
            no_browser=no_browser,
        )
        credential = await self._credential_store.save_login(name, candidate)
        self._credential_status[name] = token_state(credential.token)
        self.credential_store_error = None
        await self.refresh()

    async def logout(self, name: str, *, scope: Scope = "effective") -> None:
        if self._snapshot.generation == 0:
            await self.refresh()
        self.show(name, scope=scope)
        await self._credential_store.delete(name)
        self._credential_status[name] = "unauthorized"
        self.credential_store_error = None
        await self.refresh()


__all__ = [
    "MCPManagementError",
    "MCPManagementService",
    "MCPManagerSnapshot",
    "ManagedServer",
    "Scope",
]
