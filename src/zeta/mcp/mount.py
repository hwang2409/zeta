"""Discover MCP tools and mount them into zeta's tool registry."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from ..core.abort import AbortSignal
from ..tools.registry import ToolRegistry
from ..types import StructuredToolResult
from .client import MCPClient, MCPError, MCPTool
from .config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    load_mcp_config,
    mcp_log_path,
)
from .http import StreamableHTTPMCPClient
from .stdio import StdioMCPClient

logger = logging.getLogger(__name__)
# One quiet server cannot hold session startup longer than this bound.
SERVER_SETUP_TIMEOUT_SECONDS = 10.0
MCPServerState = Literal["mounted", "failed", "skipped-missing-env", "timed-out"]


@dataclass(frozen=True, slots=True)
class MCPServerStatus:
    """Current connection state for one configured MCP server."""

    name: str
    transport: str
    state: MCPServerState
    tool_count: int = 0
    reason: str | None = None
    stderr_log_path: str = ""


NoticeSink = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class _SetupResult:
    public: MCPServerStatus
    client: MCPClient | None = None
    tools: tuple[MCPTool, ...] = ()


@dataclass(slots=True, init=False)
class MCPMount:
    """Connected MCP clients owned by one zeta session."""

    registry: ToolRegistry | None
    configs: dict[str, MCPServerConfig]
    statuses: dict[str, MCPServerStatus]
    _clients: dict[str, MCPClient] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _schema_refresh: Callable[[MCPMount], None] | None = None
    _closed: bool = False

    def __init__(
        self,
        registry: ToolRegistry | tuple[MCPClient, ...],
        configs: dict[str, MCPServerConfig] | None = None,
        statuses: dict[str, MCPServerStatus] | None = None,
        _clients: dict[str, MCPClient] | None = None,
        _locks: dict[str, asyncio.Lock] | None = None,
        _schema_refresh: Callable[[MCPMount], None] | None = None,
        _closed: bool = False,
    ) -> None:
        if isinstance(registry, ToolRegistry):
            clients = _clients or {}
            server_configs = configs or {}
        else:
            clients = {client.config.name: client for client in registry}
            server_configs = {
                client.config.name: client.config for client in registry
            }
            registry = None
        self.registry = registry
        self.configs = server_configs
        self.statuses = statuses or {}
        self._clients = clients
        self._locks = _locks or {
            name: asyncio.Lock() for name in server_configs
        }
        self._schema_refresh = _schema_refresh
        self._closed = _closed

    @property
    def clients(self) -> tuple[MCPClient, ...]:
        """Return connected clients in configured order."""

        return tuple(self._clients.values())

    @property
    def summary(self) -> str:
        mounted = sum(status.state == "mounted" for status in self.statuses.values())
        failed = sum(
            status.state in {"failed", "timed-out"}
            for status in self.statuses.values()
        )
        return f"mcp: {mounted} mounted, {failed} failed"

    def render(self) -> str:
        lines = [self.summary]
        for status in self.statuses.values():
            line = (
                f"{status.name}: {status.state} | transport: {status.transport} | "
                f"tools: {status.tool_count} | stderr: {status.stderr_log_path}"
            )
            if status.reason:
                line += f" | reason: {status.reason}"
            lines.append(line)
        return "\n".join(lines)

    async def reconnect(
        self,
        name: str,
        *,
        notice_sink: NoticeSink | None = None,
    ) -> MCPServerStatus:
        """Replace one server without touching other server clients."""

        if name not in self.configs:
            raise ValueError(f"unknown MCP server: {name}")
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if self._closed:
                raise ValueError("MCP mount is closed")
            client = self._clients.get(name)
            self._transition(name, client=client)
            if client is not None:
                await _close_failed_client(client)
            setup = await _setup_server(
                self.configs[name],
                notice_sink=notice_sink,
                client_ready=lambda replacement: self._arm_client(
                    name, replacement
                ),
            )
            self._finish_setup(setup)
            return setup.public

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in tuple(set(self.configs) | set(self._clients)):
            lock = self._locks.setdefault(name, asyncio.Lock())
            async with lock:
                client = self._clients.get(name)
                self._transition(name, client=client, allow_closed=True)
                if client is not None:
                    await _close_failed_client(client)

    def set_schema_refresh(self, callback: Callable[[MCPMount], None]) -> None:
        """Bind the owner of provider tool schemas after initial setup."""

        self._schema_refresh = callback
        callback(self)

    def _arm_client(self, name: str, client: MCPClient) -> None:
        self._clients[name] = client
        self._attach_failure_handler(name, client)

    def _finish_setup(self, setup: _SetupResult) -> None:
        client = setup.client
        accepted = self._transition(
            setup.public.name,
            client=client,
            state=setup.public.state,
            reason=setup.public.reason,
            tools=setup.tools,
        )
        if not accepted and client is not None:
            asyncio.create_task(_close_failed_client(client))

    def _transition(
        self,
        name: str,
        *,
        client: MCPClient | None,
        state: MCPServerState | None = None,
        reason: str | None = None,
        tools: tuple[MCPTool, ...] = (),
        allow_closed: bool = False,
    ) -> bool:
        """Apply one server state and keep registry and schemas in sync."""

        if self._closed and not allow_closed:
            return False
        current = self._clients.get(name)
        if client is not None and current is not None and current is not client:
            return False
        if state == "mounted":
            if client is None or current is not client:
                return False
            self.statuses[name] = replace(
                self.statuses.get(name)
                or MCPServerStatus(
                    name,
                    self.configs[name].transport,
                    "mounted",
                ),
                state="mounted",
                tool_count=len(tools),
                reason=None,
            )
            for tool in tools:
                _register_tool(self.registry, client, tool, mount=self)
        else:
            self._unregister_tools(name)
            if current is client:
                self._clients.pop(name, None)
            if state is not None:
                previous = self.statuses.get(name) or MCPServerStatus(
                    name,
                    self.configs[name].transport,
                    state,
                )
                self.statuses[name] = replace(
                    previous,
                    state=state,
                    tool_count=0,
                    reason=reason,
                )
        if self._schema_refresh is not None:
            self._schema_refresh(self)
        return True

    def _unregister_tools(self, server_name: str) -> None:
        prefix = f"{server_name}:"
        if self.registry is None:
            return
        for name in tuple(self.registry.definitions_by_name):
            if name.startswith(prefix):
                self.registry.unregister(name)

    def _attach_failure_handler(self, name: str, client: MCPClient) -> None:
        set_failure_sink = getattr(client, "set_failure_sink", None)
        if set_failure_sink is not None:
            set_failure_sink(lambda reason: self._mark_failed(name, client, reason))

    def _mark_failed(self, name: str, client: MCPClient, reason: str) -> None:
        if self._transition(name, client=client, state="failed", reason=reason):
            asyncio.create_task(_close_failed_client(client))


async def mount_mcp_servers(
    registry: ToolRegistry,
    config: MCPConfig | None = None,
    *,
    notice_sink: NoticeSink | None = None,
) -> MCPMount:
    """Connect configured servers and register each discovered tool."""

    if config is None:
        try:
            config = load_mcp_config()
        except MCPConfigError as exc:
            logger.error("%s", exc)
            return MCPMount(registry, {}, {})

    configs = config.configured_servers
    mount = MCPMount(
        registry,
        configs,
        {},
        _locks={name: asyncio.Lock() for name in configs},
    )
    try:
        results = await asyncio.gather(
            *(
                _setup_server(
                    server_config,
                    notice_sink=notice_sink,
                    client_ready=lambda client, name=server_config.name: mount._arm_client(
                        name, client
                    ),
                )
                for server_config in configs.values()
            )
        )
    except BaseException:
        await mount.close()
        raise
    for setup in results:
        mount._finish_setup(setup)
    return mount


async def _setup_server(
    server_config: MCPServerConfig,
    *,
    notice_sink: NoticeSink | None = None,
    client_ready: Callable[[MCPClient], None] | None = None,
) -> _SetupResult:
    if server_config.missing_env:
        variables = ", ".join(server_config.missing_env)
        reason = f"missing environment variable(s): {variables}"
        public = MCPServerStatus(
            server_config.name,
            server_config.transport,
            "skipped-missing-env",
            reason=reason,
            stderr_log_path=str(mcp_log_path(server_config.name)),
        )
        _notice(
            notice_sink,
            f"mcp · {server_config.name} skipped-missing-env ({variables})",
        )
        return _SetupResult(public)

    client: MCPClient | None = None
    try:
        client = _build_client(server_config)
        if client_ready is not None:
            client_ready(client)
        tools = await asyncio.wait_for(
            _connect_and_list(client), timeout=SERVER_SETUP_TIMEOUT_SECONDS
        )
    except TimeoutError:
        reason = f"after {SERVER_SETUP_TIMEOUT_SECONDS:.1f}s"
        logger.warning(
            "timed out connecting to MCP server %s %s", server_config.name, reason
        )
        if client is not None:
            await _close_failed_client(client)
        public = MCPServerStatus(
            server_config.name,
            server_config.transport,
            "timed-out",
            reason=reason,
            stderr_log_path=str(mcp_log_path(server_config.name)),
        )
        _notice(notice_sink, f"mcp · {server_config.name} timed-out ({reason})")
        return _SetupResult(public, client)
    except Exception as exc:  # noqa: BLE001 - isolate one bad server
        reason = _error_text(exc)
        logger.warning("failed to mount MCP server %s: %s", server_config.name, reason)
        if client is not None:
            await _close_failed_client(client)
        public = MCPServerStatus(
            server_config.name,
            server_config.transport,
            "failed",
            reason=reason,
            stderr_log_path=str(mcp_log_path(server_config.name)),
        )
        _notice(notice_sink, f"mcp · {server_config.name} failed: {reason}")
        return _SetupResult(public, client)
    except BaseException:
        if client is not None:
            await _close_failed_client(client)
        raise
    public = MCPServerStatus(
        server_config.name,
        server_config.transport,
        "mounted",
        tool_count=len(tools),
        stderr_log_path=str(mcp_log_path(server_config.name)),
    )
    _notice(notice_sink, f"mcp · {server_config.name} mounted ({len(tools)} tools)")
    return _SetupResult(public, client, tuple(tools))


def _notice(sink: NoticeSink | None, message: str) -> None:
    if sink is not None:
        sink(message)


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask the failure
        return type(error).__name__


def _build_client(config: MCPServerConfig) -> MCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return StreamableHTTPMCPClient(config)


async def _connect_and_list(client: MCPClient) -> list[MCPTool]:
    await client.connect()
    return await client.list_tools()


async def _close_failed_client(client: MCPClient) -> None:
    try:
        await client.close()
    except Exception:
        logger.exception("failed to close MCP server %s", client.config.name)


def _register_tool(
    registry: ToolRegistry,
    client: MCPClient,
    tool: MCPTool,
    *,
    mount: MCPMount,
) -> None:
    name = f"{client.config.name}:{tool.name}"

    async def handler(
        arguments: dict[str, object], abort_signal: AbortSignal
    ) -> StructuredToolResult:
        try:
            result = await client.call_tool(tool.name, arguments, abort_signal)
        except MCPError as exc:
            mount._mark_failed(client.config.name, client, _error_text(exc))
            raise
        return result

    try:
        registry.register(
            name,
            handler,
            description=tool.description,
            parameters=tool.input_schema,
            validate_arguments=False,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("skipping MCP tool %s: invalid input schema: %s", name, exc)


__all__ = ["MCPMount", "MCPServerState", "MCPServerStatus", "mount_mcp_servers"]
