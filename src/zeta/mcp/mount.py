"""Discover MCP tools and mount them into zeta's tool registry."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from ..core.abort import AbortSignal
from ..tools.registry import ToolRegistry
from ..types import StructuredToolResult
from .client import MCPClient, MCPError, MCPTool, make_error_result
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
AUTO_RECONNECT_BASE_DELAY_SECONDS = 1.0
AUTO_RECONNECT_MAX_DELAY_SECONDS = 30.0
MCPServerState = Literal[
    "mounted",
    "degraded",
    "failed",
    "skipped-missing-env",
    "timed-out",
    "malformed",
]


@dataclass(frozen=True, slots=True)
class MCPServerStatus:
    """Current connection state for one configured MCP server."""

    name: str
    transport: str
    state: MCPServerState
    tool_count: int = 0
    reason: str | None = None
    stderr_log_path: str = ""
    next_retry_at: float | None = None


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
    sources: dict[str, Path]
    _clients: dict[str, MCPClient] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    _auto_remount_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _failure_tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict)
    _failure_counts: dict[str, int] = field(default_factory=dict)
    _generations: dict[str, int] = field(default_factory=dict)
    _config_lock: asyncio.Lock = field(init=False)
    _schema_refresh: Callable[[MCPMount], None] | None = None
    _closed: bool = False

    def __init__(
        self,
        registry: ToolRegistry | tuple[MCPClient, ...],
        configs: dict[str, MCPServerConfig] | None = None,
        statuses: dict[str, MCPServerStatus] | None = None,
        _clients: dict[str, MCPClient] | None = None,
        _locks: dict[str, asyncio.Lock] | None = None,
        _auto_remount_tasks: dict[str, asyncio.Task[None]] | None = None,
        _failure_tasks: dict[int, asyncio.Task[None]] | None = None,
        _failure_counts: dict[str, int] | None = None,
        _generations: dict[str, int] | None = None,
        _schema_refresh: Callable[[MCPMount], None] | None = None,
        _closed: bool = False,
        sources: dict[str, Path] | None = None,
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
        self.sources = sources or {}
        self._clients = clients
        self._locks = _locks or {
            name: asyncio.Lock() for name in server_configs
        }
        self._auto_remount_tasks = _auto_remount_tasks or {}
        self._failure_tasks = _failure_tasks or {}
        self._failure_counts = _failure_counts or {}
        self._generations = _generations or {
            name: 1 for name in server_configs
        }
        self._config_lock = asyncio.Lock()
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
            status.state in {"degraded", "failed", "timed-out", "malformed"}
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
            if status.state == "degraded":
                line += f" | next retry: {_retry_text(status.next_retry_at)}"
            lines.append(line)
        return "\n".join(lines)

    async def reconnect(
        self,
        name: str,
        *,
        notice_sink: NoticeSink | None = None,
    ) -> MCPServerStatus:
        """Replace one server without touching other server clients.

        Manual reconnects start at once and bypass automatic backoff.
        """

        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            server_config = self.configs.get(name)
            if server_config is None:
                raise ValueError(f"unknown MCP server: {name}")
        return await self._mount_one(name, server_config, notice_sink=notice_sink)

    async def add_server(
        self,
        server_config: MCPServerConfig,
        *,
        source: Path,
        notice_sink: NoticeSink | None = None,
        prepare: Callable[[], None] | None = None,
        rollback: Callable[[], None] | None = None,
    ) -> MCPServerStatus:
        """Mount one new server live, without touching existing clients."""

        name = server_config.name
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with self._config_lock, lock:
            if self._closed:
                raise ValueError("MCP mount is closed")
            if name in self.configs:
                raise ValueError(f"MCP server already configured: {name}")
            if prepare is not None:
                prepare()
            generation = self._generations.get(name, 0) + 1
            self._transition(
                name,
                client=None,
                config=server_config,
                source=source,
                new_generation=generation,
            )
        try:
            return await self._mount_one(name, server_config, notice_sink=notice_sink)
        except BaseException:
            client = None
            accepted = False
            async with self._config_lock, lock:
                client = self._clients.get(name)
                accepted = self._transition(
                    name,
                    client=client,
                    generation=generation,
                    allow_closed=True,
                    remove_config=True,
                    remove_source=True,
                )
                if accepted and rollback is not None:
                    try:
                        rollback()
                    except BaseException:
                        logger.exception("failed to roll back MCP config %s", name)
            if accepted and client is not None:
                await _close_failed_client(
                    client, mount=self, generation=generation
                )
            raise

    async def replace_server(
        self,
        name: str,
        *,
        persist: Callable[[Path], None],
        replacement: Callable[[Path], tuple[MCPServerConfig, Path] | None],
        notice_sink: NoticeSink | None = None,
    ) -> None:
        """Replace one server without holding its lifecycle lock during I/O."""

        lock = self._locks.setdefault(name, asyncio.Lock())
        async with self._config_lock, lock:
            if name not in self.configs:
                raise ValueError(f"unknown MCP server: {name}")
            source = self.sources.get(name)
            if source is None:
                raise ValueError(f"unknown MCP server: {name}")
            next_server = replacement(source)
            persist(source)
            client = self._clients.get(name)
            generation = self._generations.get(name)
            self._transition(
                name,
                client=client,
                generation=generation,
                allow_closed=True,
                remove_config=True,
                remove_source=True,
            )
            if next_server is None:
                server_config = None
                replacement_generation = None
            else:
                server_config, source = next_server
                next_generation = self._generations.get(name, 0) + 1
                replacement_generation = next_generation
                self._transition(
                    name,
                    client=None,
                    config=server_config,
                    source=source,
                    new_generation=next_generation,
                )
        if client is not None:
            await _close_failed_client(client, mount=self, generation=generation)
        if server_config is None:
            return
        try:
            await self._mount_one(name, server_config, notice_sink=notice_sink)
        except BaseException as exc:
            async with lock:
                self._transition(
                    name,
                    client=self._clients.get(name),
                    generation=replacement_generation,
                    state="failed",
                    reason=_error_text(exc),
                )
            raise

    async def remove_server(self, name: str) -> None:
        """Unmount one server live and drop its bookkeeping."""

        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name not in self.configs:
                raise ValueError(f"unknown MCP server: {name}")
            client = self._clients.get(name)
            generation = self._generations[name]
            await self._remove_server_locked(name)
        if client is not None:
            await _close_failed_client(client, mount=self, generation=generation)

    async def _remove_server_locked(self, name: str) -> None:
        """Remove one server while its lifecycle lock is held."""

        client = self._clients.get(name)
        generation = self._generations[name]
        self._transition(
            name,
            client=client,
            generation=generation,
            allow_closed=True,
            remove_config=True,
            remove_source=True,
        )
    async def _mount_one(
        self,
        name: str,
        server_config: MCPServerConfig,
        *,
        notice_sink: NoticeSink | None,
    ) -> MCPServerStatus:
        """Mount one server without holding its lifecycle lock during I/O."""

        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if self._closed:
                raise ValueError("MCP mount is closed")
            if self.configs.get(name) is not server_config:
                raise ValueError(f"unknown MCP server: {name}")
            generation = self._generations[name]
            previous = self.statuses.get(name)
            preserve_degraded = previous is not None and previous.state == "degraded"
            client = self._clients.get(name)
            if client is not None:
                self._transition(
                    name,
                    client=client,
                    generation=generation,
                    state="failed",
                    reason="reconnecting",
                )
        try:
            if client is not None:
                await _close_failed_client(
                    client, mount=self, generation=generation
                )
            setup = await _setup_server(
                server_config,
                notice_sink=notice_sink,
                mount=self,
                generation=generation,
                client_ready=lambda replacement: self._arm_client(
                    name, replacement, generation
                ),
            )
        except BaseException as exc:
            async with lock:
                current_status = self.statuses.get(name)
                if isinstance(exc, asyncio.CancelledError):
                    if preserve_degraded and current_status is not None:
                        self._transition(
                            name,
                            client=self._clients.get(name),
                            generation=generation,
                            state="degraded",
                            reason=current_status.reason,
                            next_retry_at=current_status.next_retry_at,
                            failure_count=self._failure_counts.get(name, 0),
                        )
                    else:
                        self._transition(
                            name,
                            client=self._clients.get(name),
                            generation=generation,
                            state="failed",
                            reason=_error_text(exc),
                        )
                    raise
                if (
                    preserve_degraded
                    and current_status is not None
                    and current_status.state == "degraded"
                ):
                    return current_status
                self._transition(
                    name,
                    client=self._clients.get(name),
                    generation=generation,
                    state="failed",
                    reason=_error_text(exc),
                )
            raise
        stale_client: MCPClient | None = None
        async with lock:
            current_status = self.statuses.get(name)
            if (
                self._closed
                or self.configs.get(name) is not server_config
                or self._generations.get(name) != generation
            ):
                stale_client = setup.client
            elif (
                preserve_degraded
                and current_status is not None
                and current_status.state == "degraded"
                and setup.public.state != "mounted"
            ):
                self._transition(
                    name,
                    client=setup.client,
                    generation=generation,
                    state="degraded",
                    reason=current_status.reason,
                    next_retry_at=current_status.next_retry_at,
                    failure_count=self._failure_counts.get(name, 0),
                )
                return current_status
            else:
                self._finish_setup(setup, generation)
                return self.statuses[name]
        if stale_client is not None:
            await _close_failed_client(
                stale_client, mount=self, generation=generation
            )
        return self.statuses.get(name, setup.public)

    async def close(self) -> None:
        async with self._config_lock:
            if self._closed:
                return
            self._transition("", client=None, mark_closed=True)
            auto_tasks = tuple(self._auto_remount_tasks.values())
        for task in auto_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*auto_tasks, return_exceptions=True)
        for name in tuple(set(self.configs) | set(self._clients)):
            lock = self._locks.setdefault(name, asyncio.Lock())
            async with lock:
                client = self._clients.get(name)
                generation = self._generations.get(name)
                self._transition(
                    name,
                    client=client,
                    state="failed",
                    reason="MCP mount is closed",
                    allow_closed=True,
                )
            if client is not None:
                await _close_failed_client(
                    client, mount=self, generation=generation
                )

    def set_schema_refresh(self, callback: Callable[[MCPMount], None]) -> None:
        """Bind the owner of provider tool schemas after initial setup."""

        self._schema_refresh = callback
        callback(self)

    def _arm_client(self, name: str, client: MCPClient, generation: int) -> None:
        self._transition(name, client=client, generation=generation)

    def _finish_setup(self, setup: _SetupResult, generation: int) -> None:
        client = setup.client
        accepted = self._transition(
            setup.public.name,
            client=client,
            state=setup.public.state,
            reason=setup.public.reason,
            tools=setup.tools,
            generation=generation,
        )
        if not accepted and client is not None:
            asyncio.create_task(
                _close_failed_client(
                    client, mount=self, generation=generation
                )
            )

    def _transition(
        self,
        name: str,
        *,
        client: MCPClient | None,
        config: MCPServerConfig | None = None,
        source: Path | None = None,
        state: MCPServerState | None = None,
        reason: str | None = None,
        tools: tuple[MCPTool, ...] = (),
        next_retry_at: float | None = None,
        failure_count: int = 0,
        generation: int | None = None,
        new_generation: int | None = None,
        auto_task: asyncio.Task[None] | None = None,
        clear_auto_task: asyncio.Task[None] | None = None,
        failure_task: asyncio.Task[None] | None = None,
        clear_failure_task: asyncio.Task[None] | None = None,
        failure_client: MCPClient | None = None,
        clear_failure_client: MCPClient | None = None,
        mark_closed: bool = False,
        allow_closed: bool = False,
        remove_config: bool = False,
        remove_source: bool = False,
    ) -> bool:
        """Apply one server state and keep registry and schemas in sync."""

        if mark_closed:
            self._closed = True
            return True
        if self._closed and not allow_closed:
            return False
        if generation is not None and self._generations.get(name) != generation:
            return False
        if new_generation is not None:
            self._generations[name] = new_generation
        task_only = any(
            task is not None
            for task in (
                auto_task,
                clear_auto_task,
                failure_task,
                clear_failure_task,
                failure_client,
                clear_failure_client,
            )
        )
        if auto_task is not None:
            self._auto_remount_tasks[name] = auto_task
        if (
            clear_auto_task is not None
            and self._auto_remount_tasks.get(name) is clear_auto_task
        ):
            self._auto_remount_tasks.pop(name, None)
        if failure_task is not None and failure_client is not None:
            self._failure_tasks[id(failure_client)] = failure_task
        if (
            clear_failure_task is not None
            and clear_failure_client is not None
            and self._failure_tasks.get(id(clear_failure_client)) is clear_failure_task
        ):
            self._failure_tasks.pop(id(clear_failure_client), None)
        if task_only:
            return True
        current = self._clients.get(name)
        if client is not None and current is not None and current is not client:
            return False
        if config is not None:
            self.configs[name] = config
        if source is not None:
            self.sources[name] = source
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
                next_retry_at=None,
            )
            self._failure_counts.pop(name, None)
            self._unregister_tools(name)
            for tool in tools:
                _register_tool(self.registry, client, tool, mount=self)
        elif state == "degraded":
            if client is not None and current is not client:
                return False
            previous = self.statuses.get(name) or MCPServerStatus(
                name,
                self.configs[name].transport,
                "degraded",
            )
            self.statuses[name] = replace(
                previous,
                state="degraded",
                reason=reason,
                next_retry_at=next_retry_at,
            )
            if current is not None and current is client:
                self._clients.pop(name, None)
            self._failure_counts[name] = failure_count
        elif state is None and client is not None and current is None:
            self._clients[name] = client
            self._attach_failure_handler(name, client)
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
                    next_retry_at=next_retry_at,
                )
                if state != "degraded":
                    self._failure_counts.pop(name, None)
        if remove_config:
            self._generations[name] = self._generations.get(name, 0) + 1
            self.configs.pop(name, None)
            self.statuses.pop(name, None)
        if remove_source:
            self.sources.pop(name, None)
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
            generation = self._generations[name]
            set_failure_sink(
                lambda reason: self._mark_failed(
                    name, client, generation, reason
                )
            )

    def _mark_failed(
        self, name: str, client: MCPClient, generation: int, reason: str
    ) -> None:
        asyncio.create_task(self._degrade_client(name, client, generation, reason))

    async def _degrade_client(
        self, name: str, client: MCPClient, generation: int, reason: str
    ) -> None:
        lock = self._locks.setdefault(name, asyncio.Lock())
        task = asyncio.current_task()
        async with lock:
            current = self._clients.get(name)
            if (
                current is not client
                or name not in self.configs
                or self._generations.get(name) != generation
            ):
                return
            self._transition(
                name,
                client=None,
                failure_task=task,
                failure_client=client,
                generation=generation,
            )
            self._transition(
                name,
                client=client,
                generation=generation,
                state="degraded",
                reason=reason,
                next_retry_at=time.monotonic(),
                failure_count=self._failure_counts.get(name, 0),
            )
        await _close_failed_client(client, mount=self, generation=generation)
        if task is not None:
            async with lock:
                self._transition(
                    name,
                    client=None,
                    generation=generation,
                    clear_failure_task=task,
                    clear_failure_client=client,
                    allow_closed=True,
                )

    async def _wait_for_failure(self, client: MCPClient, generation: int) -> None:
        lock = self._locks.setdefault(client.config.name, asyncio.Lock())
        async with lock:
            if self._generations.get(client.config.name) != generation:
                return
            task = self._failure_tasks.get(id(client))
        if task is not None:
            await asyncio.shield(task)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
        abort_signal: AbortSignal,
        *,
        generation: int | None = None,
    ) -> StructuredToolResult:
        """Call one tool while coordinating degradation and remounts."""

        lock = self._locks.setdefault(server_name, asyncio.Lock())
        auto_task: asyncio.Task[None] | None = None
        started_auto_remount = False
        async with self._config_lock, lock:
            current_generation = self._generations.get(server_name)
            if self._closed or (
                generation is not None and generation != current_generation
            ):
                return make_error_result(
                    f"MCP server '{server_name}' is unavailable. "
                    f"Use /mcp reconnect {server_name}."
                )
            generation = current_generation
            status = self.statuses.get(server_name)
            client = self._clients.get(server_name)
            if status is not None and status.state == "mounted" and client is not None:
                pass
            elif status is None or status.state != "degraded":
                return make_error_result(
                    f"MCP server '{server_name}' is unavailable. "
                    f"Use /mcp reconnect {server_name}."
                )
            else:
                auto_task = self._auto_remount_tasks.get(server_name)
                if auto_task is None or auto_task.done():
                    if auto_task is not None:
                        self._transition(
                            server_name,
                            client=None,
                            generation=generation,
                            clear_auto_task=auto_task,
                        )
                    retry_at = status.next_retry_at
                    if retry_at is not None and time.monotonic() < retry_at:
                        return _degraded_result(server_name, status)
                    auto_task = asyncio.create_task(
                        self._auto_remount(server_name, generation)
                    )
                    accepted = self._transition(
                        server_name,
                        client=None,
                        generation=generation,
                        auto_task=auto_task,
                    )
                    if not accepted:
                        auto_task.cancel()
                        return _degraded_result(server_name, status)
                    started_auto_remount = True
                else:
                    return _degraded_result(server_name, status)
        if client is not None and status is not None and status.state == "mounted":
            return await self._call_mounted_tool(
                server_name,
                client,
                tool_name,
                arguments,
                abort_signal,
                generation,
            )
        if not started_auto_remount or auto_task is None:
            return _degraded_result(server_name, status)
        await asyncio.shield(auto_task)
        return await self._call_mounted_tool(
            server_name,
            None,
            tool_name,
            arguments,
            abort_signal,
            generation,
        )

    async def _call_mounted_tool(
        self,
        server_name: str,
        expected_client: MCPClient | None,
        tool_name: str,
        arguments: dict[str, object],
        abort_signal: AbortSignal,
        generation: int,
    ) -> StructuredToolResult:
        close_client: MCPClient | None = None
        lock = self._locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            status = self.statuses.get(server_name)
            client = self._clients.get(server_name)
            if (
                self._closed
                or self._generations.get(server_name) != generation
                or status is None
                or status.state != "mounted"
                or client is None
                or (expected_client is not None and client is not expected_client)
            ):
                return _degraded_result(server_name, status)
        failure_reason: str | None = None
        try:
            result = await client.call_tool(tool_name, arguments, abort_signal)
        except MCPError as exc:
            failure_reason = _error_text(exc)
            result = make_error_result(failure_reason)
        except Exception as exc:  # noqa: BLE001 - transport adapters fail closed
            failure_reason = _error_text(exc)
            result = make_error_result(failure_reason)
        async with lock:
            if (
                self._generations.get(server_name) != generation
                or self._clients.get(server_name) is not client
            ):
                return result
            if failure_reason is not None:
                self._transition(
                    server_name,
                    client=client,
                    generation=generation,
                    state="degraded",
                    reason=failure_reason,
                    next_retry_at=time.monotonic(),
                    failure_count=self._failure_counts.get(server_name, 0),
                )
                close_client = client
        await self._wait_for_failure(client, generation)
        if close_client is not None:
            await _close_failed_client(
                close_client, mount=self, generation=generation
            )
        async with lock:
            status = self.statuses.get(server_name)
            if status is not None and status.state == "degraded":
                return _degraded_result(server_name, status)
        return result

    async def _auto_remount(self, name: str, generation: int) -> None:
        lock = self._locks.setdefault(name, asyncio.Lock())
        task = asyncio.current_task()
        try:
            async with lock:
                config = self.configs.get(name)
                status = self.statuses.get(name)
                if (
                    self._generations.get(name) != generation
                    or config is None
                    or status is None
                    or status.state != "degraded"
                ):
                    return
            setup = await _setup_server(
                config,
                mount=self,
                generation=generation,
                client_ready=lambda client: self._arm_client(
                    name, client, generation
                ),
            )
            stale_client: MCPClient | None = None
            async with lock:
                current_config = self.configs.get(name)
                current_status = self.statuses.get(name)
                if (
                    self._closed
                    or current_config is not config
                    or self._generations.get(name) != generation
                    or current_status is None
                    or current_status.state != "degraded"
                ):
                    stale_client = setup.client
                elif setup.public.state == "mounted" and setup.client is not None:
                    self._finish_setup(setup, generation)
                else:
                    failure_count = self._failure_counts.get(name, 0) + 1
                    exponent = min(max(failure_count - 1, 0), 30)
                    delay = min(
                        AUTO_RECONNECT_MAX_DELAY_SECONDS,
                        AUTO_RECONNECT_BASE_DELAY_SECONDS * 2**exponent,
                    )
                    self._transition(
                        name,
                        client=None,
                        generation=generation,
                        state="degraded",
                        reason=setup.public.reason or "automatic remount failed",
                        next_retry_at=time.monotonic() + delay,
                        failure_count=failure_count,
                    )
            if stale_client is not None:
                await _close_failed_client(
                    stale_client, mount=self, generation=generation
                )
        finally:
            if task is not None:
                async with lock:
                    self._transition(
                        name,
                        client=None,
                        generation=generation,
                        clear_auto_task=task,
                        allow_closed=True,
                    )


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
        sources=dict(config.sources),
    )
    try:
        results = await asyncio.gather(
            *(
                _setup_server(
                    server_config,
                    notice_sink=notice_sink,
                    mount=mount,
                    generation=mount._generations[server_config.name],
                    client_ready=lambda client,
                    name=server_config.name,
                    generation=mount._generations[server_config.name]: mount._arm_client(
                        name, client, generation
                    ),
                )
                for server_config in configs.values()
            )
        )
    except BaseException:
        await mount.close()
        raise
    for setup in results:
        mount._finish_setup(setup, mount._generations[setup.public.name])
    return mount


async def _setup_server(
    server_config: MCPServerConfig,
    *,
    notice_sink: NoticeSink | None = None,
    mount: MCPMount | None = None,
    generation: int | None = None,
    client_ready: Callable[[MCPClient], None] | None = None,
) -> _SetupResult:
    if server_config.malformed_reason is not None:
        public = MCPServerStatus(
            server_config.name,
            server_config.transport,
            "malformed",
            reason=server_config.malformed_reason,
            stderr_log_path=str(mcp_log_path(server_config.name)),
        )
        _notice(
            notice_sink,
            f"mcp · {server_config.name} malformed ({server_config.malformed_reason})",
        )
        return _SetupResult(public)
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
            await _close_failed_client(
                client, mount=mount, generation=generation
            )
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
            await _close_failed_client(
                client, mount=mount, generation=generation
            )
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
            await _close_failed_client(
                client, mount=mount, generation=generation
            )
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


def _retry_text(retry_at: float | None) -> str:
    if retry_at is None:
        return "when backoff expires"
    remaining = retry_at - time.monotonic()
    if remaining <= 0:
        return "now"
    return f"in {math.ceil(remaining * 10) / 10:.1f}s"


def _degraded_result(
    name: str, status: MCPServerStatus | None
) -> StructuredToolResult:
    reason = (
        status.reason
        if status is not None and status.reason
        else "transport failure"
    )
    return make_error_result(
        f"MCP server '{name}' is degraded: {reason}. "
        f"Use /mcp reconnect {name}."
    )


def _build_client(config: MCPServerConfig) -> MCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return StreamableHTTPMCPClient(config)


async def _connect_and_list(client: MCPClient) -> list[MCPTool]:
    await client.connect()
    return await client.list_tools()


async def _close_failed_client(
    client: MCPClient,
    *,
    mount: MCPMount | None = None,
    generation: int | None = None,
) -> None:
    if mount is not None and generation is not None:
        lock = mount._locks.setdefault(client.config.name, asyncio.Lock())
        async with lock:
            current = mount._clients.get(client.config.name)
            if (
                current is client
                and mount._generations.get(client.config.name) != generation
            ):
                return
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
    generation = mount._generations[client.config.name]

    async def handler(
        arguments: dict[str, object], abort_signal: AbortSignal
    ) -> StructuredToolResult:
        return await mount.call_tool(
            client.config.name,
            tool.name,
            arguments,
            abort_signal,
            generation=generation,
        )

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
