"""Discover MCP tools and wire one lifecycle actor per server."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from ..core.abort import AbortSignal
from ..tools.registry import ToolRegistry
from .client import MCPClient, MCPTool
from .config import MCPConfig, MCPConfigError, MCPServerConfig, load_mcp_config
from .http import StreamableHTTPMCPClient
from .server_actor import (
    AUTO_RECONNECT_BASE_DELAY_SECONDS,
    AUTO_RECONNECT_MAX_DELAY_SECONDS,
    MCPServerActor,
    MCPServerState,
    MCPServerStatus,
    NoticeSink,
    SERVER_SETUP_TIMEOUT_SECONDS,
    _degraded_result,
    _retry_text,
)
from .stdio import StdioMCPClient

logger = logging.getLogger(__name__)


def _build_client(config: MCPServerConfig) -> MCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return StreamableHTTPMCPClient(config)


async def _connect_and_list(client: MCPClient) -> list[MCPTool]:
    await client.connect()
    return await client.list_tools()


class MCPMount:
    """Published MCP snapshots and one actor handle for each server."""

    def __init__(
        self,
        registry: ToolRegistry | tuple[MCPClient, ...],
        configs: dict[str, MCPServerConfig] | None = None,
        statuses: dict[str, MCPServerStatus] | None = None,
        *,
        sources: dict[str, Path] | None = None,
    ) -> None:
        if isinstance(registry, ToolRegistry):
            self.registry: ToolRegistry | None = registry
            server_configs = dict(configs or {})
            clients: dict[str, MCPClient] = {}
        else:
            self.registry = None
            clients = {client.config.name: client for client in registry}
            server_configs = {
                client.config.name: client.config for client in registry
            }
        self.configs = server_configs
        self.statuses = dict(statuses or {})
        self.sources = dict(sources or {})
        self._clients = clients
        self._actors: dict[str, MCPServerActor] = {}
        self._removed_actors: dict[str, MCPServerActor] = {}
        self._config_lock = asyncio.Lock()
        self._schema_refresh: Callable[[MCPMount], None] | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def clients(self) -> tuple[MCPClient, ...]:
        """Return connected-client snapshots in configured order."""

        return tuple(
            self._clients[name]
            for name in self.configs
            if name in self._clients
        )

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
        for name in self.configs:
            status = self.statuses.get(name)
            if status is None:
                continue
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
        actor = await self._actor_for(name)
        if self._closed:
            return actor.status
        if actor.is_terminal:
            actor = await self._replace_terminal_actor(name, actor)
        return await actor.reconnect(notice_sink=notice_sink)

    async def add_server(
        self,
        server_config: MCPServerConfig,
        *,
        source: Path,
        notice_sink: NoticeSink | None = None,
        prepare: Callable[[], None] | None = None,
        rollback: Callable[[], None] | None = None,
    ) -> MCPServerStatus:
        """Commit a config entry, then start its lifecycle actor."""

        name = server_config.name
        async with self._config_lock:
            if self._closed:
                raise ValueError("MCP mount is closed")
            if name in self._actors:
                raise ValueError(f"MCP server already configured: {name}")
            if prepare is not None:
                prepare()
            actor = self._make_actor(
                server_config,
                source,
            )
            self.configs[name] = server_config
            self.sources[name] = source
            self._actors[name] = actor
            actor.start()
        try:
            return await actor.wait_started(notice_sink)
        except BaseException:
            async with self._config_lock:
                if self._actors.get(name) is actor:
                    self._actors.pop(name, None)
                    self.configs.pop(name, None)
                    self.statuses.pop(name, None)
                    self.sources.pop(name, None)
                    self._clients.pop(name, None)
                    self._refresh_schemas()
                    if rollback is not None:
                        try:
                            rollback()
                        except BaseException:
                            logger.exception(
                                "failed to roll back MCP config %s", name
                            )
            await actor.close()
            raise

    async def replace_server(
        self,
        name: str,
        *,
        persist: Callable[[Path], None],
        replacement: Callable[[Path], tuple[MCPServerConfig, Path] | None],
        notice_sink: NoticeSink | None = None,
    ) -> None:
        """Commit a config replacement and send it to the server actor."""

        async with self._config_lock:
            actor = self._actors.get(name)
            if actor is None:
                raise ValueError(f"unknown MCP server: {name}")
            source = self.sources.get(name)
            if source is None:
                raise ValueError(f"unknown MCP server: {name}")
            next_server = replacement(source)
            persist(source)
            if next_server is None:
                self._actors.pop(name, None)
                self.configs.pop(name, None)
                self.statuses.pop(name, None)
                self.sources.pop(name, None)
                self._clients.pop(name, None)
                self._removed_actors[name] = actor
                self._refresh_schemas()
            else:
                server_config, next_source = next_server
                self.configs[name] = server_config
                self.sources[name] = next_source
        if next_server is None:
            await actor.remove_when_idle()
            async with self._config_lock:
                if self._removed_actors.get(name) is actor:
                    self._removed_actors.pop(name, None)
            return
        await actor.replace(
            server_config,
            next_source,
            notice_sink=notice_sink,
        )

    async def remove_server(self, name: str) -> None:
        """Remove one configured server and await its actor cleanup."""

        async with self._config_lock:
            actor = self._actors.get(name)
            if actor is None:
                raise ValueError(f"unknown MCP server: {name}")
            self._actors.pop(name, None)
            self.configs.pop(name, None)
            self.statuses.pop(name, None)
            self.sources.pop(name, None)
            self._clients.pop(name, None)
            self._removed_actors[name] = actor
            self._refresh_schemas()
        await self._remove_server_locked(name)

    async def _remove_server_locked(self, name: str) -> None:
        actor = self._removed_actors.get(name)
        if actor is not None:
            try:
                await actor.remove()
            finally:
                async with self._config_lock:
                    if self._removed_actors.get(name) is actor:
                        self._removed_actors.pop(name, None)

    async def close(self) -> None:
        async with self._config_lock:
            if self._close_task is None:
                self._closed = True
                actors = tuple(self._actors.values())
                actors += tuple(self._removed_actors.values())
                self._close_task = asyncio.create_task(_close_actors(actors))
            close_task = self._close_task
        await asyncio.shield(close_task)

    def set_schema_refresh(self, callback: Callable[[MCPMount], None]) -> None:
        """Bind the owner of provider tool schemas after initial setup."""

        self._schema_refresh = callback
        callback(self)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
        abort_signal: AbortSignal,
        *,
        generation: int,
    ):
        actor = self._actors.get(server_name)
        if actor is None:
            return _degraded_result(server_name, self.statuses.get(server_name))
        return await actor.call_tool(
            tool_name,
            arguments,
            abort_signal,
            generation=generation,
        )

    async def _actor_for(self, name: str) -> MCPServerActor:
        async with self._config_lock:
            actor = self._actors.get(name)
            if actor is None:
                raise ValueError(f"unknown MCP server: {name}")
            return actor

    async def _replace_terminal_actor(
        self,
        name: str,
        actor: MCPServerActor,
    ) -> MCPServerActor:
        async with self._config_lock:
            current = self._actors.get(name)
            if current is None:
                raise ValueError(f"unknown MCP server: {name}")
            if current is not actor or not current.is_terminal:
                return current
            replacement = self._make_actor(current.config, current.source)
            self._actors[name] = replacement
            replacement.start()
        await actor.close()
        return replacement

    def _make_actor(
        self,
        config: MCPServerConfig,
        source: Path,
    ) -> MCPServerActor:
        return MCPServerActor(
            config,
            source=source,
            registry=self.registry,
            publish=self._publish,
            build_client=_build_client,
            connect_and_list=_connect_and_list,
            setup_timeout=SERVER_SETUP_TIMEOUT_SECONDS,
            auto_base_delay=AUTO_RECONNECT_BASE_DELAY_SECONDS,
            auto_max_delay=AUTO_RECONNECT_MAX_DELAY_SECONDS,
        )

    def _publish(
        self,
        actor: MCPServerActor,
        status: MCPServerStatus,
        client: MCPClient | None,
    ) -> None:
        if self._actors.get(actor.name) is not actor:
            return
        self.configs[actor.name] = actor.config
        self.sources[actor.name] = actor.source
        self.statuses[actor.name] = status
        if client is None:
            self._clients.pop(actor.name, None)
        else:
            self._clients[actor.name] = client
        self._refresh_schemas()

    def _refresh_schemas(self) -> None:
        if self._schema_refresh is not None:
            self._schema_refresh(self)

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

    mount = MCPMount(
        registry,
        config.configured_servers,
        {},
        sources=dict(config.sources),
    )
    actors: list[MCPServerActor] = []
    for server_config in config.configured_servers.values():
        actor = mount._make_actor(
            server_config,
            config.sources.get(server_config.name, config.path),
        )
        mount._actors[server_config.name] = actor
        actor.start()
        actors.append(actor)
    try:
        await asyncio.gather(
            *(actor.wait_started(notice_sink) for actor in actors)
        )
    except BaseException:
        await mount.close()
        raise
    return mount


async def _close_actors(actors: tuple[MCPServerActor, ...]) -> None:
    unique = tuple(dict.fromkeys(actors))
    await asyncio.gather(*(actor.close() for actor in unique))


__all__ = ["MCPMount", "MCPServerState", "MCPServerStatus", "mount_mcp_servers"]
