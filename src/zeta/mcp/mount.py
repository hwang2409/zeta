"""Discover MCP tools and wire one lifecycle actor per server."""

from __future__ import annotations

import asyncio
import logging
import time  # noqa: F401 - kept as the monkey-patch seam for tests
from collections.abc import Callable
from pathlib import Path

from ..core.abort import AbortSignal
from ..tools.registry import ToolRegistry
from .client import MCPClient, MCPPrompt, MCPResource, MCPTool
from .config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    load_mcp_config,
    tool_prefix,
)
from .http import StreamableHTTPMCPClient
from .resources import ResourceAttachment, fetch_resource
from .resources import list_resources as fetch_resources
from .server_actor import (
    AUTO_RECONNECT_BASE_DELAY_SECONDS,
    AUTO_RECONNECT_MAX_DELAY_SECONDS,
    SERVER_SETUP_TIMEOUT_SECONDS,
    MCPServerActor,
    MCPServerState,
    MCPServerStatus,
    NoticeSink,
    _retry_text,
)
from .stdio import StdioMCPClient

logger = logging.getLogger(__name__)


class _ActorResourceClient:
    def __init__(
        self, actor: MCPServerActor, abort_signal: AbortSignal | None = None
    ) -> None:
        self._actor = actor
        self._abort_signal = abort_signal

    async def list_resources(self) -> list[MCPResource]:
        return await self._actor.list_resources(
            generation=self._actor.generation, abort_signal=self._abort_signal
        )

    async def read_resource(self, uri: str) -> str:
        return await self._actor.read_resource(
            uri, generation=self._actor.generation, abort_signal=self._abort_signal
        )


def _build_client(config: MCPServerConfig) -> MCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return StreamableHTTPMCPClient(config, home=_current_home())


_HOME_CONTEXT: list[str | None] = [None]


def _current_home() -> str | None:
    return _HOME_CONTEXT[-1]


def _make_build_client(
    home: str | None,
) -> Callable[[MCPServerConfig], MCPClient]:
    def build(config: MCPServerConfig) -> MCPClient:
        _HOME_CONTEXT.append(home)
        try:
            return _build_client(config)
        finally:
            _HOME_CONTEXT.pop()

    return build


async def _connect_and_list(client: MCPClient) -> list[MCPTool]:
    await client.connect()
    capabilities = getattr(client, "capabilities", None)
    if type(capabilities) is dict and "tools" not in capabilities:
        return []
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
        home: str | None = None,
    ) -> None:
        if isinstance(registry, ToolRegistry):
            self.registry: ToolRegistry | None = registry
            server_configs = dict(configs or {})
            clients: dict[str, MCPClient] = {}
        else:
            self.registry = None
            clients = {client.config.name: client for client in registry}
            server_configs = {client.config.name: client.config for client in registry}
        self.configs = server_configs
        self.statuses = dict(statuses or {})
        if self.registry is not None:
            # The session clone copies this pointer; activation still writes only
            # to the registry performing the discovery call.
            self.registry._mcp_mount = self
            # MCP discovery is a mount capability, not an optional catalog tool;
            # this also keeps minimal/test registries usable.
            from ..tools.mcp_discovery import register as register_discovery

            if self.registry._register_builtin:
                register_discovery(self.registry)
        self.sources = dict(sources or {})
        self.home = home
        self._clients = clients
        self._actors: dict[str, MCPServerActor] = {}
        self._catalog: dict[str, tuple[MCPServerActor, MCPTool]] = {}
        self._removed_actors: set[MCPServerActor] = set()
        self._removal_tasks: set[asyncio.Task[None]] = set()
        self._config_lock = asyncio.Lock()
        self._schema_refresh: Callable[[MCPMount], None] | None = None
        self._prompt_refresh: Callable[[MCPMount], None] | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def client_for(self, name: str) -> MCPClient | None:
        """Return the live client for one server, or None if it is not mounted."""

        return self._clients.get(name)

    @property
    def clients(self) -> tuple[MCPClient, ...]:
        """Return connected-client snapshots in configured order."""

        return tuple(
            self._clients[name] for name in self.configs if name in self._clients
        )

    @property
    def prompt_entries(self) -> tuple[tuple[str, str, MCPPrompt], ...]:
        """Return live prompt commands in configured server order."""

        entries: list[tuple[str, str, MCPPrompt]] = []
        for name in self.configs:
            actor = self._actors.get(name)
            if actor is None:
                continue
            for prompt in actor.prompts:
                entries.append((f"{name}:{prompt.name}", name, prompt))
        return tuple(entries)

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> str:
        server, separator, prompt_name = name.partition(":")
        if not separator or not server or not prompt_name:
            raise ValueError(f"unknown MCP prompt: {name}")
        actor = self._actors.get(server)
        if actor is None:
            raise ValueError(f"unknown MCP prompt: {name}")
        prompt = next(
            (candidate for candidate in actor.prompts if candidate.name == prompt_name),
            None,
        )
        if prompt is None:
            raise ValueError(f"unknown MCP prompt: {name}")
        return await actor.get_prompt(
            prompt.name,
            arguments,
            generation=actor.generation,
        )

    def search_tools(
        self, query: str, *, server: str | None = None, limit: int = 20
    ) -> tuple[tuple[str, str, bool], ...]:
        """Search names/descriptions locally and return a bounded catalog slice."""
        terms = tuple(part.casefold() for part in query.split() if part)
        results: list[tuple[str, str, bool]] = []
        for name in self.configs:
            if server is not None and name != server:
                continue
            actor = self._actors.get(name)
            if actor is None:
                continue
            for tool in actor._tools:
                haystack = f"{tool.name} {tool.description}".casefold()
                if terms and not all(term in haystack for term in terms):
                    continue
                qualified = f"{tool_prefix(name)}{tool.name}"
                results.append(
                    (
                        qualified,
                        tool.description,
                        qualified
                        in (self.registry.registered_names if self.registry else ()),
                    )
                )
                if len(results) >= max(1, min(limit, 20)):
                    return tuple(results)
        return tuple(results)

    def activate_tools(
        self, registry: ToolRegistry, names: list[str]
    ) -> tuple[list[str], list[str]]:
        activated: list[str] = []
        rejected: list[str] = []
        for name in dict.fromkeys(names):
            if name in registry._mcp_excluded_names:
                rejected.append(f"{name}: excluded from this session")
                continue
            entry = self._catalog.get(name)
            actor, tool = entry if entry is not None else (None, None)
            if actor is None or tool is None:
                rejected.append(f"{name}: unavailable")
            elif actor.register_tool_for(registry, tool) or registry.is_mcp_owned(
                name, actor, actor.generation
            ):
                activated.append(name)
            else:
                rejected.append(f"{name}: name collision or invalid schema")
        if activated:
            # Provider owners replace their schema snapshot synchronously; the
            # next completion therefore sees a deterministic activated set.
            self._refresh_schemas()
        return activated, rejected

    async def list_resources(
        self, server: str, *, limit: int = 50, abort_signal: AbortSignal | None = None
    ) -> list[MCPResource]:
        actor = self._actors.get(server)
        if actor is None:
            raise ValueError(f"{server} is not connected")
        resources = await fetch_resources(
            _ActorResourceClient(actor, abort_signal), server=server
        )
        return resources[: max(1, min(limit, 50))]

    async def read_resource(
        self, server: str, uri: str, *, abort_signal: AbortSignal | None = None
    ) -> ResourceAttachment:
        actor = self._actors.get(server)
        if actor is None:
            raise ValueError(f"{server} is not connected")
        return await fetch_resource(
            _ActorResourceClient(actor, abort_signal), server=server, uri=uri
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
        old_actor: MCPServerActor | None = None
        async with self._config_lock:
            actor = self._actors.get(name)
            if actor is None:
                raise ValueError(f"unknown MCP server: {name}")
            if self._closed:
                return actor.status
            if actor.is_terminal:
                old_actor = actor
                replacement = self._make_actor(
                    self.configs[name],
                    self.sources[name],
                )
                self._actors[name] = replacement
                replacement.start()
                actor = replacement
        if old_actor is not None:
            await old_actor.close()
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
                            logger.exception("failed to roll back MCP config %s", name)
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
                self._removed_actors.add(actor)
                self._refresh_schemas()
            else:
                server_config, next_source = next_server
                self.configs[name] = server_config
                self.sources[name] = next_source
        if next_server is None:
            await actor.remove_when_idle()
            async with self._config_lock:
                self._removed_actors.discard(actor)
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
            self._removed_actors.add(actor)
            self._refresh_schemas()
        removal_task = asyncio.create_task(self._remove_server_locked(actor))
        self._removal_tasks.add(removal_task)
        removal_task.add_done_callback(
            lambda task, removed_actor=actor: self._removal_finished(
                removed_actor, task
            )
        )
        await asyncio.shield(removal_task)

    async def _remove_server_locked(self, actor: MCPServerActor) -> None:
        await actor.remove()

    def _removal_finished(
        self,
        actor: MCPServerActor,
        task: asyncio.Task[None],
    ) -> None:
        self._removal_tasks.discard(task)
        self._removed_actors.discard(actor)

    async def close(self) -> None:
        async with self._config_lock:
            if self._close_task is None:
                self._closed = True
                actors = tuple(self._actors.values())
                actors += tuple(self._removed_actors)
                removal_tasks = tuple(self._removal_tasks)
                self._close_task = asyncio.create_task(
                    _close_actors(actors, removal_tasks)
                )
            close_task = self._close_task
        await asyncio.shield(close_task)

    def set_schema_refresh(self, callback: Callable[[MCPMount], None]) -> None:
        """Bind the owner of provider tool schemas after initial setup."""

        self._schema_refresh = callback
        callback(self)

    def set_prompt_refresh(self, callback: Callable[[MCPMount], None]) -> None:
        """Bind the owner of slash prompt commands after initial setup."""

        self._prompt_refresh = callback
        callback(self)

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
            build_client=_make_build_client(self.home),
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
        self._rebuild_catalog()
        self._refresh_schemas()

    def _rebuild_catalog(self) -> None:
        self._catalog = {
            f"{tool_prefix(name)}{tool.name}": (actor, tool)
            for name, actor in self._actors.items()
            for tool in actor._tools
        }

    def _refresh_schemas(self) -> None:
        if self._schema_refresh is not None:
            self._schema_refresh(self)
        if self._prompt_refresh is not None:
            self._prompt_refresh(self)


async def mount_mcp_servers(
    registry: ToolRegistry,
    config: MCPConfig | None = None,
    *,
    notice_sink: NoticeSink | None = None,
    home: str | None = None,
) -> MCPMount:
    """Connect configured servers and register each discovered tool."""

    if config is None:
        try:
            config = load_mcp_config()
        except MCPConfigError as exc:
            logger.error("%s", exc)
            return MCPMount(registry, {}, {}, home=home)

    mount = MCPMount(
        registry,
        config.configured_servers,
        {},
        sources=dict(config.sources),
        home=home,
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
        await asyncio.gather(*(actor.wait_started(notice_sink) for actor in actors))
    except BaseException:
        await mount.close()
        raise
    return mount


async def _close_actors(
    actors: tuple[MCPServerActor, ...],
    removal_tasks: tuple[asyncio.Task[None], ...] = (),
) -> None:
    unique = tuple(dict.fromkeys(actors))
    await asyncio.gather(
        *(actor.close() for actor in unique),
        *removal_tasks,
    )


__all__ = ["MCPMount", "MCPServerState", "MCPServerStatus", "mount_mcp_servers"]
