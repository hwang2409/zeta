"""MCP definition publication shared by server actors.

A server actor mounts one MCP catalog and then keeps a set of live tool
registries in sync with it: the primary registry plus every session clone that
adopted the actor's definitions. This unit owns that publication contract --
registering tools, remembering the active MCP-name set each live registry holds,
and re-publishing new-generation definitions after a reconnect so scoped
activations survive instead of collapsing into the primary registry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..core.abort import AbortSignal
from ..protocol.types import StructuredToolResult
from ..tools.registry import ToolRegistry
from .client import MCPClient, MCPTool, error_text, notice
from .config import MCPServerConfig, mcp_log_path, tool_prefix

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolsListChanged:
    client: MCPClient


@dataclass(frozen=True, slots=True)
class ToolsListRefreshed:
    client: MCPClient
    task: asyncio.Task[list[MCPTool]]


class MCPDefinitionPublisher:
    def register_registry(self, registry: ToolRegistry) -> None:
        self._owned_registries.add(registry)
        # Seed the remembered set from whatever this registry already holds for
        # us (a session clone copies live definitions before it is registered).
        self._active_names.setdefault(registry, set()).update(
            registry.mcp_owned_names(self)
        )

    def unregister_registry(self, registry: ToolRegistry) -> None:
        """Detach a registry so a reconnect never republishes into it again.

        Closing a registry is a lifecycle boundary: the actor drops it from the
        live set and forgets its remembered active names so a later reconnect
        cannot resurrect stale, actor-owned definitions in the closed session.
        """
        self._owned_registries.discard(registry)
        self._active_names.pop(registry, None)

    def register_tool_for(self, registry: ToolRegistry, tool: MCPTool) -> bool:
        self.register_registry(registry)
        return self._register_tool(tool, self._generation, registry=registry)

    def _register_tool(
        self, tool: MCPTool, generation: int, *, registry: ToolRegistry | None = None
    ) -> bool:
        target = registry or self._registry
        if (
            target is None
            or self._client is None
            or not self.config.allows_tool(tool.name)
        ):
            return False
        if getattr(target, "_closed", False):
            return False
        name = f"{tool_prefix(self.name)}{tool.name}"
        if name in target.registered_names:
            return False

        async def handler(
            arguments: dict[str, object], abort_signal: AbortSignal
        ) -> StructuredToolResult:
            return await self.call_tool(
                tool.name, arguments, abort_signal, generation=generation
            )

        try:
            registered = target.register_mcp(
                name,
                handler,
                owner=self,
                generation=generation,
                description=tool.description,
                parameters=tool.input_schema,
                approval_subject=self.config.approval_subjects.get(tool.name),
                validate_arguments=False,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("skipping MCP tool %s: invalid input schema: %s", name, exc)
            return False
        if registered:
            self._active_names.setdefault(target, set()).add(name)
        return registered

    def _unregister_tools(self, *, keep_primary: bool = False) -> None:
        # Remove the live definitions but keep the remembered active-name set so
        # the next generation can be republished into the same registries.
        for registry in tuple(self._owned_registries):
            if keep_primary and registry is self._registry:
                hide = getattr(registry, "hide_mcp_owner", None)
                if hide is not None:
                    hide(self)
                continue
            registry.unregister_mcp_owner(self)

    def _republish_definitions(self) -> None:
        """Re-register new-generation definitions into every still-live registry.

        The primary registry mirrors an eager catalog in full. Every other
        registry -- and the primary when the catalog is deferred -- regains
        exactly the names it held before the reconnect (its remembered active
        set), intersected with the new generation and minus its own exclusions.
        This keeps each child session's scoped activation alive across reconnects
        instead of dropping it into the primary registry alone.
        """
        from .server_actor import MCP_EAGER_TOOL_LIMIT

        tools_by_name = {tool.name: tool for tool in self._tools}
        prefix = tool_prefix(self.name)
        eager = len(self._tools) <= MCP_EAGER_TOOL_LIMIT
        for registry in tuple(self._owned_registries):
            if getattr(registry, "_closed", False):
                continue
            if registry is self._registry:
                desired = [
                    f"{prefix}{tool.name}"
                    for tool in self._tools
                    if eager
                    or (
                        registry.tool_allow is not None
                        and registry.tool_is_allowed(f"{prefix}{tool.name}")
                    )
                ]
            else:
                desired = list(self._active_names.get(registry, set()))
            for full in desired:
                if not full.startswith(prefix):
                    continue
                base = full[len(prefix) :]
                tool = tools_by_name.get(base)
                if tool is None or full in registry._mcp_excluded_names:
                    continue
                self._register_tool(tool, self._generation, registry=registry)

    def _filter_tools(
        self,
        tools: list[MCPTool],
        notice_sink: Callable[[str], None] | None,
        *,
        config: MCPServerConfig | None = None,
    ) -> list[MCPTool]:
        server_config = config or self.config
        names = {tool.name for tool in tools}
        unknown = server_config.unknown_tool_patterns(names)
        if unknown:
            detail = "; ".join(
                f"unknown {field}: {', '.join(patterns)}"
                for field, patterns in unknown
            )
            warning = f"mcp · {self.name} {detail}"
            logger.warning(warning)
            notice(notice_sink, warning)
        return [tool for tool in tools if server_config.allows_tool(tool.name)]

    def _notify(self, client: MCPClient, method: str) -> None:
        if method == "notifications/tools/list_changed":
            if (
                (client is self._client or client is self._setup_client)
                and self._status.state != "mounted"
            ):
                self._setup_notification_pending = True
            else:
                self._queue.put_nowait(ToolsListChanged(client))

    def _handle_tools_list_changed(self, message: ToolsListChanged) -> None:
        if message.client is not self._client or self._status.state != "mounted":
            return
        if self._tool_refresh_task is not None:
            self._tool_refresh_pending = True
            return
        task = asyncio.create_task(
            asyncio.wait_for(message.client.list_tools(), self._setup_timeout)
        )
        self._tool_refresh_task = task
        self._children.add(task)
        task.add_done_callback(
            lambda done, client=message.client: self._queue.put_nowait(
                ToolsListRefreshed(client, done)
            )
        )

    def _handle_tools_list_refreshed(self, message: ToolsListRefreshed) -> None:
        from .server_actor import MCPServerStatus

        self._children.discard(message.task)
        if self._tool_refresh_task is message.task:
            self._tool_refresh_task = None
        if message.client is not self._client or self._status.state != "mounted":
            self._tool_refresh_pending = False
            return
        try:
            tools = self._filter_tools(message.task.result(), self._notice_sink)
        except Exception as exc:  # noqa: BLE001 - refresh failure degrades the server
            self._tool_refresh_pending = False
            self._degrade_current(error_text(exc))
            return
        self._unregister_tools()
        self._tools = tuple(tools)
        self._set_status(
            MCPServerStatus(
                self.name,
                self.config.transport,
                "mounted",
                tool_count=len(self._tools),
                stderr_log_path=str(mcp_log_path(self.name)),
            )
        )
        self._republish_definitions()
        self._publish_callback(self, self._status, self._client)
        if not tools and (
            self.config.allowed_tools is not None or self.config.disallowed_tools
        ):
            notice(self._notice_sink, f"mcp · {self.name} mounted with zero tools")
        if self._tool_refresh_pending:
            self._tool_refresh_pending = False
            self._handle_tools_list_changed(ToolsListChanged(message.client))
