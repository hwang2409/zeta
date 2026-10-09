"""MCP definition publication shared by server actors.

A server actor mounts one MCP catalog and then keeps a set of live tool
registries in sync with it: the primary registry plus every session clone that
adopted the actor's definitions. This unit owns that publication contract --
registering tools, remembering the active MCP-name set each live registry holds,
and re-publishing new-generation definitions after a reconnect so scoped
activations survive instead of collapsing into the primary registry.
"""

import logging

from ..core.abort import AbortSignal
from ..protocol.types import StructuredToolResult
from ..tools.registry import ToolRegistry
from .client import MCPTool
from .config import tool_prefix

logger = logging.getLogger(__name__)


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
