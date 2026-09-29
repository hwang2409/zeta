"""MCP definition publication shared by server actors."""

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

    def register_tool_for(self, registry: ToolRegistry, tool: MCPTool) -> bool:
        self.register_registry(registry)
        return self._register_tool(tool, self._generation, registry=registry)

    def _register_tool(
        self, tool: MCPTool, generation: int, *, registry: ToolRegistry | None = None
    ) -> bool:
        target = registry or self._registry
        if target is None or self._client is None:
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
            return target.register_mcp(
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

    def _unregister_tools(self, *, keep_primary: bool = False) -> None:
        for registry in tuple(self._owned_registries):
            if keep_primary and registry is self._registry:
                hide = getattr(registry, "hide_mcp_owner", None)
                if hide is not None:
                    hide(self)
                continue
            registry.unregister_mcp_owner(self)


