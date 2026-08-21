"""Discover MCP tools and mount them into zeta's tool registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.abort import AbortSignal
from ..tools.registry import ToolRegistry
from ..types import StructuredToolResult
from .client import MCPClient, MCPTool
from .config import MCPConfig, MCPConfigError, MCPServerConfig, load_mcp_config
from .http import StreamableHTTPMCPClient
from .stdio import StdioMCPClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MCPMount:
    """Connected MCP clients owned by one zeta session."""

    clients: tuple[MCPClient, ...]

    async def close(self) -> None:
        for client in self.clients:
            try:
                await client.close()
            except Exception:
                logger.exception("failed to close MCP server %s", client.config.name)


async def mount_mcp_servers(registry: ToolRegistry, config: MCPConfig | None = None) -> MCPMount:
    """Connect configured servers and register each discovered tool."""

    if config is None:
        try:
            config = load_mcp_config()
        except MCPConfigError as exc:
            logger.error("%s", exc)
            return MCPMount(())
    connected: list[MCPClient] = []
    for server_config in config.servers.values():
        client = _build_client(server_config)
        try:
            await client.connect()
            tools = await client.list_tools()
        except Exception as exc:  # noqa: BLE001 - isolate one bad server
            logger.warning("skipping MCP server %s: %s", server_config.name, exc)
            await client.close()
            continue
        connected.append(client)
        for tool in tools:
            _register_tool(registry, client, tool)
    return MCPMount(tuple(connected))


def _build_client(config: MCPServerConfig) -> MCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return StreamableHTTPMCPClient(config)


def _register_tool(registry: ToolRegistry, client: MCPClient, tool: MCPTool) -> None:
    name = f"{client.config.name}:{tool.name}"

    async def handler(arguments: dict[str, object], abort_signal: AbortSignal) -> StructuredToolResult:
        return await client.call_tool(tool.name, arguments, abort_signal)

    try:
        registry.register(name, handler, description=tool.description, parameters=tool.input_schema)
    except (TypeError, ValueError) as exc:
        logger.warning("skipping MCP tool %s: invalid input schema: %s", name, exc)


__all__ = ["MCPMount", "mount_mcp_servers"]
