"""Policy-checked construction for MCP clients."""

from __future__ import annotations

from ..tools.registry import ToolRegistry
from .client import MCPClient
from .config import MCPServerConfig
from .http import StreamableHTTPMCPClient
from .stdio import StdioMCPClient


def mcp_server_allowed(registry: ToolRegistry | None, server: str) -> bool:
    """Return whether the active session policy permits a server namespace."""

    return registry is None or registry.tool_policy.allows_mcp_server(server)


class MCPServerPolicyError(RuntimeError):
    """Raised before a client is built for a blocked server namespace."""


def build_mcp_client(
    config: MCPServerConfig,
    *,
    registry: ToolRegistry | None,
    home: str | None = None,
) -> MCPClient:
    """Build a client only after the shared namespace-policy gate permits it."""

    if not mcp_server_allowed(registry, config.name):
        raise MCPServerPolicyError(f"{config.name}: skipped by tool policy")
    spill_store = registry.spills if registry is not None else None
    if config.transport == "stdio":
        return StdioMCPClient(config, spill_store=spill_store)
    return StreamableHTTPMCPClient(config, home=home, spill_store=spill_store)


__all__ = ["MCPServerPolicyError", "build_mcp_client", "mcp_server_allowed"]
