"""MCP client adapters for zeta."""

from .client import MCPClient, MCPTool
from .config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    default_config_path,
    load_mcp_config,
)
from .http import StreamableHTTPMCPClient
from .mount import MCPMount, mount_mcp_servers
from .stdio import StdioMCPClient

__all__ = [
    "MCPClient", "MCPConfig", "MCPConfigError", "MCPMount", "MCPServerConfig",
    "MCPTool", "StdioMCPClient", "StreamableHTTPMCPClient", "default_config_path",
    "load_mcp_config", "mount_mcp_servers",
]
