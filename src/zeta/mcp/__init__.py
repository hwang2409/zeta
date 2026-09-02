"""MCP client adapters for zeta."""

from .client import MCPClient, MCPTool
from .config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    default_config_path,
    load_mcp_config,
    mcp_log_path,
)
from .http import StreamableHTTPMCPClient
from .mount import MCPMount, MCPServerState, MCPServerStatus, mount_mcp_servers
from .stdio import StdioMCPClient

__all__ = [
    "MCPClient",
    "MCPConfig",
    "MCPConfigError",
    "MCPMount",
    "MCPServerConfig",
    "MCPServerState",
    "MCPServerStatus",
    "MCPTool",
    "StdioMCPClient",
    "StreamableHTTPMCPClient",
    "default_config_path",
    "load_mcp_config",
    "mcp_log_path",
    "mount_mcp_servers",
]
