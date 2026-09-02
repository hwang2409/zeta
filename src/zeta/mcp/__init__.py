"""MCP client adapters for zeta."""

from .client import MCPClient, MCPTool
from .config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    default_config_path,
    home_config_path,
    load_mcp_config,
    load_mcp_config_overlay,
    mcp_log_path,
    project_config_path,
    read_mcp_config_file,
    server_to_json,
    write_mcp_config,
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
    "home_config_path",
    "load_mcp_config",
    "load_mcp_config_overlay",
    "mcp_log_path",
    "mount_mcp_servers",
    "project_config_path",
    "read_mcp_config_file",
    "server_to_json",
    "write_mcp_config",
]
