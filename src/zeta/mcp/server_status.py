from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import MCPServerConfig, mcp_log_path

MCPServerState = Literal[
    "mounted",
    "degraded",
    "failed",
    "skipped-missing-env",
    "skipped-policy",
    "timed-out",
    "malformed",
]


@dataclass(frozen=True, slots=True)
class MCPServerStatus:
    """Current connection state for one configured MCP server."""

    name: str
    transport: str
    state: MCPServerState
    tool_count: int = 0
    reason: str | None = None
    stderr_log_path: str = ""
    next_retry_at: float | None = None


def unavailable_status(name: str, config: MCPServerConfig) -> MCPServerStatus:
    return MCPServerStatus(
        name,
        config.transport,
        "failed",
        reason="MCP mount is closed",
        stderr_log_path=str(mcp_log_path(name)),
    )
