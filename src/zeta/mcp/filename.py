"""Collision-free filename encoding for MCP server names."""

from urllib.parse import quote

_SAFE_FILENAME_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


def encode_mcp_filename(name: str) -> str:
    """Encode a complete server name while preserving existing safe names."""

    return quote(name, safe=_SAFE_FILENAME_CHARS)


__all__ = ["encode_mcp_filename"]
