"""Shared bounded cursor pagination for MCP list methods."""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from .client import MCPProtocolError

T = TypeVar("T")


async def drain_pages(  # noqa: UP047
    request: Callable[[str, dict[str, object]], Awaitable[dict[str, object]]],
    method: str,
    parser: Callable[[dict[str, object]], list[T]],
    label: str,
    *,
    max_pages: int = 1_000,
    max_items: int = 10_000,
) -> list[T]:
    """Follow a cursor chain, rejecting cycles and unbounded responses."""
    items: list[T] = []
    cursor: str | None = None
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        params = {} if cursor is None else {"cursor": cursor}
        result = await request(method, params)
        items.extend(parser(result))
        if len(items) > max_items:
            raise MCPProtocolError(f"MCP {label} exceeded the {max_items} item limit")
        next_cursor = result.get("nextCursor")
        if next_cursor is None or next_cursor == "":
            return items
        if type(next_cursor) is not str:
            raise MCPProtocolError(f"MCP {label} returned malformed nextCursor")
        if next_cursor in seen:
            raise MCPProtocolError(f"MCP {label} cursor repeated")
        seen.add(next_cursor)
        cursor = next_cursor
    raise MCPProtocolError(f"MCP {label} exceeded the {max_pages} page limit")
