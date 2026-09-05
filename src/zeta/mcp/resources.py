"""MCP resource attachment: list, read, and format for the composer."""

from __future__ import annotations

from dataclasses import dataclass

from .client import MCPClient, MCPResource

RESOURCE_MAX_BYTES = 200_000


class MCPResourceError(RuntimeError):
    """User-facing failure reading or listing MCP resources."""


class MCPResourceTooLargeError(MCPResourceError):
    """Raised when a resource payload exceeds the size bound."""


@dataclass(frozen=True, slots=True)
class ResourceAttachment:
    """A resolved MCP resource formatted as text for a user turn."""

    server: str
    uri: str
    text: str
    labeled_text: str


async def list_resources(client: MCPClient, *, server: str) -> list[MCPResource]:
    """Return every resource declared by one live MCP server."""

    try:
        return await client.list_resources()
    except Exception as exc:
        raise MCPResourceError(
            f"could not list resources for {server}: {exc}"
        ) from exc


async def fetch_resource(
    client: MCPClient,
    *,
    server: str,
    uri: str,
    max_bytes: int = RESOURCE_MAX_BYTES,
) -> ResourceAttachment:
    """Read one resource and return the labeled text block for attachment."""

    try:
        text = await client.read_resource(uri)
    except Exception as exc:
        raise MCPResourceError(
            f"could not read resource {uri} from {server}: {exc}"
        ) from exc
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        raise MCPResourceTooLargeError(
            f"resource {uri} from {server} is {size} bytes; limit is {max_bytes} bytes"
        )
    labeled = f"[mcp-resource: {server}:{uri} · {size} bytes]\n{text}"
    return ResourceAttachment(server=server, uri=uri, text=text, labeled_text=labeled)


def format_resource_list(server: str, resources: list[MCPResource]) -> str:
    """Render one server's resource list for /mcp resources output."""

    if not resources:
        return f"{server}: no resources"
    lines = [f"{server}: {len(resources)} resources"]
    for resource in resources:
        parts = [resource.uri]
        if resource.name:
            parts.append(f"name={resource.name}")
        if resource.mime_type:
            parts.append(f"mime={resource.mime_type}")
        if resource.description:
            parts.append(f"description={resource.description}")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


__all__ = [
    "RESOURCE_MAX_BYTES",
    "MCPResourceError",
    "MCPResourceTooLargeError",
    "ResourceAttachment",
    "fetch_resource",
    "format_resource_list",
    "list_resources",
]
