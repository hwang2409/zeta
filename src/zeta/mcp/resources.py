"""MCP resource attachment: list, read, spill, and format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..tools._spill import SpillStore
from .client import MCPClient, MCPResource, MCPResourceContent

RESOURCE_MAX_BYTES = 200_000


class MCPResourceError(RuntimeError):
    """User-facing failure reading or listing MCP resources."""


class MCPResourceTooLargeError(MCPResourceError):
    """Deprecated compatibility name; large resources now spill."""


@dataclass(frozen=True, slots=True)
class ResourceAttachment:
    """A resolved MCP resource formatted for a user turn."""

    server: str
    uri: str
    text: str
    labeled_text: str
    spill_paths: tuple[Path, ...] = ()


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
    spill_store: SpillStore | None = None,
    max_bytes: int = RESOURCE_MAX_BYTES,
) -> ResourceAttachment:
    """Read one resource, keeping large text and every blob in spill storage."""

    try:
        raw = await client.read_resource(uri)
    except Exception as exc:
        raise MCPResourceError(
            f"could not read resource {uri} from {server}: {exc}"
        ) from exc
    contents = (
        (MCPResourceContent(raw),)
        if type(raw) is str
        else raw
    )
    if not contents or not all(isinstance(item, MCPResourceContent) for item in contents):
        raise MCPResourceError(f"resource {uri} from {server} has invalid content")

    encoded = [
        item.data.encode("utf-8") if type(item.data) is str else item.data
        for item in contents
    ]
    total_size = sum(len(data) for data in encoded)
    needs_spill = total_size > max_bytes or any(type(item.data) is bytes for item in contents)
    paths: tuple[Path, ...] = ()
    if needs_spill:
        if spill_store is None:
            spill_store = SpillStore()
        published = await spill_store.awrite_group(
            "mcp-resource",
            f"{server}-{uri}",
            {str(index): [data] for index, data in enumerate(encoded)},
        )
        paths = tuple(published[str(index)] for index in range(len(encoded)))

    remaining = max_bytes
    previews: list[str] = []
    notices: list[str] = []
    for index, (item, data) in enumerate(zip(contents, encoded, strict=True)):
        mime = item.mime_type or (
            "text/plain" if type(item.data) is str else "application/octet-stream"
        )
        if type(item.data) is str and remaining > 0:
            preview_data = data[:remaining]
            while preview_data:
                try:
                    preview = preview_data.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    preview_data = preview_data[:-1]
            else:
                preview = ""
            previews.append(preview)
            remaining -= len(preview_data)
        if needs_spill:
            notices.append(
                f"content[{index}] mime={mime} · {len(data)} bytes · full content at {paths[index]}"
            )

    text = "\n".join(previews)
    label = f"[mcp-resource: {server}:{uri} · {total_size} bytes]"
    body = "\n".join(part for part in (text, *notices) if part)
    return ResourceAttachment(
        server=server,
        uri=uri,
        text=text,
        labeled_text=f"{label}\n{body}" if body else label,
        spill_paths=paths,
    )


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
