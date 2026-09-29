"""Small model-facing MCP catalog and activation tool."""

from __future__ import annotations

from ..core.abort import AbortSignal
from ..mcp.resources import MCPResourceError
from ..protocol.types import StructuredToolResult
from .registry import ToolRegistry, _success_result, text_block

MAX_DISCOVERY_RESULTS = 20
MAX_RESOURCE_RESULTS = 50


def _text(value: str) -> StructuredToolResult:
    return _success_result(text_block(value))


async def _discover(
    registry: ToolRegistry,
    arguments: dict[str, object],
    _abort: AbortSignal,
) -> StructuredToolResult:
    mount = registry._mcp_mount
    if mount is None:
        return _text("No MCP servers are mounted.")
    action = arguments.get("action", "search")
    if type(action) is not str:
        return _text("action must be search, activate, resources, or read_resource")
    if action == "search":
        query = arguments.get("query", "")
        if type(query) is not str:
            return _text("query must be a string")
        server = arguments.get("server")
        if server is not None and type(server) is not str:
            return _text("server must be a string")
        matches = mount.search_tools(query, server=server, limit=MAX_DISCOVERY_RESULTS)
        if not matches:
            return _text("No matching MCP tools.")
        lines = [
            f"{name}: {description or '(no description)'}"
            for name, description, active in matches
        ]
        suffix = "\nUse action=activate with names from this list."
        return _text("\n".join(lines) + suffix)
    if action == "activate":
        names = arguments.get("names")
        if type(names) is not list or not names or len(names) > MAX_DISCOVERY_RESULTS:
            return _text(f"names must contain 1-{MAX_DISCOVERY_RESULTS} tool names")
        if any(type(name) is not str for name in names):
            return _text("names must contain strings")
        activated, rejected = mount.activate_tools(registry, names)
        message = []
        if activated:
            message.append("Activated for the next turn: " + ", ".join(activated))
        if rejected:
            message.append("Not activated: " + "; ".join(rejected))
        return _text("\n".join(message) or "No tools activated.")
    server = arguments.get("server")
    if type(server) is not str or not server:
        return _text("server is required for resource actions")
    if action == "resources":
        try:
            resources = await mount.list_resources(server, limit=MAX_RESOURCE_RESULTS)
        except (MCPResourceError, ValueError) as exc:
            return _text(f"MCP resources unavailable: {exc}")
        if not resources:
            return _text(f"{server}: no resources")
        return _text(
            "\n".join(
                f"{r.uri}: {r.name or r.description or '(resource)'}" for r in resources
            )
        )
    if action == "read_resource":
        uri = arguments.get("uri")
        if type(uri) is not str or not uri:
            return _text("uri is required")
        try:
            attachment = await mount.read_resource(server, uri)
        except (MCPResourceError, ValueError) as exc:
            return _text(f"MCP resource unavailable: {exc}")
        return _text(attachment.labeled_text)
    return _text("action must be search, activate, resources, or read_resource")


def register(registry: ToolRegistry) -> None:
    # A user/custom tool must never be overwritten by a built-in.
    if "mcp_discover" in registry.registered_names:
        return
    registry.register_session_tool(
        "mcp_discover",
        _discover,
        description=(
            "Search the mounted MCP catalog without exposing it, activate selected "
            "tools for the next turn, or list/read bounded MCP resources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "activate", "resources", "read_resource"],
                },
                "query": {
                    "type": "string",
                    "description": "Text to match in tool names and descriptions.",
                },
                "server": {"type": "string"},
                "names": {"type": "array", "items": {"type": "string"}},
                "uri": {"type": "string"},
            },
            "additionalProperties": False,
        },
        requires_approval=False,
        validate_arguments=True,
    )
