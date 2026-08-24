"""Shared MCP protocol types and result translation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..core.abort import AbortSignal
from ..tools.registry import text_block
from ..types import (
    StructuredToolResult,
    ToolTextBlock,
    flatten_tool_content,
    validate_tool_content_block,
)
from .config import MCPServerConfig


class MCPError(RuntimeError):
    """Base error for MCP transport and protocol failures."""


class MCPProtocolError(MCPError):
    """Raised for a malformed or error JSON-RPC response."""


class MCPHTTPError(MCPError):
    """Raised for an unsuccessful streamable-http response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"MCP HTTP {status_code}: {detail}")
        self.status_code = status_code


class MCPCanceled(MCPError):
    """Raised when an MCP request is aborted by the caller."""


@dataclass(frozen=True, slots=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, object]

    def __init__(self, name: str, description: str, input_schema: Mapping[str, object]) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "input_schema", dict(input_schema))


class MCPClient(Protocol):
    config: MCPServerConfig
    protocol_version: str | None

    async def connect(self) -> None:
        """Open the transport and complete initialize."""

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools exposed by the server."""

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
        abort_signal: AbortSignal,
    ) -> StructuredToolResult:
        """Call one remote tool."""

    async def close(self) -> None:
        """Close the transport and release child processes or clients."""


def initialize_params() -> dict[str, object]:
    return {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "zeta", "version": "0.1.0"},
    }


def make_error_result(message: str) -> StructuredToolResult:
    return {"content": [text_block(message)], "isError": True, "structuredContent": None}


def canceled_result() -> StructuredToolResult:
    return make_error_result("tool execution canceled")


def translate_call_result(value: object) -> StructuredToolResult:
    if type(value) is not dict:
        return make_error_result("MCP response result must be an object")
    content_value = value.get("content", [])
    if type(content_value) is not list:
        return make_error_result("MCP tool result content must be an array")
    blocks: list[ToolTextBlock] = []
    for index, item in enumerate(content_value):
        if type(item) is not dict:
            return make_error_result(f"MCP content[{index}] must be an object")
        if item.get("type") != "text" or type(item.get("text")) is not str:
            try:
                block = validate_tool_content_block(index, item)
            except ValueError:
                rendered = f"[unsupported MCP block: {item.get('type', 'unknown')}]"
            else:
                rendered = flatten_tool_content([block])
            blocks.append(text_block(rendered))
            continue
        blocks.append(text_block(item["text"]))
    is_error = value.get("isError", False)
    if type(is_error) is not bool:
        return make_error_result("MCP tool result isError must be a boolean")
    structured = value.get("structuredContent")
    if structured is not None:
        if type(structured) is not dict:
            return make_error_result("MCP structuredContent must be an object")
        try:
            _validate_json_value(structured)
        except ValueError as exc:
            return make_error_result(f"MCP structuredContent is invalid: {exc}")
        structured_content = structured
    else:
        structured_content = None
    return {"content": blocks, "isError": is_error, "structuredContent": structured_content}


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {str, int, bool, float}:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("floating point values must be finite")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("object keys must be strings")
            _validate_json_value(item)
        return
    raise ValueError("must contain JSON values")


def parse_rpc_response(value: object, request_id: int) -> dict[str, object]:
    if type(value) is not dict:
        raise MCPProtocolError("MCP JSON-RPC response must be an object")
    if value.get("jsonrpc") != "2.0" or value.get("id") != request_id:
        raise MCPProtocolError("MCP JSON-RPC response has an invalid id")
    error = value.get("error")
    if error is not None:
        if type(error) is not dict:
            raise MCPProtocolError("MCP JSON-RPC error must be an object")
        raise MCPProtocolError(str(error.get("message", "unknown MCP error")))
    result = value.get("result")
    if type(result) is not dict:
        raise MCPProtocolError("MCP JSON-RPC response result must be an object")
    return result


def tools_from_result(value: Mapping[str, object]) -> list[MCPTool]:
    raw_tools = value.get("tools")
    if type(raw_tools) is not list:
        raise MCPProtocolError("MCP tools/list result must contain tools")
    tools: list[MCPTool] = []
    for item in raw_tools:
        if type(item) is not dict:
            raise MCPProtocolError("MCP tool declaration must be an object")
        name = item.get("name")
        description = item.get("description", "")
        schema = item.get("inputSchema", {"type": "object", "properties": {}})
        if type(name) is not str or not name:
            raise MCPProtocolError("MCP tool name must be a nonempty string")
        if type(description) is not str:
            description = ""
        if type(schema) is not dict:
            raise MCPProtocolError(f"MCP tool {name} inputSchema must be an object")
        tools.append(MCPTool(name, description, schema))
    return tools


__all__ = [
    "MCPCanceled",
    "MCPClient",
    "MCPError",
    "MCPHTTPError",
    "MCPProtocolError",
    "MCPTool",
    "canceled_result",
    "initialize_params",
    "make_error_result",
    "parse_rpc_response",
    "tools_from_result",
    "translate_call_result",
]
