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
    ToolContentBlock,
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
        if status_code == 401:
            message = f"MCP HTTP 401: bearer token rejected by MCP server: {detail}"
        elif status_code == 0:
            message = f"MCP HTTP 0: MCP connection failed: {detail}"
        else:
            message = f"MCP HTTP {status_code}: {detail}"
        super().__init__(message)
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


@dataclass(frozen=True, slots=True)
class MCPPromptArgument:
    name: str
    description: str = ""
    required: bool = False


@dataclass(frozen=True, slots=True)
class MCPPrompt:
    name: str
    description: str = ""
    arguments: tuple[MCPPromptArgument, ...] = ()


class MCPClient(Protocol):
    config: MCPServerConfig
    protocol_version: str | None
    capabilities: dict[str, object]

    async def connect(self) -> None:
        """Open the transport and complete initialize."""

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools exposed by the server."""

    async def list_prompts(self) -> list[MCPPrompt]:
        """Discover prompts exposed by the server."""

    async def get_prompt(self, name: str, arguments: Mapping[str, str]) -> str:
        """Resolve one prompt into user-facing text."""

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
        raise MCPProtocolError("MCP response result must be an object")
    content_value = value.get("content", [])
    if type(content_value) is not list:
        raise MCPProtocolError("MCP tool result content must be an array")
    blocks: list[ToolContentBlock] = []
    for index, item in enumerate(content_value):
        if type(item) is not dict:
            raise MCPProtocolError(f"MCP content[{index}] must be an object")
        if item.get("type") == "text":
            if type(item.get("text")) is not str:
                raise MCPProtocolError(
                    f"invalid MCP content[{index}].text: must be a string"
                )
            blocks.append(text_block(item["text"]))
            continue
        try:
            block = validate_tool_content_block(index, item)
        except ValueError as exc:
            raise MCPProtocolError(f"invalid MCP content[{index}]: {exc}") from exc
        if block["type"] == "text":
            blocks.append(text_block(block["text"]))
        else:
            blocks.append(block)
    is_error = value.get("isError", False)
    if type(is_error) is not bool:
        raise MCPProtocolError("MCP tool result isError must be a boolean")
    structured = value.get("structuredContent")
    if structured is not None:
        if type(structured) is not dict:
            raise MCPProtocolError("MCP structuredContent must be an object")
        try:
            _validate_json_value(structured)
        except ValueError as exc:
            raise MCPProtocolError(f"MCP structuredContent is invalid: {exc}") from exc
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
        code = error.get("code")
        message = error.get("message")
        if type(code) is not int or type(message) is not str:
            raise MCPProtocolError("MCP JSON-RPC error has an invalid shape")
        return make_error_result(message)
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


def prompts_from_result(value: Mapping[str, object]) -> list[MCPPrompt]:
    raw_prompts = value.get("prompts")
    if type(raw_prompts) is not list:
        raise MCPProtocolError("MCP prompts/list result must contain prompts")
    prompts: list[MCPPrompt] = []
    for item in raw_prompts:
        if type(item) is not dict:
            raise MCPProtocolError("MCP prompt declaration must be an object")
        name = item.get("name")
        description = item.get("description", "")
        raw_arguments = item.get("arguments", [])
        if type(name) is not str or not name:
            raise MCPProtocolError("MCP prompt name must be a nonempty string")
        if type(description) is not str:
            description = ""
        if type(raw_arguments) is not list:
            raise MCPProtocolError(f"MCP prompt {name} arguments must be an array")
        arguments: list[MCPPromptArgument] = []
        for argument in raw_arguments:
            if type(argument) is not dict:
                raise MCPProtocolError(
                    f"MCP prompt {name} argument must be an object"
                )
            argument_name = argument.get("name")
            argument_description = argument.get("description", "")
            required = argument.get("required", False)
            if type(argument_name) is not str or not argument_name:
                raise MCPProtocolError(
                    f"MCP prompt {name} argument name must be a nonempty string"
                )
            if type(argument_description) is not str:
                argument_description = ""
            if type(required) is not bool:
                raise MCPProtocolError(
                    f"MCP prompt {name} argument {argument_name} required must be boolean"
                )
            arguments.append(
                MCPPromptArgument(argument_name, argument_description, required)
            )
        prompts.append(MCPPrompt(name, description, tuple(arguments)))
    return prompts


def prompt_text_from_result(value: Mapping[str, object]) -> str:
    messages = value.get("messages")
    if messages is None and value.get("isError") is True:
        content = value.get("content")
        if type(content) is list and content and type(content[0]) is dict:
            message = content[0].get("text")
            if type(message) is str:
                raise MCPProtocolError(message)
        raise MCPProtocolError("MCP prompts/get failed")
    if type(messages) is not list:
        raise MCPProtocolError("MCP prompts/get result must contain messages")
    text: list[str] = []
    for index, message in enumerate(messages):
        if type(message) is not dict:
            raise MCPProtocolError(f"MCP prompt message[{index}] must be an object")
        content = message.get("content")
        if type(content) is not dict or content.get("type") != "text":
            raise MCPProtocolError(
                f"MCP prompt message[{index}] content must be text"
            )
        message_text = content.get("text")
        if type(message_text) is not str:
            raise MCPProtocolError(
                f"MCP prompt message[{index}] text must be a string"
            )
        text.append(message_text)
    return "\n".join(text)


__all__ = [
    "MCPCanceled",
    "MCPClient",
    "MCPError",
    "MCPHTTPError",
    "MCPPrompt",
    "MCPPromptArgument",
    "MCPProtocolError",
    "MCPTool",
    "canceled_result",
    "initialize_params",
    "make_error_result",
    "parse_rpc_response",
    "prompt_text_from_result",
    "prompts_from_result",
    "tools_from_result",
    "translate_call_result",
]
