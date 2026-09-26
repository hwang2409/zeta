"""Text blocks and normalized tool execution results."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..protocol.types import (
    StructuredContentValue,
    StructuredToolResult,
    ToolContentBlock,
    ToolResult,
    ToolTextBlock,
)
from ._validation import validate_tool_result

_logger = logging.getLogger("zeta.tools.registry")


class _BoundedText:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._parts: list[str] = []
        self._length = 0
        self._full_size = 0
        self._has_line = False
        self.truncated = False

    @property
    def retained_chars(self) -> int:
        return self._length

    @property
    def full_size(self) -> int:
        return self._full_size

    def append(self, value: str) -> None:
        self._full_size += len(value.encode("utf-8"))
        remaining = self.limit - self._length
        if remaining > 0:
            retained = value[:remaining]
            self._parts.append(retained)
            self._length += len(retained)
        if len(value) > remaining:
            self.truncated = True

    def append_captured(self, value: str, full_size: int) -> None:
        self.append(value)
        self._full_size += max(0, full_size - len(value.encode("utf-8")))

    def begin_line(self) -> None:
        if self._has_line:
            self.append("\n")
        self._has_line = True

    def append_line(self, value: str) -> None:
        self.begin_line()
        self.append(value)

    def render(self, *, full_size: int | None = None) -> ToolTextBlock:
        return text_block(
            "".join(self._parts),
            full_size=self.full_size if full_size is None else full_size,
        )


def text_block(
    text: str,
    *,
    cap: int | None = None,
    full_size: int | None = None,
) -> ToolTextBlock:
    """Build a text block and expose any output cap to the caller."""

    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("text block cap must be a positive integer")
    if full_size is not None and (type(full_size) is not int or full_size < 0):
        raise ValueError("text block full_size must be a nonnegative integer")
    original_size = len(text.encode("utf-8")) if full_size is None else full_size
    shown = text if cap is None else text[:cap]
    truncated = shown != text or original_size > len(shown.encode("utf-8"))
    return {
        "type": "text",
        "text": shown,
        "truncated": truncated,
        "full_size": original_size,
    }


def _success_result(
    block: ToolTextBlock,
    *,
    structured_content: Mapping[str, StructuredContentValue] | None = None,
) -> StructuredToolResult:
    normalized_content = (
        None if structured_content is None else dict(structured_content)
    )
    return {
        "content": [block],
        "isError": False,
        "structuredContent": normalized_content,
    }


def _error_result(
    message: str,
    *,
    kind: str = "error",
    hint: str = "",
) -> StructuredToolResult:
    return {
        "content": [text_block(message)],
        "isError": True,
        "structuredContent": {
            "error": {"kind": kind, "hint": hint, "message": message},
        },
    }


_ERROR_HINTS: dict[str, str] = {
    "timeout": "increase the timeout or use run_background for long-running work",
    "exit_nonzero": "check stderr; the process ran but exited nonzero",
    "unknown_tool": "call one of the registered tools listed in the schemas",
    "invalid_arguments": "reread the tool schema and retry with correct arguments",
    "denied": "the user denied approval; do not retry without new context",
    "canceled": "the tool call was canceled; retry only if still useful",
    "sandbox_violation": "retarget to a path inside the session cwd",
    "invalid_result": "the tool handler returned a malformed result",
    "error": "",
}
_ERROR_KINDS: frozenset[str] = frozenset(_ERROR_HINTS)


def _extract_error_message(result: StructuredToolResult) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                return text
    return ""


def _infer_error_kind(
    result: StructuredToolResult, structured: Mapping[str, Any]
) -> str:
    if structured.get("timed_out") is True:
        return "timeout"
    exit_code = structured.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return "exit_nonzero"
    message = _extract_error_message(result).lower()
    if not message:
        return "error"
    if message.startswith("tool execution canceled"):
        return "canceled"
    if message.startswith("tool execution denied"):
        return "denied"
    if message.startswith("unknown tool"):
        return "unknown_tool"
    if message.startswith("invalid arguments"):
        return "invalid_arguments"
    if message.startswith("invalid tool "):
        return "invalid_result"
    if "path escaped" in message or "sandbox integrity" in message:
        return "sandbox_violation"
    return "error"


def _apply_error_governance(
    result: StructuredToolResult, tool_name: str
) -> StructuredToolResult:
    """Ensure every tool error carries structuredContent.error = {tool, kind, hint}.

    ``kind`` is normalized against the closed :data:`_ERROR_KINDS` taxonomy;
    unknown values are logged and remapped through :func:`_infer_error_kind`.
    ``tool`` and ``hint`` follow the same fallback pattern: caller-provided
    strings win, and the seam only fills in defaults when the caller left
    the field empty.
    """

    if not result.get("isError"):
        return result
    raw_structured = result.get("structuredContent")
    structured: dict[str, Any] = (
        dict(raw_structured) if isinstance(raw_structured, Mapping) else {}
    )
    existing = structured.get("error")
    error: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    kind = error.get("kind")
    if not isinstance(kind, str) or not kind:
        kind = _infer_error_kind(result, structured)
    elif kind not in _ERROR_KINDS:
        _logger.warning(
            "unknown error kind %r from tool %r; normalizing to inferred kind",
            kind,
            tool_name,
        )
        kind = _infer_error_kind(result, structured)
    hint = error.get("hint")
    if not isinstance(hint, str) or not hint:
        hint = _ERROR_HINTS.get(kind, "")
    existing_tool = error.get("tool")
    if not isinstance(existing_tool, str) or not existing_tool:
        error["tool"] = tool_name
    error["kind"] = kind
    error["hint"] = hint
    if "message" not in error:
        message = _extract_error_message(result)
        if message:
            error["message"] = message
    structured["error"] = error
    return {**result, "structuredContent": structured}


def _legacy_result(result: ToolResult) -> StructuredToolResult:
    if type(result.content) is not str:
        return _error_result("invalid tool result: content", kind="invalid_result")
    blocks = (
        result.content_blocks
        if result.content_blocks is not None
        else [text_block(result.content)]
    )
    try:
        structured_result: dict[str, object] = {
            "content": blocks,
            "isError": result.is_error,
            "structuredContent": None,
        }
        if result.is_canceled:
            structured_result["isCanceled"] = True
        return validate_tool_result(structured_result)
    except ValueError as exc:
        return _error_result(f"invalid tool result: {exc}", kind="invalid_result")


def _normalize_result(
    result: StructuredToolResult,
    max_output_chars: int,
) -> StructuredToolResult:
    content: list[ToolContentBlock] = []
    remaining = max_output_chars
    for block in result["content"]:
        if block["type"] != "text":
            content.append(block)
            continue
        full_size = block["full_size"]
        shown = block["text"][:remaining]
        normalized = text_block(shown, full_size=full_size)
        if "annotations" in block:
            normalized["annotations"] = block["annotations"]
        normalized["truncated"] = block["truncated"] or shown != block["text"]
        if "full_size_chars" in block:
            normalized["full_size_chars"] = block["full_size_chars"]
        if "next_offset" in block:
            normalized["next_offset"] = block["next_offset"]
        remaining -= len(shown)
        content.append(normalized)
    return {**result, "content": content}
