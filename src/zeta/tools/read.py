"""The built-in file reading tool."""

from __future__ import annotations

import codecs
import hashlib
from typing import Any

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from .registry import (
    ToolRegistry,
    _BoundedText,
    _success_result,
    _yield_for_abort,
)


async def _read(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> StructuredToolResult:
    path = registry._path(arguments["path"])
    if not path.is_file():
        raise ValueError(f"not a file: {arguments['path']}")
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    output = _BoundedText(registry.max_output_chars)
    digest = hashlib.sha256()
    line_count = 0
    last_byte: int | None = None
    decoder = codecs.getincrementaldecoder("utf-8")()
    line_index = 0
    selected_count = 0
    line_has_data = False
    line_started = False
    pending_carriage_return = False

    def append_newline() -> None:
        nonlocal line_has_data, line_index, line_started, selected_count
        selected = line_index >= offset and (
            limit is None or selected_count < limit
        )
        if selected:
            if not line_started:
                output.begin_line()
            selected_count += 1
        line_index += 1
        line_has_data = False
        line_started = False

    def append_character(character: str) -> None:
        nonlocal line_has_data, line_started, pending_carriage_return
        if pending_carriage_return:
            pending_carriage_return = False
            append_newline()
            if character == "\n":
                return
        if character == "\r":
            pending_carriage_return = True
            return
        if character == "\n":
            append_newline()
            return
        line_has_data = True
        if line_index < offset or (limit is not None and selected_count >= limit):
            return
        if not line_started:
            output.begin_line()
            line_started = True
        output.append(character)

    try:
        await _yield_for_abort(abort_signal)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    await _yield_for_abort(abort_signal)
                    break
                digest.update(chunk)
                line_count += chunk.count(b"\n")
                last_byte = chunk[-1]
                for character in decoder.decode(chunk, final=False):
                    append_character(character)
                await _yield_for_abort(abort_signal)
            for character in decoder.decode(b"", final=True):
                append_character(character)
            if pending_carriage_return:
                pending_carriage_return = False
                append_newline()
            if line_has_data:
                selected = line_index >= offset and (
                    limit is None or selected_count < limit
                )
                if selected:
                    if not line_started:
                        output.begin_line()
                    selected_count += 1
    except OSError as exc:
        raise ValueError(f"could not read file: {exc}") from exc
    if last_byte is not None and last_byte != ord("\n"):
        line_count += 1
    structured_content = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "line_count": line_count,
    }
    return _success_result(
        output.render(),
        structured_content=structured_content,
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        "read",
        lambda arguments, abort_signal: _read(registry, arguments, abort_signal),
        description="Read a UTF-8 file. Relative paths use the session cwd.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )
