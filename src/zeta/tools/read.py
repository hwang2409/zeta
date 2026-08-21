"""The built-in file reading tool."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from .registry import (
    ToolRegistry,
    _BoundedText,
    _success_result,
    _yield_for_abort,
)


def _file_metadata(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    line_count = 0
    last_byte: int | None = None
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1]
    if last_byte is not None and last_byte != ord("\n"):
        line_count += 1
    return digest.hexdigest(), line_count


async def _read(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> StructuredToolResult:
    path = registry._path(arguments["path"])
    if not path.is_file():
        raise ValueError(f"not a file: {arguments['path']}")
    sha256, line_count = _file_metadata(path)
    structured_content = {
        "path": str(path),
        "sha256": sha256,
        "line_count": line_count,
    }
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    output = _BoundedText(registry.max_output_chars)
    line_index = 0
    selected_count = 0
    line_has_data = False
    line_started = False
    try:
        with path.open("r", encoding="utf-8", newline=None) as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                for character in chunk:
                    if character == "\n":
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
                        if limit is not None and selected_count >= limit:
                            return _success_result(
                                output.render(),
                                structured_content=structured_content,
                            )
                        continue
                    line_has_data = True
                    if line_index < offset or (
                        limit is not None and selected_count >= limit
                    ):
                        continue
                    if not line_started:
                        output.begin_line()
                        line_started = True
                    output.append(character)
                await _yield_for_abort(abort_signal)
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
