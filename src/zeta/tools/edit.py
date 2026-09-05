"""The built-in UTF-8 str_replace file editing tool."""

from __future__ import annotations

import hashlib
import os
from typing import TypedDict

from ..types import StructuredToolResult
from ._sandbox import _path_from_fd, open_target
from .registry import (
    AbortSignal,
    ToolRegistry,
    _error_result,
    _success_result,
    text_block,
)


class EditArguments(TypedDict):
    path: str
    old_string: str
    new_string: str


class EditStructuredContent(TypedDict):
    path: str
    bytes_before: int
    bytes_after: int
    sha256_after: str


def _count_overlapping(content: str, old_string: str) -> int:
    count = 0
    start = 0
    while (match := content.find(old_string, start)) != -1:
        count += 1
        start = match + 1
    return count


async def _edit(
    registry: ToolRegistry,
    arguments: EditArguments,
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    try:
        arguments["old_string"].encode("utf-8")
        arguments["new_string"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("old_string and new_string must be valid UTF-8") from exc

    with open_target(
        registry,
        arguments["path"],
        flags=os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
    ) as (file_descriptor, _resolved_path):
        path = _path_from_fd(file_descriptor)
        try:
            handle = os.fdopen(file_descriptor, "r+b")
        except (OSError, ValueError):
            os.close(file_descriptor)
            raise
        with handle:
            content_bytes = handle.read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"file is not valid UTF-8: {path}") from exc

            match_count = _count_overlapping(content, arguments["old_string"])
            if match_count == 0:
                return _error_result(
                    f"old_string not found in {arguments['path']}"
                )
            if match_count > 1:
                return _error_result(
                    f"old_string found {match_count} times in {arguments['path']}; "
                    "must be unique"
                )

            updated_content = content.replace(
                arguments["old_string"], arguments["new_string"], 1
            )
            updated_bytes = updated_content.encode("utf-8")
            try:
                handle.seek(0)
                handle.truncate()
                handle.write(updated_bytes)
                handle.flush()
            except OSError as exc:
                raise ValueError(f"could not write file: {path}: {exc}") from exc
    structured_content: EditStructuredContent = {
        "path": path,
        "bytes_before": len(content_bytes),
        "bytes_after": len(updated_bytes),
        "sha256_after": hashlib.sha256(updated_bytes).hexdigest(),
    }
    message = f"edited {path}: {len(content_bytes)} bytes → {len(updated_bytes)} bytes"
    return _success_result(
        text_block(message),
        structured_content=structured_content,
    )


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "edit",
        _edit,
        description=(
            "Replace one unique UTF-8 string in a file. "
            "Relative paths use the session cwd; "
            "~ and absolute paths outside the cwd are allowed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    )
