"""The built-in UTF-8 str_replace file editing tool."""

from __future__ import annotations

import hashlib
import os
from itertools import pairwise
from typing import NotRequired, TypedDict

from ...protocol.types import StructuredToolResult
from .._shared.sandbox import _path_from_fd, open_target
from ..registry import (
    AbortSignal,
    ToolExecutionContext,
    ToolRegistry,
    _error_result,
    _success_result,
    text_block,
)


class EditReplacement(TypedDict):
    old_string: str
    new_string: str


class EditArguments(TypedDict):
    path: str
    old_string: NotRequired[str]
    new_string: NotRequired[str]
    edits: NotRequired[list[EditReplacement]]


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
    *,
    execution_context: ToolExecutionContext,
) -> StructuredToolResult:
    batch = arguments.get("edits")
    if batch is None:
        if "old_string" not in arguments or "new_string" not in arguments:
            return _error_result(
                "invalid arguments: supply old_string and new_string, or edits",
                kind="invalid_arguments",
            )
        replacements = [(arguments["old_string"], arguments["new_string"])]
    else:
        if "old_string" in arguments or "new_string" in arguments:
            return _error_result(
                "invalid arguments: use edits or old_string/new_string, not both",
                kind="invalid_arguments",
            )
        replacements = [(edit["old_string"], edit["new_string"]) for edit in batch]

    try:
        for old_string, new_string in replacements:
            old_string.encode("utf-8")
            new_string.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("old_string and new_string must be valid UTF-8") from exc

    with open_target(
        registry,
        arguments["path"],
        flags=os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        execution_context=execution_context,
    ) as (file_descriptor, _resolved_path):
        path = _path_from_fd(file_descriptor)
        with os.fdopen(file_descriptor, "r+b", closefd=False) as handle:
            content_bytes = handle.read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"file is not valid UTF-8: {path}") from exc

            matches: list[tuple[int, int, str, int]] = []
            for index, (old_string, new_string) in enumerate(replacements):
                label = f"edits {index + 1}: " if batch is not None else ""
                match_count = _count_overlapping(content, old_string)
                if match_count == 0:
                    return _error_result(
                        f"{label}old_string not found in {arguments['path']}"
                    )
                if match_count > 1:
                    return _error_result(
                        f"{label}old_string found {match_count} times in "
                        f"{arguments['path']}; must be unique"
                    )
                start = content.find(old_string)
                matches.append((start, start + len(old_string), new_string, index))

            matches.sort(key=lambda match: match[0])
            for previous, current in pairwise(matches):
                if current[0] < previous[1]:
                    return _error_result(
                        f"edits {previous[3] + 1} and {current[3] + 1} overlap "
                        f"in {arguments['path']}"
                    )

            parts: list[str] = []
            position = 0
            for start, end, new_string, _index in matches:
                parts.extend((content[position:start], new_string))
                position = end
            parts.append(content[position:])
            updated_content = "".join(parts)
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
    if batch is not None:
        message += f" ({len(replacements)} replacements)"
    return _success_result(
        text_block(message),
        structured_content=structured_content,
    )


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "edit",
        _edit,
        approval_subject="path",
        description=(
            "Replace one unique UTF-8 string, or several non-overlapping unique "
            "strings in one file with edits. Match all edits against the original "
            "file and validate them before one write. "
            "Relative paths use the session cwd; "
            "~ and absolute paths outside the cwd are allowed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_string": {
                    "type": "string",
                    "description": "Single replacement: exact text to find once.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Single replacement: text to insert.",
                },
                "edits": {
                    "type": "array",
                    "description": "Multiple replacements in this file, matched against its original content.",
                    "minItems": 1,
                    "maxItems": 50,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string", "minLength": 1},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )
