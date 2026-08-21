"""The built-in UTF-8 str_replace file editing tool.

Sandbox
-------
Sandbox is best-effort per peer-tool convention: lexical outside-cwd rejection
plus O_NOFOLLOW anchored walk. Direct-truncate write means a partial-failure
mid-write can leave a corrupt file. TOCTOU races (parent rename, hard-link,
ancestor symlink swaps) are out of scope, tracked as ZETA-22.
"""

from __future__ import annotations

import hashlib
import os
from typing import TypedDict

from ..types import StructuredToolResult
from ._sandbox import _path_from_fd, _path_open_error, open_parent
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

    with open_parent(registry, arguments["path"], create_parents=False) as parent:
        components, parent_fd, fallback_path = parent
        file_descriptor: int | None = None
        try:
            try:
                file_descriptor = os.open(
                    components[-1],
                    os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
            except OSError as open_error:
                raise _path_open_error(fallback_path, open_error) from open_error
            try:
                path = _path_from_fd(file_descriptor)
            except OSError as exc:
                raise ValueError(
                    f"could not resolve path: {fallback_path}: {exc}"
                ) from exc

            with os.fdopen(file_descriptor, "r+b") as handle:
                file_descriptor = None
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

        except (OSError, ValueError):
            if file_descriptor is not None:
                os.close(file_descriptor)
            raise

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
    registry.register(
        "edit",
        lambda arguments, abort_signal: _edit(registry, arguments, abort_signal),
        description=(
            "Replace one unique UTF-8 string in a file. "
            "Relative paths use the session cwd."
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
