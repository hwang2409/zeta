"""The built-in UTF-8 str_replace file editing tool.

Sandbox
-------
The sandbox is best effort per the peer-tool convention. Lexical path checks,
anchored openat walks, and O_NOFOLLOW checks defer race hardening to ZETA-22.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypedDict

from ..types import StructuredToolResult
from . import write
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


def _open_existing(
    registry: ToolRegistry,
    raw_path: str,
) -> tuple[int, str]:
    components = write._anchored_components(registry, raw_path)
    fallback_path = write._path_for_components(registry, components)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    sandbox_fd = registry._open_cwd()

    parent_fd = sandbox_fd
    file_descriptor: int | None = None
    try:
        for index, component in enumerate(components[:-1]):
            component_path = write._path_for_components(
                registry, components[: index + 1]
            )
            try:
                child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except OSError as exc:
                raise write._path_open_error(component_path, exc) from exc
            if parent_fd != sandbox_fd:
                os.close(parent_fd)
            parent_fd = child_fd

        try:
            file_descriptor = os.open(
                components[-1],
                os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise write._path_open_error(fallback_path, exc) from exc
        try:
            actual_path = write._path_from_fd(file_descriptor)
        except OSError as exc:
            raise ValueError(
                f"could not resolve path: {fallback_path}: {exc}"
            ) from exc
        result = file_descriptor, actual_path
        file_descriptor = None
        return result
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_fd != sandbox_fd:
            os.close(parent_fd)
        os.close(sandbox_fd)


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

    file_descriptor, path = _open_existing(registry, arguments["path"])
    try:
        with os.fdopen(file_descriptor, "r+b") as handle:
            file_descriptor = -1
            content_bytes = handle.read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"file is not valid UTF-8: {path}") from exc

            match_count = content.count(arguments["old_string"])
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
        if file_descriptor >= 0:
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
