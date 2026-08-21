"""The built-in file writing tool.

Sandbox
-------
Layer A rejects lexical paths outside the session cwd. The anchored openat
walk, O_EXCL/O_TRUNC selection, and O_NOFOLLOW checks are best effort.
TOCTOU races, hard-link writes, and ancestor-symlink swaps are out of scope
for ZETA-15 and tracked in ZETA-22.
"""

from __future__ import annotations

import hashlib
import os
from typing import NotRequired, TypedDict

from ..types import StructuredToolResult
from ._sandbox import _path_from_fd as _sandbox_path_from_fd
from ._sandbox import _path_open_error, open_parent
from .registry import AbortSignal, ToolRegistry, _success_result, text_block


class WriteArguments(TypedDict):
    path: str
    content: str
    create_parents: NotRequired[bool]


class WriteStructuredContent(TypedDict):
    path: str
    bytes_written: int
    sha256: str
    was_created: bool
    was_overwritten: bool


def _path_from_fd(file_descriptor: int) -> str:
    return _sandbox_path_from_fd(file_descriptor)


def _open_anchored(
    registry: ToolRegistry,
    raw_path: str,
    create_parents: bool,
) -> tuple[int, bool, str]:
    with open_parent(registry, raw_path, create_parents=create_parents) as parent:
        components, parent_fd, fallback_path = parent
        file_descriptor: int | None = None
        try:
            try:
                file_descriptor = os.open(
                    components[-1],
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                    0o666,
                    dir_fd=parent_fd,
                )
                was_created = True
            except FileExistsError:
                try:
                    file_descriptor = os.open(
                        components[-1],
                        os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=parent_fd,
                    )
                except OSError as open_error:
                    raise _path_open_error(fallback_path, open_error) from open_error
                was_created = False
            except OSError as open_error:
                raise _path_open_error(fallback_path, open_error) from open_error

            actual_path = _path_from_fd(file_descriptor)
            if not was_created:
                os.ftruncate(file_descriptor, 0)
            result = file_descriptor, was_created, actual_path
            file_descriptor = None
            return result
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)


async def _write(
    registry: ToolRegistry,
    arguments: WriteArguments,
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    try:
        encoded_content = arguments["content"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("content must be valid UTF-8") from exc

    file_descriptor, was_created, path = _open_anchored(
        registry,
        arguments["path"],
        arguments.get("create_parents", False),
    )

    try:
        handle = os.fdopen(file_descriptor, "wb")
    except (OSError, ValueError):
        os.close(file_descriptor)
        raise

    try:
        with handle:
            handle.write(encoded_content)
    except OSError as exc:
        raise ValueError(f"could not write file: {path}: {exc}") from exc

    structured_content: WriteStructuredContent = {
        "path": str(path),
        "bytes_written": len(encoded_content),
        "sha256": hashlib.sha256(encoded_content).hexdigest(),
        "was_created": was_created,
        "was_overwritten": not was_created,
    }
    message = f"wrote {len(encoded_content)} bytes to {path}"
    return _success_result(
        text_block(message),
        structured_content=structured_content,
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        "write",
        lambda arguments, abort_signal: _write(registry, arguments, abort_signal),
        description="Write UTF-8 text to a file. Relative paths use the session cwd.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
                "create_parents": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )
