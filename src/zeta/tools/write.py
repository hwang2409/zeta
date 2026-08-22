"""The built-in file writing tool."""

from __future__ import annotations

import hashlib
import os
from typing import NotRequired, TypedDict

from ..types import StructuredToolResult
from ._sandbox import _path_from_fd as _sandbox_path_from_fd
from ._sandbox import open_target
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


def _write_target(
    registry: ToolRegistry,
    raw_path: str,
    content: bytes,
    *,
    create_parents: bool,
    was_created: bool,
) -> str:
    flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    if was_created:
        flags |= os.O_CREAT | os.O_EXCL
    with open_target(
        registry,
        raw_path,
        flags=flags,
        mode=0o666,
        create_parents=create_parents,
    ) as (file_descriptor, _resolved_path):
        path = _path_from_fd(file_descriptor)
        if not was_created:
            os.ftruncate(file_descriptor, 0)
        try:
            handle = os.fdopen(file_descriptor, "wb")
        except (OSError, ValueError):
            os.close(file_descriptor)
            raise
        try:
            with handle:
                handle.write(content)
        except OSError as exc:
            raise ValueError(f"could not write file: {path}: {exc}") from exc
    return path


async def _write(
    registry: ToolRegistry,
    arguments: WriteArguments,
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    try:
        encoded_content = arguments["content"].encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("content must be valid UTF-8") from exc

    try:
        path = _write_target(
            registry,
            arguments["path"],
            encoded_content,
            create_parents=arguments.get("create_parents", False),
            was_created=True,
        )
        was_created = True
    except FileExistsError:
        try:
            path = _write_target(
                registry,
                arguments["path"],
                encoded_content,
                create_parents=arguments.get("create_parents", False),
                was_created=False,
            )
        except ValueError as exc:
            if str(exc).startswith("parent directory does not exist:"):
                raise ValueError("path escaped sandbox") from exc
            raise
        was_created = False

    structured_content: WriteStructuredContent = {
        "path": path,
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
