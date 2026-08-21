"""The built-in file writing tool."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import NotRequired, TypedDict

from ..types import StructuredToolResult
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


def _resolve_path(registry: ToolRegistry, raw_path: str) -> Path:
    path = registry._path(raw_path).resolve()
    try:
        path.relative_to(registry.cwd)
    except ValueError as exc:
        raise ValueError(f"path outside session cwd: {path}") from exc
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

    path = _resolve_path(registry, arguments["path"])
    parent = path.parent
    if arguments.get("create_parents", False):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"could not create parent directory: {parent}: {exc}"
            ) from exc
    elif not parent.is_dir():
        if not parent.exists():
            raise ValueError(f"parent directory does not exist: {parent}")
        raise ValueError(f"parent is not a directory: {parent}")

    try:
        file_descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        was_created = True
    except FileExistsError:
        try:
            file_descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
        except OSError as exc:
            raise ValueError(f"could not open file: {path}: {exc}") from exc
        was_created = False

    try:
        with os.fdopen(file_descriptor, "wb") as handle:
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
