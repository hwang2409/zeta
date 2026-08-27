"""The built-in file reading tool."""

from __future__ import annotations

import codecs
import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from ._sandbox import open_target
from .registry import (
    ToolRegistry,
    _BoundedText,
    _success_result,
    _yield_for_abort,
)


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


async def _read_handle(
    handle: BinaryIO,
    path: Path,
    offset: int,
    limit: int | None,
    output: _BoundedText,
    digest: _Digest,
    abort_signal: AbortSignal,
) -> StructuredToolResult:
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

    await _yield_for_abort(abort_signal)
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
    if last_byte is not None and last_byte != ord("\n"):
        line_count += 1
    return _success_result(
        output.render(),
        structured_content={
            "path": str(path),
            "sha256": digest.hexdigest(),
            "line_count": line_count,
        },
    )


def _is_external_path(path: Path, cwd: Path) -> bool:
    try:
        path.relative_to(cwd)
    except ValueError:
        return True
    return False


async def _read(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> StructuredToolResult:
    raw_path = arguments["path"]
    path = registry._path(raw_path)
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    output = _BoundedText(registry.max_output_chars)
    digest = hashlib.sha256()
    try:
        if _is_external_path(path, registry.cwd):
            if not path.is_file():
                if path.is_dir():
                    raise ValueError(
                        f"{raw_path} is a directory; use bash (e.g. `ls`) to list its contents"
                    )
                raise ValueError(f"not a file: {raw_path}")
            with path.open("rb") as handle:
                return await _read_handle(
                    handle, path, offset, limit, output, digest, abort_signal
                )
        with open_target(
            registry,
            raw_path,
            flags=os.O_RDONLY | os.O_CLOEXEC,
        ) as (file_descriptor, resolved_path):
            try:
                handle = os.fdopen(file_descriptor, "rb")
            except (OSError, ValueError):
                os.close(file_descriptor)
                raise
            with handle:
                return await _read_handle(
                    handle,
                    resolved_path,
                    offset,
                    limit,
                    output,
                    digest,
                    abort_signal,
                )
    except OSError as exc:
        raise ValueError(f"could not read file: {exc}") from exc


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "read",
        _read,
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
