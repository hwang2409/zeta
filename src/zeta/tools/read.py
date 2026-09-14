"""The built-in file reading tool."""

from __future__ import annotations

import base64
import codecs
import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from ..core.abort import AbortSignal
from ..types import StructuredToolResult, detect_image_media_type
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


IMAGE_MAX_BYTES = 4 * 1024 * 1024


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


async def _read(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> StructuredToolResult:
    raw_path = arguments["path"]
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    output = _BoundedText(registry.max_output_chars)
    digest = hashlib.sha256()
    try:
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
                file_size = os.fstat(file_descriptor).st_size
                sniffed_type = detect_image_media_type(
                    os.pread(file_descriptor, 12, 0)
                )
                if sniffed_type is not None:
                    data = handle.read(IMAGE_MAX_BYTES + 1)
                    media_type = detect_image_media_type(data, complete=True)
                    if media_type is None:
                        handle.seek(0)
                        return await _read_handle(
                            handle,
                            resolved_path,
                            offset,
                            limit,
                            output,
                            digest,
                            abort_signal,
                        )
                    if "offset" in arguments or "limit" in arguments:
                        raise ValueError(
                            "offset and limit are not supported for image reads"
                        )
                    if file_size > IMAGE_MAX_BYTES:
                        raise ValueError(
                            f"image is {file_size} bytes; cap is "
                            f"{IMAGE_MAX_BYTES} bytes (4 MiB)"
                        )
                    if len(data) > IMAGE_MAX_BYTES:
                        raise ValueError(
                            f"image is {len(data)} bytes; cap is "
                            f"{IMAGE_MAX_BYTES} bytes (4 MiB)"
                        )
                    file_size = len(data)
                    format_name = media_type.removeprefix("image/")
                    filename = resolved_path.name
                    receipt = (
                        f"filename={filename} bytes={file_size} "
                        f"format={format_name}"
                    )
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": receipt,
                                "truncated": False,
                                "full_size": len(receipt.encode("utf-8")),
                            },
                            {
                                "type": "image",
                                "data": base64.b64encode(data).decode("ascii"),
                                "mimeType": media_type,
                                "path": str(resolved_path),
                                "size": file_size,
                            },
                        ],
                        "isError": False,
                        "structuredContent": {
                            "path": str(resolved_path),
                            "filename": filename,
                            "bytes": file_size,
                            "format": format_name,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        },
                    }
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
        approval_subject="path",
        description=(
            "Read a UTF-8 text file or a PNG, JPEG, GIF, or WebP image. "
            "Image reads return the image bytes. Relative paths use the session cwd; "
            "~ and absolute paths outside the cwd are allowed."
        ),
        parallel_safe=True,
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
