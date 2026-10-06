"""The built-in file reading tool."""

from __future__ import annotations

import base64
import codecs
import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from ...core.abort import AbortSignal
from ...media.image_normalization import prepare_image
from ...media.images import detect_image_media_type
from ...protocol.types import StructuredToolResult
from .._shared.sandbox import open_target
from ..registry import (
    ToolExecutionContext,
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
        selected = line_index >= offset and (limit is None or selected_count < limit)
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
        selected = line_index >= offset and (limit is None or selected_count < limit)
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
    *,
    execution_context: ToolExecutionContext,
) -> StructuredToolResult:
    raw_path = arguments["path"]
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    output = _BoundedText(registry.max_output_chars)
    digest = hashlib.sha256()
    try:
        with (
            open_target(
                registry,
                raw_path,
                flags=os.O_RDONLY | os.O_CLOEXEC,
                execution_context=execution_context,
            ) as (file_descriptor, resolved_path),
            os.fdopen(file_descriptor, "rb", closefd=False) as handle,
        ):
            file_size = os.fstat(file_descriptor).st_size
            sniffed_type = detect_image_media_type(os.pread(file_descriptor, 12, 0))
            if sniffed_type is not None:
                if "offset" in arguments or "limit" in arguments:
                    raise ValueError(
                        "offset and limit are not supported for image reads"
                    )
                normalized = await prepare_image(
                    os.dup(file_descriptor),
                    file_size=file_size,
                    policy=registry.image_policy,
                )
                if normalized.utf8_text:
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
                if normalized.error is not None:
                    raise ValueError(normalized.error)
                prepared = normalized.image
                if prepared is None:
                    raise ValueError("image worker returned no result")
                filename = resolved_path.name
                original = {
                    "bytes": prepared.original_bytes,
                    "width": prepared.original_width,
                    "height": prepared.original_height,
                    "format": prepared.original_format,
                }
                sent = {
                    "bytes": len(prepared.data) if prepared.data is not None else 0,
                    "width": prepared.sent_width,
                    "height": prepared.sent_height,
                    "format": prepared.sent_format,
                }
                receipt = (
                    f"filename={filename} original="
                    f"{original['width']}x{original['height']} "
                    f"{original['bytes']}B {original['format']} sent="
                    f"{sent['width']}x{sent['height']} {sent['bytes']}B "
                    f"{sent['format']} original_unchanged=true path={resolved_path}"
                )
                if prepared.note:
                    receipt = f"{receipt} note={prepared.note}"
                content: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": receipt,
                        "truncated": False,
                        "full_size": len(receipt.encode("utf-8")),
                    }
                ]
                if prepared.data is not None:
                    content.append(
                        {
                            "type": "image",
                            "data": base64.b64encode(prepared.data).decode("ascii"),
                            "mimeType": prepared.media_type,
                            "path": str(resolved_path),
                            "size": len(prepared.data),
                        }
                    )
                return {
                    "content": content,
                    "isError": False,
                    "structuredContent": {
                        "path": str(resolved_path),
                        "filename": filename,
                        "bytes": len(prepared.data) if prepared.data is not None else 0,
                        "format": prepared.sent_format,
                        "sha256": prepared.original_sha256,
                        "original": original,
                        "sent": sent,
                        "original_path": str(resolved_path),
                        "original_unchanged": True,
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
