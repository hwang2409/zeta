"""Row-backed content reads for the serve protocol."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from .ergonomics import history_projection
from .protocol import ProtocolError, bounded

MAX_CONTENT_READ = 524_288
_ROW_PREFIX = "r1"


def row_handle(entry_id: str, field: str, block_index: int) -> str:
    token = {
        "message.content": "mc",
        "message.tool_result.content_blocks": "tc",
    }.get(field)
    if token is None:
        raise ValueError("unsupported content row field")
    return f"{_ROW_PREFIX}.{entry_id}.{token}.{block_index}"


def _not_found(handle: object) -> ProtocolError:
    return ProtocolError(-32012, "content not found", {"code": "content_not_found"})


def _decode_handle(handle: object) -> tuple[str, str, int] | None:
    if not isinstance(handle, str):
        return None
    parts = handle.split(".")
    if len(parts) != 4 or parts[0] != _ROW_PREFIX:
        return None
    entry_id, field, index = parts[1:]
    if field not in {"mc", "tc"} or not entry_id or not index.isdigit():
        return None
    return entry_id, field, int(index)


def read(
    store: Any, handle: object, *, offset: object = 0, limit: object = MAX_CONTENT_READ
) -> dict[str, object]:
    locator = _decode_handle(handle)
    if locator is None:
        raise _not_found(handle)
    if type(offset) is not int or type(limit) is not int or offset < 0 or limit < 1:
        raise ProtocolError(
            -32602,
            "invalid content read",
            {"code": "invalid_content_read"},
        )
    limit = min(limit, MAX_CONTENT_READ)
    entry_id, field, block_index = locator
    projection = history_projection(store)
    entry = projection.entry_index.get(entry_id)
    if entry is None:
        raise _not_found(handle)
    message = entry.data.get("message") if entry.type == "message" else None
    if not isinstance(message, dict):
        raise _not_found(handle)
    blocks = (
        message.get("content")
        if field == "mc"
        else (message.get("tool_result") or {}).get("content_blocks")
    )
    if not isinstance(blocks, list) or block_index >= len(blocks):
        raise _not_found(handle)
    block = blocks[block_index]
    if not isinstance(block, dict) or block.get("type") != "image":
        raise _not_found(handle)
    data = block.get("data")
    mime_type = block.get("mimeType")
    if (
        not isinstance(data, str)
        or not data
        or not isinstance(mime_type, str)
        or not mime_type
    ):
        raise _not_found(handle)
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        raise _not_found(handle) from None
    if offset > len(raw):
        raise ProtocolError(
            -32602, "invalid content read", {"code": "invalid_content_read"}
        )
    chunk = raw[offset : offset + limit]
    title = block.get("caption")
    if not isinstance(title, str) or not title:
        title = Path(block.get("path") or "image").name
    return {
        "handle": handle,
        "media_type": mime_type,
        "title": bounded(title, 200),
        "size": len(raw),
        "offset": offset,
        "data": base64.b64encode(chunk).decode("ascii"),
        "next_offset": offset + len(chunk),
        "eof": offset + len(chunk) >= len(raw),
    }
