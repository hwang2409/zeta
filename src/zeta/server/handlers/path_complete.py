"""Controller-only `path_complete` request body."""

from __future__ import annotations

from typing import Any

from ...core.path_completion import complete_paths
from ..protocol import ErrorDescriptor, ProtocolError
from . import validation

MAX_TEXT_BEFORE_CURSOR_BYTES = 4096


async def path_complete(client, request_id, params: dict[str, Any]) -> dict[str, object]:
    store = validation.active(client.server.runtime, params)
    text = params.get("text_before_cursor")
    if not isinstance(text, str):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "text_before_cursor must be a string",
        )
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "text_before_cursor must be valid Unicode",
        ) from exc
    if encoded_size > MAX_TEXT_BEFORE_CURSOR_BYTES:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "text_before_cursor must be at most 4096 UTF-8 bytes",
        )
    base_dir = store.cwd
    rows, truncated = await client.server.control._run_blocking(
        lambda: complete_paths(text, base_dir)
    )
    return {
        "entries": [
            {
                "text": row.text,
                "start_position": row.start_position,
                "display": row.display,
                "kind": row.kind,
            }
            for row in rows
        ],
        "truncated": truncated,
    }
