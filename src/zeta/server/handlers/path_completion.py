"""Controller-only `path_complete` scanning and result projection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ...core.path_completion import PathCompletion, complete_paths
from ..protocol import ErrorDescriptor, ProtocolError

MAX_TEXT_BEFORE_CURSOR_BYTES = 4096

RunBlocking = Callable[
    [Callable[[], tuple[list[PathCompletion], bool]]],
    Awaitable[tuple[list[PathCompletion], bool]],
]


async def complete(
    params: dict[str, Any],
    *,
    base_dir: Path,
    run_blocking: RunBlocking,
) -> dict[str, object]:
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
    rows, truncated = await run_blocking(lambda: complete_paths(text, base_dir))
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


__all__ = ["complete"]
