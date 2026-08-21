"""The built-in directory listing tool."""

from __future__ import annotations

import heapq
import os
from pathlib import Path
from typing import Any

from ..core.abort import AbortSignal
from .registry import ToolRegistry, _BoundedText, _ToolCanceled, _yield_for_abort


async def _list(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> str:
    relative_path = arguments.get("path", ".")
    path = registry._path(relative_path)
    if not path.is_dir():
        raise ValueError(f"not a directory: {relative_path}")
    depth = arguments.get("depth", 1)
    output = _BoundedText(registry.max_output_chars)
    await _list_children(registry, path, depth, output, abort_signal)
    if abort_signal.is_set():
        raise _ToolCanceled()
    return output.render()


async def _list_children(
    registry: ToolRegistry,
    path: Path,
    depth: int,
    output: _BoundedText,
    abort_signal: AbortSignal,
) -> bool:
    if abort_signal.is_set():
        raise _ToolCanceled()
    try:
        entries = heapq.nsmallest(
            max(1, registry.max_output_chars - output.retained_chars),
            path.iterdir(),
            key=lambda item: item.name,
        )
    except OSError as exc:
        raise ValueError(f"could not list directory: {exc}") from exc
    for index, entry in enumerate(entries):
        if abort_signal.is_set():
            raise _ToolCanceled()
        if index % 64 == 0:
            await _yield_for_abort(abort_signal)
        try:
            relative = os.fspath(entry.relative_to(registry.cwd))
        except ValueError:
            relative = os.fspath(entry)
        if entry.is_dir() and not entry.is_symlink():
            relative += "/"
        output.append_line(relative)
        if output.truncated:
            return True
        if depth > 1 and entry.is_dir() and not entry.is_symlink():
            if await _list_children(registry, entry, depth - 1, output, abort_signal):
                return True
    return False


def register(registry: ToolRegistry) -> None:
    registry.register(
        "list",
        lambda arguments, abort_signal: _list(registry, arguments, abort_signal),
        description="List a directory. Relative paths use the session cwd.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "depth": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    )
