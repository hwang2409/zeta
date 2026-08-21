"""The built-in directory listing tool."""

from __future__ import annotations

import heapq
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from .registry import (
    ToolRegistry,
    _BoundedText,
    _success_result,
    _ToolCanceled,
    _yield_for_abort,
)


MAX_ENTRIES_SHOWN = 1_024


async def _list(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> StructuredToolResult:
    relative_path = arguments.get("path", ".")
    path = registry._path(relative_path)
    if not path.is_dir():
        raise ValueError(f"not a directory: {relative_path}")
    depth = arguments.get("depth", 1)
    output = _BoundedText(registry.max_output_chars)
    entry_count = [0]
    full_size = [0]
    has_full_line = [False]
    await _list_children(
        registry,
        path,
        depth,
        output,
        abort_signal,
        entry_count,
        full_size,
        has_full_line,
    )
    if abort_signal.is_set():
        raise _ToolCanceled()
    return _success_result(
        output.render(full_size=full_size[0]),
        structured_content={
            "root": str(path),
            "entry_count": entry_count[0],
            "full_size": entry_count[0],
            "truncated": entry_count[0] > MAX_ENTRIES_SHOWN,
        },
    )


async def _list_children(
    registry: ToolRegistry,
    path: Path,
    depth: int,
    output: _BoundedText,
    abort_signal: AbortSignal,
    entry_count: list[int],
    full_size: list[int],
    has_full_line: list[bool],
    render_output: bool = True,
) -> None:
    if abort_signal.is_set():
        raise _ToolCanceled()
    try:
        entry_limit = min(
            MAX_ENTRIES_SHOWN,
            max(1, registry.max_output_chars - output.retained_chars),
        )
        entries_seen = 0

        def iter_entries() -> Iterator[Path]:
            nonlocal entries_seen
            for entry in path.iterdir():
                entries_seen += 1
                entry_count[0] += 1
                relative = _relative_entry(registry, entry)
                if has_full_line[0]:
                    full_size[0] += 1
                full_size[0] += len(relative.encode("utf-8"))
                has_full_line[0] = True
                yield entry

        entries = heapq.nsmallest(entry_limit, iter_entries(), key=lambda item: item.name)
    except OSError as exc:
        raise ValueError(f"could not list directory: {exc}") from exc
    if entries_seen > entry_limit:
        output.truncated = True
    for index, entry in enumerate(entries):
        if abort_signal.is_set():
            raise _ToolCanceled()
        if index % 64 == 0:
            await _yield_for_abort(abort_signal)
        relative = _relative_entry(registry, entry)
        if render_output:
            output.append_line(relative)
        if depth > 1 and entry.is_dir() and not entry.is_symlink():
            await _list_children(
                registry,
                entry,
                depth - 1,
                output,
                abort_signal,
                entry_count,
                full_size,
                has_full_line,
                render_output,
            )

    if depth <= 1:
        return
    selected_directories = {
        entry for entry in entries if entry.is_dir() and not entry.is_symlink()
    }
    try:
        for index, entry in enumerate(path.iterdir()):
            if index % 64 == 0:
                await _yield_for_abort(abort_signal)
            if (
                entry not in selected_directories
                and entry.is_dir()
                and not entry.is_symlink()
            ):
                await _list_children(
                    registry,
                    entry,
                    depth - 1,
                    output,
                    abort_signal,
                    entry_count,
                    full_size,
                    has_full_line,
                    False,
                )
    except OSError as exc:
        raise ValueError(f"could not list directory: {exc}") from exc


def _relative_entry(registry: ToolRegistry, entry: Path) -> str:
    try:
        relative = os.fspath(entry.relative_to(registry.cwd))
    except ValueError:
        relative = os.fspath(entry)
    if entry.is_dir() and not entry.is_symlink():
        relative += "/"
    return relative


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
