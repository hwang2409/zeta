"""Bounded metadata pages of durable child runs, without opening a conversation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from .... import storage_format
from ....core.agent_state import _validate_status_lifecycle
from ....core.checkpoints import load_session_json
from ....core.child_runs import (
    NONTERMINAL_AGENT_STATES,
    TERMINAL_AGENT_STATES,
    child_run_directories,
    child_run_for_features,
    project_agent_status,
    project_child_run_row,
    run_row_identity,
)
from ....core.session import validate_session_id
from ....core.session_files import child_directory, open_session_file, session_root
from ....core.store._child_runs import parse_child_run_id

STATES = NONTERMINAL_AGENT_STATES | TERMINAL_AGENT_STATES
_MAX_CANDIDATES = 128
_MAX_METADATA_BYTES = 256 * 1024
Position = tuple[int, str, str]


class CatalogReadError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_root_id(session_id: str) -> None:
    validate_session_id(session_id)
    if ":" in session_id or len(session_id.encode()) > 253:
        raise ValueError("invalid root session id")


def _roots(fd: int) -> list[tuple[int, str]]:
    roots = []
    with os.scandir(fd) as entries:
        for entry in entries:
            try:
                validate_root_id(entry.name)
                if entry.is_dir(follow_symlinks=False):
                    stamp = entry.stat(follow_symlinks=False).st_mtime_ns
                    roots.append((max(0, stamp), entry.name))
            except (OSError, ValueError):
                continue
    return sorted(roots, key=lambda root: (-root[0], root[1]))


def _metadata(fd: int, name: str) -> Any:
    with os.fdopen(open_session_file(fd, name, os.O_RDONLY), "rb") as source:
        raw = source.read(_MAX_METADATA_BYTES + 1)
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("catalog metadata exceeds read limit")
    return load_session_json(raw)


def _follow_ups(fd: int, run_id: str) -> bool:
    try:
        markers = _metadata(fd, "session_state.json").get("agent_children", {})
        return any(
            isinstance(marker, dict) and marker.get("child_instance_id") == run_id
            and marker.get("accepts_follow_ups") is True for marker in markers.values()
        )
    except (OSError, ValueError, AttributeError):
        return False


def _row(fd: int, parent_fd: int, run_id: str, state: str | None, child_recovery: bool) -> dict[str, Any] | None:
    try:
        lifecycle = _validate_status_lifecycle(_metadata(fd, "agent_lifecycle.json"))
        item = project_agent_status(lifecycle, run_id, notifications=())
        session_id, parent_run_id = run_row_identity(run_id)
        row = project_child_run_row(
            item, session_id=session_id, parent_run_id=parent_run_id,
            accepts_follow_ups=False,
        )
        row = child_run_for_features(row, child_recovery=child_recovery)
        if state is not None and row["status"] != state:
            return None
        row["accepts_follow_ups"] = _follow_ups(parent_fd, run_id)
        return row
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def _children(
    parent_fd: int, root_id: str, prefix: tuple[str, ...], after: tuple[str, ...],
) -> Iterator[tuple[tuple[str, ...], int | None, int]]:
    # A disappeared child still advances the cursor. Its siblings are independent.
    try:
        names = sorted(child_run_directories(parent_fd))
        if not names:
            return
        with child_directory(parent_fd, "agents") as agents_fd:
            for name in names:
                parts = (*prefix, name)
                if parts <= after and parts != after[:len(parts)]:
                    continue
                try:
                    parse_child_run_id(root_id, f"{root_id}:{':'.join(parts)}")
                except ValueError:
                    continue
                try:
                    with child_directory(agents_fd, name) as fd:
                        if parts > after:
                            yield parts, fd, parent_fd
                        yield from _children(fd, root_id, parts, after)
                except OSError:
                    if parts > after:
                        yield parts, None, parent_fd
    except OSError:
        return


def _candidates(
    fd: int, root_id: str | None, after: Position | None,
) -> Iterator[tuple[Position, int | None, int]]:
    start_order = (-after[0], after[1]) if after is not None else None
    for stamp, sid in ([(0, root_id)] if root_id is not None else _roots(fd)):
        order = (-stamp, sid)
        if start_order is not None and order < start_order:
            continue
        if order == start_order and after is not None:
            child_start = tuple(after[2].split(":")) if after[2] else ()
        else:
            yield (stamp, sid, ""), None, fd
            child_start = ()
        try:
            with child_directory(fd, sid) as root_fd:
                for parts, child_fd, parent_fd in _children(root_fd, sid, (), child_start):
                    yield (stamp, sid, ":".join(parts)), child_fd, parent_fd
        except OSError:
            continue


def read_page(
    home: Path, *, root_id: str | None, state: str | None, after: Position | None, limit: int, child_recovery: bool = False,
) -> tuple[list[dict[str, Any]], list[Position], Position | None]:
    """Return public rows, their keys, and a continuation after the last scan position.

    Only names and stats are gathered for ordering. A page reads at most two
    bounded metadata files per candidate, and probes at most 128 roots/runs.
    """
    rows = []
    keys = []
    scanned = after
    try:
        with (
            storage_format.open_reader(home),
            session_root(home / "sessions") as fd,
            closing(_candidates(fd, root_id, after)) as candidates,
        ):
            for index, (key, child_fd, parent_fd) in enumerate(candidates):
                if index == _MAX_CANDIDATES or len(rows) == limit:
                    return rows, keys, scanned
                scanned = key
                if child_fd is not None:
                    row = _row(child_fd, parent_fd, f"{key[1]}:{key[2]}", state, child_recovery)
                    if row is not None:
                        rows.append(row)
                        keys.append(key)
    except FileNotFoundError:
        pass  # A formatted home can have no sessions directory yet.
    except (OSError, storage_format.StorageFormatError):
        raise CatalogReadError("state_unavailable") from None
    return rows, keys, None
