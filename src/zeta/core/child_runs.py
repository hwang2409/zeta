"""Identity and durable payload rules for child runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_MAX_RUN_ID_BYTES = 256
_MAX_RUN_DEPTH = 64
_RUN_SEGMENT = re.compile(r"[1-9][0-9]*\Z")


def parse_child_run_id(parent_session_id: str, run_id: str) -> tuple[int, str | None]:
    """Validate a public run id and return its leaf number and parent id."""
    if type(parent_session_id) is not str or not parent_session_id:
        raise ValueError("invalid parent session id")
    if type(run_id) is not str or len(run_id.encode("utf-8")) > _MAX_RUN_ID_BYTES:
        raise ValueError("invalid child run id")
    prefix = f"{parent_session_id}:"
    if not run_id.startswith(prefix):
        raise ValueError("child run id has the wrong parent")
    segments = run_id[len(prefix) :].split(":")
    if not 1 <= len(segments) <= _MAX_RUN_DEPTH:
        raise ValueError("invalid child run id depth")
    if any(_RUN_SEGMENT.fullmatch(segment) is None for segment in segments):
        raise ValueError("invalid child run id segment")
    return int(segments[-1]), run_id.rpartition(":")[0] if len(segments) > 1 else None


def validate_child_run_started_data(data: Mapping[str, Any]) -> None:
    """Validate the bounded internal start-row payload."""
    run_id = data.get("run_id")
    parent_run_id = data.get("parent_run_id")
    name = data.get("name")
    task = data.get("task")
    started_at = data.get("started_at")
    if (
        set(data) - {"run_id", "parent_run_id", "name", "task", "started_at"}
        or type(run_id) is not str
        or not run_id
        or len(run_id.encode("utf-8")) > _MAX_RUN_ID_BYTES
        or type(parent_run_id) not in {str, type(None)}
        or parent_run_id == ""
        or type(task) is not str
        or not task
        or len(task) > 10_000
        or type(started_at) is not str
        or not started_at
        or len(started_at) > 128
        or (
            name is not None
            and (type(name) is not str or not name or len(name) > 1_000)
        )
    ):
        raise ValueError("invalid child run started payload")
    parts = run_id.split(":")
    if len(parts) < 2 or any(
        _RUN_SEGMENT.fullmatch(part) is None for part in parts[1:]
    ):
        raise ValueError("invalid child run id")
    expected_parent = ":".join(parts[:-1]) if len(parts) > 2 else None
    if parent_run_id != expected_parent:
        raise ValueError("invalid child run parent")
