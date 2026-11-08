"""The one child-run row projection serve consumes.

stand-in until z5 child_runs merges; replace with z5's projection
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# the serve methods whose features child_run_events relies on; serve advertises it only
# with every one negotiated. interim set: the z5 merge must make it final by adding
# z5's child run list method (feature child_runs) before this feature ships.
CHILD_RUN_EVENTS_USES = ("session_history_and_subscribe", "child_run_history")
# the complete encoded row is at most this many bytes (JSON, no envelope)
MAX_CHILD_RUN_ROW_BYTES = 10_000
_MAX_TEXT_JSON_BYTES = 4_000
_MAX_SHORT_JSON_BYTES = 256


def _json_size(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def _bounded(value: object, limit: int) -> str:
    """Cut text so its encoded JSON string, with escapes, fits ``limit`` bytes."""
    text = value if type(value) is str else ""
    while _json_size(text) > limit:
        text = text[: max(0, len(text) - max(1, (_json_size(text) - limit) // 6))]
    return text


def project_child_run_row(
    lifecycle: Mapping[str, Any], *, accepts_follow_ups: bool
) -> dict[str, object]:
    """Project one lifecycle snapshot to the shared z5 row, bounded."""
    run_id = str(lifecycle["handle"])
    parent = run_id.rpartition(":")[0]
    row: dict[str, object] = {
        "run_id": run_id,
        "session_id": run_id.split(":", 1)[0],
        "parent_run_id": parent if ":" in parent else None,
        "description": _bounded(lifecycle.get("description"), _MAX_TEXT_JSON_BYTES),
        "agent_type": _bounded(lifecycle.get("agent_type"), _MAX_SHORT_JSON_BYTES),
        "status": _bounded(lifecycle.get("state"), _MAX_SHORT_JSON_BYTES),
        "accepts_follow_ups": accepts_follow_ups,
        "turns_used": int(lifecycle.get("turns_used") or 0),
        "tool_count": int(lifecycle.get("tool_calls") or 0),
        "current_step": _bounded(lifecycle.get("current_step"), _MAX_TEXT_JSON_BYTES),
        "started_at": _bounded(lifecycle.get("started_at"), _MAX_SHORT_JSON_BYTES),
    }
    finished_at = lifecycle.get("finished_at")
    if finished_at is not None:
        row["finished_at"] = _bounded(finished_at, _MAX_SHORT_JSON_BYTES)
    elapsed = lifecycle.get("elapsed")
    row["elapsed_seconds"] = (
        round(float(elapsed), 3) if type(elapsed) in {int, float} else 0.0
    )
    assert _json_size(row) <= MAX_CHILD_RUN_ROW_BYTES
    return row


def started_child_run_row(
    data: Mapping[str, Any], *, accepts_follow_ups: bool
) -> dict[str, object]:
    """The row a durable ``child_run_started`` entry presents."""
    return project_child_run_row(
        {
            "handle": data["run_id"],
            "description": data.get("task"),
            "agent_type": data.get("name") or "general",
            "state": "running",
            "current_step": "starting",
            "started_at": data.get("started_at"),
        },
        accepts_follow_ups=accepts_follow_ups,
    )


def child_run_row(session_dir: Path, session_id: str, run_id: str) -> dict[str, object] | None:
    """Return the row for one run of the top session at ``session_dir``, or None.

    Reads files: call it off the serve loop.
    """
    from .agent_progress import live_child_store, read_agent_lifecycle
    from .store._child_runs import child_run_segments

    try:
        segments = child_run_segments(session_id, run_id)
    except ValueError:
        return None
    path = session_dir
    for segment in segments:
        path = path / "agents" / str(segment)
    try:
        lifecycle = read_agent_lifecycle(path / "agent_lifecycle.json")
    except (OSError, ValueError):
        return None
    if not isinstance(lifecycle, dict) or lifecycle.get("handle") != run_id:
        return None
    store = live_child_store(run_id)
    follow_ups = bool(getattr(store, "accepts_follow_ups", False)) and (
        lifecycle.get("finished_at") is None
    )
    return project_child_run_row(lifecycle, accepts_follow_ups=follow_ups)
