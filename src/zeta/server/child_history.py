"""Durable child lifecycle rows in a parent's serve history.

A parent's history presents each child it started as one ``child_started`` row (its
durable ``child_run_started`` entry) and each terminal receipt it installed as one
``child_ended`` row in the receipt's group. Both carry the immediate-parent row
``sequence`` that live ``child_started``/``child_ended`` events also carry.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.child_run_row import child_run_row, started_child_run_row


def started_entry(store: Any, entry: Any, runs: dict[str, str]) -> Any:
    """Attach the start row to a ``child_run_started`` entry."""
    run_id = entry.data["run_id"]
    runs[run_id.rsplit(":", 1)[1]] = run_id
    marker = store.agent_children().get(run_id) or {}
    row = started_child_run_row(
        entry.data, accepts_follow_ups=marker.get("accepts_follow_ups") is True
    )
    return replace(entry, data={**entry.data, "_row": row})


def _receipt_run(entry: Any, runs: dict[str, str]) -> str | None:
    """The run whose terminal receipt this entry is, or None."""
    data = entry.data
    if entry.type == "notification":
        if data.get("kind", "agent_completion") != "agent_completion" or data.get(
            "display_only"
        ):
            return None
        run_id = data.get("child_instance_id")
        return run_id if run_id in runs.values() else None
    result = data["message"].get("tool_result")
    structured = result.get("structured_content") if isinstance(result, dict) else None
    if not isinstance(structured, dict) or structured.get("status") == "running":
        return None
    run_id = structured.get("child_instance_id")
    if type(run_id) is not str:
        path = structured.get("child_session_path")
        run_id = runs.get(Path(path).name) if type(path) is str else None
    return run_id if run_id in runs.values() else None


def ended_entry(store: Any, entry: Any, runs: dict[str, str]) -> Any | None:
    """A ``child_ended`` entry for a terminal receipt, or None.

    Reads the child's lifecycle file: call it off the serve loop.
    """
    if entry.type not in {"message", "notification"}:
        return None
    run_id = _receipt_run(entry, runs)
    if run_id is None:
        return None
    cache = getattr(store, "_serve_child_end_rows", None)
    if cache is None:
        cache = store._serve_child_end_rows = {}
    row = cache.get(run_id)
    if row is None:
        row = child_run_row(Path(store.session_dir), run_id)
        if row is None or "finished_at" not in row:
            return None
        cache[run_id] = row
    return replace(
        entry,
        id=f"{entry.id}:child_ended",
        type="child_ended",
        data={"run_id": run_id, "_row": row},
    )


def history_rows(entry: Any) -> list[dict[str, object]]:
    row = entry.data["_row"]
    return [
        {
            "id": entry.id,
            "role": "system",
            "event": "child_started" if entry.type == "child_run_started" else "child_ended",
            "session_id": row["session_id"],
            "run_id": row["run_id"],
            "sequence": entry.seq,
            "child": row,
        }
    ]
