"""durable child lifecycle rows in a parent's serve history."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ...core.child_runs import (
    project_child_run_row,
    read_child_lifecycle,
    run_row_identity,
)
from ...core.store._child_runs import child_run_segments

BASELINE_ROW_RESERVE = 4_096


def started_entry(store: Any, entry: Any, runs: dict[str, str]) -> Any:
    run_id = entry.data["run_id"]
    runs[run_id.rsplit(":", 1)[1]] = run_id
    marker = store.agent_children().get(run_id) or {}
    session_id, parent_run_id = run_row_identity(run_id)
    row = project_child_run_row(
        {
            "handle": run_id,
            "description": entry.data.get("task", ""),
            "agent_type": entry.data.get("name") or "general",
            "state": "running",
            "turns_used": 0,
            "tool_calls": 0,
            "current_step": "starting",
            "started_at": entry.data.get("started_at"),
            "finished_at": None,
            "elapsed": 0.0,
        },
        session_id=session_id,
        parent_run_id=parent_run_id,
        accepts_follow_ups=marker.get("accepts_follow_ups") is True,
    )
    return replace(entry, data={**entry.data, "_row": row})


def _receipt_run(entry: Any, runs: dict[str, str]) -> tuple[str, str | None] | None:
    data = entry.data
    if entry.type == "notification":
        if data.get("kind", "agent_completion") != "agent_completion" or data.get("display_only"):
            return None
        run_id = data.get("child_instance_id")
        path = data.get("child_session_path")
    else:
        result = data["message"].get("tool_result")
        structured = result.get("structured_content") if isinstance(result, dict) else None
        if not isinstance(structured, dict) or structured.get("status") == "running":
            return None
        run_id = structured.get("child_instance_id")
        path = structured.get("child_session_path")
        if type(run_id) is not str:
            run_id = runs.get(Path(path).name) if type(path) is str else None
    if run_id not in runs.values():
        return None
    return run_id, path if type(path) is str else None


def ended_entry(store: Any, entry: Any, runs: dict[str, str]) -> Any | None:
    if entry.type not in {"message", "notification"}:
        return None
    receipt = _receipt_run(entry, runs)
    if receipt is None:
        return None
    run_id, receipt_path = receipt
    marker = store.agent_children().get(run_id) or {}
    path = receipt_path or marker.get("child_session_path")
    if type(path) is not str:
        try:
            path = store.session_dir.joinpath(
                *(component for segment in child_run_segments(store.session_id, run_id) for component in ("agents", str(segment)))
            )
        except (OSError, ValueError):
            return None
    try:
        lifecycle, _ = read_child_lifecycle(store, Path(path))
    except (OSError, ValueError):
        return None
    if not isinstance(lifecycle, dict):
        lifecycle = getattr(lifecycle, "value", None)
    if not isinstance(lifecycle, dict) or lifecycle.get("finished_at") is None:
        return None
    session_id, parent_run_id = run_row_identity(run_id)
    row = project_child_run_row(
        lifecycle,
        session_id=session_id,
        parent_run_id=parent_run_id,
        accepts_follow_ups=False,
    )
    return replace(entry, id=f"{entry.id}:child_ended", type="child_ended", data={"run_id": run_id, "_row": row})


def history_rows(entry: Any) -> list[dict[str, object]]:
    row = entry.data["_row"]
    return [{"id": entry.id, "role": "system", "event": "child_started" if entry.type == "child_run_started" else "child_ended", "session_id": row["session_id"], "run_id": row["run_id"], "sequence": entry.seq, "child": row}]


def baseline_reserve(group: tuple[dict[str, object], ...]) -> int:
    return BASELINE_ROW_RESERVE * sum(row.get("event") == "child_started" for row in group)
