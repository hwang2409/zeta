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


def _ended_row(store: Any, run_id: str, path: Path) -> dict[str, Any] | None:
    """Read the grandchild lifecycle at ``path`` and project a finished row, or
    None when it is unreadable or has not finished yet."""
    try:
        lifecycle, _ = read_child_lifecycle(store, path)
    except (OSError, ValueError):
        return None
    if not isinstance(lifecycle, dict):
        lifecycle = getattr(lifecycle, "value", None)
    if not isinstance(lifecycle, dict) or lifecycle.get("finished_at") is None:
        return None
    session_id, parent_run_id = run_row_identity(run_id)
    return project_child_run_row(
        lifecycle,
        session_id=session_id,
        parent_run_id=parent_run_id,
        accepts_follow_ups=False,
    )


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
    row = _ended_row(store, run_id, Path(path))
    if row is None:
        return None
    return replace(entry, id=f"{entry.id}:child_ended", type="child_ended", data={"run_id": run_id, "_row": row})


def orphan_ended_entries(
    store: Any,
    started_entries: list[Any],
    ended_run_ids: set[str],
    anchor_seq: int,
) -> tuple[list[Any], bool]:
    """Derive a child_ended for each started run whose completion receipt landed
    in an ancestor store (adopt, sealed re-resolve), leaving a closed child's
    history with a child_started and no child_ended.

    Only a closed store (its own lifecycle has ``finished_at``) is derived from;
    a live store's receipt is still in transit and covered by the loop. Reads the
    grandchild's own lifecycle, which whoever received the receipt wrote. Returns
    ``(entries, has_unended)``; ``has_unended`` flags a run whose grandchild has
    not finished yet, so the caller must not cache the projection (a closed
    store's generation does not change when that grandchild ends later)."""
    lifecycle = store.agent_lifecycle()
    if not isinstance(lifecycle, dict) or lifecycle.get("finished_at") is None:
        return [], False
    entries: list[Any] = []
    has_unended = False
    for started in started_entries:
        run_id = started.data["run_id"]
        if run_id in ended_run_ids:
            continue
        leaf = run_id.rsplit(":", 1)[1]
        row = _ended_row(store, run_id, store.session_dir / "agents" / leaf)
        if row is None:
            has_unended = True
            continue
        ended_run_ids.add(run_id)
        entries.append(
            replace(
                started,
                id=f"{started.id}:child_ended",
                type="child_ended",
                seq=anchor_seq,
                data={"run_id": run_id, "_row": row},
            )
        )
    return entries, has_unended


def history_rows(entry: Any) -> list[dict[str, object]]:
    row = entry.data["_row"]
    return [{"id": entry.id, "role": "system", "event": "child_started" if entry.type == "child_run_started" else "child_ended", "session_id": row["session_id"], "run_id": row["run_id"], "sequence": entry.seq, "child": row}]


def baseline_reserve(group: tuple[dict[str, object], ...]) -> int:
    return BASELINE_ROW_RESERVE * sum(row.get("event") == "child_started" for row in group)
