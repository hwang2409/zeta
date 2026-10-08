"""Read-only projection of live orchestrators and their attention requests."""

from __future__ import annotations

import json
import math
import os
import stat
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .attention_forks import attention_decision_message_id
from .attention_records import AttentionRecord, AttentionStore
from .session_liveness import session_is_live

_MAX_PANEL_FILE_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class PanelLane:
    kind: str
    label: str
    status: str
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class PanelAttention:
    record: AttentionRecord
    status: str


@dataclass(frozen=True, slots=True)
class PanelSession:
    session_id: str
    name: str
    updated_at: str
    live: bool
    lanes: tuple[PanelLane, ...]
    tasks: tuple[PanelLane, ...]
    attention: tuple[PanelAttention, ...]


@dataclass(frozen=True, slots=True)
class PanelProject:
    project_id: str | None
    name: str
    sessions: tuple[PanelSession, ...]


@dataclass(frozen=True, slots=True)
class PanelSnapshot:
    projects: tuple[PanelProject, ...]


def _elapsed(started_at: object, ended_at: object = None) -> float | None:
    if type(started_at) in {int, float}:
        end = ended_at if type(ended_at) in {int, float} else time.monotonic()
        if math.isfinite(started_at) and math.isfinite(end):
            return max(0.0, end - started_at)
        return None
    if not isinstance(started_at, str):
        return None
    try:
        end = (
            datetime.fromisoformat(ended_at)
            if isinstance(ended_at, str)
            else datetime.now(UTC)
        )
        return max(0.0, (end - datetime.fromisoformat(started_at)).total_seconds())
    except ValueError:
        return None


def _recent(record: AttentionRecord) -> bool:
    if record.status == "open" or record.resolved_at is None:
        return True
    try:
        return (
            datetime.now(UTC) - datetime.fromisoformat(record.resolved_at)
        ).total_seconds() <= 600
    except ValueError:
        return False


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("panel input must be a regular, unshared file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            data = stream.read(_MAX_PANEL_FILE_BYTES + 1)
        if len(data) > _MAX_PANEL_FILE_BYTES:
            raise ValueError("panel input is too large")
        value = json.loads(data)
        if not isinstance(value, (dict, list)):
            raise TypeError("panel input must be a JSON object or array")
        return value
    finally:
        os.close(fd)


def _project_names(home: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    try:
        project_dirs = tuple((home / "projects").iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return names
    for directory in project_dirs:
        try:
            record = _read_json(directory / "project.json")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(record, dict)
            and record.get("project_id") == directory.name
            and isinstance(record.get("name"), str)
        ):
            names[directory.name] = record["name"]
    return names


def _delivery_pending(home: Path, record: AttentionRecord) -> bool:
    if record.status != "resolved" or record.project_id is None:
        return False
    message_id = attention_decision_message_id(record.id)
    try:
        message = _read_json(
            home
            / "projects"
            / record.project_id
            / "inbox"
            / "new"
            / f"{message_id}.json"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(message, dict)
        and message.get("id") == message_id
        and message.get("to_project") == record.project_id
        and message.get("to_session") == record.session_id
    )


def _attention_rows(
    home: Path, session_dir: Path, *, live: bool
) -> tuple[PanelAttention, ...]:
    rows: list[PanelAttention] = []
    for record in AttentionStore(session_dir).list():
        pending = _delivery_pending(home, record)
        if not live and not pending:
            continue
        if not _recent(record):
            continue
        rows.append(
            PanelAttention(
                record,
                "delivery pending" if pending else record.status,
            )
        )
    return tuple(rows)


def _child_lanes(session_dir: Path, state: dict[str, Any]) -> tuple[PanelLane, ...]:
    children = state.get("agent_children", {})
    running_markers: dict[str, dict[str, Any]] = {}
    for marker in children.values() if isinstance(children, dict) else ():
        if not isinstance(marker, dict):
            continue
        path = marker.get("child_session_path")
        if not isinstance(path, str):
            continue
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(session_dir)
            except ValueError:
                continue
        running_markers[str(candidate)] = marker
    rows: list[PanelLane] = []
    seen_running: set[str] = set()
    try:
        agent_dirs = tuple((session_dir / "agents").iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        agent_dirs = ()
    for child_dir in agent_dirs:
        relative_path = str(child_dir.relative_to(session_dir))
        try:
            lifecycle = _read_json(child_dir / "agent_lifecycle.json")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(lifecycle, dict):
            continue
        marker = running_markers.get(relative_path)
        if marker is not None:
            seen_running.add(relative_path)
            rows.append(
                PanelLane(
                    "child",
                    str(marker.get("description", child_dir.name)),
                    "running",
                    _elapsed(lifecycle.get("started_at")),
                )
            )
            continue
        description = lifecycle.get("description")
        status = lifecycle.get("state")
        elapsed = lifecycle.get("elapsed")
        if isinstance(description, str) and isinstance(status, str):
            rows.append(
                PanelLane(
                    "child",
                    description,
                    status,
                    float(elapsed) if type(elapsed) in {int, float} else None,
                )
            )
    rows.extend(
        PanelLane("child", str(marker.get("description", path)), "running")
        for path, marker in running_markers.items()
        if path not in seen_running
    )
    return tuple(rows)


def _background_tasks(session_dir: Path) -> tuple[PanelLane, ...]:
    try:
        rows = _read_json(session_dir / "background_tasks.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(rows, list):
        return ()
    return tuple(
        PanelLane(
            "task",
            row["command"],
            "running" if row.get("running") else "finished",
            _elapsed(row.get("started_at"), row.get("ended_at")),
        )
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("command"), str)
    )


def panel_snapshot(home: Path) -> PanelSnapshot:
    """Project session state without opening a ConversationStore or writing files."""
    home = Path(home)
    names = _project_names(home)
    grouped: dict[str | None, list[PanelSession]] = {}
    try:
        session_dirs = tuple((home / "sessions").iterdir())
    except FileNotFoundError:
        session_dirs = ()
    for session_dir in session_dirs:
        if not session_dir.is_dir():
            continue
        live = session_is_live(session_dir)
        try:
            meta = _read_json(session_dir / "meta.json")
            if not isinstance(meta, dict):
                continue
            state_value = _read_json(session_dir / "session_state.json") if live else {}
            if not isinstance(state_value, dict):
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        attention = _attention_rows(home, session_dir, live=live)
        if not live and not attention:
            continue
        project_id = (
            meta.get("project_id") if isinstance(meta.get("project_id"), str) else None
        )
        grouped.setdefault(project_id, []).append(
            PanelSession(
                session_id=session_dir.name,
                name=str(meta.get("name") or session_dir.name[:8]),
                updated_at=str(meta.get("updated_at", "")),
                live=live,
                lanes=_child_lanes(session_dir, state_value) if live else (),
                tasks=_background_tasks(session_dir) if live else (),
                attention=attention,
            )
        )
    projects = tuple(
        PanelProject(
            project_id,
            names.get(project_id, "Unassigned"),
            tuple(sorted(sessions, key=lambda item: item.updated_at, reverse=True)),
        )
        for project_id, sessions in sorted(
            grouped.items(), key=lambda item: names.get(item[0], "Unassigned")
        )
    )
    return PanelSnapshot(projects)
