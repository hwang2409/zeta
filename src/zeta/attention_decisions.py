"""Open attention decisions gathered across live sessions.

The TUI's decisions popup and status-bar count read this projection instead of
opening any ``ConversationStore`` or writing files. It lists only the open
attention records of sessions whose runtime lease is still live, so a finished
orchestrator's stale request never counts as a pending decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attention_records import AttentionRecord, AttentionStore
from .project_registry import ProjectRegistry
from .session_liveness import session_is_live


@dataclass(frozen=True, slots=True)
class DecisionItem:
    session_id: str
    project_id: str | None
    project_name: str
    record: AttentionRecord


def _project_names(home: Path) -> dict[str, str]:
    try:
        projects = ProjectRegistry(home / "projects").list_projects()
    except OSError:
        return {}
    return {project.project_id: project.name for project in projects}


def open_decisions(home: Path) -> tuple[DecisionItem, ...]:
    """Return every open attention record owned by a live session.

    Sorted oldest first so the popup is stable and the longest-waiting decision
    sits at the top.
    """

    home = Path(home)
    names = _project_names(home)
    try:
        session_dirs = tuple((home / "sessions").iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return ()
    items: list[DecisionItem] = []
    for session_dir in session_dirs:
        if not session_dir.is_dir() or not session_is_live(session_dir):
            continue
        for record in AttentionStore(session_dir).list():
            if record.status != "open":
                continue
            items.append(
                DecisionItem(
                    session_id=session_dir.name,
                    project_id=record.project_id,
                    project_name=names.get(record.project_id or "", "Unassigned"),
                    record=record,
                )
            )
    return tuple(sorted(items, key=lambda item: item.record.created_at))


__all__ = ["DecisionItem", "open_decisions"]
