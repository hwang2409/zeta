"""Open attention decisions for one live session.

The TUI's decisions popup and status-bar count read this projection instead of
opening any ``ConversationStore`` or writing files. It lists only the open
attention records of a session whose runtime lease is still live.
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


def open_decisions(home: Path, session_id: str) -> tuple[DecisionItem, ...]:
    """Return open attention records owned by one live session.

    Sorted oldest first so the popup is stable and the longest-waiting decision
    sits at the top.
    """

    home = Path(home)
    session_dir = home / "sessions" / session_id
    if not session_dir.is_dir() or not session_is_live(session_dir):
        return ()
    names = _project_names(home)
    items = [
        DecisionItem(
            session_id=session_id,
            project_id=record.project_id,
            project_name=names.get(record.project_id or "", "Unassigned"),
            record=record,
        )
        for record in AttentionStore(session_dir).list()
        if record.status == "open"
    ]
    return tuple(sorted(items, key=lambda item: item.record.created_at))


__all__ = ["DecisionItem", "open_decisions"]
