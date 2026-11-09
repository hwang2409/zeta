"""Open attention decisions for one live session.

The TUI's decisions popup and status-bar count read this projection instead of
opening any ``ConversationStore`` or writing files. It lists only the open
attention records of a session whose runtime lease is still live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attention_records import (
    AttentionRecord,
    AttentionStore,
    Broken,
    observed_epoch_for_session,
)
from .project_errors import ProjectRegistryError
from .project_registry import ProjectRegistry
from .runtime.project_registry_state import (
    ProjectRegistryState,
    available_project_registry,
)
from .session_liveness import session_is_live


@dataclass(frozen=True, slots=True)
class DecisionItem:
    session_id: str
    project_id: str | None
    project_name: str
    record: AttentionRecord | Broken


def _project_names(registry: ProjectRegistry | None) -> dict[str, str]:
    if registry is None:
        return {}
    try:
        projects = registry.list_projects()
    except (OSError, ProjectRegistryError):
        return {}
    return {project.project_id: project.name for project in projects}


def _decision_item(
    session_id: str, record: AttentionRecord | Broken, names: dict[str, str]
) -> DecisionItem:
    if isinstance(record, Broken):
        return DecisionItem(session_id, None, "Unassigned", record)
    return DecisionItem(
        session_id,
        record.project_id,
        names.get(record.project_id or "", "Unassigned"),
        record,
    )


def _decision_sort_key(item: DecisionItem) -> tuple[bool, str]:
    record = item.record
    if isinstance(record, Broken):
        return True, record.path
    return record.kind != "decision", record.created_at


def open_decisions(
    home: Path,
    session_id: str,
    registry_state: ProjectRegistryState | None = None,
) -> tuple[DecisionItem, ...]:
    """Return open attention records owned by one live session.

    Sorted oldest first so the popup is stable and the longest-waiting decision
    sits at the top.
    """

    home = Path(home)
    session_dir = home / "sessions" / session_id
    if not session_dir.is_dir() or not session_is_live(session_dir):
        return ()
    state = registry_state or available_project_registry(home / "projects")
    names = _project_names(state.registry)
    items = (
        _decision_item(session_id, record, names)
        for record in AttentionStore(session_dir, writer_epoch=observed_epoch_for_session(session_dir)).list()
        if isinstance(record, Broken) or record.status == "open"
    )
    return tuple(sorted(items, key=_decision_sort_key))


__all__ = ["DecisionItem", "open_decisions"]
