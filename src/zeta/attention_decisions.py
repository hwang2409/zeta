"""Open attention decisions for one live session.

The TUI's decisions popup and status-bar count read this projection instead of
opening any ``ConversationStore`` or writing files. It lists only the open
attention records of a session whose runtime lease is still live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .attention_records import (
    AttentionRecord,
    AttentionStore,
    Broken,
    observed_epoch_for_session,
)
from .core.session import SessionManager
from .core.session_files import SessionError
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

    @property
    def is_fyi(self) -> bool:
        return isinstance(self.record, AttentionRecord) and not self.record.requires_action


@dataclass(frozen=True, slots=True)
class ProjectAttentionItem:
    session_id: str
    record: AttentionRecord | Broken


@dataclass(frozen=True, slots=True)
class ProjectAttentionPage:
    items: tuple[ProjectAttentionItem, ...]
    total: int
    action_required_total: int
    info_total: int
    next_offset: int


def _top_level_sessions(home: Path, project_id: str) -> tuple[tuple[str, Path], ...]:
    sessions = home / "sessions"
    try:
        entries = tuple(sorted(sessions.iterdir(), key=lambda path: path.name))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return ()
    manager = SessionManager(home)
    result: list[tuple[str, Path]] = []
    try:
        for session_dir in entries:
            if not session_dir.is_dir() or session_dir.name == "agents":
                continue
            try:
                metadata = manager.read_metadata(session_dir.name)
            except (OSError, SessionError, ValueError):
                continue
            if (
                metadata.project_id == project_id
                and metadata.parent_session_id is None
            ):
                result.append((metadata.session_id, session_dir))
    finally:
        manager._storage_lease.close()
    return tuple(result)


def open_project_attention(
    home: Path, project_id: str, *, offset: int = 0, limit: int = 100, fyi: bool = False
) -> ProjectAttentionPage:
    """Read open attention from every top-level session in one project."""
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if type(limit) is not int or limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    rows: list[ProjectAttentionItem] = []
    actions = info = broken = 0
    for session_id, session_dir in _top_level_sessions(Path(home), project_id):
        try:
            epoch = observed_epoch_for_session(session_dir)
            records = AttentionStore(session_dir, writer_epoch=epoch).list()
        except (OSError, ValueError):
            continue
        for record in records:
            if isinstance(record, Broken):
                broken += 1
                if not fyi:
                    rows.append(ProjectAttentionItem(session_id, record))
            elif record.status == "open" and record.project_id == project_id and record.session_id == session_id:
                actions += record.requires_action
                info += not record.requires_action
                if record.requires_action != fyi:
                    rows.append(ProjectAttentionItem(session_id, record))
    rows.sort(
        key=lambda item: (
            item.record.created_at if isinstance(item.record, AttentionRecord) else "",
            item.session_id,
            item.record.id,
        )
    )
    return ProjectAttentionPage(
        tuple(rows[offset : offset + limit]), actions + info + broken, actions, info,
        offset + limit if offset + limit < len(rows) else 0,
    )


def compact_age(created_at: str, *, now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    try:
        seconds = max(0, int((current - datetime.fromisoformat(created_at)).total_seconds()))
    except ValueError:
        return "unknown age"
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


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
    return not record.requires_action, record.created_at


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


def format_open_decisions(items: tuple[DecisionItem, ...], *, fyi: bool | None) -> str:
    if fyi is not None:
        items = tuple(item for item in items if item.is_fyi == fyi)
    if not items:
        return "no open FYI items" if fyi else "no open attention items"
    title = "open FYI items:" if fyi else "open attention items:"
    return title + "\n" + "\n".join(
        f"  {record.id} [broken] {record.path}: {record.reason}"
        if isinstance(record := item.record, Broken)
        else f"  {record.id} [{record.kind}] {record.title} [{item.project_name}]"
        for item in items
    )


__all__ = ["DecisionItem", "open_decisions"]
