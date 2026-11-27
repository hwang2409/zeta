"""The open project attention queue and its decision views."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .attention_records import (
    AttentionRecord,
    AttentionStore,
    Broken,
    Loaded,
    state_attention_slots,
)
from .core.session import SessionMetadata, read_session_metadata
from .core.session_files import SessionError
from .project_errors import ProjectRegistryError
from .project_registry import ProjectRegistry
from .runtime.project_registry_state import (
    ProjectRegistryState,
    available_project_registry,
)
from .storage_format import verify


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

    @property
    def is_fyi(self) -> bool:
        return isinstance(self.record, AttentionRecord) and not self.record.requires_action


@dataclass(frozen=True, slots=True)
class ProjectAttentionPage:
    items: tuple[ProjectAttentionItem, ...]
    total: int
    action_required_total: int
    info_total: int
    next_offset: int


def attention_owners(
    home: Path, project_id: str | None, *, unassigned_session: str | None = None,
) -> Iterator[tuple[SessionMetadata, Path]]:
    """Find top-level owners with an attention store, live or dead."""
    sessions = home / "sessions"
    try:
        entries = (sessions / unassigned_session,) if unassigned_session else tuple(sessions.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return
    for session_dir in entries:
        # Most sessions have never created attention; skip their metadata reads.
        if not (session_dir / "attention").is_dir():
            continue
        sid = session_dir.name
        try:
            metadata = read_session_metadata(sessions, sid)
        except (OSError, SessionError, ValueError):
            continue
        owner_project = metadata.project_id
        if metadata.parent_session_id is not None or (
            project_id is not None and owner_project != project_id
        ) or (unassigned_session is not None and owner_project is not None):
            continue
        yield metadata, session_dir


def _is_open_for_owner(record: AttentionRecord, owner: SessionMetadata) -> bool:
    return (
        record.status == "open"
        and record.project_id == owner.project_id
        and record.session_id == owner.session_id
    )


def session_attention_counts(sessions: Path, owner: SessionMetadata, epoch: int) -> tuple[int, int]:
    """Return stored slots and the session's open action count in one store walk."""
    session_dir = sessions / owner.session_id
    store = AttentionStore(session_dir, writer_epoch=epoch)
    stored = action_required = 0
    for attention_id in state_attention_slots(session_dir):
        stored += 1
        if owner.parent_session_id is None:
            state = store.record_state(attention_id)
            if isinstance(state, Loaded) and _is_open_for_owner(state.value, owner) and state.value.requires_action:
                action_required += 1
    return stored, action_required


def open_attention_items(
    home: Path,
    project_id: str | None,
    *,
    unassigned_session: str | None = None,
    epoch: int | None = None,
) -> Iterator[ProjectAttentionItem]:
    """Read validated open records, including dead top-level predecessors.

    Without a project, serve reads all queues; an unassigned interactive session
    instead reads only its own records.
    """
    for metadata, session_dir in attention_owners(home, project_id, unassigned_session=unassigned_session):
        try:
            if epoch is None:
                epoch = verify(home).writer_epoch
            records = AttentionStore(session_dir, writer_epoch=epoch).list()
        except (OSError, ValueError):
            continue
        for record in records:
            if isinstance(record, Broken) or _is_open_for_owner(record, metadata):
                yield ProjectAttentionItem(metadata.session_id, record)


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
    for item in open_attention_items(Path(home), project_id):
        record = item.record
        if isinstance(record, Broken):
            broken += 1
        else:
            actions += record.requires_action
            info += not record.requires_action
        if item.is_fyi == fyi:
            rows.append(item)
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
    """Show the queue for the viewed session's project."""
    home = Path(home)
    try:
        project_id = read_session_metadata(home / "sessions", session_id).project_id
    except (OSError, SessionError, ValueError):
        return ()
    state = registry_state or available_project_registry(home / "projects")
    names = _project_names(state.registry)
    items = (
        _decision_item(item.session_id, item.record, names)
        for item in open_attention_items(home, project_id, unassigned_session=session_id if project_id is None else None)
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
        else f"  {record.id} [{record.kind}] {record.title} [{item.project_name} · {item.session_id[:8]}]"
        for item in items
    )


__all__ = ["DecisionItem", "open_decisions"]
