"""Attention records, read-only panel projection, and discussion forks."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .core.session import SessionManager
from .core.session_files import child_directory, session_directory, write_session_file
from .protocol.types import Message, MessageRole, TextContent

_MAX_RECORD_BYTES = 256 * 1024
_FORK_TOOLS = (
    "read",
    "fetch",
    "websearch",
    "recall_history",
    "project",
    "mcp_discover",
    "resolve_attention",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class AttentionRecord:
    id: str
    created_at: str
    session_id: str
    project_id: str | None
    entry_id: str | None
    entry_seq: int | None
    lane: str
    title: str
    why: str
    options: tuple[str, ...]
    recommendation: str | None
    status: str
    resolved_at: str | None = None
    fork_session_id: str | None = None
    decision: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AttentionRecord:
        allowed = {field for field in cls.__dataclass_fields__}
        if not set(value).issubset(allowed):
            raise ValueError("unknown attention record fields")
        normalized = dict(value)
        normalized["options"] = tuple(value.get("options", ()))
        record = cls(**normalized)
        if (
            len(record.id) != 32
            or any(ch not in "0123456789abcdef" for ch in record.id)
            or record.status not in {"open", "resolved"}
            or not record.title
            or not record.why
        ):
            raise ValueError("invalid attention record")
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "entry_id": self.entry_id,
            "entry_seq": self.entry_seq,
            "lane": self.lane,
            "title": self.title,
            "why": self.why,
            "options": list(self.options),
            "recommendation": self.recommendation,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "fork_session_id": self.fork_session_id,
            "decision": self.decision,
        }


@dataclass(frozen=True, slots=True)
class AttentionFork:
    forked_from_session: str
    forked_at_entry: str
    attention_id: str


class AttentionStore:
    """Own atomic attention records behind one small session-local interface."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.directory = self.session_dir / "attention"

    def request(
        self,
        *,
        session_id: str,
        project_id: str | None,
        entry_id: str | None,
        entry_seq: int | None,
        title: str,
        why: str,
        options: list[str] | tuple[str, ...] = (),
        recommendation: str | None = None,
        lane: str = "orchestrator",
    ) -> AttentionRecord:
        if (
            not title.strip()
            or not why.strip()
            or any(not item.strip() for item in options)
        ):
            raise ValueError("attention title, why, and options must be nonempty")
        record = AttentionRecord(
            id=uuid.uuid4().hex,
            created_at=_now(),
            session_id=session_id,
            project_id=project_id,
            entry_id=entry_id,
            entry_seq=entry_seq,
            lane=lane,
            title=title.strip(),
            why=why.strip(),
            options=tuple(item.strip() for item in options),
            recommendation=recommendation.strip() if recommendation else None,
            status="open",
        )
        with (
            session_directory(self.session_dir.parent, session_id) as (_, session_fd),
            child_directory(session_fd, "attention", create=True) as attention_fd,
        ):
            write_session_file(
                attention_fd,
                f"{record.id}.json",
                (
                    json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode(),
            )
        return record

    def get(self, attention_id: str) -> AttentionRecord:
        path = self.directory / f"{attention_id}.json"
        data = path.read_bytes()
        if len(data) > _MAX_RECORD_BYTES:
            raise ValueError("attention record is too large")
        return AttentionRecord.from_dict(json.loads(data))

    def list(self) -> tuple[AttentionRecord, ...]:
        try:
            paths = tuple(self.directory.iterdir())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return ()
        records: list[AttentionRecord] = []
        for path in paths:
            if path.suffix != ".json":
                continue
            try:
                records.append(self.get(path.stem))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return tuple(sorted(records, key=lambda item: item.created_at))

    def replace(self, record: AttentionRecord) -> None:
        with (
            session_directory(self.session_dir.parent, record.session_id) as (
                _,
                session_fd,
            ),
            child_directory(session_fd, "attention") as attention_fd,
        ):
            write_session_file(
                attention_fd,
                f"{record.id}.json",
                (
                    json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode(),
            )


def read_attention_fork(session_dir: Path) -> AttentionFork | None:
    try:
        value = _bounded_json(Path(session_dir) / "attention_fork.json")
    except FileNotFoundError:
        return None
    if not isinstance(value, dict) or set(value) != {
        "forked_from_session",
        "forked_at_entry",
        "attention_id",
    }:
        raise ValueError("invalid attention fork metadata")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError("invalid attention fork metadata")
    return AttentionFork(**value)


@dataclass(frozen=True, slots=True)
class PanelLane:
    kind: str
    label: str
    status: str
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class PanelSession:
    session_id: str
    name: str
    updated_at: str
    lanes: tuple[PanelLane, ...]
    tasks: tuple[PanelLane, ...]
    attention: tuple[AttentionRecord, ...]


@dataclass(frozen=True, slots=True)
class PanelProject:
    project_id: str | None
    name: str
    sessions: tuple[PanelSession, ...]


@dataclass(frozen=True, slots=True)
class PanelSnapshot:
    projects: tuple[PanelProject, ...]


def session_is_live(session_dir: Path) -> bool:
    """Use the same cooperative directory lease test as project inbox recovery."""
    try:
        fd = os.open(session_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _elapsed(started_at: object, ended_at: object = None) -> float | None:
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


def _recent_attention(record: AttentionRecord) -> bool:
    if record.status == "open" or record.resolved_at is None:
        return True
    try:
        return (
            datetime.now(UTC) - datetime.fromisoformat(record.resolved_at)
        ).total_seconds() <= 600
    except ValueError:
        return False


def _bounded_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("rb") as stream:
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("panel input is too large")
    value = json.loads(data)
    if not isinstance(value, (dict, list)):
        raise TypeError("panel input must be a JSON object or array")
    return value


def _project_names(home: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    try:
        project_dirs = tuple((home / "projects").iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return names
    for directory in project_dirs:
        try:
            record = _bounded_json(directory / "project.json")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(record, dict)
            and record.get("project_id") == directory.name
            and isinstance(record.get("name"), str)
        ):
            names[directory.name] = record["name"]
    return names


def panel_snapshot(home: Path) -> PanelSnapshot:
    """Project live sessions without opening a ConversationStore or writing files."""
    home = Path(home)
    names = _project_names(home)
    grouped: dict[str | None, list[PanelSession]] = {}
    try:
        session_dirs = tuple((home / "sessions").iterdir())
    except FileNotFoundError:
        session_dirs = ()
    for session_dir in session_dirs:
        if not session_dir.is_dir() or not session_is_live(session_dir):
            continue
        try:
            meta = _bounded_json(session_dir / "meta.json")
            state = _bounded_json(session_dir / "session_state.json")
            assert isinstance(meta, dict) and isinstance(state, dict)
        except (OSError, ValueError, AssertionError, json.JSONDecodeError):
            continue
        children = state.get("agent_children", {})
        running_paths = {
            str(marker.get("child_session_path"))
            for marker in (children.values() if isinstance(children, dict) else ())
            if isinstance(marker, dict)
        }
        lane_rows = [
            PanelLane("child", str(marker.get("description", key)), "running")
            for key, marker in (children.items() if isinstance(children, dict) else ())
            if isinstance(marker, dict)
        ]
        try:
            agent_dirs = tuple((session_dir / "agents").iterdir())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            agent_dirs = ()
        for child_dir in agent_dirs:
            if str(child_dir.relative_to(session_dir)) in running_paths:
                continue
            try:
                lifecycle = _bounded_json(child_dir / "agent_lifecycle.json")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if not isinstance(lifecycle, dict):
                continue
            description = lifecycle.get("description")
            status = lifecycle.get("state")
            elapsed = lifecycle.get("elapsed")
            if isinstance(description, str) and isinstance(status, str):
                lane_rows.append(
                    PanelLane(
                        "child",
                        description,
                        status,
                        float(elapsed) if type(elapsed) in {int, float} else None,
                    )
                )
        lanes = tuple(lane_rows)
        tasks: tuple[PanelLane, ...] = ()
        try:
            task_rows = _bounded_json(session_dir / "background_tasks.json")
            if isinstance(task_rows, list):
                tasks = tuple(
                    PanelLane(
                        "task",
                        row["command"],
                        "running" if row.get("running") else "finished",
                        _elapsed(row.get("started_at"), row.get("ended_at")),
                    )
                    for row in task_rows
                    if isinstance(row, dict) and isinstance(row.get("command"), str)
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        project_id = (
            meta.get("project_id") if isinstance(meta.get("project_id"), str) else None
        )
        grouped.setdefault(project_id, []).append(
            PanelSession(
                session_id=session_dir.name,
                name=str(meta.get("name") or session_dir.name[:8]),
                updated_at=str(meta.get("updated_at", "")),
                lanes=lanes,
                tasks=tasks,
                attention=tuple(
                    record
                    for record in AttentionStore(session_dir).list()
                    if _recent_attention(record)
                ),
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


def create_discussion_fork(
    home: Path, source_session_id: str, attention_id: str
) -> str:
    """Create or reuse the read-only discussion fork for one attention record."""
    manager = SessionManager(home)
    source = manager.open(source_session_id, _read_only=True)
    try:
        attention_store = AttentionStore(source.store.session_dir)
        record = attention_store.get(attention_id)
        if record.fork_session_id:
            return record.fork_session_id
        branch = source.store.replay()
        anchor_index = next(
            (
                index
                for index, entry in enumerate(branch)
                if entry.id == record.entry_id
            ),
            None,
        )
        if anchor_index is None:
            raise ValueError("attention anchor is not on the active branch")
        selected = branch[: anchor_index + 1]
        metadata = source.metadata
        fork = manager.create(
            provider=metadata.provider,
            model=metadata.model,
            cwd=metadata.cwd,
            retained_tail=metadata.retained_tail,
            compaction_budget=metadata.compaction_budget,
            compaction=metadata.compaction,
            system_prompt=metadata.system_prompt,
            context_files=metadata.context_files,
            vim_mode=metadata.vim_mode,
            name=f"Discussion: {record.title}",
            project_id=metadata.project_id,
            project_role="session",
            parent_session_id=source_session_id,
            tool_allow=_FORK_TOOLS,
            auto_project=False,
        )
        with session_directory(manager.sessions_dir, fork.store.session_id) as (
            _,
            fork_fd,
        ):
            write_session_file(
                fork_fd,
                "attention_fork.json",
                (
                    json.dumps(
                        {
                            "forked_from_session": source_session_id,
                            "forked_at_entry": record.entry_id,
                            "attention_id": record.id,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode(),
            )
        fork.store.close()
        log_path = fork.store.session_dir / "conversation.jsonl"
        header = log_path.read_bytes().splitlines(keepends=True)[0]
        log = header + b"".join(
            (
                json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            for entry in selected
        )
        log_path.write_bytes(log)
        reopened = manager.open(fork.store.session_id)
        note = (
            f"You are a discussion fork for attention item {record.id}: {record.title}. "
            "The original session keeps running. Read, search, and discuss only. "
            "When the user reaches a decision, send it with resolve_attention."
        )
        reopened.store.append_message(
            Message(
                MessageRole.SYSTEM,
                [TextContent(note)],
                metadata={"origin": "harness", "kind": "attention_fork"},
            )
        )
        reopened.store.close()
        updated = AttentionRecord.from_dict(
            {**record.to_dict(), "fork_session_id": fork.store.session_id}
        )
        attention_store.replace(updated)
        return fork.store.session_id
    finally:
        source.store.close()
