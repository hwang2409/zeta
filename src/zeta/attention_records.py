"""Atomic session-local storage for user attention requests."""

from __future__ import annotations

import json
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .core.session_files import (
    child_directory,
    open_session_file,
    session_directory,
    write_session_file,
)

_MAX_RECORD_BYTES = 256 * 1024


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
        allowed = set(cls.__dataclass_fields__)
        if not set(value).issubset(allowed):
            raise ValueError("unknown attention record fields")
        normalized = dict(value)
        normalized["options"] = tuple(value.get("options", ()))
        record = cls(**normalized)
        if (
            not isinstance(record.id, str)
            or len(record.id) != 32
            or any(ch not in "0123456789abcdef" for ch in record.id)
            or record.status not in {"open", "resolved"}
            or not isinstance(record.title, str)
            or not record.title
            or not isinstance(record.why, str)
            or not record.why
            or any(
                not isinstance(option, str) or not option for option in record.options
            )
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


class AttentionStore:
    """Own attention records behind one small session-local interface."""

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
            not isinstance(title, str)
            or not title.strip()
            or len(title) > 160
            or not isinstance(why, str)
            or not why.strip()
            or len(why) > 32_000
            or len(options) > 20
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 2_000
                for item in options
            )
            or (
                recommendation is not None
                and (
                    not isinstance(recommendation, str)
                    or not recommendation.strip()
                    or len(recommendation) > 8_000
                )
            )
        ):
            raise ValueError("attention request fields are invalid or too large")
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
        self.replace(record, create_directory=True)
        return record

    def get(self, attention_id: str) -> AttentionRecord:
        if len(attention_id) != 32 or any(
            character not in "0123456789abcdef" for character in attention_id
        ):
            raise ValueError("invalid attention id")
        value = _bounded_json(self.directory / f"{attention_id}.json")
        if not isinstance(value, dict):
            raise TypeError("attention record must be a JSON object")
        return AttentionRecord.from_dict(value)

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

    def replace(
        self, record: AttentionRecord, *, create_directory: bool = False
    ) -> None:
        with (
            session_directory(self.session_dir.parent, record.session_id) as (
                _,
                session_fd,
            ),
            child_directory(
                session_fd, "attention", create=create_directory
            ) as attention_fd,
        ):
            write_session_file(
                attention_fd,
                f"{record.id}.json",
                (
                    json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode(),
            )


def _read_bounded_json_fd(fd: int) -> dict[str, Any] | list[Any]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("attention record must be a regular, unshared file")
    with os.fdopen(os.dup(fd), "rb") as stream:
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("attention record is too large")
    value = json.loads(data)
    if not isinstance(value, (dict, list)):
        raise TypeError("attention record must be a JSON object or array")
    return value


def _bounded_json(path: Path) -> dict[str, Any] | list[Any]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        return _read_bounded_json_fd(fd)
    finally:
        os.close(fd)


def read_bounded_session_json(
    directory_fd: int, name: str
) -> dict[str, Any] | list[Any]:
    """Read one bounded session JSON file through an already-verified directory."""
    fd = open_session_file(directory_fd, name, os.O_RDONLY | os.O_NONBLOCK)
    try:
        return _read_bounded_json_fd(fd)
    finally:
        os.close(fd)
