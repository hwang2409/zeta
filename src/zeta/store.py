"""Append-only session persistence."""

from __future__ import annotations

import json
import os
import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .types import Message


SCHEMA = "zeta.conversation.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    seq: int
    id: str
    parent_id: str | None
    lane: str
    type: str
    data: dict[str, Any]

    @property
    def entry_type(self) -> str:
        return self.type

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "id": self.id,
            "parent_id": self.parent_id,
            "lane": self.lane,
            "type": self.type,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConversationEntry:
        return cls(
            seq=int(value["seq"]),
            id=str(value["id"]),
            parent_id=(str(value["parent_id"]) if value.get("parent_id") else None),
            lane=str(value["lane"]),
            type=str(value["type"]),
            data=dict(value.get("data", {})),
        )


class ConversationStore:
    def __init__(
        self,
        session_dir: str | Path | None = None,
        *,
        session_id: str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.session_dir = Path(session_dir or Path.home() / ".zeta" / "sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "conversation.jsonl"
        self.session_id = session_id or uuid.uuid4().hex
        self.cwd = str(cwd or Path.cwd())
        self._entries: list[ConversationEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            header = {
                "schema": SCHEMA,
                "session_id": self.session_id,
                "cwd": self.cwd,
                "created_at": _now(),
            }
            self._write_line({"type": "header", "data": header})
            return

        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        valid_rows: list[dict[str, Any]] = []
        torn = False
        for index, line in enumerate(lines):
            is_final = index == len(lines) - 1
            if is_final and not line.endswith(b"\n"):
                try:
                    valid_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    torn = True
                continue
            valid_rows.append(json.loads(line))

        if not valid_rows:
            raise ValueError(f"conversation file is empty: {self.path}")
        header = valid_rows[0]
        if header.get("type") != "header" or header.get("data", {}).get("schema") != SCHEMA:
            raise ValueError(f"unsupported conversation schema: {self.path}")
        self.session_id = str(header["data"]["session_id"])
        self.cwd = str(header["data"]["cwd"])
        self._entries = [ConversationEntry.from_dict(row) for row in valid_rows[1:]]

        if torn:
            self.path.write_bytes(
                b"".join(
                    json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
                    for row in valid_rows
                )
            )
            self._append_row(
                "warning",
                {"message": "dropped torn final conversation line"},
            )
            warnings.warn(
                f"dropped torn final conversation line from {self.path}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _write_line(self, row: dict[str, Any]) -> None:
        with self.path.open("ab") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _append_row(self, entry_type: str, data: dict[str, Any], parent_id: str | None = None) -> ConversationEntry:
        entry = ConversationEntry(
            seq=(self._entries[-1].seq + 1 if self._entries else 1),
            id=uuid.uuid4().hex,
            parent_id=(parent_id if parent_id is not None else (self._entries[-1].id if self._entries else None)),
            lane="main",
            type=entry_type,
            data=data,
        )
        self._write_line(entry.to_dict())
        self._entries.append(entry)
        return entry

    def append_message(self, message: Message, *, parent_id: str | None = None) -> ConversationEntry:
        return self._append_row("message", {"message": message.to_dict()}, parent_id)

    def append_compaction_marker(
        self,
        summary: str,
        source_seq_start: int,
        source_seq_end: int,
        *,
        parent_id: str | None = None,
    ) -> ConversationEntry:
        return self._append_row(
            "compaction",
            {
                "summary": summary,
                "source_seq_start": source_seq_start,
                "source_seq_end": source_seq_end,
            },
            parent_id,
        )

    def replay(self) -> list[ConversationEntry]:
        if not self._entries:
            return []
        by_id = {entry.id: entry for entry in self._entries}
        current = self._entries[-1]
        branch: list[ConversationEntry] = []
        while current is not None:
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        return list(reversed(branch))

    def messages(self) -> list[Message]:
        messages: list[Message] = []
        for entry in self.replay():
            if entry.type == "message":
                messages.append(Message.from_dict(entry.data["message"]))
        return messages

    @property
    def entries(self) -> list[ConversationEntry]:
        return list(self._entries)
