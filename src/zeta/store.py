"""Append-only session persistence."""

from __future__ import annotations

import copy
import json
import os
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import fcntl

from .types import Message, MessageRole, ToolCall, ToolUseContent


SCHEMA = "zeta.conversation.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationIntegrityError(ValueError):
    """Raised when a session file violates the conversation schema."""


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
        seq = value.get("seq")
        entry_id = value.get("id")
        parent_id = value.get("parent_id")
        lane = value.get("lane")
        entry_type = value.get("type")
        data = value.get("data")
        if type(seq) is not int:
            raise ConversationIntegrityError("conversation seq must be an integer")
        if type(entry_id) is not str or not entry_id:
            raise ConversationIntegrityError("conversation id must be a nonempty string")
        if parent_id is not None and (type(parent_id) is not str or not parent_id):
            raise ConversationIntegrityError(
                "conversation parent_id must be null or a nonempty string"
            )
        if type(lane) is not str or lane != "main":
            raise ConversationIntegrityError("conversation lane must be 'main'")
        if type(entry_type) is not str or not entry_type:
            raise ConversationIntegrityError("conversation type must be a nonempty string")
        if type(data) is not dict:
            raise ConversationIntegrityError("conversation data must be an object")
        return cls(
            seq=seq,
            id=entry_id,
            parent_id=parent_id,
            lane=lane,
            type=entry_type,
            data=data,
        )


class ConversationStore:
    def __init__(
        self,
        session_dir: str | Path | None = None,
        *,
        session_id: str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(session_dir or Path.home() / ".zeta" / "sessions")
        self.session_id = uuid.uuid4().hex if session_id is None else session_id
        session_path = Path(self.session_id)
        if (
            not self.session_id
            or self.session_id in {".", ".."}
            or "\x00" in self.session_id
            or session_path.is_absolute()
            or session_path.parts != (self.session_id,)
        ):
            raise ConversationIntegrityError(
                f"session id must be one safe path component: {self.session_id!r}"
            )
        self.session_dir = self.root_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "conversation.jsonl"
        self.lock_path = self.session_dir / ".lock"
        self.cwd = str(cwd or Path.cwd())
        self._entries: list[ConversationEntry] = []
        with self._append_lock():
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._entries = []
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
        torn_offset: int | None = None
        offset = 0
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, RecursionError) as exc:
                if index != len(lines) - 1:
                    raise ConversationIntegrityError(
                        f"invalid conversation row {index + 1}: {self.path}"
                    ) from exc
                if line.endswith(b"\n"):
                    raise ConversationIntegrityError(
                        f"invalid terminated conversation row {index + 1}: {self.path}"
                    ) from exc
                torn_offset = offset
                break
            if not isinstance(row, dict):
                raise ConversationIntegrityError(
                    f"conversation row {index + 1} is not an object: {self.path}"
                )
            valid_rows.append(row)
            offset += len(line)

        if not valid_rows:
            raise ConversationIntegrityError(f"conversation file is empty: {self.path}")
        header = valid_rows[0]
        header_data = header.get("data")
        if (
            type(header.get("type")) is not str
            or header.get("type") != "header"
            or type(header_data) is not dict
            or type(header_data.get("schema")) is not str
            or header_data.get("schema") != SCHEMA
        ):
            raise ConversationIntegrityError(f"unsupported conversation schema: {self.path}")
        header_session_id = header_data.get("session_id")
        cwd = header_data.get("cwd")
        created_at = header_data.get("created_at")
        if (
            type(header_session_id) is not str
            or not header_session_id
            or type(cwd) is not str
            or type(created_at) is not str
        ):
            raise ConversationIntegrityError(
                f"conversation header is incomplete: {self.path}"
            )
        self.cwd = cwd
        if header_session_id != self.session_id:
            raise ConversationIntegrityError(
                f"conversation header session id mismatch: {self.path}"
            )
        try:
            self._entries = [ConversationEntry.from_dict(row) for row in valid_rows[1:]]
        except ConversationIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationIntegrityError(
                f"invalid conversation entry: {self.path}"
            ) from exc
        self._validate_entries()
        for entry in self._entries:
            self._validate_entry_payload(entry)

        if torn_offset is not None:
            with self.path.open("r+b") as handle:
                handle.truncate(torn_offset)
                handle.flush()
                os.fsync(handle.fileno())
            self._append_row_unlocked(
                "warning",
                {"message": "dropped torn final conversation line"},
            )
            warnings.warn(
                f"dropped torn final conversation line from {self.path}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif not raw.endswith(b"\n"):
            with self.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _validate_entries(self) -> None:
        ids: set[str] = set()
        approval_requests: dict[str, ConversationEntry] = {}
        approval_resolutions: set[str] = set()
        by_id = {entry.id: entry for entry in self._entries}
        active_ids: set[str] = set()
        current = self._entries[-1] if self._entries else None
        while current is not None:
            if current.id in active_ids:
                break
            active_ids.add(current.id)
            current = by_id.get(current.parent_id) if current.parent_id else None
        for expected_seq, entry in enumerate(self._entries, start=1):
            if entry.seq != expected_seq:
                raise ConversationIntegrityError(
                    f"non-monotonic conversation sequence at {entry.id}"
                )
            if entry.id in ids:
                raise ConversationIntegrityError(f"duplicate conversation id: {entry.id}")
            if expected_seq > 1 and entry.parent_id is None:
                raise ConversationIntegrityError(
                    f"conversation entry {entry.id} is an orphaned root"
                )
            if entry.parent_id is not None and entry.parent_id not in ids:
                raise ConversationIntegrityError(
                    f"missing prior parent {entry.parent_id} for {entry.id}"
                )
            if entry.id not in active_ids:
                ids.add(entry.id)
                continue
            if entry.type == "message":
                requests = entry.data.get("approval_requests", [])
                if type(requests) is list:
                    for request in requests:
                        if type(request) is not dict:
                            continue
                        request_id = request.get("request_id")
                        if type(request_id) is str and request_id:
                            if request_id in approval_requests:
                                raise ConversationIntegrityError(
                                    f"duplicate approval request: {request_id}"
                                )
                            approval_requests[request_id] = entry
            elif entry.type == "approval_request":
                raise ConversationIntegrityError(
                    "standalone approval requests are not supported"
                )
            elif entry.type == "approval_resolution":
                request_id = entry.data.get("request_id")
                request = (
                    approval_requests.get(request_id)
                    if type(request_id) is str
                    else None
                )
                if request is None or not self._is_ancestor(
                    request.id, entry.id, by_id
                ):
                    raise ConversationIntegrityError(
                        f"approval resolution is not linked to request: {request_id}"
                    )
                if request_id in approval_resolutions:
                    raise ConversationIntegrityError(
                        f"duplicate approval resolution: {request_id}"
                    )
                approval_resolutions.add(request_id)
            ids.add(entry.id)

    @staticmethod
    def _is_ancestor(
        ancestor_id: str,
        descendant_id: str,
        by_id: Mapping[str, ConversationEntry],
    ) -> bool:
        current = by_id.get(descendant_id)
        seen: set[str] = set()
        while current is not None and current.parent_id is not None:
            if current.id in seen:
                return False
            seen.add(current.id)
            if current.parent_id == ancestor_id:
                return True
            current = by_id.get(current.parent_id)
        return False

    def _validate_entry_payload(self, entry: ConversationEntry) -> None:
        try:
            if entry.type == "message":
                message = entry.data.get("message")
                if type(message) is not dict:
                    raise ValueError("message entry payload must contain an object")
                parsed_message = Message.from_dict(message)
                approval_requests = entry.data.get("approval_requests", [])
                if type(approval_requests) is not list:
                    raise ValueError("message approval_requests must be an array")
                request_ids: set[str] = set()
                for request in approval_requests:
                    if type(request) is not dict:
                        raise ValueError("message approval request must be an object")
                    request_id = request.get("request_id")
                    tool_call = request.get("tool_call")
                    if type(request_id) is not str or not request_id:
                        raise ValueError(
                            "approval request id must be a nonempty string"
                        )
                    if request_id in request_ids:
                        raise ValueError(f"duplicate approval request: {request_id}")
                    request_ids.add(request_id)
                    if type(tool_call) is not dict:
                        raise ValueError(
                            "approval request tool_call must be an object"
                        )
                    parsed_tool_call = ToolCall.from_dict(tool_call)
                    anchored_call = next(
                        (
                            block.tool_call
                            for block in parsed_message.content
                            if isinstance(block, ToolUseContent)
                            and block.tool_call.id == parsed_tool_call.id
                        ),
                        None,
                    )
                    if anchored_call != parsed_tool_call:
                        raise ValueError(
                            "approval request must match an anchored tool call"
                        )
            elif entry.type == "compaction":
                summary = entry.data.get("summary")
                source_start = entry.data.get("source_seq_start")
                source_end = entry.data.get("source_seq_end")
                if type(summary) is not str:
                    raise ValueError("compaction summary must be a string")
                if type(source_start) is not int or type(source_end) is not int:
                    raise ValueError("compaction source sequence must be integers")
            elif entry.type == "warning":
                if type(entry.data.get("message")) is not str:
                    raise ValueError("warning message must be a string")
            elif entry.type == "approval_resolution":
                request_id = entry.data.get("request_id")
                decision = entry.data.get("decision")
                if type(request_id) is not str or not request_id:
                    raise ValueError("approval resolution id must be a nonempty string")
                if decision not in {"allow", "deny", "abort"}:
                    raise ValueError("invalid approval resolution")
            else:
                raise ValueError(f"unsupported conversation entry type: {entry.type}")
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationIntegrityError(
                f"invalid payload for conversation entry {entry.id}"
            ) from exc

    def _write_line(self, row: dict[str, Any]) -> None:
        with self.path.open("ab") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def _append_lock(self) -> Iterator[None]:
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append_row(
        self,
        entry_type: str,
        data: dict[str, Any],
        parent_id: str | None = None,
    ) -> ConversationEntry:
        with self._append_lock():
            self._load()
            entry = self._append_row_unlocked(entry_type, data, parent_id)
            return self._snapshot_entry(entry)

    def _append_row_unlocked(
        self,
        entry_type: str,
        data: dict[str, Any],
        parent_id: str | None = None,
    ) -> ConversationEntry:
        prior_ids = {entry.id for entry in self._entries}
        if parent_id is not None and parent_id not in prior_ids:
            raise ConversationIntegrityError(f"missing prior parent {parent_id}")
        entry_id = uuid.uuid4().hex
        if entry_id in prior_ids:
            raise ConversationIntegrityError(f"duplicate conversation id: {entry_id}")
        entry = ConversationEntry(
            seq=(self._entries[-1].seq + 1 if self._entries else 1),
            id=entry_id,
            parent_id=(parent_id if parent_id is not None else (self._entries[-1].id if self._entries else None)),
            lane="main",
            type=entry_type,
            data=copy.deepcopy(data),
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

    def append_message_with_approval_requests(
        self,
        message: Message,
        approval_requests: Iterable[tuple[str, ToolCall]] = (),
        *,
        parent_id: str | None = None,
    ) -> ConversationEntry:
        request_data: list[dict[str, Any]] = []
        request_ids: set[str] = set()
        anchored_calls = {
            block.tool_call.id: block.tool_call
            for block in message.content
            if isinstance(block, ToolUseContent)
        }
        for request_id, tool_call in approval_requests:
            if type(request_id) is not str or not request_id:
                raise ValueError("approval request id must be a nonempty string")
            normalized_tool_call = ToolCall.from_dict(tool_call.to_dict())
            if request_id in request_ids:
                raise ValueError(f"duplicate approval request: {request_id}")
            if anchored_calls.get(request_id) != normalized_tool_call:
                raise ValueError(
                    "approval request must match an anchored tool call"
                )
            request_ids.add(request_id)
            request_data.append(
                {
                    "request_id": request_id,
                    "tool_call": normalized_tool_call.to_dict(),
                }
            )
        data: dict[str, Any] = {"message": message.to_dict()}
        if request_data:
            data["approval_requests"] = request_data
        with self._append_lock():
            self._load()
            current_branch = self.replay()
            resolved_parent = (
                parent_id
                if parent_id is not None
                else (current_branch[-1].id if current_branch else None)
            )
            current_ids = {entry.id for entry in current_branch}
            branch = (
                current_branch
                if resolved_parent in current_ids
                else self._branch_to_parent(resolved_parent)
            )
            target_entries = [
                entry
                for entry in branch
                if entry.type == "message" and entry.parent_id == resolved_parent
            ]
            if request_data:
                for entry in target_entries:
                    if entry.data == data:
                        return self._snapshot_entry(entry)
                request_ids = {request["request_id"] for request in request_data}
                for entry in target_entries:
                    existing_ids = {
                        request["request_id"]
                        for request in entry.data.get("approval_requests", [])
                    }
                    if existing_ids and existing_ids < request_ids:
                        entry = self._append_row_unlocked("message", data, parent_id)
                        return self._snapshot_entry(entry)

            persisted_requests = self._approval_request_entries(branch)
            missing_requests: list[dict[str, Any]] = []
            for request in request_data:
                existing = persisted_requests.get(request["request_id"])
                if existing is None:
                    missing_requests.append(request)
                    continue
                existing_request = next(
                    candidate
                    for candidate in existing.data["approval_requests"]
                    if candidate["request_id"] == request["request_id"]
                )
                if existing_request["tool_call"] != request["tool_call"]:
                    raise ConversationIntegrityError(
                        f"approval request tool call mismatch: {request['request_id']}"
                    )
            append_data = copy.deepcopy(data)
            if missing_requests:
                append_data["approval_requests"] = missing_requests
            else:
                append_data.pop("approval_requests", None)
            entry = self._append_row_unlocked("message", append_data, parent_id)
            return self._snapshot_entry(entry)

    def _branch_to_parent(self, parent_id: str | None) -> list[ConversationEntry]:
        if parent_id is None:
            return []
        by_id = {entry.id: entry for entry in self._entries}
        current = by_id.get(parent_id)
        branch: list[ConversationEntry] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(
                    f"conversation parent cycle at {current.id}"
                )
            seen.add(current.id)
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        return [self._snapshot_entry(entry) for entry in reversed(branch)]

    def append_approval_request(
        self,
        request_id: str,
        tool_call: ToolCall,
        *,
        parent_id: str | None = None,
    ) -> ConversationEntry:
        return self.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(tool_call)]),
            [(request_id, tool_call)],
            parent_id=parent_id,
        )

    def append_approval_resolution(
        self,
        request_id: str,
        decision: str,
        *,
        parent_id: str | None = None,
    ) -> ConversationEntry:
        if type(request_id) is not str or not request_id:
            raise ValueError("approval resolution id must be a nonempty string")
        if decision not in {"allow", "deny", "abort"}:
            raise ValueError(f"invalid approval resolution: {decision}")
        with self._append_lock():
            self._load()
            entry = self._resolve_approval_unlocked(
                request_id,
                decision,
                parent_id,
            )
            if entry is None:
                raise ConversationIntegrityError(
                    f"approval request is already resolved or unknown: {request_id}"
                )
            return self._snapshot_entry(entry)

    def resolve_approval(self, request_id: str, decision: str) -> bool:
        """Resolve one pending request atomically.

        The state check and resolution append share one file lock.
        """

        if type(request_id) is not str or not request_id:
            raise ValueError("approval resolution id must be a nonempty string")
        if decision not in {"allow", "deny", "abort"}:
            raise ValueError(f"invalid approval resolution: {decision}")
        with self._append_lock():
            self._load()
            return self._resolve_approval_unlocked(request_id, decision) is not None

    def _resolve_approval_unlocked(
        self,
        request_id: str,
        decision: str,
        parent_id: str | None = None,
    ) -> ConversationEntry | None:
        branch = self.replay()
        states = self._approval_states_from_branch(branch)
        state = states.get(request_id)
        if state is None or state[1] is not None:
            return None
        request_entry = self._request_entry(branch, request_id)
        if request_entry is None:
            return None
        active_ids = {entry.id for entry in branch}
        resolved_parent = parent_id or branch[-1].id
        if resolved_parent not in active_ids:
            raise ConversationIntegrityError(
                f"approval resolution parent is not on the active branch: {resolved_parent}"
            )
        by_id = {entry.id: entry for entry in branch}
        if resolved_parent != request_entry.id and not self._is_ancestor(
            request_entry.id,
            resolved_parent,
            by_id,
        ):
            raise ConversationIntegrityError(
                f"approval resolution parent is not descended from request: {request_id}"
            )
        return self._append_row_unlocked(
            "approval_resolution",
            {"request_id": request_id, "decision": decision},
            parent_id,
        )

    def approval_states(self) -> dict[str, tuple[ToolCall, str | None]]:
        """Return the latest durable state for each approval request."""

        with self._append_lock():
            self._load()
            return self._approval_states_from_branch(self.replay())

    @staticmethod
    def _request_entry(
        branch: list[ConversationEntry],
        request_id: str,
    ) -> ConversationEntry | None:
        for entry in branch:
            if entry.type == "message":
                for request in entry.data.get("approval_requests", []):
                    if request["request_id"] == request_id:
                        return entry
        return None

    @staticmethod
    def _approval_request_entries(
        entries: Iterable[ConversationEntry],
    ) -> dict[str, ConversationEntry]:
        return {
            request["request_id"]: entry
            for entry in entries
            if entry.type == "message"
            for request in entry.data.get("approval_requests", [])
        }

    @staticmethod
    def _approval_states_from_branch(
        branch: list[ConversationEntry],
    ) -> dict[str, tuple[ToolCall, str | None]]:
        states: dict[str, tuple[ToolCall, str | None]] = {}
        for entry in branch:
            if entry.type == "message":
                requests = entry.data.get("approval_requests", [])
                for request in requests:
                    request_id = request["request_id"]
                    states[request_id] = (
                        ToolCall.from_dict(request["tool_call"]),
                        None,
                    )
            elif entry.type == "approval_resolution":
                request_id = entry.data["request_id"]
                if request_id in states:
                    tool_call, _ = states[request_id]
                    states[request_id] = (tool_call, entry.data["decision"])
        return states

    def pending_approvals(self) -> list[tuple[str, ToolCall]]:
        return [
            (request_id, tool_call)
            for request_id, (tool_call, decision) in self.approval_states().items()
            if decision is None
        ]

    def replay(self) -> list[ConversationEntry]:
        if not self._entries:
            return []
        by_id = {entry.id: entry for entry in self._entries}
        current = self._entries[-1]
        branch: list[ConversationEntry] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(
                    f"conversation parent cycle at {current.id}"
                )
            seen.add(current.id)
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        return [self._snapshot_entry(entry) for entry in reversed(branch)]

    @staticmethod
    def _snapshot_entry(entry: ConversationEntry) -> ConversationEntry:
        return ConversationEntry(
            seq=entry.seq,
            id=entry.id,
            parent_id=entry.parent_id,
            lane=entry.lane,
            type=entry.type,
            data=copy.deepcopy(entry.data),
        )

    def messages(self) -> list[Message]:
        messages: list[Message] = []
        for entry in self.replay():
            if entry.type == "message":
                messages.append(Message.from_dict(entry.data["message"]))
        return messages

    @property
    def entries(self) -> list[ConversationEntry]:
        return [self._snapshot_entry(entry) for entry in self._entries]
