"""Append-only session persistence."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import tempfile
import time
import uuid
import warnings
import weakref
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Self

from ..agent_receipt import encode_json
from ..types import Message, MessageRole, ToolCall, ToolUseContent
from .agent_state import AgentStateMixin, _apply_agent_state, _parse_agent_state
from .checkpoints import (
    CheckpointForkMixin,
    ConversationEntry,
    ConversationIntegrityError,
    _now,
    load_session_json,
)
from .session_files import session_directory
from .todo import TodoItem, parse_todo_items

SCHEMA = "zeta.conversation.v1"
MAX_AGENT_NOTIFICATION_TEXT = 10_000
MAX_PENDING_PROMPT_TEXT = 16_000


class PendingPromptsClosedError(RuntimeError):
    """Raised when a run has already decided to finish and refuses new prompts."""


class PendingPromptCommitTimeoutError(TimeoutError):
    """Raised when a pending-prompt commit misses its pre-write deadline."""


class PendingPromptQueue:
    """Own every durable append, acknowledgement, close, and timeout decision."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = weakref.proxy(store)

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise PendingPromptCommitTimeoutError(
                "pending prompt commit deadline exceeded"
            )

    def append(
        self, text: str, *, deadline: float | None = None
    ) -> ConversationEntry:
        if type(text) is not str or not text.strip():
            raise ValueError("pending prompt text must be a nonempty string")
        if len(text) > MAX_PENDING_PROMPT_TEXT:
            raise ValueError("pending prompt text is too long")
        with self._store._append_lock(deadline=deadline):
            self._store._load()
            self._check_deadline(deadline)
            if self._queue_closed_unlocked():
                raise PendingPromptsClosedError("pending prompt queue is closed")
            entry = self._store._append_row_unlocked(
                "pending_prompt", {"text": text}, deadline=deadline
            )
            return self._store._snapshot_entry(entry)

    def pending(self) -> list[ConversationEntry]:
        with self._store._append_lock():
            self._store._load()
            return self._pending_unlocked()

    def acknowledge(self, prompt_id: str) -> None:
        with self._store._append_lock():
            self._store._load()
            branch = self._store.replay()
            prompts = [entry for entry in branch if entry.type == "pending_prompt"]
            if not any(entry.id == prompt_id for entry in prompts):
                raise ValueError(f"unknown pending prompt: {prompt_id}")
            acknowledged = {
                entry.data["prompt_id"]
                for entry in branch
                if entry.type == "pending_prompt_ack"
            }
            if prompt_id not in acknowledged:
                self._store._append_row_unlocked(
                    "pending_prompt_ack", {"prompt_id": prompt_id}
                )

    def close_if_empty(self) -> list[ConversationEntry]:
        with self._store._append_lock():
            self._store._load()
            pending = self._pending_unlocked()
            if not pending and not self._queue_closed_unlocked():
                self._store._append_row_unlocked("pending_queue_closed", {})
            return pending

    def close(self) -> None:
        with self._store._append_lock():
            self._store._load()
            if not self._queue_closed_unlocked():
                self._store._append_row_unlocked("pending_queue_closed", {})

    def _queue_closed_unlocked(self) -> bool:
        return any(
            entry.type == "pending_queue_closed" for entry in self._store._entries
        )

    def _pending_unlocked(self) -> list[ConversationEntry]:
        branch = self._store.replay()
        acknowledged = {
            entry.data["prompt_id"]
            for entry in branch
            if entry.type == "pending_prompt_ack"
        }
        return [
            self._store._snapshot_entry(entry)
            for entry in branch
            if entry.type == "pending_prompt" and entry.id not in acknowledged
        ]


def _valid_agent_stats(value: object) -> bool:
    if type(value) is not dict:
        return False
    return (
        type(value.get("turns_used")) is int
        and value["turns_used"] >= 0
        and type(value.get("elapsed")) in {int, float}
        and math.isfinite(value["elapsed"])
        and value["elapsed"] >= 0
        and type(value.get("tool_calls")) is int
        and value["tool_calls"] >= 0
        and type(value.get("error")) is bool
        and type(value.get("canceled")) is bool
    )


class ConversationStore(AgentStateMixin, CheckpointForkMixin):
    def __init__(
        self,
        session_dir: str | Path | None = None,
        *,
        session_id: str | None = None,
        cwd: str | Path | None = None,
        bash_cwd: str | Path | None = None,
        _lock_deadline: float | None = None,
        _read_only: bool = False,
        _must_exist: bool = False,
    ) -> None:
        default_home = Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))
        self.root_dir = Path(session_dir or default_home / "sessions")
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
        self._read_only = _read_only
        self._closed = False
        if not _read_only and not _must_exist:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "conversation.jsonl"
        self.state_path = self.session_dir / "session_state.json"
        self.agent_lifecycle_path = self.session_dir / "agent_lifecycle.json"
        self.lock_path = self.session_dir / ".lock"
        self.cwd = str(cwd or Path.cwd())
        self.bash_cwd = str(bash_cwd or self.cwd)
        self._entries: list[ConversationEntry] = []
        self._todo_items: list[TodoItem] = []
        self._todo_revision = 0
        self._todo_dismissed = False
        self._agent_counter = 0
        self._agent_children: dict[str, dict[str, Any]] = {}
        self._agent_parent: dict[str, Any] | None = None
        self._agent_canceled: dict[str, Any] | None = None
        self._agent_lifecycle: dict[str, Any] | None = None
        self._write_deadline: float | None = None
        self.pending_prompt_queue = PendingPromptQueue(self)
        with ExitStack() as lease:
            lease.enter_context(session_directory(self.root_dir, self.session_id))
            # Discovery validates without creating locks/state or repairing the log.
            with nullcontext() if _read_only else self._append_lock(deadline=_lock_deadline):
                self._load()
                self._load_session_state()
            self._release_lease = weakref.finalize(self, lease.pop_all().close)

    def close(self) -> None:
        """Release the activity lease after the caller stops using this store."""
        self._closed = True
        self._release_lease()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _load(self) -> None:
        if not self.path.exists():
            if self._read_only:
                raise ConversationIntegrityError(f"conversation file is missing: {self.path}")
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
                row = load_session_json(line)
            except ConversationIntegrityError as exc:
                # Only a malformed, unterminated tail is recoverable. Depth
                # violations are corrupt documents, never evidence of a torn write.
                if (
                    self._read_only
                    or index != len(lines) - 1
                    or line.endswith(b"\n")
                    or not isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError))
                ):
                    kind = "terminated " if line.endswith(b"\n") else ""
                    raise ConversationIntegrityError(
                        f"invalid {kind}conversation row {index + 1}: {self.path}"
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
            or not cwd
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
            if entry.type == "fork":
                self._validate_fork_entry(entry)

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
        elif not self._read_only and not raw.endswith(b"\n"):
            with self.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    def set_bash_cwd(self, cwd: str | Path) -> None:
        """Persist the shell's current directory outside the append-only log."""

        resolved = str(cwd)
        if not resolved:
            raise ValueError("bash cwd must be a nonempty string")
        with self._append_lock():
            self._load()
            self._write_session_state(resolved, self._todo_items)
            self.bash_cwd = resolved

    def todo_items(self) -> list[TodoItem]:
        """Return a detached snapshot of the current session todo list."""

        return [dict(item) for item in self._todo_items]

    @property
    def todo_revision(self) -> int:
        """Return the in-process revision of the todo list."""

        return self._todo_revision

    @property
    def todo_dismissed(self) -> bool:
        """Return whether the terminal todo receipt was dismissed."""

        return self._todo_dismissed

    def dismiss_todo(self) -> None:
        """Persist dismissal of the terminal todo receipt."""

        with self._append_lock():
            self._load()
            self._todo_dismissed = True
            self._write_session_state(self.bash_cwd, self._todo_items)

    def set_todo_items(self, items: object) -> None:
        """Replace the session todo list in one atomic state-file update."""

        normalized = parse_todo_items(items)
        with self._append_lock():
            self._load()
            self._todo_items = [dict(item) for item in normalized]
            self._todo_revision += 1
            self._todo_dismissed = False
            self._write_session_state(self.bash_cwd, normalized)

    def _load_session_state(self) -> None:
        if self.state_path.exists():
            value = load_session_json(self.state_path)
        else:
            if not self._read_only:
                self._write_session_state(self.cwd, ())
            value = {"bash_cwd": self.cwd}
        bash_cwd = value.get("bash_cwd") if isinstance(value, dict) else None
        if type(bash_cwd) is not str or not bash_cwd:
            raise ConversationIntegrityError(
                f"session state bash cwd is invalid: {self.state_path}"
            )
        try:
            todo_items = parse_todo_items(value.get("todo_items", []))
        except ValueError as exc:
            raise ConversationIntegrityError(
                f"session state todo list is invalid: {self.state_path}"
            ) from exc
        agent_state = _parse_agent_state(value, self.state_path)
        self.bash_cwd = bash_cwd
        self._todo_items = todo_items
        todo_dismissed = value.get("todo_dismissed", False)
        if type(todo_dismissed) is not bool:
            raise ConversationIntegrityError(
                f"session state todo dismissal is invalid: {self.state_path}"
            )
        self._todo_dismissed = todo_dismissed
        self._agent_counter = agent_state["agent_counter"]
        self._agent_children = agent_state["agent_children"]
        self._agent_parent = agent_state["agent_parent"]
        self._agent_canceled = agent_state["agent_canceled"]
        self._agent_lifecycle = None
        if self.agent_lifecycle_path.exists():
            lifecycle = load_session_json(self.agent_lifecycle_path)
            if type(lifecycle) is not dict:
                raise ConversationIntegrityError(
                    f"agent lifecycle is invalid: {self.agent_lifecycle_path}"
                )
            self._agent_lifecycle = lifecycle

    def _write_session_state(
        self, bash_cwd: str, todo_items: Iterable[TodoItem]
    ) -> None:
        state: dict[str, Any] = {"bash_cwd": bash_cwd}
        normalized_items = [dict(item) for item in todo_items]
        if normalized_items:
            state["todo_items"] = normalized_items
        if self._todo_dismissed:
            state["todo_dismissed"] = True
        _apply_agent_state(
            state,
            agent_counter=self._agent_counter,
            agent_children=self._agent_children,
            agent_parent=self._agent_parent,
            agent_canceled=self._agent_canceled,
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.session_dir,
                prefix=".session_state.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(state, temporary, separators=(",", ":"))
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.state_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _validate_entries(self) -> None:
        ids: set[str] = set()
        approval_requests: dict[str, ConversationEntry] = {}
        approval_resolutions: set[str] = set()
        notifications: set[str] = set()
        notification_acks: set[str] = set()
        pending_prompts: set[str] = set()
        pending_prompt_acks: set[str] = set()
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
            elif entry.type == "notification":
                notifications.add(entry.id)
            elif entry.type == "notification_ack":
                notification_id = entry.data.get("notification_id")
                if notification_id not in notifications:
                    raise ConversationIntegrityError(
                        f"notification acknowledgement is not linked: {notification_id}"
                    )
                if notification_id in notification_acks:
                    raise ConversationIntegrityError(
                        f"duplicate notification acknowledgement: {notification_id}"
                    )
                notification_acks.add(notification_id)
            elif entry.type == "pending_prompt":
                pending_prompts.add(entry.id)
            elif entry.type == "pending_prompt_ack":
                prompt_id = entry.data.get("prompt_id")
                if prompt_id not in pending_prompts:
                    raise ConversationIntegrityError(
                        f"pending prompt acknowledgement is not linked: {prompt_id}"
                    )
                if prompt_id in pending_prompt_acks:
                    raise ConversationIntegrityError(
                        f"duplicate pending prompt acknowledgement: {prompt_id}"
                    )
                pending_prompt_acks.add(prompt_id)
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
                replaces = entry.data.get("replaces", [])
                if type(summary) is not str or not summary.strip():
                    raise ValueError("compaction summary must be a nonempty string")
                if (
                    type(source_start) is not int
                    or type(source_end) is not int
                    or source_start <= 0
                    or source_end < source_start
                ):
                    raise ValueError("compaction source sequence must be integers")
                if (
                    type(replaces) is not list
                    or any(type(entry_id) is not str or not entry_id for entry_id in replaces)
                    or len(replaces) != len(set(replaces))
                ):
                    raise ValueError("compaction replaces must be unique string IDs")
            elif entry.type == "warning":
                if type(entry.data.get("message")) is not str:
                    raise ValueError("warning message must be a string")
            elif entry.type in {"checkpoint", "fork"}:
                self._validate_checkpoint_or_fork_payload(entry)
            elif entry.type == "approval_resolution":
                request_id = entry.data.get("request_id")
                decision = entry.data.get("decision")
                if type(request_id) is not str or not request_id:
                    raise ValueError("approval resolution id must be a nonempty string")
                if decision not in {"allow", "deny", "abort"}:
                    raise ValueError("invalid approval resolution")
            elif entry.type == "notification":
                if (
                    type(entry.data.get("child_instance_id")) is not str
                    or not entry.data["child_instance_id"]
                    or type(entry.data.get("child_session_path")) is not str
                    or not entry.data["child_session_path"]
                    or type(entry.data.get("description")) is not str
                    or not entry.data["description"]
                    or entry.data.get("status") not in {"completed", "error", "canceled"}
                    or type(entry.data.get("text")) is not str
                    or not entry.data["text"]
                    or len(entry.data["text"]) > MAX_AGENT_NOTIFICATION_TEXT
                    or (
                        "stats" in entry.data
                        and not _valid_agent_stats(entry.data["stats"])
                    )
                ):
                    raise ValueError("invalid agent notification")
            elif entry.type == "notification_ack":
                notification_id = entry.data.get("notification_id")
                if type(notification_id) is not str or not notification_id:
                    raise ValueError("notification id must be a nonempty string")
            elif entry.type == "pending_prompt":
                text = entry.data.get("text")
                if type(text) is not str or not text:
                    raise ValueError("pending prompt text must be a nonempty string")
                if len(text) > MAX_PENDING_PROMPT_TEXT:
                    raise ValueError("pending prompt text is too long")
            elif entry.type == "pending_prompt_ack":
                prompt_id = entry.data.get("prompt_id")
                if type(prompt_id) is not str or not prompt_id:
                    raise ValueError("pending prompt id must be a nonempty string")
            elif entry.type == "pending_queue_closed":
                if entry.data != {}:
                    raise ValueError("pending queue closed marker takes no fields")
            else:
                raise ValueError(f"unsupported conversation entry type: {entry.type}")
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationIntegrityError(
                f"invalid payload for conversation entry {entry.id}"
            ) from exc

    def _write_line(self, row: dict[str, Any]) -> None:
        encoded = encode_json(row) + b"\n"
        if (
            self._write_deadline is not None
            and time.monotonic() >= self._write_deadline
        ):
            raise PendingPromptCommitTimeoutError(
                "pending prompt commit deadline exceeded"
            )
        self._write_bytes(encoded)

    def _write_bytes(self, line: bytes) -> None:
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def _append_lock(self, *, deadline: float | None = None) -> Iterator[None]:
        if self._closed:
            raise ValueError("session store is closed")
        with self.lock_path.open("a+") as handle:
            if deadline is None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            else:
                while True:
                    if time.monotonic() >= deadline:
                        raise PendingPromptCommitTimeoutError(
                            "pending prompt commit deadline exceeded"
                        )
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        time.sleep(max(0, min(0.01, deadline - time.monotonic())))
                    else:
                        break
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
        *,
        deadline: float | None = None,
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
        previous_deadline = self._write_deadline
        self._write_deadline = deadline
        try:
            self._write_line(entry.to_dict())
        finally:
            self._write_deadline = previous_deadline
        self._entries.append(entry)
        return entry

    def append_message(self, message: Message, *, parent_id: str | None = None) -> ConversationEntry:
        return self._append_row("message", {"message": message.to_dict()}, parent_id)

    def append_agent_notification(
        self,
        child_instance_id: str,
        *,
        child_session_path: str,
        description: str,
        status: str,
        text: str,
        stats: dict[str, Any] | None = None,
    ) -> ConversationEntry:
        """Persist one bounded background-child completion notification."""

        if (
            not child_instance_id
            or not child_session_path
            or not description
            or status not in {"completed", "error", "canceled"}
            or not text
        ):
            raise ValueError("invalid agent notification")
        if stats is not None and not _valid_agent_stats(stats):
            raise ValueError("invalid agent notification stats")
        data: dict[str, Any] = {
            "child_instance_id": child_instance_id,
            "child_session_path": child_session_path,
            "description": description,
            "status": status,
            "text": text,
        }
        if stats is not None:
            data["stats"] = dict(stats)
        return self._append_row(
            "notification",
            data,
        )

    def agent_notifications(
        self, *, pending_only: bool = True
    ) -> list[ConversationEntry]:
        """Return durable background-child notifications on the active branch."""

        branch = self.replay()
        acknowledged = {
            entry.data["notification_id"]
            for entry in branch
            if entry.type == "notification_ack"
        }
        return [
            entry
            for entry in branch
            if entry.type == "notification"
            and (not pending_only or entry.id not in acknowledged)
        ]

    def acknowledge_agent_notification(self, notification_id: str) -> None:
        """Durably mark one notification as rendered by the parent UI."""

        notifications = self.agent_notifications(pending_only=False)
        if not any(entry.id == notification_id for entry in notifications):
            raise ValueError(f"unknown agent notification: {notification_id}")
        if any(
            entry.data["notification_id"] == notification_id
            for entry in self.replay()
            if entry.type == "notification_ack"
        ):
            return
        self._append_row("notification_ack", {"notification_id": notification_id})

    def append_pending_prompt(self, text: str) -> ConversationEntry:
        """Queue a follow-up through the pending-prompt queue owner."""

        return self.pending_prompt_queue.append(text)

    def close_pending_queue_if_empty(self) -> list[ConversationEntry]:
        """Return pending prompts or close the queue through its owner."""

        return self.pending_prompt_queue.close_if_empty()

    def close_pending_queue(self) -> None:
        """Close the pending-prompt queue through its owner."""

        self.pending_prompt_queue.close()

    def pending_prompts(self) -> list[ConversationEntry]:
        """Return queued follow-ups through the pending-prompt queue owner."""

        return self.pending_prompt_queue.pending()

    def acknowledge_pending_prompt(self, prompt_id: str) -> None:
        """Acknowledge a follow-up through the pending-prompt queue owner."""

        self.pending_prompt_queue.acknowledge(prompt_id)

    def append_compaction_marker(
        self,
        summary: str,
        source_seq_start: int,
        source_seq_end: int,
        *,
        replaces: Iterable[str] = (),
        parent_id: str | None = None,
        expected_parent_id: str | None = None,
    ) -> ConversationEntry:
        data = {
            "summary": summary,
            "source_seq_start": source_seq_start,
            "source_seq_end": source_seq_end,
            "replaces": list(replaces),
        }
        if expected_parent_id is None:
            return self._append_row("compaction", data, parent_id)
        with self._append_lock():
            self._load()
            branch = self.replay()
            current_parent_id = branch[-1].id if branch else None
            if current_parent_id != expected_parent_id:
                raise ConversationIntegrityError(
                    "active branch changed while appending compaction"
                )
            entry = self._append_row_unlocked(
                "compaction",
                data,
                expected_parent_id,
            )
            return self._snapshot_entry(entry)

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
            branch = self._branch_to_parent(resolved_parent)
            active_child = next(
                (
                    entry
                    for entry in current_branch
                    if entry.type == "message" and entry.parent_id == resolved_parent
                ),
                None,
            )
            target_entries = [active_child] if request_data and active_child else []
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

    def compaction_marker_count(self) -> int:
        return sum(entry.type == "compaction" for entry in self.replay())

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
