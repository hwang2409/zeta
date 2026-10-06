"""Append-only session persistence."""

from __future__ import annotations

import copy
import fcntl
import json  # noqa: F401 - re-exported by the compatibility store facade
import math  # noqa: F401 - re-exported by the compatibility store facade
import os
import time
import uuid
import warnings  # noqa: F401 - re-exported by the compatibility store facade
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Self

from ...agent.receipt import encode_json
from ...protocol.types import Message, MessageRole, ToolCall, ToolResult, ToolUseContent
from ..agent_state import AgentStateMixin, _apply_agent_state, _parse_agent_state
from ..checkpoints import (
    CheckpointForkMixin,
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from ..session_files import (
    child_directory,
    open_session_file,
    read_session_file,
    session_directory,
    session_root,
    write_session_json,
)
from ..todo import TodoItem, parse_todo_items
from ._approval_display import normalize_approval_requests, validated_approval_display
from ._async_writes import AsyncDurableWritesMixin
from ._incremental_validation import IncrementalValidationMixin
from ._log import ConversationLogMixin
from ._notifications import NotificationStateMixin
from ._validation import (
    AGENT_COMPLETION_NOTIFICATION_KIND,
    MAX_AGENT_NOTIFICATION_TEXT,  # noqa: F401 - re-exported by store facade
    TASK_EXITED_NOTIFICATION_KIND,
    validate_agent_notification_data,
)

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

    def append(self, text: str, *, deadline: float | None = None) -> ConversationEntry:
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


class ConversationStore(
    AsyncDurableWritesMixin,
    ConversationLogMixin,
    IncrementalValidationMixin,
    NotificationStateMixin,
    AgentStateMixin,
    CheckpointForkMixin,
):
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
        self.on_persisted_activity: Callable[[int], None] | None = None
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
        self._must_exist = _must_exist
        self._closed = False
        self._closing = False
        self._initialize_async_writes()
        if not _read_only and not _must_exist:
            with session_root(self.root_dir, create=True) as root_fd:
                # Serialize first publication of the permanent append lock. On
                # Darwin, racing O_CREAT | O_NOFOLLOW opens can transiently
                # report ENOENT even though neither constructor removes it.
                fcntl.flock(root_fd, fcntl.LOCK_EX)
                try:
                    with child_directory(
                        root_fd, self.session_id, create=True
                    ) as directory_fd:
                        os.close(
                            open_session_file(
                                directory_fd, ".lock", os.O_RDWR | os.O_CREAT
                            )
                        )
                finally:
                    fcntl.flock(root_fd, fcntl.LOCK_UN)
        self.path = self.session_dir / "conversation.jsonl"
        self.state_path = self.session_dir / "session_state.json"
        self.agent_lifecycle_path = self.session_dir / "agent_lifecycle.json"
        self.lock_path = self.session_dir / ".lock"
        self.cwd = str(cwd or Path.cwd())
        self.bash_cwd = str(bash_cwd or self.cwd)
        self._entries: list[ConversationEntry] = []
        # Task-exit task ids for O(1) append_task_notification dedupe (task ids
        # are unique and exit once, so this mirrors the active-branch scan).
        self._task_notification_ids: set[str] = set()
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
            _, self.directory_fd = lease.enter_context(
                session_directory(self.root_dir, self.session_id)
            )
            # Discovery validates without creating locks/state or repairing the log.
            with (
                nullcontext()
                if _read_only
                else self._append_lock(deadline=_lock_deadline)
            ):
                self._load()
                self._load_session_state()
            self._release_lease = weakref.finalize(self, lease.pop_all().close)

    def close(self) -> None:
        """Drain durable writers, then release this store's activity lease."""
        if self._drain_durable_writes_for_close():
            self.directory_fd = -1
            self._release_lease()

    def refresh(self) -> None:
        """Reload durable state without acquiring ownership of the session."""

        if self._closed:
            raise ConversationIntegrityError("conversation store is closed")
        with nullcontext() if self._read_only else self._append_lock():
            self._load()
            self._load_session_state()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

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
        try:
            value = load_session_json(
                read_session_file(self.directory_fd, "session_state.json")
            )
        except FileNotFoundError:
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
        try:
            lifecycle = load_session_json(
                read_session_file(self.directory_fd, "agent_lifecycle.json")
            )
        except FileNotFoundError:
            pass
        else:
            if type(lifecycle) is not dict:
                raise ConversationIntegrityError(
                    f"agent lifecycle is invalid: {self.agent_lifecycle_path}"
                )
            self._agent_lifecycle = self._sanitize_lifecycle(lifecycle)

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
        write_session_json(self.directory_fd, "session_state.json", state)

    def _validate_entries(self) -> None:
        ids: set[str] = set()
        approval_requests: dict[str, ConversationEntry] = {}
        approval_resolutions: set[str] = set()
        notifications: set[str] = set()
        notification_acks: set[str] = set()
        notification_tui_presentations: set[str] = set()
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
                raise ConversationIntegrityError(
                    f"duplicate conversation id: {entry.id}"
                )
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
            elif entry.type == "notification_tui_presented":
                notification_id = entry.data.get("notification_id")
                if notification_id not in notifications:
                    raise ConversationIntegrityError(
                        f"notification TUI presentation is not linked: {notification_id}"
                    )
                if notification_id in notification_tui_presentations:
                    raise ConversationIntegrityError(
                        f"duplicate notification TUI presentation: {notification_id}"
                    )
                notification_tui_presentations.add(notification_id)
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
                        raise ValueError("approval request tool_call must be an object")
                    parsed_tool_call = ToolCall.from_dict(tool_call)
                    if "approval_display" in request:
                        validated_approval_display(request["approval_display"])
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
                pinned_message = entry.data.get("pinned_message")
                kind = entry.data.get("kind", "summary")
                view = entry.data.get("view")
                telemetry = entry.data.get("telemetry")
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
                    or any(
                        type(entry_id) is not str or not entry_id
                        for entry_id in replaces
                    )
                    or len(replaces) != len(set(replaces))
                ):
                    raise ValueError("compaction replaces must be unique string IDs")
                if pinned_message is not None:
                    if type(pinned_message) is not dict:
                        raise ValueError("compaction pinned message must be an object")
                    pinned = Message.from_dict(pinned_message)
                    if pinned.role is not MessageRole.USER:
                        raise ValueError("compaction pinned message must be a user message")
                if kind not in {"summary", "evict"}:
                    raise ValueError("unknown compaction kind")
                if kind == "evict":
                    if type(view) is not list or not view:
                        raise ValueError("eviction view must be a nonempty array")
                    for item in view:
                        if type(item) is not dict or type(item.get("seq")) is not int:
                            raise ValueError("invalid eviction view item")
                        message = item.get("message")
                        if type(message) is not dict:
                            raise ValueError("invalid eviction view message")
                        Message.from_dict(message)
                    if telemetry is not None and type(telemetry) is not dict:
                        raise ValueError("eviction telemetry must be an object")
                elif view is not None or telemetry is not None:
                    raise ValueError("summary compaction cannot contain an eviction view")
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
                kind = entry.data.get("kind", AGENT_COMPLETION_NOTIFICATION_KIND)
                if kind == TASK_EXITED_NOTIFICATION_KIND and (
                    type(entry.data.get("task_id")) is not str
                    or not entry.data["task_id"]
                    or type(entry.data.get("headline")) is not str
                    or type(entry.data.get("exit_code")) not in {int, type(None)}
                    or type(entry.data.get("output_tail", "")) is not str
                ):
                    raise ValueError("invalid task notification")
                if kind == AGENT_COMPLETION_NOTIFICATION_KIND:
                    validate_agent_notification_data(entry.data)
                # Unknown notification kinds are tolerated for forward compat.
            elif entry.type == "notification_ack":
                notification_id = entry.data.get("notification_id")
                if type(notification_id) is not str or not notification_id:
                    raise ValueError("notification id must be a nonempty string")
            elif entry.type == "notification_tui_presented":
                notification_id = entry.data.get("notification_id")
                if type(notification_id) is not str or not notification_id:
                    raise ValueError("notification id must be a nonempty string")
                if set(entry.data) != {"notification_id"}:
                    raise ValueError("notification marker has unexpected fields")
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

    @contextmanager
    def _append_lock(self, *, deadline: float | None = None) -> Iterator[None]:
        if self._closed:
            raise ValueError("session store is closed")
        with os.fdopen(
            open_session_file(self.directory_fd, ".lock", os.O_RDWR | os.O_CREAT), "r+"
        ) as handle:
            if deadline is None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            else:
                while True:
                    if time.monotonic() >= deadline:
                        raise PendingPromptCommitTimeoutError(
                            "pending prompt commit deadline exceeded"
                        )
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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

    def append_many(
        self, rows: Iterable[tuple[str, dict[str, Any]]]
    ) -> list[ConversationEntry]:
        """Durably append ordered rows with one flock acquisition and one fsync.

        Rows become visible together at the byte-write boundary and are ordered
        exactly as supplied. Each later row is parented to the preceding row.
        """
        materialized = [(entry_type, dict(data)) for entry_type, data in rows]
        if not materialized:
            return []
        with self._append_lock():
            self._load()
            return [
                self._snapshot_entry(entry)
                for entry in self._append_many_unlocked(materialized)
            ]

    def _append_many_unlocked(
        self, rows: list[tuple[str, dict[str, Any]]]
    ) -> list[ConversationEntry]:
        prior_ids = set(self._entry_ids)
        parent_id = self._entries[-1].id if self._entries else None
        next_seq = self._entries[-1].seq + 1 if self._entries else 1
        entries: list[ConversationEntry] = []
        for index, (entry_type, data) in enumerate(rows):
            entry_id = uuid.uuid4().hex
            if entry_id in prior_ids:
                raise ConversationIntegrityError(
                    f"duplicate conversation id: {entry_id}"
                )
            prior_ids.add(entry_id)
            entry = ConversationEntry(
                seq=next_seq + index,
                id=entry_id,
                parent_id=parent_id,
                lane="main",
                type=entry_type,
                data=copy.deepcopy(data),
            )
            self._validate_entry_payload(entry)
            entries.append(entry)
            parent_id = entry_id
        # Validate the complete candidate sequence before any bytes reach disk.
        # Batches are intentionally small; ordinary one-row appends retain the
        # O(1) incremental integrity path.
        self._entries.extend(entries)
        try:
            self._validate_entries()
        finally:
            del self._entries[-len(entries) :]
        encoded = b"".join(encode_json(entry.to_dict()) + b"\n" for entry in entries)
        self._write_bytes(encoded)
        for entry in entries:
            self._entries.append(entry)
            self._entry_ids.add(entry.id)
            self._record_active_entry(entry)
            if entry.type == "notification" and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND:
                task_id = entry.data.get("task_id")
                if type(task_id) is str and task_id:
                    self._task_notification_ids.add(task_id)
        if self.on_persisted_activity is not None:
            self.on_persisted_activity(entries[-1].seq)
        return entries

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
            parent_id=(
                parent_id
                if parent_id is not None
                else (self._entries[-1].id if self._entries else None)
            ),
            lane="main",
            type=entry_type,
            data=copy.deepcopy(data),
        )
        # Reject with the same payload validator used while loading before any
        # bytes reach disk. This keeps every append path from creating a row the
        # next store open cannot read.
        self._validate_entry_payload(entry)
        previous_deadline = self._write_deadline
        self._write_deadline = deadline
        try:
            self._write_line(entry.to_dict())
        finally:
            self._write_deadline = previous_deadline
        linear = not self._entries or entry.parent_id == self._entries[-1].id
        self._entries.append(entry)
        self._entry_ids.add(entry.id)
        if linear:
            self._record_active_entry(entry)
        else:
            # Explicit branch appends are rare; rebuild active-branch indexes.
            self._validate_entries()
        if self.on_persisted_activity is not None:
            self.on_persisted_activity(entry.seq)
            self._rebuild_incremental_validation_state()
        if entry.type == "notification" and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND:
            task_id = entry.data.get("task_id")
            if type(task_id) is str and task_id:
                self._task_notification_ids.add(task_id)
        return entry

    def append_message(
        self, message: Message, *, parent_id: str | None = None
    ) -> ConversationEntry:
        return self._append_row("message", {"message": message.to_dict()}, parent_id)

    async def append_message_async(
        self, message: Message, *, parent_id: str | None = None
    ) -> ConversationEntry:
        """Append off the event loop, serialized with other async writes."""
        if parent_id is None:
            return await self._to_thread_durable(self.append_message, message)
        return await self._to_thread_durable(
            self.append_message, message, parent_id=parent_id
        )

    def append_task_notification(
        self,
        *,
        task_id: str,
        command: str,
        exit_code: int | None,
        output_tail: str = "",
        log_path: str | None = None,
        note: str | None = None,
        background_metadata: tuple[str, str] = ("run_background", "natural_exit"),
    ) -> ConversationEntry:
        """Persist a bounded notification for a model-owned process exit."""
        if not task_id or not command or type(exit_code) not in {int, type(None)}:
            raise ValueError("invalid task notification")
        with self._append_lock():
            self._load()
            # Confirm id-set hits against the branch, preserving old scan behavior.
            if task_id in self._task_notification_ids:
                existing = next(
                    (
                        entry
                        for entry in self.agent_notifications(pending_only=False)
                        if entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND
                        and entry.data.get("task_id") == task_id
                    ),
                    None,
                )
                if existing is not None:
                    return existing
            if len(output_tail) > 2_048:
                raise ValueError("task notification output is too long")
            data: dict[str, Any] = {
                "kind": TASK_EXITED_NOTIFICATION_KIND,
                "task_id": task_id,
                "headline": command,
                "exit_code": exit_code,
                "output_tail": output_tail,
                "background_owner": background_metadata[0],
                "background_phase": background_metadata[1],
            }
            if log_path is not None:
                data["log_path"] = log_path
            if note is not None:
                data["note"] = note
            entry = self._append_row_unlocked("notification", data)
            self._task_notification_ids.add(task_id)
            return self._snapshot_entry(entry)

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
        pinned_message: Message | None = None,
        parent_id: str | None = None,
        expected_parent_id: str | None = None,
        kind: str = "summary",
        view: list[dict[str, Any]] | None = None,
        telemetry: Mapping[str, Any] | None = None,
    ) -> ConversationEntry:
        data = {
            "summary": summary,
            "source_seq_start": source_seq_start,
            "source_seq_end": source_seq_end,
            "replaces": list(replaces),
        }
        if kind != "summary":
            data["kind"] = kind
        if view is not None:
            data["view"] = view
        if telemetry is not None:
            data["telemetry"] = dict(telemetry)
        if pinned_message is not None:
            if pinned_message.role is not MessageRole.USER:
                raise ValueError("compaction pinned message must be a user message")
            data["pinned_message"] = pinned_message.to_dict()
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

    async def append_message_with_approval_requests_async(
        self,
        message: Message,
        approval_requests: Iterable[
            tuple[str, ToolCall] | tuple[str, ToolCall, Mapping[str, object]]
        ] = (),
        *,
        parent_id: str | None = None,
    ) -> ConversationEntry:
        """Append an approval-bearing message through the async writer gate."""
        materialized = list(approval_requests)
        if parent_id is None:
            return await self._to_thread_durable(
                self.append_message_with_approval_requests, message, materialized
            )
        return await self._to_thread_durable(
            self.append_message_with_approval_requests,
            message,
            materialized,
            parent_id=parent_id,
        )

    def append_message_with_approval_requests(
        self,
        message: Message,
        approval_requests: Iterable[
            tuple[str, ToolCall] | tuple[str, ToolCall, Mapping[str, object]]
        ] = (),
        *,
        parent_id: str | None = None,
    ) -> ConversationEntry:
        request_data = normalize_approval_requests(message, approval_requests)
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
                if (
                    "approval_display" in request
                    and existing_request.get("approval_display")
                    != request["approval_display"]
                ):
                    raise ConversationIntegrityError(
                        f"approval request display mismatch: {request['request_id']}"
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
        """Return strict approval state after waiting for concurrent writers."""
        with self._append_lock():
            self._load()
            return self._approval_states_from_indexes()

    def _approval_states_from_indexes(self) -> dict[str, tuple[ToolCall, str | None]]:
        states: dict[str, tuple[ToolCall, str | None]] = {}
        for request_id, entry in self._active_approval_requests.items():
            request = next(
                request
                for request in entry.data.get("approval_requests", [])
                if request["request_id"] == request_id
            )
            states[request_id] = (
                ToolCall.from_dict(request["tool_call"]),
                self._active_approval_resolutions.get(request_id),
            )
        return states

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
        """Return latency-tolerant display state; use approval_states for decisions."""
        self._sync_log_without_waiting()
        states = self._approval_states_from_indexes()
        return [
            (request_id, tool_call)
            for request_id, (tool_call, decision) in states.items()
            if decision is None
        ]

    def compaction_marker_count(self) -> int:
        return sum(entry.type == "compaction" for entry in self.replay())

    def _active_branch(self) -> tuple[ConversationEntry, ...]:
        """Return resident active-branch entries for store-internal queries."""

        if not self._entries:
            return ()
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
        return tuple(reversed(branch))

    def replay(self) -> list[ConversationEntry]:
        return [self._snapshot_entry(entry) for entry in self._active_branch()]

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

    def tool_result(self, tool_call_id: str) -> ToolResult | None:
        """Return a strict tool result after waiting for concurrent writers."""
        with self._append_lock():
            self._load()
            return self._active_tool_results.get(tool_call_id)

    @property
    def entries(self) -> list[ConversationEntry]:
        return [self._snapshot_entry(entry) for entry in self._entries]
