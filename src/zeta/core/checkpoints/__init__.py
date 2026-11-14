"""Conversation checkpoint and fork support."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...protocol.types import (
    Message,
    MessageRole,
    TextContent,
)
from ..session_files import read_session_file, session_directory
from .entry_types import CONVERSATION_ENTRY_SPECS


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationIntegrityError(ValueError):
    """Raised when a session file violates the conversation schema."""


MAX_SESSION_JSON_DEPTH = 64
CONVERSATION_ENTRY_TYPES = frozenset(CONVERSATION_ENTRY_SPECS)


def load_session_json(source: Path | bytes) -> Any:
    """Decode a session file or JSONL row with a bounded container depth."""

    try:
        if isinstance(source, Path):
            path = source.absolute()
            with session_directory(path.parent.parent, path.parent.name) as (
                _,
                directory_fd,
            ):
                source = read_session_file(directory_fd, path.name)
        value = json.loads(source)
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            if isinstance(item, (dict, list)):
                depth += 1
                if depth > MAX_SESSION_JSON_DEPTH:
                    raise ValueError(f"JSON depth exceeds {MAX_SESSION_JSON_DEPTH}")
                children = item.values() if isinstance(item, dict) else item
                pending.extend((child, depth) for child in children)
        return value
    except (OSError, ValueError, RecursionError) as exc:
        location = str(source) if isinstance(source, Path) else "conversation row"
        raise ConversationIntegrityError(
            f"session JSON could not be read ({location}): {exc}"
        ) from exc


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
            raise ConversationIntegrityError(
                "conversation id must be a nonempty string"
            )
        if parent_id is not None and (type(parent_id) is not str or not parent_id):
            raise ConversationIntegrityError(
                "conversation parent_id must be null or a nonempty string"
            )
        if type(lane) is not str or lane != "main":
            raise ConversationIntegrityError("conversation lane must be 'main'")
        if type(entry_type) is not str or entry_type not in CONVERSATION_ENTRY_TYPES:
            raise ConversationIntegrityError(
                f"unsupported conversation entry type: {entry_type!r}"
            )
        if type(data) is not dict:
            raise ConversationIntegrityError("conversation data must be an object")
        normalized_data = dict(data)
        created_at = normalized_data.get("created_at")
        if entry_type == "checkpoint" and type(created_at) is str:
            try:
                timestamp = datetime.fromisoformat(created_at)
            except ValueError:
                pass
            else:
                if timestamp.tzinfo is None:
                    normalized_data["created_at"] = timestamp.replace(
                        tzinfo=UTC
                    ).isoformat()
        return cls(
            seq=seq,
            id=entry_id,
            parent_id=parent_id,
            lane=lane,
            type=entry_type,
            data=normalized_data,
        )


def active_branch(
    entries: Iterable[ConversationEntry],
) -> tuple[ConversationEntry, ...]:
    """Return the parent-linked branch ending at the newest entry."""

    ordered = tuple(entries)
    if not ordered:
        return ()
    by_id = {entry.id: entry for entry in ordered}
    current: ConversationEntry | None = ordered[-1]
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


@dataclass(frozen=True, slots=True)
class ForkPoint:
    """Compact coordinates for one user-message fork target."""

    seq: int
    id: str


@dataclass(frozen=True, slots=True)
class BranchInfo:
    """One leaf of the parent-linked conversation tree."""

    head: ConversationEntry
    divergence: ConversationEntry | None
    preview: str
    message_count: int
    is_current: bool


def _message_preview(entry: ConversationEntry) -> str | None:
    message = Message.from_dict(entry.data["message"])
    text = " ".join(
        block.text.strip()
        for block in message.content
        if isinstance(block, TextContent) and block.text.strip()
    )
    if text:
        return " ".join(text.split())[:80]
    return "[non-text message]"


def list_branches_from_entries(
    entries: Iterable[ConversationEntry],
) -> list[BranchInfo]:
    """Enumerate every leaf from one conversation snapshot."""

    ordered = tuple(entries)
    by_id = {entry.id: entry for entry in ordered}
    child_count: dict[str, int] = {}
    for entry in ordered:
        if entry.parent_id is not None:
            child_count[entry.parent_id] = child_count.get(entry.parent_id, 0) + 1
    branch = active_branch(ordered)
    active_head_id = branch[-1].id if branch else None
    branches: list[BranchInfo] = []
    for entry in ordered:
        if child_count.get(entry.id, 0) > 0:
            continue
        walker: ConversationEntry | None = entry
        chain: list[ConversationEntry] = []
        divergence: ConversationEntry | None = None
        while walker is not None:
            chain.append(walker)
            if (
                divergence is None
                and walker.id != entry.id
                and child_count.get(walker.id, 0) > 1
            ):
                divergence = walker
            if walker.parent_id is None:
                break
            walker = by_id.get(walker.parent_id)
        chain.reverse()
        preview = ""
        for candidate in reversed(chain):
            if candidate.type != "message":
                continue
            candidate_message = Message.from_dict(candidate.data["message"])
            if candidate_message.role is MessageRole.USER:
                preview = _message_preview(candidate) or ""
                break
        if not preview:
            preview = f"seq {entry.seq}"
        message_count = sum(
            1
            for candidate in chain
            if candidate.type == "message"
            and Message.from_dict(candidate.data["message"]).role is MessageRole.USER
        )
        branches.append(
            BranchInfo(
                head=entry,
                divergence=divergence,
                preview=preview,
                message_count=message_count,
                is_current=(entry.id == active_head_id),
            )
        )
    branches.sort(key=lambda info: info.head.seq)
    return branches


class CheckpointForkMixin:
    def _validate_checkpoint_or_fork_payload(self, entry: ConversationEntry) -> None:
        if entry.type == "checkpoint":
            label = entry.data.get("label")
            created_at = entry.data.get("created_at")
            if type(label) is not str or not label.strip():
                raise ValueError("checkpoint label must be a nonempty string")
            self._validate_checkpoint_label(label.strip())
            if type(created_at) is not str or not created_at:
                raise ValueError("checkpoint created_at must be a string")
            todo_dismissed = entry.data.get("todo_dismissed", False)
            if type(todo_dismissed) is not bool:
                raise ValueError("checkpoint todo dismissal must be a boolean")
        elif entry.type == "fork":
            from_entry_id = entry.data.get("from_entry_id")
            from_seq = entry.data.get("from_seq")
            label = entry.data.get("label")
            if type(from_entry_id) is not str or not from_entry_id:
                raise ValueError("fork source entry id must be a nonempty string")
            if type(from_seq) is not int or from_seq <= 0:
                raise ValueError("fork source sequence must be positive")
            if type(label) is not str or not label.strip():
                raise ValueError("fork label must be a nonempty string")

    def _validate_fork_entry(self, entry: ConversationEntry) -> None:
        source_id = entry.data["from_entry_id"]
        source_ref = next(
            (candidate for candidate in self.resident_refs() if candidate.id == source_id),
            None,
        )
        if (
            source_ref is None
            or entry.parent_id != source_ref.id
            or entry.data["from_seq"] != source_ref.seq
        ):
            raise ConversationIntegrityError(
                f"fork source is not its parent: {entry.id}"
            )

    def append_checkpoint(self, label: str | None = None) -> ConversationEntry:
        """Append a checkpoint at the active turn boundary."""

        with self._append_lock():
            self._load()
            if not self.is_turn_boundary():
                raise ConversationIntegrityError(
                    "cannot create a checkpoint while a turn is in flight"
                )
            resolved_label = label.strip() if label is not None else ""
            if not resolved_label:
                resolved_label = self._default_checkpoint_label()
            self._validate_checkpoint_label(resolved_label)
            entry = self._append_row_unlocked(
                "checkpoint",
                {
                    "label": resolved_label,
                    "created_at": _now(),
                    "todo_dismissed": self._todo_dismissed,
                },
            )
            return self._snapshot_entry(entry)

    def append_fork(self, selector: str) -> ConversationEntry:
        """Append a fork entry whose parent is an active checkpoint."""

        if not selector.strip():
            raise ValueError("fork requires a checkpoint label or sequence")
        with self._append_lock():
            self._load()
            if not self.is_turn_boundary():
                raise ConversationIntegrityError(
                    "cannot fork while a turn is in flight"
                )
            checkpoint = self._resolve_checkpoint(
                selector.strip(), self.relation_payloads({"checkpoint"})
            )
            entry = self._append_row_unlocked(
                "fork",
                {
                    "from_entry_id": checkpoint.id,
                    "from_seq": checkpoint.seq,
                    "label": checkpoint.data["label"],
                    "source_type": "checkpoint",
                },
                parent_id=checkpoint.id,
            )
            self._todo_dismissed = checkpoint.data.get("todo_dismissed", False)
            self._write_session_state(self.bash_cwd, self._todo_items)
            return self._snapshot_entry(entry)

    def is_turn_boundary(self) -> bool:
        """Return whether the active branch ends between complete turns."""

        last = self.last_message_ref()
        if last is None:
            return True
        message = Message.from_dict(self.entry(last).data["message"])
        return (
            message.role is not MessageRole.USER
            and not self._delivery_fold.open_tool_calls
        )

    def turn_in_flight(self) -> bool:
        return not self.is_turn_boundary()

    def has_outstanding_tool_calls(self) -> bool:
        """Return whether any tool_call lacks a matching tool_result (O(1))."""

        return bool(self._delivery_fold.open_tool_calls)

    def list_checkpoints(self) -> list[tuple[ConversationEntry, str | None]]:
        """Return active checkpoints with the next message preview."""

        refs = self.branch_refs()
        result: list[tuple[ConversationEntry, str | None]] = []
        for index, ref in enumerate(refs):
            if ref.type != "checkpoint":
                continue
            preview: str | None = None
            next_message = next(
                (following for following in refs[index + 1 :] if following.type == "message"),
                None,
            )
            if next_message is not None:
                preview = getattr(next_message.facts, "message_preview", None)
                if preview is None:
                    preview = "[non-text message]"
            label, created_at, todo_dismissed = ref.facts
            result.append(
                (
                    ConversationEntry(
                        seq=ref.seq,
                        id=ref.id,
                        parent_id=ref.parent_id,
                        lane=ref.lane,
                        type=ref.type,
                        data={
                            "label": label,
                            "created_at": created_at,
                            "todo_dismissed": todo_dismissed,
                        },
                    ),
                    preview,
                )
            )
        return result

    def checkpoint_count(self) -> int:
        return len(self.list_checkpoints())

    def list_user_message_forkpoints(
        self,
    ) -> list[tuple[int, ForkPoint, str]]:
        """Return ``(index, target, preview)`` for USER messages on the branch."""

        result: list[tuple[int, ForkPoint, str]] = []
        for ref in self.branch_refs():
            if not getattr(ref.facts, "is_user_forkpoint", False):
                continue
            preview = getattr(ref.facts, "message_preview", None) or ""
            result.append((len(result) + 1, ForkPoint(ref.seq, ref.id), preview))
        return result

    def append_message_fork(self, entry_id: str) -> ConversationEntry:
        """Fork from a prior USER message (implicit boundary checkpoint).

        The target must be a USER message on the active branch; the fork
        re-anchors the tail at that message and abandons everything after.
        Because the fork itself is a pure append that never splits an
        outstanding tool_call/tool_result pair, mid-turn tails are allowed
        as long as the target is a user boundary.
        """

        if not entry_id or not entry_id.strip():
            raise ValueError("fork requires a user message entry id")
        with self._append_lock():
            self._load()
            source_ref = next(
                (candidate for candidate in self.branch_refs() if candidate.id == entry_id),
                None,
            )
            if source_ref is None:
                raise ValueError(
                    f"user message is not on the active branch: {entry_id!r}"
                )
            if source_ref.type != "message":
                raise ValueError(f"fork source is not a message: {entry_id!r}")
            if not getattr(source_ref.facts, "is_user_forkpoint", False):
                raise ValueError(f"fork source is not a user message: {entry_id!r}")
            source = self.entry(source_ref)
            label = self._message_fork_label(source)
            entry = self._append_row_unlocked(
                "fork",
                {
                    "from_entry_id": source.id,
                    "from_seq": source.seq,
                    "label": label,
                    "source_type": "message",
                },
                parent_id=source.id,
            )
            return self._snapshot_entry(entry)

    def list_branches(self) -> list[BranchInfo]:
        """Enumerate every leaf in the parent-linked tree."""

        all_rows = tuple(self._resident_rows.entry(ref) for ref in self._resident_rows.rows)
        return list_branches_from_entries(all_rows)

    def switch_to_branch(self, head_entry_id: str) -> ConversationEntry:
        """Re-anchor the active branch on a leaf via a fork entry.

        The re-anchor itself is a pure append (a new ``fork`` row parented
        at the target leaf), so the current branch's turn boundary is not
        required — abandoning an in-flight turn is fine.

        The target branch's own tail is whatever the historical tree
        recorded — a leaf may sit mid-turn between a tool_call and its
        tool_result. Callers that need a clean tail must land on a
        boundary head themselves; this method does not repair one.
        """

        if not head_entry_id or not head_entry_id.strip():
            raise ValueError("branch switch requires a head entry id")
        with self._append_lock():
            self._load()
            refs = self.resident_refs()
            refs_by_id = {ref.id: ref for ref in refs}
            source_ref = refs_by_id.get(head_entry_id)
            if source_ref is None:
                raise ValueError(f"branch head not found: {head_entry_id!r}")
            if any(ref.parent_id == source_ref.id for ref in refs):
                raise ValueError(
                    f"entry has children; not a branch head: {head_entry_id!r}"
                )
            current_refs = self.branch_refs()
            if current_refs and current_refs[-1].id == source_ref.id:
                raise ValueError("already on that branch")
            source = self.entry(source_ref)
            label = self._branch_switch_label(source_ref, refs_by_id)
            entry = self._append_row_unlocked(
                "fork",
                {
                    "from_entry_id": source.id,
                    "from_seq": source.seq,
                    "label": label,
                    "source_type": "branch",
                },
                parent_id=source.id,
            )
            walker = source_ref
            dismissed = False
            while walker is not None:
                if walker.type == "checkpoint":
                    dismissed = bool(
                        self.entry(walker).data.get("todo_dismissed", False)
                    )
                    break
                walker = refs_by_id.get(walker.parent_id) if walker.parent_id else None
            self._todo_dismissed = dismissed
            self._write_session_state(self.bash_cwd, self._todo_items)
            return self._snapshot_entry(entry)

    def _default_checkpoint_label(self) -> str:
        refs = self.branch_refs()
        count = sum(ref.type == "checkpoint" for ref in refs) + 1
        for ref in reversed(refs):
            text = getattr(ref.facts, "message_preview", None)
            if text and not self._is_numeric_selector(text):
                return text[:48]
        return f"checkpoint {count}"

    @staticmethod
    def _is_numeric_selector(value: str) -> bool:
        try:
            int(value)
        except ValueError:
            return False
        return True

    @classmethod
    def _validate_checkpoint_label(cls, label: str) -> None:
        if cls._is_numeric_selector(label):
            raise ValueError(
                "checkpoint label cannot be numeric; numeric values are reserved "
                "for sequence selectors"
            )

    @staticmethod
    def _message_fork_label(entry: ConversationEntry) -> str:
        message = Message.from_dict(entry.data["message"])
        text = " ".join(
            block.text.strip()
            for block in message.content
            if isinstance(block, TextContent) and block.text.strip()
        )
        if text:
            return " ".join(text.split())[:48]
        return f"message seq {entry.seq}"

    @staticmethod
    def _branch_switch_label(head, by_id) -> str:
        walker = head
        while walker is not None:
            if getattr(walker.facts, "is_user", False):
                text = getattr(walker.facts, "message_preview", None)
                if text:
                    return text[:48]
            walker = by_id.get(walker.parent_id) if walker.parent_id else None
        return f"branch seq {head.seq}"

    @staticmethod
    def _resolve_checkpoint(
        selector: str,
        branch: list[ConversationEntry],
    ) -> ConversationEntry:
        checkpoints = [entry for entry in branch if entry.type == "checkpoint"]
        if CheckpointForkMixin._is_numeric_selector(selector):
            sequence = int(selector)
            sequence_matches = [entry for entry in checkpoints if entry.seq == sequence]
            if sequence_matches:
                return sequence_matches[0]
            raise ValueError(f"checkpoint not found: {selector!r}")
        label_matches = [
            entry for entry in checkpoints if entry.data["label"] == selector
        ]
        if len(label_matches) > 1:
            matches = ", ".join(str(entry.seq) for entry in label_matches)
            raise ValueError(
                f"ambiguous checkpoint label {selector!r}; matches seq {matches}"
            )
        if label_matches:
            return label_matches[0]
        raise ValueError(f"checkpoint not found: {selector!r}")
