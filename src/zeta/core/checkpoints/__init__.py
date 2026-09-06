"""Conversation checkpoint and fork support."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ...types import Message, MessageRole, TextContent, ToolUseContent


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


@dataclass(frozen=True, slots=True)
class BranchInfo:
    """One leaf of the parent-linked conversation tree."""

    head: ConversationEntry
    divergence: ConversationEntry | None
    preview: str
    message_count: int
    is_current: bool


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
        source = next(
            (candidate for candidate in self._entries if candidate.id == source_id),
            None,
        )
        if (
            source is None
            or entry.parent_id != source.id
            or entry.data["from_seq"] != source.seq
        ):
            raise ConversationIntegrityError(
                f"fork source is not its parent: {entry.id}"
            )

    def append_checkpoint(self, label: str | None = None) -> ConversationEntry:
        """Append a checkpoint at the active turn boundary."""

        with self._append_lock():
            self._load()
            branch = self.replay()
            if not self.is_turn_boundary(branch):
                raise ConversationIntegrityError(
                    "cannot create a checkpoint while a turn is in flight"
                )
            resolved_label = label.strip() if label is not None else ""
            if not resolved_label:
                resolved_label = self._default_checkpoint_label(branch)
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
            branch = self.replay()
            if not self.is_turn_boundary(branch):
                raise ConversationIntegrityError(
                    "cannot fork while a turn is in flight"
                )
            checkpoint = self._resolve_checkpoint(selector.strip(), branch)
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

    @staticmethod
    def is_turn_boundary(entries: Iterable[ConversationEntry]) -> bool:
        """Return whether a branch ends between complete turns."""

        messages = [entry for entry in entries if entry.type == "message"]
        if not messages:
            return True
        last = messages[-1]
        if Message.from_dict(last.data["message"]).role is MessageRole.USER:
            return False
        outstanding: set[str] = set()
        for entry in messages:
            parsed = Message.from_dict(entry.data["message"])
            outstanding.update(
                block.tool_call.id
                for block in parsed.content
                if isinstance(block, ToolUseContent)
            )
            if parsed.tool_result is not None:
                outstanding.discard(parsed.tool_result.tool_call_id)
        return not outstanding

    def turn_in_flight(self) -> bool:
        return not self.is_turn_boundary(self.replay())

    @staticmethod
    def has_outstanding_tool_calls(entries: Iterable[ConversationEntry]) -> bool:
        """Return whether any tool_call lacks a matching tool_result."""

        outstanding: set[str] = set()
        for entry in entries:
            if entry.type != "message":
                continue
            parsed = Message.from_dict(entry.data["message"])
            outstanding.update(
                block.tool_call.id
                for block in parsed.content
                if isinstance(block, ToolUseContent)
            )
            if parsed.tool_result is not None:
                outstanding.discard(parsed.tool_result.tool_call_id)
        return bool(outstanding)

    def list_checkpoints(self) -> list[tuple[ConversationEntry, str | None]]:
        """Return active checkpoints with the next message preview."""

        branch = self.replay()
        result: list[tuple[ConversationEntry, str | None]] = []
        for index, entry in enumerate(branch):
            if entry.type != "checkpoint":
                continue
            preview: str | None = None
            for following in branch[index + 1 :]:
                if following.type != "message":
                    continue
                preview = self._message_preview(following)
                break
            result.append((entry, preview))
        return result

    def checkpoint_count(self) -> int:
        return len(self.list_checkpoints())

    def list_user_message_forkpoints(
        self,
    ) -> list[tuple[int, ConversationEntry, str]]:
        """Return ``(index, entry, preview)`` for USER messages on the branch."""

        branch = self.replay()
        result: list[tuple[int, ConversationEntry, str]] = []
        for entry in branch:
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            if message.role is not MessageRole.USER:
                continue
            preview = self._message_preview(entry) or ""
            result.append((len(result) + 1, entry, preview))
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
            branch = self.replay()
            source = next(
                (candidate for candidate in branch if candidate.id == entry_id),
                None,
            )
            if source is None:
                raise ValueError(
                    f"user message is not on the active branch: {entry_id!r}"
                )
            if source.type != "message":
                raise ValueError(
                    f"fork source is not a message: {entry_id!r}"
                )
            message = Message.from_dict(source.data["message"])
            if message.role is not MessageRole.USER:
                raise ValueError(
                    f"fork source is not a user message: {entry_id!r}"
                )
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

        by_id = {entry.id: entry for entry in self._entries}
        child_count: dict[str, int] = {}
        for entry in self._entries:
            if entry.parent_id is not None:
                child_count[entry.parent_id] = (
                    child_count.get(entry.parent_id, 0) + 1
                )
        active_branch = self.replay()
        active_head_id = active_branch[-1].id if active_branch else None
        branches: list[BranchInfo] = []
        for entry in self._entries:
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
                    preview = self._message_preview(candidate) or ""
                    break
            if not preview:
                preview = f"seq {entry.seq}"
            message_count = sum(
                1
                for candidate in chain
                if candidate.type == "message"
                and Message.from_dict(candidate.data["message"]).role
                is MessageRole.USER
            )
            branches.append(
                BranchInfo(
                    head=self._snapshot_entry(entry),
                    divergence=(
                        self._snapshot_entry(divergence)
                        if divergence is not None
                        else None
                    ),
                    preview=preview,
                    message_count=message_count,
                    is_current=(entry.id == active_head_id),
                )
            )
        branches.sort(key=lambda info: info.head.seq)
        return branches

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
            current_branch = self.replay()
            by_id = {entry.id: entry for entry in self._entries}
            source = by_id.get(head_entry_id)
            if source is None:
                raise ValueError(
                    f"branch head not found: {head_entry_id!r}"
                )
            for entry in self._entries:
                if entry.parent_id == source.id:
                    raise ValueError(
                        f"entry has children; not a branch head: {head_entry_id!r}"
                    )
            if current_branch and current_branch[-1].id == source.id:
                raise ValueError("already on that branch")
            label = self._branch_switch_label(source, by_id)
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
            walker: ConversationEntry | None = source
            dismissed = False
            while walker is not None:
                if walker.type == "checkpoint":
                    dismissed = bool(walker.data.get("todo_dismissed", False))
                    break
                if walker.parent_id is None:
                    break
                walker = by_id.get(walker.parent_id)
            self._todo_dismissed = dismissed
            self._write_session_state(self.bash_cwd, self._todo_items)
            return self._snapshot_entry(entry)

    @staticmethod
    def _default_checkpoint_label(branch: list[ConversationEntry]) -> str:
        count = sum(entry.type == "checkpoint" for entry in branch) + 1
        for entry in reversed(branch):
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            text = " ".join(
                block.text.strip()
                for block in message.content
                if isinstance(block, TextContent)
                and block.text.strip()
            )
            if text and not CheckpointForkMixin._is_numeric_selector(text):
                return " ".join(text.split())[:48]
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
    def _branch_switch_label(
        head: ConversationEntry,
        by_id: Mapping[str, ConversationEntry],
    ) -> str:
        walker: ConversationEntry | None = head
        while walker is not None:
            if walker.type == "message":
                message = Message.from_dict(walker.data["message"])
                if message.role is MessageRole.USER:
                    text = " ".join(
                        block.text.strip()
                        for block in message.content
                        if isinstance(block, TextContent) and block.text.strip()
                    )
                    if text:
                        return " ".join(text.split())[:48]
            if walker.parent_id is None:
                break
            walker = by_id.get(walker.parent_id)
        return f"branch seq {head.seq}"

    @staticmethod
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
