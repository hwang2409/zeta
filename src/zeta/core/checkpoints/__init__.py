"""Conversation checkpoint and fork support."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Iterable

from ...types import Message, MessageRole, TextContent, ToolUseContent

if TYPE_CHECKING:
    from ..store import ConversationEntry


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ConversationIntegrityError(ValueError):
    """Raised when a session file violates the conversation schema."""


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
            or source.type != "checkpoint"
            or entry.parent_id != source.id
            or entry.data["from_seq"] != source.seq
        ):
            raise ConversationIntegrityError(
                f"fork source is not its checkpoint parent: {entry.id}"
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
                {"label": resolved_label, "created_at": _now()},
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
                },
                parent_id=checkpoint.id,
            )
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
