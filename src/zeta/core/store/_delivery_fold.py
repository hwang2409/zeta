"""One incrementally folded projection of the notification/prompt/receipt family.

The fold answers every hot delivery read (active notifications, presentations,
pending prompts, receipts, tool results, the terminal transition) in O(1) off
one resident object, instead of the O(n) branch scans used before. It is branch-scoped: a branch switch or rewind rebuilds it from the
selected branch, exactly where the store rebuilt the old per-field indexes.

Immutable after install
-----------------------
An installed fold is never mutated in place. Every change builds a new fold with
``staged()`` (copied containers), applies the new entries to that working copy,
and swaps it in by one reference. ``__copy__``/``__deepcopy__`` return ``self``,
so there is no per-command deep copy of the whole fold. Equality is by value
behind an identity fast path, so the cross-instance worker patch protocol detects
an owner that swapped its fold to an equal-valued object as unchanged and still
installs the writer's fold on the ``replace_entries`` refresh route.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...protocol.types import Message, ToolResult, ToolUseContent
from ..checkpoints import ConversationEntry, ConversationIntegrityError
from ._validation import AGENT_COMPLETION_NOTIFICATION_KIND

if TYPE_CHECKING:
    from ._incremental_validation import _ApprovalValidationOverlay


class DeliveryFold:
    """Resident notification/prompt/receipt relations for one active branch."""

    __slots__ = (
        "active_notifications",
        "active_prompts",
        "agent_call_ids",
        "agent_receipts",
        "completions_by_child",
        "delivery_message",
        "last_message",
        "last_terminal_transition",
        "last_turn_end",
        "notification_ids",
        "open_tool_calls",
        "presented",
        "prompt_ids",
        "prompt_messages",
        "tool_result_receipts",
        "tool_result_seqs",
    )

    def __init__(self) -> None:
        self.tool_result_seqs: dict[str, int] = {}
        self.tool_result_receipts: dict[str, tuple[str, str]] = {}
        self.open_tool_calls: set[str] = set()
        self.agent_call_ids: set[str] = set()
        self.agent_receipts: list[tuple[str, str]] = []
        self.notification_ids: set[str] = set()
        self.active_notifications: dict[str, ConversationEntry] = {}
        self.completions_by_child: dict[str, ConversationEntry] = {}
        self.presented: set[str] = set()
        self.prompt_ids: set[str] = set()
        self.active_prompts: dict[str, ConversationEntry] = {}
        self.prompt_messages: dict[str, Message] = {}
        self.delivery_message: Message | None = None
        self.last_message: Message | None = None
        self.last_turn_end: ConversationEntry | None = None
        self.last_terminal_transition: ConversationEntry | None = None

    # An installed fold is immutable, so a copy is the same object. This removes
    # the per-command deep copy of the whole fold.
    def __copy__(self) -> DeliveryFold:
        return self

    def __deepcopy__(self, memo: object) -> DeliveryFold:
        return self

    # The worker patch protocol compares a baseline value to the resident one to
    # decide whether to install the writer's value (``_async_writes`` install
    # check). That comparison must be by value, not identity: the owner can swap
    # its fold to a new object with unchanged values while a command is in
    # flight, and an identity test would then drop the writer's fold (with an
    # external notification in it) on the replace route. The identity fast path
    # keeps the common "same object" case free.
    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, DeliveryFold):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)

    __hash__ = None

    def staged(self) -> DeliveryFold:
        """Return a mutable working fold with its own containers for one change."""
        new = DeliveryFold.__new__(DeliveryFold)
        new.tool_result_seqs = dict(self.tool_result_seqs)
        new.tool_result_receipts = dict(self.tool_result_receipts)
        new.open_tool_calls = set(self.open_tool_calls)
        new.agent_call_ids = set(self.agent_call_ids)
        new.agent_receipts = list(self.agent_receipts)
        new.notification_ids = set(self.notification_ids)
        new.active_notifications = dict(self.active_notifications)
        new.completions_by_child = dict(self.completions_by_child)
        new.presented = set(self.presented)
        new.prompt_ids = set(self.prompt_ids)
        new.active_prompts = dict(self.active_prompts)
        new.prompt_messages = dict(self.prompt_messages)
        new.delivery_message = self.delivery_message
        new.last_message = self.last_message
        new.last_turn_end = self.last_turn_end
        new.last_terminal_transition = self.last_terminal_transition
        return new

    def apply(
        self,
        entry: ConversationEntry,
        approvals: _ApprovalValidationOverlay | None = None,
    ) -> None:
        """Record one active-branch entry into this (working) fold.

        Raises the same ``ConversationIntegrityError`` texts the store raised
        before for an unlinked or duplicate notification, presentation, or
        pending-prompt acknowledgement.
        """
        if approvals is not None:
            approvals.record(entry)
        entry_type = entry.type
        if entry_type == "turn_end":
            self.last_turn_end = entry
        elif entry_type == "message":
            self._apply_message(entry)
        elif entry_type in {"approval_request", "approval_resolution"}:
            pass
        elif entry_type == "notification":
            self._apply_notification(entry)
        elif entry_type == "notification_ack":
            notification_id = entry.data.get("notification_id")
            if notification_id not in self.notification_ids:
                raise ConversationIntegrityError(
                    f"notification acknowledgement is not linked: {notification_id}"
                )
            if notification_id not in self.active_notifications:
                raise ConversationIntegrityError(
                    f"duplicate notification acknowledgement: {notification_id}"
                )
            del self.active_notifications[notification_id]
        elif entry_type == "notification_tui_presented":
            notification_id = entry.data.get("notification_id")
            if notification_id not in self.notification_ids:
                raise ConversationIntegrityError(
                    f"notification TUI presentation is not linked: {notification_id}"
                )
            if notification_id in self.presented:
                raise ConversationIntegrityError(
                    f"duplicate notification TUI presentation: {notification_id}"
                )
            self.presented.add(notification_id)
        elif entry_type == "pending_prompt":
            self.prompt_ids.add(entry.id)
            self.active_prompts[entry.id] = entry
        elif entry_type == "pending_prompt_ack":
            prompt_id = entry.data.get("prompt_id")
            if prompt_id not in self.prompt_ids:
                raise ConversationIntegrityError(
                    f"pending prompt acknowledgement is not linked: {prompt_id}"
                )
            if prompt_id not in self.active_prompts:
                raise ConversationIntegrityError(
                    f"duplicate pending prompt acknowledgement: {prompt_id}"
                )
            del self.active_prompts[prompt_id]
            self.prompt_messages.pop(prompt_id, None)
        elif entry_type == "terminal_transition":
            self.last_terminal_transition = entry

    def _apply_message(self, entry: ConversationEntry) -> None:
        message = Message.from_dict(entry.data["message"])
        self.last_message = message
        prompt_id = message.metadata.get("zeta.pending_prompt_id")
        if type(prompt_id) is str:
            self.prompt_messages[prompt_id] = message
        identity = message.metadata.get("zeta.step_delivery")
        if isinstance(identity, dict) and identity.get("notification_ids"):
            self.delivery_message = message
        tool_result = message.tool_result
        if tool_result is not None:
            self.tool_result_seqs[tool_result.tool_call_id] = entry.seq
            receipt = self._agent_receipt(tool_result)
            if receipt is not None:
                self.tool_result_receipts[tool_result.tool_call_id] = receipt
        for block in message.content:
            if not isinstance(block, ToolUseContent):
                continue
            # Track every open tool call so has_outstanding_tool_calls() is an
            # O(1) fold read. Add the ids first, then discard the result id
            # below, matching the per-message order of the branch scan it
            # replaces.
            self.open_tool_calls.add(block.tool_call.id)
            if block.tool_call.name.casefold() == "agent":
                self.agent_call_ids.add(block.tool_call.id)
                prior_receipt = self.tool_result_receipts.get(block.tool_call.id)
                if prior_receipt is not None:
                    self.agent_receipts.append(prior_receipt)
        if tool_result is not None:
            self.open_tool_calls.discard(tool_result.tool_call_id)
            if tool_result.tool_call_id in self.agent_call_ids:
                self._record_agent_receipt(tool_result)

    def _apply_notification(self, entry: ConversationEntry) -> None:
        handle = entry.data.get("child_instance_id")
        child_path = entry.data.get("child_session_path")
        if type(handle) is str and handle and type(child_path) is str and child_path:
            self.agent_receipts.append((handle, child_path))
        self.notification_ids.add(entry.id)
        self.active_notifications[entry.id] = entry
        if (
            entry.data.get("kind", AGENT_COMPLETION_NOTIFICATION_KIND)
            == AGENT_COMPLETION_NOTIFICATION_KIND
            and type(child_id := entry.data.get("child_instance_id")) is str
        ):
            self.completions_by_child.setdefault(child_id, entry)

    @staticmethod
    def _agent_receipt(result: ToolResult) -> tuple[str, str] | None:
        structured = result.structured_content
        if structured is None:
            return None
        handle = structured.get("child_instance_id")
        child_path = structured.get("child_session_path")
        if type(handle) is str and handle and type(child_path) is str and child_path:
            return handle, child_path
        return None

    def _record_agent_receipt(self, result: ToolResult) -> None:
        receipt = self._agent_receipt(result)
        if receipt is not None:
            self.agent_receipts.append(receipt)
