"""Compaction-marker projection for provider context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...context_accounting import cooperative_pause
from ...context_eviction import (
    EVICTION_KIND,
    _fact_floor,
    normalize_evicted_tool_result,
    valid_eviction_view,
)
from ...protocol.types import Message, MessageRole, TextContent
from ..store import ConversationEntry
from .types import _ContextItem


class ContextMarkerMixin:
    """Project persisted compaction markers into provider messages."""

    @staticmethod
    def _marker_items(
        entry: ConversationEntry,
        active_source_seqs: set[int],
        source_entries: Sequence[ConversationEntry] = (),
    ) -> list[_ContextItem]:
        if entry.data.get("kind") == EVICTION_KIND:
            view = entry.data["view"]
            valid = valid_eviction_view(entry, active_source_seqs)
            if valid:
                pinned = entry.data.get("pinned_message")
                pinned_message = None if pinned is None else Message.from_dict(pinned)
                if pinned_message is not None and "source_seq" not in pinned_message.metadata:
                    pinned_seq = ContextMarkerMixin._legacy_pinned_seq(
                        entry, pinned_message, source_entries
                    )
                    if pinned_seq is not None:
                        pinned_message = Message(
                            pinned_message.role,
                            list(pinned_message.content),
                            tool_result=pinned_message.tool_result,
                            metadata={**pinned_message.metadata, "source_seq": pinned_seq},
                        )
                messages = ContextMarkerMixin._eviction_view_messages(
                    view,
                    pinned_message,
                    _fact_floor.projected_fact_floor(entry.data.get("fact_floor")),
                )
            else:
                messages = [
                    normalize_evicted_tool_result(
                        Message(
                            message.role,
                            list(message.content),
                            tool_result=message.tool_result,
                            metadata={
                                **message.metadata,
                                "eviction_view_invalid": True,
                            },
                        )
                    )
                    for item in view
                    if isinstance(item.get("message"), Mapping)
                    for message in [Message.from_dict(item["message"])]
                ]
            items = [_ContextItem(entry, message) for message in messages]
        else:
            messages = ContextMarkerMixin._marker_messages(
                entry.data["source_seq_start"],
                entry.data["source_seq_end"],
                entry.data["summary"],
            )
            items = [_ContextItem(entry, message, fixed=True) for message in messages]
        if entry.data.get("kind") != EVICTION_KIND:
            pinned = entry.data.get("pinned_message")
            if pinned is not None:
                items.append(_ContextItem(entry, Message.from_dict(pinned)))
        return items

    @staticmethod
    def _legacy_pinned_seq(
        entry: ConversationEntry,
        pinned: Message,
        source_entries: Sequence[ConversationEntry],
    ) -> int | None:
        matches = [
            source.seq
            for source in source_entries
            if entry.data["source_seq_start"] <= source.seq <= entry.data["source_seq_end"]
            and source.type == "message"
            and Message.from_dict(source.data["message"]) == pinned
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _eviction_view_messages(
        view: Sequence[Mapping[str, Any]],
        pinned: Message | None = None,
        floor: Sequence[Message] = (),
    ) -> list[Message]:
        """Project a stored or proposed eviction view into request messages."""
        projected_pinned = pinned
        pinned_seq = None if pinned is None else pinned.metadata.get("source_seq")
        if pinned is not None and type(pinned_seq) is int:
            projected_pinned = Message(
                pinned.role,
                list(pinned.content),
                tool_result=pinned.tool_result,
                metadata={key: value for key, value in pinned.metadata.items() if key != "source_seq"},
            )
        messages = ContextMarkerMixin._merge_eviction_view(view, projected_pinned, pinned_seq)
        if floor:
            pin_index = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if message is projected_pinned
                    or (pinned is not None and message == projected_pinned)
                ),
                len(messages),
            )
            messages[pin_index:pin_index] = floor
        return messages

    @staticmethod
    def _merge_eviction_view(
        view: Sequence[Mapping[str, Any]],
        pinned: Message | None,
        pinned_seq: object,
    ) -> list[Message]:
        messages: list[Message] = []
        inserted = False
        for index, item in enumerate(view):
            seq = item["seq"]
            if pinned is not None and type(pinned_seq) is int and seq == pinned_seq:
                messages.append(pinned)
                inserted = True
            elif pinned is not None and not inserted and type(pinned_seq) is int and seq > pinned_seq:
                messages.extend((pinned, ContextMarkerMixin._eviction_view_message(item)))
                inserted = True
            else:
                messages.append(ContextMarkerMixin._eviction_view_message(item))
            if index % 4 == 3:
                cooperative_pause()
        if pinned is not None and not inserted:
            if type(pinned_seq) is not int and any(message == pinned for message in messages):
                return messages
            if type(pinned_seq) is not int:
                # Legacy markers may not have the original row. This preserves the old safe fallback.
                last_user = max(
                    (i for i, message in enumerate(messages) if message.role is MessageRole.USER),
                    default=-1,
                )
                messages.insert(last_user + 1, pinned)
            else:
                messages.append(pinned)
        return messages

    @staticmethod
    def _eviction_view_message(item: Mapping[str, Any]) -> Message:
        message = Message.from_dict(item["message"])
        return normalize_evicted_tool_result(
            Message(
                message.role,
                list(message.content),
                tool_result=message.tool_result,
                metadata={**message.metadata, "source_seq": item["seq"]},
            )
        )

    @staticmethod
    def _marker_messages(
        source_start: int,
        source_end: int,
        summary: str,
    ) -> list[Message]:
        marker_text = f"[compaction marker: entries {source_start}–{source_end}]"
        metadata = {
            "source_seq_start": source_start,
            "source_seq_end": source_end,
        }
        return [
            Message(
                MessageRole.COMPACTION,
                [TextContent(marker_text)],
                metadata=metadata,
            ),
            Message(
                MessageRole.ASSISTANT,
                [TextContent(summary)],
                metadata={"compaction_summary": True, **metadata},
            ),
        ]
