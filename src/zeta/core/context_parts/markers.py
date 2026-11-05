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
        entry: ConversationEntry, active_source_seqs: set[int]
    ) -> list[_ContextItem]:
        if entry.data.get("kind") == EVICTION_KIND:
            view = entry.data["view"]
            valid = valid_eviction_view(entry, active_source_seqs)
            if valid:
                messages = ContextMarkerMixin._eviction_view_messages(view)
                messages.extend(
                    _fact_floor.projected_fact_floor(entry.data.get("fact_floor"))
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
        pinned = entry.data.get("pinned_message")
        if pinned is not None:
            items.append(_ContextItem(entry, Message.from_dict(pinned)))
        return items

    @staticmethod
    def _eviction_view_messages(view: Sequence[Mapping[str, Any]]) -> list[Message]:
        """Project a stored or proposed eviction view into request messages."""
        messages: list[Message] = []
        for index, item in enumerate(view):
            messages.append(ContextMarkerMixin._eviction_view_message(item))
            if index % 4 == 3:
                cooperative_pause()
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
