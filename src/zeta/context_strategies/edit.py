"""Structurally safe, durable model-authored context replacements."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.store import ConversationEntry, ConversationStore
from ..protocol.types import Message, MessageRole, TextContent
from . import ContextTelemetry
from .archive import (
    ContextBlock,
    _estimate_tokens,
    _validate_target,
    snap_range,
)

MAX_REPLACEMENT_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class ReplaceResult:
    seq_start: int
    seq_end: int
    tokens: int


def replace_context(
    store: ConversationStore,
    *,
    seq_start: int,
    seq_end: int,
    replacement: str,
) -> ReplaceResult:
    """Persist a plain-text assistant note replacing an active-branch range."""

    if type(replacement) is not str or not replacement.strip():
        raise ValueError("replacement must be a nonempty string")
    if len(replacement) > MAX_REPLACEMENT_CHARS:
        raise ValueError("replacement must be at most 4000 characters")
    entries = store.replay()
    start, end = snap_range(entries, seq_start, seq_end)
    selected = _validate_target(entries, start, end, retained_tail=None)
    tokens = _estimate_tokens(selected)
    store._append_row(
        "context_replace",
        {
            "source_seq_start": start,
            "source_seq_end": end,
            "replacement": replacement,
            "tokens": tokens,
        },
    )
    ContextTelemetry().emit(
        "replace", kind="replace", range=[start, end], tokens=tokens
    )
    return ReplaceResult(start, end, tokens)


def render_edit(edit: ConversationEntry) -> ContextBlock:
    """Render one replacement as typed assistant text, never reparsed content."""

    start = edit.data["source_seq_start"]
    end = edit.data["source_seq_end"]
    message = Message(
        MessageRole.ASSISTANT,
        [
            TextContent(
                f"[model-authored context note replacing seq {start}–{end}]\n"
                f"{edit.data['replacement']}"
            )
        ],
        metadata={
            "context_model_note": True,
            "context_strategy_fixed": True,
            "source_seq_start": start,
            "source_seq_end": end,
        },
    )
    return ContextBlock(edit, message, start, end, fixed=True)
