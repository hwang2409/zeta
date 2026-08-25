"""Helpers for grouping streamed transcript content."""

from __future__ import annotations

from ..types import (
    RedactedThinkingContent,
    StreamEvent,
    TextContent,
    ThinkingContent,
)


def stream_key(
    event: StreamEvent,
) -> tuple[str | None, tuple[str, object] | None]:
    content = event.content
    index = event.data.get("index")
    identity = (
        ("index", index)
        if isinstance(index, (int, str))
        else None
    )
    if isinstance(content, ThinkingContent):
        return "thinking", identity or (
            ("signature", content.signature)
            if content.signature is not None
            else ("kind", "thinking")
        )
    if isinstance(content, RedactedThinkingContent):
        return "redacted-thinking", identity or ("data", content.data)
    if isinstance(content, TextContent) or event.delta is not None:
        return "assistant", ("kind", "assistant")
    return None, None
