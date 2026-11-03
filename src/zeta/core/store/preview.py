from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

from ..checkpoints import (
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from ..session_files import open_session_file, session_directory

# On Henry's 307 real sessions the first user message ends within 23.5 KB and 7 rows.
MAX_PREFIX_BYTES = 256 * 1024
MAX_PREFIX_ROWS = 64


def first_user_message(sessions_dir: Path, session_id: str) -> str:
    """Read the first user message from a bounded conversation-log prefix."""

    with session_directory(sessions_dir, session_id) as (_, directory_fd), os.fdopen(
        open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY), "rb"
    ) as conversation:
        for value in _prefix_rows(conversation):
            text = _user_text(value)
            if text is not None:
                return text[:512]
    return ""


def _prefix_rows(conversation: IO[bytes]) -> Iterator[dict[str, Any]]:
    """Yield complete rows until a byte or row cap; never read past the caps."""

    remaining = MAX_PREFIX_BYTES
    for _ in range(MAX_PREFIX_ROWS):
        line = conversation.readline(remaining)
        if not line.endswith(b"\n"):
            return  # end of file, a partial last row, or a row longer than the cap
        remaining -= len(line)
        value = load_session_json(line)
        if not isinstance(value, dict):
            raise ConversationIntegrityError("conversation row must be an object")
        yield value


def _user_text(value: dict[str, Any]) -> str | None:
    """Return the text of a user message row, or None for any other row."""

    if value.get("type") == "header" and "seq" not in value:
        return None
    entry = ConversationEntry.from_dict(value)
    message = entry.data.get("message") if entry.type == "message" else None
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content", [])
    if not isinstance(content, list):
        raise ConversationIntegrityError("message content must be an array")
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {"text", "thinking"}
        and isinstance(block.get("text"), str)
    )
