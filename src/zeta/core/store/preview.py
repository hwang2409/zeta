from __future__ import annotations

import os
from pathlib import Path

from ..checkpoints import (
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from ..session_files import open_session_file, session_directory


def first_user_message(sessions_dir: Path, session_id: str) -> str:
    """Read the first user message from a bounded conversation-log prefix."""

    with session_directory(sessions_dir, session_id) as (_, directory_fd), os.fdopen(
        open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY), "rb"
    ) as conversation:
        for line in conversation:
            value = load_session_json(line)
            if not isinstance(value, dict):
                raise ConversationIntegrityError("conversation row must be an object")
            if value.get("type") == "header" and "seq" not in value:
                continue
            entry = ConversationEntry.from_dict(value)
            if entry.type != "message":
                continue
            message = entry.data.get("message")
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                raise ConversationIntegrityError("message content must be an array")
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"text", "thinking"} and isinstance(
                    block.get("text"), str
                ):
                    parts.append(block["text"])
            return "".join(parts)[:512]
    return ""
