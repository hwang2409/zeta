"""Identify nested approval audit pairs without changing the stored schema."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ...protocol.types import ASSISTANT_RESPONSE_SYNTHETIC
from ..store import ConversationEntry


def nested_approval_anchors(entries: Sequence[ConversationEntry]) -> dict[str, str]:
    outers: set[str] = set()
    anchors: dict[str, str] = {}
    for entry in entries:
        if entry.type != "message":
            continue
        message = entry.data.get("message", {})
        if message.get("role") != "assistant":
            continue
        blocks = message.get("content", ())
        synthetic = message.get("metadata", {}).get("response_state") == ASSISTANT_RESPONSE_SYNTHETIC
        calls = [block["tool_call"] for block in blocks if block.get("type") == "tool_use"]
        if not synthetic:
            outers.update(call["id"] for call in calls if call.get("name") == "code")
        elif len(blocks) == 1 and len(calls) == 1:
            call_id = calls[0]["id"]
            parent, marker, index = call_id.rpartition(":code:")
            if (marker and parent in outers and re.fullmatch(r"[1-9][0-9]*", index)
                    and any(row.get("request_id") == call_id for row in entry.data.get("approval_requests", ()))):
                anchors[call_id] = entry.id
    return anchors


def nested_call_data(call_id: str, parent: str) -> dict[str, object]:
    prefix = parent + ":code:"
    index = call_id.removeprefix(prefix)
    if call_id.startswith(prefix) and re.fullmatch(r"(?:[1-9]|[1-4][0-9]|50)", index):
        return {"code_parent_id": parent, "code_call_index": int(index)}
    return {}


def code_projection(structured: Mapping[str, object] | None) -> Mapping[str, object]:
    code = (structured or {}).get("code_mode")
    return code if isinstance(code, dict) and code.get("version") == 1 else {}
