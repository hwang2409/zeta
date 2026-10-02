"""Shared append/load validation for durable store payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ...agent.receipt import valid_killed_task_fields
from ...protocol.types import Message, MessageRole

MAX_AGENT_NOTIFICATION_TEXT = 10_000
AGENT_COMPLETION_NOTIFICATION_KIND = "agent_completion"
TASK_EXITED_NOTIFICATION_KIND = "task_exited"


def valid_agent_stats(value: object) -> bool:
    if type(value) is not dict:
        return False
    return (
        type(value.get("turns_used")) is int
        and value["turns_used"] >= 0
        and type(value.get("elapsed")) in {int, float}
        and math.isfinite(value["elapsed"])
        and value["elapsed"] >= 0
        and type(value.get("tool_calls")) is int
        and value["tool_calls"] >= 0
        and type(value.get("error")) is bool
        and type(value.get("canceled")) is bool
    )


def validate_agent_notification_data(data: Mapping[str, Any]) -> None:
    """Apply the shared append/load contract for an agent notification."""

    if (
        type(data.get("child_instance_id")) is not str
        or not data["child_instance_id"]
        or type(data.get("child_session_path")) is not str
        or not data["child_session_path"]
        or type(data.get("description")) is not str
        or not data["description"]
        or data.get("status") not in {"completed", "error", "canceled"}
        or type(data.get("text")) is not str
        or not data["text"]
        or len(data["text"]) > MAX_AGENT_NOTIFICATION_TEXT
        or ("stats" in data and not valid_agent_stats(data["stats"]))
        or not valid_killed_task_fields(data)
    ):
        raise ValueError("invalid agent notification")


def validate_compaction_data(data: Mapping[str, Any]) -> None:
    """Validate summary and deterministic-eviction marker payloads."""

    summary = data.get("summary")
    source_start = data.get("source_seq_start")
    source_end = data.get("source_seq_end")
    replaces = data.get("replaces", [])
    if type(summary) is not str or not summary.strip():
        raise ValueError("compaction summary must be a nonempty string")
    if (
        type(source_start) is not int
        or type(source_end) is not int
        or source_start <= 0
        or source_end < source_start
    ):
        raise ValueError("compaction source sequence must be integers")
    if (
        type(replaces) is not list
        or any(type(entry_id) is not str or not entry_id for entry_id in replaces)
        or len(replaces) != len(set(replaces))
    ):
        raise ValueError("compaction replaces must be unique string IDs")
    kind = data.get("kind", "summary")
    view = data.get("view")
    if kind not in {"summary", "eviction"}:
        raise ValueError("compaction kind is invalid")
    if kind == "eviction":
        if type(view) is not list or not view:
            raise ValueError("eviction view must be a nonempty array")
        for row in view:
            if (
                type(row) is not dict
                or type(row.get("seq")) is not int
                or type(row.get("message")) is not dict
            ):
                raise ValueError("eviction view row is invalid")
            Message.from_dict(row["message"])
    pinned_message = data.get("pinned_message")
    if pinned_message is not None:
        if type(pinned_message) is not dict:
            raise ValueError("compaction pinned message must be an object")
        pinned = Message.from_dict(pinned_message)
        if pinned.role is not MessageRole.USER:
            raise ValueError("compaction pinned message must be a user message")
