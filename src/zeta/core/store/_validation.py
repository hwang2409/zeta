"""Shared append/load validation for durable store payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ...agent.receipt import valid_killed_task_fields

MAX_AGENT_NOTIFICATION_TEXT = 10_000


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
