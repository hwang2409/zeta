"""Shared append/load validation for durable store payloads."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from ...agent.receipt import valid_killed_task_fields

MAX_AGENT_NOTIFICATION_TEXT = 10_000
AGENT_COMPLETION_NOTIFICATION_KIND = "agent_completion"
CHILD_IDLE_NOTIFICATION_KIND = "child_idle"
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


def _valid_explicit_finish_fields(data: Mapping[str, Any]) -> bool:
    fields = {"report_status", "summary", "head_sha", "verdict"} & data.keys()
    if not fields:
        return True
    head_sha = data.get("head_sha")
    return (
        data.get("report_status") in {"done", "blocked", "failed"}
        and type(data.get("summary")) is str
        and bool(data["summary"])
        and len(data["summary"]) <= 8_000
        and (
            head_sha is None
            or type(head_sha) is str
            and re.fullmatch(r"[0-9a-f]{7,64}", head_sha) is not None
        )
        and data.get("verdict") in {None, "pass", "fail", "inconclusive"}
    )


def validate_terminal_transition_data(data: Mapping[str, Any]) -> None:
    """Validate one durable finish-or-cancel fence winner."""
    action = data.get("action")
    report = data.get("report")
    if action not in {"finish", "cancel"}:
        raise ValueError("invalid terminal transition")
    if action == "cancel":
        if set(data) != {"action"}:
            raise ValueError("cancel transition cannot contain a report")
        return
    if (
        set(data) != {"action", "report"}
        or type(report) is not dict
        or set(report) - {"status", "summary", "head_sha", "verdict"}
        or report.get("status") not in {"done", "blocked", "failed"}
        or type(report.get("summary")) is not str
        or not report["summary"]
        or len(report["summary"]) > 8_000
        or "head_sha" in report
        and (
            type(report["head_sha"]) is not str
            or not report["head_sha"]
            or len(report["head_sha"]) > 128
        )
        or "verdict" in report
        and report["verdict"] not in {"pass", "fail", "inconclusive"}
    ):
        raise ValueError("invalid explicit finish report")


def validate_child_idle_notification_data(data: Mapping[str, Any]) -> None:
    """Apply the bounded schema for a nonterminal child-idle notice."""

    allowed = {
        "origin",
        "kind",
        "child_instance_id",
        "child_session_path",
        "description",
        "status",
        "text",
        "ask_parent_question_id",
        "display_only",
    }
    question_id = data.get("ask_parent_question_id")
    if (
        set(data) - allowed
        or data.get("origin") != "notification"
        or data.get("kind") != CHILD_IDLE_NOTIFICATION_KIND
        or data.get("status") != CHILD_IDLE_NOTIFICATION_KIND
        or type(data.get("child_instance_id")) is not str
        or not data["child_instance_id"]
        or len(data["child_instance_id"]) > 512
        or type(data.get("child_session_path")) is not str
        or not data["child_session_path"]
        or len(data["child_session_path"]) > 4_096
        or type(data.get("description")) is not str
        or not data["description"]
        or len(data["description"]) > 1_000
        or type(data.get("text")) is not str
        or not data["text"]
        or len(data["text"]) > MAX_AGENT_NOTIFICATION_TEXT
        or question_id is not None
        and (type(question_id) is not str or not question_id or len(question_id) > 512)
        or ("display_only" in data and data["display_only"] is not True)
    ):
        raise ValueError("invalid child idle notification")


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
        or ("display_only" in data and data["display_only"] is not True)
        or not valid_killed_task_fields(data)
        or not _valid_explicit_finish_fields(data)
    ):
        raise ValueError("invalid agent notification")
