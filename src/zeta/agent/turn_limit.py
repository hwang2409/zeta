"""Child lifetime turn admission and epoch-free result projection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocol.types import Message, harness_notice
from .receipt import _without_agent_receipt_suffix, has_agent_receipt_suffix

HANDOFF_NAME = "turn-limit-handoff.md"
_MARKER_PREFIX = "[zeta:turn-limit:v1] "


@dataclass(frozen=True, slots=True)
class TurnLimitProjection:
    status: str
    handoff_path: str
    readable_text: str


class ChildTurnControl:
    """Reserve completions against one child's lifetime limit."""

    def __init__(self, limit: int, child_session_path: str) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("child max_turns must be a positive integer")
        self.limit = limit
        self.handoff_path = str(Path(child_session_path) / HANDOFF_NAME)
        self.used = 0
        self._reserved = False
        self._warned = False

    def begin_completion(self) -> Message | None:
        if self._reserved:
            return None
        if self.used >= self.limit:
            return None
        self._reserved = True
        number = self.used + 1
        if not self._warned and number == math.ceil(0.9 * self.limit):
            self._warned = True
            remaining = self.limit - self.used
            return harness_notice(
                "child turn limit",
                f"{remaining} model completion(s) remain, including this one. "
                "finish current work now; do not start new scope. Commit local work "
                f"if applicable and write the handoff to {self.handoff_path}.",
                event="child_turn_limit",
            )
        return None

    def finish_completion(self) -> bool:
        if not self._reserved:
            raise RuntimeError("no child completion is reserved")
        self._reserved = False
        self.used += 1
        return self.exhausted

    def abandon_completion(self) -> None:
        self._reserved = False

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def reserved(self) -> bool:
        return self._reserved


def handoff_path(child_session_path: str | Path) -> str:
    return str(Path(child_session_path) / HANDOFF_NAME)


def marker_for(child_session_path: str | Path) -> str:
    payload = json.dumps(
        {"handoff_path": handoff_path(child_session_path)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _MARKER_PREFIX + payload


def build_turn_limit_result(receipt: str, child_session_path: str | Path) -> str:
    """Append the trusted marker after one canonical completed receipt."""
    if not has_agent_receipt_suffix(receipt):
        raise ValueError("turn-limit result must end with a canonical receipt")
    return f"{receipt}\n{marker_for(child_session_path)}"


def parse_turn_limit_result(
    text: object, child_session_path: str | Path
) -> TurnLimitProjection | None:
    if type(text) is not str:
        return None
    separator = text.rfind("\n")
    if separator < 0:
        return None
    readable = text[:separator]
    marker = text[separator + 1 :]
    if not has_agent_receipt_suffix(readable) or marker != marker_for(child_session_path):
        return None
    return TurnLimitProjection("turn_limit", handoff_path(child_session_path), readable)


def project_notification_data(data: dict[str, Any]) -> dict[str, Any]:
    """Project a validated stored completion without changing its stored shape."""
    if data.get("status") != "completed":
        return data
    projection = parse_turn_limit_result(
        data.get("text"), data.get("child_session_path", "")
    )
    if projection is None:
        return data
    return {
        **data,
        "status": projection.status,
        "handoff_path": projection.handoff_path,
        "text": projection.readable_text,
    }


def project_payload(payload: dict[str, object], child_session_path: str) -> dict[str, object]:
    """Return the public non-error payload for a live turn-limit completion."""
    projected = dict(payload)
    structured = dict(projected.get("structuredContent", {}))
    structured.update(
        {"status": "turn_limit", "handoff_path": handoff_path(child_session_path)}
    )
    projected["structuredContent"] = structured
    return projected


def fallback_handoff(
    *,
    control: ChildTurnControl,
    child_instance_id: str,
    child_session_path: str,
    final_text: str,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
) -> bytes:
    lines = [
        "# child turn limit handoff",
        "",
        f"the configured turn limit was reached ({control.used}/{control.limit} completions).",
        f"child: {child_instance_id}",
        f"session: {child_session_path}",
        "",
        "## final assistant text",
        _without_agent_receipt_suffix(final_text).strip()
        or "no final assistant text",
    ]
    if children:
        lines.extend(("", "## unfinished direct children"))
        lines.extend(f"- {handle}: {description}" for handle, description in children)
    if tasks:
        lines.extend(("", "## unfinished tasks"))
        lines.extend(f"- {task_id}: {headline}" for task_id, headline in tasks)
    return ("\n".join(lines) + "\n").encode()
