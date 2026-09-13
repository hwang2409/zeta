"""Immutable automation definitions and durable execution snapshots."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..core.approval import parse_approval_rule
from .trigger import Trigger, parse_trigger


@dataclass(frozen=True)
class Job:
    name: str
    prompt: str
    trigger: Trigger
    servers: tuple[str, ...]
    allow: tuple[str, ...]
    deliver: str
    provider: str
    model: str
    cwd: str

    def document(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("name")
        value["servers"] = list(self.servers)
        value["allow"] = list(self.allow)
        return value


@dataclass(frozen=True)
class JobState:
    job: Job
    revision: int
    approved_at: datetime | None = None
    recipient: str | None = None
    last_run: datetime | None = None
    last_check: datetime | None = None
    last_due: datetime | None = None
    enabled: bool = False


@dataclass(frozen=True)
class DueOccurrence:
    name: str
    revision: int
    due_at: datetime
    checked_at: datetime
    last_run: datetime


@dataclass(frozen=True)
class PollEvent:
    id: str
    timestamp: datetime
    context: str


@dataclass(frozen=True)
class RunRecord:
    id: str
    name: str
    revision: int
    due_at: str
    status: str
    session_id: str | None
    detail: str
    delivery: str


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone aware")
    return value.astimezone(UTC).isoformat()


def instant(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    timestamp(result)
    return result.astimezone(UTC)


def parse_job(name: str, value: object) -> Job:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError(
            "job name must contain 1..80 letters, numbers, dots, dashes or underscores"
        )
    if not isinstance(value, dict):
        raise TypeError("job must be an object")
    required = {
        "prompt",
        "trigger",
        "servers",
        "allow",
        "deliver",
        "provider",
        "model",
        "cwd",
    }
    if set(value) != required:
        raise ValueError(f"job fields must be: {', '.join(sorted(required))}")
    for field in ("prompt", "deliver", "provider", "model", "cwd"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    for field in ("servers", "allow"):
        if not isinstance(value[field], (list, tuple)) or any(
            not isinstance(item, str) or not item for item in value[field]
        ):
            raise ValueError(f"{field} must be a list of strings")
        if len(value[field]) != len(set(value[field])):
            raise ValueError(f"{field} must not contain duplicates")
    if value["provider"] not in {"claude", "codex", "fake"}:
        raise ValueError("unknown provider")
    if not Path(value["cwd"]).is_absolute():
        raise ValueError("cwd must be an absolute path on the daemon host")
    if not value["deliver"].startswith("slack:") or not value["deliver"][6:].strip():
        raise ValueError("deliver must be slack:<user or channel>")
    for rule_text in value["allow"]:
        rule = parse_approval_rule(rule_text)
        if "__" in rule.tool and rule.tool.split("__", 1)[0] not in value["servers"]:
            raise ValueError(f"allow rule references an unselected server: {rule.tool}")
    return Job(
        name,
        value["prompt"],
        parse_trigger(value["trigger"]),
        tuple(value["servers"]),
        tuple(value["allow"]),
        value["deliver"],
        value["provider"],
        value["model"],
        value["cwd"],
    )
