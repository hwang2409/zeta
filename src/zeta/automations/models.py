"""Immutable automation definitions and durable execution snapshots."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..core.approval import parse_approval_rule
from ..models.catalog import validate_provider
from .trigger import Trigger, Webhook, parse_trigger


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
    account: str | None = None

    def document(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("name")
        if self.account is None:
            value.pop("account")
        if isinstance(self.trigger, Webhook):
            trigger: dict[str, object] = {
                "kind": "webhook",
                "verify": self.trigger.verify,
            }
            if self.trigger.verify == "hmac-sha256":
                trigger.update(
                    signature_header=self.trigger.signature_header,
                    signature_prefix=self.trigger.signature_prefix,
                )
                if self.trigger.timestamp_header is not None:
                    trigger["timestamp_header"] = self.trigger.timestamp_header
            value["trigger"] = trigger
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
    inbox_target_ids: tuple[str, ...] = ()


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
        "provider",
        "model",
        "cwd",
    }
    missing = required - set(value)
    extra = set(value) - required - {"deliver", "account"}
    if missing or extra:
        fields = required | {"deliver"} | set(value)
        raise ValueError(f"job fields must be: {', '.join(sorted(fields))}")
    value = dict(value)
    value.setdefault("deliver", "none")
    value.setdefault("account", None)
    for field in ("prompt", "provider", "model", "cwd"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    for field in ("servers", "allow"):
        if not isinstance(value[field], (list, tuple)) or any(
            not isinstance(item, str) or not item for item in value[field]
        ):
            raise ValueError(f"{field} must be a list of strings")
        if len(value[field]) != len(set(value[field])):
            raise ValueError(f"{field} must not contain duplicates")
    validate_provider(value["provider"])
    if value["account"] is not None and (
        not isinstance(value["account"], str)
        or not value["account"].startswith(
            "anthropic:" if value["provider"] == "claude" else f"{value['provider']}:"
        )
    ):
        raise ValueError("account must match the job provider")
    if not Path(value["cwd"]).is_absolute():
        raise ValueError("cwd must be an absolute path on the daemon host")
    if value["deliver"] != "none" and (
        not value["deliver"].startswith("slack:") or not value["deliver"][6:].strip()
    ):
        raise ValueError("deliver must be slack:<user or channel> or none")
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
        value["account"],
    )
