"""Clock-free trigger parsing and cron matching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Schedule:
    cron: str
    timezone: str = "America/Toronto"
    kind: str = "schedule"


@dataclass(frozen=True)
class Poll:
    condition: str
    interval_seconds: int = 300
    kind: str = "poll"


@dataclass(frozen=True)
class Webhook:
    verify: str
    signature_header: str
    signature_prefix: str
    timestamp_header: str | None = None
    delivery_header: str | None = None
    kind: str = "webhook"


Trigger = Schedule | Poll | Webhook

_GITHUB = Webhook(
    verify="github",
    signature_header="X-Hub-Signature-256",
    signature_prefix="sha256=",
    delivery_header="X-GitHub-Delivery",
)


def _field(expression: str, low: int, high: int) -> frozenset[int]:
    values: set[int] = set()
    for item in expression.split(","):
        base, separator, step_text = item.partition("/")
        step = int(step_text) if separator else 1
        if step < 1:
            raise ValueError("cron steps must be positive")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            left, right = base.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = int(base)
            end = high if separator else start
        if not low <= start <= end <= high:
            raise ValueError(f"cron field outside {low}..{high}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def cron_fields(expression: str) -> tuple[frozenset[int], ...]:
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError("cron requires five numeric fields")
    return tuple(
        _field(value, *bounds)
        for value, bounds in zip(
            fields,
            ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7)),
            strict=True,
        )
    )


def cron_matches(trigger: Schedule, instant: datetime) -> bool:
    local = instant.astimezone(ZoneInfo(trigger.timezone))
    if local.fold:
        return False
    minute, hour, day, month, weekday = cron_fields(trigger.cron)
    day_match = local.day in day
    dow = (local.weekday() + 1) % 7
    week_match = dow in weekday or (dow == 0 and 7 in weekday)
    fields = trigger.cron.split()
    calendar_match = (
        day_match and week_match
        if fields[2].startswith("*") or fields[4].startswith("*")
        else day_match or week_match
    )
    return (
        local.minute in minute
        and local.hour in hour
        and local.month in month
        and calendar_match
    )


def _header(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{field} must be a bounded nonempty HTTP header name")
    if any(character not in "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for character in value):
        raise ValueError(f"{field} must be an HTTP token")
    return value


def _webhook(value: dict[str, object]) -> Webhook:
    verify = value.get("verify")
    if verify == "github":
        if set(value) != {"kind", "verify"}:
            raise ValueError("github webhook accepts only kind and verify")
        return _GITHUB
    if verify != "hmac-sha256":
        raise ValueError("unknown or missing webhook verification scheme")
    allowed = {
        "kind",
        "verify",
        "signature_header",
        "signature_prefix",
        "timestamp_header",
    }
    if set(value) - allowed:
        raise ValueError("unknown webhook fields")
    required = {"kind", "verify", "signature_header", "signature_prefix"}
    if not required <= set(value):
        raise ValueError("hmac-sha256 requires signature_header and signature_prefix")
    prefix = value["signature_prefix"]
    if not isinstance(prefix, str) or len(prefix) > 64:
        raise ValueError("signature_prefix must be a bounded string")
    timestamp_header = value.get("timestamp_header")
    return Webhook(
        verify="hmac-sha256",
        signature_header=_header(value["signature_header"], "signature_header"),
        signature_prefix=prefix,
        timestamp_header=(
            _header(timestamp_header, "timestamp_header")
            if timestamp_header is not None
            else None
        ),
    )


def parse_trigger(value: object) -> Trigger:
    if not isinstance(value, dict):
        raise TypeError("trigger must be an object with a kind")
    if value.get("kind") == "schedule":
        if set(value) - {"kind", "cron", "timezone"}:
            raise ValueError("unknown schedule fields")
        cron = value.get("cron")
        timezone = value.get("timezone", "America/Toronto")
        if not isinstance(cron, str) or not isinstance(timezone, str):
            raise ValueError("cron and timezone must be strings")
        cron_fields(cron)
        ZoneInfo(timezone)
        return Schedule(cron, timezone)
    if value.get("kind") == "webhook":
        return _webhook(value)
    if value.get("kind") == "poll":
        if set(value) - {"kind", "condition", "interval_seconds"}:
            raise ValueError("unknown poll fields")
        condition = value.get("condition")
        interval = value.get("interval_seconds", 300)
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("poll requires a nonempty condition")
        if type(interval) is not int or interval < 300:
            raise ValueError("poll interval_seconds must be at least 300")
        return Poll(condition, interval)
    raise ValueError(f"unsupported trigger kind: {value.get('kind')!r}")
