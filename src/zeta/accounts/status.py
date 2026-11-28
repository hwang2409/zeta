"""Bounded account facts for local run status, from the existing usage files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, TypedDict

from ..models.catalog import DEFAULT_MODELS
from .availability import AccountDisplay
from .registry import AccountId

if TYPE_CHECKING:
    from .usage import UsageState

# a cached hold that ends sooner than this is left for the child's own hold to wait out
SPAWN_REFUSE_HOLD_OVER = timedelta(minutes=10)
_MODEL_PROVIDER = {"anthropic": "claude", "codex": "codex"}


class AccountStatus(TypedDict):
    account: str
    held: bool
    usable_alternatives: int
    cached_used_percent: float | None


def account_status(
    displays: Sequence[AccountDisplay], account: AccountId, *, held: bool = False
) -> AccountStatus:
    now = datetime.now(UTC)
    alternatives = 0
    used_percent = None
    for item in displays:
        usage = item.usage
        if item.account.provider != account.provider or usage is None:
            continue
        if item.account != account:
            alternatives += _has_headroom(item, now)
            continue
        if usage.known_exhausted(now):
            used_percent = 100.0
        if (windows := _live_windows(usage, now)) and usage.freshness == "now":
            used_percent = max(windows)
    return {
        "account": str(account),
        "held": held,
        "usable_alternatives": alternatives,
        "cached_used_percent": used_percent,
    }


def spawn_hold_refusal(
    usage: UsageState,
    account: AccountId,
    displays: Callable[[], Sequence[AccountDisplay]],
) -> str | None:
    """Name a long cached hold on ``account`` and the accounts with cached headroom.

    ``displays`` reads every account, so it runs only when a refusal is returned.
    """
    now = datetime.now(UTC)
    end = usage.exhausted_until(now)
    if end is None or end - now <= SPAWN_REFUSE_HOLD_OVER:
        return None
    local = end.astimezone().strftime("%b %d %H:%M %Z").lower()
    others = [
        f"{item.account} (model {DEFAULT_MODELS[_MODEL_PROVIDER[item.account.provider]]}"
        + (
            ")"
            if item.account.provider != account.provider
            else "; switch with /accounts)"
        )
        for item in displays()
        if item.account != account and _has_headroom(item, now)
    ]
    return (
        f"agent error: {account} is at its subscription limit until "
        f"{end.isoformat(timespec='seconds')} ({local}, cached); the child was not started. "
        + (
            "accounts with cached headroom: " + ", ".join(others)
            if others
            else "no other account has confirmed cached headroom"
        )
        + ". pass model=... for another provider, or wait for the reset."
    )


def _live_windows(usage: UsageState, now: datetime) -> list[float]:
    snapshot = usage.snapshot
    if snapshot is None:
        return []
    return [
        window.used_percent for window in snapshot.windows if window.resets_at > now
    ]


def _has_headroom(item: AccountDisplay, now: datetime) -> bool:
    usage = item.usage
    return bool(
        item.available
        and usage is not None
        and usage.freshness == "now"
        and _live_windows(usage, now)
        and not usage.known_exhausted(now)
    )


def account_status_text(status: Mapping[str, object]) -> str:
    account = status["account"]
    provider = str(account).partition(":")[0]
    count = status["usable_alternatives"]
    used = status["cached_used_percent"]
    usage = f"{used:g}% used" if type(used) in {int, float} else "usage unknown"
    held = " held" if status["held"] else ""
    noun = "account" if count == 1 else "accounts"
    return f"{account}{held} (cached: {usage}; {count} other {provider} {noun} with headroom)"


def project_account_status(value: object) -> AccountStatus | None:
    if not isinstance(value, Mapping):
        return None
    account = value.get("account")
    held = value.get("held")
    count = value.get("usable_alternatives")
    used = value.get("cached_used_percent")
    if (
        not isinstance(account, str) or type(held) is not bool
        or type(count) is not int or count < 0
    ):
        return None
    if used is not None and (type(used) not in {int, float} or not 0 <= used <= 100):
        return None
    try:
        canonical = AccountId(*account.split(":", 1))
    except (TypeError, ValueError):
        return None
    return {
        "account": str(canonical), "held": held,
        "usable_alternatives": count, "cached_used_percent": used,
    }
