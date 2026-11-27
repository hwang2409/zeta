"""Bounded account facts for local run status, from the existing usage files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TypedDict

from .availability import AccountDisplay
from .registry import AccountId


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
        snapshot = usage.snapshot
        if item.account == account and usage.known_exhausted(now):
            used_percent = 100.0
        if snapshot is None or usage.freshness != "now":
            continue
        windows = [
            window.used_percent for window in snapshot.windows if window.resets_at > now
        ]
        if not windows:
            continue
        if item.account == account:
            used_percent = max(windows)
        elif item.available and not usage.known_exhausted(now):
            alternatives += 1
    return {
        "account": str(account),
        "held": held,
        "usable_alternatives": alternatives,
        "cached_used_percent": used_percent,
    }


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
