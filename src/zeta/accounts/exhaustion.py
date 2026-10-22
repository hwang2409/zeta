"""Verified subscription-limit classification and recovery choices."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from .store import AccountId
from .usage import UsageSnapshot, UsageState, UsageWindow


@dataclass(frozen=True, slots=True)
class SubscriptionLimit:
    account: AccountId
    resets_at: datetime
    window_minutes: int

    def usage_state(self, captured_at: datetime) -> UsageState:
        snapshot = UsageSnapshot(
            captured_at.astimezone(UTC),
            "provider_error",
            (UsageWindow("primary", 100.0, self.window_minutes, self.resets_at),),
        )
        return UsageState(snapshot, "now")

    def recovery_event(
        self, *, frozen: bool, alternatives: tuple[AccountId, ...]
    ) -> dict[str, object]:
        candidate = alternatives[0] if not frozen and len(alternatives) == 1 else None
        return {
            "account": str(self.account),
            "resets_at": self.resets_at.isoformat().replace("+00:00", "Z"),
            "window_minutes": self.window_minutes,
            "reset_local": self.resets_at.astimezone().strftime("%b %d %H:%M %Z").lower(),
            "action": "none" if frozen else ("switch" if candidate else "accounts"),
            "alternative": str(candidate) if candidate else None,
        }


def classify_subscription_limit(
    provider: str, account: AccountId, failure: object
) -> SubscriptionLimit | None:
    """Accept only the verified Codex Responses HTTP 429 shape."""

    if provider != "codex" or getattr(failure, "status_code", None) != 429:
        return None
    body = getattr(failure, "response_body", None)
    if not isinstance(body, bytes):
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, Mapping) or error.get("type") != "usage_limit_reached":
        return None
    resets_at = error.get("resets_at")
    window_minutes = error.get("limit_window_minutes")
    if (
        type(resets_at) is not int
        or resets_at < 0
        or type(window_minutes) is not int
        or window_minutes <= 0
    ):
        return None
    try:
        reset = datetime.fromtimestamp(resets_at, UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return SubscriptionLimit(account, reset, window_minutes)


def recovery_alternatives(
    current: AccountId,
    states: Mapping[AccountId, UsageState],
    now: datetime,
) -> tuple[AccountId, ...]:
    eligible = tuple(
        account
        for account, state in states.items()
        if account.provider == current.provider
        and account != current
        and state.freshness == "now"
        and state.snapshot is not None
        and not state.known_exhausted(now)
    )
    return eligible if len(eligible) == 1 else ()
