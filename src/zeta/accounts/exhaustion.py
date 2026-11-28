"""Verified subscription-limit classification and recovery choices."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .store import AccountId
from .usage import UsageSnapshot, UsageState, UsageWindow


@dataclass(frozen=True, slots=True)
class SubscriptionLimit:
    account: AccountId
    resets_at: datetime
    window_minutes: int
    # the provider's own recovery hint, shown with the hold when it names the fix
    detail: str | None = None

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
    """Accept only the verified Codex and Anthropic exhaustion shapes."""

    error = _error_object(failure)
    if error is None:
        return None
    if provider == "anthropic":
        return _anthropic_limit(account, failure, error)
    if (
        provider != "codex"
        or getattr(failure, "status_code", None) != 429
        or error.get("type") != "usage_limit_reached"
    ):
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


def _error_object(failure: object) -> Mapping[str, object] | None:
    body = getattr(failure, "response_body", None)
    if not isinstance(body, bytes):
        return None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, Mapping) else None
    return error if isinstance(error, Mapping) else None


# Claude subscription windows: five hours and seven days.
_FIVE_HOURS_MINUTES = 5 * 60
_SEVEN_DAYS_MINUTES = 7 * 24 * 60
_EXTRA_USAGE_PREFIX = "Third-party apps now draw from your extra usage"


def _anthropic_limit(
    account: AccountId, failure: object, error: Mapping[str, object]
) -> SubscriptionLimit | None:
    """Accept the recorded Anthropic Messages exhaustion responses (F073)."""

    from ..providers.completion import MAX_PROVIDER_RETRY_SECONDS
    from ..providers.rate_limit_hold import RATE_LIMIT_PROBE_SECONDS

    status = getattr(failure, "status_code", None)
    now = datetime.now(UTC)
    if status == 429 and error.get("type") == "rate_limit_error":
        # a Retry-After the turn budget can never wait out means the plan window is spent
        retry_after = getattr(failure, "retry_after", None)
        if (
            type(retry_after) not in {int, float}
            or retry_after <= MAX_PROVIDER_RETRY_SECONDS
        ):
            return None
        window = (
            _FIVE_HOURS_MINUTES
            if retry_after <= _FIVE_HOURS_MINUTES * 60
            else _SEVEN_DAYS_MINUTES
        )
        return SubscriptionLimit(account, now + timedelta(seconds=retry_after), window)
    message = error.get("message")
    if (
        status == 400
        and error.get("type") == "invalid_request_error"
        and isinstance(message, str)
        and message.startswith(_EXTRA_USAGE_PREFIX)
    ):
        # no reset time: hold one bounded probe, the same cap as a long rate-limit hold
        return SubscriptionLimit(
            account,
            now + timedelta(seconds=RATE_LIMIT_PROBE_SECONDS),
            int(RATE_LIMIT_PROBE_SECONDS // 60),
            detail=message,
        )
    return None


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
