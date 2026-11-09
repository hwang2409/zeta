"""Decide and describe provider rate-limit holds."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

# cap a long provider retry at a bounded probe; credential changes can wake it sooner.
RATE_LIMIT_PROBE_SECONDS = 300.0
# poll credential and account generations while a rate-limit hold is active.
RATE_LIMIT_CREDENTIAL_POLL_SECONDS = 5.0
# floor for a subscription-limit probe, so a reset that already passed cannot spin.
SUBSCRIPTION_LIMIT_MIN_PROBE_SECONDS = 60.0
# a background child held longer than this wakes its parent once per hold.
LONG_HOLD_WAKE_SECONDS = 15 * 60.0


def retry_after_seconds(
    error: BaseException | object,
    event_data: Mapping[str, Any] | None = None,
) -> float | None:
    """Read a provider retry-after value from an error or emitted event."""
    value = getattr(error, "retry_after", None)
    if value is None and event_data is not None:
        value = event_data.get("retry_after")
    return max(0.0, float(value)) if type(value) in {int, float} else None


def is_rate_limit_hold(error: BaseException | object, retry_after: float | None) -> bool:
    """Return whether a retry should wait as a rate-limit hold."""
    return retry_after is not None and getattr(error, "status_code", None) in {429, 503, 529}


def credential_fingerprint(token_store: object) -> tuple[int, int] | None:
    """Stat the on-disk credential file; a re-login changes it and wakes a hold."""
    path = getattr(token_store, "path", None)
    if not isinstance(path, Path):
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def subscription_limit_wait(resets_at: float, now: float) -> float:
    """Wait until the subscription reset; the hold wakes sooner on a re-login or switch."""
    return max(resets_at - now, SUBSCRIPTION_LIMIT_MIN_PROBE_SECONDS)


def rate_limit_hold_text(data: Mapping[str, Any]) -> str:
    """Describe a rate-limit hold event as local wall time plus the next probe."""

    until = datetime.fromtimestamp(data["until"]).astimezone()
    today = datetime.now().astimezone().date()
    stamp = until.strftime("%H:%M" if until.date() == today else "%b %d %H:%M")
    if data.get("limit") == "subscription":
        return f"subscription limit until {stamp}; switch account or log in to resume"
    return (
        f"rate-limited until {stamp} (provider asks {format_duration(data['retry_after'])}); "
        f"checking again in {format_duration(data['delay'])}"
    )


def format_duration(seconds: float) -> str:
    """Format a non-negative duration for a hold notice."""

    total = max(0, int(seconds))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = [
        f"{value}{unit}"
        for value, unit in ((days, "d"), (hours, "h"), (minutes, "m"))
        if value
    ]
    if secs and not parts:
        parts.append(f"{secs}s")
    return " ".join(parts) or "0s"
