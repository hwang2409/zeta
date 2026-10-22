"""Verified explicit subscription-usage refresh for named accounts."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .store import AccountId, AccountStore

FRESH_FOR = timedelta(minutes=15)
REFRESH_COOLDOWN = timedelta(seconds=60)
REQUEST_TIMEOUT_SECONDS = 15.0
IN_FLIGHT_GRACE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class UsageWindow:
    kind: str
    used_percent: float
    window_minutes: int
    resets_at: datetime


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    captured_at: datetime
    source: str
    windows: tuple[UsageWindow, ...]


@dataclass(frozen=True, slots=True)
class UsageState:
    snapshot: UsageSnapshot | None
    freshness: str
    error: str | None = None
    last_attempt_at: datetime | None = None


def usage_freshness(snapshot: UsageSnapshot | None, now: datetime) -> str:
    if snapshot is None:
        return "usage unknown"
    return "now" if now - snapshot.captured_at <= FRESH_FOR else "stale"


def normalize_usage(provider: str, value: object, captured_at: datetime) -> UsageSnapshot:
    if not isinstance(value, Mapping):
        raise TypeError("invalid usage response")
    if provider == "anthropic":
        windows = tuple(
            _anthropic_window(value, source, kind, minutes)
            for source, kind, minutes in (
                ("five_hour", "five_hour", 300),
                ("seven_day", "seven_day", 10_080),
            )
            if value.get(source) is not None
        )
    elif provider == "codex":
        rate_limit = value.get("rate_limit")
        if not isinstance(rate_limit, Mapping):
            raise ValueError("invalid usage response")
        windows = (_codex_window(rate_limit, "primary_window", "primary"),)
    else:
        raise ValueError("unknown usage provider")
    if not windows:
        raise ValueError("invalid usage response")
    return UsageSnapshot(_utc(captured_at), "usage_endpoint", windows)


def _percent(value: object) -> float:
    if type(value) not in {int, float}:
        raise ValueError("invalid usage response")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 100:
        raise ValueError("invalid usage response")
    return result


def _iso_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("invalid usage response")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid usage response") from exc
    if parsed.tzinfo is None:
        raise ValueError("invalid usage response")
    return parsed.astimezone(UTC)


def _anthropic_window(
    value: Mapping[str, object], source: str, kind: str, minutes: int
) -> UsageWindow:
    window = value.get(source)
    if not isinstance(window, Mapping):
        raise TypeError("invalid usage response")
    return UsageWindow(
        kind,
        _percent(window.get("utilization")),
        minutes,
        _iso_datetime(window.get("resets_at")),
    )


def _codex_window(
    rate_limit: Mapping[str, object], source: str, kind: str
) -> UsageWindow:
    window = rate_limit.get(source)
    if not isinstance(window, Mapping):
        raise TypeError("invalid usage response")
    seconds = window.get("limit_window_seconds")
    reset_after = window.get("reset_after_seconds")
    reset_at = window.get("reset_at")
    if (
        type(seconds) is not int
        or seconds <= 0
        or seconds % 60
        or type(reset_after) is not int
        or reset_after < 0
        or type(reset_at) is not int
        or reset_at < 0
    ):
        raise ValueError("invalid usage response")
    try:
        resets_at = datetime.fromtimestamp(reset_at, UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("invalid usage response") from exc
    return UsageWindow(
        kind,
        _percent(window.get("used_percent")),
        seconds // 60,
        resets_at,
    )


def read_usage(store: AccountStore, account: AccountId, *, now: datetime | None = None) -> UsageState:
    now = _utc(now or datetime.now(UTC))
    try:
        raw = store._read_usage_bytes(account)
        if raw is None:
            return UsageState(None, "usage unknown")
        value = json.loads(raw)
        if not isinstance(value, Mapping) or value.get("version") != 1:
            raise ValueError
        snapshot = _snapshot_from_json(value) if value.get("captured_at") else None
        attempt = (
            _iso_datetime(value["last_started_at"])
            if value.get("last_started_at") is not None
            else None
        )
        if value.get("last_finished_at") is not None:
            _iso_datetime(value["last_finished_at"])
        outcome = value.get("outcome")
        if outcome is not None and outcome not in {"in_flight", "succeeded", "failed"}:
            raise ValueError
        error = value.get("last_error")
        if error is not None and error not in {
            "authentication failed",
            "rate limited",
            "invalid usage response",
            "usage request failed",
        }:
            raise ValueError
        if (
            outcome == "in_flight"
            and attempt is not None
            and now - attempt > timedelta(
                seconds=REQUEST_TIMEOUT_SECONDS + IN_FLIGHT_GRACE_SECONDS
            )
        ):
            error = "usage request failed"
        return UsageState(snapshot, usage_freshness(snapshot, now), error, attempt)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return UsageState(None, "usage unknown", "invalid usage response")


class AccountUsage:
    """Refresh stale accounts sequentially and preserve safe durable state."""

    def __init__(
        self,
        store: AccountStore,
        *,
        clock: Callable[[], datetime] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.store = store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._transport = transport

    async def refresh_stale(self) -> dict[AccountId, UsageState]:
        results: dict[AccountId, UsageState] = {}
        timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            follow_redirects=False,
        ) as client:
            for account in self.store.list():
                now = _utc(self._clock())
                state = read_usage(self.store, account, now=now)
                if state.freshness == "now":
                    continue
                result = await self._refresh_one(account, state, now, client)
                if result is not None:
                    results[account] = result
        return results

    async def _refresh_one(
        self,
        account: AccountId,
        state: UsageState,
        now: datetime,
        client: httpx.AsyncClient,
    ) -> UsageState | None:
        with (
            self.store.lifecycle_guard(account) as home_fd,
            self.store.usage_guard(account),
        ):
            state = read_usage(self.store, account, now=now)
            if state.freshness == "now" or (
                state.last_attempt_at is not None
                and now - state.last_attempt_at < REFRESH_COOLDOWN
            ):
                return None
            self._write(account, state.snapshot, now, None, "in_flight")
            error: str | None = None
            try:
                response = await asyncio.wait_for(
                    self._request(account, home_fd, client), REQUEST_TIMEOUT_SECONDS
                )
                if response.status_code in {401, 403}:
                    error = "authentication failed"
                elif response.status_code == 429:
                    error = "rate limited"
                elif response.status_code != 200:
                    error = "usage request failed"
                else:
                    snapshot = normalize_usage(account.provider, response.json(), now)
                    self._write(account, snapshot, now, None, "succeeded", finished=now)
                    return UsageState(snapshot, "now", None, now)
            except (httpx.HTTPError, TimeoutError):
                error = "usage request failed"
            except RuntimeError:
                error = "authentication failed"
            except (TypeError, ValueError, json.JSONDecodeError):
                error = "invalid usage response"
            self._write(account, state.snapshot, now, error, "failed", finished=now)
            return UsageState(state.snapshot, usage_freshness(state.snapshot, now), error, now)

    async def _request(
        self, account: AccountId, home_fd: int, client: httpx.AsyncClient
    ) -> httpx.Response:
        from ..codex import CodexCredentialDriver
        from ..providers.anthropic import AnthropicCredentialDriver
        from .store import AccountCredentials

        driver = (
            AnthropicCredentialDriver()
            if account.provider == "anthropic"
            else CodexCredentialDriver()
        )
        credentials = AccountCredentials(self.store, account, driver)
        token = await credentials._access_token_locked(home_fd, client)
        headers = credentials.auth_headers(token)
        headers["accept"] = "application/json"
        if account.provider == "anthropic":
            headers.update(
                {
                    "anthropic-beta": credentials.beta_header(),
                    "user-agent": "claude-code",
                }
            )
            endpoint = "https://api.anthropic.com/api/oauth/usage"
        else:
            headers["user-agent"] = "codex-cli"
            endpoint = "https://chatgpt.com/backend-api/wham/usage"
        return await client.get(endpoint, headers=headers)

    def _write(
        self,
        account: AccountId,
        snapshot: UsageSnapshot | None,
        attempt: datetime,
        error: str | None,
        outcome: str,
        *,
        finished: datetime | None = None,
    ) -> None:
        value: dict[str, Any] = {
            "version": 1,
            "captured_at": None,
            "source": None,
            "windows": [],
            "last_started_at": _iso(attempt),
            "last_finished_at": _iso(finished) if finished else None,
            "outcome": outcome,
            "last_error": error,
        }
        if snapshot is not None:
            value.update(_snapshot_json(snapshot))
        data = (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()
        self.store._write_usage_bytes(account, data)


def format_usage_lines(state: UsageState) -> tuple[str, ...]:
    if state.snapshot is None:
        lines = ["usage unknown"]
    else:
        now = datetime.now().astimezone()
        lines = []
        for window in state.snapshot.windows:
            reset = window.resets_at.astimezone()
            reset_text = (
                reset.strftime("%H:%M")
                if reset.date() == now.date()
                else reset.strftime("%b %d %H:%M").lower()
            )
            percent = f"{window.used_percent:g}%"
            lines.append(
                f"{_window_label(window)} {percent} · resets {reset_text}"
            )
        lines.append(state.freshness)
    if state.error:
        lines.append(f"refresh failed: {state.error}")
    return tuple(lines)


def _window_label(window: UsageWindow) -> str:
    if window.window_minutes % (24 * 60) == 0:
        return f"{window.window_minutes // (24 * 60)}d"
    if window.window_minutes % 60 == 0:
        return f"{window.window_minutes // 60}h"
    return f"{window.window_minutes}m"


def usage_json(state: UsageState) -> dict[str, object]:
    return {
        "freshness": state.freshness,
        "error": state.error,
        "snapshot": _snapshot_json(state.snapshot) if state.snapshot else None,
    }


def _snapshot_json(snapshot: UsageSnapshot) -> dict[str, object]:
    return {
        "version": 1,
        "captured_at": _iso(snapshot.captured_at),
        "source": snapshot.source,
        "windows": [
            {
                "kind": window.kind,
                "used_percent": window.used_percent,
                "window_minutes": window.window_minutes,
                "resets_at": _iso(window.resets_at),
            }
            for window in snapshot.windows
        ],
    }


def _snapshot_from_json(value: Mapping[str, object]) -> UsageSnapshot:
    source = value.get("source")
    windows = value.get("windows")
    if source != "usage_endpoint" or not isinstance(windows, list) or not windows:
        raise ValueError
    parsed: list[UsageWindow] = []
    for raw in windows:
        if not isinstance(raw, Mapping):
            raise TypeError
        kind = raw.get("kind")
        minutes = raw.get("window_minutes")
        if kind not in {"five_hour", "seven_day", "primary"}:
            raise ValueError
        if type(minutes) is not int or minutes <= 0:
            raise ValueError
        parsed.append(
            UsageWindow(
                kind,
                _percent(raw.get("used_percent")),
                minutes,
                _iso_datetime(raw.get("resets_at")),
            )
        )
    return UsageSnapshot(_iso_datetime(value.get("captured_at")), source, tuple(parsed))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")
