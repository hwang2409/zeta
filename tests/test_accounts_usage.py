from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, AccountStore
from zeta.accounts.usage import (
    AccountUsage,
    UsageSnapshot,
    UsageWindow,
    normalize_usage,
    read_usage,
    usage_freshness,
)
from zeta.oauth import OAuthTokens

FIXTURES = Path(__file__).parent / "fixtures" / "usage"
NOW = datetime(2026, 10, 22, 16, 0, tzinfo=UTC)


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text())["response"]


def _tokens() -> OAuthTokens:
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "account-test"
                }
            }
        ).encode()
    ).decode().rstrip("=")
    return OAuthTokens(f"e30.{payload}.signature", "refresh", 9_999_999_999)


def _seed(store: AccountStore, *accounts: AccountId) -> None:
    for account in accounts:
        named_account_credentials(store.home, account.provider, account.name).save(_tokens())


def test_anthropic_fixture_normalizes_five_hour_and_seven_day() -> None:
    snapshot = normalize_usage("anthropic", _fixture("anthropic_oauth_usage.json"), NOW)

    assert snapshot == UsageSnapshot(
        captured_at=NOW,
        source="usage_endpoint",
        windows=(
            UsageWindow(
                "five_hour",
                3.0,
                300,
                datetime(2026, 10, 22, 18, 49, 59, 770796, tzinfo=UTC),
            ),
            UsageWindow(
                "seven_day",
                21.0,
                10_080,
                datetime(2026, 10, 28, 14, 59, 59, 770826, tzinfo=UTC),
            ),
        ),
    )


def test_codex_fixture_normalizes_primary() -> None:
    snapshot = normalize_usage("codex", _fixture("codex_wham_usage.json"), NOW)

    assert snapshot.windows == (
        UsageWindow(
            "primary",
            6.0,
            10_080,
            datetime.fromtimestamp(1_793_251_485, UTC),
        ),
    )


def test_rejects_out_of_range_nonfinite_and_unknown_values() -> None:
    base = {
        "five_hour": {"utilization": 10, "resets_at": "2026-10-22T18:00:00Z"},
        "seven_day": {"utilization": 20, "resets_at": "2026-10-29T18:00:00Z"},
    }
    invalid = [
        {**base, "five_hour": {**base["five_hour"], "utilization": -1}},
        {**base, "five_hour": {**base["five_hour"], "utilization": 101}},
        {**base, "five_hour": {**base["five_hour"], "utilization": math.nan}},
        {**base, "five_hour": {"resets_at": "2026-10-22T18:00:00Z"}},
        {**base, "five_hour": {**base["five_hour"], "resets_at": "unknown"}},
    ]
    for value in invalid:
        with pytest.raises(ValueError):
            normalize_usage("anthropic", value, NOW)
    with pytest.raises(ValueError):
        normalize_usage("other", base, NOW)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(401), "authentication failed"),
        (httpx.Response(429), "rate limited"),
        (httpx.Response(200, json={"changed": True}), "invalid usage response"),
    ],
)
def test_failed_refresh_keeps_previous_snapshot_and_reports_safe_error(
    tmp_path: Path, response: httpx.Response, expected: str
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "work")
    _seed(store, account)
    good = _fixture("anthropic_oauth_usage.json")
    replies = iter((httpx.Response(200, json=good), response))
    usage = AccountUsage(
        store,
        clock=lambda: NOW,
        transport=httpx.MockTransport(lambda request: next(replies)),
    )
    assert asyncio.run(usage.refresh_stale())[account].error is None

    later = NOW + timedelta(minutes=16)
    usage = AccountUsage(
        store,
        clock=lambda: later,
        transport=httpx.MockTransport(lambda request: next(replies)),
    )
    result = asyncio.run(usage.refresh_stale())[account]

    assert result.error == expected
    assert result.snapshot is not None
    assert result.snapshot.captured_at == NOW
    assert expected in store.list_display()[0].usage.error
    assert asyncio.run(usage.refresh_stale()) == {}


def test_refresh_is_sequential_one_request_per_account_and_respects_cooldown(
    tmp_path: Path,
) -> None:
    store = AccountStore(tmp_path)
    accounts = (AccountId("anthropic", "a"), AccountId("codex", "b"))
    _seed(store, *accounts)
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        payload = (
            _fixture("anthropic_oauth_usage.json")
            if "anthropic" in request.url.host
            else _fixture("codex_wham_usage.json")
        )
        return httpx.Response(200, json=payload)

    usage = AccountUsage(
        store,
        clock=lambda: NOW,
        transport=httpx.MockTransport(handler),
    )
    first = asyncio.run(usage.refresh_stale())
    second = asyncio.run(usage.refresh_stale())

    assert list(first) == list(accounts)
    assert second == {}
    assert requested == [
        "https://api.anthropic.com/api/oauth/usage",
        "https://chatgpt.com/backend-api/wham/usage",
    ]


def test_refresh_has_total_deadline_and_releases_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "work")
    _seed(store, account)
    monkeypatch.setattr("zeta.accounts.usage.REQUEST_TIMEOUT_SECONDS", 0.05)

    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b"x"
                await asyncio.sleep(0.02)

        async def aclose(self) -> None:
            pass

    usage = AccountUsage(
        store,
        clock=lambda: NOW,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Drip())
        ),
    )
    started = time.monotonic()
    result = asyncio.run(usage.refresh_stale())[account]
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert result.error == "usage request failed"
    with store.lifecycle_guard(account):
        pass


def _write_in_flight(store: AccountStore, account: AccountId, started: datetime) -> None:
    store._write_usage_bytes(
        account,
        (
            json.dumps(
                {
                    "version": 1,
                    "captured_at": NOW.isoformat(),
                    "source": "usage_endpoint",
                    "windows": [
                        {
                            "kind": "five_hour",
                            "used_percent": 1,
                            "window_minutes": 300,
                            "resets_at": "2026-10-22T18:00:00Z",
                        }
                    ],
                    "last_started_at": started.isoformat(),
                    "last_finished_at": None,
                    "outcome": "in_flight",
                    "last_error": None,
                }
            )
            + "\n"
        ).encode(),
    )


def test_abandoned_in_flight_attempt_shows_safe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "work")
    _seed(store, account)
    monkeypatch.setattr("zeta.accounts.usage.REQUEST_TIMEOUT_SECONDS", 0.05)
    _write_in_flight(store, account, NOW - timedelta(seconds=2))

    state = read_usage(store, account, now=NOW)
    assert state.error == "usage request failed"


def test_active_in_flight_attempt_is_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "work")
    _seed(store, account)
    monkeypatch.setattr("zeta.accounts.usage.REQUEST_TIMEOUT_SECONDS", 15)
    _write_in_flight(store, account, NOW)

    state = read_usage(store, account, now=NOW)
    assert state.error is None


def test_freshness_now_stale_unknown() -> None:
    snapshot = UsageSnapshot(NOW, "usage_endpoint", ())
    assert usage_freshness(snapshot, NOW + timedelta(minutes=15)) == "now"
    assert usage_freshness(snapshot, NOW + timedelta(minutes=15, seconds=1)) == "stale"
    assert usage_freshness(None, NOW) == "usage unknown"


def test_usage_write_is_atomic_and_display_read_is_lock_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AccountStore(tmp_path)
    account = AccountId("anthropic", "work")
    _seed(store, account)
    replacements: list[tuple[object, object]] = []
    original_replace = os.replace

    def replace(source: object, target: object, **kwargs: object) -> None:
        replacements.append((source, target))
        original_replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    usage = AccountUsage(
        store,
        clock=lambda: NOW,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=_fixture("anthropic_oauth_usage.json")
            )
        ),
    )
    asyncio.run(usage.refresh_stale())
    usage_path = tmp_path / "accounts" / "anthropic" / "work" / "usage.json"
    assert usage_path.stat().st_mode & 0o777 == 0o600
    assert any(target == "usage.json" for _, target in replacements)

    monkeypatch.setattr(
        AccountStore,
        "usage_guard",
        lambda *args, **kwargs: pytest.fail("display read took usage lock"),
    )
    display = store.list_display()[0]
    assert display.usage.snapshot is not None
