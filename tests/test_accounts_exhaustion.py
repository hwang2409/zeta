from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.credentials import named_account_credentials
from zeta.accounts import AccountId, AccountStore
from zeta.accounts.exhaustion import (
    SubscriptionLimit,
    classify_subscription_limit,
    recovery_alternatives,
)
from zeta.accounts.usage import (
    UsageSnapshot,
    UsageState,
    UsageWindow,
    format_usage_lines,
    read_usage,
    record_subscription_limit,
)
from zeta.oauth import OAuthTokens
from zeta.providers.retry_policy import ProviderRetryBudget

FIXTURE = Path(__file__).parent / "fixtures/usage/codex_responses_429_usage_limit_reached.json"
NOW = datetime(2026, 10, 22, 16, 0, tzinfo=UTC)
ACCOUNT = AccountId("codex", "pro")


def _failure(status: int, body: object) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, response_body=json.dumps(body).encode())


def _state(percent: float, *, age: timedelta = timedelta(), provider: str = "codex") -> UsageState:
    kind = "primary" if provider == "codex" else "five_hour"
    return UsageState(
        UsageSnapshot(NOW - age, "usage_endpoint", (UsageWindow(kind, percent, 10_080, NOW + timedelta(days=1)),)),
        "now" if age <= timedelta(minutes=15) else "stale",
    )


def test_codex_usage_limit_reached_fixture_is_classified() -> None:
    payload = json.loads(FIXTURE.read_text())
    result = classify_subscription_limit("codex", ACCOUNT, _failure(429, {"error": payload["error"]}))
    assert result == SubscriptionLimit(ACCOUNT, datetime.fromtimestamp(1793082708, UTC), 10080)


def test_unknown_429_and_changed_schema_are_not_exhaustion() -> None:
    assert classify_subscription_limit("codex", ACCOUNT, _failure(429, {"error": {"type": "other"}})) is None
    assert classify_subscription_limit("codex", ACCOUNT, _failure(429, {"error": {"type": "usage_limit_reached", "resets_at": "soon", "limit_window_minutes": 10080}})) is None
    assert classify_subscription_limit("codex", ACCOUNT, _failure(500, {"error": {"type": "usage_limit_reached", "resets_at": 1793082708, "limit_window_minutes": 10080}})) is None


def test_anthropic_429_is_not_classified_without_fixture() -> None:
    assert classify_subscription_limit("anthropic", AccountId("anthropic", "pro"), _failure(429, {"error": {"type": "usage_limit_reached", "resets_at": 1793082708, "limit_window_minutes": 10080}})) is None


def test_classified_limit_is_not_retried_and_other_retries_unchanged() -> None:
    exhausted = _failure(429, json.loads(FIXTURE.read_text()))
    exhausted.subscription_limit = classify_subscription_limit("codex", ACCOUNT, exhausted)
    budget = ProviderRetryBudget()
    budget.start_attempt("test")
    assert budget.plan(exhausted, owner="test") is None
    assert budget.plan(SimpleNamespace(status_code=429), owner="test") is not None


def test_limit_marks_account_known_exhausted_until_reset(tmp_path: Path) -> None:
    named_account_credentials(tmp_path, "codex", "pro").save(
        OAuthTokens("access", "refresh", 9_999_999_999)
    )
    store = AccountStore(tmp_path)
    limit = SubscriptionLimit(ACCOUNT, NOW + timedelta(hours=1), 10080)
    state = limit.usage_state(NOW)
    assert state.snapshot is not None
    record_subscription_limit(store, ACCOUNT, state.snapshot, NOW)

    persisted = read_usage(store, ACCOUNT, now=NOW)
    assert persisted.known_exhausted(NOW)
    assert format_usage_lines(persisted)[0].startswith("limit reached · resets ")
    assert not persisted.known_exhausted(NOW + timedelta(hours=1, seconds=1))


def test_offers_switch_only_with_exactly_one_fresh_unexhausted_alternative() -> None:
    a, b, c = AccountId("codex", "a"), AccountId("codex", "b"), AccountId("codex", "c")
    assert recovery_alternatives(a, {a: _state(100), b: _state(20)}, NOW) == (b,)
    assert recovery_alternatives(a, {a: _state(100)}, NOW) == ()
    assert recovery_alternatives(a, {a: _state(100), b: _state(20), c: _state(30)}, NOW) == ()
    assert recovery_alternatives(a, {a: _state(100), b: _state(20, age=timedelta(minutes=16))}, NOW) == ()
    assert recovery_alternatives(a, {a: _state(100), b: _state(100)}, NOW) == ()


def test_switch_never_replays_failed_request() -> None:
    # Recovery returns a selection only; the caller must wait for new user input.
    other = AccountId("codex", "other")
    assert recovery_alternatives(ACCOUNT, {ACCOUNT: _state(100), other: _state(1)}, NOW) == (other,)


def test_child_limit_failure_never_switches_parent() -> None:
    limit = SubscriptionLimit(ACCOUNT, NOW + timedelta(hours=1), 10080)
    assert limit.recovery_event(frozen=True, alternatives=(AccountId("codex", "other"),))["action"] == "none"


@pytest.mark.asyncio
async def test_limit_recovery_does_not_block_event_loop_while_usage_lock_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zeta.accounts.session import SessionAccountContext

    entered = threading.Event()
    release = threading.Event()

    def blocking_recovery(_self: object, _failure: object) -> None:
        entered.set()
        release.wait(timeout=2)

    monkeypatch.setattr(SessionAccountContext, "_subscription_limit_event", blocking_recovery)
    context = SessionAccountContext.__new__(SessionAccountContext)
    task = asyncio.create_task(context.subscription_limit_event(object()))
    await asyncio.sleep(0)
    assert entered.wait(timeout=1)
    ticks = 0
    for _ in range(3):
        await asyncio.sleep(0)
        ticks += 1
    assert ticks == 3
    release.set()
    await task
