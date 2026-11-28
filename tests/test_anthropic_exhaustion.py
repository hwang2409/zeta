"""Anthropic account exhaustion holds the turn like the Codex subscription limit (F073)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from tests.support.credentials import named_account_credentials
from tests.support.owned import await_or_fail
from tests.test_anthropic import SSE
from tests.test_rate_limit_hold import _collect
from zeta.accounts import AccountId, AccountStore, SessionAccountContext
from zeta.accounts.exhaustion import classify_subscription_limit
from zeta.accounts.usage import read_usage
from zeta.core.session import SessionManager
from zeta.oauth import OAuthTokens
from zeta.protocol.types import MessageOrigin, StreamEventType
from zeta.providers.anthropic import AnthropicBackend
from zeta.providers.anthropic_errors import http_error
from zeta.providers.completion import CompletionRetryBudget
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.storage_format import advance
from zeta.tui.render import render_event

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")

FIXTURES = Path(__file__).parent / "fixtures/usage"
ACCOUNT = AccountId("anthropic", "first")
EXTRA_USAGE = (
    "Third-party apps now draw from your extra usage, not your plan limits. "
    "Add more at claude.ai/settings/usage and keep going."
)


def _recorded(name: str) -> tuple[int, dict[str, str], bytes]:
    payload = json.loads((FIXTURES / name).read_text())
    meta = payload.pop("_fixture")
    return meta["http_status"], meta["headers"], json.dumps(payload).encode()


RATE_LIMIT_429 = _recorded("anthropic_messages_429_rate_limit_error.json")
EXTRA_USAGE_400 = _recorded("anthropic_messages_400_extra_usage.json")


def _error(recorded: tuple[int, dict[str, str], bytes]):
    status, headers, body = recorded
    return http_error(status, body, headers)


def test_recorded_long_retry_after_429_is_account_exhaustion() -> None:
    before = datetime.now(UTC)
    limit = classify_subscription_limit("anthropic", ACCOUNT, _error(RATE_LIMIT_429))
    assert limit is not None, "a 2.5-day Retry-After 429 is an exhausted account"
    assert limit.account == ACCOUNT
    assert before + timedelta(seconds=213277) <= limit.resets_at
    assert limit.resets_at <= datetime.now(UTC) + timedelta(seconds=213277)
    # only the seven-day window can reset more than five hours out
    assert limit.window_minutes == 7 * 24 * 60


def test_recorded_extra_usage_400_is_account_exhaustion() -> None:
    before = datetime.now(UTC)
    limit = classify_subscription_limit("anthropic", ACCOUNT, _error(EXTRA_USAGE_400))
    assert limit is not None, "the extra-usage 400 is an exhausted account"
    # no reset time: hold for one bounded probe, like a long rate-limit hold
    assert before + timedelta(seconds=300) <= limit.resets_at
    assert limit.resets_at <= datetime.now(UTC) + timedelta(seconds=300)
    assert limit.detail == EXTRA_USAGE


def test_short_retry_after_429_still_retries_inside_budget() -> None:
    status, _headers, body = RATE_LIMIT_429
    error = http_error(status, body, {"retry-after": "5"})
    assert classify_subscription_limit("anthropic", ACCOUNT, error) is None
    budget = CompletionRetryBudget()
    budget.start_attempt("completion", 0.0)
    plan = budget.plan(error, 0.0)
    assert plan is not None and not plan.is_rate_limit_hold
    assert plan.subscription_limit is None
    assert plan.delay == 5.0


@pytest.mark.parametrize("value", ["inf", "1e15", "Fri, 31 Dec 9999 23:59:59 GMT"])
def test_absurd_retry_after_is_no_value(value: str) -> None:
    status, _headers, body = RATE_LIMIT_429
    try:
        error = http_error(status, body, {"retry-after": value}, account=ACCOUNT)
    except OverflowError:
        error = None
    assert error is not None, "an absurd Retry-After must not replace the HTTP error"
    assert error.status_code == 429
    assert error.retry_after is None
    assert getattr(error, "subscription_limit", None) is None


def test_other_anthropic_400_is_not_exhaustion() -> None:
    body = json.dumps(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    ).encode()
    assert (
        classify_subscription_limit("anthropic", ACCOUNT, http_error(400, body)) is None
    )


def test_codex_classifier_ignores_anthropic_shapes() -> None:
    for recorded in (RATE_LIMIT_429, EXTRA_USAGE_400):
        failure = _error(recorded)
        assert (
            classify_subscription_limit("codex", AccountId("codex", "pro"), failure)
            is None
        )


class _Held:
    """Two anthropic accounts; the first returns one recorded exhaustion response."""

    def __init__(
        self, tmp_path: Path, recorded: tuple[int, dict[str, str], bytes]
    ) -> None:
        self.recorded = recorded
        # the first account answers once with `recorded`, then recovers
        self.exhausted_once = False
        home = tmp_path / "home"
        advance(home, expected_epoch=0)
        self.first = ACCOUNT
        self.second = AccountId("anthropic", "second")
        for account in (self.first, self.second):
            named_account_credentials(home, "anthropic", account.name).save(
                OAuthTokens(f"token-{account.name}", "refresh", 4_000_000_000)
            )
        store = AccountStore(home)
        store.set_default(self.first)
        manager = SessionManager(home)
        self.opened = manager.create(
            provider="anthropic", model="claude-opus-5-5", cwd=tmp_path
        )
        self.context = SessionAccountContext.load(home, manager, self.opened.metadata)
        self.requests: list[httpx.Request] = []
        self.waits: list[float] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        self.backends = {
            account: AnthropicBackend(
                client=self.client,
                token_store=named_account_credentials(home, "anthropic", account.name),
                base_url="https://test.invalid/v1/messages",
            )
            for account in (self.first, self.second)
        }
        self.loop = AgentLoop(
            self.backends[self.first],
            self.opened.store,
            skill_catalog=SkillCatalog.empty(),
        )
        self.loop.manager = self.context.manager
        self.loop.session_metadata = self.context.metadata
        self.loop.account_context = self.context

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers["authorization"] == "Bearer token-first" and (
            not self.exhausted_once or len(self.requests) == 1
        ):
            status, headers, body = self.recorded
            return httpx.Response(
                status, headers=headers, content=body, request=request
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=SSE,
            request=request,
        )

    async def run(self, *, user_switches: bool = True):
        async def user_switches_on_first_poll(delay: float) -> None:
            self.waits.append(delay)
            if not user_switches:
                return
            await self.context.switch_active(
                self.loop,
                self.second,
                boundary=self.opened.store.active_branch_head_sequence(),
                provider="anthropic",
                build_backend=lambda account: self.backends[account],
            )

        for backend in self.backends.values():
            backend._completion_wait = user_switches_on_first_poll
        try:
            return await await_or_fail(
                _collect(self.loop.run_turn("hello", origin=MessageOrigin.USER)),
                what="held anthropic turn",
            )
        finally:
            await self.loop.close()
            await self.client.aclose()
            self.opened.store.close()


@pytest.mark.parametrize(
    ("recorded", "text"),
    [
        (
            RATE_LIMIT_429,
            (
                r"^anthropic:first subscription limit until \w{3} \d\d \d\d:\d\d; "
                r"switch account or log in to resume$"
            ),
        ),
        (
            EXTRA_USAGE_400,
            (
                r"^anthropic:first subscription limit until (\w{3} \d\d )?\d\d:\d\d; "
                r"switch account or log in to resume \("
                + re.escape(EXTRA_USAGE)
                + r"\)$"
            ),
        ),
    ],
    ids=["429-retry-after", "400-extra-usage"],
)
@pytest.mark.asyncio
async def test_recorded_exhaustion_holds_turn_with_plain_message(
    tmp_path: Path, recorded: tuple[int, dict[str, str], bytes], text: str
) -> None:
    held = await asyncio.to_thread(_Held, tmp_path, recorded)
    events = await held.run()

    errors = [event.error for event in events if event.type is StreamEventType.ERROR]
    assert errors == [], (
        "an exhausted account must hold the turn, not end it with raw JSON"
    )
    hold = next(event for event in events if event.type is StreamEventType.RETRY)
    assert hold.data["limit"] == "subscription"
    assert hold.data["account"] == "anthropic:first"
    assert re.match(text, hold.data["text"]), hold.data["text"]
    assert render_event(hold).plain == hold.data["text"]
    assert "{" not in hold.data["text"]
    suggestion = next(
        event for event in events if event.type is StreamEventType.ACCOUNT_LIMIT
    )
    assert suggestion.data["account"] == "anthropic:first"
    assert render_event(suggestion).plain.startswith(
        "account limit · anthropic:first · resets "
    )
    # the hold record makes status and recovery treat the account as out of usage
    state = read_usage(held.context.store, held.first, now=datetime.now(UTC))
    assert state.known_exhausted(datetime.now(UTC))
    assert held.waits == [5]
    assert [request.headers["authorization"] for request in held.requests] == [
        "Bearer token-first",
        "Bearer token-second",
    ]
    assert any(event.delta == "hello" for event in events)


@pytest.mark.asyncio
async def test_short_limit_holds_exactly_retry_after_and_never_switches_by_itself(
    tmp_path: Path,
) -> None:
    status, _headers, body = RATE_LIMIT_429
    held = await asyncio.to_thread(
        _Held, tmp_path, (status, {"retry-after": "120"}, body)
    )
    held.exhausted_once = True
    before = time.time()
    events = await held.run(user_switches=False)
    after = time.time()

    hold = next(event for event in events if event.type is StreamEventType.RETRY)
    assert hold.data["limit"] == "subscription"
    # the hold ends at now + Retry-After, with no longer default hold
    assert before + 120 <= hold.data["until"] <= after + 120
    assert 119 <= hold.data["delay"] <= 120
    assert sum(held.waits) == pytest.approx(hold.data["delay"])
    state = read_usage(held.context.store, held.first, now=datetime.now(UTC))
    assert state.snapshot is not None
    assert [w.window_minutes for w in state.snapshot.windows] == [300]
    # another account exists, yet only the user may switch to it
    assert [request.headers["authorization"] for request in held.requests] == [
        "Bearer token-first",
        "Bearer token-first",
    ]
    assert held.loop.backend is held.backends[held.first]
    assert any(event.delta == "hello" for event in events)
