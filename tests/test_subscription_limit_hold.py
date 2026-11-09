"""A Codex subscription limit holds the turn until reset, a re-login, or a user account switch."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from tests.support.credentials import named_account_credentials
from tests.support.owned import await_or_fail
from tests.test_accounts import _codex_hello_sse
from tests.test_accounts_exhaustion import _account_context, _codex_token
from tests.test_rate_limit_hold import _collect, _serve_mapping
from zeta.accounts import AccountUsage
from zeta.accounts.usage import UsageSnapshot, UsageWindow
from zeta.oauth import OAuthTokens
from zeta.protocol.types import MessageOrigin, StreamEventType
from zeta.providers.codex import CodexBackend
from zeta.providers.rate_limit_hold import rate_limit_hold_text
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.render import render_event

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")

RESETS_AT = int(time.time()) + 3 * 86400
HOLD_TEXT = re.compile(
    r"^codex:first subscription limit until \w{3} \d\d \d\d:\d\d; "
    r"switch account or log in to resume$"
)


def _usage_limit_reached(request: httpx.Request) -> httpx.Response:
    # the verified Codex Responses shape that classify_subscription_limit accepts
    return httpx.Response(
        429,
        json={
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "resets_at": RESETS_AT,
                "limit_window_minutes": 10080,
            }
        },
        request=request,
    )


class _Held:
    """Two codex accounts, the first at its subscription limit until RESETS_AT."""

    def __init__(self, tmp_path: Path) -> None:
        self.context, self.opened, self.first, self.second = _account_context(tmp_path)
        now = datetime.now(UTC)
        # exactly one fresh alternative, so recovery suggests switching to it
        AccountUsage(self.context.store)._write(
            self.second,
            UsageSnapshot(
                now,
                "usage_endpoint",
                (UsageWindow("primary", 25, 10_080, now + timedelta(days=1)),),
            ),
            now,
            None,
            "succeeded",
            finished=now,
        )
        self.requests: list[httpx.Request] = []
        self.waits: list[float] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        self.backends = {
            account: CodexBackend(
                client=self.client,
                token_store=named_account_credentials(
                    self.context.home, "codex", account.name
                ),
                base_url="https://test.invalid/codex/responses",
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
        if request.headers["authorization"] == f"Bearer {_codex_token('first')}":
            return _usage_limit_reached(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_codex_hello_sse(),
            request=request,
        )

    def tokens(self) -> list[str]:
        return [request.headers["authorization"] for request in self.requests]

    def on_wait(self, wait) -> None:
        for backend in self.backends.values():
            backend._completion_wait = wait

    async def switch_to_second(self) -> None:
        await self.context.switch_active(
            self.loop,
            self.second,
            boundary=self.opened.store.active_branch_head_sequence(),
            provider="codex",
            build_backend=lambda account: self.backends[account],
        )

    async def close(self) -> None:
        await self.loop.close()
        await self.client.aclose()
        self.opened.store.close()


@pytest.mark.asyncio
async def test_subscription_limit_holds_without_auto_switch_until_user_switches(
    tmp_path: Path,
) -> None:
    held = await asyncio.to_thread(_Held, tmp_path)

    async def user_switches_after_three_polls(delay: float) -> None:
        held.waits.append(delay)
        # one alternative account exists, yet nothing may use it before the user acts
        assert held.tokens() == [f"Bearer {_codex_token('first')}"]
        if len(held.waits) == 3:
            await held.switch_to_second()

    held.on_wait(user_switches_after_three_polls)
    try:
        events = await await_or_fail(
            _collect(held.loop.run_turn("hello", origin=MessageOrigin.USER)),
            what="held turn",
        )
    finally:
        await held.close()

    errors = [event.error for event in events if event.type is StreamEventType.ERROR]
    assert errors == [], "a subscription limit must hold the turn, not end it"
    retries = [event for event in events if event.type is StreamEventType.RETRY]
    assert [event.data.get("kind") for event in retries] == ["provider_rate_limit_hold"]
    hold = retries[0].data
    assert hold["limit"] == "subscription"
    assert hold["account"] == "codex:first"
    assert hold["until"] == RESETS_AT
    assert HOLD_TEXT.match(hold["text"]), hold["text"]
    # the TUI, serve, and child progress (agent_status) show the same hold
    assert render_event(retries[0]).plain == hold["text"]
    frames = await _serve_mapping(retries[0])
    assert [name for name, _fields in frames] == ["retry"]
    served = frames[0][1].get("data", frames[0][1])
    assert {key: served[key] for key in ("kind", "limit", "account", "until", "text")} == {
        key: hold[key] for key in ("kind", "limit", "account", "until", "text")
    }
    assert hold["text"] == f"codex:first {rate_limit_hold_text(hold)}"
    assert held.waits == [5, 5, 5], "the hold polls credentials and never probes before reset"
    suggestion = next(event for event in events if event.type is StreamEventType.ACCOUNT_LIMIT)
    assert suggestion.data["action"] == "switch"
    assert suggestion.data["alternative"] == "codex:second"
    assert events.index(retries[0]) < events.index(suggestion)
    assert held.tokens() == [
        f"Bearer {_codex_token('first')}",
        f"Bearer {_codex_token('second')}",
    ]
    assert held.loop.backend is held.backends[held.second]
    assert any(event.delta == "hello" for event in events)


@pytest.mark.asyncio
async def test_relogin_wakes_subscription_hold_and_probes(tmp_path: Path) -> None:
    held = await asyncio.to_thread(_Held, tmp_path)
    relogin = _codex_token("relogin")

    async def relogin_during_hold(delay: float) -> None:
        held.waits.append(delay)
        if len(held.waits) == 1:
            await asyncio.to_thread(
                named_account_credentials(held.context.home, "codex", "first").save,
                OAuthTokens(relogin, "refresh", 4_000_000_000),
            )

    held.on_wait(relogin_during_hold)
    try:
        events = await await_or_fail(
            _collect(held.loop.run_turn("hello", origin=MessageOrigin.USER)),
            what="held turn",
        )
    finally:
        await held.close()

    assert not any(event.type is StreamEventType.ERROR for event in events)
    assert held.waits == [5], "a re-login must wake the hold at the next poll"
    assert held.tokens() == [f"Bearer {_codex_token('first')}", f"Bearer {relogin}"]
    assert held.loop.backend is held.backends[held.first]
    assert any(event.delta == "hello" for event in events)


@pytest.mark.asyncio
async def test_cancel_during_subscription_hold_ends_turn_at_once(tmp_path: Path) -> None:
    held = await asyncio.to_thread(_Held, tmp_path)
    holding = asyncio.Event()

    async def wait_forever(delay: float) -> None:
        held.waits.append(delay)
        holding.set()
        # only the test's cancel ends this wait
        await asyncio.wait_for(asyncio.Event().wait(), 3600)

    held.on_wait(wait_forever)
    turn = asyncio.ensure_future(
        _collect(held.loop.run_turn("hello", origin=MessageOrigin.USER))
    )
    try:
        await await_or_fail(holding.wait(), what="subscription hold")
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(turn, what="cancelled held turn")
    finally:
        turn.cancel()
        await held.close()

    assert held.waits == [5]
    assert held.tokens() == [f"Bearer {_codex_token('first')}"]
