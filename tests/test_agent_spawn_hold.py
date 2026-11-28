"""F053: `agent` refuses to start a child on an account with a long cached hold."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.support.credentials import named_account_credentials
from tests.support.fake_backend import PerActorFakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.test_accounts_exhaustion import _account_context, _codex_token
from tests.test_rate_limit_hold import _collect
from zeta.accounts.store import AccountId
from zeta.accounts.usage import UsageSnapshot, UsageWindow, record_usage_snapshot
from zeta.oauth import OAuthTokens
from zeta.protocol.types import MessageOrigin, StreamEventType, TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


class SpawnBackend(PerActorFakeBackend):
    provider = "codex"

    def __init__(self, arguments: dict[str, object]) -> None:
        call = ToolCall(
            "spawn",
            "agent",
            {
                "prompt": "inspect",
                "description": "worker",
                "background": True,
                **arguments,
            },
        )
        super().__init__(
            [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("noted")])],
            {"inspect": [ScriptedTurn([TextContent("child done")])]},
            parent_prompt="start",
            gated_child_prompts=set(),
            parent_waits_for_children=False,
        )


def _seed(
    context,
    account: AccountId,
    *,
    used: float,
    resets_in: timedelta,
    also_resets_in: timedelta | None = None,
    age: timedelta = timedelta(0),
) -> None:
    now = datetime.now(UTC)
    windows = [UsageWindow("primary", used, 10_080, now + resets_in)]
    if also_resets_in is not None:
        windows.append(UsageWindow("five_hour", used, 300, now + also_resets_in))
    record_usage_snapshot(
        context.store,
        account,
        UsageSnapshot(now - age, "usage_endpoint", tuple(windows)),
        now,
    )


async def _spawn(tmp_path: Path, arguments: dict[str, object], seed):
    """Run one real parent turn that calls `agent`; return (result, loop, opened)."""

    context, opened, first, second = await asyncio.to_thread(_account_context, tmp_path)
    await asyncio.to_thread(seed, context, first, second)
    loop = AgentLoop(
        SpawnBackend(arguments),
        opened.store,
        max_turns=2,
        skill_catalog=SkillCatalog.empty(),
    )
    loop.manager = context.manager
    loop.session_metadata = context.metadata
    loop.account_context = context
    loop.tool_registry.account_context = context
    events = await await_or_fail(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), what="parent turn"
    )
    [result] = [
        event.tool_result
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_END
        and event.tool_call is not None
        and event.tool_call.name == "agent"
    ]
    return result, loop, opened


async def _children(loop: AgentLoop) -> list[object]:
    status = await loop.tool_registry.execute(ToolCall("status", "agent_status", {}))
    return list(status["structuredContent"].get("children", []))


async def _close(loop: AgentLoop, opened) -> None:
    try:
        await asyncio.wait_for(loop._background_owner.wait(), 10)
    finally:
        await loop.close()
        opened.store.close()


@pytest.mark.asyncio
async def test_spawn_on_codex_model_refuses_a_four_day_cached_hold(
    tmp_path: Path,
) -> None:
    hold = timedelta(days=4)

    def seed(context, first, second) -> None:
        # the session pins codex:first; it is held for days (its other spent
        # window resets sooner), codex:second has room, codex:third only has a
        # stale low reading, so it is not confirmed headroom
        _seed(
            context,
            first,
            used=100.0,
            resets_in=hold,
            also_resets_in=timedelta(hours=2),
        )
        _seed(context, second, used=10.0, resets_in=timedelta(hours=1))
        third = AccountId("codex", "third")
        named_account_credentials(context.store.home, "codex", "third").save(
            OAuthTokens(_codex_token("third"), "refresh", 4_000_000_000)
        )
        _seed(
            context,
            third,
            used=10.0,
            resets_in=timedelta(hours=1),
            age=timedelta(hours=1),
        )

    before = datetime.now(UTC)
    result, loop, opened = await _spawn(tmp_path, {"model": "gpt-5.6-luna"}, seed)
    try:
        text = result.content
        assert result.is_error, text
        assert "codex:first" in text, text
        end = datetime.fromisoformat(text.split(" until ", 1)[1].split(" ", 1)[0])
        assert (
            before + hold - timedelta(minutes=1) <= end <= datetime.now(UTC) + hold
        ), f"the hold must end at the last spent window's reset: {text}"
        assert "codex:second" in text, "the refusal must name the account with headroom"
        assert "codex:third" not in text, (
            "a stale account must not be named as headroom"
        )
        assert await _children(loop) == [], (
            "a refused spawn must not create a child row"
        )
        assert not (opened.store.session_dir / "agents").exists()
    finally:
        await _close(loop, opened)


def _no_hold(context, first, second) -> None:
    del context, first, second


def _short_hold_on_pin(context, first, second) -> None:
    del second
    _seed(context, first, used=100.0, resets_in=timedelta(minutes=5))


def _long_hold_on_unpinned_default(context, first, second) -> None:
    # the session keeps its codex:first pin; only the new default is held
    context.store.set_default(second)
    _seed(context, second, used=100.0, resets_in=timedelta(days=4))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed", [_no_hold, _short_hold_on_pin, _long_hold_on_unpinned_default]
)
async def test_spawn_without_a_long_hold_on_the_child_account_starts(
    tmp_path: Path, seed
) -> None:
    result, loop, opened = await _spawn(tmp_path, {}, seed)
    try:
        assert not result.is_error, result.content
        assert len(await _children(loop)) == 1
    finally:
        await _close(loop, opened)
