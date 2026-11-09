"""A background child that enters a long provider hold wakes its parent once."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.fake_backend import (
    PerActorFakeBackend,
    ScriptedTurn,
    _public_input_prompt,
)
from tests.support.owned import await_or_fail
from tests.test_accounts_exhaustion import _account_context
from tests.test_rate_limit_hold import _collect
from zeta.accounts.exhaustion import classify_subscription_limit
from zeta.accounts.store import AccountId
from zeta.protocol.types import (
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.server.control import NotificationPresentation, ServeControl
from zeta.skills import SkillCatalog
from zeta.tui.render import render_event

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")

LONG_RETRY_AFTER = 524749.0


def _rate_limited(retry_after: float) -> BaseException:
    error = RuntimeError("busy")
    error.status_code = 429
    error.retry_after = retry_after
    error.retryable = True
    return error


def _usage_limit_reached(resets_at: int) -> BaseException:
    error = RuntimeError("The usage limit has been reached")
    error.status_code = 429
    error.response_body = json.dumps(
        {
            "error": {
                "type": "usage_limit_reached",
                "resets_at": resets_at,
                "limit_window_minutes": 10080,
            }
        }
    ).encode()
    # the same classification the codex adapter applies to this 429 body
    error.subscription_limit = classify_subscription_limit(
        "codex", AccountId("codex", "first"), error
    )
    assert error.subscription_limit is not None
    return error


class HeldChildBackend(PerActorFakeBackend):
    """The child's first requests fail; the first hold wait blocks until released."""

    provider = "codex"

    def __init__(self, failures: list[BaseException], *, parent_turns: int) -> None:
        call = ToolCall(
            "background-agent",
            "agent",
            {"prompt": "inspect", "description": "worker", "background": True},
        )
        super().__init__(
            [ScriptedTurn(tool_calls=[call])]
            + [ScriptedTurn([TextContent("noted")])] * (parent_turns - 1),
            {"inspect": [ScriptedTurn([TextContent("child done")])]},
            parent_prompt="start",
            gated_child_prompts=set(),
            parent_waits_for_children=False,
        )
        self.failures = failures
        self.waits: list[float] = []
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self._completion_wait = self._wait

    async def _wait(self, delay: float) -> None:
        # a fake clock: hold waits return at once, except the first, which the test owns
        self.waits.append(delay)
        if len(self.waits) == 1:
            self.held.set()
            await asyncio.wait_for(self.release.wait(), 10)

    @owned_completion
    async def complete(self, messages, tool_schemas):
        prompt = _public_input_prompt(messages, {*self.parent_prompts, *self.children})
        if prompt == "inspect" and self.failures:
            raise self.failures.pop(0)
        async for event in PerActorFakeBackend.complete.__wrapped__(
            self, messages, tool_schemas
        ):
            yield event


class HeldTree:
    """A root loop with named codex accounts and one background child in a hold."""

    def __init__(self, tmp_path: Path, backend: HeldChildBackend) -> None:
        self.context, self.opened, _first, _second = _account_context(tmp_path)
        self.backend = backend
        self.loop = AgentLoop(
            backend, self.opened.store, max_turns=1, skill_catalog=SkillCatalog.empty()
        )
        self.loop.manager = self.context.manager
        self.loop.session_metadata = self.context.metadata
        self.loop.account_context = self.context
        self.wakes = 0
        self.loop._background_owner.set_wake_callback(self._woke)
        self.child_ended: asyncio.Future[object] | None = None

    def _woke(self) -> None:
        self.wakes += 1

    async def start_and_hold(self) -> None:
        await _collect(self.loop.run_turn("start", origin=MessageOrigin.USER))
        self.child_ended = asyncio.ensure_future(
            asyncio.wait_for(self.loop._background_owner.wait(), 10)
        )
        await await_or_fail(self.backend.held.wait(), what="child hold")
        assert not self.child_ended.done(), "the child must hold, not fail"

    def holds(self, *, pending_only: bool = True) -> list[dict[str, object]]:
        return [
            entry.data
            for entry in self.opened.store.agent_notifications(pending_only=pending_only)
            if entry.data.get("kind") == "child_hold"
        ]

    async def child_state(self) -> dict[str, object]:
        status = await self.loop.tool_registry.execute(
            ToolCall("status-call", "agent_status", {})
        )
        return status["structuredContent"]["children"][0]

    async def close(self) -> None:
        # closing the root cancels a still-held child without running out its hold
        await self.loop.close()
        if self.child_ended is not None:
            await asyncio.gather(self.child_ended, return_exceptions=True)
        self.opened.store.close()


async def _serve_frames(event: StreamEvent) -> list[tuple[str, dict[str, object]]]:
    control = object.__new__(ServeControl)
    control.server = SimpleNamespace(
        runtime=SimpleNamespace(state=None),
        notification_presentations=NotificationPresentation(),
    )
    control._protocol_version = lambda: "1.2"
    frames: list[tuple[str, dict[str, object]]] = []

    async def notify(name: object, session_id: str | None, **fields: object) -> bool:
        del session_id
        frames.append((getattr(name, "value", name), fields))
        return True

    control._notify = notify
    control._end_stale_approvals = lambda _session_id: None
    await control._event(event, session_id="session", background=True)
    return frames


@pytest.mark.asyncio
async def test_long_rate_limit_hold_wakes_parent_once_across_reprobes(
    tmp_path: Path,
) -> None:
    # two 429s: the second probe re-enters the same hold
    backend = HeldChildBackend(
        [_rate_limited(LONG_RETRY_AFTER), _rate_limited(LONG_RETRY_AFTER - 300)],
        parent_turns=2,
    )
    tree = await asyncio.to_thread(HeldTree, tmp_path, backend)
    before = time.time()
    try:
        await tree.start_and_hold()
        holds = tree.holds()
        assert len(holds) == 1, "a long hold must wake the parent once"
        hold = holds[0]
        assert tree.wakes >= 1, "the hold notice must wake the parent"
        assert hold["reason"] == "rate_limit"
        assert hold["provider"] == "codex"
        assert hold["account"] == "codex:first"
        assert hold["retry_after"] == LONG_RETRY_AFTER
        until = datetime.fromisoformat(str(hold["until"])).timestamp()
        assert before + LONG_RETRY_AFTER <= until <= time.time() + LONG_RETRY_AFTER
        assert hold.get("display_only") is None

        child = await tree.child_state()
        assert child["state"] == "running", "a held child is not idle or stalled"
        assert child.get("idle_since") is None
        assert not [
            entry
            for entry in tree.opened.store.agent_notifications(pending_only=False)
            if entry.data.get("kind") == "child_idle"
        ]

        # the parent's next step receives the wake as a delivered notification
        events = await await_or_fail(
            _collect(tree.loop.run_notification_turn()), what="parent wake turn"
        )
        delivered = [
            event
            for event in events
            if event.type is StreamEventType.AGENT_NOTIFICATION
            and event.data.get("kind") == "child_hold"
        ]
        assert len(delivered) == 1
        rendered = _plain(render_event(delivered[0]))
        assert "hold ←" in rendered and "worker" in rendered
        assert f"rate limit until {hold['until']}" in rendered
        frames = await _serve_frames(delivered[0])
        assert [name for name, _fields in frames] == ["child_hold"]
        served = frames[0][1]["data"]
        assert {key: served[key] for key in ("reason", "until", "account")} == {
            key: hold[key] for key in ("reason", "until", "account")
        }

        backend.release.set()
        assert tree.child_ended is not None
        await await_or_fail(tree.child_ended, what="held child end")
    finally:
        await tree.close()

    assert len(backend.waits) > 1, "the second probe must re-enter the hold"
    assert len(tree.holds(pending_only=False)) == 1, "a re-probe is the same hold"


@pytest.mark.asyncio
async def test_short_rate_limit_hold_does_not_wake_parent(tmp_path: Path) -> None:
    backend = HeldChildBackend([_rate_limited(120.0)], parent_turns=1)
    tree = await asyncio.to_thread(HeldTree, tmp_path, backend)
    try:
        await tree.start_and_hold()
        assert tree.holds() == []
        backend.release.set()
        assert tree.child_ended is not None
        await await_or_fail(tree.child_ended, what="held child end")
    finally:
        await tree.close()
    assert tree.holds(pending_only=False) == []


@pytest.mark.asyncio
async def test_subscription_limit_days_away_wakes_parent_once(tmp_path: Path) -> None:
    resets_at = int(time.time()) + 3 * 86400
    backend = HeldChildBackend([_usage_limit_reached(resets_at)], parent_turns=1)
    tree = await asyncio.to_thread(HeldTree, tmp_path, backend)
    try:
        await tree.start_and_hold()
        holds = tree.holds()
        assert len(holds) == 1, "a subscription hold days away must wake the parent"
        hold = holds[0]
        assert hold["reason"] == "subscription_limit"
        assert hold["account"] == "codex:first"
        assert hold["until"] == datetime.fromtimestamp(resets_at, UTC).isoformat()
        assert "subscription limit" in str(hold["text"])
        assert (await tree.child_state())["state"] == "running"
    finally:
        # the reset is days away; closing the tree cancels the held child
        await tree.close()


def _plain(renderable: object) -> str:
    from rich.console import Console

    console = Console(width=160, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()
