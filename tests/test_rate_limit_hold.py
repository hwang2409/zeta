"""A long provider Retry-After holds the turn; the hold is visible and ends on an account switch."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.support.credentials import named_account_credentials
from tests.support.fake_backend import (
    PerActorFakeBackend,
    ScriptedTurn,
    _public_input_prompt,
)
from tests.support.owned import await_or_fail
from tests.test_accounts import _codex_hello_sse
from tests.test_accounts_exhaustion import _account_context, _codex_token
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
)
from zeta.providers.codex import CodexBackend
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.server.control import ServeControl
from zeta.skills import SkillCatalog

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")

HOLD_TEXT = re.compile(
    r"rate-limited until .+ \(provider asks 6d 1h 45m\); checking again in 5m$"
)


class LongRateLimitError(RuntimeError):
    status_code = 429
    retry_after = 524749.0
    retryable = True


@pytest.mark.asyncio
async def test_account_switch_ends_hold_and_probes_with_new_account(
    tmp_path: Path,
) -> None:
    context, opened, first, second = await asyncio.to_thread(_account_context, tmp_path)
    requests: list[httpx.Request] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "524749"},
                json={"error": {"message": "busy"}},
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_codex_hello_sse(),
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backends = {
        account: CodexBackend(
            client=client,
            token_store=named_account_credentials(context.home, "codex", account.name),
            base_url="https://test.invalid/codex/responses",
        )
        for account in (first, second)
    }
    loop = AgentLoop(backends[first], opened.store, skill_catalog=SkillCatalog.empty())
    loop.manager = context.manager
    loop.session_metadata = context.metadata
    loop.account_context = context

    async def switch_during_hold(delay: float) -> None:
        waits.append(delay)
        if len(waits) == 1:
            await context.switch_active(
                loop,
                second,
                boundary=opened.store.active_branch_head_sequence(),
                provider="codex",
                build_backend=lambda account: backends[account],
            )

    for backend in backends.values():
        backend._completion_wait = switch_during_hold
    try:
        events = await await_or_fail(
            _collect(loop.run_turn("hello", origin=MessageOrigin.USER)),
            what="held turn",
        )
    finally:
        await loop.close()
        await client.aclose()
        opened.store.close()

    assert not any(event.type is StreamEventType.ERROR for event in events)
    retries = [event for event in events if event.type is StreamEventType.RETRY]
    assert [event.data.get("kind") for event in retries] == ["provider_rate_limit_hold"]
    assert waits == [5], "the switch must end the hold at once, not after 300 s"
    assert [request.headers["authorization"] for request in requests] == [
        f"Bearer {_codex_token('first')}",
        f"Bearer {_codex_token('second')}",
    ]
    assert loop.backend is backends[second]
    assert any(event.delta == "hello" for event in events)


class RateLimitedChildBackend(PerActorFakeBackend):
    """The child's first request gets a six-day Retry-After; its hold waits until released."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self._completion_wait = self._wait

    async def _wait(self, _delay: float) -> None:
        self.held.set()
        await asyncio.wait_for(self.release.wait(), 10)

    @owned_completion
    async def complete(self, messages, tool_schemas):
        prompt = _public_input_prompt(messages, {*self.parent_prompts, *self.children})
        if prompt == "inspect" and self.child_calls["inspect"] == 0:
            self.child_calls["inspect"] += 1
            raise LongRateLimitError("busy")
        async for event in PerActorFakeBackend.complete.__wrapped__(
            self, messages, tool_schemas
        ):
            yield event


@pytest.mark.asyncio
async def test_held_child_reports_rate_limited_until_and_stays_running(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    call = ToolCall(
        "background-agent",
        "agent",
        {"prompt": "inspect", "description": "background", "background": True},
    )
    backend = RateLimitedChildBackend(
        [ScriptedTurn(tool_calls=[call])],
        {"inspect": [ScriptedTurn([TextContent("child done")])]},
        parent_prompt="start",
        gated_child_prompts=set(),
        parent_waits_for_children=False,
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)
    held = asyncio.ensure_future(asyncio.wait_for(backend.held.wait(), 10))
    child_ended: asyncio.Future[object] | None = None
    try:
        await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
        child_ended = asyncio.ensure_future(
            asyncio.wait_for(loop._background_owner.wait(), 10)
        )
        await asyncio.wait(
            {held, child_ended}, timeout=10, return_when=asyncio.FIRST_COMPLETED
        )
        assert held.done() and not child_ended.done(), "the child must hold, not fail"

        status = await loop.tool_registry.execute(
            ToolCall("status-call", "agent_status", {})
        )
        child = status["structuredContent"]["children"][0]
        assert child["state"] == "running", "a held child is not idle or stalled"
        assert child.get("idle_since") is None
        assert HOLD_TEXT.search(child["current_step"]), child["current_step"]
        assert child["current_step"] in status["content"][0]["text"]

        progress = [
            event
            for event in background_events
            if event.type is StreamEventType.TOOL_EXECUTION_UPDATE
            and "rate-limited until" in (event.delta or "")
        ]
        assert len(progress) == 1
        serve_frames = await _serve_mapping(progress[0])
        assert [name for name, _fields in serve_frames] == ["tool_output"]
        assert serve_frames[0][1]["output"] == progress[0].delta
        assert HOLD_TEXT.search(progress[0].delta.rstrip("\n"))

        backend.release.set()
        await await_or_fail(child_ended, what="loop._background_owner.wait()")
        done = await loop.tool_registry.execute(
            ToolCall("status-call-2", "agent_status", {"handle": child["handle"]})
        )
        finished = done["structuredContent"]["children"][0]
        assert finished["state"] == "completed"
        assert finished["final_result"].startswith("child done")
    finally:
        held.cancel()
        backend.release.set()
        if child_ended is not None:
            await await_or_fail(child_ended, what="child end")
        await loop.close()
        store.close()


async def _serve_mapping(event: StreamEvent) -> list[tuple[str, dict[str, object]]]:
    control = object.__new__(ServeControl)
    control.server = SimpleNamespace(runtime=SimpleNamespace(state=None))
    frames: list[tuple[str, dict[str, object]]] = []

    async def notify(name: object, session_id: str | None, **fields: object) -> None:
        del session_id
        frames.append((getattr(name, "value", name), fields))

    control._notify = notify

    control._end_stale_approvals = lambda _session_id: None
    await control._event(event, session_id="session", background=True)
    return frames


async def _collect(events):
    return [event async for event in events]
