"""One ``zeta.context.compact`` span per compaction through every real compaction path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.owned import await_or_fail
from tests.telemetry import test_agent_spans
from tests.telemetry.test_agent_spans import (
    PROMPT,
    Escape,
    ScriptedBackend,
    _assert_no_content,
    _by_name,
)
from zeta.core.commands.slash_registry import SlashSurface
from zeta.core.context import ContextAssembler
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ErrorInfo,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.server.slash_commands import ServerSlashSession
from zeta.skills import SkillCatalog
from zeta.telemetry import Telemetry
from zeta.tui.app import TUIApp

pytest.importorskip("opentelemetry.sdk")

MODEL = "claude-opus-5-5"
CHAT = f"chat {MODEL}"
COMPACT = "zeta.context.compact"
OVERFLOW = ErrorInfo("context_length_exceeded", PROMPT)
# the shared in-memory exporter fixture
traced = test_agent_spans.traced


def _count(message: Message) -> int:
    if message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION} or message.metadata.get("compaction_summary") or message.metadata.get("context_evicted"):
        return 1
    return 30


def _loop(
    tmp_path: Path, backend: ScriptedBackend, telemetry: Telemetry, *, history: int, budget: int, role: MessageRole = MessageRole.USER
) -> AgentLoop:
    """``history`` old messages of ``role``: user ones need a summary, assistant ones are evicted without one."""
    store = ConversationStore(tmp_path / "sessions")
    for index in range(history):
        message = Message(role, [TextContent(f"secret {index}")])
        store.append_message(with_message_origin(message, MessageOrigin.USER) if role is MessageRole.USER else message)
    assembler = ContextAssembler(store, backend=backend, token_budget=budget, retained_tail=1, token_counter=_count)
    return AgentLoop(backend, store, context_assembler=assembler, skill_catalog=SkillCatalog.empty(), telemetry=telemetry)


async def _turn(loop: AgentLoop) -> None:
    async for _event in loop.run_turn(PROMPT, origin=MessageOrigin.USER):
        pass


async def _closing(loop: AgentLoop, work) -> object:
    try:
        return await work
    finally:
        await loop.close()
        loop.store.close()


def _compactions(memory) -> list:
    return _by_name(memory.get_finished_spans()).get(COMPACT, [])


def _children(spans, parent) -> list:
    return [span for span in spans if span.parent is not None and span.parent.span_id == parent.context.span_id]


def test_automatic_compaction_is_one_span_under_the_turn_with_the_summaries_as_children(tmp_path, traced) -> None:
    telemetry, memory = traced
    loop = _loop(tmp_path, ScriptedBackend("claude", MODEL, ["reply"] * 4), telemetry, history=2, budget=40)
    asyncio.run(_closing(loop, _turn(loop)))
    spans = memory.get_finished_spans()
    names = _by_name(spans)
    (turn,), (compact,) = names["zeta.turn"], names[COMPACT]
    assert sorted(span.name for span in _children(spans, turn)) == [CHAT, COMPACT]
    # two chunk summaries and their reduce, each one inference span
    summaries = _children(spans, compact)
    assert [span.name for span in summaries] == [CHAT] * 3
    assert len(names[CHAT]) == 4
    assert dict(compact.attributes) == {"zeta.emergency": False, "zeta.tokens.before": 90, "zeta.tokens.after": 32, "zeta.marker.count": 1}
    for summary in summaries:
        assert summary.attributes["gen_ai.provider.name"] == "anthropic"
        assert summary.attributes["gen_ai.usage.total_tokens"] == 20
        assert summary.attributes["gen_ai.response.finish_reasons"] == "end_turn"
    assert all(span.attributes.get("error.type") is None for span in spans)
    _assert_no_content(spans)


def test_context_that_fits_opens_no_compaction_span(tmp_path, traced) -> None:
    telemetry, memory = traced
    loop = _loop(tmp_path, ScriptedBackend("claude", MODEL, ["reply"]), telemetry, history=0, budget=1_000)
    asyncio.run(_closing(loop, _turn(loop)))
    assert sorted(span.name for span in memory.get_finished_spans()) == sorted(["invoke_agent zeta", "zeta.turn", CHAT])


def test_forced_retry_compaction_is_one_emergency_span_in_the_same_turn(tmp_path, traced) -> None:
    telemetry, memory = traced
    backend = ScriptedBackend("claude", MODEL, [OVERFLOW, "reply"])
    loop = _loop(tmp_path, backend, telemetry, history=4, budget=1_000, role=MessageRole.ASSISTANT)
    asyncio.run(_closing(loop, _turn(loop)))
    spans = memory.get_finished_spans()
    names = _by_name(spans)
    (turn,), (compact,) = names["zeta.turn"], names[COMPACT]
    assert sorted(span.name for span in _children(spans, turn)) == [CHAT, CHAT, COMPACT]
    # emergency eviction is deterministic: no summary call
    assert _children(spans, compact) == []
    assert dict(compact.attributes) == {"zeta.emergency": True, "zeta.tokens.before": 150, "zeta.tokens.after": 31, "zeta.marker.count": 1}
    assert compact.attributes.get("error.type") is None
    _assert_no_content(spans)


def _tui_compact(loop: AgentLoop) -> str:
    app = TUIApp(loop, provider="claude", model=MODEL)

    async def run() -> str:
        try:
            return await create_slash_registry(project_eligible=False, skill_catalog=SkillCatalog.empty()).dispatch_async(app, "/compact")
        finally:
            await app.close()

    return asyncio.run(run())


def _serve_compact(loop: AgentLoop) -> str:
    app = TUIApp(loop, provider="claude", model=MODEL)
    runtime = SimpleNamespace(loop=loop, state=SimpleNamespace(usage_tracker=app._usage_tracker, model=MODEL))

    async def run() -> str:
        try:
            return await create_slash_registry(
                project_eligible=False, skill_catalog=SkillCatalog.empty(), surface=SlashSurface.SERVE
            ).dispatch_async(ServerSlashSession(runtime), "/compact")
        finally:
            await app.close()

    return asyncio.run(run())


@pytest.mark.parametrize("surface", [_tui_compact, _serve_compact], ids=["tui", "serve"])
def test_manual_compact_is_one_root_span_with_the_summary_as_child(tmp_path, traced, surface) -> None:
    telemetry, memory = traced
    loop = _loop(tmp_path, ScriptedBackend("claude", MODEL, ["reply"]), telemetry, history=3, budget=1_000)
    assert surface(loop) == "compacted entries 1\u20132; tokens after: 32"
    spans = memory.get_finished_spans()
    assert sorted(span.name for span in spans) == [CHAT, COMPACT]
    (compact,), (summary,) = _by_name(spans)[COMPACT], _by_name(spans)[CHAT]
    assert compact.parent is None
    assert summary.parent.span_id == compact.context.span_id
    assert dict(compact.attributes) == {"zeta.emergency": False, "zeta.tokens.before": 90, "zeta.tokens.after": 32, "zeta.marker.count": 1}
    _assert_no_content(spans)


# rows: (scripted summary attempts, what the turn raises, error.type of each summary span, of compaction, of turn)
@pytest.mark.parametrize(
    ("attempts", "raised", "summary_errors", "compact_error", "turn_error"),
    [
        ([ErrorInfo("invalid_request_error", PROMPT)] * 2, None, ["_OTHER", "_OTHER"], "SummaryCompletionError", None),
        (["stall", "reply"], asyncio.CancelledError, ["CancelledError", "CancelledError"], "CancelledError", "CancelledError"),
        ([Escape(), "reply"], Escape, ["CancelledError", "Escape"], "Escape", "Escape"),
    ],
    ids=["error-event", "cancelled", "base-exception"],
)
def test_failed_compaction_ends_its_span_once_with_the_summaries_under_it(
    tmp_path, traced, attempts, raised, summary_errors, compact_error, turn_error
) -> None:
    telemetry, memory = traced
    backend = ScriptedBackend("claude", MODEL, [*attempts, "reply", "reply"])
    loop = _loop(tmp_path, backend, telemetry, history=2, budget=40)

    async def scenario() -> None:
        task = asyncio.create_task(_closing(loop, _turn(loop)))
        if attempts[0] == "stall":
            await await_or_fail(backend.stalled.wait(), what="the summary call to start")
            task.cancel()
        await await_or_fail(task, what="the turn to finish")

    if raised is None:
        asyncio.run(scenario())
    else:
        with pytest.raises(raised):
            asyncio.run(scenario())
    spans = memory.get_finished_spans()
    assert sorted(span.name for span in spans) == sorted(["invoke_agent zeta", "zeta.turn", COMPACT, CHAT, CHAT])
    (turn,), (compact,) = _by_name(spans)["zeta.turn"], _by_name(spans)[COMPACT]
    assert compact.parent.span_id == turn.context.span_id
    summaries = _children(spans, compact)
    assert sorted(span.attributes.get("error.type") for span in summaries) == summary_errors
    assert [compact.attributes.get("error.type"), turn.attributes.get("error.type")] == [compact_error, turn_error]
    assert "zeta.tokens.after" not in compact.attributes
    _assert_no_content(spans)


def test_disabled_compaction_creates_no_span(tmp_path, monkeypatch) -> None:
    def no_span(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disabled telemetry reached span construction")

    # test-contract: seam the disabled facade must never reach span construction
    monkeypatch.setattr(Telemetry, "_span", no_span)
    loop = _loop(tmp_path, ScriptedBackend("claude", MODEL, ["reply"] * 4), Telemetry.disabled(), history=2, budget=40)
    asyncio.run(_closing(loop, _turn(loop)))
    assert loop.store.compaction_marker_count() == 1
