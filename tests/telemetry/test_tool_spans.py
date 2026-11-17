"""Tool, approval, and child-agent spans through the real dispatch, registry, and agent runner."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

import pytest

from tests.support.agent_lifecycle import wait_background_owner
from tests.support.owned import await_or_fail
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.telemetry import Telemetry

pytest.importorskip("opentelemetry.sdk")

MODEL = "claude-opus-5-5"
PROMPT = "start the secret plan"
CHILD_PROMPT = "inspect the secret child task"
SECRET = "secret-tool-text"


class ToolFailure(RuntimeError):
    """A tool failure whose message is content."""


class RoutedBackend:
    """Shared by parent and child: the first user text names the conversation.

    The parent's first call returns ``calls``; every later call, and every child
    call, replies with text. A child waits for ``release_child`` first.
    """

    provider = "claude"
    model = MODEL

    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.calls = list(calls)
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()
        self.release_child.set()

    async def _completion_wait(self, delay: float) -> None:
        del delay

    @owned_completion
    async def complete(self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        first_user = next(block.text for message in messages if message.role is MessageRole.USER for block in message.content if isinstance(block, TextContent))
        answered = any(message.tool_result is not None for message in messages)
        if first_user == CHILD_PROMPT:
            self.child_started.set()
            await await_or_fail(self.release_child.wait(), what="the test to release the child")
        blocks = [ToolUseContent(call) for call in self.calls] if first_user == PROMPT and not answered else [TextContent(SECRET)]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, blocks), data={"stop_reason": "end_turn"})


@pytest.fixture
def traced() -> Iterator[tuple[Telemetry, object, object]]:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    class DropExporter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def export(self, spans: object) -> SpanExportResult:
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    class Lifecycle(SpanProcessor):
        """Counts span starts and ends by span id."""

        def __init__(self) -> None:
            self.started: dict[int, int] = {}
            self.ended: dict[int, int] = {}

        def on_start(self, span: object, parent_context: object = None) -> None:
            self.started[span.context.span_id] = self.started.get(span.context.span_id, 0) + 1

        def on_end(self, span: object) -> None:
            self.ended[span.context.span_id] = self.ended.get(span.context.span_id, 0) + 1

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trace_exporter, "OTLPSpanExporter", DropExporter)
        telemetry = Telemetry.configure("http://127.0.0.1:4318/v1/traces")
    memory, lifecycle = InMemorySpanExporter(), Lifecycle()
    telemetry._implementation._provider.add_span_processor(SimpleSpanProcessor(memory))
    telemetry._implementation._provider.add_span_processor(lifecycle)
    try:
        yield telemetry, memory, lifecycle
    finally:
        telemetry.shutdown()


A = "invoke_agent zeta"
T = "zeta.turn"
C = f"chat {MODEL}"
CHILD = "invoke_agent general"


def _path(*names: str) -> str:
    return "/".join(names)


def _agent(call_id: str, *, background: bool) -> ToolCall:
    return ToolCall(call_id, "agent", {"prompt": CHILD_PROMPT, "description": "child", "background": background})


_PARENT = [(_path(A), None), (_path(A, T), None), (_path(A, T, C), None), (_path(A, T), None), (_path(A, T, C), None)]


def _child(tool: str) -> list[tuple[str, str | None]]:
    return [(_path(A, T, tool, CHILD), None), (_path(A, T, tool, CHILD, T), None), (_path(A, T, tool, CHILD, T, C), None)]


# rows: (tool calls the model makes, ask for approval, cancel the turn at the tool await,
#        every span as (name path from the root, error.type))
CASES = [
    pytest.param([ToolCall("toolu_ok", "probe", {"text": SECRET})], False, False,
                 [*_PARENT, (_path(A, T, "execute_tool probe"), None)], id="success"),
    pytest.param([ToolCall("toolu_err", "broken", {"text": SECRET})], False, False,
                 [*_PARENT, (_path(A, T, "execute_tool broken"), "_OTHER")], id="tool-error"),
    pytest.param([ToolCall("toolu_stall", "stall", {"text": SECRET})], False, True,
                 [(_path(A), "CancelledError"), (_path(A, T), "CancelledError"), (_path(A, T, C), None),
                  (_path(A, T, "execute_tool stall"), "CancelledError")], id="cancelled-at-await"),
    pytest.param([ToolCall("toolu_left", "pair", {"text": SECRET}), ToolCall("toolu_right", "pair", {"text": SECRET})], False, False,
                 [*_PARENT, (_path(A, T, "execute_tool pair"), None), (_path(A, T, "execute_tool pair"), None)], id="parallel-siblings"),
    pytest.param([ToolCall("toolu_ask", "probe", {"text": SECRET})], True, False,
                 [*_PARENT, (_path(A, T, "execute_tool probe"), None)], id="approval-pair"),
    pytest.param([_agent("toolu_fg", background=False)], False, False,
                 [*_PARENT, (_path(A, T, "execute_tool agent"), None), *_child("execute_tool agent")], id="foreground-child"),
    pytest.param([_agent("toolu_bg", background=True)], False, False,
                 [*_PARENT, (_path(A, T, "execute_tool agent"), None), *_child("execute_tool agent")], id="background-child"),
]


def _loop(tmp_path: Path, backend: RoutedBackend, telemetry: Telemetry, *, ask: bool) -> tuple[AgentLoop, asyncio.Event, asyncio.Event]:
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(default="ask", store=store) if ask else None
    loop = AgentLoop(backend, store, approval_policy=policy, max_turns=4, skill_catalog=SkillCatalog.empty(), telemetry=telemetry)
    stalled, both = asyncio.Event(), asyncio.Event()
    pair_started: list[str] = []

    async def probe(arguments: dict[str, object]) -> str:
        return "ok"

    async def broken(arguments: dict[str, object]) -> str:
        raise ToolFailure(SECRET)

    async def stall(arguments: dict[str, object]) -> str:
        stalled.set()
        await await_or_fail(asyncio.Event().wait(), what="the test to cancel the stalled tool")
        return "unreachable"

    async def pair(arguments: dict[str, object]) -> str:
        # both calls must be in flight at once, so each runs in its own task
        pair_started.append("started")
        if len(pair_started) == 2:
            both.set()
        await await_or_fail(both.wait(), what="the sibling tool call to start")
        return "ok"

    registry = loop.tool_registry
    registry.register("probe", probe, requires_approval=ask)
    registry.register("broken", broken, requires_approval=False)
    registry.register("stall", stall, requires_approval=False)
    registry.register("pair", pair, parallel_safe=True, requires_approval=False)
    return loop, stalled, both


def _tree(spans: Sequence) -> list[tuple[str, str | None]]:
    by_id = {span.context.span_id: span for span in spans}

    def path(span: object) -> str:
        names = [span.name]
        while span.parent is not None:
            span = by_id[span.parent.span_id]  # KeyError: a parent that never finished
            names.append(span.name)
        return "/".join(reversed(names))

    return sorted((path(span), span.attributes.get("error.type")) for span in spans)


def _assert_no_content(spans: Sequence) -> None:
    for span in spans:
        texts = [span.name, span.status.description or ""]
        texts += [f"{key}={value}" for key, value in span.attributes.items()]
        for event in span.events:
            texts += [event.name, *(f"{key}={value}" for key, value in event.attributes.items())]
        for text in texts:
            assert "secret" not in text and "ok" != text.rsplit("=", 1)[-1], text
            assert not text.startswith(("gen_ai.tool.call.arguments", "gen_ai.tool.call.result")), text


@pytest.mark.parametrize(("calls", "ask", "cancel", "expected"), CASES)
def test_each_tool_and_child_span_ends_once_under_its_parent(tmp_path, traced, caplog, calls, ask, cancel, expected) -> None:
    from opentelemetry.trace import StatusCode

    telemetry, memory, lifecycle = traced
    backend = RoutedBackend(calls)
    background = any(call.arguments.get("background") for call in calls)
    if background:
        backend.release_child.clear()
    loop, stalled, _both = _loop(tmp_path, backend, telemetry, ask=ask)
    finished_at_turn_end: list[str] = []

    async def turn() -> None:
        async for event in loop.run_turn(PROMPT, origin=MessageOrigin.USER):
            if event.type is StreamEventType.TOOL_APPROVAL_START:
                assert loop.tool_registry.approval_policy.approve(event.tool_call.id)

    async def scenario() -> None:
        try:
            task = asyncio.create_task(turn())
            if cancel:
                await await_or_fail(stalled.wait(), what="the tool to start")
                task.cancel()
            await await_or_fail(task, what="the turn to finish")
        finally:
            if background:
                # the parent turn and its tool span have ended; the child is still running
                await await_or_fail(backend.child_started.wait(), what="the background child to start")
                finished_at_turn_end.extend(span.name for span in memory.get_finished_spans())
                backend.release_child.set()
                await wait_background_owner(loop)
            await loop.close()
            loop.store.close()

    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(scenario())
        else:
            asyncio.run(scenario())
    spans = memory.get_finished_spans()
    assert _tree(spans) == sorted(expected)
    # one owner: every started span ended exactly once, and nothing wrote to a span after it ended
    assert lifecycle.started == lifecycle.ended == {span.context.span_id: 1 for span in spans}
    assert [record.getMessage() for record in caplog.records] == []
    for span in spans:
        assert (span.status.status_code is StatusCode.ERROR) == (span.attributes.get("error.type") is not None)
    tools = [span for span in spans if span.name.startswith("execute_tool ")]
    assert sorted(span.attributes["zeta.tool.call.id"] for span in tools) == sorted(call.id for call in calls)
    for span in tools:
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["zeta.tool.type"] == ("agent" if span.name == "execute_tool agent" else "function")
        assert span.attributes["zeta.agent.name"] == "zeta"
        assert span.attributes["gen_ai.conversation.id"] == loop.store.session_id
        if span.attributes.get("error.type") != "CancelledError":  # a cancelled call has no result to size
            assert span.attributes["zeta.tool.call.arguments.size"] > 0
    probes = [span for span in tools if span.name == "execute_tool probe"]
    for span in probes:
        # sizes only: {"text": SECRET} and the one "ok" text block, never the text itself
        assert span.attributes["zeta.tool.call.arguments.size"] == len(f'{{"text": "{SECRET}"}}')
        assert span.attributes["zeta.tool.call.result.size"] > len("ok")
    approvals = [(event.name, dict(event.attributes)) for span in tools for event in span.events]
    for _name, attributes in approvals:
        assert type(attributes.pop("duration", 0)) is int
    assert approvals == ([("zeta.approval.requested", {"decision": "ask"}), ("zeta.approval.resolved", {"decision": "allow"})] if ask else [])
    children = [span for span in spans if span.name == CHILD]
    for child in children:
        assert child.attributes["zeta.background"] is background
        assert child.attributes["zeta.agent.depth"] == 1
        assert child.attributes["zeta.agent.name"] == "general"
        assert child.attributes["gen_ai.conversation.id"] == f"{loop.store.session_id}:1"
    if background:
        assert "execute_tool agent" in finished_at_turn_end and A in finished_at_turn_end
        assert CHILD not in finished_at_turn_end
    _assert_no_content(spans)


def test_disabled_tool_and_child_turn_creates_no_span(tmp_path, monkeypatch) -> None:
    def no_span(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disabled telemetry reached span construction")

    # test-contract: seam the disabled facade must never reach span construction
    monkeypatch.setattr(Telemetry, "_span", no_span)
    backend = RoutedBackend([ToolCall("toolu_ok", "probe", {}), _agent("toolu_fg", background=False)])
    loop, _stalled, _both = _loop(tmp_path, backend, Telemetry.disabled(), ask=True)

    async def scenario() -> list[StreamEvent]:
        try:
            events = []
            async for event in loop.run_turn(PROMPT, origin=MessageOrigin.USER):
                events.append(event)
                if event.type is StreamEventType.TOOL_APPROVAL_START:
                    assert loop.tool_registry.approval_policy.approve(event.tool_call.id)
            return events
        finally:
            await loop.close()
            loop.store.close()

    events = asyncio.run(scenario())
    ends = [event.tool_result for event in events if event.type is StreamEventType.TOOL_EXECUTION_END]
    assert [result.is_error for result in ends] == [False, False]
