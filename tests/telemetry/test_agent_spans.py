"""Agent, turn, and inference spans through the real AgentLoop turn."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ErrorInfo,
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
from zeta.providers.completion import PrematureCompletionError, owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.telemetry import Telemetry

pytest.importorskip("opentelemetry.sdk")

PROMPT = "summarize the secret plan"
REPLY = "the plan stays private"
USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2}


class ScriptedFailure(RuntimeError):
    """A non-retryable provider failure carrying content in its message."""


class Escape(BaseException):
    """A BaseException that is neither cancellation nor generator close."""


class ScriptedBackend:
    """One scripted attempt per provider call; ``"stall"`` waits until cancelled.

    ``("truncated", usage)`` sends a truncated MESSAGE_END carrying ``usage`` and then
    fails retryably, so the provider retries after reporting that attempt's usage.
    """

    def __init__(self, provider: str, model: str, attempts: Sequence[object], start: Mapping[str, str] | None = None) -> None:
        self.provider = provider
        self.model = model
        self.attempts = list(attempts)
        self.start = dict(start or {})
        self.stalled = asyncio.Event()

    async def _completion_wait(self, delay: float) -> None:
        del delay  # retries do not sleep in tests

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        step = self.attempts.pop(0)
        yield StreamEvent(StreamEventType.MESSAGE_START, data=dict(self.start))
        if step == "stall":
            self.stalled.set()
            await await_or_fail(asyncio.Event().wait(), what="the test to cancel the stalled provider call")
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, ErrorInfo):
            yield StreamEvent(StreamEventType.ERROR, error=step)
            return
        if isinstance(step, tuple):
            yield StreamEvent(StreamEventType.MESSAGE_END, data={"usage": dict(step[1]), "truncated": True})
            raise PrematureCompletionError(PROMPT)
        blocks = [TextContent(REPLY)] if step == "reply" else [ToolUseContent(ToolCall("toolu_01", "noop", {"text": PROMPT}))]
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
            data={"usage": dict(USAGE), "stop_reason": "end_turn" if step == "reply" else "tool_use"},
        )


@pytest.fixture
def traced() -> Iterator[tuple[Telemetry, list]]:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
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

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trace_exporter, "OTLPSpanExporter", DropExporter)
        telemetry = Telemetry.configure("http://127.0.0.1:4318/v1/traces")
    memory = InMemorySpanExporter()
    telemetry._implementation._provider.add_span_processor(SimpleSpanProcessor(memory))
    try:
        yield telemetry, memory
    finally:
        telemetry.shutdown()


def _loop(tmp_path: Path, backend: ScriptedBackend, telemetry: Telemetry, **options: object) -> AgentLoop:
    async def noop(arguments: dict[str, object], abort_signal: object, publisher: object) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    return AgentLoop(
        backend,
        ConversationStore(tmp_path / "sessions"),
        tools={"noop": noop},
        max_turns=4,
        skill_catalog=SkillCatalog.empty(),
        telemetry=telemetry,
        **options,
    )


async def _run(loop: AgentLoop) -> list[StreamEvent]:
    try:
        return [event async for event in loop.run_turn(PROMPT, origin=MessageOrigin.USER)]
    finally:
        await loop.close()
        loop.store.close()


def _by_name(spans: Sequence) -> dict[str, list]:
    found: dict[str, list] = {}
    for span in spans:
        found.setdefault(span.name, []).append(span)
    return found


def _assert_no_content(spans: Sequence) -> None:
    texts = [span.name for span in spans]
    for span in spans:
        texts += [f"{key}={value}" for key, value in span.attributes.items()]
        texts += [span.status.description or ""]
        for event in span.events:
            texts += [event.name, *(f"{key}={value}" for key, value in event.attributes.items())]
    for text in texts:
        assert PROMPT not in text and REPLY not in text and "secret" not in text, text
        assert not text.startswith(("gen_ai.input", "gen_ai.output", "gen_ai.system_instructions", "gen_ai.tool.call.arguments")), text


# MESSAGE_START data in each provider's own shape: anthropic.py message_start, codex.py response.created, ollama.py first frame
@pytest.mark.parametrize(
    ("provider", "model", "genai_provider", "start", "response_model"),
    [
        ("claude", "claude-opus-5-5", "anthropic", {"id": "msg_01", "model": "claude-opus-5-5-20261101", "role": "assistant"}, "claude-opus-5-5-20261101"),
        ("codex", "gpt-5.6-luna", "openai", {"id": "resp_01", "model": "gpt-5.6-luna-2026-11-01", "status": "in_progress"}, "gpt-5.6-luna-2026-11-01"),
        ("ollama", "qwen3:4b", "ollama", {"model": "qwen3:4b"}, "qwen3:4b"),
    ],
    ids=["anthropic", "openai", "ollama"],
)
def test_turns_and_inference_nest_under_one_agent_span(tmp_path, traced, provider, model, genai_provider, start, response_model) -> None:
    from opentelemetry.trace import SpanKind

    telemetry, memory = traced
    loop = _loop(tmp_path, ScriptedBackend(provider, model, ["tool", "reply"], start), telemetry)
    asyncio.run(_run(loop))
    spans = memory.get_finished_spans()
    names = _by_name(spans)
    assert sorted(names) == sorted(["invoke_agent zeta", "zeta.turn", f"chat {model}", "execute_tool noop"])
    (agent,) = names["invoke_agent zeta"]
    turns = sorted(names["zeta.turn"], key=lambda span: span.attributes["zeta.turn.number"])
    chats = names[f"chat {model}"]
    assert agent.parent is None
    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent.attributes["gen_ai.request.model"] == model
    assert agent.attributes["gen_ai.conversation.id"] == loop.store.session_id
    assert [turn.parent.span_id for turn in turns] == [agent.context.span_id] * 2
    assert [turn.attributes["zeta.tool.count"] for turn in turns] == [1, 0]
    assert [chat.parent.span_id for chat in chats] == [turn.context.span_id for turn in turns]
    (tool,) = names["execute_tool noop"]
    assert tool.parent.span_id == turns[0].context.span_id
    for chat in chats:
        assert chat.kind is SpanKind.CLIENT
        assert chat.attributes["gen_ai.provider.name"] == genai_provider
        assert chat.attributes["gen_ai.request.model"] == model
        assert chat.attributes["gen_ai.response.model"] == response_model
        assert chat.attributes["gen_ai.usage.input_tokens"] == 15
        assert chat.attributes["gen_ai.usage.output_tokens"] == 5
        assert chat.attributes["gen_ai.usage.total_tokens"] == 20
    assert [chat.attributes["gen_ai.response.finish_reasons"] for chat in chats] == ["tool_use", "end_turn"]
    assert all(span.attributes.get("error.type") is None for span in spans)
    _assert_no_content(spans)


def test_provider_retries_are_events_on_one_inference_span(tmp_path, traced) -> None:
    telemetry, memory = traced
    backend = ScriptedBackend("claude", "claude-opus-5-5", [PrematureCompletionError(PROMPT), PrematureCompletionError(PROMPT), "reply"])
    asyncio.run(_run(_loop(tmp_path, backend, telemetry)))
    (chat,) = _by_name(memory.get_finished_spans())["chat claude-opus-5-5"]
    retries = [event for event in chat.events if event.name == "gen_ai.client.retry"]
    assert [event.attributes["gen_ai.retry.attempt"] for event in retries] == [2, 3]
    assert {event.attributes["gen_ai.retry.reason"] for event in retries} == {"stream_error"}
    assert all(event.attributes["gen_ai.retry.after"] is False for event in retries)
    assert chat.attributes.get("error.type") is None
    _assert_no_content(memory.get_finished_spans())


def test_retry_usage_sums_every_attempt_on_the_one_span(tmp_path, traced) -> None:
    telemetry, memory = traced
    failed = {"input_tokens": 7, "output_tokens": 1, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 0}
    backend = ScriptedBackend("claude", "claude-opus-5-5", [("truncated", failed), "reply"])
    asyncio.run(_run(_loop(tmp_path, backend, telemetry)))
    (chat,) = _by_name(memory.get_finished_spans())["chat claude-opus-5-5"]
    assert [event.name for event in chat.events] == ["gen_ai.client.retry"]
    # 11 + 15 input (cache reads and writes included), 1 + 5 output
    assert chat.attributes["gen_ai.usage.input_tokens"] == 26
    assert chat.attributes["gen_ai.usage.output_tokens"] == 6
    assert chat.attributes["gen_ai.usage.total_tokens"] == 32
    assert chat.attributes["gen_ai.response.finish_reasons"] == "end_turn"


# rows: (attempts, the exception the turn raises, error.type on chat, on turn, on agent)
@pytest.mark.parametrize(
    ("attempts", "raised", "chat_error", "turn_error", "agent_error"),
    [
        (["reply"], None, None, None, None),
        ([ScriptedFailure(PROMPT)], None, "ScriptedFailure", "_OTHER", "_OTHER"),
        ([ErrorInfo("invalid_request_error", PROMPT)], None, "_OTHER", "_OTHER", "_OTHER"),
        ([Escape()], Escape, "Escape", "Escape", "Escape"),
        (["stall"], asyncio.CancelledError, "CancelledError", "CancelledError", "CancelledError"),
        ([PrematureCompletionError(PROMPT), "reply"], None, None, None, None),
    ],
    ids=["success", "provider-exception", "error-event", "base-exception", "cancelled-at-stream", "retry-then-success"],
)
def test_every_exit_ends_each_span_once_under_its_parent(tmp_path, traced, attempts, raised, chat_error, turn_error, agent_error) -> None:
    from opentelemetry.trace import StatusCode

    telemetry, memory = traced
    backend = ScriptedBackend("claude", "claude-opus-5-5", attempts)
    loop = _loop(tmp_path, backend, telemetry)

    async def scenario() -> None:
        task = asyncio.create_task(_run(loop))
        if attempts == ["stall"]:
            await await_or_fail(backend.stalled.wait(), what="the provider call to start")
            task.cancel()
        await await_or_fail(task, what="the turn to finish")

    if raised is None:
        asyncio.run(scenario())
    else:
        with pytest.raises(raised):
            asyncio.run(scenario())
    spans = memory.get_finished_spans()
    assert sorted(span.name for span in spans) == sorted(["invoke_agent zeta", "zeta.turn", "chat claude-opus-5-5"])
    names = _by_name(spans)
    (agent,), (turn,), (chat,) = names["invoke_agent zeta"], names["zeta.turn"], names["chat claude-opus-5-5"]
    assert agent.parent is None
    assert turn.parent.span_id == agent.context.span_id
    assert chat.parent.span_id == turn.context.span_id
    assert [span.attributes.get("error.type") for span in (chat, turn, agent)] == [chat_error, turn_error, agent_error]
    assert (chat.status.status_code is StatusCode.ERROR) == (chat_error is not None)
    _assert_no_content(spans)


def test_a_stream_error_event_fails_the_turn_and_agent_spans(tmp_path, traced) -> None:
    from opentelemetry.trace import StatusCode

    telemetry, memory = traced
    # a pre-send rejection: the queued prompt cannot fit, the loop yields ERROR and returns without an exception
    loop = _loop(tmp_path, ScriptedBackend("claude", "claude-opus-5-5", []), telemetry, token_budget=2_000)
    loop.store.pending_prompt_queue.append("p" * 15_000, origin=MessageOrigin.AGENT_SEND)
    events = asyncio.run(_run(loop))
    assert [event.error.code for event in events if event.type is StreamEventType.ERROR] == ["context_length_exceeded"]
    names = _by_name(memory.get_finished_spans())
    for span in (*names["invoke_agent zeta"], *names["zeta.turn"]):
        assert span.attributes["zeta.outcome"] == "failed", span.name
        assert span.status.status_code is StatusCode.ERROR, span.name
        assert span.attributes["zeta.error.code"] == "context_length", span.name
    _assert_no_content(memory.get_finished_spans())


def test_disabled_turn_creates_no_span(tmp_path, monkeypatch) -> None:
    def no_span(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disabled telemetry reached span construction")

    # test-contract: seam the disabled facade must never reach span construction
    monkeypatch.setattr(Telemetry, "_span", no_span)
    events = asyncio.run(_run(_loop(tmp_path, ScriptedBackend("claude", "claude-opus-5-5", ["tool", "reply"]), Telemetry.disabled())))
    assert [event.type for event in events].count(StreamEventType.TURN_END) == 2
