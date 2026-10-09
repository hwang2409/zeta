import asyncio
import json
import re
import shlex
import sys
import threading
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from io import StringIO
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import zeta.runtime.execution as execution_module
import zeta.runtime.loop.agent as agent_loop_module
import zeta.tools.agent_send as agent_send_module
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.background import (
    BackgroundAgentOwner,
    adopt_agent_children,
    finish_background_child,
)
from zeta.agent.presets import (
    AGENT_PRESETS,
    GENERAL_PRESET,
)
from zeta.agent.runner import _child_base_system_prompt
from zeta.core.abort import AbortGenerationRegistry
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.store import (
    MAX_AGENT_NOTIFICATION_TEXT,
    ConversationStore,
    PendingPromptsClosedError,
)
from zeta.mcp import MCPMount
from zeta.protocol.types import (
    CompletionBackend,
    ErrorInfo,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.tool_schema import canonical_tool_schemas
from zeta.skills import SkillCatalog, SkillMeta
from zeta.tools import ToolRegistry
from zeta.tools._action_metadata import ApprovalBinding, ResolvedCapability
from zeta.tools.agent import ChildApprovalPolicy, send_to_run
from zeta.tui.agent_card import AgentRunCommandMixin
from zeta.tui.app import TUIApp
from zeta.tui.render import render_approval_card, render_event
from zeta.tui.todo import TodoWidget


async def _collect(events):
    return [event async for event in events]


def _approval_capability(
    name: str, arguments: dict[str, object]
) -> ResolvedCapability:
    subject = "path" if name in {"read", "write", "edit"} else "command"
    binding = ApprovalBinding.PATH if subject == "path" else ApprovalBinding.CWD
    return ResolvedCapability(
        name,
        None,
        True,
        subject,
        arguments.get(subject),
        binding,
        None,
        arguments,
    )


@pytest.mark.parametrize("error_event", [False, True])
@pytest.mark.asyncio
async def test_context_overflow_compacts_and_retries_same_turn(
    tmp_path: Path, error_event: bool
) -> None:
    class ContextLimitBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                if error_event:
                    yield StreamEvent(
                        StreamEventType.ERROR,
                        error=ErrorInfo("context_length_exceeded", "stream error"),
                    )
                    return
                error = RuntimeError("context_length_exceeded: stream error")
                error.code = "context_length_exceeded"
                raise error
            answer = "summary" if not tool_schemas else "done"
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent(answer)]),
            )

    backend = ContextLimitBackend()
    store = ConversationStore(tmp_path)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("old reasoning " * 100), TextContent("old work")],
        )
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        token_budget=10_000,
        retained_tail=1,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_turn("current request", origin=MessageOrigin.USER))

    assert len(backend.calls) == 2
    assert store.compaction_marker_count() == 1
    assert sum(event.type is StreamEventType.TURN_START for event in events) == 1
    assert sum(event.type is StreamEventType.TURN_END for event in events) == 1
    assert any(event.type is StreamEventType.RETRY for event in events)
    assert not any(event.type is StreamEventType.ERROR for event in events)
    assert not any(message.metadata.get("turn_failed") for message in store.messages())
    await loop.close()


@pytest.mark.asyncio
async def test_context_overflow_forces_emergency_eviction_and_retries_once(
    tmp_path: Path,
) -> None:
    class ContextLimitBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                error = RuntimeError(
                    "prompt is too long: 120 tokens > 100 maximum"
                )
                error.code = "context_length_exceeded"
                error.provider_prompt_tokens = 120
                raise error
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("recovered")]),
            )

    store = ConversationStore(tmp_path)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("old reasoning " * 100), TextContent("old result")],
        )
    )
    backend = ContextLimitBackend()
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        backend=backend,
        token_counter=lambda message: 1
        if message.metadata.get("context_evicted")
        else 40,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        context_assembler=assembler,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_turn("current request", origin=MessageOrigin.USER))

    assert len(backend.calls) == 2
    assert sum(map(assembler.token_counter, backend.calls[1])) <= 50
    assert sum(map(assembler.token_counter, backend.calls[1])) < sum(
        map(assembler.token_counter, backend.calls[0])
    )
    assert assembler.calibration_ratio == pytest.approx(1.5)
    completed = store.messages()[-1]
    assert completed.metadata["context_calibration_ratio"] == pytest.approx(1.5)
    assert not any(event.type is StreamEventType.ERROR for event in events)
    await loop.close()


@pytest.mark.asyncio
async def test_context_overflow_retry_failure_persists_turn_once(
    tmp_path: Path,
) -> None:
    class AlwaysOverflowBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            self.calls.append(list(messages))
            error = RuntimeError("prompt is too long: 120 tokens > 100 maximum")
            error.code = "context_length_exceeded"
            error.provider_prompt_tokens = 120
            raise error
            yield  # pragma: no cover

    store = ConversationStore(tmp_path)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("old reasoning " * 100), TextContent("old result")],
        )
    )
    backend = AlwaysOverflowBackend()
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        token_counter=lambda message: 1
        if message.metadata.get("context_evicted")
        else 40,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        context_assembler=assembler,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_turn("current request", origin=MessageOrigin.USER))

    assert len(backend.calls) == 2
    messages = store.messages()
    assert sum(
        message.role is MessageRole.USER
        and message.content == [TextContent("current request")]
        for message in messages
    ) == 1
    failed = [
        message
        for message in messages
        if message.role is MessageRole.ASSISTANT
        and message.metadata.get("turn_failed")
    ]
    assert len(failed) == 1
    assert not any(message.role is MessageRole.TOOL_RESULT for message in messages)
    assert sum(event.type is StreamEventType.RETRY for event in events) == 1
    await loop.close()


@pytest.mark.asyncio
async def test_overflow_without_reduction_fails_with_diagnostic(tmp_path: Path) -> None:
    class AlwaysOverflowBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            self.calls += 1
            error = RuntimeError("prompt is too long: 220 tokens > 200 maximum")
            error.code = "context_length_exceeded"
            raise error
            yield  # pragma: no cover

    backend = AlwaysOverflowBackend()
    store = ConversationStore(tmp_path)
    store.append_message(
        with_message_origin(
            Message(MessageRole.USER, [TextContent("old request")]),
            MessageOrigin.USER,
        )
    )
    assembler = ContextAssembler(
        store,
        token_budget=200,
        retained_tail=1,
        system_prompt="",
        token_counter=lambda _: 1,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        context_assembler=assembler,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_turn("only request", origin=MessageOrigin.USER))

    errors = [event.error for event in events if event.type is StreamEventType.ERROR]
    assert backend.calls == 1
    assert errors and "could not reduce" in errors[0].message
    assert "raise --token-budget" in errors[0].message
    await loop.close()


@pytest.mark.asyncio
async def test_agent_loop_uses_fallback_after_empty_summary_retries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("old request")]), MessageOrigin.USER))
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent(" ")]),
            ScriptedTurn(),
            ScriptedTurn([TextContent("\n")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )

    def token_count(message: Message) -> int:
        if (
            message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION}
            or message.metadata.get("compaction_summary")
        ):
            return 1
        return 30

    assembler = ContextAssembler(
        store,
        token_budget=40,
        retained_tail=1,
        system_prompt="",
        backend=backend,
        token_counter=token_count,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        context_assembler=assembler,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_turn("current request", origin=MessageOrigin.USER))

    assert len(backend.calls) == 4
    assert not any(event.type is StreamEventType.ERROR for event in events)
    assert any(event.type is StreamEventType.TURN_END for event in events)
    marker = next(entry for entry in store.entries if entry.type == "compaction")
    assert marker.data["summary"].startswith(
        "[automatic fallback summary: model returned no summary]"
    )
    assert "user: old request" in marker.data["summary"]
    assert assembler.last_compaction_telemetry["fallback_count"] == 1
    assert assembler.last_compaction_telemetry["retries"] == 2
    assert "model returned no summary" in caplog.text
    await loop.close()


@pytest.mark.asyncio
async def test_agent_loop_compacts_oversized_tool_output_instead_of_failing(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("inspect the output")]), MessageOrigin.USER))
    call = ToolCall("call-noisy", "bash", {"command": "print lots"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(call.id, "noisy output\n" * 8_000),
        )
    )
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("The noisy command completed.")])
            for _ in range(40)
        ]
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        token_budget=10_000,
        retained_tail=8,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_turn("continue", origin=MessageOrigin.USER))

    assert any(event.type is StreamEventType.TURN_END for event in events)
    assert not any(event.type is StreamEventType.ERROR for event in events)
    assert store.compaction_marker_count() == 1
    assert backend.calls[-1][0][-1].content == [TextContent("continue")]
    await loop.close()


@pytest.mark.asyncio
async def test_context_overflow_after_partial_output_is_not_retried(
    tmp_path: Path,
) -> None:
    class PartialBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            self.calls += 1
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE, content=TextContent("partial")
            )
            error = RuntimeError("context_length_exceeded: stream error")
            error.code = "context_length_exceeded"
            raise error

    backend = PartialBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    events = await _collect(loop.run_turn("current request", origin=MessageOrigin.USER))

    assert backend.calls == 1
    assert store.compaction_marker_count() == 0
    assert any(
        event.type is StreamEventType.ERROR
        and event.error is not None
        and event.error.code == "context_length_exceeded"
        for event in events
    )
    await loop.close()


@pytest.mark.asyncio
async def test_background_owner_waits_for_child_close_before_unregister(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(tmp_path / "child")
    call = _agent_call()
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="task research",
    )
    child_store.mark_agent_parent(call.id)
    owner = BackgroundAgentOwner(parent_store)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def close_child() -> None:
        close_started.set()
        await release_close.wait()

    def cleanup() -> None:
        cleanup_called.set()
        owner.unregister("child-1")

    watcher = asyncio.create_task(
        finish_background_child(
            child_task=asyncio.create_task(
                asyncio.sleep(
                    0,
                    result={
                        "content": [{"text": "done"}],
                        "isError": False,
                    },
                )
            ),
            child_store=child_store,
            parent_store=parent_store,
            notification_store=parent_store,
            tool_call=call,
            child_instance_id="child-1",
            child_path=str(child_store.session_dir),
            description="task research",
            child_turns=lambda: 0,
            build_result=lambda text, error, status: {
                "content": [{"text": text}],
                "isError": error,
                "structuredContent": {"status": status},
            },
            validate_result=lambda result, tool_call_id: ToolResult(
                tool_call_id,
                result["content"][0]["text"],
                is_error=result["isError"],
                structured_content=result["structuredContent"],
            ),
            publish_event=lambda event: None,
            cleanup=cleanup,
            close_child=close_child,
            error_message=lambda exc: str(exc),
            background_owner=owner,
        )
    )
    owner.register("child-1", lambda: None, watcher, parent_store)

    await close_started.wait()
    waiting = asyncio.create_task(owner.wait())
    await asyncio.sleep(0)
    assert not waiting.done()
    assert not cleanup_called.is_set()

    release_close.set()
    await watcher
    await waiting
    assert cleanup_called.is_set()


@pytest.mark.asyncio
async def test_background_completion_truncates_long_killed_task_id(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(tmp_path / "child")
    call = _agent_call("long-task-id")
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="task research",
    )
    long_task_id = "task-" + "x" * 80

    await finish_background_child(
        child_task=asyncio.create_task(
            asyncio.sleep(
                0,
                result={"content": [{"text": "done"}], "isError": False},
            )
        ),
        child_store=child_store,
        parent_store=parent_store,
        notification_store=parent_store,
        tool_call=call,
        child_instance_id="child-1",
        child_path=str(child_store.session_dir),
        description="task research",
        child_turns=lambda: 0,
        build_result=lambda text, error, status: {
            "content": [{"text": text}],
            "isError": error,
            "structuredContent": {"status": status},
        },
        validate_result=lambda result, tool_call_id: ToolResult(
            tool_call_id,
            result["content"][0]["text"],
            is_error=result["isError"],
            structured_content=result["structuredContent"],
        ),
        publish_event=lambda event: None,
        cleanup=lambda: None,
        close_child=lambda: asyncio.sleep(0, result=(long_task_id,)),
        error_message=str,
    )

    notification = parent_store.agent_notifications()[0]
    assert notification.data["killed_task_ids"] == [long_task_id[:64]]
    assert notification.data["killed_task_count"] == 1
    assert notification.data["killed_task_ids_truncated"] is True
    child_store.close()
    parent_store.close()


def _agent_call(call_id: str = "agent-1", agent_type: str | None = None) -> ToolCall:
    arguments = {"prompt": "inspect the task", "description": "task research"}
    if agent_type is not None:
        arguments["agent_type"] = agent_type
    return ToolCall(
        call_id,
        "agent",
        arguments,
    )


class ParallelChildrenBackend(CompletionBackend):
    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.parent_calls = list(calls)
        self.call_count = 0
        self.child_prompts = {
            call.arguments["prompt"] for call in calls if "prompt" in call.arguments
        }
        self.started_prompts: set[object] = set()
        self.children_started = asyncio.Event()
        self.release_children = {
            prompt: asyncio.Event() for prompt in self.child_prompts
        }
        self.children_completed = {
            prompt: asyncio.Event() for prompt in self.child_prompts
        }

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(call) for call in self.parent_calls]
        else:
            prompt = next(
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            )
            self.started_prompts.add(prompt)
            if self.started_prompts == self.child_prompts:
                self.children_started.set()
            await self.release_children[prompt].wait()
            blocks = [TextContent(f"response for {prompt}")]
            self.children_completed[prompt].set()
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ParallelApprovalBackend(CompletionBackend):
    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.parent_calls = list(calls)
        self.call_count = 0
        self.child_count = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(call) for call in self.parent_calls]
        elif self.call_count <= 3:
            self.child_count += 1
            blocks = [
                ToolUseContent(
                    ToolCall(
                        f"child-bash-{self.child_count}",
                        "bash",
                        {"cmd": f"echo child-{self.child_count}"},
                    )
                )
            ]
        else:
            blocks = [TextContent(f"child-final-{self.call_count}")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _parallel_agent_calls() -> list[ToolCall]:
    return [
        ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": f"inspect {index}", "description": f"task {index}"},
        )
        for index in (1, 2)
    ]


class BackgroundBackend(CompletionBackend):
    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.calls = list(calls)
        self.child_text = "child complete"
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if any(
            message.role is MessageRole.SYSTEM
            and any(
                isinstance(block, TextContent)
                and block.text.startswith(
                    "durable notifications (kind is agent_completion when omitted):"
                )
                for block in message.content
            )
            for message in messages
        ):
            blocks = [TextContent("parent reacted to completion")]
        elif last_user == "start":
            blocks = [ToolUseContent(call) for call in self.calls]
        elif last_user == "inspect the task":
            self.child_started.set()
            await self.release_child.wait()
            blocks = [TextContent(self.child_text)]
        else:
            blocks = [TextContent("parent continued")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class NotificationWakeBackend(CompletionBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.wake_started = asyncio.Event()
        self.release_wake = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        self.calls.append(list(messages))
        is_notification = any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        )
        if is_notification:
            self.wake_started.set()
            await self.release_wake.wait()
        blocks = [TextContent("wake response" if is_notification else "user response")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class NestedBlockingBackend(CompletionBackend):
    def __init__(self) -> None:
        self.call_count = 0
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(_agent_call("child"))]
        elif self.call_count == 2:
            blocks = [ToolUseContent(_agent_call("grandchild"))]
        else:
            self.grandchild_started.set()
            await self.release_grandchild.wait()
            blocks = [TextContent("grandchild complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class NestedBackgroundBackend(CompletionBackend):
    def __init__(self) -> None:
        self.call_count = 0
        self.grandchild_started = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(_background_agent_call("child"))]
        elif self.call_count == 2:
            nested = _background_agent_call("grandchild")
            nested.arguments["description"] = "grandchild"
            blocks = [ToolUseContent(nested)]
        else:
            self.grandchild_started.set()
            await asyncio.Event().wait()
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ForegroundNestedBackgroundBackend(CompletionBackend):
    def __init__(self) -> None:
        self.child_nested = False
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            child = _agent_call("child")
            child.arguments["prompt"] = "child prompt"
            blocks = [ToolUseContent(child)]
        elif last_user == "child prompt" and not self.child_nested:
            self.child_nested = True
            grandchild = _background_agent_call("grandchild")
            grandchild.arguments["prompt"] = "grandchild prompt"
            grandchild.arguments["description"] = "grandchild"
            blocks = [ToolUseContent(grandchild)]
        elif last_user == "grandchild prompt":
            self.grandchild_started.set()
            await self.release_grandchild.wait()
            blocks = [TextContent("grandchild complete")]
        else:
            blocks = [TextContent("child complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ParallelNestedReadBackend(CompletionBackend):
    def __init__(self, count: int) -> None:
        self.calls = []
        for index in range(count):
            call = _agent_call(f"agent-{index}")
            call.arguments["prompt"] = f"inspect {index}"
            self.calls.append(call)
        self.seen_prompts: set[str] = set()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            blocks = [ToolUseContent(call) for call in self.calls]
        elif last_user not in self.seen_prompts:
            self.seen_prompts.add(last_user)
            blocks = [
                ToolUseContent(
                    ToolCall(
                        f"{last_user}-read",
                        "read",
                        {"path": "missing"},
                    )
                )
            ]
        else:
            blocks = [TextContent("child complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _background_agent_call(call_id: str = "background-1") -> ToolCall:
    return ToolCall(
        call_id,
        "agent",
        {
            "prompt": "inspect the task",
            "description": "background research",
            "background": True,
        },
    )


async def _wait_for_notification(store: ConversationStore, status: str) -> object:
    for _ in range(100):
        notifications = store.agent_notifications()
        if notifications and notifications[-1].data["status"] == status:
            return notifications[-1]
        await asyncio.sleep(0.01)
    raise AssertionError(f"missing {status} background notification")


async def _wait_for_no_agent_children(store: ConversationStore) -> None:
    for _ in range(100):
        if not store.agent_children():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("background child cleanup did not finish")


@pytest.mark.asyncio
async def test_background_agent_returns_handle_and_parent_continues(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    assert result.structured_content["description"] == "background research"
    await asyncio.wait_for(backend.child_started.wait(), timeout=5)

    parent_events = await _collect(loop.run_turn("follow up", origin=MessageOrigin.USER))
    assert any(event.type is StreamEventType.TURN_END for event in parent_events)
    assert store.agent_notifications() == []

    backend.release_child.set()
    notification = await _wait_for_notification(store, "completed")
    assert notification.data["text"].startswith("child complete")
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_wakes_idle_parent(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    wake_events: list[StreamEvent] = []
    wake_task: asyncio.Task[None] | None = None
    loop.set_background_event_sink(background_events.append)

    async def consume_wake() -> None:
        wake_events.extend(await _collect(loop.run_notification_turn()))

    def wake() -> None:
        nonlocal wake_task
        if wake_task is None or wake_task.done():
            wake_task = asyncio.create_task(consume_wake())

    loop.set_background_wake_callback(wake)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    backend.release_child.set()
    while wake_task is None:
        await asyncio.sleep(0)
    await wake_task

    assert wake_events[0].type is StreamEventType.AGENT_NOTIFICATION
    assert wake_events[0].data["text"].startswith("child complete")
    assert wake_events[0].data["text"].count("error=false") == 1
    assert wake_events[0].data["text"].count("canceled=false") == 1
    rendered = render_event(wake_events[0])
    assert rendered is not None
    assert rendered.plain.startswith("⏺ background research · completed · ")
    assert "error=false" not in rendered.plain
    assert "canceled=false" not in rendered.plain
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.content.count("error=false") == 1
    wake_message = next(
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "agent_notifications"
    )
    assert wake_message.role is MessageRole.SYSTEM
    assert "child complete" in wake_message.content[0].text
    assert loop.store.agent_notifications() == []
    await loop.close()


@pytest.mark.asyncio
async def test_user_submission_waits_behind_notification_wake(
    tmp_path: Path,
) -> None:
    backend = NotificationWakeBackend()
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    app = TUIApp(
        AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()),
        provider="codex",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submissions.wake()
    await backend.wake_started.wait()
    app._submissions.submit("follow up")
    backend.release_wake.set()

    async with asyncio.timeout(5):
        while len(backend.calls) < 2 or app._submissions.active:
            await asyncio.sleep(0)

    assert len(backend.calls) == 2
    assert any(
        message.role is MessageRole.USER
        and any(
            isinstance(block, TextContent) and block.text == "follow up"
            for block in message.content
        )
        for message in backend.calls[1]
    )
    await app.close()


@pytest.mark.asyncio
async def test_notification_wake_waits_for_resumed_durable_tool(
    tmp_path: Path,
) -> None:
    from zeta.submission.pipeline import SubmissionPipeline, _DurableToolDone

    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
    starts: list[bool] = []
    release = asyncio.Event()

    async def provider_turn() -> None:
        starts.append(True)
        await release.wait()

    host = SimpleNamespace(
        loop=loop,
        _start_turn=lambda *_args, **_kwargs: asyncio.create_task(provider_turn()),
    )
    pipeline = SubmissionPipeline(host)
    durable = asyncio.create_task(asyncio.sleep(0))
    await durable
    pipeline._durable_tasks["approval-1"] = durable
    try:
        pipeline.wake()
        await asyncio.sleep(0)
        assert starts == []
        pipeline._send(_DurableToolDone("approval-1", durable))
        for _ in range(3):
            await asyncio.sleep(0)
        assert starts == [True]
    finally:
        release.set()
        await pipeline.close()
        await loop.close()




@pytest.mark.asyncio
async def test_cancel_before_tool_dispatch_persists_canceled_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = [
        ToolCall("call-before-dispatch-1", "echo", {}),
        ToolCall("call-before-dispatch-2", "echo", {}),
    ]
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=calls)]),
        store,
        tools={"echo": lambda _: "must not run"},
        approval_policy=policy,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    assistant_persisted = asyncio.Event()
    original_append = store.append_message_with_approval_requests

    def append_and_signal(message, approval_requests):
        original_append(message, approval_requests)
        assistant_persisted.set()

    monkeypatch.setattr(store, "append_message_with_approval_requests", append_and_signal)

    async def stalled_dispatch(*args, **kwargs):
        await asyncio.Event().wait()
        yield  # pragma: no cover

    monkeypatch.setattr(agent_loop_module, "dispatch_tool_calls", stalled_dispatch)
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await assistant_persisted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert policy.pending_requests() == []
    messages = [message for message in store.messages() if message.tool_result is not None]
    assert len(messages) == len(calls)
    assert {message.tool_result.tool_call_id for message in messages} == {
        call.id for call in calls
    }
    assert all(message.tool_result.is_canceled for message in messages)
    store.close()
    reopened = ConversationStore(tmp_path)
    assert reopened.pending_approvals() == []
    await loop.close()
    reopened.close()


@pytest.mark.asyncio
async def test_generator_exit_before_tool_dispatch_persists_canceled_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = [
        ToolCall("call-generator-exit-1", "echo", {}),
        ToolCall("call-generator-exit-2", "echo", {}),
    ]
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=calls)]),
        store,
        tools={"echo": lambda _: "must not run"},
        approval_policy=policy,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    class GeneratorExitDispatch:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise GeneratorExit

        async def aclose(self):
            return None

    monkeypatch.setattr(
        agent_loop_module,
        "dispatch_tool_calls",
        lambda *args, **kwargs: GeneratorExitDispatch(),
    )
    with pytest.raises(GeneratorExit):
        await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert policy.pending_requests() == []
    receipts = [message for message in store.messages() if message.tool_result is not None]
    assert len(receipts) == len(calls)
    assert {message.tool_result.tool_call_id for message in receipts} == {
        call.id for call in calls
    }
    assert all(message.tool_result.is_canceled for message in receipts)
    store.close()
    reopened = ConversationStore(tmp_path)
    assert reopened.pending_approvals() == []
    await loop.close()
    reopened.close()


@pytest.mark.asyncio
async def test_notification_batch_reaches_provider_context(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("follow up", origin=MessageOrigin.USER))

    assert any(
        message.metadata.get("zeta_event") == "agent_notifications"
        for message in backend.calls[0][0]
    )
    assert store.agent_notifications() == []
    await loop.close()


@pytest.mark.asyncio
async def test_aborted_notification_wake_commits_durable_notifications(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done 1",
    )
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
    wake = asyncio.Event()
    loop.set_background_wake_callback(wake.set)

    appended_during_wake = False

    async def consume() -> None:
        nonlocal appended_during_wake
        task = asyncio.current_task()
        assert task is not None
        async for event in loop.run_notification_turn():
            if (
                event.type is StreamEventType.AGENT_NOTIFICATION
                and not appended_during_wake
            ):
                appended_during_wake = True
                store.append_agent_notification(
                    "child-2",
                    child_session_path="/tmp/child-2",
                    description="child",
                    status="completed",
                    text="done 2",
                )
                task.cancel()

    wake_task = asyncio.create_task(consume())
    with pytest.raises(asyncio.CancelledError):
        await wake_task
    try:
        assert store.agent_notifications() == []
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 2
        assert not wake.is_set()
    finally:
        await loop.close()


@pytest.mark.asyncio
async def test_failed_notification_wake_commits_durable_batch(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="done",
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn()], close_error=RuntimeError("boom")),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await _collect(loop.run_notification_turn())

    assert any(event.type is StreamEventType.ERROR for event in events)
    assert store.agent_notifications() == []
    assert sum(
        message.metadata.get("zeta_event") == "agent_notifications"
        for message in store.messages()
    ) == 1
    assert loop.notification_turn_state == "idle"
    await loop.close()


@pytest.mark.asyncio
async def test_notification_completion_during_wake_continues_same_turn(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="first",
    )
    serialized_calls = 0

    def serialize(
        _messages: Sequence[Message], _schemas: Sequence[ToolSchema]
    ) -> bytes:
        nonlocal serialized_calls
        serialized_calls += 1
        if serialized_calls == 1:
            store.append_agent_notification(
                "child-2",
                child_session_path="/tmp/child-2",
                description="child",
                status="completed",
                text="second",
            )
        return b"request"

    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first response")]),
            ScriptedTurn([TextContent("second response")]),
        ],
        request_serializer=serialize,
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    events = await _collect(loop.run_notification_turn())

    assert len(backend.calls) == 2
    assert [
        event.data["child_instance_id"]
        for event in events
        if event.type is StreamEventType.AGENT_NOTIFICATION
    ] == ["child-1", "child-2"]
    assert not any(
        event.type is StreamEventType.ERROR
        and event.error is not None
        and event.error.code == "max_turns"
        for event in events
    )
    await loop.close()


@pytest.mark.asyncio
async def test_background_multibyte_receipt_fits_persisted_limit(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    backend.child_text = "😀" * 1_800
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.content.count("error=false") == 1
    assert terminal.tool_result.content.count("canceled=false") == 1
    notification = store.agent_notifications(pending_only=False)[0]
    notification_row = next(
        row
        for row in store.path.read_bytes().splitlines()
        if b'"type":"notification"' in row
    )
    assert len(notification_row) <= 10_000
    assert notification.data["text"].count("error=false") == 1
    assert notification.data["text"].count("canceled=false") == 1
    persisted = Message(
        MessageRole.TOOL_RESULT,
        [TextContent(terminal.tool_result.content)],
        tool_result=terminal.tool_result,
    )
    assert (
        len(json.dumps(persisted.to_dict(), ensure_ascii=False).encode("utf-8"))
        <= 10_000
    )
    await loop.close()


@pytest.mark.asyncio
async def test_background_large_configured_receipt_fits_notification_store(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    prioritized_tail = "reply-priority-tail"
    backend.child_text = "x" * 20_000 + prioritized_tail
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        max_output_chars=50_000,
        skill_catalog=SkillCatalog.empty(),
    )
    loop = AgentLoop(
        backend,
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    backend.release_child.set()
    notification = await _wait_for_notification(store, "completed")

    text = notification.data["text"]
    assert len(text) <= MAX_AGENT_NOTIFICATION_TEXT
    assert prioritized_tail in text
    assert "[earlier report truncated]" in text
    assert text.count("error=false") == 1
    assert text.count("canceled=false") == 1
    reopened = ConversationStore(tmp_path, session_id=store.session_id)
    assert reopened.agent_notifications(pending_only=False)[0].data["text"] == text
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"] == text
    child.close()
    reopened.close()
    await loop.close()


@pytest.mark.asyncio
async def test_notification_during_wake_setup_waits_for_next_delivery(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="first",
    )
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first response")]),
            ScriptedTurn([TextContent("second response")]),
        ]
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    async def delayed_setup() -> None:
        store.append_agent_notification(
            "child-2",
            child_session_path="/tmp/child-2",
            description="child",
            status="completed",
            text="second",
        )

    loop._ensure_mcp_servers = delayed_setup
    events = await _collect(loop.run_notification_turn())

    notification_events_seen = [
        event.data["child_instance_id"]
        for event in events
        if event.type is StreamEventType.AGENT_NOTIFICATION
    ]
    assert notification_events_seen == ["child-1", "child-2"]
    assert len(backend.calls) == 1
    wake_batches = [
        [entry["child_instance_id"] for entry in message.metadata["notifications"]]
        for message in backend.calls[0][0]
        if message.metadata.get("zeta_event") == "agent_notifications"
    ]
    assert wake_batches == [["child-1"], ["child-2"]]
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_during_setup_is_drained_at_turn_start(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skip_mcp_mount=True,
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    setup_started = asyncio.Event()

    async def delayed_setup() -> None:
        setup_started.set()
        backend.release_child.set()
        while not store.agent_notifications():
            await asyncio.sleep(0)

    loop._ensure_mcp_servers = delayed_setup
    events = await _collect(loop.run_turn("follow up", origin=MessageOrigin.USER))

    assert setup_started.is_set()
    assert events[0].type is StreamEventType.AGENT_NOTIFICATION
    assert events[0].data["status"] == "completed"
    assert events[1].type is StreamEventType.AGENT_START
    assert store.agent_notifications() == []
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_leaves_no_pending_abort_waiter(
    tmp_path: Path,
) -> None:
    baseline = set(asyncio.all_tasks())
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop._background_owner.wait()

    assert not [
        task for task in asyncio.all_tasks() if task not in baseline and not task.done()
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_parent_abort_cancels_background_agent(tmp_path: Path) -> None:
    call = _background_agent_call()
    backend = BackgroundBackend([call])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    loop.abort()
    notification = await _wait_for_notification(store, "canceled")
    assert "parent session exited" not in notification.data["text"]
    assert notification.data["text"].count("error=false") == 1
    assert notification.data["text"].count("canceled=true") == 1
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.is_error is False
    assert terminal.tool_result.is_canceled is True
    rendered = render_event(terminal)
    assert rendered is not None
    assert rendered.plain.count("error=false") == 1
    assert rendered.plain.count("canceled=true") == 1
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_canceled() == {
        "tool_call_id": call.id,
        "content": "tool execution canceled",
    }
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_parent_abort_cancels_background_grandchild(
    tmp_path: Path,
) -> None:
    backend = NestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    loop.abort()
    await _wait_for_notification(store, "canceled")

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert child_store.agent_notifications(pending_only=False)[0].data["status"] == (
        "canceled"
    )
    assert grandchild_store.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }
    await _wait_for_no_agent_children(store)
    await loop.close()


@pytest.mark.asyncio
async def test_foreground_cancel_cancels_owned_background_descendant_only(
    tmp_path: Path,
) -> None:
    backend = ForegroundNestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))

    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=5)
    sibling_done = asyncio.Event()
    sibling_task = asyncio.create_task(sibling_done.wait())
    loop._background_owner.register(
        "root:sibling",
        sibling_task.cancel,
        sibling_task,
        parent_store=store,
    )
    foreground_handle = f"{store.session_id}:1"
    loop._background_owner.cancel_subtree(foreground_handle)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    grandchild_store = ConversationStore(
        store.session_dir / "agents" / "1" / "agents", session_id="1"
    )
    assert grandchild_store.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }
    assert not sibling_task.done()
    loop._background_owner.cancel("root:sibling")
    await asyncio.gather(sibling_task, return_exceptions=True)
    await loop.close()


@pytest.mark.asyncio
async def test_foreground_child_does_not_wait_for_background_grandchild(
    tmp_path: Path,
) -> None:
    backend = ForegroundNestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    events = await asyncio.wait_for(task, timeout=1)

    assert any(event.type is StreamEventType.TURN_END for event in events)
    assert store.agent_notifications() == []
    marker = next(iter(store.agent_children().values()))
    assert marker["child_session_path"] == str(
        store.session_dir / "agents" / "1" / "agents" / "1"
    )
    assert loop._background_owner.owns_running(f"{store.session_id}:1:1")

    backend.release_grandchild.set()
    notification = await _wait_for_notification(store, "completed")
    assert notification.data["text"].startswith("grandchild complete")
    await loop.close()
    assert not store.agent_children()


@pytest.mark.asyncio
async def test_background_and_foreground_tools_mix_in_one_turn(
    tmp_path: Path,
) -> None:
    background = _background_agent_call()
    foreground = ToolCall("read-1", "read", {"path": "missing"})
    backend = BackgroundBackend([background, foreground])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    results = [
        message.tool_result for message in store.messages() if message.tool_result
    ]
    assert len(results) == 2
    assert any(
        result is not None
        and result.structured_content is not None
        and result.structured_content.get("status") == "running"
        for result in results
    )
    assert any(result is not None and result.is_error for result in results)
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


def _persist_background_receipt(
    store: ConversationStore, call: ToolCall, child: ConversationStore
) -> None:
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("start")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("background agent started")],
            tool_result=ToolResult(
                call.id,
                "background agent started",
                structured_content={
                    "turns_used": 0,
                    "child_session_path": str(child.session_dir),
                    "status": "running",
                    "child_instance_id": f"{store.session_id}:1",
                    "description": "background research",
                },
            ),
        )
    )


def test_resume_cancels_live_background_child(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _background_agent_call()
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.mark_agent_parent(call.id)
    _persist_background_receipt(store, call, child)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="background research",
        background=True,
    )

    resumed = ConversationStore(tmp_path, session_id="parent")
    AgentLoop(BackgroundBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "canceled"
    assert "session exited" in notification.data["text"]
    assert not resumed.agent_children()
    assert ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": call.id,
        "content": "tool execution canceled",
    }


def test_resume_keeps_completed_unnotified_background_notification(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _background_agent_call()
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.mark_agent_parent(call.id)
    _persist_background_receipt(store, call, child)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="background research",
        background=True,
    )
    store.append_agent_notification(
        "parent:1",
        child_session_path=str(child.session_dir),
        description="background research",
        status="completed",
        text="child complete",
    )

    resumed = ConversationStore(tmp_path, session_id="parent")
    AgentLoop(BackgroundBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "completed"
    assert notification.data["text"].startswith("child complete")
    assert not resumed.agent_children()
    assert (
        ConversationStore(store.session_dir / "agents", session_id="1").agent_canceled()
        is None
    )


def test_resume_recovers_killed_task_provenance_from_terminal_lifecycle(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _background_agent_call("recover-receipt")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1", started_at="2026-09-30T00:00:00Z", depth=1,
        agent_type="agent", description="background research",
    )
    child.finish_agent_lifecycle(
        "completed", final_result="done · 1 turns · 0.1s · 0 tool calls · error=false · canceled=false",
        killed_task_ids=["task-a"], killed_task_count=100,
        killed_task_ids_truncated=True,
    )
    child.mark_agent_parent(call.id)
    _persist_background_receipt(store, call, child)
    store.allocate_agent_index()
    store.register_agent_child(
        call, child_session_path=str(child.session_dir),
        description="background research", background=True,
    )

    resumed = ConversationStore(tmp_path, session_id="parent")
    AgentLoop(BackgroundBackend([]), resumed, skill_catalog=SkillCatalog.empty())
    notification = resumed.agent_notifications()[0]
    assert notification.data["killed_task_ids"] == ["task-a"]
    assert notification.data["killed_task_count"] == 100
    assert notification.data["killed_task_ids_truncated"] is True


@pytest.mark.asyncio
async def test_resume_drops_corrupt_lifecycle_killed_task_metadata(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _background_agent_call("recover-corrupt-receipt")
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-09-30T00:00:00Z",
        depth=1,
        agent_type="agent",
        description="background research",
    )
    child.finish_agent_lifecycle(
        "completed",
        final_result=(
            "done · 1 turns · 0.1s · 0 tool calls · error=false · canceled=false"
        ),
        killed_task_ids=["task-a"],
        killed_task_count=1,
        killed_task_ids_truncated=False,
    )
    child.mark_agent_parent(call.id)
    _persist_background_receipt(store, call, child)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="background research",
        background=True,
    )
    lifecycle = json.loads(child.agent_lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["killed_task_ids"] = [""]
    child.agent_lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")

    resumed = ConversationStore(tmp_path, session_id="parent")
    with pytest.warns(RuntimeWarning, match="dropping it"):
        loop = AgentLoop(
            BackgroundBackend([]), resumed, skill_catalog=SkillCatalog.empty()
        )

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "completed"
    assert notification.data["text"].startswith("done")
    assert "killed_task_ids" not in notification.data
    assert "killed_task_count" not in notification.data
    assert "killed_task_ids_truncated" not in notification.data
    await loop.close()
    resumed.close()
    child.close()
    store.close()


def test_resume_cancels_adopted_background_grandchild(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    child_call = _background_agent_call("child")
    child.mark_agent_parent(child_call.id)
    child.finish_agent_parent()
    grandchild_call = _background_agent_call("grandchild")
    grandchild.mark_agent_parent(grandchild_call.id)
    root.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        background=True,
        child_instance_id="root:1:1",
    )

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "canceled"
    assert "session exited" in notification.data["text"]
    assert not resumed.agent_children()
    assert ConversationStore(
        child.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": grandchild_call.id,
        "content": "tool execution canceled",
    }


def test_resume_keeps_same_id_adopted_background_grandchildren_separate(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    children = [
        ConversationStore(root.session_dir / "agents", session_id=str(index))
        for index in (1, 2)
    ]
    grandchild_call = _background_agent_call("same-grandchild")

    for index, child in enumerate(children, start=1):
        grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
        grandchild.mark_agent_parent(grandchild_call.id)
        child.register_agent_child(
            grandchild_call,
            child_session_path=str(grandchild.session_dir),
            description=f"grandchild {index}",
            background=True,
            child_instance_id=f"root:{index}:1",
        )
        adopt_agent_children(child, root)

    assert set(root.agent_children()) == {"root:1:1", "root:2:1"}
    root.finish_agent_child("root:1:1")
    assert set(root.agent_children()) == {"root:2:1"}

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notifications = resumed.agent_notifications(pending_only=False)
    assert [entry.data["child_instance_id"] for entry in notifications] == ["root:2:1"]
    recovered = ConversationStore(children[1].session_dir / "agents", session_id="1")
    assert recovered.agent_canceled() == {
        "tool_call_id": "same-grandchild",
        "content": "tool execution canceled",
    }


@pytest.mark.asyncio
async def test_background_persistence_failure_does_not_block_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = _background_agent_call()
    backend = BackgroundBackend([call])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    def fail_notification(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("persistence failed")

    monkeypatch.setattr(store, "append_agent_notification", fail_notification)
    backend.release_child.set()
    await asyncio.wait_for(loop.close(), timeout=1)

    assert not loop.background_children_running


@pytest.mark.asyncio
async def test_parallel_agent_calls_overlap_and_keep_child_results(
    tmp_path: Path,
) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelChildrenBackend(calls)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.wait_for(backend.children_started.wait(), timeout=30)
    backend.release_children["inspect 2"].set()
    await asyncio.wait_for(backend.children_completed["inspect 2"].wait(), timeout=30)
    backend.release_children["inspect 1"].set()
    await asyncio.wait_for(task, timeout=30)

    results = {
        message.tool_result.tool_call_id: message.tool_result.content.split(" · ", 1)[0]
        for message in store.messages()
        if message.tool_result is not None
    }
    assert results == {
        "agent-1": "response for inspect 1",
        "agent-2": "response for inspect 2",
    }
    assert sorted(path.name for path in (store.session_dir / "agents").iterdir()) == [
        "1",
        "2",
    ]


@pytest.mark.asyncio
async def test_parallel_nested_lifecycle_events_survive_large_batch(
    tmp_path: Path,
) -> None:
    backend = ParallelNestedReadBackend(44)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await asyncio.wait_for(_collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=30)

    read_lifecycle = [
        event
        for event in events
        if event.tool_call is not None
        and event.tool_call.name == "read"
        and event.type
        in {
            StreamEventType.TOOL_EXECUTION_START,
            StreamEventType.TOOL_EXECUTION_END,
        }
    ]
    assert len(read_lifecycle) == 88
    errors = [
        event.error
        for event in events
        if event.type is StreamEventType.ERROR and event.error is not None
    ]
    assert [error.code for error in errors] == ["max_turns"]
    await loop.close()


@pytest.mark.asyncio
async def test_duplicate_parallel_agent_ids_fail_before_child_dispatch(
    tmp_path: Path,
) -> None:
    calls = [_agent_call("same-id"), _agent_call("same-id")]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=calls),
            ScriptedTurn([TextContent("child one")]),
            ScriptedTurn([TextContent("child two")]),
        ]
    )
    store = ConversationStore(tmp_path)

    with pytest.raises(ValueError, match="duplicate tool call id"):
        await _collect(
            AgentLoop(
                backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
            ).run_turn("start", origin=MessageOrigin.USER)
        )

    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_parent_abort_cancels_all_parallel_children(tmp_path: Path) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelChildrenBackend(calls)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.wait_for(backend.children_started.wait(), timeout=30)
    loop.abort()
    await asyncio.wait_for(task, timeout=30)

    results = [
        message.tool_result for message in store.messages() if message.tool_result
    ]
    assert len(results) == 2
    assert all(
        result is not None and result.content.startswith("tool execution canceled")
        for result in results
    )
    for index, call in enumerate(calls, start=1):
        child_store = ConversationStore(
            store.session_dir / "agents", session_id=str(index)
        )
        assert child_store.agent_canceled() == {
            "tool_call_id": call.id,
            "content": "tool execution canceled",
        }


@pytest.mark.asyncio
async def test_parallel_delegated_approvals_resolve_by_child_instance(
    tmp_path: Path,
) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelApprovalBackend(calls)
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=policy,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    for _ in range(100):
        if len(policy.pending_requests()) == 2:
            break
        await asyncio.sleep(0.01)
    pending = policy.pending_requests()
    assert len(pending) == 2
    assert {request.child_instance_id for request in pending} == {
        f"{store.session_id}:1",
        f"{store.session_id}:2",
    }
    allow, deny = pending
    assert policy.approve(allow.key)
    assert policy.deny(deny.key)
    await asyncio.wait_for(task, timeout=1)
    assert policy.pending_requests() == []


@pytest.mark.asyncio
async def test_agent_returns_child_text_and_persists_child_session(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("done")
    child_dir = store.session_dir / "agents" / "1"
    assert (child_dir / "conversation.jsonl").exists()
    assert [
        message.role
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
    ] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert "agent" in {schema["name"] for schema in backend.calls[1][1]}
    assert {schema["name"] for schema in backend.calls[1][1]} == {
        schema["name"]
        for schema in backend.calls[0][1]
        if schema["name"] != "request_attention"
    }


@pytest.mark.asyncio
async def test_provider_tool_order_is_canonical_across_registry_insertion_order(
    tmp_path: Path,
) -> None:
    def make_registry(order: Sequence[str]) -> ToolRegistry:
        registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
        for name in order:
            registry.register(
                name,
                lambda arguments: arguments,
                description=f"{name} tool",
            )
        return registry

    first_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    second_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    first_loop = AgentLoop(
        first_backend,
        ConversationStore(tmp_path / "first"),
        registry=make_registry(["zulu", "alpha"]),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    second_loop = AgentLoop(
        second_backend,
        ConversationStore(tmp_path / "second"),
        registry=make_registry(["alpha", "zulu"]),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(first_loop.run_turn("start", origin=MessageOrigin.USER))
    await _collect(second_loop.run_turn("start", origin=MessageOrigin.USER))

    first_names = [schema["name"] for schema in first_backend.calls[0][1]]
    assert first_names == sorted(first_names)
    assert {"alpha", "zulu"} <= set(first_names)
    assert first_backend.calls[0][1] == second_backend.calls[0][1]
    assert first_backend.request_bytes[0] == second_backend.request_bytes[0]


@pytest.mark.parametrize("name", [None, "", 42, []])
def test_provider_tool_schema_malformed_names_are_rejected(name: object) -> None:
    schema = {} if name is None else {"name": name}

    with pytest.raises(
        ValueError, match="provider-visible tool schema name must be a non-empty string"
    ):
        canonical_tool_schemas([schema])


def test_provider_tool_schema_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate provider-visible tool schema name"):
        canonical_tool_schemas(
            [
                {"name": "same", "description": "first"},
                {"name": "same", "description": "second"},
            ]
        )


def test_plan_mode_filters_before_provider_schema_validation(tmp_path: Path) -> None:
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        tool_schemas=[{"name": "read"}, {}],
        skill_catalog=SkillCatalog.empty(),
    )
    loop.set_plan_mode(True)

    assert [schema["name"] for schema in loop._active_tool_schemas()] == ["read"]


def test_agent_schema_uses_preset_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = replace(
        AGENT_PRESETS["explore"],
        name="custom",  # type: ignore[arg-type]
    )
    monkeypatch.setitem(AGENT_PRESETS, "explore", custom)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(
        FakeBackend([]), store, registry=registry, skill_catalog=SkillCatalog.empty()
    )

    agent_schema = next(
        schema for schema in registry.schemas if schema["name"] == "agent"
    )
    agent_type_schema = agent_schema["parameters"]["properties"]["agent_type"]
    assert agent_type_schema["enum"] == [
        preset.name for preset in AGENT_PRESETS.values()
    ]
    expected_description = (
        "Choose one of: "
        + "; ".join(
            f"{preset.name}: {preset.selection_guidance}"
            for preset in AGENT_PRESETS.values()
        )
        + "."
    )
    assert agent_type_schema["description"] == expected_description


@pytest.mark.asyncio
async def test_general_agent_markers_keep_legacy_state_bytes(tmp_path: Path) -> None:
    omitted = ConversationStore(tmp_path / "omitted", cwd=tmp_path)
    explicit = ConversationStore(tmp_path / "explicit", cwd=tmp_path)
    omitted_backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(call_id="omitted")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    explicit_backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    _agent_call(call_id="explicit", agent_type=GENERAL_PRESET.name)
                ]
            ),
            ScriptedTurn([TextContent("done")]),
        ]
    )

    await _collect(
        AgentLoop(
            omitted_backend, omitted, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )
    await _collect(
        AgentLoop(
            explicit_backend, explicit, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    omitted_child_state = omitted.session_dir / "agents" / "1" / "session_state.json"
    explicit_child_state = explicit.session_dir / "agents" / "1" / "session_state.json"
    assert omitted_child_state.read_bytes() == explicit_child_state.read_bytes()
    assert json.loads(omitted_child_state.read_text()) == {"bash_cwd": str(tmp_path)}
    assert omitted.state_path.read_bytes() == explicit.state_path.read_bytes()


@pytest.mark.asyncio
async def test_explore_child_has_read_only_tools_and_rejects_exec(
    tmp_path: Path,
) -> None:
    child_exec = ToolCall("child-exec", "bash", {"command": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn(tool_calls=[child_exec]),
            ScriptedTurn([TextContent("explore complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    child_schemas = {schema["name"] for schema in backend.calls[1][1]}
    assert child_schemas == {
        "agent",
        "agent_output",
        "agent_status",
        "fetch",
        "read",
        "skill",
        "websearch",
    }
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    child_result = next(
        message.tool_result for message in child_messages if message.tool_result
    )
    assert child_result.is_error
    assert child_result.content == "unknown tool: bash"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_type",
    ["explore", "plan"],
)
async def test_restricted_child_cannot_use_mounted_mcp_write_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_type: str,
) -> None:
    mount_calls = 0

    async def mount_write_tool(
        registry: ToolRegistry, config=None, *, notice_sink=None, home=None
    ) -> MCPMount:
        del config, notice_sink, home
        nonlocal mount_calls
        mount_calls += 1

        async def remote_write(
            arguments: dict[str, object], abort_signal
        ) -> dict[str, object]:
            del arguments, abort_signal
            return {
                "content": [{"type": "text", "text": "write executed"}],
                "isError": False,
                "structuredContent": None,
            }

        registry.register(
            "remote:write",
            remote_write,
            parameters={"type": "object"},
            requires_approval=False,
        )
        return MCPMount(())

    monkeypatch.setattr(
        "zeta.runtime.loop.mcp_session.mount_mcp_servers", mount_write_tool
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type=agent_type)]),
            ScriptedTurn(tool_calls=[ToolCall("remote-write", "remote:write", {})]),
            ScriptedTurn([TextContent("restricted complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    assert mount_calls == 1
    child_result = next(
        message.tool_result
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
        if message.tool_result
    )
    assert child_result.is_error
    assert child_result.content == "unknown tool: remote:write"


@pytest.mark.asyncio
async def test_plan_child_includes_todo_and_only_read_only_tools(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="plan")]),
            ScriptedTurn([TextContent("plan complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    assert {schema["name"] for schema in backend.calls[1][1]} == {
        "agent",
        "agent_output",
        "agent_status",
        "fetch",
        "read",
        "skill",
        "todo",
        "websearch",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", ["explore", "plan"])
async def test_typed_child_exceeds_former_turn_cap_and_completes(
    tmp_path: Path, agent_type: str
) -> None:
    """Delegated presets no longer terminate at their former turn caps."""
    child_call = ToolCall("child-read", "read", {"path": "missing"})
    turns = 26
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call(agent_type=agent_type)])]
        + [
            ScriptedTurn([TextContent(f"step-{turn}")], tool_calls=[child_call])
            for turn in range(1, turns + 1)
        ]
        + [ScriptedTurn([TextContent("child complete")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None and not result.is_error
    assert result.content.startswith("child complete")
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_lifecycle()["turns_used"] == turns + 1


@pytest.mark.asyncio
async def test_unknown_agent_type_returns_loud_error(tmp_path: Path) -> None:
    call = _agent_call()
    call = ToolCall(
        call.id,
        call.name,
        {**call.arguments, "agent_type": "unknown"},
    )
    backend = FakeBackend([])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    result = await loop._run_agent_tool(
        call,
        call.arguments,
        AbortGenerationRegistry().new_generation(),
        None,
    )

    assert result["isError"] is True
    assert "unknown agent_type" in result["content"][0]["text"]
    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_typed_preamble_composes_with_child_system_prompt(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            system_prompt="existing child instructions",
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    child_system = backend.calls[1][0][0]
    system_text = " ".join(
        block.text for block in child_system.content if isinstance(block, TextContent)
    )
    assert "You are an explore sub-agent." in system_text
    assert "existing child instructions" in system_text


@pytest.mark.asyncio
async def test_child_registry_preserves_parent_pre_execution_hook(
    tmp_path: Path,
) -> None:
    child_call = ToolCall("child-bash", "bash", {"cmd": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("hook denied")]),
        ]
    )
    observed: list[str] = []

    def deny_bash(name: str, arguments: dict[str, object]) -> str | None:
        del arguments
        observed.append(name)
        return "denied by test hook" if name == "bash" else None

    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        store.cwd, pre_execute_hook=deny_bash, skill_catalog=SkillCatalog.empty()
    )
    await _collect(
        AgentLoop(
            backend,
            store,
            registry=registry,
            max_turns=1,
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    assert observed[-1] == "bash"
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    denied = next(
        message.tool_result for message in child_messages if message.tool_result
    )
    assert denied.is_error
    assert "denied by test hook" in denied.content


@pytest.mark.asyncio
async def test_child_registry_isolates_todo_store_and_other_session_tools(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "sessions"
    parent_store = ConversationStore(sessions, session_id="parent", cwd=tmp_path)
    child_store = ConversationStore(sessions, session_id="child", cwd=tmp_path)
    parent_store.set_todo_items([{"content": "parent work", "status": "pending"}])
    (tmp_path / "nested").mkdir()
    registry = ToolRegistry(
        parent_store.cwd, session_store=parent_store, skill_catalog=SkillCatalog.empty()
    )
    child_registry = registry.clone_for_session(child_store)
    child_loop = AgentLoop(
        FakeBackend([]),
        child_store,
        registry=child_registry,
        skill_catalog=SkillCatalog.empty(),
    )
    widget = TodoWidget(parent_store)

    await child_loop.tool_registry.execute(
        ToolCall(
            "child-todo",
            "todo",
            {"items": [{"content": "child work", "status": "pending"}]},
        )
    )
    await child_loop.tool_registry.execute(
        ToolCall("child-bash", "bash", {"cmd": "cd nested && pwd"})
    )

    assert parent_store.todo_items() == [
        {"content": "parent work", "status": "pending"}
    ]
    assert child_store.todo_items() == [{"content": "child work", "status": "pending"}]
    assert widget.visible
    assert parent_store.bash_cwd == str(tmp_path)
    assert child_store.bash_cwd == str(tmp_path / "nested")

    await child_loop.close()
    await registry.close()


@pytest.mark.asyncio
async def test_plan_child_writes_todo_items_to_its_own_store(tmp_path: Path) -> None:
    plan_items = [{"content": "review findings", "status": "pending"}]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="plan")]),
            ScriptedTurn(
                tool_calls=[ToolCall("plan-todo", "todo", {"items": plan_items})]
            ),
            ScriptedTurn([TextContent("plan complete")]),
        ]
    )
    store = ConversationStore(tmp_path)
    store.set_todo_items([{"content": "parent work", "status": "pending"}])
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert store.todo_items() == [{"content": "parent work", "status": "pending"}]
    assert child_store.todo_items() == plan_items
    await loop.close()


@pytest.mark.asyncio
async def test_child_todos_do_not_make_an_empty_parent_widget_visible(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    child_store = ConversationStore(parent_store.session_dir / "agents", session_id="1")
    registry = ToolRegistry(
        parent_store.cwd,
        session_store=parent_store,
        skill_catalog=SkillCatalog.empty(),
    )
    child_registry = registry.clone_for_session(child_store)
    child_loop = AgentLoop(
        FakeBackend([]),
        child_store,
        registry=child_registry,
        skill_catalog=SkillCatalog.empty(),
    )

    await child_loop.tool_registry.execute(
        ToolCall(
            "child-todo",
            "todo",
            {"items": [{"content": "child work", "status": "pending"}]},
        )
    )

    assert not TodoWidget(parent_store).visible
    assert child_store.todo_items() == [{"content": "child work", "status": "pending"}]
    await child_loop.close()
    await registry.close()


@pytest.mark.asyncio
async def test_delegated_approvals_use_child_instance_keys(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    call_a = ToolCall("same-request", "bash", {"cmd": "a"})
    call_b = ToolCall("same-request", "bash", {"cmd": "b"})
    child_a = ConversationStore(tmp_path / "children", session_id="a")
    child_b = ConversationStore(tmp_path / "children", session_id="b")
    policy_a = ChildApprovalPolicy(policy, child_a, "a", "child-a")
    policy_b = ChildApprovalPolicy(policy, child_b, "b", "child-b")

    child_a_signal = AbortGenerationRegistry().new_generation()
    child_b_signal = AbortGenerationRegistry().new_generation()
    task_a = asyncio.create_task(policy_a.authorize(call_a, child_a_signal, capability=_approval_capability(call_a.name, call_a.arguments)))
    task_b = asyncio.create_task(policy_b.authorize(call_b, child_b_signal, capability=_approval_capability(call_b.name, call_b.arguments)))
    while len(policy.pending_requests()) < 2:
        await asyncio.sleep(0)

    pending = {request.key for request in policy.pending_requests()}
    assert pending == {("child-a", "same-request"), ("child-b", "same-request")}
    assert policy.approve(("child-a", "same-request"))
    assert policy.deny(("child-b", "same-request"))
    assert await task_a == ApprovalDecision.ALLOW
    assert await task_b == ApprovalDecision.DENY
    child_a_signal.abort()
    child_b_signal.abort()


def test_parent_relative_allow_rule_does_not_authorize_child_other_repo(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    child_cwd = tmp_path / "child"
    parent_cwd.mkdir()
    child_cwd.mkdir()
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    policy = ApprovalPolicy(store=parent_store, always_allow={"write(src/**)"})
    child_policy = ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=parent_cwd,
        child_cwd=child_cwd,
    )

    assert child_policy.decide(_approval_capability("write", {"path": "src/file.py"})) is ApprovalDecision.ASK


@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
def test_parent_shell_allow_does_not_authorize_child_other_cwd(
    tmp_path: Path, tool_name: str
) -> None:
    parent_cwd = tmp_path / "parent"
    child_cwd = tmp_path / "child"
    parent_cwd.mkdir()
    child_cwd.mkdir()
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    policy = ApprovalPolicy(
        store=parent_store, always_allow={f"{tool_name}(git clean*)"}
    )
    child_policy = ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=parent_cwd,
        child_cwd=child_cwd,
    )

    assert child_policy.decide(_approval_capability(tool_name, {"command": "git clean -fd"})) is ApprovalDecision.ASK


def test_parent_shell_allow_applies_in_same_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "repo"
    cwd.mkdir()
    parent_store = ConversationStore(tmp_path / "sessions", cwd=cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=cwd)
    policy = ApprovalPolicy(store=parent_store, always_allow={"bash(git status*)"})
    child_policy = ChildApprovalPolicy(
        policy, child_store, "child", "child-1", parent_cwd=cwd, child_cwd=cwd
    )

    assert child_policy.decide(_approval_capability("bash", {"command": "git status --short"})) is ApprovalDecision.ALLOW

    alias = tmp_path / "repo-alias"
    alias.symlink_to(cwd, target_is_directory=True)
    aliased_child = ChildApprovalPolicy(
        policy,
        ConversationStore(tmp_path / "aliased-children", cwd=alias),
        "child",
        "child-2",
        parent_cwd=cwd,
        child_cwd=alias,
    )
    assert aliased_child.decide(_approval_capability("bash", {"command": "git status --short"})) is ApprovalDecision.ALLOW


def test_child_shell_explicit_cwd_and_persisted_cd_respected(tmp_path: Path) -> None:
    parent_cwd = tmp_path / "repo"
    other_cwd = parent_cwd / "other"
    other_cwd.mkdir(parents=True)
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=parent_cwd)
    policy = ApprovalPolicy(store=parent_store, always_allow={"bash(git status*)"})
    child_policy = ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=parent_cwd,
        child_cwd=parent_cwd,
    )

    assert child_policy.decide(_approval_capability("bash", {"command": "git status", "cwd": "other"})) is ApprovalDecision.ASK
    child_store.set_bash_cwd(str(other_cwd))
    assert child_policy.decide(_approval_capability("bash", {"command": "git status"})) is ApprovalDecision.ASK


def test_parent_allow_rule_still_applies_when_child_path_resolves_inside_it(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    child_cwd = parent_cwd / "src"
    child_cwd.mkdir(parents=True)
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    policy = ApprovalPolicy(store=parent_store, always_allow={"write(src/**)"})
    child_policy = ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=parent_cwd,
        child_cwd=child_cwd,
    )

    assert child_policy.decide(_approval_capability("write", {"path": "nested/file.py"})) is ApprovalDecision.ALLOW


def test_symlink_out_of_allowed_tree_not_auto_allowed(tmp_path: Path) -> None:
    parent_cwd = tmp_path / "parent"
    outside = tmp_path / "outside"
    (parent_cwd / "src").mkdir(parents=True)
    outside.mkdir()
    (parent_cwd / "src" / "link").symlink_to(outside, target_is_directory=True)
    policy = _child_path_policy(tmp_path, parent_cwd)

    assert policy.decide(_approval_capability("write", {"path": "src/link/file.py"})) is ApprovalDecision.ASK


def test_symlink_into_allowed_tree_behavior(tmp_path: Path) -> None:
    """A child alias into the canonical allowed tree remains auto-allowed."""

    parent_cwd = tmp_path / "parent"
    child_cwd = tmp_path / "child"
    (parent_cwd / "src").mkdir(parents=True)
    child_cwd.mkdir()
    (child_cwd / "alias").symlink_to(parent_cwd / "src", target_is_directory=True)
    policy = _child_path_policy(tmp_path, parent_cwd, child_cwd=child_cwd)

    assert policy.decide(_approval_capability("write", {"path": "alias/file.py"})) is ApprovalDecision.ALLOW


def test_nonexistent_target_under_symlinked_ancestor_canonicalized(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    outside = tmp_path / "outside"
    (parent_cwd / "src").mkdir(parents=True)
    outside.mkdir()
    (parent_cwd / "src" / "link").symlink_to(outside, target_is_directory=True)
    policy = _child_path_policy(tmp_path, parent_cwd)

    assert policy.decide(_approval_capability("write", {"path": "src/link/missing/directory/file.py"})) is ApprovalDecision.ASK


def _child_path_policy(
    tmp_path: Path, parent_cwd: Path, *, child_cwd: Path | None = None
) -> ChildApprovalPolicy:
    child_cwd = child_cwd or parent_cwd
    parent_store = ConversationStore(tmp_path / "path-sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "path-children", cwd=child_cwd)
    policy = ApprovalPolicy(store=parent_store, always_allow={"write(src/**)"})
    return ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-path",
        parent_cwd=parent_cwd,
        child_cwd=child_cwd,
    )


def test_child_approval_card_shows_effective_cwd_and_resolved_path(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    child_cwd = tmp_path / "child"
    parent_cwd.mkdir()
    child_cwd.mkdir()
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=parent_cwd,
        child_cwd=child_cwd,
    )
    call = ToolCall(
        "write-request", "write", {"path": "src/file.py", "content": "x"}
    )
    request = child_policy.prepare(
        call, capability=_approval_capability(call.name, call.arguments)
    )

    assert request is not None
    assert request.effective_cwd is None
    assert request.resolved_path == str(child_cwd / "src/file.py")
    output = StringIO()
    Console(file=output, force_terminal=False, width=200).print(
        render_approval_card(
            request.tool_call.name,
            request.tool_call.arguments,
            execution_display=(request.effective_cwd, request.resolved_path),
        )
    )
    card = output.getvalue()
    assert "cwd=" not in card
    assert f"resolved_path={child_cwd / 'src/file.py'}" in card


def test_child_shell_approval_shows_cwd(tmp_path: Path) -> None:
    parent_cwd = tmp_path / "parent"
    child_cwd = tmp_path / "child"
    shell_cwd = child_cwd / "nested"
    parent_cwd.mkdir()
    shell_cwd.mkdir(parents=True)
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=parent_cwd,
        child_cwd=child_cwd,
    )
    calls = (
        ToolCall("bash-request", "bash", {"command": "pwd", "cwd": "nested"}),
        ToolCall(
            "background-request",
            "run_background",
            {"command": "pwd", "cwd": "nested"},
        ),
    )
    for call in calls:
        request = child_policy.prepare(
            call, capability=_approval_capability(call.name, call.arguments)
        )
        assert request is not None
        assert request.effective_cwd == str(shell_cwd)
        assert request.resolved_path is None
        output = StringIO()
        Console(file=output, force_terminal=False, width=200).print(
            render_approval_card(
                request.tool_call.name,
                request.tool_call.arguments,
                execution_display=(request.effective_cwd, request.resolved_path),
            )
        )
        assert f"cwd={shell_cwd}" in output.getvalue()


@pytest.mark.asyncio
async def test_child_approval_cleanup_removes_pending_requests(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")
    signal = AbortGenerationRegistry().new_generation()
    call = ToolCall("pending", "bash", {"cmd": "wait"})
    task = asyncio.create_task(
        child_policy.authorize(
            call,
            signal,
            capability=_approval_capability(call.name, call.arguments),
        )
    )
    while not policy.pending_requests():
        await asyncio.sleep(0)

    child_policy.cleanup()
    assert await task == ApprovalDecision.DENY
    assert policy.pending_requests() == []
    signal.abort()


@pytest.mark.asyncio
async def test_child_policy_inherits_scoped_rules_and_delegates_prompts(
    tmp_path: Path,
) -> None:
    """ZETA-86: scoped rules apply to children; unmatched calls still surface."""

    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(
        store=parent_store,
        always_allow={"bash(git status*)"},
        always_deny={"bash(rm *)"},
    )
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")

    assert (
        child_policy.decide(_approval_capability("bash", {"command": "git status"})) is ApprovalDecision.ALLOW
    )
    assert child_policy.decide(_approval_capability("bash", {"command": "rm -rf /"})) is ApprovalDecision.DENY
    prepared_call = ToolCall("ok", "bash", {"command": "git status"})
    assert child_policy.prepare(
        prepared_call,
        capability=_approval_capability(
            prepared_call.name, prepared_call.arguments
        ),
    ) is None
    assert child_policy.notices == ()

    signal = AbortGenerationRegistry().new_generation()
    call = ToolCall("pending", "bash", {"command": "git push"})
    task = asyncio.create_task(child_policy.authorize(call, signal, capability=_approval_capability(call.name, call.arguments)))
    while not policy.pending_requests():
        await asyncio.sleep(0)

    assert [request.key for request in policy.pending_requests()] == [
        ("child-1", "pending")
    ]
    assert policy.approve(("child-1", "pending"))
    assert await task is ApprovalDecision.ALLOW
    signal.abort()


@pytest.mark.asyncio
async def test_child_loop_inherits_argument_scoped_approval_rules(
    tmp_path: Path,
) -> None:
    """ZETA-86: the agent tool's child loop is gated by the parent's scoped rules."""

    allowed_call = ToolCall("child-echo", "bash", {"command": "echo scoped-ok"})
    denied_call = ToolCall("child-rm", "bash", {"command": "rm -rf nothing-here"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[allowed_call, denied_call]),
            ScriptedTurn([TextContent("child done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(
        store=store,
        always_allow={"bash(echo *)"},
        default=ApprovalDecision.DENY,
    )

    await _collect(
        AgentLoop(
            backend,
            store,
            approval_policy=policy,
            max_turns=1,
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    results = {
        message.tool_result.tool_call_id: message.tool_result
        for message in child_messages
        if message.tool_result
    }
    assert not results["child-echo"].is_error
    assert "scoped-ok" in results["child-echo"].content
    assert results["child-rm"].is_error
    assert results["child-rm"].content == "tool execution denied"


@pytest.mark.asyncio
async def test_child_abort_closes_all_loop_tasks(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def wait_forever(arguments: dict[str, object], abort_signal) -> str:
        del arguments
        started.set()
        await abort_signal.wait()
        return "stopped"

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[ToolCall("child-wait", "wait", {})]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(store.cwd, skill_catalog=SkillCatalog.empty())
    registry.register("wait", wait_forever)
    loop = AgentLoop(
        backend,
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await started.wait()
    loop.abort()

    await task
    await loop.close()

    assert not [
        child_task
        for child_task in asyncio.all_tasks()
        if child_task not in baseline and not child_task.done()
    ]


@pytest.mark.asyncio
async def test_child_agent_call_allows_one_grandchild(tmp_path: Path) -> None:
    nested = _agent_call("nested")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("grandchild complete")]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("child complete")
    child_result = next(
        message.tool_result
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
        if message.tool_result
    )
    assert child_result.content.startswith("grandchild complete")
    grandchild_schemas = {schema["name"] for schema in backend.calls[2][1]}
    assert "agent" not in grandchild_schemas


@pytest.mark.asyncio
async def test_nested_typed_child_only_tightens_tools(tmp_path: Path) -> None:
    nested = _agent_call("grandchild")
    child = _agent_call("child", "explore")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("grandchild complete")]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    child_tools = {schema["name"] for schema in backend.calls[1][1]}
    grandchild_tools = {schema["name"] for schema in backend.calls[2][1]}
    assert child_tools == {
        "agent",
        "agent_output",
        "agent_status",
        "fetch",
        "read",
        "skill",
        "websearch",
    }
    assert grandchild_tools == {
        "agent_status",
        "agent_output",
        "fetch",
        "read",
        "skill",
        "websearch",
    }


@pytest.mark.asyncio
async def test_nested_and_parallel_children_do_not_share_a_terminating_budget(
    tmp_path: Path,
) -> None:
    """Nested and sibling delegation completes without a tree turn budget."""
    grandchildren = [_agent_call("grandchild-1"), _agent_call("grandchild-2")]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("child")]),
            ScriptedTurn(tool_calls=grandchildren),
            ScriptedTurn([TextContent("one")]),
            ScriptedTurn([TextContent("two")]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None and not result.is_error
    assert result.content.startswith("child complete")
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    child_results = [
        message.tool_result
        for message in child_store.messages()
        if message.tool_result is not None
    ]
    assert len(child_results) == 2
    assert all(not item.is_error for item in child_results)


def test_legacy_tree_budget_fields_are_tolerated(tmp_path: Path) -> None:
    """Old lifecycle markers remain readable after the budget API is removed."""
    store = ConversationStore(tmp_path)
    store.start_agent_lifecycle(
        handle="parent:child",
        started_at="2026-09-08T00:00:00+00:00",
        depth=1,
        agent_type="general",
        description="legacy child",
    )
    import json

    marker = store.session_dir / "agent_lifecycle.json"
    value = json.loads(marker.read_text())
    value.update({"tree_budget": 25, "max_turns": 25, "turns_used": 7})
    marker.write_text(json.dumps(value))
    reopened = ConversationStore(store.root_dir, session_id=store.session_id)
    lifecycle = reopened.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["state"] == "running"
    assert lifecycle["turns_used"] == 7


@pytest.mark.asyncio
async def test_background_grandchild_keeps_its_own_notification(tmp_path: Path) -> None:
    nested = _agent_call("grandchild")
    nested.arguments["background"] = True
    child = _background_agent_call("child")
    child.arguments["prompt"] = "inspect the task"
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("child complete")]),
            ScriptedTurn([TextContent("child finished")]),
        ]
    )
    store = ConversationStore(tmp_path)
    root_woke = asyncio.Event()
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    loop.set_background_wake_callback(root_woke.set)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await _wait_for_notification(store, "completed")
    await asyncio.wait_for(root_woke.wait(), timeout=1)

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    nested_result = next(
        message.tool_result for message in child_store.messages() if message.tool_result
    )
    assert nested_result.structured_content is not None
    assert nested_result.structured_content["status"] == "running"
    assert [
        entry.data["status"]
        for entry in child_store.agent_notifications(pending_only=False)
    ] == ["completed"]
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert grandchild_store.agent_canceled() is None
    await loop.close()


@pytest.mark.asyncio
async def test_abort_propagates_through_two_nested_levels(tmp_path: Path) -> None:
    backend = NestedBlockingBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))

    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    loop.abort()
    await asyncio.wait_for(task, timeout=1)

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert child_store.agent_canceled() == {
        "tool_call_id": "child",
        "content": "tool execution canceled",
    }
    assert grandchild_store.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }
    await loop.close()


def test_resume_cancels_nested_tree_markers(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    child_call = _agent_call("child")
    grandchild_call = _agent_call("grandchild")
    child.mark_agent_parent(child_call.id)
    grandchild.mark_agent_parent(grandchild_call.id)
    child.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        child_instance_id="root:1:1",
    )
    root.register_agent_child(
        child_call,
        child_session_path=str(child.session_dir),
        description="child",
    )

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    assert not resumed.agent_children()
    assert ConversationStore(
        root.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": "child",
        "content": "tool execution canceled",
    }
    assert ConversationStore(
        child.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }


def test_resume_cancels_nested_background_tree_markers(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    child_call = _background_agent_call("child")
    grandchild_call = _background_agent_call("grandchild")
    child_call.arguments["description"] = "child"
    grandchild_call.arguments["description"] = "grandchild"
    child.mark_agent_parent(child_call.id)
    grandchild.mark_agent_parent(grandchild_call.id)
    _persist_background_receipt(child, grandchild_call, grandchild)
    child.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        background=True,
        child_instance_id="root:1:1",
    )
    _persist_background_receipt(root, child_call, child)
    root.register_agent_child(
        child_call,
        child_session_path=str(child.session_dir),
        description="child",
        background=True,
        child_instance_id="root:1",
    )

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    assert resumed.agent_notifications()[0].data["status"] == "canceled"
    resumed_child = ConversationStore(resumed.session_dir / "agents", session_id="1")
    resumed_grandchild = ConversationStore(
        resumed_child.session_dir / "agents", session_id="1"
    )
    assert resumed_child.agent_notifications()[0].data["status"] == "canceled"
    assert not resumed.agent_children()
    assert not resumed_child.agent_children()
    assert resumed_grandchild.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }


@pytest.mark.asyncio
async def test_child_approval_uses_parent_policy(tmp_path: Path) -> None:
    child_call = ToolCall("child-bash", "bash", {"cmd": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("approval handled")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    events = []
    async for event in AgentLoop(
        backend,
        store,
        approval_policy=policy,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    ).run_turn("start", origin=MessageOrigin.USER):
        events.append(event)
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            assert [request.tool_call.id for request in policy.pending_requests()] == [
                child_call.id
            ]
            assert policy.pending_requests()[0].label == "task research: bash"
            assert all(
                message.tool_result is None
                or message.tool_result.tool_call_id != child_call.id
                for message in store.messages()
            )
            policy.deny(child_call.id)

    assert any(event.type is StreamEventType.TOOL_APPROVAL_END for event in events)
    assert not policy.pending_requests()
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL_RESULT,
    ]
    assert child_call.id not in {
        block.tool_call.id
        for message in store.messages()
        for block in message.content
        if isinstance(block, ToolUseContent)
    }
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    denied = next(
        message.tool_result for message in child_messages if message.tool_result
    )
    assert denied.is_error
    assert denied.content == "tool execution denied"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("approve", "nested"),
        ("deny", "tool execution denied"),
        ("abort", "tool execution canceled"),
    ],
)
async def test_grandchild_approval_composes_with_parent_policy(
    tmp_path: Path, action: str, expected: str
) -> None:
    grandchild_bash = ToolCall("grandchild-bash", "bash", {"cmd": "echo nested"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("child")]),
            ScriptedTurn(tool_calls=[_agent_call("grandchild")]),
            ScriptedTurn(tool_calls=[grandchild_bash]),
            ScriptedTurn([TextContent("grandchild finished")]),
            ScriptedTurn([TextContent("child finished")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=policy,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    for _ in range(100):
        pending = policy.pending_requests()
        if pending:
            break
        await asyncio.sleep(0.01)
    assert len(pending) == 1
    request = pending[0]
    assert request.child_instance_id == f"{store.session_id}:1:1"
    assert getattr(policy, action)(request.key)
    await asyncio.wait_for(task, timeout=5)

    grandchild_store = ConversationStore(
        store.session_dir / "agents" / "1" / "agents", session_id="1"
    )
    bash_result = next(
        message.tool_result
        for message in grandchild_store.messages()
        if message.tool_result is not None
    )
    assert expected in bash_result.content
    await loop.close()


@pytest.mark.asyncio
async def test_agent_result_metadata_survives_parent_replay(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    result = next(
        message.tool_result for message in replayed.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["turns_used"] == 1
    assert result.structured_content["child_session_path"] == str(
        store.session_dir / "agents" / "1"
    )


@pytest.mark.asyncio
async def test_agent_normal_completion_reports_all_child_turns(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(
                [TextContent("first")],
                tool_calls=[ToolCall("child-read", "read", {"path": "missing"})],
            ),
            ScriptedTurn([TextContent("second")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["turns_used"] == 2


@pytest.mark.asyncio
async def test_typed_child_type_survives_completion_and_reopen(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_type() == "explore"
    assert json.loads(child.state_path.read_text())["agent_parent"] == {
        "tool_call_id": "agent-1",
        "agent_type": "explore",
        "status": "finished",
    }
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"


@pytest.mark.asyncio
async def test_empty_child_final_message_returns_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn(), ScriptedTurn()]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert "empty final assistant message" in result.content


@pytest.mark.asyncio
async def test_child_answer_with_cancellation_prefix_is_success(
    tmp_path: Path,
) -> None:
    answer = "tool execution canceled, but this is the answer"
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent(answer)]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None
    assert result.is_error is False
    assert result.content.startswith(answer)
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_canceled() is None


@pytest.mark.asyncio
async def test_failed_agent_receipt_stats_match_is_error(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn()])
    store = ConversationStore(tmp_path)

    events = await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None
    assert result.is_error is True
    assert "error=true" in result.content
    assert "canceled=false" in result.content
    rendered = render_event(
        next(
            event
            for event in events
            if event.type is StreamEventType.TOOL_EXECUTION_END
        )
    )
    assert rendered is not None
    assert rendered.plain.count("error=true") == 1
    assert rendered.plain.count("canceled=false") == 1


@pytest.mark.asyncio
async def test_failed_background_receipt_matches_error_flag(
    tmp_path: Path,
) -> None:
    call = _background_agent_call()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call]), ScriptedTurn()]),
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await loop._background_owner.wait()
    notification = store.agent_notifications(pending_only=False)[0]
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )

    assert notification.data["status"] == "error"
    assert notification.data["text"].count("error=true") == 1
    assert notification.data["text"].count("canceled=false") == 1
    assert terminal.tool_result is not None
    assert terminal.tool_result.is_error is True
    assert terminal.tool_result.is_canceled is False
    rendered = render_event(terminal)
    assert rendered is not None
    assert rendered.plain.count("error=true") == 1
    assert rendered.plain.count("canceled=false") == 1
    await loop.close()


@pytest.mark.asyncio
async def test_multibyte_agent_receipt_stays_within_response_limit(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("😀" * 1_800)]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    persisted = next(message for message in store.messages() if message.tool_result)
    result = persisted.tool_result
    assert result is not None
    assert (
        len(json.dumps(persisted.to_dict(), ensure_ascii=False).encode("utf-8"))
        <= 10_000
    )
    receipt_row = next(
        row for row in store.path.read_bytes().splitlines() if b'"tool_result"' in row
    )
    assert len(receipt_row) <= 10_000
    assert result.content.count("error=false") == 1
    assert result.content.count("canceled=false") == 1


@pytest.mark.asyncio
async def test_parent_abort_cancels_child(tmp_path: Path) -> None:
    child_started = asyncio.Event()

    class StartedBackend(FakeBackend):
        async def complete(
            self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
        ) -> AsyncIterator[StreamEvent]:
            async for event in super().complete(messages, tool_schemas):
                if (
                    len(self.calls) == 2
                    and event.type is StreamEventType.MESSAGE_UPDATE
                ):
                    child_started.set()
                yield event

    backend = StartedBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.wait_for(child_started.wait(), timeout=10)
    loop.abort()

    events = await task
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("tool execution canceled")
    assert "error=false" in result.content
    assert "canceled=true" in result.content
    assert result.structured_content == {
        "turns_used": 0,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "child_instance_id": f"{store.session_id}:1",
    }
    rendered = render_event(
        next(
            event
            for event in events
            if event.type is StreamEventType.TOOL_EXECUTION_END
        )
    )
    assert rendered is not None
    assert rendered.plain.count("error=false") == 1
    assert rendered.plain.count("canceled=true") == 1
    child_state = (
        store.session_dir / "agents" / "1" / "session_state.json"
    ).read_text()
    assert '"agent_parent"' not in child_state
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_canceled() == {
        "tool_call_id": "agent-1",
        "content": "tool execution canceled",
    }
    assert not store.agent_children()


@pytest.mark.asyncio
async def test_parent_abort_after_child_turn_reports_completed_turns(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(
                [TextContent("first")],
                tool_calls=[ToolCall("child-read", "read", {"path": "missing"})],
            ),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    async for event in loop.run_turn("start", origin=MessageOrigin.USER):
        if (
            event.type is StreamEventType.TOOL_EXECUTION_UPDATE
            and event.delta is not None
            and "turn 2: thinking" in event.delta
        ):
            loop.abort()

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content == {
        "turns_used": 1,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "child_instance_id": f"{store.session_id}:1",
    }


@pytest.mark.asyncio
async def test_typed_child_type_survives_cancellation_and_reopen(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.sleep(0.05)
    loop.abort()

    await task

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_type() == "explore"


@pytest.mark.asyncio
async def test_parent_result_append_precedes_marker_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    def fail_cleanup(tool_call_id: str) -> None:
        del tool_call_id
        raise RuntimeError("crash after parent result")

    monkeypatch.setattr(store, "finish_agent_child", fail_cleanup)
    with pytest.raises(RuntimeError, match="crash after parent result"):
        await _collect(
            AgentLoop(
                backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
            ).run_turn("start", origin=MessageOrigin.USER)
        )

    assert store.agent_children()
    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    AgentLoop(
        FakeBackend([]), replayed, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    results = [
        message.tool_result for message in replayed.messages() if message.tool_result
    ]
    assert len(results) == 1
    assert results[0].content.startswith("done")
    assert not replayed.agent_children()


@pytest.mark.parametrize("receipt_limit", [5_000, 50_000])
@pytest.mark.asyncio
async def test_resume_reuses_exact_foreground_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_limit: int,
) -> None:
    answer = "x" * 30_000 + "canonical-tail"
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent(answer)])]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        max_output_chars=receipt_limit,
        skill_catalog=SkillCatalog.empty(),
    )
    original_receipts: list[str] = []
    append_message = store.append_message

    def crash_before_parent_result(message: Message) -> None:
        if message.tool_result is not None:
            original_receipts.append(message.tool_result.content)
            raise RuntimeError("crash before parent result")
        append_message(message)

    monkeypatch.setattr(store, "append_message", crash_before_parent_result)
    with pytest.raises(RuntimeError, match="crash before parent result"):
        await _collect(
            AgentLoop(
                backend,
                store,
                registry=registry,
                max_turns=1,
                skill_catalog=SkillCatalog.empty(),
            ).run_turn("start", origin=MessageOrigin.USER)
        )

    assert len(original_receipts) == 1
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"] == original_receipts[0]

    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    AgentLoop(
        FakeBackend([]), replayed, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    recovered = [
        message.tool_result for message in replayed.messages() if message.tool_result
    ]
    assert len(recovered) == 1
    assert recovered[0].content == original_receipts[0]


@pytest.mark.asyncio
async def test_setup_failure_persists_canonical_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend([ScriptedTurn(tool_calls=[_agent_call()])])
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        max_output_chars=1_000,
        skill_catalog=SkillCatalog.empty(),
    )

    def fail_setup(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("child setup failed")

    monkeypatch.setattr(registry, "clone_for_session", fail_setup)
    await _collect(
        AgentLoop(
            backend,
            store,
            registry=registry,
            max_turns=1,
            skill_catalog=SkillCatalog.empty(),
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"] == result.content
    assert lifecycle["final_result_is_receipt"] is True
    assert result.content.endswith("error=true · canceled=false")


def test_resume_resolves_dead_child_marker(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _agent_call()
    child = ConversationStore(
        store.session_dir / "agents", session_id="1", cwd=store.cwd
    )
    child.mark_agent_parent(call.id)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="task research",
    )
    store.update_agent_child_turns(call.id, 2)

    AgentLoop(FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty())

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("tool execution canceled")
    assert result.structured_content == {
        "turns_used": 2,
        "child_session_path": str(child.session_dir),
        "child_instance_id": "parent:1",
    }
    assert not store.agent_children()
    assert '"agent_parent"' not in (child.state_path).read_text()


def test_resume_preserves_typed_child_receipt(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _agent_call(agent_type="explore")
    child = ConversationStore(
        store.session_dir / "agents", session_id="1", cwd=store.cwd
    )
    child.mark_agent_parent(call.id, agent_type="explore")
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="task research",
        agent_type="explore",
    )
    store.update_agent_child_turns(call.id, 2)

    AgentLoop(FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty())

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"
    reopened_child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert reopened_child.agent_type() == "explore"


def _model_agent_call(
    model: str = "gpt-5.4",
    call_id: str = "agent-model-1",
    **extra: object,
) -> ToolCall:
    arguments: dict[str, object] = {
        "prompt": "inspect the task",
        "description": "cross provider research",
        "model": model,
    }
    arguments.update(extra)
    return ToolCall(call_id, "agent", arguments)


class _ValidTokens:
    def is_valid(self, *, skew: float = 60) -> bool:
        del skew
        return True


class _FakeCredentialStore:
    """Stand in for an OAuth store without touching the real credential files."""

    def __init__(self, tokens: object | None) -> None:
        self._tokens = tokens

    def read(self) -> object | None:
        return self._tokens

    def bootstrap(self) -> object | None:
        return None


def _stub_backend_factory(
    monkeypatch: pytest.MonkeyPatch,
    child_backend: CompletionBackend,
    *,
    tokens: object | None = None,
) -> list[tuple[str, str | None]]:
    """Record what the runner asks the factory for, and hand back child_backend."""

    requested: list[tuple[str, str | None]] = []

    def build(provider: str, model: str | None, **kwargs: object):
        del kwargs
        requested.append((provider, model))
        return child_backend, model or ""

    monkeypatch.setattr("zeta.agent.runner.build_backend", build)
    monkeypatch.setattr(
        "zeta.agent.runner.credential_store",
        lambda provider, **kwargs: _FakeCredentialStore(
            _ValidTokens() if tokens is None else tokens
        ),
    )
    return requested


def test_provider_for_model_maps_each_catalog_entry() -> None:
    from zeta.models.catalog import PROVIDER_MODELS, provider_for_model

    for provider, models in PROVIDER_MODELS.items():
        for model in models:
            assert provider_for_model(model) == provider


def test_provider_for_model_rejects_an_unknown_name() -> None:
    from zeta.models.catalog import provider_for_model

    with pytest.raises(ValueError) as excinfo:
        provider_for_model("gpt-nonexistent")
    message = str(excinfo.value)
    assert "gpt-nonexistent" in message
    assert "claude-opus-5" in message


def test_agent_schema_offers_every_known_model(tmp_path: Path) -> None:
    from zeta.models.catalog import known_model_names

    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(
        FakeBackend([]), store, registry=registry, skill_catalog=SkillCatalog.empty()
    )

    agent_schema = next(
        schema for schema in registry.schemas if schema["name"] == "agent"
    )
    model_schema = agent_schema["parameters"]["properties"]["model"]
    assert model_schema["enum"] == known_model_names()
    assert "claude-opus-5" in model_schema["enum"]
    assert "gpt-5.4" in model_schema["enum"]


@pytest.mark.asyncio
async def test_agent_without_a_model_still_inherits_the_parent_backend(
    tmp_path: Path,
) -> None:
    """The pre-existing spawn path must keep using loop.backend."""

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("done")
    # Parent and child both ran on the one backend, so it saw both turns.
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_agent_with_a_model_runs_the_child_on_that_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("codex done")])])
    requested = _stub_backend_factory(monkeypatch, child_backend)
    parent_backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_model_agent_call(background=False)])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            parent_backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    assert requested == [("codex", "gpt-5.4")]
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("codex done")
    # The child talked to the substitute, never to the parent's backend.
    assert len(child_backend.calls) == 1
    assert len(parent_backend.calls) == 1


@pytest.mark.asyncio
async def test_child_usage_is_counted_separately_from_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend(
        [
            ScriptedTurn(
                [TextContent("done")], usage={"input_tokens": 20, "output_tokens": 2}
            )
        ]
    )
    _stub_backend_factory(monkeypatch, child_backend)
    parent_backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[_model_agent_call(background=False)],
                usage={"input_tokens": 5, "output_tokens": 1},
            )
        ]
    )
    loop = AgentLoop(
        parent_backend,
        ConversationStore(tmp_path),
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert loop.context_assembler.descendant_usage == {
        "input_tokens": 20,
        "output_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": len(child_backend.request_bytes[0]),
    }
    assert loop.context_assembler.descendant_usage_by_model == {
        "gpt-5.4": loop.context_assembler.descendant_usage
    }
    assert loop.context_assembler.uncached_input_tokens_this_session == 5


def test_nested_child_usage_propagates_to_root_once(tmp_path: Path) -> None:
    root = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "root"),
        skill_catalog=SkillCatalog.empty(),
    )
    child = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "child"),
        skill_catalog=SkillCatalog.empty(),
        usage_sink=root.context_assembler.record_descendant_usage,
    )
    grandchild = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "grandchild"),
        skill_catalog=SkillCatalog.empty(),
        usage_sink=child.context_assembler.record_descendant_usage,
    )

    grandchild.context_assembler.record_usage(
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
            "_zeta_model": "gpt-5.6-sol",
        }
    )

    assert (
        root.context_assembler.descendant_usage
        == child.context_assembler.descendant_usage
        == {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
        }
    )
    assert root.context_assembler.descendant_usage_by_model == {
        "gpt-5.6-sol": root.context_assembler.descendant_usage
    }
    root.context_assembler.record_descendant_usage({"input_tokens": 1})
    assert (
        root.context_assembler.descendant_usage_by_model["unknown"]["input_tokens"] == 1
    )


@pytest.mark.asyncio
async def test_agent_model_implies_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend(
        [
            ScriptedTurn(
                [TextContent("codex done")],
                usage={"input_tokens": 8, "output_tokens": 2},
            )
        ]
    )
    _stub_backend_factory(monkeypatch, child_backend)
    parent_backend = FakeBackend([ScriptedTurn(tool_calls=[_model_agent_call()])])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        parent_backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    await loop._background_owner.wait()
    assert loop.context_assembler.descendant_usage["input_tokens"] == 8
    assert loop.context_assembler.descendant_usage["output_tokens"] == 2
    await loop.close()


@pytest.mark.asyncio
async def test_agent_rejects_an_unknown_model_before_spawning(tmp_path: Path) -> None:
    """The schema enum catches a bad model name before the runner is reached."""

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_model_agent_call(model="gpt-nonexistent")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert "model is not an allowed value" in result.content
    assert not (store.session_dir / "agents").exists()


def test_resolve_child_backend_guards_bad_models(tmp_path: Path) -> None:
    """Second line of defence, for any caller that skips schema validation."""

    from zeta.agent.runner import resolve_child_backend

    parent_backend = FakeBackend([])
    loop = AgentLoop(
        parent_backend, ConversationStore(tmp_path), skill_catalog=SkillCatalog.empty()
    )

    # No model at all keeps the parent's backend.
    assert resolve_child_backend(loop, None) == (parent_backend, None)

    backend, error = resolve_child_backend(loop, "gpt-nonexistent")
    assert backend is None
    assert error is not None and "unknown model" in error

    backend, error = resolve_child_backend(loop, "")
    assert backend is None
    assert error is not None and "nonempty string" in error

    backend, error = resolve_child_backend(loop, 7)
    assert backend is None
    assert error is not None and "nonempty string" in error


def test_resolve_child_backend_accepts_bootstrapped_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.agent.runner import resolve_child_backend

    parent = FakeBackend([])
    child = FakeBackend([])
    loop = AgentLoop(
        parent, ConversationStore(tmp_path), skill_catalog=SkillCatalog.empty()
    )

    class BootstrapStore(_FakeCredentialStore):
        def bootstrap(self) -> object:
            return _ValidTokens()

    monkeypatch.setattr(
        "zeta.agent.runner.credential_store", lambda provider: BootstrapStore(None)
    )
    monkeypatch.setattr(
        "zeta.agent.runner.build_backend",
        lambda provider, model, *, token_budget=None: (child, model),
    )
    assert resolve_child_backend(loop, "gpt-5.6-luna") == (child, None)


@pytest.mark.asyncio
async def test_agent_reports_a_missing_provider_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("unreachable")])])
    _stub_backend_factory(monkeypatch, child_backend, tokens=None)
    monkeypatch.setattr(
        "zeta.agent.runner.credential_store",
        lambda provider, **kwargs: _FakeCredentialStore(None),
    )
    backend = FakeBackend([ScriptedTurn(tool_calls=[_model_agent_call()])])
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert "not logged in to codex" in result.content
    assert "zeta login --provider codex" in result.content
    assert not child_backend.calls


def test_agent_loop_turn_cap_allows_long_runs(tmp_path: Path) -> None:
    """50 turns was too few for real work; the default has to clear it."""

    import inspect

    default = inspect.signature(AgentLoop.__init__).parameters["max_turns"].default
    assert default == 150
    assert (
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ).max_turns
        == 150
    )


@pytest.mark.asyncio
async def test_background_start_text_names_handle_and_polling_tools(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    handle = result.structured_content["child_instance_id"]
    assert type(handle) is str and handle
    assert result.structured_content["status"] == "running"
    assert f"handle={handle}" in result.content
    assert "agent_status" in result.content
    assert "agent_output" in result.content
    assert "Completion is announced automatically" in result.content
    assert "poll with" not in result.content
    assert "task_output" not in result.content
    assert result.content.startswith("background agent started:")

    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_delegated_turn_count_stays_accurate_without_a_denominator(
    tmp_path: Path,
) -> None:
    call = _agent_call()
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    turns = 27
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn([TextContent(f"step-{turn}")], tool_calls=[child_read])
            for turn in range(1, turns + 1)
        ]
        + [ScriptedTurn([TextContent("child done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_lifecycle()["turns_used"] == turns + 1
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None and not result.is_error
    assert result.content.startswith("child done")


class RunBackend(CompletionBackend):
    """Drive a long run whose first turn can be held open mid-flight."""

    def __init__(self) -> None:
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()
        self.child_prompts: list[str] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            blocks = [ToolUseContent(_run_agent_call())]
        elif last_user == "work the big task":
            self.child_prompts.append(last_user)
            self.child_started.set()
            await self.release_child.wait()
            blocks = [TextContent("first pass done")]
        else:
            self.child_prompts.append(last_user)
            blocks = [TextContent("follow-up handled")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _run_agent_call(call_id: str = "run-1") -> ToolCall:
    return ToolCall(
        call_id,
        "agent",
        {
            "prompt": "work the big task",
            "description": "long horizon run",
            "agent_type": "run",
        },
    )


class _RunCommands(AgentRunCommandMixin):
    """Minimal host for the mixin: it only needs loop.store."""

    def __init__(self, loop: AgentLoop) -> None:
        self.loop = loop


def test_run_preset_is_registered_without_a_turn_cap(tmp_path: Path) -> None:
    from zeta.agent.presets import AGENT_PRESETS, RUN_PRESET

    assert not hasattr(RUN_PRESET, "turn_cap")
    assert RUN_PRESET.tool_names is None
    assert AGENT_PRESETS["run"] is RUN_PRESET

    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(
        FakeBackend([]), store, registry=registry, skill_catalog=SkillCatalog.empty()
    )
    agent_schema = next(
        schema for schema in registry.schemas if schema["name"] == "agent"
    )
    assert "run" in agent_schema["parameters"]["properties"]["agent_type"]["enum"]


@pytest.mark.asyncio
async def test_a_run_does_not_draw_on_the_shared_sibling_budget(
    tmp_path: Path,
) -> None:
    """A run and a restricted sibling complete independently without a tree budget."""

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("explore-1", agent_type="explore")]),
            ScriptedTurn([TextContent("explore done")]),
            ScriptedTurn(tool_calls=[_run_agent_call()]),
            ScriptedTurn([TextContent("run done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await _collect(loop.run_turn("now start the run", origin=MessageOrigin.USER))
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_a_run_goes_to_the_background_without_being_asked(
    tmp_path: Path,
) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


def test_pending_prompt_queue_round_trips(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="run")
    first = store.append_pending_prompt("also update the docs")
    store.append_pending_prompt("and run the linter")
    assert [entry.data["text"] for entry in store.pending_prompts()] == [
        "also update the docs",
        "and run the linter",
    ]

    store.acknowledge_pending_prompt(first.id)
    assert [entry.data["text"] for entry in store.pending_prompts()] == [
        "and run the linter"
    ]
    # Acknowledging twice is a no-op rather than an integrity error.
    store.acknowledge_pending_prompt(first.id)

    # A separate handle sees the queue, and the queue survives a reload.
    assert [
        entry.data["text"]
        for entry in ConversationStore(tmp_path, session_id="run").pending_prompts()
    ] == ["and run the linter"]

    with pytest.raises(ValueError):
        store.append_pending_prompt("   ")
    with pytest.raises(ValueError):
        store.acknowledge_pending_prompt("nonexistent")


def test_close_pending_queue_if_empty_locks_out_new_prompts(tmp_path: Path) -> None:
    """Once a run declares itself done, further follow-ups must be rejected."""

    store = ConversationStore(tmp_path, session_id="run")

    # With prompts pending the queue stays open and callers still see them.
    store.append_pending_prompt("keep working")
    pending = store.close_pending_queue_if_empty()
    assert [entry.data["text"] for entry in pending] == ["keep working"]
    store.acknowledge_pending_prompt(pending[0].id)

    # Second call finds nothing pending and closes the door.
    assert store.close_pending_queue_if_empty() == []
    with pytest.raises(PendingPromptsClosedError):
        store.append_pending_prompt("too late")
    # A separate handle sees the closed door too, not just this instance.
    other = ConversationStore(tmp_path, session_id="run")
    with pytest.raises(PendingPromptsClosedError):
        other.append_pending_prompt("also too late")


@pytest.mark.asyncio
async def test_consume_run_keeps_pending_entry_when_delivery_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed follow-up delivery must leave the queue intact for recovery."""

    from zeta.agent import runner as agent_runner

    child_store = ConversationStore(tmp_path, session_id="run")
    child_store.append_pending_prompt("follow up")

    call_prompts: list[str] = []

    async def fake_consume_child(child_loop, prompt, **kwargs):
        del child_loop, kwargs
        call_prompts.append(prompt)
        if len(call_prompts) == 1:
            return {"content": [], "isError": False, "structuredContent": None}
        return {"content": [], "isError": True, "structuredContent": None}

    monkeypatch.setattr(agent_runner, "consume_child", fake_consume_child)

    result = await agent_runner.consume_run(
        object(),  # child_loop is unused by the fake
        "initial",
        child_store=child_store,
    )
    assert result["isError"] is True
    assert call_prompts == ["initial", "follow up"]
    # Ack must not have run since the follow-up delivery errored.
    assert [entry.data["text"] for entry in child_store.pending_prompts()] == [
        "follow up"
    ]
    with pytest.raises(PendingPromptsClosedError):
        child_store.append_pending_prompt("after failure")


@pytest.mark.asyncio
async def test_restart_keeps_run_lifecycle_open_for_an_in_flight_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zeta.agent import runner as agent_runner

    child_store = ConversationStore(tmp_path, session_id="run")
    child_store.start_agent_lifecycle(
        handle="parent:run",
        started_at="2026-09-08T00:00:00+00:00",
        depth=1,
        agent_type="run",
        description="long horizon run",
    )
    child_store.append_pending_prompt("follow up")
    prompts: list[str] = []
    finish_calls = 0

    async def fake_consume_child(child_loop, prompt, **kwargs):
        del child_loop, kwargs
        prompts.append(prompt)
        return {
            "content": [
                {
                    "text": (
                        f"handled {prompt} · 2 turns · 1.2s · 3 tool calls "
                        "· error=false · canceled=false"
                    )
                }
            ],
            "isError": False,
            "structuredContent": None,
        }

    def finish_lifecycle(state: str, text: str) -> dict[str, object]:
        nonlocal finish_calls
        finish_calls += 1
        child_store.finish_agent_lifecycle(state, final_result=text)
        return {}

    original_close = child_store.close_pending_queue_if_empty
    checked_restart = False

    def close_after_restart_check():
        nonlocal checked_restart
        if not checked_restart:
            checked_restart = True
            restarted = ConversationStore(tmp_path, session_id="run")
            lifecycle = restarted.agent_lifecycle()
            assert lifecycle is not None
            assert lifecycle["finished_at"] is None
            assert restarted.pending_prompts()[0].data["text"] == "follow up"
        return original_close()

    monkeypatch.setattr(agent_runner, "consume_child", fake_consume_child)
    monkeypatch.setattr(
        child_store,
        "close_pending_queue_if_empty",
        close_after_restart_check,
    )

    result = await agent_runner.consume_run(
        object(),
        "initial",
        child_store=child_store,
        child_turns=lambda: len(prompts),
        finish_lifecycle=finish_lifecycle,
    )

    assert result["isError"] is False
    assert prompts == ["initial", "follow up"]
    assert finish_calls == 1
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["state"] == "completed"
    assert lifecycle["finished_at"] is not None
    assert lifecycle["final_result"] == "handled follow up"


@pytest.mark.asyncio
async def test_agent_send_waits_for_blocked_append_before_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(parent_store.session_dir / "agents", session_id="1")
    call = _run_agent_call()
    child_store.mark_agent_parent(call.id, agent_type="run")
    parent_store.allocate_agent_index()
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="long horizon run",
        agent_type="run",
        background=True,
        child_instance_id="parent:1",
    )

    started = threading.Event()
    release = threading.Event()
    original_send = agent_send_module.send_to_run

    def blocked_send(*args):
        started.set()
        release.wait(timeout=2)
        return original_send(*args)

    monkeypatch.setattr(agent_send_module, "send_to_run", blocked_send)
    registry = ToolRegistry(
        tmp_path, session_store=parent_store, skill_catalog=SkillCatalog.empty()
    )
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "agent-send-call",
                "agent_send",
                {"child_instance_id": "parent:1", "message": "follow up"},
            )
        )
    )
    await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    result = await task

    assert result["isError"] is False
    assert [entry.data["text"] for entry in child_store.pending_prompts()] == [
        "follow up"
    ]


@pytest.mark.asyncio
async def test_tool_registry_reports_agent_send_result_after_cleanup_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(parent_store.session_dir / "agents", session_id="1")
    call = _run_agent_call()
    child_store.mark_agent_parent(call.id, agent_type="run")
    parent_store.allocate_agent_index()
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="long horizon run",
        agent_type="run",
        background=True,
        child_instance_id="parent:1",
    )

    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    original_gather = execution_module.asyncio.gather

    async def blocked_cleanup(*args, **kwargs):
        cleanup_started.set()
        await release_cleanup.wait()
        return await original_gather(*args, **kwargs)

    monkeypatch.setattr(execution_module.asyncio, "gather", blocked_cleanup)
    registry = ToolRegistry(
        tmp_path, session_store=parent_store, skill_catalog=SkillCatalog.empty()
    )
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "agent-send-cleanup-cancel",
                "agent_send",
                {"child_instance_id": "parent:1", "message": "follow up"},
            )
        )
    )

    await asyncio.wait_for(cleanup_started.wait(), timeout=2)
    task.cancel()
    release_cleanup.set()
    result = await task

    assert result["isError"] is False
    assert [entry.data["text"] for entry in child_store.pending_prompts()] == [
        "follow up"
    ]


@pytest.mark.asyncio
async def test_agent_send_aborts_before_append_when_store_lock_is_held(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(parent_store.session_dir / "agents", session_id="1")
    call = _run_agent_call()
    child_store.mark_agent_parent(call.id, agent_type="run")
    parent_store.allocate_agent_index()
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="long horizon run",
        agent_type="run",
        background=True,
        child_instance_id="parent:1",
    )

    registry = ToolRegistry(
        tmp_path, session_store=parent_store, skill_catalog=SkillCatalog.empty()
    )
    lock = child_store._append_lock()
    lock.__enter__()
    try:
        task = asyncio.create_task(
            registry.execute(
                ToolCall(
                    "agent-send-call",
                    "agent_send",
                    {"child_instance_id": "parent:1", "message": "follow up"},
                )
            )
        )
        await asyncio.sleep(0)
        started = time.monotonic()
        task.cancel()
        result = await asyncio.wait_for(task, timeout=2)
        elapsed = time.monotonic() - started
    finally:
        lock.__exit__(None, None, None)

    assert elapsed < 1.8
    assert result["isError"] is True
    assert "timed out" in result["content"][0]["text"]
    assert child_store.pending_prompts() == []


@pytest.mark.asyncio
async def test_agent_send_reports_when_the_run_just_closed(tmp_path: Path) -> None:
    """The race the closed-queue marker prevents: a queued prompt after finish."""

    from zeta.tools.agent import send_to_run

    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)
    handle = _run_handle_from_receipt(store)

    # Simulate the race: parent looked at the marker before the run finished,
    # then the run drained and closed its queue before the parent got here.
    marker = store.agent_children()[handle]
    child_path = Path(str(marker["child_session_path"]))
    child_store = ConversationStore(
        child_path.parent, session_id=child_path.name, cwd=store.cwd
    )
    assert child_store.close_pending_queue_if_empty() == []

    error = send_to_run(store, handle, "too late")
    assert error is not None and "no live run" in error

    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_a_queued_prompt_stays_out_of_the_run_context(tmp_path: Path) -> None:
    """Only messages reach the model; a queued follow-up must not leak in early."""

    store = ConversationStore(tmp_path, session_id="run")
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("work the big task")]), MessageOrigin.USER))
    store.append_pending_prompt("secret follow-up")
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())

    assembled = await loop.context_assembler.assemble()

    rendered = "\n".join(
        block.text
        for message in assembled
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert "work the big task" in rendered
    assert "secret follow-up" not in rendered


def _run_handle_from_receipt(store: ConversationStore) -> str:
    """Return the child_instance_id a real model would receive for the run."""

    for message in store.messages():
        result = message.tool_result
        if result is None:
            continue
        structured = result.structured_content
        if structured is None:
            continue
        handle = structured.get("child_instance_id")
        if type(handle) is str and handle:
            return handle
    raise AssertionError("no run receipt with a child_instance_id")


@pytest.mark.asyncio
async def test_a_follow_up_reaches_the_run_at_its_next_turn(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)

    # The model queues follow-ups by the child_instance_id it saw in the tool
    # result, not by the provider tool_call.id, so round-trip that handle.
    handle = _run_handle_from_receipt(store)
    assert send_to_run(store, handle, "also check the tests") is None

    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    assert backend.child_prompts == ["work the big task", "also check the tests"]
    await loop.close()


@pytest.mark.asyncio
async def test_a_run_with_an_empty_queue_finishes_normally(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    assert backend.child_prompts == ["work the big task"]
    assert not store.agent_children()
    await loop.close()


def test_send_to_run_rejects_unknown_and_finished_runs(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    error = send_to_run(store, "run-1", "hello")
    assert error is not None and "no live run" in error

    assert send_to_run(store, "", "hello") == (
        "child_instance_id must be a nonempty string"
    )
    error = send_to_run(store, "run-1", "  ")
    assert error == "message must be a nonempty string"


def test_send_to_run_rejects_non_run_children(tmp_path: Path) -> None:
    """A queued prompt would rot: only consume_run drains the queue."""

    store = ConversationStore(tmp_path)
    child_call = ToolCall(
        "explore-1",
        "agent",
        {"prompt": "look", "description": "explore", "agent_type": "explore"},
    )
    store.register_agent_child(
        child_call,
        child_session_path=str(tmp_path / "agents" / "1"),
        description="explore",
        agent_type="explore",
        background=True,
        child_instance_id="sess:1",
    )

    error = send_to_run(store, "sess:1", "hello")
    assert error is not None
    assert "explore" in error and "agent_send" in error


@pytest.mark.asyncio
async def test_runs_and_send_commands_drive_a_live_run(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    commands = _RunCommands(loop)

    explore_call = ToolCall(
        "explore-1",
        "agent",
        {"prompt": "look", "description": "explore", "agent_type": "explore"},
    )
    store.register_agent_child(
        explore_call,
        child_session_path=str(tmp_path / "agents" / "explore"),
        description="explore",
        agent_type="explore",
        background=True,
        child_instance_id="sess:explore",
    )

    assert commands.slash_runs("") == "no live runs"

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)

    listing = commands.slash_runs("")
    assert "long horizon run" in listing
    assert "explore" not in listing
    handle = _run_handle_from_receipt(store)
    # /runs shows the model-facing handle, not the opaque provider tool_call.id.
    assert handle in listing
    assert "run-1" not in listing

    assert commands.slash_send("nonsense") == (
        "use /send <run-id> <message>; /runs lists the live ones"
    )
    assert "no live run" in commands.slash_send("bogus-id hello")
    assert f"queued for {handle}" in commands.slash_send(
        f"{handle} also check the tests"
    )

    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    assert backend.child_prompts == ["work the big task", "also check the tests"]
    await loop.close()


@pytest.mark.asyncio
async def test_child_thinking_only_reply_with_pending_notification_is_nudged(
    tmp_path: Path,
) -> None:
    class Backend(CompletionBackend):
        def __init__(self, store: ConversationStore) -> None:
            self.store = store
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            del tool_schemas
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                self.store.append_task_notification(
                    task_id="task", command="work", exit_code=0
                )
                blocks = [ThinkingContent("planning", "sig")]
            else:
                blocks = [TextContent("done")]
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, blocks),
            )

    store = ConversationStore(tmp_path)
    backend = Backend(store)
    loop = AgentLoop(
        backend, store, max_turns=1, agent_depth=1, skill_catalog=SkillCatalog.empty()
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    nudges = [
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "empty_turn_nudge"
    ]
    assert len(nudges) == 1
    assert len(backend.calls) == 2
    await loop.close()


@pytest.mark.asyncio
async def test_context_retry_of_notification_consumption_is_never_nudged(
    tmp_path: Path,
) -> None:
    class Backend(CompletionBackend):
        def __init__(self, store: ConversationStore) -> None:
            self.store = store
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            self.calls.append(list(messages))
            call = len(self.calls)
            if call == 1:
                self.store.append_task_notification(
                    task_id="task", command="work", exit_code=0
                )
                blocks = [TextContent("visible work " * 100)]
            elif call == 2:
                error = RuntimeError("context_length_exceeded: stream error")
                error.code = "context_length_exceeded"
                raise error
            elif not tool_schemas:
                blocks = [TextContent("summary")]
            elif call == 4:
                blocks = [ThinkingContent("quiet", "sig")]
            else:
                blocks = [TextContent("unexpected recovery")]
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, blocks),
            )

    store = ConversationStore(tmp_path)
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("old work")]), MessageOrigin.USER))
    backend = Backend(store)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        agent_depth=1,
        token_budget=10_000,
        retained_tail=1,
        skill_catalog=SkillCatalog.empty(),
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert not any(
        message.metadata.get("zeta_event") == "empty_turn_nudge"
        for message in store.messages()
    )
    assert len(backend.calls) == 3
    await loop.close()


@pytest.mark.asyncio
async def test_context_retry_of_ordinary_empty_reply_still_nudged_once(
    tmp_path: Path,
) -> None:
    class Backend(CompletionBackend):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            self.calls.append(list(messages))
            call = len(self.calls)
            if call == 1:
                error = RuntimeError("context_length_exceeded: stream error")
                error.code = "context_length_exceeded"
                raise error
            if not tool_schemas:
                blocks = [TextContent("summary")]
            elif call == 2:
                blocks = [ThinkingContent("quiet", "sig")]
            else:
                blocks = [TextContent("recovered")]
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, blocks),
            )

    store = ConversationStore(tmp_path)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("old reasoning " * 100), TextContent("old work")],
        )
    )
    backend = Backend()
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        agent_depth=1,
        token_budget=10_000,
        retained_tail=1,
        skill_catalog=SkillCatalog.empty(),
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    nudges = [
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "empty_turn_nudge"
    ]
    assert len(nudges) == 1
    assert len(backend.calls) == 3
    await loop.close()


@pytest.mark.asyncio
async def test_notification_consumption_reply_is_never_nudged(tmp_path: Path) -> None:
    class Backend(CompletionBackend):
        def __init__(self, store: ConversationStore) -> None:
            self.store = store
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            del tool_schemas
            self.calls.append(list(messages))
            notification = any(
                message.metadata.get("zeta_event") == "agent_notifications"
                for message in messages
            )
            if len(self.calls) == 1:
                self.store.append_task_notification(
                    task_id="task", command="work", exit_code=0
                )
                blocks = [TextContent("visible")]
            else:
                assert notification
                blocks = [ThinkingContent("quiet", "sig")]
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, blocks),
            )

    store = ConversationStore(tmp_path)
    backend = Backend(store)
    loop = AgentLoop(
        backend, store, max_turns=1, agent_depth=1, skill_catalog=SkillCatalog.empty()
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert not any(
        message.metadata.get("zeta_event") == "empty_turn_nudge"
        for message in store.messages()
    )
    assert len(backend.calls) == 2
    await loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "codex"])
async def test_nudge_notification_provider_shape(provider: str, tmp_path: Path) -> None:
    class Backend(CompletionBackend):
        def __init__(self, store: ConversationStore) -> None:
            self.store = store
            self.calls: list[list[Message]] = []

        async def complete(self, messages, tool_schemas):
            del tool_schemas
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                self.store.append_task_notification(
                    task_id="task", command="work", exit_code=0
                )
                blocks = [ThinkingContent("planning", "sig")]
            else:
                blocks = [TextContent("done")]
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, blocks),
            )

    store = ConversationStore(tmp_path)
    backend = Backend(store)
    loop = AgentLoop(
        backend, store, max_turns=1, agent_depth=1, skill_catalog=SkillCatalog.empty()
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    request_messages = backend.calls[-1]
    assert any(
        message.metadata.get("zeta_event") == "empty_turn_nudge"
        for message in request_messages
    )
    assert any(
        message.metadata.get("zeta_event") == "agent_notifications"
        for message in request_messages
    )

    if provider == "anthropic":
        from zeta.providers.anthropic import build_messages_payload

        payload = build_messages_payload(
            request_messages, [], model="claude-test", max_tokens=16384
        )
        roles = [message["role"] for message in payload["messages"]]
        assert all(left != right for left, right in pairwise(roles))
    else:
        from zeta.providers.codex import build_responses_payload

        payload = build_responses_payload(request_messages, [], model="codex-test")
        assert payload["input"]
    await loop.close()


@pytest.mark.asyncio
async def test_subagent_thinking_only_reply_is_nudged_before_failing(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(
                [ThinkingContent("planning", "sig-1")], stop_reason="end_turn"
            ),
            ScriptedTurn(
                [TextContent("child complete: answer")], stop_reason="end_turn"
            ),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()
        ).run_turn("start", origin=MessageOrigin.USER)
    )

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None
    assert not result.is_error
    assert "child complete: answer" in result.content
    assert "empty final assistant message" not in result.content


def _python(*parts: str) -> str:
    return shlex.join((sys.executable, "-c", *parts))


async def _wait_completion(store: ConversationStore, timeout: float = 10.0) -> object:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        completed = [
            entry
            for entry in store.agent_notifications(pending_only=False)
            if entry.data.get("kind", "agent_completion") == "agent_completion"
            and entry.data.get("status") == "completed"
        ]
        if completed:
            return completed[-1]
        await asyncio.sleep(0.02)
    raise AssertionError("no completion notification")


class _TaskOwningChildBackend(CompletionBackend):
    """Root spawns a background child; the child starts a long task then completes."""

    def __init__(self, command: str) -> None:
        self.command = command

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        started = any(
            message.role is MessageRole.TOOL_RESULT
            and any(
                isinstance(block, TextContent)
                and "started background task" in block.text
                for block in message.content
            )
            for message in messages
        )
        if last_user == "start":
            blocks: list = [
                ToolUseContent(
                    ToolCall(
                        "bg-child",
                        "agent",
                        {
                            "prompt": "own a task",
                            "description": "task owner",
                            "background": True,
                        },
                    )
                )
            ]
        elif last_user == "own a task" and not started:
            blocks = [
                ToolUseContent(
                    ToolCall("bg-task", "run_background", {"command": self.command})
                )
            ]
        else:
            blocks = [TextContent("child done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


@pytest.mark.asyncio
async def test_child_completion_reports_killed_tasks_in_receipt(tmp_path: Path) -> None:
    # S2: a child that still owns a running task when it completes has the task
    # killed without a notification; close() returns the ids and the runner
    # folds them into the child's completion receipt (text + data).
    backend = _TaskOwningChildBackend(_python("import time; time.sleep(30)"))
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    events = []
    loop.set_background_event_sink(events.append)
    events.extend(await _collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    notification = await _wait_completion(store)
    text = notification.data["text"]
    assert "background tasks killed on child completion" in text
    stats_marker = " · error=false · canceled=false"
    assert text.count("error=") == 1
    assert text.count("canceled=") == 1
    assert text.count(stats_marker) == 1
    event = next(
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_END
        and event.tool_result is not None
        and event.data.get("notification_id")
    )
    assert event.tool_result is not None
    assert event.tool_result.content.count("error=") == 1
    assert event.tool_result.content.count("canceled=") == 1
    killed = notification.data.get("killed_task_ids")
    assert isinstance(killed, list) and len(killed) == 1
    assert killed[0] in text
    assert notification.data["killed_task_count"] == 1
    assert notification.data["killed_task_ids_truncated"] is False
    child_session = Path(notification.data["child_session_path"])
    child_store = ConversationStore(child_session.parent, session_id=child_session.name)
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"] == text
    await loop.close()


@pytest.mark.asyncio
async def test_child_task_exit_does_not_wake_root(tmp_path: Path) -> None:
    # S3: a child-owned task exit is appended to the child store and must never
    # wake the shared-owner root.
    root_store = ConversationStore(tmp_path, session_id="root")
    root = AgentLoop(
        FakeBackend([]), root_store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    woke: list[int] = []
    root.set_background_wake_callback(lambda: woke.append(1))
    child_store = ConversationStore(tmp_path, session_id="child")
    child = AgentLoop(
        FakeBackend([]),
        child_store,
        max_turns=1,
        agent_depth=1,
        background_owner=root._background_owner,
        skill_catalog=SkillCatalog.empty(),
    )
    task_id, _ = await child.tool_registry.background_tasks.start(
        _python("print('x')"), Path.cwd()
    )
    await asyncio.wait_for(
        child.tool_registry.background_tasks.wait(task_id), timeout=30
    )
    await asyncio.sleep(0.05)
    child_task_exits = [
        entry
        for entry in child_store.agent_notifications(pending_only=False)
        if entry.data.get("kind") == "task_exited"
    ]
    assert len(child_task_exits) == 1
    assert child_task_exits[0].data["task_id"] == task_id
    assert woke == []
    assert [
        entry
        for entry in root_store.agent_notifications(pending_only=False)
        if entry.data.get("kind") == "task_exited"
    ] == []
    await child.close()
    await root.close()


class _ChildConsumeBackend(CompletionBackend):
    """Emit a task_exited mid-turn, then acknowledge it on the notification turn."""

    def __init__(self, store: ConversationStore) -> None:
        self.store = store
        self.saw_notification = False

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        is_notification = any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        )
        if is_notification:
            self.saw_notification = True
            blocks = [TextContent("acknowledged task exit")]
        else:
            self.store.append_task_notification(
                task_id="task-mid", command="sleep", exit_code=0
            )
            blocks = [TextContent("did work")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


@pytest.mark.asyncio
async def test_child_consumes_pending_task_notification_before_completing(
    tmp_path: Path,
) -> None:
    # S3: a child that finds pending notifications at its own turn boundary runs
    # a notification turn to consume them before completing.
    root_store = ConversationStore(tmp_path, session_id="root")
    root = AgentLoop(
        FakeBackend([]), root_store, max_turns=1, skill_catalog=SkillCatalog.empty()
    )
    child_store = ConversationStore(tmp_path, session_id="child")
    backend = _ChildConsumeBackend(child_store)
    child = AgentLoop(
        backend,
        child_store,
        max_turns=None,
        agent_depth=1,
        background_owner=root._background_owner,
        skill_catalog=SkillCatalog.empty(),
    )
    await _collect(child.run_turn("do work", origin=MessageOrigin.USER))
    assert backend.saw_notification is True
    assert child_store.agent_notifications() == []
    assert any(
        message.metadata.get("zeta_event") == "agent_notifications"
        for message in child_store.messages()
    )
    await child.close()
    await root.close()


class _RecordingBackend(CompletionBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        self.calls.append(list(messages))
        blocks = [TextContent("noted")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


@pytest.mark.asyncio
async def test_root_task_exit_wakes_idle_loop_into_notification_turn(
    tmp_path: Path,
) -> None:
    # S4: a root-owned task exit appends to the root store BEFORE the depth-aware
    # wake callback fires, waking the idle root into a notification turn whose
    # system message includes the task_exited entry.
    backend = _RecordingBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    order: list[bool] = []
    wake_task: asyncio.Task[None] | None = None

    async def consume_wake() -> None:
        async for _ in loop.run_notification_turn():
            pass

    def wake() -> None:
        nonlocal wake_task
        order.append(
            bool(
                [
                    entry
                    for entry in store.agent_notifications()
                    if entry.data.get("kind") == "task_exited"
                ]
            )
        )
        if wake_task is None:
            wake_task = asyncio.create_task(consume_wake())

    loop.set_background_wake_callback(wake)
    task_id, _ = await loop.tool_registry.background_tasks.start(
        _python("print('done')"), Path.cwd()
    )
    await asyncio.wait_for(
        loop.tool_registry.background_tasks.wait(task_id), timeout=30
    )
    while wake_task is None:
        await asyncio.sleep(0.01)
    await wake_task
    assert order and order[0] is True
    notif_message = next(
        message
        for message in store.messages()
        if message.metadata.get("zeta_event") == "agent_notifications"
    )
    text = notif_message.content[0].text
    assert "task_exited" in text
    assert task_id in text
    await loop.close()


# Agent cwd behavior tests ported from the split PR.

def _last_user_prompt(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role is MessageRole.USER:
            for block in message.content:
                if isinstance(block, TextContent):
                    return block.text
    return ""


def _pending_tool_result(messages: Sequence[Message]) -> Message | None:
    for message in reversed(messages):
        if message.role is MessageRole.TOOL_RESULT:
            return message
        if message.role is MessageRole.USER:
            return None
    return None


class ChildCwdBackend(CompletionBackend):
    """Drive one child (and optionally a grandchild) that runs a single tool.

    The child inherits the parent's backend, so one instance serves the parent
    turn, the child turns, and any grandchild turns; branches key off the last
    user prompt each turn saw.
    """

    def __init__(
        self,
        *,
        child_arguments: dict[str, object],
        child_tool_call: ToolCall | None = None,
        grandchild_arguments: dict[str, object] | None = None,
        grandchild_tool_call: ToolCall | None = None,
    ) -> None:
        self.child_arguments = child_arguments
        self.child_tool_call = child_tool_call
        self.grandchild_arguments = grandchild_arguments
        self.grandchild_tool_call = grandchild_tool_call

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = _last_user_prompt(messages)
        pending = _pending_tool_result(messages)
        child_prompt = self.child_arguments["prompt"]
        grandchild_prompt = (
            self.grandchild_arguments["prompt"]
            if self.grandchild_arguments is not None
            else None
        )
        if last_user == "start":
            blocks = [ToolUseContent(ToolCall("child-1", "agent", dict(self.child_arguments)))]
        elif grandchild_prompt is not None and last_user == grandchild_prompt:
            if pending is None and self.grandchild_tool_call is not None:
                blocks = [ToolUseContent(self.grandchild_tool_call)]
            else:
                blocks = [TextContent("grandchild done")]
        elif last_user == child_prompt:
            if pending is not None:
                blocks = [TextContent("child done")]
            elif self.grandchild_arguments is not None:
                blocks = [
                    ToolUseContent(
                        ToolCall("grandchild-1", "agent", dict(self.grandchild_arguments))
                    )
                ]
            elif self.child_tool_call is not None:
                blocks = [ToolUseContent(self.child_tool_call)]
            else:
                blocks = [TextContent("child done")]
        else:
            blocks = [TextContent("done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _first_tool_result(store: ConversationStore) -> ToolResult:
    return next(
        message.tool_result for message in store.messages() if message.tool_result
    )


@pytest.mark.asyncio
async def test_child_cwd_replacement_fails_closed_for_all_tools(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    child_cwd = tmp_path / "child"
    parent_cwd.mkdir()
    child_cwd.mkdir()
    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    parent_registry = ToolRegistry(parent_cwd, skill_catalog=SkillCatalog.empty())
    child_registry = parent_registry.clone_for_session(child_store, cwd=child_cwd)

    child_cwd.rename(tmp_path / "child-original")
    child_cwd.mkdir()

    calls = (
        ToolCall("replaced-read", "read", {"path": "target.txt"}),
        ToolCall(
            "replaced-write",
            "write",
            {"path": "target.txt", "content": "replacement"},
        ),
        ToolCall("replaced-bash", "bash", {"command": "touch bash-marker"}),
        ToolCall(
            "replaced-background",
            "run_background",
            {"command": "touch background-marker"},
        ),
    )
    for call in calls:
        result = await child_registry.execute(call)
        assert result["isError"] is True
        assert "session cwd was replaced" in result["content"][0]["text"]

    fake_loop = SimpleNamespace(
        store=parent_store,
        context_assembler=SimpleNamespace(system_prompt="parent prompt"),
        active_home=str(tmp_path / "home"),
        tool_registry=parent_registry,
        root_project_id=None,
    )
    with pytest.raises(ValueError, match="session cwd was replaced"):
        _child_base_system_prompt(fake_loop, str(child_cwd), child_registry)

    assert not (child_cwd / "target.txt").exists()
    assert not (child_cwd / "bash-marker").exists()
    assert not (child_cwd / "background-marker").exists()
    await child_registry.close()
    await parent_registry.close()


@pytest.mark.asyncio
async def test_agent_cwd_sets_child_bash_session_cwd(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "worktree child",
            "cwd": str(worktree),
        },
        child_tool_call=ToolCall(
            "child-bash", "bash", {"command": "touch bash_marker"}
        ),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert (worktree / "bash_marker").exists()
    assert not (parent_dir / "bash_marker").exists()
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_relative_paths_resolve_in_child_cwd(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "worktree child",
            "cwd": str(worktree),
        },
        child_tool_call=ToolCall(
            "child-write",
            "write",
            {"path": "notes.txt", "content": "from child"},
        ),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert (worktree / "notes.txt").read_text(encoding="utf-8") == "from child"
    assert not (parent_dir / "notes.txt").exists()
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_rejects_missing_directory(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    missing = tmp_path / "does-not-exist"
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "worktree child",
            "cwd": str(missing),
        },
        child_tool_call=ToolCall("child-bash", "bash", {"command": "true"}),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    result = _first_tool_result(store)
    assert result.is_error is True
    assert "existing directory" in result.content
    assert str(missing) in result.content
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_defaults_to_parent_cwd(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={"prompt": "child-work", "description": "inherit child"},
        child_tool_call=ToolCall(
            "child-bash", "bash", {"command": "touch default_marker"}
        ),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert (parent_dir / "default_marker").exists()
    await loop.close()


@pytest.mark.asyncio
async def test_grandchild_inherits_child_cwd(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "worktree child",
            "cwd": str(worktree),
        },
        grandchild_arguments={
            "prompt": "gc-work",
            "description": "grandchild",
        },
        grandchild_tool_call=ToolCall(
            "gc-bash", "bash", {"command": "touch gc_marker"}
        ),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert (worktree / "gc_marker").exists()
    assert not (parent_dir / "gc_marker").exists()
    await loop.close()


class ChildToolSequenceBackend(CompletionBackend):
    """Drive one child that issues a fixed sequence of tool calls, one per turn."""

    def __init__(
        self,
        *,
        child_arguments: dict[str, object],
        child_tool_calls: Sequence[ToolCall],
    ) -> None:
        self.child_arguments = child_arguments
        self.child_tool_calls = list(child_tool_calls)
        self.child_prompt = child_arguments["prompt"]

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = _last_user_prompt(messages)
        if last_user == "start":
            blocks = [
                ToolUseContent(ToolCall("child-1", "agent", dict(self.child_arguments)))
            ]
        elif last_user == self.child_prompt:
            done = sum(
                1 for message in messages if message.role is MessageRole.TOOL_RESULT
            )
            if done < len(self.child_tool_calls):
                blocks = [ToolUseContent(self.child_tool_calls[done])]
            else:
                blocks = [TextContent("child done")]
        else:
            blocks = [TextContent("done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _child_store_for(store: ConversationStore, session_id: str = "1") -> ConversationStore:
    return ConversationStore(store.session_dir / "agents", session_id=session_id)


@pytest.mark.asyncio
async def test_agent_cwd_relative_to_parent_cwd(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    (parent_dir / "worktree").mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "relative cwd child",
            "cwd": "worktree",
        },
        child_tool_call=ToolCall(
            "child-bash", "bash", {"command": "touch relative_marker"}
        ),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert (parent_dir / "worktree" / "relative_marker").exists()
    assert not (parent_dir / "relative_marker").exists()
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_applies_to_read_and_edit(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "data.txt").write_text("OLD-CONTENT", encoding="utf-8")
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildToolSequenceBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "read/edit child",
            "cwd": str(worktree),
        },
        child_tool_calls=[
            ToolCall("child-read", "read", {"path": "data.txt"}),
            ToolCall(
                "child-edit",
                "edit",
                {
                    "path": "data.txt",
                    "old_string": "OLD-CONTENT",
                    "new_string": "NEW-CONTENT",
                },
            ),
        ],
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_messages = _child_store_for(store).messages()
    results = {
        message.tool_result.tool_call_id: message.tool_result
        for message in child_messages
        if message.tool_result is not None
    }
    assert not results["child-read"].is_error
    assert "OLD-CONTENT" in results["child-read"].content
    assert not results["child-edit"].is_error
    assert (worktree / "data.txt").read_text(encoding="utf-8") == "NEW-CONTENT"
    assert not (parent_dir / "data.txt").exists()
    await loop.close()


class ChildRunBackgroundBackend(CompletionBackend):
    """Drive a child that starts a run_background task, then waits for it.

    Waiting via task_output before the child finishes keeps the assertion off
    the child-teardown race: the file exists by the time the child replies.
    """

    def __init__(
        self,
        *,
        cwd: str,
        command: str,
        background_child: bool = False,
    ) -> None:
        self.cwd = cwd
        self.command = command
        self.background_child = background_child
        self.child_prompt = "child-work"
        self.task_output_requested = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = _last_user_prompt(messages)
        if last_user == "start":
            blocks = [
                ToolUseContent(
                    ToolCall(
                        "child-1",
                        "agent",
                        {
                            "prompt": self.child_prompt,
                            "description": "run_background child",
                            "cwd": self.cwd,
                            "background": self.background_child,
                        },
                    )
                )
            ]
        elif last_user == self.child_prompt:
            results = {
                message.tool_result.tool_call_id: message.tool_result
                for message in messages
                if message.tool_result is not None
            }
            if "child-bg" not in results:
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "child-bg", "run_background", {"command": self.command}
                        )
                    )
                ]
            elif "child-wait" not in results:
                task_id = re.search(
                    r"task-[0-9a-f]+", results["child-bg"].content
                ).group(0)
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "child-wait",
                            "task_output",
                            {"task_id": task_id, "wait_seconds": 5},
                        )
                    )
                ]
                self.task_output_requested.set()
            else:
                blocks = [TextContent("child done")]
        else:
            blocks = [TextContent("done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


@pytest.mark.asyncio
async def test_agent_cwd_applies_to_child_run_background(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildRunBackgroundBackend(cwd=str(worktree), command="touch bg_marker")
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert (worktree / "bg_marker").exists()
    assert not (parent_dir / "bg_marker").exists()
    await loop.close()


@pytest.mark.asyncio
async def test_parent_abort_cancels_child_waiting_on_task_output_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_errors: list[BaseException] = []
    child_abort_calls = 0
    child_registry: ToolRegistry | None = None
    child_execute_entered = asyncio.Event()
    release_child_execute = asyncio.Event()
    child_abort_entered = asyncio.Event()
    original_execute = ToolRegistry.execute
    original_abort_approval = ToolRegistry.abort_approval

    async def pause_child_before_execute(
        registry: ToolRegistry,
        tool_call: ToolCall,
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal child_registry
        if tool_call.name == "task_output":
            assert isinstance(registry.approval_policy, ChildApprovalPolicy)
            child_registry = registry
            child_execute_entered.set()
            await release_child_execute.wait()
        return await original_execute(registry, tool_call, **kwargs)

    def record_child_abort_error(
        registry: ToolRegistry,
        tool_call: ToolCall,
    ) -> ApprovalDecision | None:
        nonlocal child_abort_calls
        is_child = registry is child_registry
        if is_child:
            child_abort_calls += 1
            child_abort_entered.set()
        try:
            return original_abort_approval(registry, tool_call)
        except BaseException as exc:
            if is_child:
                abort_errors.append(exc)
            raise

    monkeypatch.setattr(ToolRegistry, "execute", pause_child_before_execute)
    monkeypatch.setattr(ToolRegistry, "abort_approval", record_child_abort_error)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    store = ConversationStore(tmp_path / "sessions", cwd=worktree)
    backend = ChildRunBackgroundBackend(
        cwd=str(worktree),
        command=_python("import time; time.sleep(30)"),
        background_child=True,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        approval_policy=ApprovalPolicy(
            store=store,
            always_allow={"agent", "run_background", "task_output"},
        ),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(child_execute_entered.wait(), timeout=5)
    child_store = _child_store_for(store)
    for _ in range(100):
        if any(
            message.role is MessageRole.ASSISTANT
            and any(
                isinstance(block, ToolUseContent)
                and block.tool_call.name == "task_output"
                for block in message.content
            )
            for message in child_store.messages()
        ):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("child did not start task_output")

    loop.abort()
    release_child_execute.set()
    await asyncio.wait_for(child_abort_entered.wait(), timeout=5)
    assert child_abort_calls > 0
    assert abort_errors == []
    notification = await _wait_for_notification(store, "canceled")

    assert notification.data["killed_task_count"] == 1
    assert "abort_or_winner" not in notification.data["text"]
    refreshed_child = ConversationStore(
        child_store.root_dir,
        session_id=child_store.session_id,
    )
    assert refreshed_child.agent_lifecycle()["state"] == "canceled"
    await loop.close()


@pytest.mark.asyncio
async def test_child_generator_close_after_allowed_edit_does_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    parent_store = ConversationStore(tmp_path / "parent", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store, always_allow={"edit"})
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "edit child",
        "child-edit",
        child_cwd=tmp_path,
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)
    abort_errors: list[BaseException] = []
    original_abort_approval = registry.abort_approval

    def record_abort_error(tool_call: ToolCall) -> ApprovalDecision | None:
        try:
            return original_abort_approval(tool_call)
        except BaseException as exc:
            abort_errors.append(exc)
            raise

    monkeypatch.setattr(registry, "abort_approval", record_abort_error)
    call = ToolCall(
        "child-edit-call",
        "edit",
        {"path": str(target), "old_string": "before", "new_string": "after"},
    )
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call])]),
        child_store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    turn = loop.run_turn("edit the file", origin=MessageOrigin.USER)

    async for event in turn:
        if (
            event.type is StreamEventType.TOOL_EXECUTION_END
            and event.tool_result is not None
            and event.tool_result.tool_call_id == call.id
        ):
            break
    await turn.aclose()

    assert target.read_text(encoding="utf-8") == "after"
    results = [
        message.tool_result
        for message in child_store.messages()
        if message.tool_result is not None
        and message.tool_result.tool_call_id == call.id
    ]
    assert len(results) == 1
    assert results[0].is_error is False
    assert abort_errors == []
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_with_explore_preset(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "only.txt").write_text("EXPLORE-CWD-CONTENT", encoding="utf-8")
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildToolSequenceBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "explore child",
            "agent_type": "explore",
            "cwd": str(worktree),
        },
        child_tool_calls=[ToolCall("child-read", "read", {"path": "only.txt"})],
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_messages = _child_store_for(store).messages()
    read_result = next(
        message.tool_result
        for message in child_messages
        if message.tool_result is not None
        and message.tool_result.tool_call_id == "child-read"
    )
    assert not read_result.is_error
    assert "EXPLORE-CWD-CONTENT" in read_result.content
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_rejects_symlinked_directory(tmp_path: Path) -> None:
    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = ChildCwdBackend(
        child_arguments={
            "prompt": "child-work",
            "description": "symlink child",
            "cwd": str(link),
        },
        child_tool_call=ToolCall("child-bash", "bash", {"command": "true"}),
    )
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    result = _first_tool_result(store)
    assert result.is_error is True
    assert "symlink" in result.content
    await loop.close()


@pytest.mark.asyncio
async def test_agent_cwd_uses_child_worktree_agents_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child with an explicit cwd walks AGENTS.md from that cwd, not the parent's."""

    home = tmp_path / "home"
    home.mkdir()
    (home / "AGENTS.md").write_text("HOME-IDENTITY-MARKER", encoding="utf-8")
    monkeypatch.setenv("ZETA_HOME", str(home))

    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    (parent_dir / "AGENTS.md").write_text(
        "PARENT-RULE-repo-specific-content", encoding="utf-8"
    )
    child_dir = tmp_path / "worktree"
    child_dir.mkdir()
    (child_dir / "AGENTS.md").write_text(
        "CHILD-RULE-repo-specific-content", encoding="utf-8"
    )

    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "child-1",
                        "agent",
                        {
                            "prompt": "child-work",
                            "description": "worktree child",
                            "cwd": str(child_dir),
                        },
                    )
                ]
            ),
            ScriptedTurn([TextContent("child done")]),
        ]
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        system_prompt="PARENT-RULE-repo-specific-content",
        skill_catalog=SkillCatalog.empty(),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_system = backend.calls[1][0][0]
    system_text = " ".join(
        block.text
        for block in child_system.content
        if isinstance(block, TextContent)
    )
    assert "CHILD-RULE-repo-specific-content" in system_text
    assert "PARENT-RULE-repo-specific-content" not in system_text
    assert "HOME-IDENTITY-MARKER" in system_text
    await loop.close()


@pytest.mark.asyncio
async def test_child_cwd_context_uses_active_home_and_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit-cwd children use the loop's active home and skill catalog."""

    ambient_home = tmp_path / "ambient-home"
    ambient_home.mkdir()
    (ambient_home / "AGENTS.md").write_text(
        "AMBIENT-HOME-IDENTITY", encoding="utf-8"
    )
    monkeypatch.setenv("ZETA_HOME", str(ambient_home))

    custom_home = tmp_path / "custom-home"
    custom_home.mkdir()
    (custom_home / "AGENTS.md").write_text(
        "CUSTOM-HOME-IDENTITY", encoding="utf-8"
    )
    skill_path = tmp_path / "expected-skill.md"
    skill_path.write_text("skill body", encoding="utf-8")
    catalog = SkillCatalog(
        (
            SkillMeta(
                "expected-skill",
                "EXPECTED-SKILL-DESCRIPTION",
                ["expected"],
                skill_path,
            ),
        )
    )

    parent_dir = tmp_path / "primary"
    parent_dir.mkdir()
    (parent_dir / "AGENTS.md").write_text(
        "PARENT-RULE-repo-specific-content", encoding="utf-8"
    )
    child_dir = tmp_path / "worktree"
    child_dir.mkdir()
    (child_dir / "AGENTS.md").write_text(
        "CHILD-RULE-repo-specific-content", encoding="utf-8"
    )

    store = ConversationStore(tmp_path / "sessions", cwd=parent_dir)
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "child-1",
                        "agent",
                        {
                            "prompt": "child-work",
                            "description": "worktree child",
                            "cwd": str(child_dir),
                        },
                    )
                ]
            ),
            ScriptedTurn([TextContent("child done")]),
        ]
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        system_prompt="PARENT-RULE-repo-specific-content",
        skill_catalog=catalog,
    )
    loop.set_mcp_scope(home=custom_home, project_dir=parent_dir)
    assert loop.active_home == str(custom_home)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_system = backend.calls[1][0][0]
    system_text = " ".join(
        block.text
        for block in child_system.content
        if isinstance(block, TextContent)
    )
    assert "CHILD-RULE-repo-specific-content" in system_text
    assert "PARENT-RULE-repo-specific-content" not in system_text
    assert "CUSTOM-HOME-IDENTITY" in system_text
    assert "AMBIENT-HOME-IDENTITY" not in system_text
    assert "- expected-skill: EXPECTED-SKILL-DESCRIPTION" in system_text
    await loop.close()


def test_agent_status_text_shows_cwd() -> None:
    from zeta.tools.agent import _status_response

    base_row: dict[str, object] = {
        "state": "running",
        "started_at": "2024-01-01T00:00:00+00:00",
        "finished_at": "",
        "elapsed": 1.0,
        "turns_used": 1,
        "tool_calls": 1,
        "current_step": "thinking",
        "depth": 1,
        "agent_type": "general",
        "description": "child",
    }
    with_cwd = {**base_row, "handle": "sess:1", "cwd": "/tmp/worktree"}
    without_cwd = {**base_row, "handle": "sess:2", "cwd": ""}

    shown = _status_response(
        [with_cwd],
        offset=0,
        total=1,
        truncated=False,
        next_offset=None,
        finished_omitted=0,
    )
    assert "cwd: /tmp/worktree" in shown["content"][0]["text"]

    hidden = _status_response(
        [without_cwd],
        offset=0,
        total=1,
        truncated=False,
        next_offset=None,
        finished_omitted=0,
    )
    assert "cwd:" not in hidden["content"][0]["text"]


@pytest.mark.asyncio
async def test_child_cwd_inherits_parent_project_memory_and_tools(
    tmp_path: Path,
) -> None:
    from zeta.project_registry import ProjectRegistry

    parent_cwd = tmp_path / "project-a"
    child_cwd = tmp_path / "project-b"
    parent_cwd.mkdir()
    child_cwd.mkdir()
    home = tmp_path / "zeta-home"
    registry = ProjectRegistry(home / "projects")
    project_a = registry.find_or_create_for_directory(parent_cwd)
    project_b = registry.find_or_create_for_directory(child_cwd)
    registry.update_memory(project_a.project_id, {"state.md": "PROJECT A MEMORY"})
    registry.update_memory(project_b.project_id, {"state.md": "PROJECT B MEMORY"})
    (child_cwd / "AGENTS.md").write_text("CHILD CWD INSTRUCTIONS")

    parent_store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    parent_tools = ToolRegistry(
        parent_cwd,
        project_id=project_a.project_id,
        project_registry=registry,
        skill_catalog=SkillCatalog.empty(),
    )
    child_store = ConversationStore(tmp_path / "children", cwd=child_cwd)
    child_tools = parent_tools.clone_for_session(child_store, cwd=child_cwd)
    loop = SimpleNamespace(
        store=parent_store,
        context_assembler=SimpleNamespace(system_prompt="parent prompt"),
        active_home=str(home),
        tool_registry=parent_tools,
        root_project_id=project_a.project_id,
    )

    prompt = _child_base_system_prompt(loop, str(child_cwd), child_tools)

    assert "PROJECT A MEMORY" in prompt
    assert "PROJECT B MEMORY" not in prompt
    assert "CHILD CWD INSTRUCTIONS" in prompt
    assert child_tools.project_id == project_a.project_id
    assert child_tools.project_id != project_b.project_id
    result = await child_tools.execute(
        ToolCall(
            "child-update",
            "project_update",
            {"name": "state.md", "content": "UPDATED BY CHILD"},
        )
    )
    assert result["isError"] is False
    assert dict(registry.load_memory(project_a.project_id))["state.md"] == (
        "UPDATED BY CHILD"
    )
    assert dict(registry.load_memory(project_b.project_id))["state.md"] == (
        "PROJECT B MEMORY"
    )
    await child_tools.close()
    await parent_tools.close()
    child_store.close()
    parent_store.close()


@pytest.mark.asyncio
async def test_child_assignment_origin_is_agent_prompt(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("child done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assignment = next(
        message for message in child_store.messages() if message.role is MessageRole.USER
    )
    assert assignment.metadata["zeta.origin"] == MessageOrigin.AGENT_PROMPT.value
    await loop.close()


@pytest.mark.asyncio
async def test_run_followup_message_row_origin_is_agent_send(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)
    handle = _run_handle_from_receipt(store)
    marker = store.agent_children()[handle]
    child_path = Path(str(marker["child_session_path"]))
    assert send_to_run(store, handle, "also check the tests") is None
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    followup = next(
        message
        for message in child_store.messages()
        if any(
            isinstance(block, TextContent) and block.text == "also check the tests"
            for block in message.content
        )
    )
    assert followup.metadata["zeta.origin"] == MessageOrigin.AGENT_SEND.value
    await loop.close()
