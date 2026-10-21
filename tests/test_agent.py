import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from io import StringIO
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import zeta.agent.run_consumer as run_consumer_module
import zeta.agent.runner as agent_runner_module
import zeta.runtime.execution as execution_module
import zeta.runtime.loop.agent as agent_loop_module
import zeta.tools.agent_send as agent_send_module
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.agent.background import (
    BackgroundAgentOwner,
    adopt_agent_children,
    finish_background_child,
    recover_agent_children,
)
from zeta.agent.notifications import NotificationWake
from zeta.agent.presets import (
    AGENT_PRESETS,
    GENERAL_PRESET,
    AgentPreset,
)
from zeta.agent.runner import (
    _child_base_system_prompt,
    _compose_child_system_prompt,
)
from zeta.agent.terminal_fence import TerminalTransitionFence
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
from zeta.providers.codex import CodexStreamError
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.tool_schema import canonical_tool_schemas
from zeta.skills import SkillCatalog, SkillMeta
from zeta.skills.agent_catalog import AgentCatalog
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

    async def close_child() -> tuple[str, ...]:
        close_started.set()
        await release_close.wait()
        return ()

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
    def __init__(self, *, background_child: bool = False) -> None:
        self.background_child = background_child
        self.child_nested = False
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()
        self.child_notification_started = asyncio.Event()
        self.release_child_notification = asyncio.Event()

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
        notification = any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        )
        if last_user == "start":
            child = (
                _background_agent_call("child")
                if self.background_child
                else _agent_call("child")
            )
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
        elif self.background_child and notification:
            self.child_notification_started.set()
            await self.release_child_notification.wait()
            blocks = [TextContent("child complete")]
        else:
            blocks = [TextContent("child complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class FinishGateBackend(CompletionBackend):
    def __init__(self, gate_action: str, *, background_child: bool = False) -> None:
        self.gate_action = gate_action
        self.background_child = background_child
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()
        self.gate_seen = asyncio.Event()
        self.child_final_responses: list[str] = []
        self.gate_calls = 0
        self.invalid_handoff_sent = False
        self.handoff_schema_seen = False
        self.handoff_schema_seen_before_gate = False

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
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
        gate = next(
            (
                message
                for message in reversed(messages)
                if message.metadata.get("zeta_event") == "agent_finish_gate"
            ),
            None,
        )
        notification = any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        )
        schema_names = {schema["name"] for schema in tool_schemas}
        if gate is None:
            self.handoff_schema_seen_before_gate |= "agent_handoff" in schema_names
        else:
            self.handoff_schema_seen |= "agent_handoff" in schema_names
        canceled = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "cancel-grandchild"
            for message in messages
        )
        root_child_done = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "child"
            for message in messages
        )
        grandchild_started = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "grandchild"
            for message in messages
        )
        if root_child_done:
            blocks = [TextContent("root complete")]
        elif last_user == "start":
            child = (
                _background_agent_call("child")
                if self.background_child
                else _agent_call("child")
            )
            child.arguments["prompt"] = "child prompt"
            blocks = [ToolUseContent(child)]
        elif last_user == "grandchild prompt":
            self.grandchild_started.set()
            await self.release_grandchild.wait()
            blocks = [TextContent("evidence ready")]
        elif grandchild_started and gate is None and not notification:
            blocks = [TextContent("premature child answer")]
        elif last_user == "child prompt" and gate is None:
            grandchild = _background_agent_call("grandchild")
            grandchild.arguments["prompt"] = "grandchild prompt"
            grandchild.arguments["description"] = "collect evidence"
            blocks = [ToolUseContent(grandchild)]
        elif canceled:
            blocks = [TextContent("child canceled unused work and finished")]
        elif gate is not None and not notification:
            self.gate_calls += 1
            self.gate_seen.set()
            gate_text = next(
                block.text for block in gate.content if isinstance(block, TextContent)
            )
            if self.gate_action == "cancel":
                handle = re.search(r"([\w-]+(?::\d+)+): collect evidence", gate_text)
                assert handle is not None
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "cancel-grandchild",
                            "agent_cancel",
                            {"handle": handle.group(1)},
                        )
                    )
                ]
            elif self.gate_action in {"handoff", "invalid_handoff"}:
                arguments = {
                    "reason": "This agent lacks the remaining context.",
                    "outputs": "The adopted child completion receipt in the parent session.",
                }
                if self.gate_action == "invalid_handoff" and not self.invalid_handoff_sent:
                    self.invalid_handoff_sent = True
                    arguments.pop("outputs")
                call_id = (
                    "handoff-valid"
                    if self.invalid_handoff_sent and "outputs" in arguments
                    else "handoff-grandchild"
                )
                blocks = [
                    ToolUseContent(ToolCall(call_id, "agent_handoff", arguments))
                ]
            elif self.gate_action == "text_handoff":
                blocks = [TextContent("HAND OFF: use the parent receipt")]
            elif self.gate_action == "poll":
                blocks = [
                    ToolUseContent(
                        ToolCall(f"poll-{self.gate_calls}", "agent_status", {})
                    )
                ]
            elif self.gate_action == "stall":
                await asyncio.Event().wait()
            else:
                blocks = [TextContent("I will wait for the evidence.")]
        elif notification:
            blocks = [TextContent("merged result: evidence ready")]
        else:
            blocks = [TextContent("root complete")]
        for block in blocks:
            if isinstance(block, TextContent) and (
                "child" in block.text or "merged result" in block.text
            ):
                self.child_final_responses.append(block.text)
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class FollowUpFinishGateBackend(CompletionBackend):
    def __init__(self) -> None:
        self.gate_started = asyncio.Event()
        self.release_gate_turn = asyncio.Event()
        self.follow_up_processed = asyncio.Event()
        self.follow_up_prompts: list[str] = []
        self.gate_turns = 0

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
        gate = next(
            (
                message
                for message in reversed(messages)
                if message.metadata.get("zeta_event") == "agent_finish_gate"
            ),
            None,
        )
        child_started = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "child"
            for message in messages
        )
        grandchild_started = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "grandchild"
            for message in messages
        )
        if last_user == "start" and not child_started:
            child = _background_agent_call("child")
            child.arguments.update(prompt="child prompt", description="worker")
            blocks = [ToolUseContent(child)]
        elif last_user == "child prompt" and not grandchild_started:
            grandchild = _background_agent_call("grandchild")
            grandchild.arguments.update(
                prompt="grandchild prompt", description="evidence"
            )
            blocks = [ToolUseContent(grandchild)]
        elif last_user == "grandchild prompt":
            await asyncio.Event().wait()
        elif last_user.startswith("follow-up"):
            self.follow_up_prompts.append(last_user)
            self.follow_up_processed.set()
            blocks = [TextContent("processed follow-up")]
        elif grandchild_started and gate is None:
            blocks = [TextContent("child ready to finish")]
        elif gate is not None:
            self.gate_turns += 1
            self.gate_started.set()
            if self.gate_turns == 1:
                await self.release_gate_turn.wait()
                blocks = [TextContent("I will wait for my child.")]
            else:
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "handoff-grandchild",
                            "agent_handoff",
                            {
                                "reason": "The follow-up is complete.",
                                "outputs": "The grandchild result appears in the root.",
                            },
                        )
                    )
                ]
        elif child_started:
            blocks = [TextContent("root complete")]
        else:
            blocks = [TextContent("unexpected turn")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ConversationFinishGateBackend(CompletionBackend):
    def __init__(self) -> None:
        self.gate_seen = asyncio.Event()
        self.question_published = asyncio.Event()
        self.release_grandchild = asyncio.Event()
        self.notification_turns = 0
        self.gate_turns = 0
        self.follow_up_seen = False

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
        gate_messages = [
            message
            for message in messages
            if message.metadata.get("zeta_event") == "agent_finish_gate"
        ]
        gate = gate_messages[-1] if gate_messages else None
        notification = any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        )
        child_started = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "child"
            for message in messages
        )
        grandchild_started = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "grandchild"
            for message in messages
        )
        question_asked = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "ask-parent"
            for message in messages
        )
        handoff_done = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id
            in {"handoff-grandchild", "handoff-after-follow-up"}
            for message in messages
        )
        if last_user == "start" and not child_started:
            child = _background_agent_call("child")
            child.arguments.update(prompt="child prompt", description="worker")
            blocks = [ToolUseContent(child)]
        elif last_user == "child prompt" and not grandchild_started:
            grandchild = _background_agent_call("grandchild")
            grandchild.arguments.update(
                prompt="grandchild prompt", description="evidence"
            )
            blocks = [ToolUseContent(grandchild)]
        elif last_user == "grandchild prompt" and not question_asked:
            await self.gate_seen.wait()
            blocks = [
                ToolUseContent(
                    ToolCall(
                        "ask-parent",
                        "ask_parent",
                        {"question": "Which evidence should I collect?"},
                    )
                )
            ]
        elif question_asked:
            self.question_published.set()
            await self.release_grandchild.wait()
            blocks = [TextContent("grandchild complete")]
        elif grandchild_started and gate is None and not notification:
            blocks = [TextContent("child ready to finish")]
        elif last_user == "queued follow-up":
            self.follow_up_seen = True
            blocks = [TextContent("child processed follow-up")]
        elif notification and len(gate_messages) == 1 and not handoff_done:
            self.notification_turns += 1
            blocks = [
                ToolUseContent(
                    ToolCall(
                        "handoff-grandchild",
                        "agent_handoff",
                        {
                            "reason": "The parent must answer the open question.",
                            "outputs": "The grandchild completion appears in the root.",
                        },
                    )
                )
            ]
        elif last_user.startswith("You have "):
            self.gate_turns += 1
            self.gate_seen.set()
            if self.gate_turns == 1:
                blocks = [TextContent("I will wait for my child.")]
            else:
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "handoff-after-follow-up",
                            "agent_handoff",
                            {
                                "reason": "The queued follow-up is complete.",
                                "outputs": "The grandchild completion appears in root.",
                            },
                        )
                    )
                ]
        elif handoff_done:
            blocks = [TextContent("handoff complete")]
        elif child_started:
            blocks = [TextContent("root complete")]
        else:
            blocks = [TextContent("unexpected turn")]
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
    # The root's notification for the child can land before the child's own
    # record of its grandchild; both are written by different tasks.
    for _ in range(500):
        child_store.refresh()
        grandchild_store.refresh()
        child_notifications = child_store.agent_notifications(pending_only=False)
        if child_notifications and grandchild_store.agent_canceled() is not None:
            break
        await asyncio.sleep(0.01)
    assert child_notifications[0].data["status"] == "canceled"
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
async def test_child_finish_gate_waits_for_grandchild_and_merges_result(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("wait")
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    assert not task.done()
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    gate_message = next(
        message
        for message in child_store.messages()
        if message.metadata.get("zeta_event") == "agent_finish_gate"
    )
    assert gate_message.metadata["zeta.origin"] == MessageOrigin.HARNESS_NUDGE.value
    assert "collect evidence" in next(
        block.text for block in gate_message.content if isinstance(block, TextContent)
    )

    child_store.close()
    backend.release_grandchild.set()
    await asyncio.wait_for(task, timeout=2)

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"].startswith("merged result: evidence ready")
    assert backend.child_final_responses.index("premature child answer") < (
        backend.child_final_responses.index("merged result: evidence ready")
    )
    assert not store.agent_children()
    child_store.close()
    await loop.close()


@pytest.mark.asyncio
async def test_nested_question_wakes_finish_gate_once_then_delivers_follow_up(
    tmp_path: Path,
) -> None:
    backend = ConversationFinishGateBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    await asyncio.wait_for(backend.question_published.wait(), timeout=2)
    child_handle = max(store.agent_children(), key=lambda handle: handle.count(":"))
    assert send_to_run(
        store,
        child_handle,
        "queued follow-up",
        channel=loop._background_owner.conversation_channel,
    ) is None

    await _wait_for_notification(store, "completed")
    assert backend.notification_turns == 1
    assert backend.follow_up_seen
    assert backend.gate_turns == 2

    backend.release_grandchild.set()
    await loop._background_owner.wait()
    await loop.close()


@pytest.mark.asyncio
async def test_follow_up_wakes_worker_waiting_in_finish_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FollowUpFinishGateBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.gate_started.wait(), timeout=10)
    waiting_for_follow_up = asyncio.Event()
    channel = loop._background_owner.conversation_channel
    original_wait = channel.wait

    async def wait_for_follow_up(instance_id: str) -> None:
        waiting_for_follow_up.set()
        await original_wait(instance_id)

    monkeypatch.setattr(channel, "wait", wait_for_follow_up)
    backend.release_gate_turn.set()
    await asyncio.wait_for(waiting_for_follow_up.wait(), timeout=10)
    child_handle = next(iter(store.agent_children()))
    assert send_to_run(
        store,
        child_handle,
        "follow-up while waiting",
        channel=channel,
    ) is None

    await asyncio.wait_for(backend.follow_up_processed.wait(), timeout=10)
    assert backend.follow_up_prompts == ["follow-up while waiting"]
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_follow_up_between_gate_turns_is_delivered_once(tmp_path: Path) -> None:
    backend = FollowUpFinishGateBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.gate_started.wait(), timeout=2)
    child_handle = next(iter(store.agent_children()))
    assert send_to_run(
        store,
        child_handle,
        "follow-up between gate turns",
        channel=loop._background_owner.conversation_channel,
    ) is None
    backend.release_gate_turn.set()

    await asyncio.wait_for(backend.follow_up_processed.wait(), timeout=0.1)
    await _wait_for_notification(store, "completed")
    assert backend.follow_up_prompts == ["follow-up between gate turns"]
    await loop.close()


@pytest.mark.asyncio
async def test_pending_follow_up_is_processed_before_finish_gate_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FollowUpFinishGateBackend()
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store, finish_gate_timeout=60)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )

    waiting_for_follow_up = asyncio.Event()
    channel = owner.conversation_channel
    original_wait = channel.wait

    async def wait_for_follow_up(instance_id: str) -> None:
        waiting_for_follow_up.set()
        await original_wait(instance_id)

    monkeypatch.setattr(channel, "wait", wait_for_follow_up)
    try:
        await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
        await asyncio.wait_for(backend.gate_started.wait(), timeout=10)
        backend.release_gate_turn.set()
        # Send only once the child is waiting. Runner speed must not decide
        # whether the follow-up arrives before an artificially tiny deadline.
        await asyncio.wait_for(waiting_for_follow_up.wait(), timeout=10)
        child_handle = next(iter(store.agent_children()))
        assert send_to_run(
            store,
            child_handle,
            "follow-up before bound",
            channel=channel,
        ) is None

        await asyncio.wait_for(backend.follow_up_processed.wait(), timeout=10)
        assert backend.follow_up_prompts == ["follow-up before bound"]
        await _wait_for_notification(store, "completed")
    finally:
        backend.release_gate_turn.set()
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_child_finish_gate_cancel_path(tmp_path: Path) -> None:
    backend = FinishGateBackend("cancel")
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=2
    )

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"].startswith("child canceled unused work")
    assert not store.agent_children()
    child_store.close()
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_explicit_handoff_adopts_grandchild(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("handoff")
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=2
    )

    marker = next(iter(store.agent_children().values()))
    assert marker["description"] == "collect evidence"
    assert marker["child_session_path"] == str(
        store.session_dir / "agents" / "1" / "agents" / "1"
    )
    assert backend.handoff_schema_seen
    assert not backend.handoff_schema_seen_before_gate
    receipt = store.tool_result("child")
    assert receipt is not None
    assert "This agent lacks the remaining context." in receipt.content
    assert (
        "The adopted child completion receipt in the parent session." in receipt.content
    )
    backend.release_grandchild.set()
    notification = await _wait_for_notification(store, "completed")
    assert notification.data["text"].startswith("evidence ready")
    await loop.close()


@pytest.mark.asyncio
async def test_background_nested_agent_handoff_finishes_without_channel_hold(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("handoff", background_child=True)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(_wait_for_notification(store, "completed"), timeout=2)
    marker = next(iter(store.agent_children().values()))
    assert marker["description"] == "collect evidence"

    backend.release_grandchild.set()
    await loop._background_owner.wait()
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_handoff_requires_reason_and_outputs(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("invalid_handoff")
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=2
    )

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    invalid = child_store.tool_result("handoff-grandchild")
    assert invalid is not None
    assert invalid.is_error
    assert "outputs" in invalid.content
    assert store.agent_children()
    backend.release_grandchild.set()
    await _wait_for_notification(store, "completed")
    child_store.close()
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_text_handoff_is_not_control_signal(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("text_handoff")
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.wait_for(backend.gate_seen.wait(), timeout=10)
    backend.release_grandchild.set()
    await asyncio.wait_for(task, timeout=10)

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["final_result"].startswith("merged result: evidence ready")
    assert not store.agent_children()
    child_store.close()
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_counts_model_turns_and_adopts_after_bound(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("poll")
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(
        store, finish_gate_max_turns=3, finish_gate_timeout=60
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=2
    )

    assert backend.gate_calls == 3
    assert store.agent_children()
    backend.release_grandchild.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_wait_timeout_adopts(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("wait")
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(
        store, finish_gate_max_turns=8, finish_gate_timeout=0.05
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=1
    )

    assert store.agent_children()
    backend.release_grandchild.set()
    await _wait_for_notification(store, "completed")
    receipt = store.tool_result("child")
    assert receipt is not None
    assert "premature child answer" in receipt.content
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_turn_bound_falls_back_to_adoption(
    tmp_path: Path,
) -> None:
    backend = FinishGateBackend("wait")
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(
        store, finish_gate_max_turns=1, finish_gate_timeout=60
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=2
    )

    marker = next(iter(store.agent_children().values()))
    assert marker["description"] == "collect evidence"
    backend.release_grandchild.set()
    await _wait_for_notification(store, "completed")
    receipt = store.tool_result("child")
    assert receipt is not None
    assert "premature child answer" in receipt.content
    await loop.close()


@pytest.mark.asyncio
async def test_top_level_finish_is_not_gated_by_background_child(tmp_path: Path) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=2
    )

    assert not any(
        message.metadata.get("zeta_event") == "agent_finish_gate"
        for message in store.messages()
    )
    assert loop._background_owner.running
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


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
    # Each concurrent child owns several session descriptors.  Keep a large
    # batch under constrained CI limits without making the assertion depend on
    # failed child setup.
    count = 20 if os.sysconf("SC_OPEN_MAX") < 512 else 44
    backend = ParallelNestedReadBackend(count)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    baseline_fds = len(os.listdir("/dev/fd"))

    events = await asyncio.wait_for(_collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=30)
    assert len(os.listdir("/dev/fd")) <= baseline_fds + 5

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
    assert len(read_lifecycle) == count * 2
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
        if schema["name"] not in {"request_attention", "resolve_attention"}
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


def test_explicit_finish_child_prompt_explains_finish() -> None:
    prompt = _compose_child_system_prompt(
        "parent instructions",
        replace(GENERAL_PRESET, explicit_finish=True),
        explicit_finish=True,
    )

    assert "this run ends only when you call the `finish` tool" in prompt
    assert "Ending a turn without calling `finish` leaves you idle" in prompt


def test_child_without_explicit_finish_has_no_finish_guidance() -> None:
    prompt = _compose_child_system_prompt("parent instructions", GENERAL_PRESET)

    assert "this run ends only when you call the `finish` tool" not in prompt


def test_top_level_session_prompt_has_no_finish_guidance() -> None:
    prompt = "top-level instructions"

    assert "this run ends only when you call the `finish` tool" not in prompt


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
    assert "project_update" not in child_tools
    assert "project_update" not in grandchild_tools
    assert all(
        "project_update" not in {schema["name"] for schema in schemas}
        for _, schemas in backend.calls
    )
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
    backend = ForegroundNestedBackgroundBackend(background_child=True)
    store = ConversationStore(tmp_path)
    root_woke = asyncio.Event()
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    loop.set_background_wake_callback(root_woke.set)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=2)
    backend.release_grandchild.set()
    await asyncio.wait_for(backend.child_notification_started.wait(), timeout=2)

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    root_notifications = store.agent_notifications(pending_only=False)
    assert len(root_notifications) == 1
    assert root_notifications[0].data["display_only"] is True
    assert store.agent_notifications() == []
    assert not root_woke.is_set()
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
    backend.release_child_notification.set()
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


def test_resume_cancels_gated_nested_tree_without_duplicate_notifications(
    tmp_path: Path,
) -> None:
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
    child.append_message(
        with_message_origin(
            Message(
                MessageRole.USER,
                [TextContent("finish gate")],
                metadata={"zeta_event": "agent_finish_gate"},
            ),
            MessageOrigin.HARNESS_NUDGE,
        )
    )
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

    root_notification_count = len(resumed.agent_notifications(pending_only=False))
    child_notification_count = len(
        resumed_child.agent_notifications(pending_only=False)
    )
    restarted = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), restarted, skill_catalog=SkillCatalog.empty())
    restarted_child = ConversationStore(restarted.session_dir / "agents", session_id="1")
    assert len(restarted.agent_notifications(pending_only=False)) == root_notification_count
    assert (
        len(restarted_child.agent_notifications(pending_only=False))
        == child_notification_count
    )


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    lifecycle_started = threading.Event()
    original_start = ConversationStore.start_agent_lifecycle

    def observed_start(self, *args, **kwargs) -> None:
        original_start(self, *args, **kwargs)
        lifecycle_started.set()

    monkeypatch.setattr(ConversationStore, "start_agent_lifecycle", observed_start)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    assert await asyncio.to_thread(lifecycle_started.wait, 10)
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
async def test_child_setup_failure_closes_registry_cwd_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[_agent_call()])]),
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    original_clone = registry.clone_for_session
    clone_fds: list[int] = []
    cwd_fds: list[int] = []
    retained: list[ToolRegistry] = []

    def capture_clone(*args: object, **kwargs: object) -> ToolRegistry:
        child = original_clone(*args, **kwargs)
        retained.append(child)
        cwd_fds.append(child._cwd_fd)
        clone_fds.append(len(os.listdir("/dev/fd")))
        return child

    monkeypatch.setattr(registry, "clone_for_session", capture_clone)
    original_init = agent_loop_module.AgentLoop.__init__

    def fail_child_init(self, *args: object, **kwargs: object) -> None:
        if self is not loop:
            raise RuntimeError("child setup failed")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(agent_loop_module.AgentLoop, "__init__", fail_child_init)
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    assert clone_fds
    with pytest.raises(OSError):
        os.fstat(cwd_fds[0])
    await loop.close()


@pytest.mark.asyncio
async def test_child_setup_cancellation_closes_registry_cwd_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[_agent_call()])]),
        store,
        registry=registry,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    )
    entered = threading.Event()
    release = threading.Event()
    original_prompt = agent_runner_module._child_base_system_prompt

    def block_prompt(*args: object, **kwargs: object) -> str | Message:
        entered.set()
        release.wait(5)
        return original_prompt(*args, **kwargs)

    monkeypatch.setattr(agent_runner_module, "_child_base_system_prompt", block_prompt)
    original_clone = registry.clone_for_session
    clone_fds: list[int] = []
    cwd_fds: list[int] = []
    retained: list[ToolRegistry] = []

    def capture_clone(*args: object, **kwargs: object) -> ToolRegistry:
        child = original_clone(*args, **kwargs)
        retained.append(child)
        cwd_fds.append(child._cwd_fd)
        clone_fds.append(len(os.listdir("/dev/fd")))
        return child

    monkeypatch.setattr(registry, "clone_for_session", capture_clone)
    task = asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert clone_fds
    with pytest.raises(OSError):
        os.fstat(cwd_fds[0])
    await loop.close()


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
        turns_used=2,
    )

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
        turns_used=2,
    )

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


class _FakeCredentials:
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
        lambda provider, **kwargs: _FakeCredentials(
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

    class BootstrapCredentials(_FakeCredentials):
        def bootstrap(self) -> object:
            return _ValidTokens()

    monkeypatch.setattr(
        "zeta.agent.runner.credential_store", lambda provider: BootstrapCredentials(None)
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
        lambda provider, **kwargs: _FakeCredentials(None),
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


class FollowUpFinishBackend(CompletionBackend):
    def __init__(self) -> None:
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()
        self.child_prompts: list[str] = []
        self.finish_attempts = 0
        self.finish_refused_recovery = False
        self.last_finish_prompt = ""
        self.replied_after_finish = False
        self.finish_generated = asyncio.Event()

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
        elif not self.child_prompts:
            self.child_prompts.append(last_user)
            self.child_started.set()
            await self.release_child.wait()
            blocks = [TextContent("first pass done")]
        elif self.finish_attempts and last_user == self.last_finish_prompt and not self.replied_after_finish:
            self.replied_after_finish = True
            blocks = [TextContent("continue")]
        else:
            self.child_prompts.append(last_user)
            self.finish_attempts += 1
            self.last_finish_prompt = last_user
            self.replied_after_finish = False
            self.finish_generated.set()
            blocks = [
                ToolUseContent(
                    ToolCall(
                        f"finish-{self.finish_attempts}",
                        "finish",
                        {"status": "done", "summary": "complete"},
                    )
                )
            ]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class WorkerBackend(RunBackend):
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
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
        if last_user != "start":
            async for event in super().complete(messages, tool_schemas):
                yield event
            return
        call = ToolCall(
            "worker-1",
            "agent",
            {
                "prompt": "work the big task",
                "description": "worker",
                "preset": "worker",
                "background": True,
            },
        )
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=ToolUseContent(call))
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
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

    child_loop = SimpleNamespace(
        explicit_finish=False,
        _delivering_prompt_id=None,
        notification_wake=SimpleNamespace(pending_message=lambda: None),
    )
    result = await agent_runner.consume_run(
        child_loop,
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

    child_loop = SimpleNamespace(
        notification_wake=SimpleNamespace(pending_message=lambda: None),
        _background_owner=SimpleNamespace(
            conversation_channel=SimpleNamespace(has_live_children=lambda _id: False)
        ),
        agent_instance_id=None,
    )
    result = await agent_runner.consume_run(
        child_loop,
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
        accepts_follow_ups=True,
    )

    started = threading.Event()
    release = threading.Event()
    original_send = agent_send_module.send_to_run

    def blocked_send(*args, **kwargs):
        started.set()
        release.wait(timeout=2)
        return original_send(*args, **kwargs)

    monkeypatch.setattr(agent_send_module, "send_to_run", blocked_send)
    registry = ToolRegistry(
        tmp_path, session_store=parent_store, skill_catalog=SkillCatalog.empty()
    )
    registry._agent_owner = BackgroundAgentOwner(parent_store)
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
        accepts_follow_ups=True,
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
    registry._agent_owner = BackgroundAgentOwner(parent_store)
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
        accepts_follow_ups=True,
    )

    registry = ToolRegistry(
        tmp_path, session_store=parent_store, skill_catalog=SkillCatalog.empty()
    )
    registry._agent_owner = BackgroundAgentOwner(parent_store)
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
        task.cancel()
        result = await asyncio.wait_for(task, timeout=2)
    finally:
        lock.__exit__(None, None, None)

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

    error = send_to_run(
        store,
        handle,
        "too late",
        channel=loop._background_owner.conversation_channel,
    )
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
    assert send_to_run(
        store,
        handle,
        "also check the tests",
        channel=loop._background_owner.conversation_channel,
    ) is None

    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    assert backend.child_prompts == ["work the big task", "also check the tests"]
    await loop.close()


@pytest.mark.asyncio
async def test_a_follow_up_reaches_a_worker_at_its_next_turn(tmp_path: Path) -> None:
    backend = WorkerBackend()
    store = ConversationStore(tmp_path)
    worker = AgentPreset(
        name="worker",
        tool_names=None,
        preamble="",
        selection_guidance="implementation worker",
        accepts_follow_ups=True,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog((worker,)),
    )

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)
    handle = _run_handle_from_receipt(store)
    assert send_to_run(
        store,
        handle,
        "also check the tests",
        channel=loop._background_owner.conversation_channel,
    ) is None

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

    channel = BackgroundAgentOwner(store).conversation_channel
    error = send_to_run(store, "run-1", "hello", channel=channel)
    assert error is not None and "no live run" in error

    assert send_to_run(store, "", "hello", channel=channel) == (
        "child_instance_id must be a nonempty string"
    )
    error = send_to_run(store, "run-1", "  ", channel=channel)
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

    error = send_to_run(
        store,
        "sess:1",
        "hello",
        channel=BackgroundAgentOwner(store).conversation_channel,
    )
    assert error is not None
    assert "explore" in error and "agent_send" in error
    assert "reviewers are one-shot" not in error
    assert "profile does not accept follow-ups" in error


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

    worker_call = ToolCall(
        "worker-1",
        "agent",
        {"prompt": "work", "description": "worker", "agent_type": "worker"},
    )
    store.register_agent_child(
        worker_call,
        child_session_path=str(tmp_path / "agents" / "worker"),
        description="worker",
        agent_type="worker",
        background=True,
        accepts_follow_ups=True,
        child_instance_id="sess:worker",
    )

    assert "worker" in commands.slash_runs("")

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)

    listing = commands.slash_runs("")
    assert "long horizon run" in listing
    assert "[run]" in listing
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


async def _wait_completion(
    store: ConversationStore, timeout: float = 10.0, *, containing: str | None = None
) -> object:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        completed = [
            entry
            for entry in store.agent_notifications(pending_only=False)
            if entry.data.get("kind", "agent_completion") == "agent_completion"
            and entry.data.get("status") == "completed"
            and (containing is None or containing in entry.data.get("text", ""))
        ]
        if completed:
            return completed[-1]
        await asyncio.sleep(0.02)
    texts = [
        entry.data.get("text", "")
        for entry in store.agent_notifications(pending_only=False)
    ]
    raise AssertionError(f"no matching completion notification: {texts}")


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


class _TaskFinishGateBackend(CompletionBackend):
    def __init__(
        self,
        command: str,
        action: str = "wait",
        *,
        mixed_child: bool = False,
        gate_delay: bool = False,
        pause_before_gate: bool = False,
    ) -> None:
        self.command = command
        self.action = action
        self.mixed_child = mixed_child
        self.gate_delay = gate_delay
        self.pause_before_gate = pause_before_gate
        self.child_turn_paused = asyncio.Event()
        self.release_child_turn = asyncio.Event()
        self.gate_seen = asyncio.Event()
        self.gate_delay_cancelled = asyncio.Event()
        self.gate_text = ""
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()
        self.task_output_read = asyncio.Event()
        self.follow_up_processed = asyncio.Event()
        self.noise_notification_turns = 0

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
        gate = next(
            (
                message
                for message in reversed(messages)
                if message.metadata.get("zeta_event") == "agent_finish_gate"
            ),
            None,
        )
        notification = any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        )
        latest_notification = next(
            (
                message
                for message in reversed(messages)
                if message.metadata.get("zeta_event") == "agent_notifications"
            ),
            None,
        )
        noise_notification = latest_notification is not None and all(
            item.get("kind") != "task_exited"
            for item in latest_notification.metadata.get("notifications", [])
        )
        started = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "bg-task"
            for message in messages
        )
        output_read = next(
            (
                message.tool_result.content
                for message in reversed(messages)
                if message.tool_result is not None
                and message.tool_result.tool_call_id == "read-task"
            ),
            None,
        )
        killed = any(
            message.tool_result is not None
            and message.tool_result.tool_call_id == "kill-task"
            for message in messages
        )
        if last_user == "grandchild prompt":
            self.grandchild_started.set()
            await self.release_grandchild.wait()
            blocks: list = [TextContent("grandchild evidence")]
        elif last_user == "start":
            blocks = [
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
            if self.mixed_child:
                blocks.append(
                    ToolUseContent(
                        ToolCall(
                            "grandchild",
                            "agent",
                            {
                                "prompt": "grandchild prompt",
                                "description": "collect mixed evidence",
                                "background": True,
                            },
                        )
                    )
                )
        elif last_user == "follow-up while task runs":
            self.follow_up_processed.set()
            blocks = [TextContent("follow-up handled")]
        elif gate is not None and output_read is not None:
            self.task_output_read.set()
            blocks = [TextContent(f"task result: {output_read}")]
        elif gate is not None and notification and noise_notification:
            self.noise_notification_turns += 1
            blocks = [TextContent("WAIT")]
        elif gate is not None and notification and self.action == "wait":
            task_id = re.search(r"task-[0-9a-f]+", self.gate_text)
            assert task_id is not None
            blocks = [
                ToolUseContent(
                    ToolCall("read-task", "task_output", {"task_id": task_id.group()})
                )
            ]
        elif gate is not None and notification and killed:
            blocks = [TextContent("child stopped task and finished")]
        elif gate is not None:
            self.gate_text = next(
                block.text for block in gate.content if isinstance(block, TextContent)
            )
            task_id = re.search(r"task-[0-9a-f]+", self.gate_text)
            if task_id is not None:
                self.gate_seen.set()
                if self.gate_delay:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.gate_delay_cancelled.set()
                        raise
            if self.action == "kill":
                assert task_id is not None
                blocks = [
                    ToolUseContent(
                        ToolCall("kill-task", "task_kill", {"task_id": task_id.group()})
                    )
                ]
                if self.mixed_child:
                    handle = re.search(
                        r"([\w-]+(?::\d+)+): collect mixed evidence", self.gate_text
                    )
                    assert handle is not None
                    blocks.append(
                        ToolUseContent(
                            ToolCall(
                                "cancel-grandchild",
                                "agent_cancel",
                                {"handle": handle.group(1)},
                            )
                        )
                    )
            elif self.action == "handoff":
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "handoff-task",
                            "agent_handoff",
                            {
                                "reason": "CI can finish outside this worker.",
                                "outputs": "The task output remains in the child task archive.",
                            },
                        )
                    )
                ]
            else:
                blocks = [TextContent("WAIT")]
        elif started:
            if self.pause_before_gate:
                self.child_turn_paused.set()
                await self.release_child_turn.wait()
            blocks = [TextContent("premature child answer")]
        else:
            blocks = [TextContent("root complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


async def _start_task_gate(
    tmp_path: Path,
    backend: CompletionBackend,
    *,
    owner: BackgroundAgentOwner | None = None,
) -> tuple[ConversationStore, AgentLoop, asyncio.Task[list[StreamEvent]]]:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    run = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )
    return store, loop, run


@pytest.mark.asyncio
async def test_finish_gate_wait_outlives_default_bound_for_running_task(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    command = (
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done; "
        "echo task-evidence"
    )
    backend = _TaskFinishGateBackend(command, gate_delay=True)
    store = ConversationStore(tmp_path / "session")
    owner = BackgroundAgentOwner(store, finish_gate_timeout=0.05)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    run = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )

    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    await asyncio.wait_for(backend.gate_delay_cancelled.wait(), timeout=2)
    release.touch()
    await asyncio.wait_for(run, timeout=2)
    notification = await _wait_completion(store)
    assert "task-evidence" in notification.data["text"]
    assert notification.data.get("killed_task_ids") is None
    await loop.close()


@pytest.mark.asyncio
async def test_finish_gate_task_with_pending_subagent_keeps_task_past_model_bound(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    command = (
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done; "
        "echo task-evidence"
    )
    backend = _TaskFinishGateBackend(command, mixed_child=True, gate_delay=True)
    store = ConversationStore(tmp_path / "session")
    owner = BackgroundAgentOwner(store, finish_gate_timeout=0.05)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    run = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )

    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    await asyncio.wait_for(backend.gate_delay_cancelled.wait(), timeout=2)
    release.touch()
    await asyncio.wait_for(backend.task_output_read.wait(), timeout=10)
    notification = await _wait_completion(store, containing="task-evidence")
    backend.release_grandchild.set()
    await asyncio.wait_for(run, timeout=2)
    assert "task-evidence" in notification.data["text"]
    assert notification.data.get("killed_task_ids") is None
    await loop.close()


@pytest.mark.asyncio
async def test_finish_gate_task_deadline_starts_when_gate_starts(
    tmp_path: Path,
) -> None:
    backend = _TaskFinishGateBackend(
        _python("import time; time.sleep(30)"), pause_before_gate=True
    )
    store = ConversationStore(tmp_path / "session")
    owner = BackgroundAgentOwner(store, finish_gate_timeout=0.5)
    owner.finish_gate_task_timeout = 5
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    run = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )

    await asyncio.wait_for(backend.child_turn_paused.wait(), timeout=2)
    owner.finish_gate_task_timeout = 0.08
    backend.release_child_turn.set()
    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    await asyncio.wait_for(run, timeout=2)
    notification = await _wait_completion(store)
    assert "finish-gate task safety bound expired" in notification.data["text"]
    assert notification.data.get("killed_task_ids")
    await loop.close()


@pytest.mark.asyncio
async def test_finish_gate_follow_up_keeps_original_task_deadline(
    tmp_path: Path,
) -> None:
    backend = _TaskFinishGateBackend(_python("import time; time.sleep(30)"))
    store = ConversationStore(tmp_path / "session")
    owner = BackgroundAgentOwner(
        store,
        finish_gate_max_turns=100,
        finish_gate_timeout=0.05,
        finish_gate_task_timeout=0.15,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    asyncio.create_task(_collect(loop.run_turn("start", origin=MessageOrigin.USER)))

    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    owner.finish_gate_task_timeout = 1
    child_handle = next(iter(store.agent_children()))
    assert send_to_run(
        store,
        child_handle,
        "follow-up while task runs",
        channel=owner.conversation_channel,
    ) is None

    await asyncio.wait_for(backend.follow_up_processed.wait(), timeout=2)
    notification = await asyncio.wait_for(_wait_completion(store), timeout=0.4)
    assert "finish-gate task safety bound expired" in notification.data["text"]
    assert notification.data.get("killed_task_ids")
    await loop.close()


@pytest.mark.asyncio
async def test_finish_gate_turn_cap_does_not_kill_running_task(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    command = (
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done; "
        "echo task-evidence"
    )
    backend = _TaskFinishGateBackend(command)
    store = ConversationStore(tmp_path / "session")
    owner = BackgroundAgentOwner(
        store,
        finish_gate_max_turns=2,
        finish_gate_timeout=0.05,
        finish_gate_task_timeout=0.8,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    run = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )

    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    child_handle, marker = next(iter(store.agent_children().items()))
    child_path = Path(marker["child_session_path"])
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    for index in range(3):
        child_store.append_agent_notification(
            f"noise-{index}",
            child_session_path=f"/tmp/noise-{index}",
            description="noise notification",
            status="completed",
            text=f"noise notification {index}",
        )
        owner.conversation_channel.wake(child_handle)
        await asyncio.sleep(0.01)
        deadline = asyncio.get_running_loop().time() + 0.2
        while (
            backend.noise_notification_turns < min(index + 1, 1)
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0)
        assert not [
            entry
            for entry in store.agent_notifications(pending_only=False)
            if entry.data.get("status") == "completed"
        ]

    release.touch()
    await asyncio.wait_for(run, timeout=2)
    notification = await asyncio.wait_for(_wait_completion(store), timeout=2)
    assert "task-evidence" in notification.data["text"]
    assert notification.data.get("killed_task_ids") is None
    await loop.close()


@pytest.mark.asyncio
async def test_finish_gate_task_without_exit_reports_safety_bound(
    tmp_path: Path,
) -> None:
    backend = _TaskFinishGateBackend(_python("import time; time.sleep(30)"))
    store = ConversationStore(tmp_path / "session")
    owner = BackgroundAgentOwner(store, finish_gate_timeout=0.02)
    owner.finish_gate_task_timeout = 0.05
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    run = asyncio.create_task(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    )

    await asyncio.wait_for(run, timeout=2)
    notification = await _wait_completion(store)
    text = notification.data["text"]
    assert "finish-gate task safety bound expired" in text
    assert "background tasks killed on child completion" in text
    assert notification.data.get("killed_task_ids")
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_lists_task_and_wakes_on_exit(tmp_path: Path) -> None:
    release = tmp_path / "release"
    command = (
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.01; done; "
        "echo task-evidence"
    )
    backend = _TaskFinishGateBackend(command)
    store, loop, run = await _start_task_gate(tmp_path / "session", backend)

    await asyncio.wait_for(backend.gate_seen.wait(), timeout=2)
    assert "background tasks" in backend.gate_text
    assert "task-" in backend.gate_text
    assert "while [ ! -e" in backend.gate_text
    assert not [
        entry
        for entry in store.agent_notifications(pending_only=False)
        if entry.data.get("status") == "completed"
    ]
    release.touch()

    await asyncio.wait_for(run, timeout=5)
    notification = await _wait_completion(store)
    assert "task result:" in notification.data["text"]
    assert "task-evidence" in notification.data["text"]
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_task_kill_path(tmp_path: Path) -> None:
    backend = _TaskFinishGateBackend(
        _python("import time; time.sleep(30)"), action="kill"
    )
    store, loop, run = await _start_task_gate(tmp_path, backend)

    await asyncio.wait_for(run, timeout=5)
    notification = await _wait_completion(store)
    assert notification.data["text"].startswith("child stopped task and finished")
    assert notification.data.get("killed_task_ids") is None
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_lists_mixed_child_and_task(tmp_path: Path) -> None:
    backend = _TaskFinishGateBackend(
        _python("import time; time.sleep(30)"), action="kill", mixed_child=True
    )
    store, loop, run = await _start_task_gate(tmp_path, backend)

    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=2)
    await asyncio.wait_for(run, timeout=5)
    notification = await _wait_completion(store)
    assert "background sub-agents" in backend.gate_text
    assert "collect mixed evidence" in backend.gate_text
    assert "background tasks" in backend.gate_text
    assert "task-" in backend.gate_text
    assert notification.data["text"].startswith("child stopped task and finished")
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_task_handoff_kills_task(tmp_path: Path) -> None:
    backend = _TaskFinishGateBackend(
        _python("import time; time.sleep(30)"), action="handoff"
    )
    store, loop, run = await _start_task_gate(tmp_path, backend)

    await asyncio.wait_for(run, timeout=5)
    notification = await _wait_completion(store)
    assert "Handed off running background work" in notification.data["text"]
    assert "CI can finish outside this worker." in notification.data["text"]
    assert "background tasks killed on child completion" in notification.data["text"]
    assert notification.data["killed_task_count"] == 1
    await loop.close()


@pytest.mark.asyncio
async def test_child_finish_gate_task_turn_bound_defers_to_task_safety(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(
        store,
        finish_gate_max_turns=1,
        finish_gate_timeout=60,
        finish_gate_task_timeout=0.05,
    )
    backend = _TaskFinishGateBackend(_python("import time; time.sleep(30)"))
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )

    await asyncio.wait_for(
        _collect(loop.run_turn("start", origin=MessageOrigin.USER)), timeout=5
    )
    notification = await _wait_completion(store)
    assert backend.gate_seen.is_set()
    assert "finish-gate task safety bound expired" in notification.data["text"]
    assert "background tasks killed on child completion" in notification.data["text"]
    await loop.close()


@pytest.mark.asyncio
async def test_child_completion_reports_killed_tasks_in_receipt(tmp_path: Path) -> None:
    # S2: a child that still owns a running task when it completes has the task
    # killed without a notification; close() returns the ids and the runner
    # folds them into the child's completion receipt (text + data).
    backend = _TaskOwningChildBackend(_python("import time; time.sleep(30)"))
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(
        store,
        finish_gate_max_turns=1,
        finish_gate_timeout=60,
        finish_gate_task_timeout=0.05,
    )
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
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
@pytest.mark.parametrize(
    ("command", "secret"),
    [
        (
            _python("import time; time.sleep(30)")
            + " 'Authorization: Bearer auth-secret-123'",
            "auth-secret-123",
        ),
        ("TOKEN=token-secret-456 " + _python("import time; time.sleep(30)"), "token-secret-456"),
        (
            _python("import time; time.sleep(30)") + " --password=password-secret-789",
            "password-secret-789",
        ),
        ("<'/tmp/password-secret' cat; sleep 30", "password-secret"),
        ("3<'url-user:url-pass@example.com' cat; sleep 30", "url-user:url-pass@example.com"),
        (
            "<<'heredoc-secret' cat\nbody\nheredoc-secret\nsleep 30",
            "heredoc-secret",
        ),
    ],
)
async def test_killed_task_receipt_never_contains_command_secrets(
    tmp_path: Path, command: str, secret: str
) -> None:
    class FailingBackend(_TaskOwningChildBackend):
        async def complete(self, messages, tool_schemas):
            started = any(
                message.role is MessageRole.TOOL_RESULT
                and any(
                    isinstance(block, TextContent)
                    and "started background task" in block.text
                    for block in message.content
                )
                for message in messages
            )
            if started:
                raise CodexStreamError("permanent", code="model_not_found")
            async for event in super().complete(messages, tool_schemas):
                yield event

    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store, finish_gate_max_turns=1)
    loop = AgentLoop(
        FailingBackend(command),
        store,
        max_turns=1,
        background_owner=owner,
        skill_catalog=SkillCatalog.empty(),
    )
    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        notifications = store.agent_notifications(pending_only=False)
        if notifications and "background tasks killed on child completion" in notifications[-1].data.get("text", ""):
            break
        await asyncio.sleep(0.02)
    assert notifications, "background-task cleanup notification was not recorded"
    notification = notifications[-1]
    killed_task_ids = notification.data.get("killed_task_ids")
    assert isinstance(killed_task_ids, list) and len(killed_task_ids) == 1
    text = notification.data["text"]
    assert "background tasks killed on child completion" in text
    task_id = notification.data["killed_task_ids"][0]
    assert task_id in text
    assert secret not in text
    assert "background command" in text
    assert "archive: " + str(loop.store.session_dir / "agents/1/background-output.archive") in text
    assert "command unavailable" not in text
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
async def test_agent_cwd_sets_child_bash_session_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    loop_thread = threading.get_ident()
    discovery_threads: list[int] = []
    real_discover = agent_runner_module.discover_project

    def discover_project(*args: object, **kwargs: object):
        discovery_threads.append(threading.get_ident())
        return real_discover(*args, **kwargs)

    monkeypatch.setattr(agent_runner_module, "discover_project", discover_project)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    assert discovery_threads and discovery_threads != [loop_thread]
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
    """Drive a child that starts a run_background task, then reads its output.

    Using task_output before the child finishes keeps the assertion off the
    child-teardown race: the file exists by the time the child replies.
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
                            {"task_id": task_id},
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


def test_child_startup_runs_one_git_discovery_in_repo_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core import project_context
    from zeta.project_registry import ProjectRegistry

    home = tmp_path / "zeta-home"
    root = tmp_path / "repo"
    child_cwd = root / "src"
    child_cwd.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("repo", "test", root)
    real_run = subprocess.run
    commands: list[list[str]] = []

    def count_git(command: list[str], *args: object, **kwargs: object):
        commands.append(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(project_context.subprocess, "run", count_git)
    loop = SimpleNamespace(
        store=SimpleNamespace(cwd=root),
        context_assembler=SimpleNamespace(system_prompt="parent"),
        active_home=str(home),
        tool_registry=SimpleNamespace(skill_catalog=SkillCatalog.empty()),
        root_project_id=project.project_id,
    )

    _child_base_system_prompt(loop, str(child_cwd))

    assert len(commands) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("ignored_project", ["project-a", "project-b"])
async def test_child_agent_with_other_project_cwd_loads_its_own_project_file(
    tmp_path: Path, ignored_project: str,
) -> None:
    from zeta.project_registry import ProjectRegistry

    home = tmp_path / "zeta-home"
    parent_cwd = tmp_path / "project-a"
    child_cwd = tmp_path / "project-b"
    parent_cwd.mkdir()
    child_cwd.mkdir()
    registry = ProjectRegistry(home / "projects")
    parent_project = registry.create_project("project-a", "test", parent_cwd)
    child_project = registry.create_project("project-b", "test", child_cwd)
    (home / "projects" / parent_project.project_id / "AGENTS.md").write_text(
        "PARENT-PRIVATE-MARKER", encoding="utf-8"
    )
    (home / "projects" / child_project.project_id / "AGENTS.md").write_text(
        "CHILD-PRIVATE-MARKER", encoding="utf-8"
    )
    (child_cwd / "AGENTS.md").write_text("CHILD-REPO-MARKER", encoding="utf-8")
    _ignore_repo_agents = ignored_project == "project-a"
    ignored_id = (
        parent_project.project_id if _ignore_repo_agents else child_project.project_id
    )
    for project in (parent_project, child_project):
        record = json.loads(
            (home / "projects" / project.project_id / "project.json").read_text(
                encoding="utf-8"
            )
        )
        record["repo_agents"] = "include"
        (home / "projects" / project.project_id / "project.json").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
    record = json.loads(
        (home / "projects" / ignored_id / "project.json").read_text(encoding="utf-8")
    )
    record["repo_agents"] = "ignore"
    (home / "projects" / ignored_id / "project.json").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    store = ConversationStore(tmp_path / "sessions", cwd=parent_cwd)
    tools = ToolRegistry(
        parent_cwd,
        project_id=parent_project.project_id,
        project_registry=registry,
        skill_catalog=SkillCatalog.empty(),
    )
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "child-1",
                        "agent",
                        {
                            "prompt": "child-work",
                            "description": "other project child",
                            "cwd": str(child_cwd),
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
        registry=tools,
        max_turns=1,
        system_prompt="parent prompt",
        skill_catalog=SkillCatalog.empty(),
        root_project_id=parent_project.project_id,
        project_registry=registry,
    )
    loop.set_mcp_scope(home=home, project_dir=parent_cwd)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))

    child_system = backend.calls[1][0][0]
    system_text = " ".join(
        block.text for block in child_system.content if isinstance(block, TextContent)
    )
    assert "CHILD-PRIVATE-MARKER" in system_text
    assert "PARENT-PRIVATE-MARKER" not in system_text
    from zeta.core.project_context import load_project_context
    context = load_project_context(
        cwd=child_cwd,
        repo_root=child_cwd,
        zeta_home=home,
        catalog=SkillCatalog.empty(),
        session_project=parent_project,
        cwd_project=registry.show_project(child_project.project_id),
        inbox_enabled=False,
    )
    assert any("repository AGENTS.md" in notice for notice in context.notices) is (ignored_project == "project-b")
    await loop.close()


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
    registry._replace_entry_kind(project_a.project_id, "state", "PROJECT A MEMORY")
    registry._replace_entry_kind(project_b.project_id, "state", "PROJECT B MEMORY")
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
    assert send_to_run(
        store,
        handle,
        "also check the tests",
        channel=loop._background_owner.conversation_channel,
    ) is None
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
class _ExplicitFinishLoop:
    def __init__(
        self, registry: ToolRegistry, owner: BackgroundAgentOwner | None = None
    ) -> None:
        self.tool_registry = registry
        self._background_owner = owner or BackgroundAgentOwner(registry._session_store)
        self.agent_instance_id = "root:1"
        self.agent_depth = 1
        self.explicit_finish = True
        self._explicit_finish_report = None
        self.notification_wake = NotificationWake(registry._session_store)

    @property
    def background_work_descriptions(self) -> tuple[str, ...]:
        return self._background_owner.active_descriptions + tuple(
            item.headline
            for item in self.tool_registry.background_tasks.snapshot()
            if item.running
        )


def _explicit_finish_registry(
    tmp_path: Path,
) -> tuple[ConversationStore, ToolRegistry, _ExplicitFinishLoop]:
    from zeta.tools.finish import register

    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(store)
    loop = _ExplicitFinishLoop(registry)
    register(registry, child_loop=loop)
    return store, registry, loop


def test_finish_tool_description_and_child_prompt_say_finish_ends_the_run(
    tmp_path: Path,
) -> None:
    _store, registry, _loop = _explicit_finish_registry(tmp_path)
    prompt = _compose_child_system_prompt(
        "parent instructions",
        GENERAL_PRESET,
        explicit_finish=True,
    )
    description = registry.definitions_by_name["finish"].description

    assert "this run ends only when you call the `finish` tool" in prompt
    assert "Ending a turn without calling `finish` leaves you idle" in prompt
    assert "end this run" in description
    assert "status" in description
    assert "summary" in description


@pytest.mark.asyncio
async def test_explicit_finish_background_child_preserves_report_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "status": "done",
        "summary": "implemented the requested repair",
        "head_sha": "abcdef0",
        "verdict": "pass",
    }
    finish_call = ToolCall("finish-1", "finish", report)
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_run_agent_call()]),
            ScriptedTurn(tool_calls=[finish_call]),
            ScriptedTurn([TextContent("ignored after finish")]),
        ]
    )
    monkeypatch.setitem(
        AGENT_PRESETS,
        "run",
        replace(AGENT_PRESETS["run"], explicit_finish=True),
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    loop.explicit_finish = True
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    notification = await _wait_for_notification(store, "completed")
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
        and event.tool_call is not None
        and event.tool_call.name == "agent"
    )

    assert terminal.tool_result is not None
    assert terminal.tool_result.structured_content is not None
    expected = {
        "report_status": report["status"],
        "summary": report["summary"],
        "head_sha": report["head_sha"],
        "verdict": report["verdict"],
    }
    assert {
        key: terminal.tool_result.structured_content[key] for key in expected
    } == expected
    assert {key: notification.data[key] for key in expected} == expected
    from zeta.tools.agent import _read_agent_status

    status = _read_agent_status(
        store,
        requested_handle=str(notification.data["child_instance_id"]),
    )
    assert status[0]["report_status"] == "done"
    await loop.close()


@pytest.mark.parametrize("wake_kind", ["task", "child"])
@pytest.mark.asyncio
async def test_child_finishes_during_notification_turn_that_woke_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wake_kind: str,
) -> None:
    class Backend(CompletionBackend):
        def __init__(self) -> None:
            self.child_replied = asyncio.Event()
            self.finish_sent = False

        async def complete(self, messages, tool_schemas):
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
            notification = any(
                message.metadata.get("zeta_event") == "agent_notifications"
                for message in messages
            )
            if last_user == "start":
                blocks = [ToolUseContent(_run_agent_call())]
            elif notification and not self.finish_sent:
                self.finish_sent = True
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "finish-1",
                            "finish",
                            {"status": "done", "summary": f"read {wake_kind} notice"},
                        )
                    )
                ]
            elif last_user == "work the big task":
                self.child_replied.set()
                blocks = [TextContent("WAIT")]
            else:
                blocks = [TextContent("parent continued")]
            yield StreamEvent(StreamEventType.MESSAGE_START)
            for block in blocks:
                yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, blocks),
            )

    monkeypatch.setitem(
        AGENT_PRESETS,
        "run",
        replace(AGENT_PRESETS["run"], explicit_finish=True),
    )
    backend = Backend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    loop.explicit_finish = True

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_replied.wait(), timeout=2)
    await _wait_for_notification(store, "child_idle")
    handle = _run_handle_from_receipt(store)
    marker = store.agent_children()[handle]
    child_path = Path(marker["child_session_path"])
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)

    if wake_kind == "task":
        child_store.append_task_notification(
            task_id="task-1", command="true", exit_code=0
        )
    else:
        child_store.append_agent_notification_if_absent(
            f"{handle}:1",
            child_session_path=str(child_path / "agents" / "1"),
            description="direct child",
            status="completed",
            text="direct child completed",
        )
    child_store.close()
    loop._background_owner.conversation_channel.wake(handle)

    notification = await _wait_for_notification(store, "completed")
    completions = [
        item
        for item in store.agent_notifications(pending_only=False)
        if item.data.get("kind") == "agent_completion"
        and item.data.get("child_instance_id") == handle
    ]
    assert len(completions) == 1
    assert notification.data["summary"] == f"read {wake_kind} notice"
    await loop.close()


@pytest.mark.asyncio
async def test_explicit_finish_done_publishes_canonical_report(tmp_path: Path) -> None:
    store, registry, loop = _explicit_finish_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            "finish-1",
            "finish",
            {"status": "done", "summary": " shipped ", "head_sha": "abcdef0"},
        )
    )
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "status": "done",
        "summary": "shipped",
        "head_sha": "abcdef0",
    }
    assert loop._explicit_finish_report == result["structuredContent"]
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_explicit_finish_blocked_and_failed_map_to_receipts(
    tmp_path: Path,
) -> None:
    from zeta.agent.receipt import build_explicit_finish_receipt

    blocked = build_explicit_finish_receipt(
        {"status": "blocked", "summary": "need approval", "verdict": "inconclusive"},
        {},
        tool_call_id="a",
    )
    failed = build_explicit_finish_receipt(
        {"status": "failed", "summary": "tests failed", "verdict": "fail"},
        {},
        tool_call_id="b",
    )
    assert blocked["isError"] is False
    assert blocked["content"][0]["text"].startswith("blocked: need approval")
    assert failed["isError"] is True
    assert failed["content"][0]["text"].startswith("failed: tests failed")


def _reminder_loop(tmp_path: Path) -> tuple[ConversationStore, ToolRegistry, _ExplicitFinishLoop]:
    store, registry, loop = _explicit_finish_registry(tmp_path)
    store.start_agent_lifecycle(
        handle="root:1", started_at="2026-01-01T00:00:00+00:00", depth=1,
        agent_type="worker", description="work",
    )
    loop._terminal_transition_fence = TerminalTransitionFence(
        store,
        owned_work=lambda: loop.background_work_descriptions,
        delivering_prompt_id=lambda: getattr(loop, "_delivering_prompt_id", None),
    )
    loop._background_owner.conversation_channel.register_loop("root:1")
    return store, registry, loop


@pytest.mark.asyncio
async def test_real_input_wins_over_immediate_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    calls: list[str] = []
    original = loop._terminal_transition_fence.reserve_idle_nudge

    async def reserve() -> bool:
        store.pending_prompt_queue.append("real input", origin=MessageOrigin.AGENT_SEND)
        loop._background_owner.conversation_channel.wake("root:1")
        return await original()

    loop._terminal_transition_fence.reserve_idle_nudge = reserve

    async def consume(_loop, prompt, **kwargs):
        calls.append(prompt)
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "done" if len(calls) == 2 else "running"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    await asyncio.wait_for(
        agent_runner_module.consume_run(loop, "work", child_store=store), 2
    )
    assert calls == ["work", "real input"]
    assert run_consumer_module._IMMEDIATE_FINISH_REMINDER not in calls
    await registry.close()
    store.close()


@pytest.mark.parametrize("owned_kind", ["task", "child"])
@pytest.mark.asyncio
async def test_no_immediate_reminder_while_child_owns_running_task_or_live_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owned_kind: str,
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    owned_child: asyncio.Task[None] | None = None
    if owned_kind == "task":
        monkeypatch.setattr(
            registry.background_tasks,
            "snapshot",
            lambda: (
                SimpleNamespace(
                    running=True, task_id="task-owned", headline="owned task"
                ),
            ),
        )
    else:
        owned_child = asyncio.create_task(asyncio.Event().wait())
        loop._background_owner.register(
            "root:1:1",
            lambda: None,
            owned_child,
            parent_instance_id="root:1",
            description="owned child",
        )
    prompts: list[str] = []
    idle_waiting = asyncio.Event()

    async def consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    async def wait_idle(_wake, _timeout):
        idle_waiting.set()
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    run = asyncio.create_task(
        agent_runner_module.consume_run(
            loop, "work", child_store=store, idle_wait=wait_idle
        )
    )
    await asyncio.wait_for(idle_waiting.wait(), 2)
    assert prompts == ["work"]
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    if owned_child is not None:
        owned_child.cancel()
        await asyncio.gather(owned_child, return_exceptions=True)
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_reminder_not_repeated_when_child_ends_again_without_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    prompts: list[str] = []
    idle_waiting = asyncio.Event()
    release_idle = asyncio.Event()

    async def consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    async def wait_idle(_wake, _timeout):
        idle_waiting.set()
        await release_idle.wait()
        return True

    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    run = asyncio.create_task(
        agent_runner_module.consume_run(
            loop, "work", child_store=store, idle_wait=wait_idle
        )
    )
    await asyncio.wait_for(idle_waiting.wait(), 2)
    assert prompts == ["work", run_consumer_module._IMMEDIATE_FINISH_REMINDER]
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_reminded_child_that_calls_finish_completes_with_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    prompts: list[str] = []

    async def consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        if prompt == run_consumer_module._IMMEDIATE_FINISH_REMINDER:
            result = await registry.execute(
                ToolCall(
                    "finish",
                    "finish",
                    {
                        "status": "done",
                        "summary": "finished",
                        "verdict": "pass",
                    },
                )
            )
            assert result["isError"] is False
            return {
                "content": [],
                "isError": False,
                "structuredContent": {"status": "done"},
                "explicitFinishReport": loop._explicit_finish_report,
            }
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    result = await agent_runner_module.consume_run(loop, "work", child_store=store)
    expected = {
        "status": "done",
        "summary": "finished",
        "verdict": "pass",
    }
    assert prompts == ["work", run_consumer_module._IMMEDIATE_FINISH_REMINDER]
    assert result["explicitFinishReport"] == expected
    assert loop._explicit_finish_report == expected
    assert store.pending_prompt_queue.explicit_finish_report() == expected
    await registry.close()
    store.close()


@pytest.mark.parametrize("terminal_action", ["cancel", "finish"])
@pytest.mark.asyncio
async def test_cancel_or_accepted_finish_racing_reminder_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_action: str,
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    admission_reached = asyncio.Event()
    release_admission = asyncio.Event()
    idle_waiting = asyncio.Event()
    prompts: list[str] = []
    original = loop._terminal_transition_fence.reserve_idle_nudge

    async def reserve() -> bool:
        admission_reached.set()
        await release_admission.wait()
        return await original()

    async def wait_idle(_wake, _timeout):
        idle_waiting.set()
        await asyncio.Event().wait()
        return True

    async def consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    loop._terminal_transition_fence.reserve_idle_nudge = reserve
    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    run = asyncio.create_task(
        agent_runner_module.consume_run(
            loop, "work", child_store=store, idle_wait=wait_idle
        )
    )
    await asyncio.wait_for(admission_reached.wait(), 2)
    if terminal_action == "cancel":
        assert await loop._terminal_transition_fence.cancel() is True
        accepted_report = None
    else:
        result = await registry.execute(
            ToolCall(
                "finish-race",
                "finish",
                {"status": "done", "summary": "won admission race"},
            )
        )
        assert result["isError"] is False
        accepted_report = store.pending_prompt_queue.explicit_finish_report()
    release_admission.set()
    await asyncio.wait_for(idle_waiting.wait(), 2)

    report = store.pending_prompt_queue.explicit_finish_report()
    assert (terminal_action == "cancel" and report is None) or (
        terminal_action == "finish" and report == accepted_report
    )
    assert prompts == ["work"]
    assert run_consumer_module._IMMEDIATE_FINISH_REMINDER not in prompts
    assert await loop._terminal_transition_fence.cancel() is False
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_immediate_reminder_counts_toward_nudge_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    prompts: list[str] = []
    limit_reached = asyncio.Event()
    wait_calls = 0

    async def wait_idle(_wake, timeout):
        nonlocal wait_calls
        wait_calls += 1
        if timeout is not None:
            return False
        limit_reached.set()
        await asyncio.Event().wait()
        return True

    async def consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    run = asyncio.create_task(
        agent_runner_module.consume_run(
            loop, "work", child_store=store, idle_wait=wait_idle
        )
    )
    await asyncio.wait_for(limit_reached.wait(), 2)
    assert wait_calls == 3
    assert prompts == [
        "work",
        run_consumer_module._IMMEDIATE_FINISH_REMINDER,
        run_consumer_module._IDLE_NUDGE_PROMPT,
        run_consumer_module._IDLE_NUDGE_PROMPT,
    ]
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_non_explicit_finish_child_gets_no_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, registry, loop = _reminder_loop(tmp_path)
    loop.explicit_finish = False
    prompts: list[str] = []

    async def consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        return {
            "content": [{"type": "text", "text": "reply"}],
            "isError": False,
            "structuredContent": {"status": "done"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", consume)
    await asyncio.wait_for(
        agent_runner_module.consume_run(loop, "work", child_store=store), 2
    )
    assert prompts == ["work"]
    assert run_consumer_module._IMMEDIATE_FINISH_REMINDER not in prompts
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_explicit_finish_child_ending_with_text_gets_one_immediate_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    store = ConversationStore(tmp_path)
    store.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="work",
    )
    owner = BackgroundAgentOwner(store)
    owner.conversation_channel.register_loop("root:1")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(store)
    loop = SimpleNamespace(
        explicit_finish=True,
        agent_instance_id="root:1",
        _background_owner=owner,
        _terminal_transition_fence=TerminalTransitionFence(store),
        notification_wake=SimpleNamespace(pending_message=lambda: None),
        tool_registry=registry,
    )

    idle_waiting = asyncio.Event()

    async def fake_consume(*args, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["finish_lifecycle"]("idle", "WAIT")
        return {
            "content": [{"type": "text", "text": "WAIT"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    async def wait_idle(_wake, _timeout):
        idle_waiting.set()
        await asyncio.Event().wait()
        return True

    monkeypatch.setattr(agent_runner_module, "consume_child", fake_consume)
    task = asyncio.create_task(
        agent_runner_module.consume_run(
            loop,
            "work",
            child_store=store,
            child_turns=lambda: 1,
            finish_lifecycle=lambda *_: {},
            idle_wait=wait_idle,
        )
    )
    await asyncio.wait_for(idle_waiting.wait(), 2)
    assert calls == 2
    assert store.agent_lifecycle()["state"] == "idle"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_idle_child_is_running_during_wake_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_store = ConversationStore(tmp_path, session_id="parent")
    child_store = ConversationStore(parent_store.session_dir / "agents", session_id="1")
    call = _run_agent_call()
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="run",
        agent_type="run",
        background=True,
        child_instance_id="parent:1",
        accepts_follow_ups=True,
        explicit_finish=True,
    )
    child_store.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="run",
        description="run",
    )
    owner = BackgroundAgentOwner(parent_store)
    owner.conversation_channel.register_loop("parent:1")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(child_store)
    loop = SimpleNamespace(
        explicit_finish=True,
        agent_instance_id="parent:1",
        _background_owner=owner,
        _terminal_transition_fence=TerminalTransitionFence(child_store),
        notification_wake=SimpleNamespace(pending_message=lambda: None),
        tool_registry=registry,
    )
    first_idle = asyncio.Event()
    wake_started = asyncio.Event()
    release_wake = asyncio.Event()
    calls = 0

    async def fake_consume(_loop, _prompt, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            wake_started.set()
            await release_wake.wait()
        kwargs["finish_lifecycle"]("idle", "WAIT")
        if calls == 1:
            first_idle.set()
        return {
            "content": [{"type": "text", "text": "WAIT"}],
            "isError": False,
            "structuredContent": {"status": "running"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", fake_consume)
    task = asyncio.create_task(
        agent_runner_module.consume_run(
            loop,
            "work",
            child_store=child_store,
            child_turns=lambda: calls,
            finish_lifecycle=lambda *_: {},
        )
    )
    await asyncio.wait_for(first_idle.wait(), timeout=2)
    first_idle_since = child_store.agent_lifecycle()["idle_since"]
    assert send_to_run(
        parent_store,
        "parent:1",
        "continue",
        channel=owner.conversation_channel,
    ) is None
    await asyncio.wait_for(wake_started.wait(), timeout=2)
    assert child_store.agent_lifecycle()["state"] == "running"

    release_wake.set()
    async def wait_for_new_idle() -> None:
        while child_store.agent_lifecycle().get("idle_since") == first_idle_since:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_new_idle(), timeout=2)
    assert child_store.agent_lifecycle()["state"] == "idle"
    assert child_store.agent_lifecycle()["idle_since"] > first_idle_since

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    owner.conversation_channel.unregister_loop("parent:1")
    await registry.close()
    child_store.close()
    parent_store.close()


@pytest.mark.asyncio
async def test_idle_child_wakes_for_task_and_child_completion_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="work",
    )
    owner = BackgroundAgentOwner(store)
    owner.conversation_channel.register_loop("root:1")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(store)
    pending = SimpleNamespace(
        pending_message=lambda: (
            store.agent_notifications()[0] if store.agent_notifications() else None
        )
    )
    loop = SimpleNamespace(
        explicit_finish=True,
        agent_instance_id="root:1",
        _background_owner=owner,
        _terminal_transition_fence=TerminalTransitionFence(
            store, notification_pending=lambda: pending.pending_message() is not None
        ),
        notification_wake=pending,
        tool_registry=registry,
    )
    turns: list[MessageOrigin] = []

    async def fake_consume(_loop, _prompt, *, origin, **kwargs):
        turns.append(origin)
        if kwargs.get("notification_turn"):
            for entry in store.agent_notifications():
                store.acknowledge_agent_notification(entry.id)
        if len(turns) < 4:
            kwargs["finish_lifecycle"]("idle", "WAIT")
            return {
                "content": [],
                "isError": False,
                "structuredContent": {"status": "running"},
            }
        return {
            "content": [],
            "isError": False,
            "structuredContent": {"status": "done"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", fake_consume)
    run = asyncio.create_task(
        agent_runner_module.consume_run(
            loop,
            "work",
            child_store=store,
            child_turns=lambda: len(turns),
            finish_lifecycle=lambda *_: {},
        )
    )
    await asyncio.sleep(0.02)
    # Reads and unrelated notices do not call wake, so they cannot start a turn.
    assert len(turns) == 1
    store.append_pending_prompt("follow up")
    owner.conversation_channel.wake("root:1")
    await asyncio.sleep(0.02)
    store.append_task_notification(
        task_id="task-a", command="echo", exit_code=0, output_tail="done", log_path=None
    )
    owner.conversation_channel.wake("root:1")
    await asyncio.sleep(0.02)
    store.append_agent_notification(
        "root:1:1",
        child_session_path=str(tmp_path / "child"),
        description="child",
        status="completed",
        text="done",
    )
    owner.conversation_channel.wake("root:1")
    await asyncio.wait_for(run, 1)
    assert len(turns) == 4
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


async def _idle_nudge_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    idle_wait,
    fence=None,
    finish_after_calls: int | None = None,
):
    store = ConversationStore(tmp_path)
    store.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="work",
    )
    owner = BackgroundAgentOwner(store)
    owner.conversation_channel.register_loop("root:1")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(store)
    if fence is None:
        fence = TerminalTransitionFence(store)
    loop = SimpleNamespace(
        explicit_finish=True,
        agent_instance_id="root:1",
        _background_owner=owner,
        _terminal_transition_fence=fence,
        notification_wake=SimpleNamespace(pending_message=lambda: None),
        tool_registry=registry,
    )
    prompts: list[str] = []

    async def fake_consume(_loop, prompt, **kwargs):
        prompts.append(prompt)
        done = finish_after_calls is not None and len(prompts) >= finish_after_calls
        if not done:
            kwargs["finish_lifecycle"]("idle", "WAIT")
        return {
            "content": [{"type": "text", "text": "done" if done else "WAIT"}],
            "isError": False,
            "structuredContent": {"status": "done" if done else "running"},
        }

    monkeypatch.setattr(agent_runner_module, "consume_child", fake_consume)
    run = asyncio.create_task(
        agent_runner_module.consume_run(
            loop,
            "work",
            child_store=store,
            child_turns=lambda: len(prompts),
            finish_lifecycle=lambda *_: {},
            idle_clock=lambda: 0.0,
            idle_wait=idle_wait,
        )
    )
    return run, prompts, owner, registry, store


@pytest.mark.asyncio
async def test_idle_nudge_after_ten_minutes_without_running_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float | None] = []

    async def expire(_wait, timeout):
        waits.append(timeout)
        return False

    run, prompts, owner, registry, store = await _idle_nudge_run(
        tmp_path, monkeypatch, idle_wait=expire, finish_after_calls=3
    )
    await asyncio.wait_for(run, 1)
    assert waits == [600.0]
    assert prompts[1].startswith("you ended your turn without calling `finish`")
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_idle_nudge_waits_for_task_child_active_turn_and_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fence:
        def __init__(self) -> None:
            self.checks = iter([False, False, False, False, False, True, True])

        async def idle_nudge_eligible(self):
            return next(self.checks)

        async def reserve_idle_nudge(self):
            return next(self.checks)

    waits: list[float | None] = []

    async def wake_then_expire(_wait, timeout):
        waits.append(timeout)
        return len(waits) < 5

    run, prompts, owner, registry, store = await _idle_nudge_run(
        tmp_path,
        monkeypatch,
        idle_wait=wake_then_expire,
        fence=Fence(),
        finish_after_calls=2,
    )
    await asyncio.wait_for(run, 1)
    assert waits == [None, None, 600.0]
    assert len(prompts) == 2
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_idle_nudge_loses_to_prompt_committed_after_eligibility_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timed_out = False
    eligibility_checked = asyncio.Event()
    prompt_committed = asyncio.Event()
    real_fence: TerminalTransitionFence | None = None

    class PromptAfterEligibilityFence:
        async def idle_nudge_eligible(self) -> bool:
            assert real_fence is not None
            eligible = await real_fence.idle_nudge_eligible()
            if timed_out and eligible:
                eligibility_checked.set()
                await prompt_committed.wait()
            return eligible

        async def reserve_idle_nudge(self) -> bool:
            assert real_fence is not None
            eligibility_checked.set()
            await prompt_committed.wait()
            return await real_fence.reserve_idle_nudge()

    async def expire(_wait, _timeout):
        nonlocal timed_out
        timed_out = True
        return False

    fence = PromptAfterEligibilityFence()
    run, prompts, owner, registry, store = await _idle_nudge_run(
        tmp_path,
        monkeypatch,
        idle_wait=expire,
        fence=fence,
        finish_after_calls=2,
    )
    real_fence = TerminalTransitionFence(store)

    async def commit_prompt() -> None:
        await eligibility_checked.wait()
        store.append_pending_prompt("real follow-up")
        prompt_committed.set()

    commit = asyncio.create_task(commit_prompt())
    await asyncio.wait_for(run, 1)
    await commit
    assert prompts == ["work", "real follow-up"]
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_idle_wake_resets_nudge_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float | None] = []

    async def wake_then_expire(_wait, timeout):
        waits.append(timeout)
        return len(waits) == 1

    run, prompts, owner, registry, store = await _idle_nudge_run(
        tmp_path, monkeypatch, idle_wait=wake_then_expire, finish_after_calls=3
    )
    await asyncio.wait_for(run, 1)
    assert waits == [600.0, 600.0]
    assert len(prompts) == 3
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_idle_nudge_stops_after_three_lifetime_nudges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fourth_wait = asyncio.Event()

    async def expire_then_block(wait, timeout):
        if timeout is not None:
            return False
        fourth_wait.set()
        await wait()
        return True

    run, prompts, owner, registry, store = await _idle_nudge_run(
        tmp_path, monkeypatch, idle_wait=expire_then_block
    )
    await asyncio.wait_for(fourth_wait.wait(), 1)
    assert len(prompts) == 4
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_idle_child_never_auto_finishes_for_time_or_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.agent.explicit_finish import idle_resource_usage
    from zeta.tools.agent import _status_response

    blocked = asyncio.Event()

    async def expire_then_block(wait, timeout):
        if timeout is not None:
            return False
        blocked.set()
        await wait()
        return True

    run, prompts, owner, registry, store = await _idle_nudge_run(
        tmp_path, monkeypatch, idle_wait=expire_then_block
    )
    await asyncio.wait_for(blocked.wait(), 1)
    usage = idle_resource_usage([{"state": "idle"}] * 17)
    assert usage == {
        "idle_child_count": 17,
        "idle_child_fd_floor": 51,
        "idle_child_warning": True,
    }
    rows = [
        {
            "handle": f"root:{index}",
            "state": "idle",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
            "elapsed": 600.0,
            "turns_used": 1,
            "tool_calls": 0,
            "current_step": "idle, waiting for follow-up",
            "depth": 1,
            "agent_type": "worker",
            "description": "work",
            "cwd": "",
            "idle_since": "2026-01-01T00:00:00+00:00",
        }
        for index in range(17)
    ]
    status = _status_response(
        rows,
        offset=0,
        total=17,
        truncated=False,
        next_offset=None,
        finished_omitted=0,
    )
    assert status["structuredContent"]["idle_child_count"] == 17
    assert status["structuredContent"]["idle_child_fd_floor"] == 51
    assert status["structuredContent"]["idle_child_warning"] is True
    assert not run.done()
    assert len(prompts) == 4
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    owner.conversation_channel.unregister_loop("root:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_explicit_finish_refuses_owned_work_until_clear(tmp_path: Path) -> None:
    store, registry, loop = _explicit_finish_registry(tmp_path)
    watcher = asyncio.create_task(asyncio.Event().wait())
    loop._background_owner.register(
        "root:1:1",
        watcher.cancel,
        watcher,
        parent_store=store,
        description="child-a",
        parent_instance_id="root:1",
    )
    refused = await registry.execute(
        ToolCall("f1", "finish", {"status": "done", "summary": "done"})
    )
    assert refused["isError"] is True
    assert "child-a" in refused["content"][0]["text"]
    loop._background_owner.unregister("root:1:1")
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    accepted = await registry.execute(
        ToolCall("f2", "finish", {"status": "done", "summary": "done"})
    )
    assert accepted["isError"] is False
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_explicit_finish_twice_rejects_second_call(tmp_path: Path) -> None:
    store, registry, _loop = _explicit_finish_registry(tmp_path)
    first = await registry.execute(
        ToolCall("f1", "finish", {"status": "done", "summary": "done"})
    )
    second = await registry.execute(
        ToolCall("f2", "finish", {"status": "done", "summary": "again"})
    )
    assert first["isError"] is False
    assert second["isError"] is True
    assert "child is finishing" in second["content"][0]["text"]
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_tool_call_after_successful_finish_is_rejected(tmp_path: Path) -> None:
    store, registry, _loop = _explicit_finish_registry(tmp_path)
    await registry.execute(
        ToolCall("f1", "finish", {"status": "done", "summary": "done"})
    )
    result = await registry.execute(
        ToolCall("read-1", "read", {"path": str(tmp_path / "missing")})
    )
    assert result["isError"] is True
    assert "child is finishing" in result["content"][0]["text"]
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_provider_error_after_successful_finish_keeps_report(
    tmp_path: Path,
) -> None:
    class Child(_ExplicitFinishLoop):
        async def run_turn(self, *args, **kwargs):
            call = ToolCall("finish-1", "finish", {"status": "done", "summary": "kept"})
            result = ToolResult(
                call.id,
                "accepted",
                structured_content={"status": "done", "summary": "kept"},
            )
            yield StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=result,
            )
            yield StreamEvent(
                StreamEventType.ERROR,
                error=ErrorInfo("server_error", "provider failed"),
            )

    store, registry, _loop = _explicit_finish_registry(tmp_path)
    loop = Child(registry)
    result = await agent_runner_module.consume_child(
        loop,
        "work",
        origin=MessageOrigin.AGENT_PROMPT,
        child_path=str(tmp_path),
        publish=lambda _: None,
        child_turns=lambda: 0,
        update_turns=lambda _: None,
        update_step=lambda _: None,
        update_tool_calls=lambda _: None,
        finish_lifecycle=lambda *_: {},
        publish_lifecycle=lambda *a, **k: None,
        child_result=lambda text, **kw: {
            "content": [{"text": text}],
            "isError": kw.get("state") == "failed",
            "structuredContent": kw,
        },
        error_message=str,
    )
    assert result["explicitFinishReport"]["status"] == "done"
    assert result["explicitFinishReport"]["summary"] == "kept"
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_parent_exit_cancels_idle_child_subtree_and_cleans_receipts(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store)
    canceled: list[str] = []

    async def waiting() -> None:
        await asyncio.Event().wait()

    child_task = asyncio.create_task(waiting())
    grandchild_task = asyncio.create_task(waiting())
    owner.register(
        "root:1",
        lambda: (canceled.append("root:1"), child_task.cancel()),
        child_task,
        parent_store=store,
        description="idle",
        parent_instance_id=None,
    )
    owner.register(
        "root:1:1",
        lambda: (canceled.append("root:1:1"), grandchild_task.cancel()),
        grandchild_task,
        parent_store=store,
        description="nested",
        parent_instance_id="root:1",
    )
    owner.cancel_all()
    await owner.wait()
    assert set(canceled) == {"root:1", "root:1:1"}
    assert not owner.running
    store.close()


def test_idle_child_restart_recovers_as_canceled(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    call = ToolCall(
        "agent-1",
        "agent",
        {"prompt": "work", "description": "worker", "background": True},
    )
    child.mark_agent_parent(call.id, agent_type="worker")
    child.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="work",
    )
    child.update_agent_lifecycle(
        state="idle",
        current_step="idle, waiting for follow-up",
        idle_since="2026-01-01T00:01:00+00:00",
    )
    root.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="worker",
        agent_type="worker",
        background=True,
        child_instance_id="root:1",
        accepts_follow_ups=True,
        explicit_finish=True,
    )
    child.close()
    root.close()
    reopened = ConversationStore(tmp_path, session_id="root")
    loop = AgentLoop(FakeBackend([]), reopened, skill_catalog=SkillCatalog.empty())
    assert reopened.agent_children() == {}
    recovered = ConversationStore(reopened.session_dir / "agents", session_id="1")
    assert recovered.agent_lifecycle()["state"] == "canceled"
    recovered.close()
    asyncio.run(loop.close())
    reopened.close()


@pytest.mark.asyncio
async def test_explicit_finish_never_enters_run_finish_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(store)

    class Child(_ExplicitFinishLoop):
        async def run_turn(self, *args, **kwargs):
            yield StreamEvent(StreamEventType.TURN_START)
            message = Message(MessageRole.ASSISTANT, [TextContent("WAIT")])
            yield StreamEvent(
                StreamEventType.TURN_END, message=message, data={"tool_calls": 0}
            )

    loop = Child(registry)
    watcher = asyncio.create_task(asyncio.Event().wait())
    loop._background_owner.register(
        "root:1:1",
        watcher.cancel,
        watcher,
        parent_store=store,
        description="child",
        parent_instance_id="root:1",
    )

    async def fail_gate(*args, **kwargs):
        raise AssertionError("run_finish_gate entered")

    monkeypatch.setattr(agent_runner_module, "run_finish_gate", fail_gate)
    # A no-tool turn with owned work is idle for opted-in children, not inferred completion.
    result = await agent_runner_module.consume_child(
        loop,
        "work",
        origin=MessageOrigin.AGENT_PROMPT,
        child_path=str(tmp_path),
        publish=lambda _: None,
        child_turns=lambda: 0,
        update_turns=lambda _: None,
        update_step=lambda _: None,
        update_tool_calls=lambda _: None,
        finish_lifecycle=lambda *_: {},
        publish_lifecycle=lambda *a, **k: None,
        child_result=lambda text, **kw: {
            "content": [{"text": text}],
            "isError": False,
            "structuredContent": {"status": kw.get("status", kw.get("state"))},
        },
        error_message=str,
    )
    assert result["structuredContent"]["status"] == "running"
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    loop._background_owner.unregister("root:1:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_setting_off_preserves_legacy_finish_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    class Child(_ExplicitFinishLoop):
        async def run_turn(self, *args, **kwargs):
            yield StreamEvent(StreamEventType.TURN_START)
            message = Message(MessageRole.ASSISTANT, [TextContent("done")])
            yield StreamEvent(
                StreamEventType.TURN_END, message=message, data={"tool_calls": 0}
            )

    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bind_session_store(store)
    loop = Child(registry)
    loop.explicit_finish = False
    watcher = asyncio.create_task(asyncio.Event().wait())
    loop._background_owner.register(
        "root:1:1",
        watcher.cancel,
        watcher,
        parent_store=store,
        description="child",
        parent_instance_id="root:1",
    )

    async def gate(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(handoff=None, reason="stopped")

    monkeypatch.setattr(agent_runner_module, "run_finish_gate", gate)
    await agent_runner_module.consume_child(
        loop,
        "work",
        origin=MessageOrigin.AGENT_PROMPT,
        child_path=str(tmp_path),
        publish=lambda _: None,
        child_turns=lambda: 0,
        update_turns=lambda _: None,
        update_step=lambda _: None,
        update_tool_calls=lambda _: None,
        finish_lifecycle=lambda *_: {},
        publish_lifecycle=lambda *a, **k: None,
        child_result=lambda text, **kw: {
            "content": [{"text": text}],
            "isError": False,
            "structuredContent": kw,
        },
        error_message=str,
    )
    assert called is True
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    loop._background_owner.unregister("root:1:1")
    await registry.close()
    store.close()


@pytest.mark.asyncio
async def test_wait_with_running_task_wakes_parent_then_accepts_approval(
    tmp_path: Path,
) -> None:
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(tmp_path / "child")
    owner = BackgroundAgentOwner(parent)
    wake = asyncio.Event()
    owner.conversation_channel.set_root_wake(wake.set)
    child_task = asyncio.create_task(asyncio.Event().wait())
    owner.register(
        "root:1",
        child_task.cancel,
        child_task,
        parent_store=parent,
        active_store=child,
        parent_instance_id=None,
        description="review",
    )
    owner.conversation_channel.publish_idle(
        child_instance_id="root:1",
        child_session_path=str(child.session_dir),
        description="review",
        text="WAIT",
    )
    await asyncio.wait_for(wake.wait(), timeout=1)
    parent.register_agent_child(
        _run_agent_call(),
        child_session_path=str(child.session_dir),
        description="review",
        background=True,
        child_instance_id="root:1",
        accepts_follow_ups=True,
        explicit_finish=True,
    )
    assert (
        owner.conversation_channel.publish_follow_up(
            parent, "root:1", "approved", commit_timeout=1
        )
        is None
    )
    assert child.pending_prompts()[0].data["text"] == "approved"
    owner.cancel_subtree("root:1")
    await owner.wait()


def test_idle_notification_wakes_waiting_parent_without_polling(tmp_path: Path) -> None:
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(tmp_path / "child")
    owner = BackgroundAgentOwner(parent)
    woke: list[bool] = []
    owner.conversation_channel.set_root_wake(lambda: woke.append(True))
    parent.append_child_question(
        child_instance_id="root:1", question_id="q1", question="approve?"
    )
    owner.conversation_channel.publish_idle(
        child_instance_id="root:1",
        child_session_path=str(child.session_dir),
        description="review",
        text="x" * 20_000,
    )
    notice = parent.agent_notifications()[-1]
    assert notice.data["kind"] == "child_idle"
    assert notice.data["status"] == "child_idle"
    assert notice.data["ask_parent_question_id"] == "q1"
    assert len(notice.data["text"]) <= 10_000
    assert woke == [True]


def test_child_idle_then_terminal_receipt_is_appended_and_delivered_once(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_child_idle_notification(
        "root:1", child_session_path="/child", description="review", text="WAIT"
    )
    first, appended = store.append_agent_notification_if_absent(
        "root:1",
        child_session_path="/child",
        description="review",
        status="completed",
        text="done",
    )
    again, appended_again = store.append_agent_notification_if_absent(
        "root:1",
        child_session_path="/child",
        description="review",
        status="completed",
        text="duplicate",
    )
    assert appended is True and appended_again is False and first.id == again.id
    assert [item.data["kind"] for item in store.agent_notifications()] == [
        "child_idle",
        "agent_completion",
    ]


@pytest.mark.asyncio
async def test_child_idle_wakes_idle_explicit_finish_parent_tree(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path / "root")
    parent = ConversationStore(tmp_path / "parent")
    grandchild = ConversationStore(tmp_path / "grandchild")
    owner = BackgroundAgentOwner(root)
    parent_task = asyncio.create_task(asyncio.Event().wait())
    grandchild_task = asyncio.create_task(asyncio.Event().wait())
    owner.register(
        "root:1",
        parent_task.cancel,
        parent_task,
        parent_store=root,
        active_store=parent,
        parent_instance_id=None,
        description="parent",
    )
    owner.register(
        "root:1:1",
        grandchild_task.cancel,
        grandchild_task,
        parent_store=parent,
        active_store=grandchild,
        parent_instance_id="root:1",
        description="grandchild",
    )
    owner.conversation_channel.register_loop("root:1")
    parent_wake = asyncio.create_task(owner.conversation_channel.wait("root:1"))
    owner.conversation_channel.publish_idle(
        child_instance_id="root:1:1",
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        text="WAIT",
    )
    await asyncio.wait_for(parent_wake, timeout=1)
    assert parent.agent_notifications()[0].data["kind"] == "child_idle"
    owner.cancel_subtree("root:1")
    await owner.wait()


@pytest.mark.asyncio
async def test_explicit_finish_accepts_from_a_delivered_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FollowUpFinishBackend()
    monkeypatch.setitem(AGENT_PRESETS, "run", replace(AGENT_PRESETS["run"], explicit_finish=True))
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    loop.explicit_finish = True

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)
    handle = _run_handle_from_receipt(store)
    assert send_to_run(
        store, handle, "first follow-up",
        channel=loop._background_owner.conversation_channel,
    ) is None
    backend.release_child.set()
    await asyncio.wait_for(backend.finish_generated.wait(), timeout=2)
    await _wait_for_notification(store, "completed")

    assert backend.finish_attempts == 1
    assert backend.child_prompts[-1] == "first follow-up"
    await loop.close()


@pytest.mark.asyncio
async def test_explicit_finish_refuses_when_another_follow_up_is_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FollowUpFinishBackend()
    backend.finish_refused_recovery = True
    monkeypatch.setitem(AGENT_PRESETS, "run", replace(AGENT_PRESETS["run"], explicit_finish=True))
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    loop.explicit_finish = True

    await _collect(loop.run_turn("start", origin=MessageOrigin.USER))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)
    handle = _run_handle_from_receipt(store)
    channel = loop._background_owner.conversation_channel
    assert send_to_run(store, handle, "first follow-up", channel=channel) is None
    assert send_to_run(store, handle, "second follow-up", channel=channel) is None
    child_path = Path(str(store.agent_children()[handle]["child_session_path"]))
    backend.release_child.set()
    await asyncio.wait_for(backend.finish_generated.wait(), timeout=2)
    await _wait_for_notification(store, "completed")

    assert backend.finish_attempts == 2
    child_store = ConversationStore(child_path.parent, session_id=child_path.name)
    assert any(
        message.tool_result is not None
        and message.tool_result.is_error
        and "pending follow-up" in message.tool_result.content
        for message in child_store.messages()
    ), [
        message.tool_result.content
        for message in child_store.messages()
        if message.tool_result is not None
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_reviewer_follow_up_committed_at_finish_boundary_is_read(
    tmp_path: Path,
) -> None:
    store, registry, _loop = _explicit_finish_registry(tmp_path)
    store.pending_prompt_queue.append(
        "review this first", origin=MessageOrigin.AGENT_SEND
    )
    refused = await registry.execute(
        ToolCall("finish-1", "finish", {"status": "done", "summary": "done"})
    )
    assert refused["isError"] is True
    store.acknowledge_pending_prompt(store.pending_prompts()[0].id)
    accepted = await registry.execute(
        ToolCall("finish-2", "finish", {"status": "done", "summary": "done"})
    )
    assert accepted["isError"] is False
    with pytest.raises(PendingPromptsClosedError):
        store.pending_prompt_queue.append("too late", origin=MessageOrigin.AGENT_SEND)
    await registry.close()


@pytest.mark.asyncio
async def test_pending_task_or_child_notification_rejects_finish(
    tmp_path: Path,
) -> None:
    store, registry, loop = _explicit_finish_registry(tmp_path)
    store.append_task_notification(task_id="t1", command="true", exit_code=0)
    pending_task = await registry.execute(
        ToolCall("finish-1", "finish", {"status": "done", "summary": "done"})
    )
    assert pending_task["isError"] is True
    assert "pending notification" in pending_task["content"][0]["text"]
    store.acknowledge_agent_notifications(
        [store.agent_notifications()[0].id]
    )

    store.append_agent_notification_if_absent(
        "root:1:1",
        child_session_path="/child",
        description="child",
        status="completed",
        text="child completed",
    )
    pending_child = await registry.execute(
        ToolCall("finish-2", "finish", {"status": "done", "summary": "done"})
    )
    assert pending_child["isError"] is True

    assert loop.notification_wake.schedule() is True
    reserved = await registry.execute(
        ToolCall("finish-3", "finish", {"status": "done", "summary": "done"})
    )
    assert reserved["isError"] is True

    message = loop.notification_wake.begin(notification=True)
    assert message is not None
    loop.notification_wake.commit_persisted(message)
    assert loop.notification_wake.pending_message() is None
    assert loop.notification_wake.has_undelivered_claim is False
    delivered = await registry.execute(
        ToolCall("finish-4", "finish", {"status": "done", "summary": "done"})
    )
    assert delivered["isError"] is False, delivered
    await loop.notification_wake.finish(success=True)
    await registry.close()


@pytest.mark.asyncio
async def test_terminal_fence_serializes_finish_idle_nudge_cancel_and_recovery(
    tmp_path: Path,
) -> None:
    from zeta.agent.terminal_fence import TerminalTransitionFence

    prompt_store = ConversationStore(tmp_path, session_id="prompt")
    prompt_fence = TerminalTransitionFence(prompt_store)
    prompt_ready = asyncio.Event()

    async def nudge_after_prompt() -> bool:
        await prompt_ready.wait()
        return await prompt_fence.reserve_idle_nudge()

    nudge = asyncio.create_task(nudge_after_prompt())
    prompt_store.append_pending_prompt("real follow-up")
    prompt_ready.set()
    assert await nudge is False

    finish_store = ConversationStore(tmp_path, session_id="finish")
    finish_fence = TerminalTransitionFence(finish_store)
    finish_ready = asyncio.Event()

    async def nudge_after_finish() -> bool:
        await finish_ready.wait()
        return await finish_fence.reserve_idle_nudge()

    nudge = asyncio.create_task(nudge_after_finish())
    assert await finish_fence.accept_finish({"status": "done", "summary": "done"}) is None
    finish_ready.set()
    assert await nudge is False
    assert await finish_fence.cancel() is False
    assert await finish_fence.recover() == {"status": "done", "summary": "done"}

    cancel_store = ConversationStore(tmp_path, session_id="cancel")
    cancel_fence = TerminalTransitionFence(cancel_store)
    cancel_ready = asyncio.Event()

    async def nudge_after_cancel() -> bool:
        await cancel_ready.wait()
        return await cancel_fence.reserve_idle_nudge()

    nudge = asyncio.create_task(nudge_after_cancel())
    assert await cancel_fence.cancel() is True
    cancel_ready.set()
    assert await nudge is False


@pytest.mark.asyncio
async def test_explicit_finish_cancel_wins_before_acceptance(tmp_path: Path) -> None:
    from zeta.agent.terminal_fence import TerminalTransitionFence

    store = ConversationStore(tmp_path)
    fence = TerminalTransitionFence(store)
    cancel_won = asyncio.Event()

    async def finish_after_cancel() -> str | None:
        await cancel_won.wait()
        return await fence.accept_finish({"status": "done", "summary": "late"})

    finish_task = asyncio.create_task(finish_after_cancel())
    assert await fence.cancel() is True
    cancel_won.set()
    assert await finish_task == "child is canceled"


@pytest.mark.asyncio
async def test_explicit_finish_cancel_after_acceptance_keeps_report(
    tmp_path: Path,
) -> None:
    from zeta.agent.terminal_fence import TerminalTransitionFence

    store = ConversationStore(tmp_path)
    report = {"status": "blocked", "summary": "need approval"}
    fence = TerminalTransitionFence(store)
    finish_accepted = asyncio.Event()

    async def accept_finish() -> None:
        assert await fence.accept_finish(report) is None
        finish_accepted.set()

    finish_task = asyncio.create_task(accept_finish())
    await finish_accepted.wait()
    assert await fence.cancel() is False
    await finish_task
    assert store.pending_prompt_queue.explicit_finish_report() == report


def test_events_after_terminal_state_emit_no_second_receipt(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_agent_notification_if_absent(
        "root:1",
        child_session_path="/child",
        description="review",
        status="completed",
        text="done",
    )
    store.append_child_idle_notification(
        "root:1", child_session_path="/child", description="review", text="late"
    )
    _entry, appended = store.append_agent_notification_if_absent(
        "root:1",
        child_session_path="/child",
        description="review",
        status="canceled",
        text="late cancel",
    )
    assert appended is False
    assert (
        len(
            [
                item
                for item in store.agent_notifications()
                if item.data.get("kind") == "agent_completion"
            ]
        )
        == 1
    )


def test_finish_recovery_after_durable_acceptance_publishes_report_once(
    tmp_path: Path,
) -> None:
    parent = ConversationStore(tmp_path / "parent")
    child = ConversationStore(parent.session_dir / "agents", session_id="1")
    call = _run_agent_call()
    parent.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="review",
        background=True,
        child_instance_id="root:1",
        accepts_follow_ups=True,
        explicit_finish=True,
    )
    child.start_agent_lifecycle(
        handle="root:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="review",
    )
    child.pending_prompt_queue.accept_explicit_finish(
        {"status": "done", "summary": "recovered", "head_sha": "abcdef0"}
    )
    loop = SimpleNamespace(store=parent, _background_owner=BackgroundAgentOwner(parent))
    recover_agent_children(loop)
    notices = [
        item
        for item in parent.agent_notifications(pending_only=False)
        if item.data.get("kind") == "agent_completion"
    ]
    assert len(notices) == 1
    assert notices[0].data["summary"] == "recovered"
    recover_agent_children(loop)
    assert (
        len(
            [
                item
                for item in parent.agent_notifications(pending_only=False)
                if item.data.get("kind") == "agent_completion"
            ]
        )
        == 1
    )
