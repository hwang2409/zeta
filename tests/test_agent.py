import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from zeta.agent_background import (
    BackgroundAgentOwner,
    adopt_agent_children,
    finish_background_child,
)
from zeta.agent_budget import MAX_AGENT_TURN_CAP, AgentTree
from zeta.core.abort import AbortGenerationRegistry
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.mcp import MCPMount
from zeta.tools import ToolRegistry
from zeta.tools.agent import ChildApprovalPolicy
from zeta.tools.agent_presets import (
    AGENT_PRESETS,
    GENERAL_PRESET,
)
from zeta.tui.render import render_event
from zeta.tui.todo import TodoWidget
from zeta.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)


async def _collect(events):
    return [event async for event in events]


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


def _agent_call(
    call_id: str = "agent-1", agent_type: str | None = None
) -> ToolCall:
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
        self.child_count = 0
        self.children_started = asyncio.Event()
        self.release_children = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(call) for call in self.parent_calls]
        else:
            self.child_count += 1
            child_index = self.child_count
            if self.child_count == len(self.parent_calls):
                self.children_started.set()
            await self.release_children.wait()
            blocks = [TextContent(f"child-{child_index}")]
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
        if last_user == "start":
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


async def _wait_for_notification(
    store: ConversationStore, status: str
) -> object:
    for _ in range(100):
        notifications = store.agent_notifications()
        if notifications and notifications[-1].data["status"] == status:
            return notifications[-1]
        await asyncio.sleep(0.01)
    raise AssertionError(f"missing {status} background notification")


@pytest.mark.asyncio
async def test_background_agent_returns_handle_and_parent_continues(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    assert result.structured_content["description"] == "background research"
    assert backend.child_started.is_set()

    parent_events = await _collect(loop.run_turn("follow up"))
    assert any(
        event.type is StreamEventType.TURN_END for event in parent_events
    )
    assert store.agent_notifications() == []

    backend.release_child.set()
    notification = await _wait_for_notification(store, "completed")
    assert notification.data["text"].startswith("child complete")
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_notification_waits_for_next_turn_boundary(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    events = await _collect(loop.run_turn("follow up"))
    assert events[0].type is StreamEventType.AGENT_NOTIFICATION
    assert events[0].data["text"].startswith("child complete")
    assert events[0].data["text"].count("error=false") == 1
    assert events[0].data["text"].count("canceled=false") == 1
    rendered = render_event(events[0])
    assert rendered is not None
    assert rendered.plain.count("error=false") == 1
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.content.count("error=false") == 1
    assert loop.store.agent_notifications() == []
    await loop.close()


@pytest.mark.asyncio
async def test_background_multibyte_receipt_fits_persisted_limit(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    backend.child_text = "😀" * 1_800
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
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
    assert len(json.dumps(persisted.to_dict(), ensure_ascii=False).encode("utf-8")) <= 10_000
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_during_setup_is_drained_at_turn_start(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skip_mcp_mount=True)

    await _collect(loop.run_turn("start"))
    setup_started = asyncio.Event()

    async def delayed_setup() -> None:
        setup_started.set()
        backend.release_child.set()
        while not store.agent_notifications():
            await asyncio.sleep(0)

    loop._ensure_mcp_servers = delayed_setup
    events = await _collect(loop.run_turn("follow up"))

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
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    for _ in range(100):
        if not loop._tracked_tasks:
            break
        await asyncio.sleep(0)

    assert not [
        task
        for task in asyncio.all_tasks()
        if task not in baseline and not task.done()
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_parent_abort_cancels_background_agent(tmp_path: Path) -> None:
    call = _background_agent_call()
    backend = BackgroundBackend([call])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
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
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))
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
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_foreground_child_does_not_wait_for_background_grandchild(
    tmp_path: Path,
) -> None:
    backend = ForegroundNestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    events = await asyncio.wait_for(task, timeout=1)

    assert any(event.type is StreamEventType.TURN_END for event in events)
    assert store.agent_notifications() == []
    marker = next(iter(store.agent_children().values()))
    assert marker["child_session_path"] == str(
        store.session_dir / "agents" / "1" / "agents" / "1"
    )

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
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))
    results = [message.tool_result for message in store.messages() if message.tool_result]
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
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)])
    )
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
    AgentLoop(BackgroundBackend([]), resumed)

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
    AgentLoop(BackgroundBackend([]), resumed)

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "completed"
    assert notification.data["text"].startswith("child complete")
    assert not resumed.agent_children()
    assert ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).agent_canceled() is None


def test_resume_cancels_adopted_background_grandchild(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(
        child.session_dir / "agents", session_id="1"
    )
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
    AgentLoop(FakeBackend([]), resumed)

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
    AgentLoop(FakeBackend([]), resumed)

    notifications = resumed.agent_notifications(pending_only=False)
    assert [entry.data["child_instance_id"] for entry in notifications] == [
        "root:2:1"
    ]
    recovered = ConversationStore(
        children[1].session_dir / "agents", session_id="1"
    )
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
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))

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
    loop = AgentLoop(backend, store, max_turns=1)

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.wait_for(backend.children_started.wait(), timeout=1)
    backend.release_children.set()
    await asyncio.wait_for(task, timeout=1)

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert all(
        result is not None and result.content.startswith(expected)
        for result, expected in zip(results, ("child-1", "child-2"), strict=True)
    )
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
        agent_turn_budget=100,
    )

    events = await asyncio.wait_for(_collect(loop.run_turn("start")), timeout=30)

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
        await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_parent_abort_cancels_all_parallel_children(tmp_path: Path) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelChildrenBackend(calls)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.wait_for(backend.children_started.wait(), timeout=1)
    loop.abort()
    await asyncio.wait_for(task, timeout=1)

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert len(results) == 2
    assert all(
        result is not None
        and result.content.startswith("tool execution canceled")
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
    loop = AgentLoop(backend, store, approval_policy=policy, max_turns=1)

    task = asyncio.create_task(_collect(loop.run_turn("start")))
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
async def test_agent_returns_child_text_and_persists_child_session(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content.startswith("done")
    child_dir = store.session_dir / "agents" / "1"
    assert (child_dir / "conversation.jsonl").exists()
    assert [message.role for message in ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert "agent" in {
        schema["name"] for schema in backend.calls[1][1]
    }
    assert {
        schema["name"] for schema in backend.calls[1][1]
    } == {
        schema["name"] for schema in backend.calls[0][1]
    }


def test_agent_schema_uses_preset_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = replace(
        AGENT_PRESETS["explore"],
        name="custom",  # type: ignore[arg-type]
    )
    monkeypatch.setitem(AGENT_PRESETS, "explore", custom)
    registry = ToolRegistry(tmp_path)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(FakeBackend([]), store, registry=registry)

    agent_schema = next(schema for schema in registry.schemas if schema["name"] == "agent")
    agent_type_schema = agent_schema["parameters"]["properties"]["agent_type"]
    assert agent_type_schema["enum"] == [
        preset.name for preset in AGENT_PRESETS.values()
    ]
    expected_description = "Choose one of: " + "; ".join(
        f"{preset.name}: {preset.selection_guidance}"
        for preset in AGENT_PRESETS.values()
    ) + "."
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

    await _collect(AgentLoop(omitted_backend, omitted, max_turns=1).run_turn("start"))
    await _collect(AgentLoop(explicit_backend, explicit, max_turns=1).run_turn("start"))

    omitted_child_state = omitted.session_dir / "agents" / "1" / "session_state.json"
    explicit_child_state = explicit.session_dir / "agents" / "1" / "session_state.json"
    assert omitted_child_state.read_bytes() == explicit_child_state.read_bytes()
    assert json.loads(omitted_child_state.read_text()) == {"bash_cwd": str(tmp_path)}
    assert omitted.state_path.read_bytes() == explicit.state_path.read_bytes()


@pytest.mark.asyncio
async def test_explore_child_has_read_only_tools_and_rejects_exec(
    tmp_path: Path,
) -> None:
    child_exec = ToolCall("child-exec", "exec", {"command": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn(tool_calls=[child_exec]),
            ScriptedTurn([TextContent("explore complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
    assert child_result.content == "unknown tool: exec"


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

    monkeypatch.setattr("zeta.loop.mount_mcp_servers", mount_write_tool)
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type=agent_type)]),
            ScriptedTurn(
                tool_calls=[ToolCall("remote-write", "remote:write", {})]
            ),
            ScriptedTurn([TextContent("restricted complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
async def test_plan_child_includes_todo_and_only_read_only_tools(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="plan")]),
            ScriptedTurn([TextContent("plan complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
@pytest.mark.parametrize(
    ("agent_type", "turn_cap"),
    [("explore", 15), ("plan", 20)],
)
async def test_typed_child_turn_cap_is_enforced(
    tmp_path: Path, agent_type: str, turn_cap: int
) -> None:
    child_call = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call(agent_type=agent_type)])]
        + [
            ScriptedTurn(
                [TextContent(f"step-{turn}")],
                tool_calls=[child_call],
            )
            for turn in range(1, turn_cap + 1)
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert f"{turn_cap}-turn cap" in result.content
    assert result.structured_content == {
        "turns_used": turn_cap,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "agent_type": agent_type,
        "child_instance_id": f"{store.session_id}:1",
    }


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
    loop = AgentLoop(backend, store, max_turns=1)

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
        ).run_turn("start")
    )

    child_system = backend.calls[1][0][0]
    system_text = " ".join(
        block.text for block in child_system.content if isinstance(block, TextContent)
    )
    assert "You are an explore sub-agent." in system_text
    assert "existing child instructions" in system_text


@pytest.mark.asyncio
async def test_child_registry_preserves_parent_pre_execution_hook(tmp_path: Path) -> None:
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
    registry = ToolRegistry(store.cwd, pre_execute_hook=deny_bash)
    await _collect(AgentLoop(backend, store, registry=registry, max_turns=1).run_turn("start"))

    assert observed[-1] == "bash"
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    denied = next(message.tool_result for message in child_messages if message.tool_result)
    assert denied.is_error
    assert "denied by test hook" in denied.content


@pytest.mark.asyncio
async def test_child_registry_shares_todo_store_but_isolates_other_session_tools(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "sessions"
    parent_store = ConversationStore(sessions, session_id="parent", cwd=tmp_path)
    child_store = ConversationStore(sessions, session_id="child", cwd=tmp_path)
    (tmp_path / "nested").mkdir()
    registry = ToolRegistry(parent_store.cwd, session_store=parent_store)
    child_registry = registry.clone_for_session(child_store)
    child_loop = AgentLoop(FakeBackend([]), child_store, registry=child_registry)
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
        {"content": "child work", "status": "pending"}
    ]
    assert child_store.todo_items() == []
    assert widget.visible
    assert parent_store.bash_cwd == str(tmp_path)
    assert child_store.bash_cwd == str(tmp_path / "nested")

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
    task_a = asyncio.create_task(policy_a.authorize(call_a, child_a_signal))
    task_b = asyncio.create_task(policy_b.authorize(call_b, child_b_signal))
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


@pytest.mark.asyncio
async def test_child_approval_cleanup_removes_pending_requests(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")
    signal = AbortGenerationRegistry().new_generation()
    task = asyncio.create_task(
        child_policy.authorize(ToolCall("pending", "bash", {"cmd": "wait"}), signal)
    )
    while not policy.pending_requests():
        await asyncio.sleep(0)

    child_policy.cleanup()
    assert await task == ApprovalDecision.DENY
    assert policy.pending_requests() == []
    signal.abort()


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
    registry = ToolRegistry(store.cwd)
    registry.register("wait", wait_forever)
    loop = AgentLoop(backend, store, registry=registry, max_turns=1)
    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(_collect(loop.run_turn("start")))
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
async def test_agent_turn_cap_returns_loud_error(tmp_path: Path) -> None:
    child_call = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()])]
        + [
            ScriptedTurn(
                [TextContent(f"step-{turn}")],
                tool_calls=[child_call],
            )
            for turn in range(1, 26)
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.is_error
    assert "25-turn cap" in result.content
    assert "partial state is saved" in result.content
    assert "last assistant text: step-25" in result.content
    assert "turns used: 25" in result.content


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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content.startswith("child complete")
    child_result = next(
        message.tool_result
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
        if message.tool_result
    )
    assert child_result.content.startswith("grandchild complete")
    grandchild_schemas = {
        schema["name"] for schema in backend.calls[2][1]
    }
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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
async def test_shared_turn_budget_covers_generations(tmp_path: Path) -> None:
    nested = _agent_call("grandchild")
    child = _agent_call("child")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[nested]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            agent_turn_budget=1,
        ).run_turn("start")
    )

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.is_error
    assert "shared agent turn budget exhausted" in result.content
    assert result.structured_content is not None
    assert result.structured_content["error_code"] == "agent_turn_budget"
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    child_result = next(
        message.tool_result
        for message in child_store.messages()
        if message.tool_result
    )
    assert child_result.is_error
    assert "shared agent turn budget exhausted" in child_result.content


@pytest.mark.asyncio
async def test_shared_turn_budget_covers_parallel_grandchildren(tmp_path: Path) -> None:
    grandchildren = [_agent_call("grandchild-1"), _agent_call("grandchild-2")]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("child")]),
            ScriptedTurn(tool_calls=grandchildren),
            ScriptedTurn([TextContent("one")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            agent_turn_budget=2,
        ).run_turn("start")
    )

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert [result.is_error for result in results if result is not None] == [True]
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    child_results = [
        message.tool_result
        for message in child_store.messages()
        if message.tool_result is not None
    ]
    assert sorted(result.is_error for result in child_results) == [False, True]
    assert any(
        result.structured_content is not None
        and result.structured_content.get("error_code") == "agent_turn_budget"
        for result in child_results
    )


def test_agent_loop_rejects_conflicting_turn_budget_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pass only one agent turn budget"):
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path),
            agent_turn_budget=1,
            agent_tree=AgentTree(),
        )


@pytest.mark.asyncio
async def test_top_level_agent_calls_get_fresh_shared_turn_budgets(
    tmp_path: Path,
) -> None:
    first = _agent_call("first")
    second = _agent_call("second")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[first]),
            ScriptedTurn([TextContent("first complete")]),
            ScriptedTurn(tool_calls=[second]),
            ScriptedTurn([TextContent("second complete")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, agent_turn_budget=1)

    await _collect(loop.run_turn("start"))
    await _collect(loop.run_turn("follow up"))

    results = [
        message.tool_result for message in store.messages() if message.tool_result
    ]
    assert all(
        result is not None and result.content.startswith(expected)
        for result, expected in zip(
            results, ("first complete", "second complete"), strict=True
        )
    )


@pytest.mark.asyncio
async def test_parallel_top_level_agent_invocations_have_independent_budgets(
    tmp_path: Path,
) -> None:
    async def run_agent(index: int) -> str:
        call = _agent_call(f"agent-{index}")
        backend = FakeBackend(
            [
                ScriptedTurn(tool_calls=[call]),
                ScriptedTurn([TextContent(f"agent {index} complete")]),
            ]
        )
        store = ConversationStore(tmp_path / str(index))
        loop = AgentLoop(backend, store, max_turns=1, agent_turn_budget=1)
        await _collect(loop.run_turn("start"))
        result = next(
            message.tool_result
            for message in store.messages()
            if message.tool_result
        )
        return result.content

    results = await asyncio.gather(run_agent(1), run_agent(2))
    assert results[0].startswith("agent 1 complete")
    assert results[1].startswith("agent 2 complete")


@pytest.mark.asyncio
async def test_adopted_background_child_keeps_origin_tree_budget(
    tmp_path: Path,
) -> None:
    backend = ForegroundNestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, agent_turn_budget=1)

    await _collect(loop.run_turn("start"))
    notification = await _wait_for_notification(store, "error")

    assert (
        "shared agent turn budget exhausted for this agent tree"
        in notification.data["text"]
    )


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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))
    await _wait_for_notification(store, "completed")

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    nested_result = next(
        message.tool_result
        for message in child_store.messages()
        if message.tool_result
    )
    assert nested_result.structured_content is not None
    assert nested_result.structured_content["status"] == "running"
    assert [entry.data["status"] for entry in child_store.agent_notifications()] == [
        "completed"
    ]
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert grandchild_store.agent_canceled() is None


@pytest.mark.asyncio
async def test_abort_propagates_through_two_nested_levels(tmp_path: Path) -> None:
    backend = NestedBlockingBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    task = asyncio.create_task(_collect(loop.run_turn("start")))

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
    AgentLoop(FakeBackend([]), resumed)

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
    AgentLoop(FakeBackend([]), resumed)

    assert resumed.agent_notifications()[0].data["status"] == "canceled"
    resumed_child = ConversationStore(
        resumed.session_dir / "agents", session_id="1"
    )
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
        backend, store, approval_policy=policy, max_turns=1
    ).run_turn("start"):
        events.append(event)
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            assert [request.tool_call.id for request in policy.pending_requests()] == [
                child_call.id
            ]
            assert policy.pending_requests()[0].label == "task research: bash"
            assert all(
                message.tool_result is None or message.tool_result.tool_call_id != child_call.id
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
    denied = next(message.tool_result for message in child_messages if message.tool_result)
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
    loop = AgentLoop(backend, store, approval_policy=policy, max_turns=1)

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    for _ in range(100):
        pending = policy.pending_requests()
        if pending:
            break
        await asyncio.sleep(0.01)
    assert len(pending) == 1
    request = pending[0]
    assert request.child_instance_id == f"{store.session_id}:1:1"
    assert getattr(policy, action)(request.key)
    await asyncio.wait_for(task, timeout=1)

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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    result = next(message.tool_result for message in replayed.messages() if message.tool_result)
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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_type() == "explore"
    assert json.loads(child.state_path.read_text())["agent_parent"] == {
        "tool_call_id": "agent-1",
        "agent_type": "explore",
        "status": "finished",
    }
    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"


@pytest.mark.asyncio
async def test_empty_child_final_message_returns_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn()]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn()]
    )
    store = ConversationStore(tmp_path)

    events = await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
    )
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    persisted = next(message for message in store.messages() if message.tool_result)
    result = persisted.tool_result
    assert result is not None
    assert len(json.dumps(persisted.to_dict(), ensure_ascii=False).encode("utf-8")) <= 10_000
    receipt_row = next(
        row
        for row in store.path.read_bytes().splitlines()
        if b'"tool_result"' in row
    )
    assert len(receipt_row) <= 10_000
    assert result.content.count("error=false") == 1
    assert result.content.count("canceled=false") == 1


@pytest.mark.asyncio
async def test_parent_abort_cancels_child(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.sleep(0.05)
    loop.abort()

    events = await task
    result = next(message.tool_result for message in store.messages() if message.tool_result)
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
    child_state = (store.session_dir / "agents" / "1" / "session_state.json").read_text()
    assert '"agent_parent"' not in child_state
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_canceled() == {
        "tool_call_id": "agent-1",
        "content": "tool execution canceled",
    }
    assert not store.agent_children()


@pytest.mark.asyncio
async def test_parent_abort_after_child_turn_reports_completed_turns(tmp_path: Path) -> None:
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
    loop = AgentLoop(backend, store, max_turns=1)

    async for event in loop.run_turn("start"):
        if (
            event.type is StreamEventType.TOOL_EXECUTION_UPDATE
            and event.delta is not None
            and "turn 2: thinking" in event.delta
        ):
            loop.abort()

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content == {
        "turns_used": 1,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "child_instance_id": f"{store.session_id}:1",
    }


@pytest.mark.asyncio
async def test_typed_child_type_survives_cancellation_and_reopen(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.sleep(0.05)
    loop.abort()

    await task

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_type() == "explore"


@pytest.mark.asyncio
async def test_parent_result_append_precedes_marker_cleanup(tmp_path: Path, monkeypatch) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    def fail_cleanup(tool_call_id: str) -> None:
        del tool_call_id
        raise RuntimeError("crash after parent result")

    monkeypatch.setattr(store, "finish_agent_child", fail_cleanup)
    with pytest.raises(RuntimeError, match="crash after parent result"):
        await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    assert store.agent_children()
    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    AgentLoop(FakeBackend([]), replayed, max_turns=1)
    results = [message.tool_result for message in replayed.messages() if message.tool_result]
    assert len(results) == 1
    assert results[0].content.startswith("done")
    assert not replayed.agent_children()


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

    AgentLoop(FakeBackend([]), store, max_turns=1)

    result = next(message.tool_result for message in store.messages() if message.tool_result)
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

    AgentLoop(FakeBackend([]), store, max_turns=1)

    result = next(message.tool_result for message in store.messages() if message.tool_result)
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

    monkeypatch.setattr("zeta.agent_runner.build_backend", build)
    monkeypatch.setattr(
        "zeta.agent_runner.credential_store",
        lambda provider, **kwargs: _FakeCredentialStore(
            _ValidTokens() if tokens is None else tokens
        ),
    )
    return requested


def test_provider_for_model_maps_each_catalog_entry() -> None:
    from zeta.model_catalog import PROVIDER_MODELS, provider_for_model

    for provider, models in PROVIDER_MODELS.items():
        for model in models:
            assert provider_for_model(model) == provider


def test_provider_for_model_rejects_an_unknown_name() -> None:
    from zeta.model_catalog import provider_for_model

    with pytest.raises(ValueError) as excinfo:
        provider_for_model("gpt-nonexistent")
    message = str(excinfo.value)
    assert "gpt-nonexistent" in message
    assert "claude-opus-5" in message


def test_agent_schema_offers_every_known_model(tmp_path: Path) -> None:
    from zeta.model_catalog import known_model_names

    registry = ToolRegistry(tmp_path)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(FakeBackend([]), store, registry=registry)

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

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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

    await _collect(AgentLoop(parent_backend, store, max_turns=1).run_turn("start"))

    assert requested == [("codex", "gpt-5.4")]
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("codex done")
    # The child talked to the substitute, never to the parent's backend.
    assert len(child_backend.calls) == 1
    assert len(parent_backend.calls) == 1


@pytest.mark.asyncio
async def test_agent_model_implies_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("codex done")])])
    _stub_backend_factory(monkeypatch, child_backend)
    parent_backend = FakeBackend([ScriptedTurn(tool_calls=[_model_agent_call()])])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(parent_backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    await loop.close()


@pytest.mark.asyncio
async def test_agent_rejects_an_unknown_model_before_spawning(tmp_path: Path) -> None:
    """The schema enum catches a bad model name before the runner is reached."""

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_model_agent_call(model="gpt-nonexistent")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert "model is not an allowed value" in result.content
    assert not (store.session_dir / "agents").exists()


def test_resolve_child_backend_guards_bad_models(tmp_path: Path) -> None:
    """Second line of defence, for any caller that skips schema validation."""

    from zeta.agent_runner import resolve_child_backend

    parent_backend = FakeBackend([])
    loop = AgentLoop(parent_backend, ConversationStore(tmp_path))

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


@pytest.mark.asyncio
async def test_agent_reports_a_missing_provider_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("unreachable")])])
    _stub_backend_factory(monkeypatch, child_backend, tokens=None)
    monkeypatch.setattr(
        "zeta.agent_runner.credential_store",
        lambda provider, **kwargs: _FakeCredentialStore(None),
    )
    backend = FakeBackend([ScriptedTurn(tool_calls=[_model_agent_call()])])
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

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
    assert AgentLoop(FakeBackend([]), ConversationStore(tmp_path)).max_turns == 150


@pytest.mark.asyncio
async def test_background_start_text_names_handle_and_polling_tools(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)

    await _collect(loop.run_turn("start"))
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
    assert "task_output" not in result.content
    assert result.content.startswith("background agent started:")

    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_max_turns_raises_shared_tree_budget(tmp_path: Path) -> None:
    """max_turns lifts the shared tree budget without changing the outer loop cap."""

    call = _agent_call()
    call.arguments["max_turns"] = 60
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn([TextContent(f"step-{turn}")], tool_calls=[child_read])
            for turn in range(1, 41)
        ]
        + [ScriptedTurn([TextContent("child done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle["tree_budget"] == 60
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is False
    assert result.content.startswith("child done")


@pytest.mark.asyncio
async def test_max_turns_hard_cap_rejects_oversized_request(tmp_path: Path) -> None:
    call = _agent_call()
    call.arguments["max_turns"] = MAX_AGENT_TURN_CAP + 1
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is True
    assert f"hard cap of {MAX_AGENT_TURN_CAP}" in result.content
    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_max_turns_rejected_by_schema_for_non_positive_input(
    tmp_path: Path,
) -> None:
    """Schema-level minimum:1 catches zero/negative before the runner sees them."""

    call = _agent_call()
    call.arguments["max_turns"] = 0
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    store = ConversationStore(tmp_path)
    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is True
    assert "invalid arguments" in result.content
    assert "max_turns" in result.content


@pytest.mark.asyncio
async def test_max_turns_rejected_from_nested_agent_calls(tmp_path: Path) -> None:
    """Children inherit the tree budget; they can't override it mid-tree."""

    child = _agent_call("child")
    grandchild = _agent_call("grandchild")
    grandchild.arguments["max_turns"] = 5
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[grandchild]),
            ScriptedTurn([TextContent("recover")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    nested = next(
        message.tool_result
        for message in child_store.messages()
        if message.tool_result and message.tool_result.tool_call_id == grandchild.id
    )
    assert nested.is_error is True
    assert "max_turns is only accepted at the top-level" in nested.content


@pytest.mark.asyncio
async def test_budget_exhaustion_error_reports_used_and_allocated(
    tmp_path: Path,
) -> None:
    call = _agent_call()
    call.arguments["max_turns"] = 2
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn([TextContent(f"step-{turn}")], tool_calls=[child_read])
            for turn in range(1, 4)
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["error_code"] == "agent_turn_budget"
    assert "shared agent turn budget exhausted" in result.content
    assert "2 of 2 turns used" in result.content
    assert "agent_output" in result.content
    assert str(store.session_dir / "agents" / "1") in result.content


@pytest.mark.asyncio
async def test_child_transcript_survives_budget_exhaustion(tmp_path: Path) -> None:
    """After budget death the child work must still be readable via agent_output."""

    call = _agent_call()
    call.arguments["max_turns"] = 2
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn(
                [TextContent(f"work step {turn}")], tool_calls=[child_read]
            )
            for turn in range(1, 4)
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1)
    await _collect(loop.run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    handle = result.structured_content["child_instance_id"]

    output = await loop.tool_registry.execute(
        ToolCall("output-after-death", "agent_output", {"handle": handle})
    )

    assert output["isError"] is False
    text = output["content"][0]["text"]
    assert "assistant: work step 1" in text
    assert "assistant: work step 2" in text


@pytest.mark.asyncio
async def test_max_turns_bounded_by_hard_cap_constant() -> None:
    """The hard cap constant must be documented, positive, and above defaults."""

    assert type(MAX_AGENT_TURN_CAP) is int
    assert MAX_AGENT_TURN_CAP >= 100
    from zeta.tools.agent_presets import AGENT_PRESETS

    assert MAX_AGENT_TURN_CAP >= max(
        preset.turn_cap for preset in AGENT_PRESETS.values()
    )
