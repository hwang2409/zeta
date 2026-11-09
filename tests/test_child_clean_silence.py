from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.agent.presets import AgentPreset
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.skills.agent_catalog import AgentCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent import _read_agent_status

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")


def _message_events(
    blocks: list[TextContent | ThinkingContent | ToolUseContent],
) -> AsyncIterator[StreamEvent]:
    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )

    return events()


class _ParentBackend(CompletionBackend):
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.calls = 0

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.calls += 1
        blocks: list[TextContent | ToolUseContent]
        if self.calls == 1:
            blocks = [
                ToolUseContent(
                    ToolCall(
                        "child",
                        "agent",
                        {
                            "prompt": "own live work",
                            "description": "silence regression child",
                            "agent_type": self.profile,
                            "background": True,
                        },
                    )
                )
            ]
        else:
            blocks = [TextContent("parent remains available")]
        async for event in _message_events(blocks):
            yield event


class _ChildBackend(CompletionBackend):
    def __init__(self, ending: str) -> None:
        self.ending = ending
        self.calls = 0
        self.work_started = asyncio.Event()
        self.release_initial = asyncio.Event()
        self.notification_called = asyncio.Event()
        self.notification_ready = asyncio.Event()
        self.release_notification = asyncio.Event()

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.calls += 1
        if self.calls == 1:
            if self.ending == "low_capacity":
                blocks: list[TextContent | ThinkingContent | ToolUseContent] = [
                    ToolUseContent(
                        ToolCall(
                            f"grandchild-{index}",
                            "agent",
                            {
                                "prompt": f"grandchild {index}",
                                "description": "d" * 120,
                                "agent_type": "explicit-test",
                                "background": True,
                            },
                        )
                    )
                    for index in (1, 2)
                ]
            else:
                command = f"{shlex.quote(sys.executable)} -c {shlex.quote('input()')}"
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            f"task-{index}",
                            "run_background",
                            {"command": command},
                        )
                    )
                    for index in (1, 2)
                ]
        elif self.calls == 2:
            self.work_started.set()
            if self.ending == "non_explicit":
                await await_or_fail(
                    self.release_initial.wait(), what="plain child initial response"
                )
            blocks = [TextContent("working")]
        else:
            self.notification_called.set()
            thinking = ThinkingContent("still working")
            yield StreamEvent(StreamEventType.MESSAGE_START)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=thinking)
            self.notification_ready.set()
            await await_or_fail(
                self.release_notification.wait(), what="notification response release"
            )
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [thinking]),
            )
            return
        async for event in _message_events(blocks):
            yield event


class _GrandchildBackend(CompletionBackend):
    def __init__(self, index: int, *, rich_report: bool) -> None:
        self.index = index
        self.rich_report = rich_report
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.started.set()
        await await_or_fail(self.release.wait(), what=f"grandchild {self.index} release")
        report = {"status": "done", "summary": "done"}
        if self.rich_report:
            report.update(
                summary="x" * 256, verdict="pass", head_sha="a" * 40
            )
        blocks = [
            ToolUseContent(
                ToolCall(
                    f"finish-grandchild-{self.index}", "finish", report
                )
            )
        ]
        async for event in _message_events(blocks):
            yield event


def _catalog() -> AgentCatalog:
    return AgentCatalog(
        (
            AgentPreset(
                name="explicit-test",
                preamble="test explicit child",
                selection_guidance="test only",
                tool_names=None,
                accepts_follow_ups=True,
                explicit_finish=True,
            ),
            AgentPreset(
                name="plain-test",
                preamble="test plain child",
                selection_guidance="test only",
                tool_names=None,
                accepts_follow_ups=True,
                explicit_finish=False,
            ),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ending", "profile", "expected_state"),
    [
        ("thinking_only", "explicit-test", "idle"),
        ("low_capacity", "explicit-test", "idle"),
        ("non_explicit", "plain-test", "failed"),
    ],
)
async def test_real_child_clean_silence_preserves_only_explicit_finish_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
    profile: str,
    expected_state: str,
) -> None:
    parent_backend = _ParentBackend(profile)
    child_backend = _ChildBackend(ending)
    grandchildren = [
        _GrandchildBackend(index, rich_report=True) for index in (1, 2)
    ]
    store = ConversationStore(tmp_path, cwd=tmp_path)
    child_registries: list[ToolRegistry] = []
    original_clone = ToolRegistry.clone_for_session

    def capture_child_registry(
        registry: ToolRegistry, *args: object, **kwargs: object
    ) -> ToolRegistry:
        child = original_clone(registry, *args, **kwargs)
        child_registries.append(child)
        return child

    def child_backend_for(loop: AgentLoop, _model: object):
        return (
            (child_backend, None)
            if loop.agent_depth == 0
            else (grandchildren.pop(0), None)
        )

    monkeypatch.setattr(ToolRegistry, "clone_for_session", capture_child_registry)
    monkeypatch.setattr(
        "zeta.agent.runner.resolve_child_backend", child_backend_for
    )
    loop = AgentLoop(
        parent_backend,
        store,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=_catalog(),
        token_budget=8_192,
        max_turns=2,
    )
    loop.explicit_finish = True
    idle_count = 0
    initial_idle = asyncio.Event()
    silence_idle = asyncio.Event()
    silence_done = asyncio.Event()
    child_ended = asyncio.Event()
    def observe(event: StreamEvent) -> None:
        nonlocal idle_count
        if (
            event.type is StreamEventType.AGENT_NOTIFICATION
            and event.data.get("kind") == "child_idle"
        ):
            idle_count += 1
            if idle_count == 1:
                initial_idle.set()
            else:
                silence_idle.set()
                silence_done.set()
        if (
            event.type is StreamEventType.TOOL_EXECUTION_END
            and event.tool_call
            and event.tool_call.id == "child"
        ):
            child_ended.set()
            silence_done.set()

    loop.set_background_event_sink(observe)
    first, second = grandchildren
    try:
        await await_or_fail(
            _collect(loop.run_turn("start", origin=MessageOrigin.USER)),
            what=f"parent starts {ending} child",
        )
        handle = next(iter(store.agent_children()))
        flow = (
            loop._background_owner.conversation_channel._delivery_targets[handle].flow
            if ending == "low_capacity"
            else None
        )
        await await_or_fail(child_backend.work_started.wait(), what="live child work")
        if ending != "non_explicit":
            await await_or_fail(initial_idle.wait(), what="child initial idle")
        if ending == "low_capacity":
            await await_or_fail(first.started.wait(), what="first grandchild starts")
            await await_or_fail(second.started.wait(), what="second grandchild starts")
            first.release.set()
            second.release.set()
        else:
            registry = child_registries[0]
            running = [
                task for task in registry.background_tasks.snapshot() if task.running
            ]
            assert len(running) == 2
            await await_or_fail(
                registry.background_tasks.input(
                    running[0].task_id, "done\n", eof=True
                ),
                what="first background task exit",
            )
            if ending == "thinking_only":
                await await_or_fail(
                    child_backend.notification_ready.wait(),
                    what="thinking-only notification response",
                )
            if ending in {"thinking_only", "non_explicit"}:
                await await_or_fail(
                    registry.background_tasks.input(
                        running[1].task_id, "done\n", eof=True
                    ),
                    what="second background task exit",
                )
            if ending == "non_explicit":
                await await_or_fail(
                    asyncio.gather(
                        *(record.process.wait() for record in registry.background_tasks.records)
                    ),
                    what="background tasks finish",
                )
                child_backend.release_initial.set()
        if ending == "non_explicit":
            await await_or_fail(
                child_backend.notification_ready.wait(), what="plain notification turn"
            )
        if ending in {"thinking_only", "non_explicit"}:
            child_backend.release_notification.set()
        if ending == "non_explicit":
            await await_or_fail(child_ended.wait(), what="plain child failure")
        else:
            await await_or_fail(silence_done.wait(), what="clean silence result")
            if ending == "thinking_only":
                assert child_backend.notification_called.is_set()
            else:
                assert flow is not None
                assert flow._low_capacity_notifications is True
        rows = _read_agent_status(
            store,
            requested_handle=handle,
            running_handles=loop._background_owner.active_instance_ids,
        )
        assert rows[0]["state"] == expected_state
    finally:
        child_backend.release_initial.set()
        child_backend.release_notification.set()
        first.release.set()
        second.release.set()
        active = loop._background_owner.active_instance_ids
        if active:
            handle = min(active, key=lambda item: item.count(":"))
            result = await await_or_fail(
                loop.tool_registry.execute(
                    ToolCall("cancel-child", "agent_cancel", {"handle": handle})
                ),
                what="agent_cancel cleanup",
            )
            assert result["structuredContent"]["status"] in {
                "cancellation_confirmed",
                "already_finished",
            }
        await loop.close()
        store.close()


async def _collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]
