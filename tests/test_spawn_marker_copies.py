"""Per-spawn marker work stays flat as one parent spawns many background children."""

from __future__ import annotations

import asyncio
import copy
import os

import pytest

from tests.support.owned import await_or_fail
from zeta.core import agent_state
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

_PARENT_PROMPT = "spawn the fan-out"
_BATCH = 10
_SPAWNS = 70


def _markers_in(value: object) -> int:
    """Count child markers (the only state dicts naming a child session path) inside a value."""

    if type(value) is dict:
        if "child_session_path" in value:
            return 1
        return sum(_markers_in(item) for item in value.values())
    if type(value) in (list, tuple):
        return sum(_markers_in(item) for item in value)
    return 0


class _Counters:
    def __init__(self) -> None:
        self.marker_copies = 0
        self.marker_validations = 0

    def snapshot(self) -> tuple[int, int]:
        return self.marker_copies, self.marker_validations


class _FanOutBackend(CompletionBackend):
    """Parent turns each ask for ten background children; children block in their first call."""

    model, provider = "gpt-5.6-sol", "codex"

    def __init__(self, counters: _Counters) -> None:
        self.counters = counters
        self.batches: list[tuple[int, int]] = []
        self.release = asyncio.Event()
        self.next_child = 0
        self.children_started = 0

    @owned_completion
    async def complete(self, messages, tool_schemas):
        texts = [
            block.text
            for message in messages
            if message.role is MessageRole.USER
            for block in message.content
            if isinstance(block, TextContent)
        ]
        if _PARENT_PROMPT in texts:
            self.batches.append(self.counters.snapshot())
            calls = []
            while self.next_child < _SPAWNS and len(calls) < _BATCH:
                index = self.next_child
                self.next_child += 1
                calls.append(
                    ToolCall(
                        f"spawn-{index}",
                        "agent",
                        {"prompt": f"child task {index}", "description": f"child {index}", "background": True},
                    )
                )
            blocks = [ToolUseContent(call) for call in calls] if calls else [TextContent("done")]
        else:
            self.children_started += 1
            await asyncio.wait_for(self.release.wait(), 300)
            blocks = [TextContent("child done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(role=MessageRole.ASSISTANT, content=blocks),
            data={},
        )


@pytest.mark.asyncio
async def test_marker_copies_and_validations_per_spawn_do_not_grow_with_live_children(
    tmp_path, monkeypatch
) -> None:
    counters = _Counters()
    real_deepcopy = copy.deepcopy

    def counting_deepcopy(value, memo=None):
        counters.marker_copies += _markers_in(value)
        return real_deepcopy(value, memo)

    real_validate = agent_state.validate_child_resume

    def counting_validate(value):
        counters.marker_validations += 1
        return real_validate(value)

    monkeypatch.setattr(copy, "deepcopy", counting_deepcopy)
    monkeypatch.setattr(agent_state, "validate_child_resume", counting_validate)

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    backend = _FanOutBackend(counters)
    store = await asyncio.to_thread(
        ConversationStore, os.environ["ZETA_HOME"] + "/sessions", cwd=project
    )
    try:
        loop = AgentLoop(
            backend,
            store,
            skill_catalog=SkillCatalog.empty(),
            max_turns=_SPAWNS // _BATCH + 3,
        )
        try:

            async def drive() -> None:
                async for _event in loop.run_turn(_PARENT_PROMPT, origin=MessageOrigin.USER):
                    pass

            await await_or_fail(drive(), timeout=120, what="parent fan-out turn")
            assert backend.next_child == _SPAWNS
            # (copies, validations) per spawn for each batch of ten spawns.
            per_spawn = [
                tuple((after - before) / _BATCH for before, after in zip(first, second, strict=True))
                for first, second in zip(backend.batches, backend.batches[1:], strict=False)
            ]
            early, late = per_spawn[1], per_spawn[_SPAWNS // _BATCH - 1]
            assert late[0] <= early[0] + 1, f"marker deep copies per spawn grew: {per_spawn}"
            assert late[1] <= early[1] + 1, f"marker validations per spawn grew: {per_spawn}"
        finally:
            backend.release.set()
            await await_or_fail(loop.close(), timeout=120, what="close the fan-out parent")
    finally:
        store.close()
