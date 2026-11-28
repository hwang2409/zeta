"""Background spawns through the real `agent` tool do not re-read the parent log.

One parent turn spawns many background children; each child blocks in its
first provider call. A linear parent append or durable-writer reload used to
re-decode the whole parent branch, so row decodes per spawn grew with the
number of children already spawned.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

import zeta.runtime.loop.agent as loop_agent
from tests.support.owned import await_or_fail
from zeta.core.store import ConversationStore
from zeta.core.store._rows import ResidentRows
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

SPAWNS = 120
BATCH = 10
PARENT_PROMPT = "spawn the children"


class _SpawningBackend(CompletionBackend):
    """The parent asks for BATCH background children per turn; children wait."""

    model, provider = "gpt-5.6-sol", "codex"

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.next_child = 0
        self.children_started = 0

    @owned_completion
    async def complete(self, messages, tool_schemas):
        is_parent = any(
            block.text == PARENT_PROMPT
            for message in messages
            if message.role is MessageRole.USER
            for block in message.content
            if isinstance(block, TextContent)
        )
        if is_parent:
            calls = []
            while self.next_child < SPAWNS and len(calls) < BATCH:
                index = self.next_child
                self.next_child += 1
                calls.append(
                    ToolCall(
                        f"spawn-{index}",
                        "agent",
                        {
                            "prompt": f"child task {index}",
                            "description": f"child {index}",
                            "background": True,
                        },
                    )
                )
            blocks = [ToolUseContent(call) for call in calls] or [TextContent("done")]
        else:
            self.children_started += 1
            await await_or_fail(
                self.release.wait(), timeout=600, what="test released children"
            )
            blocks = [TextContent("child done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(role=MessageRole.ASSISTANT, content=blocks),
            data={},
        )


def _in_context_preparation() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename.endswith(
            os.path.join("context_parts", "preparation.py")
        ):
            return True
        frame = frame.f_back
    return False


async def test_row_decodes_per_background_spawn_stay_flat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = [0]
    decodes: dict[int, int] = {}
    original_tool = loop_agent.run_agent_tool
    original_decode = ResidentRows._decode

    async def counted_tool(*args, **kwargs):
        spawned[0] += 1
        return await original_tool(*args, **kwargs)

    def counted_decode(self, fd, ref):
        # Parent context preparation reads the branch once per parent turn
        # (not per spawn); it is outside the append path this test pins.
        if not _in_context_preparation():
            decodes[spawned[0]] = decodes.get(spawned[0], 0) + 1
        return original_decode(self, fd, ref)

    monkeypatch.setattr(loop_agent, "run_agent_tool", counted_tool)
    # test-contract: complexity row decodes per spawn are the O(n) cost this pins
    monkeypatch.setattr(ResidentRows, "_decode", counted_decode)

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    backend = _SpawningBackend()
    store = await asyncio.to_thread(
        ConversationStore, Path(os.environ["ZETA_HOME"]) / "sessions", cwd=project
    )
    loop = AgentLoop(
        backend,
        store,
        skill_catalog=SkillCatalog.empty(),
        max_turns=SPAWNS // BATCH + 5,
    )
    try:

        async def run_turn() -> None:
            async for _event in loop.run_turn(PARENT_PROMPT, origin=MessageOrigin.USER):
                pass

        await await_or_fail(run_turn(), timeout=600, what="parent spawn turn")
    finally:
        backend.release.set()
        await await_or_fail(loop.close(), timeout=300, what="loop close")
        store.close()

    assert spawned[0] == SPAWNS
    early = sum(decodes.get(k, 0) for k in range(41, 51))
    late = sum(decodes.get(k, 0) for k in range(SPAWNS - 9, SPAWNS + 1))
    # Bounded by a small constant: 5 decodes per spawn, independent of k.
    assert late <= early + 5 * 10, (
        f"row decodes over spawns 40-49: {early}, over spawns "
        f"{SPAWNS - 10}-{SPAWNS - 1}: {late}"
    )
