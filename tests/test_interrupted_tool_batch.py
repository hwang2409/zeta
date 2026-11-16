"""A process that dies mid tool batch must not make the next turn repeat writes."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

from tests.support.branch_reads import branch_messages
from tests.support.owned import await_or_fail
from zeta.agent.tool_results import validated_tool_result
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
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
    ToolResult,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.completion import owned_completion
from zeta.providers.ollama import _messages as ollama_messages
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop import agent as agent_module
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry

TITLES = ("Alpha", "Beta", "Gamma")
NOT_RUN = "not run: the turn was interrupted"
MAYBE_RUN = "interrupted: may or may not have taken effect; check before retrying"


class AddTaskModel(CompletionBackend):
    """Scripted model that adds only titles it cannot see as added.

    It reads the real provider request (the Codex Responses payload). A
    title with no result, or with a "not run" result, is added again; a title
    with a real result or a "may or may not" result is not.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[list[dict[str, Any]]] = []
        self.anthropic: list[list[dict[str, Any]]] = []
        self.ollama: list[list[dict[str, Any]]] = []
        self.turn = 0

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        items = build_responses_payload(messages, tool_schemas, model="gpt-5")["input"]
        self.requests.append(items)
        self.anthropic.append(
            build_messages_payload(
                messages, tool_schemas, model="claude-sonnet-4-5", max_tokens=10_000
            )["messages"]
        )
        self.ollama.append(ollama_messages(messages))
        titles = {
            item["call_id"]: json.loads(item["arguments"])["title"]
            for item in items
            if item.get("type") == "function_call"
        }
        seen = {
            titles[item["call_id"]]
            for item in items
            if item.get("type") == "function_call_output" and item["output"] != NOT_RUN
        }
        last_user = max(
            index for index, item in enumerate(items) if item.get("role") == "user"
        )
        first_request = not any(
            item.get("type") == "function_call" for item in items[last_user:]
        )
        missing = [title for title in TITLES if title not in seen]
        if first_request and missing:
            content = [
                ToolUseContent(
                    ToolCall(f"{self.name}-{self.turn}-{title}", "add_task", {"title": title})
                )
                for title in missing
            ]
        else:
            content = [TextContent("done")]
        self.turn += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in content:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, content),
        )


def _registry(
    root: Path, ran: list[str], *, parallel: bool, added: asyncio.Event
) -> ToolRegistry:
    registry = ToolRegistry(
        root, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )

    def add_task(arguments: dict[str, str]) -> str:
        ran.append(arguments["title"])
        added.set()
        return f"added {arguments['title']}"

    registry.register("add_task", add_task, parallel_safe=parallel)
    return registry


async def _die_at_beta_approval(
    root: Path,
    dead: Path,
    ran: list[str],
    *,
    parallel: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one batch, approve Alpha, and copy the session at Beta's approval.

    The copy is what SIGKILL leaves on disk: the live process never writes again.
    It is taken once the dispatcher has taken Alpha's result, while Beta waits.
    """

    alpha_taken = asyncio.Event()

    def validated(result: object, call_id: str) -> ToolResult:
        checked = validated_tool_result(result, call_id)
        if call_id.endswith("-Alpha"):
            alpha_taken.set()
        return checked

    monkeypatch.setattr(agent_module, "_validated_tool_result", validated)

    store = ConversationStore(root, session_id="session")
    policy = ApprovalPolicy(default="ask", store=store)
    alpha_added = asyncio.Event()
    registry = _registry(root, ran, parallel=parallel, added=alpha_added)
    loop = AgentLoop(
        AddTaskModel("dead"),
        store,
        registry=registry,
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
    )
    beta_waiting = asyncio.Event()

    async def drive() -> None:
        async with aclosing(
            loop.run_turn("add Alpha, Beta, Gamma", origin=MessageOrigin.USER)
        ) as events:
            async for event in events:
                if event.type is not StreamEventType.TOOL_APPROVAL_START:
                    continue
                assert event.tool_call is not None
                if event.tool_call.arguments["title"] == "Alpha":
                    assert policy.approve(event.tool_call.id)
                elif event.tool_call.arguments["title"] == "Beta":
                    beta_waiting.set()

    turn = asyncio.create_task(drive())
    try:
        await await_or_fail(beta_waiting.wait(), what="Beta's approval request")
        await await_or_fail(alpha_added.wait(), what="Alpha's execution")
        await await_or_fail(alpha_taken.wait(), what="Alpha's result")
        shutil.copytree(root, dead)
    finally:
        turn.cancel()
        await await_or_fail(
            asyncio.gather(turn, return_exceptions=True), what="dead turn cancel"
        )
        await loop.close()
        await registry.close()
        store.close()
    assert ran == ["Alpha"]


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "parallel"])
async def test_continue_after_a_dead_batch_does_not_repeat_completed_calls(
    tmp_path: Path, parallel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    dead = tmp_path / "dead"
    await await_or_fail(
        _die_at_beta_approval(
            tmp_path / "live", dead, ran, parallel=parallel, monkeypatch=monkeypatch
        ),
        what="batch reaching Beta's approval",
    )

    store = ConversationStore(dead, session_id="session")
    stored = [
        (message.tool_result.tool_call_id, message.tool_result.content)
        for message in branch_messages(store)
        if message.tool_result is not None
    ]
    # Alpha finished before the process died, so its real result is on disk.
    assert stored == [("dead-0-Alpha", "added Alpha")], stored
    policy = ApprovalPolicy(default="allow", store=store)
    registry = _registry(dead, ran, parallel=parallel, added=asyncio.Event())
    model = AddTaskModel("resumed")
    loop = AgentLoop(
        model,
        store,
        registry=registry,
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        async with aclosing(loop.run_turn("continue", origin=MessageOrigin.USER)) as events:
            await await_or_fail(
                _drain(events), what="resumed turn after continue"
            )

        # A parallel retry may run Beta and Gamma in either order.
        assert sorted(ran) == list(TITLES), "a completed call ran again"
        request = model.requests[0]
        batch = [
            (item["type"], item["call_id"], item.get("output"))
            for item in request
            if item.get("call_id", "").startswith("dead-")
        ]
        assert batch == [
            ("function_call", "dead-0-Alpha", None),
            ("function_call", "dead-0-Beta", None),
            ("function_call", "dead-0-Gamma", None),
            ("function_call_output", "dead-0-Alpha", "added Alpha"),
            ("function_call_output", "dead-0-Beta", NOT_RUN),
            ("function_call_output", "dead-0-Gamma", NOT_RUN),
        ]

        # "continue" closed the dead turn's requests: approving Beta now cannot
        # run it a second time.
        assert await policy.pending_requests() == []
        assert not policy.approve("dead-0-Beta")
        assert sorted(ran) == list(TITLES)
    finally:
        await loop.close()
        await registry.close()
        store.close()


async def _drain(events: AsyncIterator[StreamEvent]) -> None:
    async for _event in events:
        pass


@pytest.mark.asyncio
async def test_live_batch_with_a_late_approval_anchor_stays_one_paired_batch(
    tmp_path: Path,
) -> None:
    """A stricter policy mid-batch stores an approval anchor inside the batch.

    The anchor repeats a call of the batch; it must not split the batch or
    send that call twice, on any provider.
    """

    store = ConversationStore(tmp_path, session_id="session")
    policy = ApprovalPolicy(default="allow", store=store)
    ran: list[str] = []
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )

    def add_task(arguments: dict[str, str]) -> str:
        ran.append(arguments["title"])
        policy.default = ApprovalDecision.ASK
        return f"added {arguments['title']}"

    registry.register("add_task", add_task)
    model = AddTaskModel("live")
    loop = AgentLoop(
        model,
        store,
        registry=registry,
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        async with aclosing(
            loop.run_turn("add Alpha, Beta, Gamma", origin=MessageOrigin.USER)
        ) as events:

            async def answer() -> None:
                async for event in events:
                    if event.type is StreamEventType.TOOL_APPROVAL_START:
                        assert event.tool_call is not None
                        assert policy.approve(event.tool_call.id)

            await await_or_fail(answer(), what="live turn")

        assert ran == list(TITLES)
        ids = [f"live-0-{title}" for title in TITLES]
        after_batch = model.requests[1]
        assert [
            (item["type"], item["call_id"], item.get("output"))
            for item in after_batch
            if item.get("call_id", "").startswith("live-")
        ] == [
            *(("function_call", call_id, None) for call_id in ids),
            *(
                ("function_call_output", call_id, f"added {title}")
                for call_id, title in zip(ids, TITLES, strict=True)
            ),
        ]
        uses = [
            block["id"]
            for message in model.anthropic[1]
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        results = [
            block["tool_use_id"]
            for message in model.anthropic[1]
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert uses == ids
        assert results == ids
        assert len(model.ollama) == 2
    finally:
        await loop.close()
        await registry.close()
        store.close()
