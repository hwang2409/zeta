import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from zeta.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.tools import ToolRegistry
from zeta.types import MessageRole, StreamEvent, TextContent, ToolCall, ToolResult


async def collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


def test_policy_rules_and_default_decision() -> None:
    policy = ApprovalPolicy(
        always_allow={"read"},
        always_deny={"delete"},
        default=ApprovalDecision.DENY,
    )

    assert policy.decide("read", {}) is ApprovalDecision.ALLOW
    assert policy.decide("delete", {}) is ApprovalDecision.DENY
    assert policy.decide("other", {"value": 1}) is ApprovalDecision.DENY
    assert ApprovalPolicy(default="allow").decide("other", {}) is ApprovalDecision.ALLOW


@pytest.mark.asyncio
async def test_durable_pending_request_is_re_emitted_and_resolves_after_restart(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "echo", {"value": "approved"})
    store = ConversationStore(tmp_path, session_id="session")
    first_policy = ApprovalPolicy(default="ask")
    first_registry = ToolRegistry(
        tmp_path,
        approval_policy=first_policy,
        approval_store=store,
        register_builtin=False,
    )
    executed: list[str] = []

    async def echo(arguments: dict[str, str]) -> str:
        executed.append(arguments["value"])
        return arguments["value"]

    first_registry.register("echo", echo)
    pending_task = asyncio.create_task(first_registry.execute(call))
    await asyncio.sleep(0.06)
    pending_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_task

    restarted_store = ConversationStore(tmp_path, session_id="session")
    restarted_policy = ApprovalPolicy(default="ask", store=restarted_store)
    pending = restarted_policy.pending_requests()

    assert pending == [ApprovalRequest(call.id, call)]
    assert [entry.type for entry in restarted_store.entries][-1] == "approval_request"

    assert restarted_policy.approve(call.id)
    restarted_registry = ToolRegistry(
        tmp_path,
        approval_policy=restarted_policy,
        approval_store=restarted_store,
        register_builtin=False,
    )
    restarted_registry.register("echo", echo)
    result = await restarted_registry.execute(call)

    assert result == ToolResult(call.id, "approved")
    assert executed == ["approved"]
    assert restarted_policy.pending_requests() == []


@pytest.mark.asyncio
async def test_ask_resolution_deny_returns_error_result(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
        register_builtin=False,
    )
    registry.register("echo", lambda arguments: "must not run")
    call = ToolCall("call-1", "echo", {})
    task = asyncio.create_task(registry.execute(call))

    for _ in range(20):
        if policy.pending_requests():
            break
        await asyncio.sleep(0.01)
    assert policy.pending_requests()
    assert policy.deny(call.id)

    assert await task == ToolResult(call.id, "tool execution denied", True)


@pytest.mark.asyncio
async def test_deny_returns_error_and_loop_continues(tmp_path: Path) -> None:
    call = ToolCall("call-1", "danger", {})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn(content=[TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("danger", lambda arguments: "must not run")

    await collect(
        AgentLoop(
            backend,
            store,
            registry=registry,
            approval_policy=ApprovalPolicy(always_deny={"danger"}),
        ).run_turn("start")
    )

    result = store.messages()[2].tool_result
    assert result is not None
    assert result == ToolResult(call.id, "tool execution denied", True)
    assert store.messages()[-1].content[0].text == "done"


@pytest.mark.asyncio
async def test_abort_pending_request_cancels_and_closes_tool_call_history(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "step", {})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn(content=[TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(default="ask")
    registry = ToolRegistry(tmp_path, register_builtin=False)
    registry.register("step", lambda arguments: "ran")
    task = asyncio.create_task(
        collect(
            AgentLoop(
                backend,
                store,
                registry=registry,
                approval_policy=policy,
            ).run_turn("start")
        )
    )

    for _ in range(20):
        if policy.pending_requests():
            break
        await asyncio.sleep(0.01)
    assert policy.pending_requests()
    registry.abort()
    events = await asyncio.wait_for(task, timeout=1)

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert results == [ToolResult(call.id, "tool execution canceled", True)]
    assert policy.pending_requests() == []
    assert events[-1].type.value == "agent_end"
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL_RESULT,
        MessageRole.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_approval_and_pre_execution_hook_compose(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(always_allow={"echo"}, store=store)
    seen: list[str] = []

    def hook(name: str, arguments: dict[str, object]) -> bool:
        seen.append(name)
        return False

    registry = ToolRegistry(
        tmp_path,
        pre_execute_hook=hook,
        approval_policy=policy,
        approval_store=store,
        register_builtin=False,
    )
    registry.register("echo", lambda arguments: "ran")

    result = await registry.execute(ToolCall("call-1", "echo", {}))

    assert result == ToolResult("call-1", "tool execution denied", True)
    assert seen == ["echo"]
