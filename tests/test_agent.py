import asyncio
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.types import MessageRole, StreamEventType, TextContent, ToolCall


async def _collect(events):
    return [event async for event in events]


def _agent_call(call_id: str = "agent-1") -> ToolCall:
    return ToolCall(
        call_id,
        "agent",
        {"prompt": "inspect the task", "description": "task research"},
    )


@pytest.mark.asyncio
async def test_agent_returns_child_text_and_persists_child_session(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content == "done"
    child_dir = store.session_dir / "agents" / "1"
    assert (child_dir / "conversation.jsonl").exists()
    assert [message.role for message in ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert "agent" not in {
        schema["name"] for schema in backend.calls[1][1]
    }


@pytest.mark.asyncio
async def test_agent_turn_cap_returns_loud_error(tmp_path: Path) -> None:
    child_call = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()])]
        + [ScriptedTurn(tool_calls=[child_call]) for _ in range(25)]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.is_error
    assert "25-turn cap" in result.content
    assert "partial state is saved" in result.content


@pytest.mark.asyncio
async def test_child_agent_call_is_rejected(tmp_path: Path) -> None:
    nested = _agent_call("nested")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("nested rejected")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content == "nested rejected"
    child_result = next(
        message.tool_result
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
        if message.tool_result
    )
    assert child_result.is_error
    assert "unknown tool: agent" in child_result.content


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
            policy.deny(child_call.id)

    assert any(event.type is StreamEventType.TOOL_APPROVAL_END for event in events)
    assert not policy.pending_requests()
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    denied = next(message.tool_result for message in child_messages if message.tool_result)
    assert denied.is_error
    assert denied.content == "tool execution denied"


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

    await task
    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content == "tool execution canceled"
    child_state = (store.session_dir / "agents" / "1" / "session_state.json").read_text()
    assert '"agent_parent"' not in child_state
    assert not store.agent_children()


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

    AgentLoop(FakeBackend([]), store, max_turns=1)

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content == "tool execution canceled"
    assert not store.agent_children()
    assert '"agent_parent"' not in (child.state_path).read_text()
