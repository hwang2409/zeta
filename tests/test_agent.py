import asyncio
from pathlib import Path

import pytest

from zeta.core.abort import AbortGenerationRegistry
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tools import ToolRegistry
from zeta.tools.agent import ChildApprovalPolicy
from zeta.types import MessageRole, StreamEventType, TextContent, ToolCall, ToolUseContent


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
async def test_child_registry_isolates_session_bound_builtin_tools(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    parent_store = ConversationStore(sessions, session_id="parent", cwd=tmp_path)
    child_store = ConversationStore(sessions, session_id="child", cwd=tmp_path)
    (tmp_path / "nested").mkdir()
    registry = ToolRegistry(parent_store.cwd, session_store=parent_store)
    child_registry = registry.clone_for_session(child_store)

    await child_registry.execute(
        ToolCall(
            "child-todo",
            "todo",
            {"items": [{"content": "child work", "status": "pending"}]},
        )
    )
    await child_registry.execute(
        ToolCall("child-bash", "bash", {"cmd": "cd nested && pwd"})
    )

    assert parent_store.todo_items() == []
    assert child_store.todo_items() == [
        {"content": "child work", "status": "pending"}
    ]
    assert parent_store.bash_cwd == str(tmp_path)
    assert child_store.bash_cwd == str(tmp_path / "nested")

    await child_registry.close()
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
    assert result.structured_content == {
        "turns_used": 0,
        "child_session_path": str(store.session_dir / "agents" / "1"),
    }
    child_state = (store.session_dir / "agents" / "1" / "session_state.json").read_text()
    assert '"agent_parent"' not in child_state
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_canceled() == {
        "tool_call_id": "agent-1",
        "content": "tool execution canceled",
    }
    assert not store.agent_children()


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
    assert results[0].content == "done"
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

    AgentLoop(FakeBackend([]), store, max_turns=1)

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content == "tool execution canceled"
    assert not store.agent_children()
    assert '"agent_parent"' not in (child.state_path).read_text()
