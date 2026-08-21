import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from zeta.core.abort import AbortGenerationRegistry
from zeta.core.approval import ApprovalDecision, ApprovalGate, ApprovalPolicy, ApprovalRequest
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationIntegrityError, ConversationStore
from zeta.tools import ToolAbortSignal, ToolRegistry
from zeta.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


async def collect(events: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in events]


def test_abort_generation_registry_is_monotonic_and_sticky() -> None:
    registry = AbortGenerationRegistry()
    first = registry.new_generation()
    second = registry.new_generation()

    assert (first.generation, second.generation) == (1, 2)
    first.abort()
    first.abort()

    assert first.is_set()
    assert not second.is_set()


@pytest.mark.asyncio
async def test_approval_gate_is_directly_testable(tmp_path: Path) -> None:
    called = False

    def hook(name: str, arguments: dict[str, object]) -> bool:
        nonlocal called
        del name, arguments
        called = True
        return True

    gate = ApprovalGate(
        ApprovalPolicy(always_deny={"danger"}, store=ConversationStore(tmp_path)),
        hook,
    )
    signal = AbortGenerationRegistry().new_generation()
    call = ToolCall("gate-call", "danger", {})

    result, execution_signal = await gate.run(
        call,
        {},
        signal,
        lambda current: current,
    )

    assert result == ToolResult(call.id, "tool execution denied", True)
    assert execution_signal is signal
    assert not called


@pytest.fixture(scope="module")
def approval_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("approval")


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
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
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
    assert restarted_store.entries[-1].type == "message"
    assert restarted_store.entries[-1].data["approval_requests"]

    assert restarted_policy.approve(call.id)
    restarted_registry = ToolRegistry(
        tmp_path,
        approval_policy=restarted_policy,
        approval_store=restarted_store,
        register_builtin=False,
    )
    restarted_registry.register("echo", echo)
    result = await restarted_registry.execute(call)

    assert result["isError"] is False
    assert result["content"][0]["text"] == "approved"
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
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    task = asyncio.create_task(registry.execute(call))

    for _ in range(20):
        if policy.pending_requests():
            break
        await asyncio.sleep(0.01)
    assert policy.pending_requests()
    assert policy.deny(call.id)

    result = await task
    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution denied"


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

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution denied"
    assert seen == ["echo"]


@pytest.mark.asyncio
async def test_torn_request_write_does_not_leave_a_tool_call_or_start_event(
    approval_root: Path,
) -> None:
    call = ToolCall("call-1", "echo", {})
    store = ConversationStore(approval_root)
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    policy = ApprovalPolicy(default="ask", store=store)
    registry = ToolRegistry(approval_root, approval_policy=policy, register_builtin=False)
    registry.register("echo", lambda arguments: "must not run")
    original_write = store._write_line

    def torn_write(row: dict[str, object]) -> None:
        if row.get("type") == "message" and row.get("data", {}).get(
            "approval_requests"
        ):
            with store.path.open("ab") as handle:
                handle.write(b'{"seq":2,"id":"torn"')
                handle.flush()
            raise RuntimeError("simulated crash")
        original_write(row)

    store._write_line = torn_write  # type: ignore[method-assign]
    events: list[StreamEvent] = []
    with pytest.raises(RuntimeError, match="simulated crash"):
        async for event in AgentLoop(
            backend,
            store,
            registry=registry,
            approval_policy=policy,
        ).run_turn("start"):
            events.append(event)

    restarted = ConversationStore(approval_root, session_id=store.session_id)
    assert not any(event.type is StreamEventType.TOOL_EXECUTION_START for event in events)
    assert all(
        not any(isinstance(block, ToolUseContent) for block in message.content)
        for message in restarted.messages()
    )
    assert ApprovalPolicy(default="ask", store=restarted).pending_requests() == []


def test_resolution_rejects_a_parent_outside_the_request_ancestry(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="invalid-resolution-parent")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_approval_request(
        "call-1",
        ToolCall("call-1", "echo", {}),
        parent_id=root.id,
    )

    with pytest.raises(ConversationIntegrityError, match="parent"):
        store.append_approval_resolution(
            "call-1",
            "allow",
            parent_id=root.id,
        )

    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert not any(
        entry.type == "approval_resolution" for entry in reopened.entries
    )


def test_two_stores_do_not_abort_a_request_before_atomic_anchor_append(
    approval_root: Path,
) -> None:
    store_a = ConversationStore(approval_root, session_id="cross-store-race")
    store_b = ConversationStore(approval_root, session_id="cross-store-race")
    root = store_a.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call = ToolCall("cross-store-call", "echo", {})
    original_append = store_a.append_message_with_approval_requests
    started = Event()
    release = Event()

    def delayed_append(*args: object, **kwargs: object):
        started.set()
        assert release.wait(timeout=1)
        return original_append(*args, **kwargs)

    store_a.append_message_with_approval_requests = delayed_append  # type: ignore[method-assign]
    with ThreadPoolExecutor(max_workers=1) as executor:
        append_task = executor.submit(
            store_a.append_message_with_approval_requests,
            Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
            [(call.id, call)],
            parent_id=root.id,
        )
        assert started.wait(timeout=1)
        assert not store_b.resolve_approval(call.id, "abort")
        release.set()
        append_task.result(timeout=1)

    assert store_a.approval_states() == store_b.approval_states()
    assert store_a.approval_states()[call.id] == (call, None)


def test_two_stores_dedupe_concurrent_duplicate_request_append(
    approval_root: Path,
) -> None:
    session_id = "cross-store-duplicate"
    store_a = ConversationStore(approval_root, session_id=session_id)
    root = store_a.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store_b = ConversationStore(approval_root, session_id=session_id)
    call = ToolCall("duplicate-call", "echo", {})

    def append(store: ConversationStore) -> object:
        return store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
            [(call.id, call)],
            parent_id=root.id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        entries = list(executor.map(append, (store_a, store_b)))

    reopened = ConversationStore(approval_root, session_id=session_id)
    assert entries[0].id == entries[1].id
    assert reopened.approval_states() == {call.id: (call, None)}


@pytest.mark.asyncio
async def test_early_exit_paths_close_pending_requests(approval_root: Path) -> None:
    cases = ("unknown", "invalid", "pre-aborted")
    for case in cases:
        store = ConversationStore(approval_root, session_id=f"early-{case}")
        policy = ApprovalPolicy(default="ask", store=store)
        call = ToolCall(f"{case}-call", "echo", {})
        store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
            [(call.id, call)],
        )
        signal = ToolAbortSignal()
        registry = ToolRegistry(
            approval_root,
            approval_policy=policy,
            approval_store=store,
            abort_signal=signal,
            register_builtin=False,
        )
        if case == "invalid":
            registry.register(
                "echo",
                lambda arguments: "must not run",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )
        if case == "pre-aborted":
            signal.abort()

        result = await registry.execute(call)

        assert result["isError"] is True
        assert policy.pending_requests() == []
        assert store.approval_states()[call.id][1] == "abort"


@pytest.mark.asyncio
async def test_abort_race_honors_an_approval_that_wins_atomically(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="abort-race")
    call = ToolCall("race-call", "echo", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy = ApprovalPolicy(default="ask", store=store)
    signal = ToolAbortSignal()
    registry = ToolRegistry(
        approval_root,
        approval_policy=policy,
        approval_store=store,
        abort_signal=signal,
        register_builtin=False,
    )
    executed: list[str] = []

    async def echo(arguments: dict[str, object], abort_signal: ToolAbortSignal) -> str:
        assert not abort_signal.is_set()
        executed.append("ran")
        return "ran"

    registry.register("echo", echo)
    original_resolve = store.resolve_approval

    def approve_when_abort_resolves(request_id: str, decision: str) -> bool:
        if decision == "abort":
            assert original_resolve(request_id, "allow")
            return False
        return original_resolve(request_id, decision)

    store.resolve_approval = approve_when_abort_resolves  # type: ignore[method-assign]
    task = asyncio.create_task(registry.execute(call))
    await asyncio.sleep(0.06)
    signal.abort()

    result = await asyncio.wait_for(task, timeout=1)
    assert result["isError"] is False
    assert result["content"][0]["text"] == "ran"
    assert store.approval_states()[call.id][1] == "allow"
    assert executed == ["ran"]


@pytest.mark.asyncio
async def test_second_abort_reaches_handler_after_approval_wins(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="second-abort")
    call = ToolCall("second-abort-call", "echo", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy = ApprovalPolicy(default="ask", store=store)
    registry = ToolRegistry(
        approval_root,
        approval_policy=policy,
        approval_store=store,
        register_builtin=False,
    )
    started = asyncio.Event()
    observed_cancellation = asyncio.Event()

    async def echo(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        del arguments
        started.set()
        await abort_signal.wait()
        observed_cancellation.set()
        return "canceled"

    registry.register("echo", echo)
    original_resolve = store.resolve_approval

    def approve_when_abort_resolves(request_id: str, decision: str) -> bool:
        if decision == "abort":
            assert original_resolve(request_id, "allow")
            return False
        return original_resolve(request_id, decision)

    store.resolve_approval = approve_when_abort_resolves  # type: ignore[method-assign]
    task = asyncio.create_task(registry.execute(call))
    await asyncio.sleep(0.06)
    registry.abort()
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not task.done()

    registry.abort()

    result = await asyncio.wait_for(task, timeout=1)
    assert result["isError"] is False
    assert result["content"][0]["text"] == "canceled"
    assert observed_cancellation.is_set()
    assert store.approval_states()[call.id][1] == "allow"


@pytest.mark.asyncio
async def test_pre_aborted_approved_call_accepts_a_second_abort(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="pre-aborted-approved")
    call = ToolCall("pre-aborted-approved-call", "echo", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy = ApprovalPolicy(default="ask", store=store)
    assert policy.approve(call.id)
    signal = ToolAbortSignal()
    signal.abort()
    registry = ToolRegistry(
        approval_root,
        approval_policy=policy,
        approval_store=store,
        abort_signal=signal,
        register_builtin=False,
    )
    started = asyncio.Event()
    canceled = asyncio.Event()

    async def echo(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        del arguments
        started.set()
        await abort_signal.wait()
        canceled.set()
        return "canceled"

    registry.register("echo", echo)
    task = asyncio.create_task(registry.execute(call))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not task.done()

    registry.abort()

    result = await asyncio.wait_for(task, timeout=1)
    assert result["isError"] is False
    assert result["content"][0]["text"] == "canceled"
    assert canceled.is_set()


@pytest.mark.asyncio
async def test_execute_many_pre_aborted_approved_calls_share_one_abort_generation(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="parallel-pre-aborted")
    calls = [
        ToolCall("parallel-approved-a", "echo", {"call_id": "parallel-approved-a"}),
        ToolCall("parallel-approved-b", "echo", {"call_id": "parallel-approved-b"}),
    ]
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [*(ToolUseContent(call) for call in calls)]),
        [(call.id, call) for call in calls],
    )
    policy = ApprovalPolicy(default="ask", store=store)
    assert all(policy.approve(call.id) for call in calls)
    signal = ToolAbortSignal()
    signal.abort()
    registry = ToolRegistry(
        approval_root,
        approval_policy=policy,
        approval_store=store,
        abort_signal=signal,
        register_builtin=False,
    )
    started = asyncio.Event()
    started_count = 0
    canceled: set[str] = set()

    async def echo(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        nonlocal started_count
        call_id = str(arguments["call_id"])
        started_count += 1
        if started_count == len(calls):
            started.set()
        await abort_signal.wait()
        canceled.add(call_id)
        return "canceled"

    registry.register("echo", echo, parallel_safe=True)
    task = asyncio.create_task(registry.execute_many(calls))
    await asyncio.wait_for(started.wait(), timeout=1)

    registry.abort()

    results = await asyncio.wait_for(task, timeout=1)
    assert [result["content"][0]["text"] for result in results] == [
        "canceled",
        "canceled",
    ]
    assert all(result["isError"] is False for result in results)
    assert canceled == {call.id for call in calls}


@pytest.mark.asyncio
async def test_execute_many_pending_parallel_approval_abort_wakes_every_waiter(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="parallel-pending-abort")
    calls = [
        ToolCall("pending-parallel-a", "echo", {}),
        ToolCall("pending-parallel-b", "echo", {}),
    ]
    policy = ApprovalPolicy(default="ask", store=store)
    registry = ToolRegistry(
        approval_root,
        approval_policy=policy,
        approval_store=store,
        register_builtin=False,
    )
    registry.register("echo", lambda arguments: "must not run", parallel_safe=True)

    task = asyncio.create_task(registry.execute_many(calls))
    for _ in range(100):
        if {request.request_id for request in policy.pending_requests()} == {
            call.id for call in calls
        }:
            break
        await asyncio.sleep(0.01)
    assert {request.request_id for request in policy.pending_requests()} == {
        call.id for call in calls
    }

    registry.abort()

    results = await asyncio.wait_for(task, timeout=1)
    assert [result["content"][0]["text"] for result in results] == [
        "tool execution canceled",
        "tool execution canceled",
    ]
    assert all(result["isError"] is True for result in results)
    assert policy.pending_requests() == []


@pytest.mark.asyncio
async def test_pending_request_wins_over_policy_change_after_restart(
    approval_root: Path,
) -> None:
    call = ToolCall("call-1", "echo", {})
    first_store = ConversationStore(approval_root, session_id="session")
    first_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    restarted_store = ConversationStore(approval_root, session_id="session")
    changed_policy = ApprovalPolicy(always_allow={"echo"}, store=restarted_store)
    registry = ToolRegistry(
        approval_root,
        approval_policy=changed_policy,
        approval_store=restarted_store,
        register_builtin=False,
    )
    executed: list[bool] = []
    registry.register("echo", lambda arguments: executed.append(True) or "ran")
    task = asyncio.create_task(registry.execute(call))

    await asyncio.sleep(0.1)
    assert not task.done()
    assert executed == []
    assert changed_policy.pending_requests()

    assert changed_policy.approve(call.id)
    result = await asyncio.wait_for(task, timeout=1)
    assert result["isError"] is False
    assert result["content"][0]["text"] == "ran"
    assert executed == [True]


def test_abandoned_branch_approval_does_not_leak_into_active_branch(
    approval_root: Path,
) -> None:
    call = ToolCall("dead-call", "echo", {})
    store = ConversationStore(approval_root)
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_approval_request(call.id, call, parent_id=root.id)
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("active")]), parent_id=root.id)

    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert reopened.pending_approvals() == []
    assert ApprovalPolicy(default="ask", store=reopened).pending_requests() == []


def test_append_superset_preserves_all_approval_requests(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-superset")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call_a = ToolCall("superset-a", "echo", {})
    call_b = ToolCall("superset-b", "echo", {})
    first = store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call_a)]),
        [(call_a.id, call_a)],
        parent_id=root.id,
    )

    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call_a), ToolUseContent(call_b)]),
        [(call_a.id, call_a), (call_b.id, call_b)],
        parent_id=first.id,
    )

    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert set(reopened.approval_states()) == {call_a.id, call_b.id}


def test_same_parent_superset_keeps_the_prior_request(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-same-parent")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call_a = ToolCall("same-parent-a", "echo", {})
    call_b = ToolCall("same-parent-b", "echo", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call_a)]),
        [(call_a.id, call_a)],
        parent_id=root.id,
    )

    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call_a), ToolUseContent(call_b)]),
        [(call_a.id, call_a), (call_b.id, call_b)],
        parent_id=root.id,
    )

    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert set(reopened.approval_states()) == {call_a.id, call_b.id}


def test_append_uses_the_target_parent_branch(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-target-branch")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call_a = ToolCall("target-a", "echo", {})
    call_b = ToolCall("target-b", "echo", {})
    target = store.append_approval_request(call_a.id, call_a, parent_id=root.id)
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("active")]),
        parent_id=root.id,
    )

    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call_a), ToolUseContent(call_b)]),
        [(call_a.id, call_a), (call_b.id, call_b)],
        parent_id=target.id,
    )

    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert set(reopened.approval_states()) == {call_a.id, call_b.id}


def test_append_target_parent_ignores_active_child_requests(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-active-ancestor")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call = ToolCall("active-ancestor-call", "echo", {})
    target = store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("target")]),
        parent_id=root.id,
    )
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
        parent_id=target.id,
    )

    sibling = store.append_message_with_approval_requests(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("retry"), ToolUseContent(call)],
        ),
        [(call.id, call)],
        parent_id=target.id,
    )

    assert sibling.data["approval_requests"]
    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert set(reopened.approval_states()) == {call.id}


def test_abandoned_exact_sibling_is_not_deduped(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-exact-sibling")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call = ToolCall("exact-sibling-call", "echo", {})
    abandoned = store.append_approval_request(call.id, call, parent_id=root.id)
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("active")]),
        parent_id=root.id,
    )

    current = store.append_approval_request(call.id, call, parent_id=root.id)

    assert current.id != abandoned.id
    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert reopened.approval_states() == {call.id: (call, None)}


def test_off_branch_duplicate_request_is_not_reused(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-off-branch")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call = ToolCall("off-branch-call", "echo", {})
    abandoned = store.append_approval_request(call.id, call, parent_id=root.id)
    active = store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("active")]),
        parent_id=root.id,
    )

    current = store.append_approval_request(call.id, call, parent_id=active.id)

    assert current.id != abandoned.id
    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert reopened.approval_states() == {call.id: (call, None)}


def test_rewound_resolution_reopens_with_active_branch_scope(
    approval_root: Path,
) -> None:
    store = ConversationStore(approval_root, session_id="approval-rewind")
    root = store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    call = ToolCall("rewound-call", "echo", {})
    request = store.append_approval_request(call.id, call, parent_id=root.id)
    store.append_approval_resolution(call.id, "allow", parent_id=request.id)
    rewind = store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("rewound")]),
        parent_id=request.id,
    )

    store.append_approval_resolution(call.id, "deny", parent_id=rewind.id)

    reopened = ConversationStore(approval_root, session_id=store.session_id)
    assert reopened.approval_states() == {call.id: (call, "deny")}


def test_concurrent_approval_resolution_has_one_winner(approval_root: Path) -> None:
    call = ToolCall("call-1", "echo", {})
    store = ConversationStore(approval_root)
    policy = ApprovalPolicy(default="ask", store=store)
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda decision: policy.resolve(call.id, decision),
                ("allow", "deny"),
            )
        )

    assert sorted(results) == [False, True]
    resolutions = [
        entry
        for entry in store.entries
        if entry.type == "approval_resolution"
    ]
    assert len(resolutions) == 1
    assert resolutions[0].data["decision"] in {"allow", "deny"}
