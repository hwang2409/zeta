import asyncio
from pathlib import Path

import pytest

from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent.approval import ChildApprovalPolicy


def _persist_prepared_request(
    registry: ToolRegistry,
    child_store: ConversationStore,
    call: ToolCall,
) -> None:
    request = registry.prepare_approval(call)
    assert request is not None
    child_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(request.request_id, request.tool_call)],
    )


async def _approve_when_registered(
    parent_policy: ApprovalPolicy,
    child_instance_id: str,
    request_id: str,
) -> None:
    key = (child_instance_id, request_id)
    while key not in {request.key for request in parent_policy.pending_requests()}:
        await asyncio.sleep(0)
    assert parent_policy.approve(key)


@pytest.mark.asyncio
async def test_write_uses_binding_captured_before_persisted_request(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe"
    evil = tmp_path / "evil"
    safe.mkdir()
    evil.mkdir()
    (safe / "victim.txt").write_text("SAFE", encoding="utf-8")
    (evil / "victim.txt").write_text("EVIL", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(safe, target_is_directory=True)

    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "write-swap",
        child_cwd=tmp_path,
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)
    call = ToolCall(
        "write-swap",
        "write",
        {"path": str(alias / "victim.txt"), "content": "PWN"},
    )

    _persist_prepared_request(registry, child_store, call)
    alias.unlink()
    alias.symlink_to(evil, target_is_directory=True)
    execution = asyncio.create_task(registry.execute(call))
    try:
        await _approve_when_registered(parent_policy, "write-swap", call.id)
        result = await execution
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await registry.close()

    assert result["isError"] is False
    assert (safe / "victim.txt").read_text(encoding="utf-8") == "PWN"
    assert (evil / "victim.txt").read_text(encoding="utf-8") == "EVIL"


async def _assert_shell_uses_displayed_cwd(
    tmp_path: Path,
    tool_name: str,
) -> None:
    safe = tmp_path / "safe"
    evil = tmp_path / "evil"
    safe.mkdir()
    evil.mkdir()
    alias = tmp_path / "cwd-alias"
    alias.symlink_to(safe, target_is_directory=True)

    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        f"{tool_name}-swap",
        child_cwd=tmp_path,
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)
    arguments = {"command": "touch marker.txt", "cwd": str(alias)}
    call = ToolCall(f"{tool_name}-swap", tool_name, arguments)

    _persist_prepared_request(registry, child_store, call)
    alias.unlink()
    alias.symlink_to(evil, target_is_directory=True)
    execution = asyncio.create_task(registry.execute(call))
    try:
        await _approve_when_registered(parent_policy, f"{tool_name}-swap", call.id)
        result = await execution
        if tool_name == "run_background" and result["isError"] is False:
            task_id = result["structuredContent"]["task_id"]
            await registry.background_tasks.wait(task_id)
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await registry.close()

    assert result["isError"] is False
    assert (safe / "marker.txt").exists()
    assert not (evil / "marker.txt").exists()


@pytest.mark.asyncio
async def test_bash_uses_binding_captured_before_persisted_request(
    tmp_path: Path,
) -> None:
    await _assert_shell_uses_displayed_cwd(tmp_path, "bash")


@pytest.mark.asyncio
async def test_run_background_uses_binding_captured_before_persisted_request(
    tmp_path: Path,
) -> None:
    await _assert_shell_uses_displayed_cwd(tmp_path, "run_background")


@pytest.mark.asyncio
async def test_restart_with_pending_request_fails_closed_after_allow(
    tmp_path: Path,
) -> None:
    approved_cwd = tmp_path / "approved"
    approved_cwd.mkdir()
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    first_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "restart",
        child_cwd=tmp_path,
    )
    first_registry = ToolRegistry(
        tmp_path,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    first_registry.set_approval_policy(first_policy)
    call = ToolCall(
        "restart-pending",
        "bash",
        {"command": "touch marker.txt", "cwd": str(approved_cwd)},
    )
    _persist_prepared_request(first_registry, child_store, call)
    await first_registry.close()

    restarted_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "restart",
        child_cwd=tmp_path,
    )
    restarted_registry = ToolRegistry(
        tmp_path,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    restarted_registry.set_approval_policy(restarted_policy)
    execution = asyncio.create_task(restarted_registry.execute(call))
    try:
        await _approve_when_registered(parent_policy, "restart", call.id)
        result = await execution
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await restarted_registry.close()

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "tool execution denied: approval binding unavailable; "
        "submit a new tool request"
    )
    assert not (approved_cwd / "marker.txt").exists()


@pytest.mark.asyncio
async def test_canceled_execution_discards_its_approval_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(
        store=parent_store,
        always_allow={"bash(echo *)"},
    )
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "canceled-binding",
        parent_cwd=tmp_path,
        child_cwd=tmp_path,
    )
    registry = ToolRegistry(
        tmp_path,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)

    binding_created = asyncio.Event()
    never_release = asyncio.Event()
    original_authorize = child_policy.authorize

    async def pause_after_binding(
        tool_call: ToolCall,
        abort_signal: AbortSignal,
        *,
        execution_token: str | None = None,
    ) -> ApprovalDecision | None:
        decision = await original_authorize(
            tool_call,
            abort_signal,
            execution_token=execution_token,
        )
        binding_created.set()
        await never_release.wait()
        return decision

    monkeypatch.setattr(child_policy, "authorize", pause_after_binding)
    execution = asyncio.create_task(
        registry.execute(ToolCall("canceled", "bash", {"command": "echo safe"}))
    )
    try:
        await asyncio.wait_for(binding_created.wait(), timeout=1)
        assert len(child_policy._execution_bindings) == 1
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert child_policy._execution_bindings == {}

        monkeypatch.setattr(child_policy, "authorize", original_authorize)
        retry = await registry.execute(
            ToolCall("canceled", "bash", {"command": "echo safe"})
        )
        assert retry["isError"] is False
        assert child_policy._execution_bindings == {}
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_allow_wins_abort_transfers_pending_binding_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    parent_policy.declare_subjects({"bash": "command"})
    child_policy = ChildApprovalPolicy(
        parent_policy, child_store, "child", "race", child_cwd=tmp_path
    )
    call = ToolCall("race", "bash", {"command": "echo safe"})
    request = child_policy.prepare(call)
    assert request is not None
    child_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(request.request_id, request.tool_call)],
    )
    original_resolve = child_store.resolve_approval

    def allow_before_abort(request_id: str, decision: str) -> bool:
        if decision == "abort":
            original_resolve(request_id, "allow")
        return original_resolve(request_id, decision)

    monkeypatch.setattr(child_store, "resolve_approval", allow_before_abort)
    signal = AbortSignal()
    signal.abort()
    decision = await child_policy.authorize(call, signal, execution_token="exec")

    assert decision is ApprovalDecision.ALLOW
    assert child_policy.consume_execution_binding("exec") is not None


@pytest.mark.asyncio
async def test_replayed_allow_without_binding_fails_closed_for_cwd(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    parent_policy.declare_subjects({"bash": "command"})
    child_policy = ChildApprovalPolicy(
        parent_policy, child_store, "child", "restart", child_cwd=tmp_path
    )
    call = ToolCall("replay", "bash", {"command": "pwd"})
    child_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    assert child_store.resolve_approval(call.id, "allow")

    decision = await child_policy.authorize(
        call, AbortSignal(), execution_token="replayed"
    )

    assert decision is ApprovalDecision.DENY
    assert child_policy.consume_execution_binding("replayed") is None
