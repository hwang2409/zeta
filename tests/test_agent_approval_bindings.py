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
    child_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)])
    )
    child_policy._request(call)
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
