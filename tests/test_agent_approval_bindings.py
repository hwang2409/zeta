import asyncio
import inspect
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools._action_metadata import ResolvedCapability
from zeta.tools._shared.sandbox import open_target
from zeta.tools.agent.approval import ChildApprovalPolicy
from zeta.tools.registry import ApprovalBinding, ToolAction
from zeta.tui.cards.approval_card import render_approval_card


def _capability(call: ToolCall) -> ResolvedCapability:
    subject = "path" if call.name in {"read", "write", "edit"} else "command"
    binding = ApprovalBinding.PATH if subject == "path" else ApprovalBinding.CWD
    return ResolvedCapability(
        call.name,
        None,
        True,
        subject,
        call.arguments.get(subject),
        binding,
        None,
        call.arguments,
    )


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


def _child_policy_for_display(
    tmp_path: Path,
    subjects: dict[str, str],
) -> tuple[ApprovalPolicy, ChildApprovalPolicy]:
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    del subjects
    return parent_policy, ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "display-binding",
        child_cwd=tmp_path,
    )


@pytest.mark.parametrize("tool_name", ["read", "write", "edit"])
@pytest.mark.parametrize("alias_kind", ["parent", "target"])
def test_path_approval_display_is_canonical_binding_target(
    tmp_path: Path,
    tool_name: str,
    alias_kind: str,
) -> None:
    sensitive = tmp_path / "sensitive"
    sensitive.mkdir()
    canonical = sensitive / "credentials"
    canonical.write_text("secret", encoding="utf-8")
    if alias_kind == "parent":
        alias_parent = tmp_path / "innocent"
        alias_parent.symlink_to(sensitive, target_is_directory=True)
        requested = alias_parent / canonical.name
    else:
        requested = tmp_path / "innocent-credentials"
        requested.symlink_to(canonical)

    _parent, child = _child_policy_for_display(tmp_path, {tool_name: "path"})
    call = ToolCall(
        f"{tool_name}-{alias_kind}",
        tool_name,
        {"path": str(requested)},
    )
    request = child.prepare(call, capability=_capability(call))

    assert request is not None
    binding = child._pending_bindings[call.id]
    assert binding is not None
    assert request.resolved_path == binding.target == str(canonical)
    output = StringIO()
    Console(file=output, force_terminal=False, width=300).print(
        render_approval_card(
            tool_name,
            call.arguments,
            execution_display=(request.effective_cwd, request.resolved_path),
        )
    )
    assert f"resolved_path={canonical}" in output.getvalue()


@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
def test_shell_approval_display_is_canonical_binding_cwd(
    tmp_path: Path,
    tool_name: str,
) -> None:
    sensitive = tmp_path / "sensitive"
    sensitive.mkdir()
    alias = tmp_path / "innocent"
    alias.symlink_to(sensitive, target_is_directory=True)
    _parent, child = _child_policy_for_display(tmp_path, {tool_name: "command"})
    call = ToolCall(
        f"{tool_name}-cwd",
        tool_name,
        {"command": "pwd", "cwd": str(alias)},
    )

    request = child.prepare(call, capability=_capability(call))

    assert request is not None
    binding = child._pending_bindings[call.id]
    assert binding is not None
    assert request.effective_cwd == binding.cwd == str(sensitive)


@pytest.mark.asyncio
async def test_failed_binding_capture_is_not_approvable_and_denies_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = _child_policy_for_display(tmp_path, {"write": "path"})
    monkeypatch.setattr(
        parent,
        "capture_child_binding",
        lambda *args, **kwargs: (None, True),
    )
    call = ToolCall("capture-failed", "write", {"path": "credentials"})

    assert child.prepare(call, capability=_capability(call)) is None
    decision = await child.authorize(call, AbortSignal(), capability=_capability(call))

    assert decision is ApprovalDecision.DENY
    assert parent.pending_requests() == []
    assert child.denial_reason(call.id) == (
        "approval binding unavailable; submit a new tool request"
    )


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
    try:
        result = await restarted_registry.execute(call)
    finally:
        await restarted_registry.close()

    assert parent_policy.pending_requests() == []
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
        capability: ResolvedCapability,
    ) -> ApprovalDecision | None:
        decision = await original_authorize(
            tool_call,
            abort_signal,
            execution_token=execution_token,
            capability=capability,
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
    child_policy = ChildApprovalPolicy(
        parent_policy, child_store, "child", "race", child_cwd=tmp_path
    )
    call = ToolCall("race", "bash", {"command": "echo safe"})
    request = child_policy.prepare(call, capability=_capability(call))
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
    decision = await child_policy.authorize(
        call, signal, execution_token="exec", capability=_capability(call)
    )

    assert decision is ApprovalDecision.ALLOW
    assert child_policy.consume_execution_binding("exec") is not None


@pytest.mark.asyncio
async def test_replayed_allow_without_binding_fails_closed_for_cwd(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=tmp_path)
    parent_policy = ApprovalPolicy(store=parent_store)
    call = ToolCall("replay", "bash", {"command": "pwd"})
    child_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [
            (
                call.id,
                call,
                {
                    "effective_cwd": str(tmp_path),
                    "resolved_path": None,
                },
            )
        ],
    )
    persisted = child_store.entries[-1].data["approval_requests"][0]
    assert persisted["approval_display"] == {
        "effective_cwd": str(tmp_path),
        "resolved_path": None,
    }
    assert child_store.resolve_approval(call.id, "allow")

    # The durable display is audit-only: a restarted policy has no in-memory
    # binding and must not turn the persisted cwd into execution authority.
    restarted_store = ConversationStore(
        tmp_path / "child-sessions", session_id=child_store.session_id, cwd=tmp_path
    )
    restarted_policy = ChildApprovalPolicy(
        parent_policy, restarted_store, "child", "restart", child_cwd=tmp_path
    )
    decision = await restarted_policy.authorize(
        call,
        AbortSignal(),
        execution_token="replayed",
        capability=_capability(call),
    )

    assert decision is ApprovalDecision.DENY
    assert restarted_policy.consume_execution_binding("replayed") is None


async def _execute_after_manual_approval(
    tmp_path: Path,
    call: ToolCall,
    mutate,
    *,
    child_cwd: Path | None = None,
):
    child_cwd = child_cwd or tmp_path
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=child_cwd)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=child_cwd)
    parent_policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        parent_policy, child_store, "child", call.id, child_cwd=child_cwd
    )
    registry = ToolRegistry(
        child_cwd, session_store=child_store, skill_catalog=SkillCatalog.empty()
    )
    registry.set_approval_policy(child_policy)
    _persist_prepared_request(registry, child_store, call)
    mutate()
    execution = asyncio.create_task(registry.execute(call))
    try:
        await _approve_when_registered(parent_policy, call.id, call.id)
        return await execution
    finally:
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_read_manual_approval_rejects_symlink_parent_swap(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    safe = tmp_path / "safe"
    evil = tmp_path / "evil"
    safe.mkdir()
    evil.mkdir()
    (safe / "value").write_text("SAFE", encoding="utf-8")
    (evil / "value").write_text("SECRET", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(safe, target_is_directory=True)
    call = ToolCall("read-link-swap", "read", {"path": str(alias / "value")})

    def mutate() -> None:
        alias.unlink()
        alias.symlink_to(evil, target_is_directory=True)

    result = await _execute_after_manual_approval(
        tmp_path, call, mutate, child_cwd=work
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "SAFE"


@pytest.mark.asyncio
async def test_read_scoped_auto_allow_rejects_symlink_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    safe = tmp_path / "safe"
    evil = tmp_path / "evil"
    safe.mkdir()
    evil.mkdir()
    (safe / "value").write_text("SAFE", encoding="utf-8")
    (evil / "value").write_text("SECRET", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(safe, target_is_directory=True)
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=work)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=work)
    parent_policy = ApprovalPolicy(
        store=parent_store, always_allow={f"read({tmp_path}/**)"}
    )
    child_policy = ChildApprovalPolicy(
        parent_policy, child_store, "child", "auto-read", child_cwd=work
    )
    registry = ToolRegistry(
        work, session_store=child_store, skill_catalog=SkillCatalog.empty()
    )
    registry.set_approval_policy(child_policy)
    authorized = asyncio.Event()
    release = asyncio.Event()
    original_authorize = child_policy.authorize

    async def pause_after_binding(
        tool_call: ToolCall,
        abort_signal: AbortSignal,
        *,
        execution_token: str | None = None,
        capability: ResolvedCapability,
    ):
        decision = await original_authorize(
            tool_call,
            abort_signal,
            execution_token=execution_token,
            capability=capability,
        )
        assert execution_token in child_policy._execution_bindings
        authorized.set()
        await release.wait()
        return decision

    monkeypatch.setattr(child_policy, "authorize", pause_after_binding)
    execution = asyncio.create_task(
        registry.execute(ToolCall("auto-read", "read", {"path": str(alias / "value")}))
    )
    try:
        await asyncio.wait_for(authorized.wait(), timeout=1)
        alias.unlink()
        alias.symlink_to(evil, target_is_directory=True)
        release.set()
        result = await execution
    finally:
        release.set()
        if not execution.done():
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        await registry.close()

    assert result["isError"] is False
    assert result["content"][0]["text"] == "SAFE"


@pytest.mark.asyncio
async def test_read_manual_approval_rejects_target_inode_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "value"
    target.write_text("SAFE", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.write_text("SECRET", encoding="utf-8")
    call = ToolCall("read-inode-swap", "read", {"path": str(target)})

    def mutate() -> None:
        replacement.replace(target)

    result = await _execute_after_manual_approval(tmp_path, call, mutate)

    assert result["isError"] is True
    assert "approved target was replaced" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["write", "edit"])
async def test_absent_approved_target_that_appears_is_denied_untouched(
    tmp_path: Path, tool_name: str
) -> None:
    approved = tmp_path / "approved"
    elsewhere = tmp_path / "elsewhere"
    approved.mkdir()
    elsewhere.mkdir()
    sensitive = elsewhere / "sensitive"
    sensitive.write_text("SECRET", encoding="utf-8")
    target = approved / "newfile"
    arguments = (
        {"path": str(target), "content": "PWN"}
        if tool_name == "write"
        else {"path": str(target), "old_string": "SECRET", "new_string": "PWN"}
    )
    call = ToolCall(f"absent-appears-{tool_name}", tool_name, arguments)

    result = await _execute_after_manual_approval(
        tmp_path, call, lambda: sensitive.rename(target)
    )

    assert result["isError"] is True
    assert target.read_text(encoding="utf-8") == "SECRET"


@pytest.mark.asyncio
async def test_existing_approved_write_target_disappears_without_stray_creation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_text("ORIGINAL", encoding="utf-8")
    call = ToolCall(
        "existing-disappears", "write", {"path": str(target), "content": "PWN"}
    )

    result = await _execute_after_manual_approval(tmp_path, call, target.unlink)

    assert result["isError"] is True
    assert not target.exists()


@pytest.mark.asyncio
async def test_existing_approved_write_target_replacement_is_denied(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_text("ORIGINAL", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.write_text("SECRET", encoding="utf-8")
    call = ToolCall(
        "existing-replaced", "write", {"path": str(target), "content": "PWN"}
    )

    result = await _execute_after_manual_approval(
        tmp_path, call, lambda: replacement.replace(target)
    )

    assert result["isError"] is True
    assert target.read_text(encoding="utf-8") == "SECRET"


@pytest.mark.asyncio
async def test_all_path_binding_tools_consume_execution_binding(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    path_tools = {
        name
        for name, definition in registry._tools.items()
        if definition.approval_subject == "path"
    }
    assert path_tools == {"read", "write", "edit"}
    for name in path_tools:
        handler = registry._tools[name].handler
        assert "execution_context" in inspect.signature(handler).parameters
        assert handler.func.__globals__.get("open_target") is open_target
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "action", "subject", "binding", "arguments"),
    [
        ("task", "start", "command", "cwd", {"action": "start", "command": "pwd"}),
        ("artifact", "update", "path", "path", {"action": "update", "path": "result.txt"}),
    ],
)
async def test_replayed_action_scoped_allow_requires_original_binding(
    tmp_path: Path,
    tool_name: str,
    action: str,
    subject: str,
    binding: str,
    arguments: dict[str, str],
) -> None:
    parent_store = ConversationStore(tmp_path / "parent", cwd=tmp_path)
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    parent = ApprovalPolicy(store=parent_store)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.register(
        tool_name,
        lambda arguments, execution_context=None: "unused",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                subject: {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        actions={
            action: ToolAction(
                required_fields=frozenset({subject}),
                allowed_fields=frozenset({"action", subject}),
                requires_approval=True,
                capability_class="exec",
                approval_subject=subject,
                binding=ApprovalBinding(binding),
            )
        },
    )
    call = ToolCall("replayed-action", tool_name, arguments)
    capability = registry.resolve_call(call.name, call.arguments)
    child_store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    assert child_store.resolve_approval(call.id, "allow")
    restarted = ChildApprovalPolicy(parent, child_store, "child", "replayed", child_cwd=tmp_path)

    decision = await restarted.authorize(
        call,
        AbortSignal(),
        execution_token="replayed",
        capability=capability,
    )

    assert decision is ApprovalDecision.DENY
    assert restarted.consume_execution_binding("replayed") is None
    await registry.close()
