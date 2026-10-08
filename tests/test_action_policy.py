from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.automations.services import validate_permissions
from zeta.config.settings import load_settings
from zeta.config.tool_policy import ToolPolicy, parse_tool_selector
from zeta.core.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRule,
    ApprovedCwdExecution,
    parse_approval_rule,
)
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolUseContent,
)
from zeta.providers.anthropic import build_request_payload
from zeta.providers.codex import build_responses_payload
from zeta.providers.ollama import _tools as ollama_tools
from zeta.runtime.loop import AgentLoop
from zeta.server.server import _approval_display_fields
from zeta.skills import SkillCatalog
from zeta.tools._action_metadata import ResolvedCapability
from zeta.tools.agent.approval import ChildApprovalPolicy
from zeta.tools.registry import ApprovalBinding, ToolAction, ToolRegistry


def _actions() -> dict[str, ToolAction]:
    return {
        "start": ToolAction(
            required_fields=frozenset({"command"}),
            allowed_fields=frozenset({"action", "command"}),
            requires_approval=True,
            capability_class="exec",
            approval_subject="command",
            binding=ApprovalBinding.CWD,
        ),
        "output": ToolAction(
            required_fields=frozenset({"task_id"}),
            allowed_fields=frozenset({"action", "task_id"}),
            requires_approval=False,
            capability_class="read",
        ),
        "kill": ToolAction(
            required_fields=frozenset({"task_id"}),
            allowed_fields=frozenset({"action", "task_id"}),
            requires_approval=False,
            capability_class="exec",
        ),
    }


def _capability(
    tool: str,
    action: str | None = None,
    subject_value: object = None,
    *,
    subject_field: str | None = None,
    binding: ApprovalBinding = ApprovalBinding.NONE,
) -> ResolvedCapability:
    return ResolvedCapability(
        tool, action, True, subject_field, subject_value, binding, None
    )


def _registry(tmp_path: Path, **kwargs: object) -> ToolRegistry:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        **kwargs,
    )
    registry.register(
        "task",
        lambda arguments, execution_context=None: arguments["action"],
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "command": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        actions=_actions(),
    )
    return registry


@pytest.mark.parametrize(
    "rule",
    ["task(start )", "task()", "task(bogus x)", "task(output x)"],
)
def test_malformed_action_rule_rejected(tmp_path: Path, rule: str) -> None:
    try:
        policy = ApprovalPolicy(always_allow=(rule,), default="deny")
    except ValueError:
        return
    _registry(tmp_path, approval_policy=policy)
    assert policy.always_allow == frozenset()
    assert any("dropped rule" in notice for notice in policy.notices)


@pytest.mark.parametrize(
    "rule",
    ["task(start )", "task()", "task(bogus x)", "task(output x)"],
)
def test_malformed_action_rule_rejected_from_settings(
    tmp_path: Path, rule: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(
        f'[approval]\nallow = ["{rule}"]\n', encoding="utf-8"
    )
    loaded = load_settings(home=home, project_dir=None)
    policy = ApprovalPolicy(always_allow=loaded.settings.approval_allow, default="deny")
    _registry(tmp_path / "registry", approval_policy=policy)

    assert policy.always_allow == frozenset()


def test_persisted_malformed_action_rule_is_rejected(tmp_path: Path) -> None:
    policy = ApprovalPolicy(default="deny")
    _registry(tmp_path, approval_policy=policy)

    with pytest.raises(ValueError, match="empty subject pattern"):
        policy.always_allow = ("task(start )",)

    assert policy.always_allow == frozenset()


@pytest.mark.asyncio
async def test_policy_authorizes_resolved_capability_only(tmp_path: Path) -> None:
    class SpyPolicy:
        restricted = True

        def allows_tool(self, name: str, actions=None) -> bool:
            return True

        def allows_call(self, capability: ResolvedCapability) -> bool:
            assert capability.tool == "task"
            assert capability.action == "output"
            assert capability.arguments == {"action": "output", "task_id": "1"}
            return True

    registry = _registry(tmp_path)
    registry.tool_policy = SpyPolicy()  # type: ignore[assignment]

    result = await registry.execute(
        ToolCall("call", "task", {"action": "output", "task_id": "1"})
    )

    assert not result["isError"]


def test_automation_and_live_matching_share_one_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = ("task(output)", "task(start pytest*)")
    registry = _registry(tmp_path)
    resolve = registry.resolve_approval_rule
    resolved: list[str] = []

    def recording_resolver(rule: str | ApprovalRule) -> ApprovalRule:
        resolved.append(str(rule))
        return resolve(rule)

    monkeypatch.setattr(registry, "resolve_approval_rule", recording_resolver)
    policy = ApprovalPolicy(always_allow=rules, default="deny")
    registry.set_approval_policy(policy)
    validate_permissions(SimpleNamespace(allow=rules), registry)  # type: ignore[arg-type]
    calls = (
        ({"action": "output", "task_id": "1"}, ApprovalDecision.ALLOW),
        ({"action": "start", "command": "pytest -q"}, ApprovalDecision.ALLOW),
        ({"action": "start", "command": "ruff"}, ApprovalDecision.DENY),
    )

    assert resolved.count("task(output)") == 2
    assert resolved.count("task(start pytest*)") == 2
    assert [
        policy.decide(registry.resolve_call("task", arguments))
        for arguments, _expected in calls
    ] == [expected for _arguments, expected in calls]


@pytest.mark.parametrize(
    "rule",
    ["task(start )", "task()", "task(bogus x)", "task(output x)"],
)
def test_automation_rejects_malformed_action_rules(
    tmp_path: Path, rule: str
) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="invalid approval rule"):
        validate_permissions(SimpleNamespace(allow=(rule,)), registry)  # type: ignore[arg-type]


def test_existing_tool_payloads_match_pre_action_policy_snapshots(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    messages = [Message(MessageRole.USER, [TextContent("go")])]
    existing_schemas = [
        schema for schema in registry.schemas if schema["name"] != "request_attention"
    ]
    snapshots = {
        "schemas": existing_schemas,
        "anthropic": build_request_payload(
            messages,
            existing_schemas,
            model="claude-test",
            max_tokens=2048,
            thinking_budget=1024,
        ),
        "codex": build_responses_payload(
            messages, existing_schemas, model="codex-test"
        ),
        "ollama": ollama_tools(existing_schemas),
    }
    expected = {
        "schemas": "f1e5f3e3c8e87a51a1b9570b89f729639c52a8219249fd9da1004f27bc002d16",
        "anthropic": "e929642ad8e79ae60f09b42f175dfe347d3495a5f816b59c2186034210bb36f3",
        "codex": "e769aa54af581e1e8c31c983aa379c0ec60d2678f8e35fb287bd4ef19a50c71d",
        "ollama": "e7bf0a9f492eae89b57ce2425ff63431bb1ac71d70e501310bcf2a62366228ff",
    }

    assert {
        name: hashlib.sha256(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        for name, value in snapshots.items()
    } == expected


def test_capability_selector_parser_normalizes_bare_action_and_mcp_names() -> None:
    assert str(parse_tool_selector("agent")) == "agent"
    assert str(parse_tool_selector("agent(status)")) == "agent(status)"
    assert str(parse_tool_selector("computer__click")) == "computer__click"
    assert str(parse_tool_selector("computer__*")) == "computer__*"


@pytest.mark.parametrize("selector", ["agent()", "agent(status|output)", "agent(status *)"])
def test_capability_selector_parser_rejects_invalid_action_forms(selector: str) -> None:
    with pytest.raises(ValueError, match="tool selector"):
        parse_tool_selector(selector)


def test_layered_allowlists_intersect_and_deny_wins_per_action() -> None:
    policy = ToolPolicy.create(
        ("task(start)", "task(output)"),
        ("task(kill)", "task(output)"),
        allow_layers=(("task",), ("task(output)", "task(kill)")),
    )

    assert not policy.allows_call(_capability("task", "start"))
    assert not policy.allows_call(_capability("task", "output"))
    assert not policy.allows_call(_capability("task", "kill"))

    allowed = ToolPolicy.create(
        allow_layers=(("task",), ("task(output)", "task(kill)")),
        deny=("task(kill)",),
    )
    assert allowed.allows_call(_capability("task", "output"))
    assert not allowed.allows_call(_capability("task", "start"))
    assert not allowed.allows_call(_capability("task", "kill"))


def test_schema_action_enum_is_filtered_and_tool_hidden_when_none_remain(tmp_path: Path) -> None:
    registry = _registry(tmp_path, tool_allow=("task(output)",))

    assert registry.schemas[0]["parameters"]["properties"]["action"]["enum"] == [
        "output"
    ]
    unrestricted_schema = _registry(tmp_path / "source").schemas[0]
    assert registry.allowed_schemas([unrestricted_schema])[0]["parameters"][
        "properties"
    ]["action"]["enum"] == ["output"]

    hidden = _registry(tmp_path / "hidden", tool_deny=("task",))
    assert hidden.schemas == []


@pytest.mark.asyncio
async def test_disallowed_and_hallucinated_actions_are_rejected_before_execution(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path, tool_allow=("task(output)",))

    disallowed = await registry.execute(
        ToolCall("call-1", "task", {"action": "start", "command": "pytest"})
    )
    hallucinated = await registry.execute(
        ToolCall("call-2", "task", {"action": "dance", "task_id": "1"})
    )

    assert disallowed["structuredContent"]["error"]["kind"] == "tool_action_not_allowed"
    assert "task(start)" in disallowed["content"][0]["text"]
    assert hallucinated["structuredContent"]["error"]["kind"] == "invalid_tool_action"
    assert "unknown action 'dance'" in hallucinated["content"][0]["text"]


@pytest.mark.asyncio
async def test_per_action_fields_are_validated_before_handler(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    missing = await registry.execute(ToolCall("call-1", "task", {"action": "start"}))
    foreign = await registry.execute(
        ToolCall("call-2", "task", {"action": "output", "task_id": "1", "command": "x"})
    )

    assert "action=start requires command" in missing["content"][0]["text"]
    assert "action=output does not allow command" in foreign["content"][0]["text"]


def test_required_exact_capabilities_are_checked_by_action(tmp_path: Path) -> None:
    registry = _registry(
        tmp_path,
        tool_allow=("task(output)", "task(missing)"),
        required_tool_names=("task(output)", "task(missing)"),
    )

    assert registry.missing_required_tools == ("task(missing)",)


def test_approval_rule_is_resolved_against_action_metadata(tmp_path: Path) -> None:
    policy = ApprovalPolicy(
        always_allow={"task(start pytest*)", "task(output)"},
        always_ask={"task(start pytest -k slow*)"},
        always_deny={"task(start pytest --pdb*)"},
    )
    _registry(tmp_path, approval_policy=policy)

    assert ApprovalRule("task", subject_pattern="pytest*", action="start") in policy.always_allow
    assert ApprovalRule("task", action="output") in policy.always_allow
    assert policy.decide(_capability("task", "start", "pytest -q", subject_field="command", binding=ApprovalBinding.CWD)) is ApprovalDecision.ALLOW
    assert policy.decide(_capability("task", "start", "pytest -k slow_case", subject_field="command", binding=ApprovalBinding.CWD)) is ApprovalDecision.ASK
    assert policy.decide(_capability("task", "start", "pytest --pdb", subject_field="command", binding=ApprovalBinding.CWD)) is ApprovalDecision.DENY
    assert policy.decide(_capability("task", "output")) is ApprovalDecision.ALLOW


def test_old_no_action_approval_syntax_keeps_its_meaning() -> None:
    assert parse_approval_rule("bash(git status*)") == ApprovalRule(
        "bash", subject_pattern="git status*"
    )
    policy = ApprovalPolicy(always_allow={"bash(git status*)"})
    assert policy.decide(_capability("bash", subject_value="git status --short", subject_field="command")) is ApprovalDecision.ALLOW


def test_action_scoped_unreadable_subject_fails_closed(tmp_path: Path) -> None:
    deny = ApprovalPolicy(always_deny={"task(start rm *)"}, default="allow")
    ask = ApprovalPolicy(always_ask={"task(start git push*)"}, always_allow={"task(start)"})
    _registry(tmp_path / "deny", approval_policy=deny)
    _registry(tmp_path / "ask", approval_policy=ask)
    unreadable = _capability(
        "task", "start", subject_field="command", binding=ApprovalBinding.CWD
    )

    assert deny.decide(unreadable) is ApprovalDecision.DENY
    assert ask.decide(unreadable) is ApprovalDecision.ASK


def test_approval_request_carries_resolved_action(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "session", cwd=tmp_path)
    policy = ApprovalPolicy(default="ask", store=store)
    registry = _registry(tmp_path, approval_policy=policy)

    request = registry.prepare_approval(
        ToolCall("call-1", "task", {"action": "start", "command": "pytest"})
    )

    assert request is not None
    assert registry.prepare_approval(
        ToolCall("call-2", "task", {"action": "output", "task_id": "1"})
    ) is None
    assert request.action == "start"
    assert request.audit_facts() == {"action": "start"}
    assert request.audit_display() == {}
    assert request.always_allow_rule() == ApprovalRule("task", action="start")
    assert _approval_display_fields(request) == {"approval_action": "start"}


@pytest.mark.asyncio
async def test_provider_turn_persists_approves_and_executes_action_call(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "task", {"action": "start", "command": "pytest"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "session", cwd=tmp_path)
    policy = ApprovalPolicy(default="ask")
    registry = _registry(tmp_path / "registry")
    executed: list[str] = []
    registry.register(
        "task",
        lambda arguments, execution_context=None: executed.append(
            str(arguments["command"])
        )
        or "started",
        parameters=registry.definitions_by_name["task"].parameters,
        actions=_actions(),
    )
    loop = AgentLoop(
        backend,
        store,
        registry=registry,
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
        skip_mcp_mount=True,
    )
    events_task = asyncio.create_task(
        _collect_events(loop.run_turn("start", origin=MessageOrigin.USER))
    )
    pending: list[ApprovalRequest] = []
    for _ in range(100):
        pending = policy.pending_requests()
        if pending:
            break
        await asyncio.sleep(0.01)
    assert pending == [ApprovalRequest(call.id, call, action="start")]
    approval_record = next(
        request
        for entry in store.entries
        for request in entry.data.get("approval_requests", [])
    )
    assert approval_record["approval_facts"] == {"action": "start"}
    assert policy.approve(call.id)
    await events_task
    assert executed == ["pytest"]


async def _collect_events(events):
    return [event async for event in events]


def test_resumed_subjectless_action_argument_uses_registry_capability(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "inbox", {"action": "send"})
    store = ConversationStore(tmp_path / "session", cwd=tmp_path)
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy = ApprovalPolicy(default="ask", store=store)
    registry_cwd = tmp_path / "registry"
    registry_cwd.mkdir()
    registry = ToolRegistry(
        registry_cwd,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        approval_policy=policy,
    )
    registry.register(
        "inbox",
        lambda arguments: "sent",
        parameters={
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        },
    )

    pending = policy.pending_requests()
    assert pending == [ApprovalRequest(call.id, call, action=None)]
    assert policy.remember_allow(pending[0]) == ApprovalRule("inbox")
    assert policy.always_allow == {ApprovalRule("inbox")}
    assert policy.notices == ()


def test_child_action_approval_uses_resolved_capability(tmp_path: Path) -> None:
    parent = ApprovalPolicy(default="ask")
    _registry(tmp_path / "registry", approval_policy=parent)
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    child = ChildApprovalPolicy(
        parent,
        child_store,
        "worker",
        "child-1",
        parent_cwd=tmp_path,
        child_cwd=tmp_path,
    )
    call = ToolCall("call-1", "task", {"command": "pytest"})
    capability = _capability(
        "task",
        "start",
        "pytest",
        subject_field="command",
        binding=ApprovalBinding.CWD,
    )

    request = child.prepare(call, capability=capability)

    assert request is not None
    assert request.action == capability.action


def test_action_subject_child_cwd_binding_is_preserved(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    policy = ApprovalPolicy(always_allow={"task(start pytest*)"})
    _registry(tmp_path / "registry", approval_policy=policy)
    capability = _capability(
        "task", "start", "pytest -q", subject_field="command", binding=ApprovalBinding.CWD
    )

    same_decision, same_binding = policy.decide_for_child_with_binding(
        capability, parent_cwd=parent, child_cwd=parent
    )
    other_decision, other_binding = policy.decide_for_child_with_binding(
        capability, parent_cwd=parent, child_cwd=child
    )

    assert same_decision is ApprovalDecision.ALLOW
    assert isinstance(same_binding, ApprovedCwdExecution)
    assert other_decision is ApprovalDecision.ASK
    assert other_binding is None


def test_action_path_subject_binds_child_target(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    allowed = child / "allowed"
    parent.mkdir()
    allowed.mkdir(parents=True)
    policy = ApprovalPolicy(
        always_allow={ApprovalRule("artifact", f"{allowed}/*", "update")}
    )
    capability = _capability(
        "artifact",
        "update",
        "allowed/result.txt",
        subject_field="path",
        binding=ApprovalBinding.PATH,
    )

    decision, binding = policy.decide_for_child_with_binding(
        capability, parent_cwd=parent, child_cwd=child
    )

    assert decision is ApprovalDecision.ALLOW
    assert binding is not None
    assert binding.target == str(allowed / "result.txt")


def test_always_allow_scope_round_trips_to_current_action() -> None:
    request = ApprovalRequest(
        "call-1",
        ToolCall("call-1", "task", {"action": "start", "command": "pytest"}),
        action="start",
    )
    policy = ApprovalPolicy()
    policy.always_allow = policy.always_allow | {request.always_allow_rule()}

    assert policy.always_allow == {ApprovalRule("task", action="start")}


def test_registry_resolves_action_and_no_action_capabilities(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    action = registry.resolve_call(
        "task", {"action": "start", "command": "pytest -q"}
    )
    registry.register(
        "plain",
        lambda arguments: "ok",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        requires_approval=False,
        approval_subject="value",
    )
    plain = registry.resolve_call("plain", {"value": "item"})

    assert (
        action.tool,
        action.action,
        action.requires_approval,
        action.subject_field,
        action.subject_value,
        action.binding,
        action.capability_class,
    ) == (
        "task",
        "start",
        True,
        "command",
        "pytest -q",
        ApprovalBinding.CWD,
        "exec",
    )
    assert (
        plain.tool,
        plain.action,
        plain.requires_approval,
        plain.subject_field,
        plain.subject_value,
        plain.binding,
        plain.capability_class,
    ) == ("plain", None, False, "value", "item", ApprovalBinding.NONE, None)


def test_action_path_binding_requires_execution_context_handler(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    actions = {
        "update": ToolAction(
            required_fields=frozenset({"path"}),
            allowed_fields=frozenset({"action", "path"}),
            requires_approval=True,
            capability_class="write",
            approval_subject="path",
            binding=ApprovalBinding.PATH,
        )
    }
    with pytest.raises(ValueError, match="must accept execution_context"):
        registry.register(
            "artifact",
            lambda arguments: "side effect",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            actions=actions,
        )
