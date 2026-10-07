from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zeta.config.tool_policy import ToolPolicy, parse_tool_selector
from zeta.core.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRule,
    ApprovedCwdExecution,
    parse_approval_rule,
)
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
from zeta.providers.anthropic import build_request_payload
from zeta.providers.codex import build_responses_payload
from zeta.providers.ollama import _tools as ollama_tools
from zeta.server.server import _approval_display_fields
from zeta.skills import SkillCatalog
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
        lambda arguments: arguments["action"],
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


def test_existing_tool_payloads_match_pre_action_policy_snapshots(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    messages = [Message(MessageRole.USER, [TextContent("go")])]
    snapshots = {
        "schemas": registry.schemas,
        "anthropic": build_request_payload(
            messages,
            registry.schemas,
            model="claude-test",
            max_tokens=2048,
            thinking_budget=1024,
        ),
        "codex": build_responses_payload(
            messages, registry.schemas, model="codex-test"
        ),
        "ollama": ollama_tools(registry.schemas),
    }
    expected = {
        "schemas": "424464b3f6cb3c604c18f629a52c5679a541d171868dd9d66f3f9dd443ba7885",
        "anthropic": "8e295ad1975e0e2b6da3a2f5d0c53c61f2d586f461a8971494688a2054e59c19",
        "codex": "9bcbe9462dabc868b3b039b1f7bc1c917c396dc004790b69b511c113d74dab02",
        "ollama": "3e41107071848090fdcebd6e4aeaa85e5d6686d107840e7a32731f35afd10c37",
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

    assert not policy.allows_call("task", {"action": "start"})
    assert not policy.allows_call("task", {"action": "output"})
    assert not policy.allows_call("task", {"action": "kill"})

    allowed = ToolPolicy.create(
        allow_layers=(("task",), ("task(output)", "task(kill)")),
        deny=("task(kill)",),
    )
    assert allowed.allows_call("task", {"action": "output"})
    assert not allowed.allows_call("task", {"action": "start"})
    assert not allowed.allows_call("task", {"action": "kill"})


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


def test_approval_rule_is_resolved_against_action_metadata() -> None:
    policy = ApprovalPolicy(
        always_allow={"task(start pytest*)", "task(output)"},
        always_ask={"task(start pytest -k slow*)"},
        always_deny={"task(start pytest --pdb*)"},
    )
    policy.declare_actions(
        "task",
        {
            "start": ("command", ApprovalBinding.CWD),
            "output": (None, ApprovalBinding.NONE),
        },
    )

    assert ApprovalRule("task", subject_pattern="pytest*", action="start") in policy.always_allow
    assert ApprovalRule("task", action="output") in policy.always_allow
    assert policy.decide("task", {"action": "start", "command": "pytest -q"}, action="start") is ApprovalDecision.ALLOW
    assert policy.decide("task", {"action": "start", "command": "pytest -k slow_case"}, action="start") is ApprovalDecision.ASK
    assert policy.decide("task", {"action": "start", "command": "pytest --pdb"}, action="start") is ApprovalDecision.DENY
    assert policy.decide("task", {"action": "output", "task_id": "1"}, action="output") is ApprovalDecision.ALLOW


def test_old_no_action_approval_syntax_keeps_its_meaning() -> None:
    assert parse_approval_rule("bash(git status*)") == ApprovalRule(
        "bash", subject_pattern="git status*"
    )
    policy = ApprovalPolicy(always_allow={"bash(git status*)"})
    policy.declare_subjects({"bash": "command"})
    assert policy.decide("bash", {"command": "git status --short"}) is ApprovalDecision.ALLOW


def test_action_scoped_unreadable_subject_fails_closed() -> None:
    deny = ApprovalPolicy(always_deny={"task(start rm *)"}, default="allow")
    ask = ApprovalPolicy(always_ask={"task(start git push*)"}, always_allow={"task(start)"})
    for policy in (deny, ask):
        policy.declare_actions("task", {"start": ("command", ApprovalBinding.CWD)})

    assert deny.decide("task", {"action": "start"}, action="start") is ApprovalDecision.DENY
    assert ask.decide("task", {"action": "start"}, action="start") is ApprovalDecision.ASK


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
    assert request.audit_display()["action"] == "start"
    assert request.always_allow_rule() == ApprovalRule("task", action="start")
    assert _approval_display_fields(request) == {
        "approval_display": {"action": "start"}
    }


def test_action_subject_child_cwd_binding_is_preserved(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    policy = ApprovalPolicy(always_allow={"task(start pytest*)"})
    policy.declare_actions(
        "task", {"start": ("command", ApprovalBinding.CWD)}
    )
    arguments = {"action": "start", "command": "pytest -q"}

    same_decision, same_binding = policy.decide_for_child_with_binding(
        "task", arguments, parent_cwd=parent, child_cwd=parent
    )
    other_decision, other_binding = policy.decide_for_child_with_binding(
        "task", arguments, parent_cwd=parent, child_cwd=child
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
    policy = ApprovalPolicy(always_allow={f"artifact(update {allowed}/*)"})
    policy.declare_actions(
        "artifact", {"update": ("path", ApprovalBinding.PATH)}
    )

    decision, binding = policy.decide_for_child_with_binding(
        "artifact",
        {"action": "update", "path": "allowed/result.txt"},
        parent_cwd=parent,
        child_cwd=child,
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
