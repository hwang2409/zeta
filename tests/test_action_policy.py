from __future__ import annotations

from pathlib import Path

import pytest

from zeta.config.tool_policy import ToolPolicy, parse_tool_selector
from zeta.core.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRule,
    parse_approval_rule,
)
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
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
                "action": {"type": "string", "enum": ["start", "output", "kill"]},
                "command": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        actions=_actions(),
    )
    return registry


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
    assert request.action == "start"
    assert request.audit_display()["action"] == "start"
    assert request.always_allow_rule() == ApprovalRule("task", action="start")
