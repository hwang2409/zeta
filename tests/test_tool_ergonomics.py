"""ZETA-70 regression tests: bash/exec arg unification, timeout text,
structured error payload governance.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zeta.agent_receipt import build_agent_receipt, encode_json
from zeta.core.store import ConversationStore
from zeta.tools import ToolRegistry
from zeta.tools.registry import _apply_error_governance
from zeta.types import ToolCall


def _registry(tmp_path: Path) -> ToolRegistry:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    return ToolRegistry(tmp_path, session_store=store)


@pytest.mark.asyncio
async def test_bash_accepts_command_and_cmd_alias(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    new_result = await registry.execute(
        ToolCall("b-new", "bash", {"command": "printf hello"})
    )
    legacy_result = await registry.execute(
        ToolCall("b-legacy", "bash", {"cmd": "printf hello"})
    )

    assert new_result["isError"] is False
    assert legacy_result["isError"] is False
    assert new_result["structuredContent"]["stdout"] == "hello"
    assert legacy_result["structuredContent"]["stdout"] == "hello"


@pytest.mark.asyncio
async def test_exec_accepts_command_and_cmd_alias(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    new_result = await registry.execute(
        ToolCall("e-new", "exec", {"command": "true"})
    )
    legacy_result = await registry.execute(
        ToolCall("e-legacy", "exec", {"cmd": "true"})
    )

    assert new_result["isError"] is False
    assert legacy_result["isError"] is False


@pytest.mark.asyncio
async def test_bash_and_exec_missing_command_fail_cleanly(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    bash_missing = await registry.execute(ToolCall("b-nope", "bash", {}))
    exec_missing = await registry.execute(ToolCall("e-nope", "exec", {}))

    for result, tool in ((bash_missing, "bash"), (exec_missing, "exec")):
        assert result["isError"] is True
        error = result["structuredContent"]["error"]
        assert error["tool"] == tool
        assert "command" in error["message"]


def test_bash_and_exec_descriptions_route_long_running(tmp_path: Path) -> None:
    schemas = {schema["name"]: schema for schema in _registry(tmp_path).schemas}

    assert "run_background" in schemas["bash"]["description"]
    assert "run_background" in schemas["exec"]["description"]


def test_bash_and_exec_schemas_expose_both_command_keys(tmp_path: Path) -> None:
    schemas = {schema["name"]: schema for schema in _registry(tmp_path).schemas}
    bash_props = schemas["bash"]["parameters"]["properties"]
    exec_props = schemas["exec"]["parameters"]["properties"]

    assert "command" in bash_props and "cmd" in bash_props
    assert "command" in exec_props and "cmd" in exec_props
    assert "Deprecated" in bash_props["cmd"]["description"]
    assert "Deprecated" in exec_props["cmd"]["description"]


@pytest.mark.asyncio
async def test_exec_timeout_error_text_names_elapsed_limit(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "exec-timeout",
            "exec",
            {"command": "sleep 5", "timeout": 0.05},
        )
    )

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "timed out after 0.05s" in text
    assert result["structuredContent"]["timed_out"] is True
    assert result["structuredContent"]["timeout_seconds"] == 0.05
    error = result["structuredContent"]["error"]
    assert error["tool"] == "exec"
    assert error["kind"] == "timeout"
    assert "run_background" in error["hint"]


@pytest.mark.asyncio
async def test_todo_accepts_multiple_in_progress(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "todo-multi",
            "todo",
            {
                "items": [
                    {"content": "a", "status": "in_progress"},
                    {"content": "b", "status": "in_progress"},
                ]
            },
        )
    )

    assert result["isError"] is False
    assert result["structuredContent"]["counts"]["in_progress"] == 2


@pytest.mark.asyncio
async def test_unknown_tool_error_carries_governance(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)

    result = await registry.execute(ToolCall("call", "nope", {}))

    error = result["structuredContent"]["error"]
    assert error["tool"] == "nope"
    assert error["kind"] == "unknown_tool"
    assert error["hint"]


@pytest.mark.asyncio
async def test_invalid_arguments_error_carries_governance(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall("exec-bad", "exec", {"command": "true", "extra": "x"})
    )

    error = result["structuredContent"]["error"]
    assert error["tool"] == "exec"
    assert error["kind"] == "invalid_arguments"
    assert error["hint"]


@pytest.mark.asyncio
async def test_bash_exit_nonzero_error_carries_governance(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall("bash-fail", "bash", {"command": "exit 5"})
    )

    error = result["structuredContent"]["error"]
    assert error["tool"] == "bash"
    assert error["kind"] == "exit_nonzero"
    assert error["hint"]


@pytest.mark.asyncio
async def test_todo_error_carries_governance(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall("todo-bad", "todo", {"action": "read"})
    )

    error = result["structuredContent"]["error"]
    assert error["tool"] == "todo"
    assert error["kind"] == "invalid_arguments"
    assert error["hint"]
    assert "action" in error["message"]


@pytest.mark.asyncio
async def test_read_unknown_tilde_expansion_maps_to_error_kind(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall("read-tilde", "read", {"path": "~unknown_user_xyz/foo"})
    )

    assert result["isError"] is True
    error = result["structuredContent"]["error"]
    assert error["tool"] == "read"
    assert error["kind"] == "error"


@pytest.mark.asyncio
async def test_write_hardlinked_target_reports_sandbox_violation(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    target = tmp_path / "target"
    target.write_text("seed")
    link = tmp_path / "link"
    os.link(target, link)

    result = await registry.execute(
        ToolCall("write-link", "write", {"path": "link", "content": "next"})
    )

    assert result["isError"] is True
    error = result["structuredContent"]["error"]
    assert error["tool"] == "write"
    assert error["kind"] == "sandbox_violation"
    assert error["hint"]


@pytest.mark.asyncio
async def test_every_registered_tool_error_carries_governance(
    tmp_path: Path,
) -> None:
    """Governance is applied at the seam regardless of which tool errors."""

    registry = ToolRegistry(tmp_path, register_builtin=False)

    def broken(_arguments: object) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "boom",
                    "truncated": False,
                    "full_size": 4,
                }
            ],
            "isError": True,
            "structuredContent": None,
        }

    registry.register("broken", broken)
    result = await registry.execute(ToolCall("call", "broken", {}))

    error = result["structuredContent"]["error"]
    assert error["tool"] == "broken"
    assert error["kind"] == "error"
    assert error["message"] == "boom"


@pytest.mark.asyncio
async def test_governance_normalizes_unknown_kind_taxonomy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)

    def rogue(_arguments: object) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "rate limited",
                    "truncated": False,
                    "full_size": 12,
                }
            ],
            "isError": True,
            "structuredContent": {"error": {"kind": "rate_limited"}},
        }

    registry.register("rogue", rogue)
    with caplog.at_level("WARNING", logger="zeta.tools.registry"):
        result = await registry.execute(ToolCall("call", "rogue", {}))

    error = result["structuredContent"]["error"]
    assert error["kind"] == "error"
    assert any(
        "rate_limited" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_governance_preserves_caller_provided_tool_label(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False)

    def labeled(_arguments: object) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "downstream failure",
                    "truncated": False,
                    "full_size": 18,
                }
            ],
            "isError": True,
            "structuredContent": {"error": {"tool": "upstream-service"}},
        }

    registry.register("labeled", labeled)
    result = await registry.execute(ToolCall("call", "labeled", {}))

    error = result["structuredContent"]["error"]
    assert error["tool"] == "upstream-service"


def test_failed_agent_receipt_stays_under_cap_after_governance() -> None:
    max_bytes = 10_000
    long_answer = "x" * 20_000
    stats = {
        "turns_used": 3,
        "elapsed": 1.5,
        "tool_calls": 4,
        "error": True,
        "canceled": False,
    }
    structured_content = {"child_session_path": "/agents/1"}

    receipt = build_agent_receipt(
        "failed",
        long_answer,
        stats,
        structured_content=structured_content,
        tool_call_id="agent-1",
        max_bytes=max_bytes,
    )
    governed = _apply_error_governance(receipt, "agent")

    assert len(encode_json(governed)) <= max_bytes
