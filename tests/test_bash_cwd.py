import asyncio
import shlex
from pathlib import Path

import pytest

from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent.approval import ChildApprovalPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("cloned_child", [False, True], ids=["session", "child"])
async def test_relative_cwd_override_uses_persisted_bash_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloned_child: bool,
) -> None:
    process_cwd = tmp_path / "process"
    persisted_cwd = tmp_path / "persisted"
    (process_cwd / "child").mkdir(parents=True)
    (persisted_cwd / "child").mkdir(parents=True)
    monkeypatch.chdir(process_cwd)

    parent_store = ConversationStore(tmp_path / "sessions", cwd=process_cwd)
    parent = ToolRegistry(
        process_cwd,
        session_store=parent_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry = parent
    if cloned_child:
        child_store = ConversationStore(
            parent_store.session_dir / "agents",
            session_id="1",
            cwd=process_cwd,
        )
        registry = parent.clone_for_session(child_store)

    try:
        changed = await registry.execute(
            ToolCall(
                "change-cwd",
                "bash",
                {"command": f"cd {shlex.quote(str(persisted_cwd))}"},
            )
        )
        result = await registry.execute(
            ToolCall("relative-cwd", "bash", {"command": "pwd", "cwd": "child"})
        )

        assert changed["structuredContent"]["cwd_after"] == str(persisted_cwd)
        assert result["structuredContent"]["stdout"].strip() == str(
            persisted_cwd / "child"
        )
    finally:
        if registry is not parent:
            await registry.close()
        await parent.close()


@pytest.mark.asyncio
async def test_run_background_relative_cwd_uses_persisted_bash_cwd(
    tmp_path: Path,
) -> None:
    process_cwd = tmp_path / "process"
    persisted_cwd = tmp_path / "persisted"
    (process_cwd / "child").mkdir(parents=True)
    (persisted_cwd / "child").mkdir(parents=True)
    registry = ToolRegistry(process_cwd, skill_catalog=SkillCatalog.empty())

    try:
        changed = await registry.execute(
            ToolCall(
                "change-cwd",
                "bash",
                {"command": f"cd {shlex.quote(str(persisted_cwd))}"},
            )
        )
        started = await registry.execute(
            ToolCall(
                "relative-background-cwd",
                "run_background",
                {"command": "pwd", "cwd": "child"},
            )
        )
        task_id = started["structuredContent"]["task_id"]
        await registry.background_tasks.wait(task_id)
        result = await registry.execute(
            ToolCall("background-output", "task_output", {"task_id": task_id})
        )

        assert changed["structuredContent"]["cwd_after"] == str(persisted_cwd)
        assert result["structuredContent"]["output"].strip() == str(
            persisted_cwd / "child"
        )
    finally:
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
async def test_delegated_shell_approval_uses_persisted_bash_cwd(
    tmp_path: Path,
    tool_name: str,
) -> None:
    initial_cwd = tmp_path / "initial"
    persisted_cwd = tmp_path / "persisted"
    (initial_cwd / "child").mkdir(parents=True)
    (persisted_cwd / "child").mkdir(parents=True)
    parent_store = ConversationStore(tmp_path / "parent", cwd=initial_cwd)
    child_store = ConversationStore(tmp_path / "child", cwd=initial_cwd)
    parent_policy = ApprovalPolicy(store=parent_store)
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "child-1",
        parent_cwd=initial_cwd,
        child_cwd=initial_cwd,
    )
    registry = ToolRegistry(
        initial_cwd,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
        approval_policy=child_policy,
    )

    try:
        changed = await registry.execute(
            ToolCall(
                "change-cwd",
                "bash",
                {"command": f"cd {shlex.quote(str(persisted_cwd))}"},
            ),
            _skip_approval=True,
        )
        call = ToolCall(
            "relative-cwd",
            tool_name,
            {"command": "pwd", "cwd": "child"},
        )
        execution = asyncio.create_task(registry.execute(call))
        while not parent_policy.pending_requests():
            await asyncio.sleep(0)

        request = parent_policy.pending_requests()[0]
        assert parent_policy.approve(request.key)
        result = await execution
        if tool_name == "run_background":
            task_id = result["structuredContent"]["task_id"]
            await registry.background_tasks.wait(task_id)
            result = await registry.execute(
                ToolCall("background-output", "task_output", {"task_id": task_id})
            )
            actual_cwd = result["structuredContent"]["output"].strip()
        else:
            actual_cwd = result["structuredContent"]["stdout"].strip()

        assert changed["structuredContent"]["cwd_after"] == str(persisted_cwd)
        assert actual_cwd == str(persisted_cwd / "child")
        assert request.effective_cwd == str(persisted_cwd / "child")
        assert request.audit_display()["effective_cwd"] == str(
            persisted_cwd / "child"
        )
    finally:
        await registry.close()


@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
def test_shell_schema_explains_cwd_resolution(
    tmp_path: Path, tool_name: str
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    description = registry.definitions_by_name[tool_name].parameters["properties"][
        "cwd"
    ].get("description", "")

    assert "persisted session cwd" in description
    assert "Absolute paths are used as given" in description
