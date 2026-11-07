from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent.approval import ChildApprovalPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
async def test_delegated_shell_executes_in_open_approved_directory_after_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    parent_cwd = tmp_path / "parent"
    approved_cwd = parent_cwd / "approved"
    replacement = tmp_path / "replacement"
    parent_cwd.mkdir()
    approved_cwd.mkdir()
    replacement.mkdir()
    alias = tmp_path / "cwd-alias"
    alias.symlink_to(approved_cwd, target_is_directory=True)

    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=approved_cwd)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=approved_cwd)
    child_store.set_bash_cwd(str(alias))
    parent_policy = ApprovalPolicy(
        store=parent_store,
        always_allow={f"{tool_name}(touch *)"},
    )
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "spawn-swap",
        parent_cwd=approved_cwd,
        child_cwd=approved_cwd,
    )
    registry = ToolRegistry(
        parent_cwd,
        session_store=child_store,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)

    if tool_name == "bash":
        import zeta.tools.bash as spawn_module
    else:
        import zeta.tools._shared.process as spawn_module

    original_spawn: Callable[..., Awaitable[Any]] = (
        spawn_module.create_subprocess_shell_in_fd
    )

    async def swap_then_spawn(*args: object, **kwargs: object) -> Any:
        approved_cwd.rename(tmp_path / "approved-renamed")
        replacement.rename(approved_cwd)
        return await original_spawn(*args, **kwargs)

    monkeypatch.setattr(
        spawn_module,
        "create_subprocess_shell_in_fd",
        swap_then_spawn,
    )
    arguments = {"command": "touch marker.txt"}
    if tool_name == "run_background":
        arguments["cwd"] = str(alias)

    try:
        result = await registry.execute(ToolCall(f"fd-{tool_name}", tool_name, arguments))
        assert result["isError"] is False
        if tool_name == "run_background":
            task_id = result["structuredContent"]["task_id"]
            await registry.background_tasks.wait(task_id)
    finally:
        await registry.close()

    assert (tmp_path / "approved-renamed" / "marker.txt").exists()
    assert not (approved_cwd / "marker.txt").exists()
