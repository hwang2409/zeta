from pathlib import Path

import pytest

from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.agent.approval import ChildApprovalPolicy


@pytest.mark.asyncio
async def test_delegated_write_uses_approved_canonical_target_after_alias_swap(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    approved_dir = parent_cwd / "src"
    outside = tmp_path / "outside"
    approved_dir.mkdir(parents=True)
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(approved_dir, target_is_directory=True)
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=parent_cwd)
    parent_policy = ApprovalPolicy(
        store=parent_store,
        always_allow={"write(src/**)"},
    )
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "child-write",
        parent_cwd=parent_cwd,
        child_cwd=parent_cwd,
    )

    def swap_alias(_name: str, _arguments: dict[str, object]) -> bool:
        alias.unlink()
        alias.symlink_to(outside, target_is_directory=True)
        return True

    registry = ToolRegistry(
        parent_cwd,
        session_store=child_store,
        pre_execute_hook=swap_alias,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)
    try:
        result = await registry.execute(
            ToolCall(
                "delegated-write-swap",
                "write",
                {"path": str(alias / "pwned.txt"), "content": "approved"},
            )
        )
    finally:
        await registry.close()

    assert result["isError"] is False
    assert (approved_dir / "pwned.txt").read_text() == "approved"
    assert not (outside / "pwned.txt").exists()


@pytest.mark.asyncio
async def test_delegated_bash_uses_approved_canonical_cwd_after_alias_swap(
    tmp_path: Path,
) -> None:
    parent_cwd = tmp_path / "parent"
    outside = tmp_path / "outside"
    parent_cwd.mkdir()
    outside.mkdir()
    alias = tmp_path / "cwd-alias"
    alias.symlink_to(parent_cwd, target_is_directory=True)
    parent_store = ConversationStore(tmp_path / "parent-sessions", cwd=parent_cwd)
    child_store = ConversationStore(tmp_path / "child-sessions", cwd=parent_cwd)
    child_store.set_bash_cwd(str(alias))
    parent_policy = ApprovalPolicy(
        store=parent_store,
        always_allow={"bash(touch *)"},
    )
    child_policy = ChildApprovalPolicy(
        parent_policy,
        child_store,
        "child",
        "child-bash",
        parent_cwd=parent_cwd,
        child_cwd=parent_cwd,
    )

    def swap_alias(_name: str, _arguments: dict[str, object]) -> bool:
        alias.unlink()
        alias.symlink_to(outside, target_is_directory=True)
        return True

    registry = ToolRegistry(
        parent_cwd,
        session_store=child_store,
        pre_execute_hook=swap_alias,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_approval_policy(child_policy)
    try:
        result = await registry.execute(
            ToolCall("delegated-bash-swap", "bash", {"command": "touch pwned.txt"})
        )
    finally:
        await registry.close()

    assert result["isError"] is False
    assert (parent_cwd / "pwned.txt").exists()
    assert not (outside / "pwned.txt").exists()
