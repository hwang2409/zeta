import shlex
from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


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


def test_bash_schema_explains_cwd_resolution(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    description = registry.definitions_by_name["bash"].parameters["properties"][
        "cwd"
    ].get("description", "")

    assert "persisted session cwd" in description
    assert "Absolute paths are used as given" in description
