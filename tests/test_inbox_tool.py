from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools.inbox import _validate_action
from zeta.tools.registry import ToolRegistry


def _registry(tmp_path: Path, *, enabled: bool = True, deny: tuple[str, ...] = ()):
    home = tmp_path / ".zeta"
    projects = ProjectRegistry(home / "projects")
    project = projects.create_project("alpha", "alpha")
    store = ConversationStore(home / "sessions", session_id="a" * 32, cwd=tmp_path)
    registry = ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        project_id=project.project_id,
        project_registry=projects,
        inbox_enabled=enabled,
        tool_deny=deny,
    )
    registry.bind_session_store(store)
    return registry, store


def test_exactly_one_action_based_inbox_tool_is_registered(tmp_path: Path) -> None:
    registry, store = _registry(tmp_path)
    try:
        inbox_names = [name for name in registry.registered_names if "inbox" in name]
        assert inbox_names == ["inbox"]
        schema = registry.definitions_by_name["inbox"].parameters
        assert schema["required"] == ["action"]
        assert "oneOf" not in schema
        assert set(schema["properties"]["action"]["enum"]) == {
            "send", "list", "claim", "done", "projects"
        }
    finally:
        asyncio.run(registry.close())
        store.close()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"action": "send"}, "send action requires field(s): body, kind, project, title"),
        ({"action": "claim"}, "claim action requires field(s): id"),
        ({"action": "done", "id": "x"}, "done action requires field(s): outcome"),
        ({"action": "projects", "id": "x"}, "projects action does not accept field(s): id"),
        ({"action": "list", "title": "x"}, "list action does not accept field(s): title"),
    ],
)
def test_each_action_has_precise_validation(
    arguments: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate_action(arguments)


@pytest.mark.asyncio
async def test_tool_policy_can_deny_whole_inbox_tool(tmp_path: Path) -> None:
    registry, store = _registry(tmp_path, deny=("inbox",))
    try:
        assert "inbox" not in registry.registered_names
        result = await registry.execute(ToolCall("call", "inbox", {"action": "projects"}))
        assert result["isError"] is True
        assert "not allowed" in result["content"][0]["text"]
    finally:
        await registry.close()
        store.close()


def test_feature_switch_off_removes_inbox_tool(tmp_path: Path) -> None:
    registry, store = _registry(tmp_path, enabled=False)
    try:
        assert "inbox" not in registry.registered_names
    finally:
        asyncio.run(registry.close())
        store.close()
