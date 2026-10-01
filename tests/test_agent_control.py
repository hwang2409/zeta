from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.agent.background import BackgroundAgentOwner
from zeta.agent.presets import AGENT_PRESETS
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_owner_cancel_is_scoped_and_idempotent(
    tmp_path: Path,
) -> None:
    owner = BackgroundAgentOwner(ConversationStore(tmp_path))
    canceled: list[str] = []
    finished = asyncio.Event()

    async def watcher() -> None:
        await finished.wait()
        owner.unregister("root:1")

    task = asyncio.create_task(watcher())
    owner.register("root:1", lambda: (canceled.append("root:1"), finished.set()), task)
    assert owner.cancel("root:1") is True
    assert owner.cancel("root:1") is True
    assert canceled == ["root:1"]
    await task
    assert owner.cancel("root:1") is False


@pytest.mark.asyncio
async def test_owner_cancel_only_cancels_selected_subtree(tmp_path: Path) -> None:
    owner = BackgroundAgentOwner(ConversationStore(tmp_path))
    canceled: list[str] = []
    tasks: dict[str, asyncio.Task[None]] = {}
    for handle in ("root", "sibling", "grandchild"):
        tasks[handle] = asyncio.create_task(asyncio.sleep(10))
    owner.register("root", lambda: canceled.append("root"), tasks["root"])
    owner.register("sibling", lambda: canceled.append("sibling"), tasks["sibling"])
    owner.register(
        "grandchild", lambda: canceled.append("grandchild"), tasks["grandchild"],
        parent_instance_id="root",
    )

    assert owner.cancel("root") is True
    assert owner.cancel("root") is True
    assert canceled == ["root", "grandchild"]
    for task in tasks.values():
        task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)
    for handle in tuple(tasks):
        owner.unregister(handle)


@pytest.mark.asyncio
async def test_owner_cancel_follows_reparented_deep_descendants(tmp_path: Path) -> None:
    owner = BackgroundAgentOwner(ConversationStore(tmp_path))
    canceled: list[str] = []
    tasks = {name: asyncio.create_task(asyncio.sleep(10)) for name in (
        "survivor", "ancestor", "grandchild", "great-grandchild", "sibling"
    )}
    owner.register("survivor", lambda: canceled.append("survivor"), tasks["survivor"])
    owner.register(
        "ancestor", lambda: canceled.append("ancestor"), tasks["ancestor"],
        parent_instance_id="survivor",
    )
    owner.register(
        "grandchild", lambda: canceled.append("grandchild"), tasks["grandchild"],
        parent_instance_id="ancestor",
    )
    owner.register(
        "great-grandchild", lambda: canceled.append("great-grandchild"),
        tasks["great-grandchild"], parent_instance_id="grandchild",
    )
    owner.register("sibling", lambda: canceled.append("sibling"), tasks["sibling"])

    # Each completing ancestor adopts its live child through its current edge.
    owner.adopt("grandchild", ConversationStore(tmp_path / "survivor"), "survivor")
    owner.unregister("ancestor")
    owner.adopt("great-grandchild", ConversationStore(tmp_path / "survivor"), "survivor")
    owner.unregister("grandchild")

    assert owner.cancel("survivor") is True
    assert canceled == ["survivor", "great-grandchild"]
    assert "sibling" not in canceled
    for task in tasks.values():
        task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)
    for handle in ("survivor", "great-grandchild", "sibling"):
        owner.unregister(handle)


@pytest.mark.asyncio
async def test_owner_registration_does_not_wake_frontend(tmp_path: Path) -> None:
    owner = BackgroundAgentOwner(ConversationStore(tmp_path))
    wakes: list[str] = []
    owner.set_wake_callback(lambda: wakes.append("wake"))
    task = asyncio.create_task(asyncio.sleep(10))
    owner.register("child", lambda: None, task)
    assert wakes == []
    owner.notify_wake()
    assert wakes == ["wake"]
    owner.unregister("child")
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_agent_control_tools_execute_through_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path / "session")
    registry = ToolRegistry(
        tmp_path, session_store=store, skill_catalog=SkillCatalog.empty(),
        register_builtin=True,
    )
    owner = BackgroundAgentOwner(store)
    task = asyncio.create_task(asyncio.sleep(10))
    owner.register("child", lambda: None, task)
    registry._agent_owner = owner
    monkeypatch.setattr(
        "zeta.tools.agent._read_agent_status", lambda _: [{"handle": "child"}]
    )
    try:
        canceled = await registry.execute(ToolCall("cancel", "agent_cancel", {"handle": "child"}))
        assert canceled["structuredContent"]["status"] == "cancellation_requested"
        owner.unregister("child")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await registry.close()


@pytest.mark.asyncio
async def test_removed_agent_wait_is_an_unknown_tool_result(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path, skill_catalog=SkillCatalog.empty(), register_builtin=True
    )
    try:
        result = await registry.execute(
            ToolCall("stale", "agent_wait", {"handles": ["child"]})
        )
        assert result["isError"] is True
        assert result["content"][0]["text"] == "unknown tool: agent_wait"
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_agent_control_schemas_are_bounded_and_model_facing() -> None:
    registry = ToolRegistry(
        Path.cwd(), skill_catalog=SkillCatalog.empty(), register_builtin=True
    )
    try:
        schemas = {schema["name"]: schema for schema in registry.schemas}
        assert schemas["agent_cancel"]["parameters"]["required"] == ["handle"]
        assert "agent_cancel" in registry.registered_names
        assert "agent_wait" not in registry.registered_names
        assert all(
            preset.tool_names is None or "agent_wait" not in preset.tool_names
            for preset in AGENT_PRESETS.values()
        )
    finally:
        await registry.close()
