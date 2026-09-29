from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zeta.agent.background import BackgroundAgentOwner
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import agent_control
from zeta.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_owner_cancel_is_scoped_idempotent_and_wait_is_event_driven(
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
    assert await owner.wait_for({"root:1"}, 1) is True
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
async def test_owner_wait_arms_before_completion_and_wakes_all_waiters(
    tmp_path: Path,
) -> None:
    owner = BackgroundAgentOwner(ConversationStore(tmp_path))
    finished = asyncio.Event()
    task = asyncio.create_task(finished.wait())
    owner.register("child", lambda: None, task)
    waiters = [asyncio.create_task(owner.wait_for({"child"}, 1)) for _ in range(3)]
    await asyncio.sleep(0)
    owner.unregister("child")
    assert await asyncio.gather(*waiters) == [True, True, True]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


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
async def test_owner_wait_times_out_without_polling(tmp_path: Path) -> None:
    owner = BackgroundAgentOwner(ConversationStore(tmp_path))
    task = asyncio.create_task(asyncio.sleep(10))
    owner.register("root:1", lambda: None, task)
    assert await owner.wait_for({"root:1"}, 0.001) is False
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
        waited = await registry.execute(
            ToolCall("wait", "agent_wait", {"handles": ["child"], "timeout": 0})
        )
        assert waited["structuredContent"]["timed_out"] is False
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await registry.close()


class _RaceOwner:
    def __init__(self, result: bool, signal: AbortSignal) -> None:
        self.result = result
        self.signal = signal

    def owns_running(self, handle: str) -> bool:
        return handle == "child"

    async def wait_for(self, handles: set[str], timeout: float) -> bool:
        del handles, timeout
        self.signal.abort()
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(("completed", "canceled"), ((True, False), (False, True)))
async def test_agent_wait_abort_races_use_standard_cancel_semantics(
    completed: bool, canceled: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = AbortSignal()
    owner = _RaceOwner(completed, signal)
    registry = type("Registry", (), {
        "_agent_owner": owner, "max_output_chars": 10000,
        "session_store": object(),
    })()
    monkeypatch.setattr(
        "zeta.tools.agent._read_agent_status", lambda _: [{"handle": "child"}]
    )
    if canceled:
        with pytest.raises(asyncio.CancelledError):
            await agent_control.agent_wait(registry, {"handles": ["child"]}, signal)
    else:
        result = await agent_control.agent_wait(
            registry, {"handles": ["child"]}, signal
        )
        assert result["structuredContent"]["timed_out"] is False


@pytest.mark.asyncio
async def test_agent_control_schemas_are_bounded_and_model_facing() -> None:
    registry = ToolRegistry(
        Path.cwd(), skill_catalog=SkillCatalog.empty(), register_builtin=True
    )
    try:
        schemas = {schema["name"]: schema for schema in registry.schemas}
        assert schemas["agent_cancel"]["parameters"]["required"] == ["handle"]
        wait = schemas["agent_wait"]["parameters"]
        assert wait["required"] == ["handles"]
        assert wait["properties"]["timeout"]["maximum"] == 300
        assert "agent_cancel" in registry.registered_names
        assert "agent_wait" in registry.registered_names
    finally:
        await registry.close()
