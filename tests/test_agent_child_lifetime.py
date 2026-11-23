from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.agent import runner
from zeta.core.abort import AbortGenerationRegistry
from zeta.core.child_runs import LoadedAgentLifecycle, read_child_lifecycle
from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

SETUP_POINTS = (
    "build_child_loop", "lineage", "append_child_start", "child_snapshot",
    "start_lifecycle", "base_system_prompt", "refresh_live_config", "step_delivery",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "close"])
@pytest.mark.parametrize("point", SETUP_POINTS)
async def test_child_cancellation_during_setup(tmp_path, monkeypatch, action, point):
    await _assert_setup_canceled(tmp_path, monkeypatch, action, point)


@pytest.mark.asyncio
async def test_repeated_close_joins_setup_cleanup(tmp_path, monkeypatch):
    await _assert_setup_canceled(tmp_path, monkeypatch, "repeat_close", "build_child_loop")


async def _assert_setup_canceled(tmp_path, monkeypatch, action, point):
    entered = asyncio.Event()
    release = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    tasks = []

    async def pause(*args, **kwargs):
        entered.set()
        await await_or_fail(release.wait(), what=f"release {point}")

    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
        owner = loop._background_owner
        call = ToolCall("agent-1", "agent", {
            "prompt": "inspect the task", "description": "test child", "background": True,
        })

        # test-contract: gate at the real setup awaits, not a timed race.
        def gate_offload(original, lifecycle_name):
            async def offload(fn, *args, **kwargs):
                name = getattr(fn, "__name__", None)
                if (point, name) in {
                    ("lineage", "publish"), ("child_snapshot", "child_snapshot"),
                    ("base_system_prompt", "_child_base_system_prompt"),
                }:
                    await pause()
                result = await original(fn, *args, **kwargs)
                if point == "start_lifecycle" and name == lifecycle_name:
                    await pause()
                return result
            return offload

        monkeypatch.setattr(
            runner, "await_joined_offload",
            gate_offload(runner.await_joined_offload, "start_lifecycle"),
        )
        monkeypatch.setattr(
            asyncio, "to_thread", gate_offload(asyncio.to_thread, "publish_child_lifecycle"),
        )
        if point == "lineage":
            def publish(link):
                pass

            loop._child_lineage_publisher = SimpleNamespace(publish=publish)
        if point == "child_snapshot":
            def child_snapshot(provider):
                return None

            loop.account_context = SimpleNamespace(child_snapshot=child_snapshot)
        if point in {"append_child_start", "build_child_loop"}:
            monkeypatch.setattr(runner, f"_{point}" if point == "build_child_loop" else point, pause)
        if point in {"refresh_live_config", "step_delivery"}:
            name = "_ensure_step_delivery" if point == "step_delivery" else point
            monkeypatch.setattr(AgentLoop, name, pause)
        if action == "repeat_close":
            original_persist = runner.persist_confirmed_cancel

            async def persist(*args, **kwargs):
                cleanup_entered.set()
                await await_or_fail(release_cleanup.wait(), what="release cancellation write")
                return await original_persist(*args, **kwargs)

            monkeypatch.setattr(runner, "persist_confirmed_cancel", persist)

        def spawn():
            return loop._run_agent_tool(
                call, call.arguments, AbortGenerationRegistry().new_generation(), None,
            )

        closed = False
        try:
            task = asyncio.create_task(spawn())
            tasks.append(task)
            await await_or_fail(entered.wait(), what=f"setup reaches {point}")
            child_store, = owner._stores
            run_id, = store.agent_children()
            assert run_id not in owner.active_instance_ids
            if action == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(task, what="join canceled setup")
            else:
                closing = asyncio.create_task(loop.close())
                tasks.append(closing)
                if action == "repeat_close":
                    await await_or_fail(cleanup_entered.wait(), what="close starts cancellation write")
                    assert not closing.done()
                    assert child_store.directory_fd != -1
                    release_cleanup.set()
                await await_or_fail(closing, what="close joins setup")
                closed = True

            lifecycle, ownership = read_child_lifecycle(store, child_store.session_dir.absolute())
            assert isinstance(lifecycle, LoadedAgentLifecycle)
            assert lifecycle.value["state"] == "canceled"
            assert lifecycle.value["handle"] == run_id
            assert ownership is None
            assert not loop._agent_child_stores
            assert not loop._agent_child_turns
            assert not loop._agent_child_types
            assert not loop._background_child_cancellers
            assert not loop._background_child_watchers
            assert not owner.project_context_loops
            assert not owner.active_instance_ids
            assert not owner._setup_tasks
            assert not owner._stores
            assert not owner._store_close_tasks
            assert child_store.directory_fd == -1
            if closed:
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(task, what="join setup canceled by close")
            if action == "repeat_close":
                await await_or_fail(loop.close(), what="close again")
                with pytest.raises(asyncio.CancelledError):
                    await await_or_fail(spawn(), what="closed owner refuses setup")
        finally:
            release.set()
            release_cleanup.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await await_or_fail(asyncio.gather(*tasks, return_exceptions=True), what="drain test tasks")
            if not closed:
                await await_or_fail(loop.close(), what="close root after test")
