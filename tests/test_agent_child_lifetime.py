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


@pytest.mark.asyncio
@pytest.mark.parametrize("point", [
    "lineage", "append_child_start", "child_snapshot", "start_lifecycle",
    "build_child_loop", "base_system_prompt", "refresh_live_config", "step_delivery",
    "runtime_shutdown", "serve_receipt", "cleanup_cancel", "cancel_read", "cancel_confirmation",
    "cancel_reread", "success",
])
async def test_child_owner_lifetime(tmp_path, monkeypatch, point):
    entered = asyncio.Event()
    release = asyncio.Event()
    running = asyncio.Event()
    cleanup_entered = asyncio.Event()
    tasks = []
    closed = False

    async def pause(*args, **kwargs):
        entered.set()
        await await_or_fail(release.wait(), what=point)

    async def consume(*args, **kwargs):
        running.set()
        await await_or_fail(release.wait(), what="child runs until canceled")

    with ConversationStore(tmp_path / "sessions", cwd=tmp_path) as store:
        loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())
        owner = loop._background_owner
        call = ToolCall("agent-1", "agent", {
            "prompt": "inspect the task", "description": "test child", "background": True,
        })
        read_count = 0
        original_offload = runner.await_joined_offload

        async def offload(fn, *args, **kwargs):
            nonlocal read_count
            if fn.__name__ == "state":
                read_count += 1
                if (point, read_count) in {("cancel_read", 1), ("cancel_reread", 2)}:
                    await pause()
            if (point, fn.__name__) in {
                ("lineage", "publish"), ("child_snapshot", "child_snapshot"),
                ("base_system_prompt", "_child_base_system_prompt"),
            }:
                await pause()
            result = await original_offload(fn, *args, **kwargs)
            if point == fn.__name__ == "start_lifecycle":
                await pause()
            return result

        # test-contract: seam pause at the real owner operation, not a timed race.
        monkeypatch.setattr(runner, "await_joined_offload", offload)
        if point == "lineage":
            def publish(link):
                pass

            loop._child_lineage_publisher = SimpleNamespace(publish=publish)
        if point == "child_snapshot":
            def child_snapshot(provider):
                return None

            loop.account_context = SimpleNamespace(child_snapshot=child_snapshot)
        if point == "append_child_start":
            monkeypatch.setattr(runner, "append_child_start", pause)
        if point in {"build_child_loop", "runtime_shutdown", "cleanup_cancel"}:
            # test-contract: seam pause loop construction while setup owns the child store.
            monkeypatch.setattr(runner, "_build_child_loop", pause)
        if point == "cleanup_cancel":
            original_persist = runner.persist_confirmed_cancel

            async def persist(*args, **kwargs):
                cleanup_entered.set()
                await await_or_fail(release.wait(), what="release setup cancellation write")
                return await original_persist(*args, **kwargs)

            monkeypatch.setattr(runner, "persist_confirmed_cancel", persist)
        if point == "refresh_live_config":
            monkeypatch.setattr(AgentLoop, "refresh_live_config", pause)
        if point == "step_delivery":
            # test-contract: seam pause step delivery before setup releases ownership.
            monkeypatch.setattr(AgentLoop, "_ensure_step_delivery", pause)
        if point == "serve_receipt":
            # test-contract: seam hold the receipt write so shutdown must join it.
            monkeypatch.setattr(store, "_commit_address_async", pause)

        async def spawn():
            return await runner.spawn_child(loop, agent_type="general", task="test child", cwd=None)

        try:
            if point.startswith("cancel_") or point == "success":
                monkeypatch.setattr(runner, "consume_child", consume)
                monkeypatch.setattr(runner, "consume_run", consume)
                run_id = await asyncio.wait_for(spawn(), 10)
                await await_or_fail(running.wait(), what="child runs after spawn")
                assert owner.owns_running(run_id)
                child_store, = owner._stores
                if point == "cancel_confirmation":
                    original_confirm = runner.persist_confirmed_cancel

                    async def confirm(*args, **kwargs):
                        await pause()
                        return await original_confirm(*args, **kwargs)

                    monkeypatch.setattr(runner, "persist_confirmed_cancel", confirm)
                if point == "cancel_reread":
                    original_cancel = owner.cancel_and_wait

                    async def cancel_after_exit(run_id):
                        owner.cancel(run_id)
                        await await_or_fail(owner.wait(), what="child exits before confirmation")
                        return await original_cancel(run_id)

                    monkeypatch.setattr(owner, "cancel_and_wait", cancel_after_exit)
                if point == "success":
                    assert await asyncio.wait_for(runner.cancel_child_run(loop, run_id), 10) == "cancelling"
                else:
                    task = asyncio.create_task(runner.cancel_child_run(loop, run_id))
                    tasks.append(task)
                    await await_or_fail(entered.wait(), what=point)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 10)
                    if point == "cancel_confirmation":
                        # The confirmation is still owned after its request waiter exits.
                        release.set()
                await await_or_fail(owner.wait(), what="settle canceled child")
            else:
                shutdown = point in {"runtime_shutdown", "serve_receipt"}
                task = asyncio.create_task(spawn() if shutdown else loop._run_agent_tool(
                    call, call.arguments, AbortGenerationRegistry().new_generation(), None,
                ))
                tasks.append(task)
                await await_or_fail(entered.wait(), what=point)
                child_store, = owner._stores
                run_id, = store.agent_children()
                if shutdown:
                    assert owner.owns_running(run_id) == (point == "serve_receipt")
                    await await_or_fail(loop.close(), what="runtime joins spawn setup")
                    closed = True
                else:
                    task.cancel()
                    if point == "cleanup_cancel":
                        await await_or_fail(cleanup_entered.wait(), what="setup starts cancellation write")
                        task.cancel()
                        release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)

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
            assert not owner.conversation_channel._wakes
            assert not owner._setup_tasks
            assert not owner._stores
            assert not owner._store_close_tasks
            assert child_store.directory_fd == -1
            if not closed:
                for _ in range(2):
                    assert await asyncio.wait_for(runner.cancel_child_run(loop, run_id), 10) == "canceled"
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
            if not closed:
                await loop.close()
