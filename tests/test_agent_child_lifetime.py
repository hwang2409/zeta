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
    "lineage", "append_child_start", "child_snapshot", "start_lifecycle",
    "build_child_loop", "base_system_prompt", "refresh_live_config", "step_delivery",
)
BEFORE_MARKER = ("resolve_cwd", "resolve_child_backend", "ensure_mcp", "open_child_store", "activate_runtime_lease")


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "point"), [
    *(("cancel", point) for point in SETUP_POINTS),
    *(("close", point) for point in (*BEFORE_MARKER, *SETUP_POINTS, "serve_receipt", "cleanup_cancel")),
    *(("cancel", point) for point in ("cleanup_cancel", "cancel_read", "cancel_confirmation", "cancel_reread", "success")),
    *(("tool_close", point) for point in SETUP_POINTS),
    ("tool_repeat_close", "build_child_loop"),
    ("repeat_close", "cleanup_cancel"),
])
async def test_child_owner_lifetime(tmp_path, monkeypatch, action, point):
    entered = asyncio.Event()
    release = asyncio.Event()
    running = asyncio.Event()
    cleanup_entered = asyncio.Event()
    tasks = []
    closed = False
    tool_entry = action == "cancel" or action.startswith("tool_")
    shutdown = action != "cancel"
    repeat_close = action.endswith("repeat_close")
    gate_cleanup = point == "cleanup_cancel" or repeat_close
    opened = []

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
        if point in {"child_snapshot", "resolve_child_backend"}:
            def child_snapshot(provider):
                return None

            loop.account_context = SimpleNamespace(child_snapshot=child_snapshot)
        if point == "append_child_start":
            monkeypatch.setattr(runner, "append_child_start", pause)
        if point in {"build_child_loop", "cleanup_cancel"}:
            # test-contract: seam pause loop construction while setup owns the child store.
            monkeypatch.setattr(runner, "_build_child_loop", pause)
        if gate_cleanup:
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

        if point in {"resolve_cwd", "resolve_child_backend"}:
            original_thread = asyncio.to_thread

            async def to_thread(fn, *args, **kwargs):
                if fn.__name__ == point:
                    await pause()
                return await original_thread(fn, *args, **kwargs)

            monkeypatch.setattr(asyncio, "to_thread", to_thread)
        if point == "ensure_mcp":
            # test-contract: seam pause MCP setup before any child store is opened.
            monkeypatch.setattr(loop, "_ensure_mcp_servers", pause)
        if point in {"open_child_store", "activate_runtime_lease"}:
            original_store_offload = ConversationStore._to_thread_store

            async def store_offload(self, fn):
                result = await original_store_offload(self, fn)
                if fn.__name__ == "open_child_store":
                    opened.append(result)
                if fn.__name__ == point:
                    await pause()
                return result

            # test-contract: seam cancel on the return edge of joined store work.
            monkeypatch.setattr(ConversationStore, "_to_thread_store", store_offload)

        async def spawn(*, via_tool=False):
            if via_tool:
                return await loop._run_agent_tool(
                    call, call.arguments, AbortGenerationRegistry().new_generation(), None,
                )
            return await runner.spawn_child(
                loop, agent_type="general", task="test child",
                cwd=str(tmp_path) if point == "resolve_cwd" else None,
            )

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
                task = asyncio.create_task(spawn(via_tool=tool_entry))
                tasks.append(task)
                await await_or_fail(entered.wait(), what=point)
                if point not in BEFORE_MARKER:
                    child_store, = owner._stores
                    run_id, = store.agent_children()
                    assert owner.owns_running(run_id) == (point == "serve_receipt")
                if shutdown:
                    closing = asyncio.create_task(loop.close())
                    tasks.append(closing)
                    if gate_cleanup:
                        await await_or_fail(cleanup_entered.wait(), what="close starts cancellation write")
                        assert not closing.done()
                        assert child_store.directory_fd != -1
                        release.set()
                    await await_or_fail(closing, what="close joins setup")
                    closed = True
                    if repeat_close:
                        await await_or_fail(loop.close(), what="repeat close")
                        with pytest.raises(asyncio.CancelledError):
                            await await_or_fail(spawn(via_tool=tool_entry), what="closed owner refuses setup")
                else:
                    task.cancel()
                    if point == "cleanup_cancel":
                        await await_or_fail(cleanup_entered.wait(), what="setup starts cancellation write")
                        task.cancel()
                        release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)

            if point in BEFORE_MARKER:
                assert not store.agent_children()
                assert not owner.active_instance_ids
                assert not owner._setup_tasks
                assert not owner._stores
                assert len(opened) == (point in {"open_child_store", "activate_runtime_lease"})
                assert all(child.directory_fd == -1 for child in opened)
                return

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
            await await_or_fail(asyncio.gather(*tasks, return_exceptions=True), what="drain test tasks")
            if not closed:
                await await_or_fail(loop.close(), what="close root after test")
