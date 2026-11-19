from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from io import StringIO

import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail, close_owned_app
from tests.support.replay_transport import connect_replay, negotiate, replay_server
from zeta.agent.terminal_fence import TerminalTransitionFence
from zeta.protocol.types import StreamEventType, TextContent, ToolCall
from zeta.providers.completion import owned_completion
from zeta.tui.app import TUIApp

pytest_plugins = ("tests.support.socket_fixtures",)
_FEATURES = ("child_run_control_v1", "child_run_events", "child_run_history", "child_runs")


class _GatedChild:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.backend = FakeBackend([
            ScriptedTurn(tool_calls=[ToolCall("finish-child", "finish", {
                "status": "done", "summary": "child finished",
            })]),
            ScriptedTurn([TextContent("parent noted completion")]),
        ])

    @owned_completion
    async def complete(self, messages, tool_schemas):
        self.started.set()
        await await_or_fail(self.release.wait(), what="release child completion")
        async for event in FakeBackend.complete.__wrapped__(self.backend, messages, tool_schemas):
            yield event


@asynccontextmanager
async def _session(tmp_path, backend):
    server = replay_server(tmp_path, "unix", lambda *_args: (backend, "offline"))
    async with AsyncExitStack() as cleanup:
        await server.start()
        cleanup.push_async_callback(server.close)
        client = await connect_replay(server, "unix")
        cleanup.push_async_callback(client.close)
        await negotiate(client, extra=_FEATURES, protocol_version="1.2")
        result = await client.request(2, "new_session", {"provider": "codex"})
        yield server, client, result["result"]["session"]["session_id"]


@pytest.mark.asyncio
async def test_child_control_rejects_observer_and_invalid_inputs_before_persistence(tmp_path):
    backend = FakeBackend([])
    async with _session(tmp_path, backend) as (server, controller, session_id):
        store = server.runtime.loop.store
        head = store.active_branch_head_id()
        observer = await connect_replay(server, "unix")
        try:
            await negotiate(observer, observer=True, extra=_FEATURES, protocol_version="1.2")
            for index, method in enumerate(("child_run_spawn", "child_run_cancel"), 3):
                reply = await observer.request(index, method, {"unknown": "invalid"})
                assert reply["error"]["code"] == -32008
                assert reply["error"]["data"] == {
                    "code": "observer_read_only", "reason": "observer_read_only", "method": method,
                }
        finally:
            await observer.close()

        args = {"session_id": session_id, "agent_type": "general", "task": "test spawn"}
        for index, change in enumerate((
            {"agent_type": "unknown-preset"}, {"cwd": str(tmp_path / "missing")},
            {"model": "override"}, {"tools": []}, {"background": False},
        ), 10):
            reply = await controller.request(index, "child_run_spawn", {**args, **change})
            assert reply["error"]["code"] == -32602
        for index, (method, params) in enumerate((
            ("child_run_spawn", args), ("child_run_cancel", {"run_id": f"{session_id}:1"}),
        ), 20):
            reply = await controller.request(index, method, {**params, "session_id": "other-session"})
            assert reply["error"]["code"] == -32003
            assert reply["error"]["data"]["reason"] == "session_mismatch"
        missing = await controller.request(22, "child_run_cancel", {
            "session_id": session_id, "run_id": f"{session_id}:1",
        })
        assert missing["error"]["code"] == -32012
        assert store.active_branch_head_id() == head
        assert not store.agent_children()
        assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["completed", "canceled"])
async def test_serve_spawn_lifecycle_and_cancel_race(tmp_path, monkeypatch, winner):
    backend = _GatedChild()
    async with _session(tmp_path, backend) as (server, controller, session_id):
        loop = server.runtime.loop
        loop.explicit_finish = True
        # test-contract: seam make this run use the real explicit-finish fence
        catalog = loop.tool_registry.agent_catalog
        monkeypatch.setattr(loop.tool_registry, "agent_catalog", replace(catalog, agents=tuple(
            replace(preset, explicit_finish=True) if preset.name == "run" else preset
            for preset in catalog.agents
        )))
        sink = loop._background_event_sink
        output = StringIO()
        tui = TUIApp(loop, provider="codex", model="offline", console=Console(file=output))

        progress = []

        def render(card):
            console = Console(file=StringIO(), width=120)
            with console.capture() as rendered:
                console.print(card)
            return rendered.get()

        def publish(event):
            sink(event)
            tui._handle_background_event(event)

            if event.type is StreamEventType.TOOL_EXECUTION_UPDATE and tui._presenter.tool_region:
                progress.append(render(tui._presenter.tool_region.renderable))

        loop.set_background_event_sink(publish)
        try:
            joined = await controller.request(3, "session_history_and_subscribe", {"session_id": session_id})
            assert "result" in joined, joined
            response = await controller.request(4, "child_run_spawn", {
                "session_id": session_id, "agent_type": "run", "task": "inspect child lifecycle",
                "cwd": str(tmp_path),
            })
            assert "result" in response, response
            run_id = response["result"]["data"]["run_id"]
            assert response["result"] == {"data": {"accepted": True, "run_id": run_id}}
            await await_or_fail(backend.started.wait(), what="child enters model call")
            for kind in ("child_started", "child_progress"):
                frame = await controller.event(kind, {"run_id": run_id})
                assert frame["child"]["run_id"] == run_id
            listed = await controller.request(8, "child_run_list", {"session_id": session_id})
            assert [row["run_id"] for row in listed["result"]["runs"]] == [run_id]
            assert "inspect child lifecycle" in output.getvalue()
            assert "running" in output.getvalue()
            unit, = tui._presenter._tool_region_units.values()
            assert any("thinking" in snapshot for snapshot in progress)
            # test-contract: seam force finish to complete while cancellation is at the fence
            original_cancel = TerminalTransitionFence.cancel

            async def finish_before_cancel(fence):
                backend.release.set()
                await await_or_fail(loop._background_owner.wait(), what="finish wins")
                return await original_cancel(fence)

            if winner == "completed":
                monkeypatch.setattr(TerminalTransitionFence, "cancel", finish_before_cancel)
            params = {"session_id": session_id, "run_id": run_id, "reason": "user_cancelled"}
            cancelled = await controller.request(5, "child_run_cancel", params)
            assert cancelled["result"]["data"]["state"] == (
                "completed" if winner == "completed" else "cancelling"
            ), cancelled
            ended = await controller.event("child_ended", {"run_id": run_id})
            assert ended["child"]["status"] == winner
            first = await controller.request(6, "child_run_cancel", params)
            second = await controller.request(7, "child_run_cancel", params)
            assert first["result"] == second["result"] == {
                "data": {"accepted": True, "run_id": run_id, "state": winner},
            }
            await await_or_fail(loop._background_owner.wait(), what="publish terminal TUI card")
            assert unit.finished
            # The expanded card shows the child transcript; collapse it for the receipt.
            assert winner in render(unit.card.toggle())
        finally:
            backend.release.set()
            loop.set_background_event_sink(sink)
            await close_owned_app(tui)
