from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import PerActorFakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server import TIMEOUT, _connect, _request
from zeta.agent import runner
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.tools.agent.approval import ChildApprovalPolicy

pytest_plugins = ("tests.support.socket_fixtures",)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_nested_foreground_child_forwards_agent_start(
    tmp_path: Path, transport: Transport
) -> None:
    child = ToolCall(
        "child",
        "agent",
        {"prompt": "child-work", "description": "child", "background": False},
    )
    grandchild = ToolCall(
        "grandchild",
        "agent",
        {"prompt": "grand-work", "description": "grandchild", "background": False},
    )
    backend = PerActorFakeBackend(
        [ScriptedTurn(tool_calls=[child]), ScriptedTurn([TextContent("parent done")])],
        {
            "child-work": [
                ScriptedTurn(tool_calls=[grandchild]),
                ScriptedTurn([TextContent("child done")]),
            ],
            "grand-work": [ScriptedTurn([TextContent("grandchild done")])],
        },
        parent_prompt="delegate",
        gated_child_prompts=(),
        parent_waits_for_children=False,
    )
    server = replay_server(
        tmp_path, transport, lambda provider, model, home: (backend, model or "offline")
    )
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    try:
        await negotiate(controller)
        await negotiate(observer, observer=True)
        created = await controller.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        joined = await observer.request(
            3, "session_history_and_subscribe", {"session_id": session_id}
        )
        assert "error" not in joined, joined
        await controller.request(4, "send", {"text": "delegate"})
        events: list[dict[str, Any]] = []
        while not events or events[-1].get("event") != "agent_end":
            frame = await await_or_fail(
                observer.read(), what=f"nested foreground child completion; events={events!r}"
            )
            if "params" in frame:
                events.append(frame["params"])
        starts = [event for event in events if event.get("event") == "agent_start"]
        assert len(starts) == 2, "nested child did not forward agent_start"
        assert starts[1]["seq"] > starts[0]["seq"], (
            "nested forwarded agent_start must follow its parent start"
        )
    finally:
        await observer.close()
        await controller.close()
        await server.close()


@pytest.mark.asyncio
async def test_disconnect_during_foreground_child_cleanup_closes_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup_entered = asyncio.Event()
    cleanup_cancelled = asyncio.Event()
    release_cleanup = asyncio.Event()
    child_loop: AgentLoop | None = None
    build_child_loop = runner._build_child_loop
    cleanup = ChildApprovalPolicy.cleanup

    async def capture_child(*args: Any, **kwargs: Any) -> tuple[AgentLoop, ChildApprovalPolicy | None]:
        nonlocal child_loop
        child_loop, policy = await build_child_loop(*args, **kwargs)
        return child_loop, policy

    async def gate_cleanup(policy: ChildApprovalPolicy) -> None:
        cleanup_entered.set()
        try:
            await asyncio.wait_for(release_cleanup.wait(), TIMEOUT)
        except asyncio.CancelledError:
            cleanup_cancelled.set()
            raise
        await cleanup(policy)

    # test-contract: seam capture the owned loop and pause only its approval cleanup.
    monkeypatch.setattr(runner, "_build_child_loop", capture_child)
    monkeypatch.setattr(ChildApprovalPolicy, "cleanup", gate_cleanup)
    call = ToolCall(
        "child", "agent",
        {"prompt": "child-work", "description": "child", "background": False},
    )
    backend = PerActorFakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("parent done")])],
        {"child-work": [ScriptedTurn([TextContent("child done")])]},
        parent_prompt="delegate",
        gated_child_prompts=(),
        parent_waits_for_children=False,
    )
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    try:
        reader, writer = await _connect(server)
        try:
            await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
            await _request(reader, writer, 2, "new_session", {"provider": "codex"})
            await _request(reader, writer, 3, "send", {"text": "delegate"})
            await asyncio.wait_for(cleanup_entered.wait(), TIMEOUT)
            writer.close()
            await asyncio.wait_for(cleanup_cancelled.wait(), TIMEOUT)
        finally:
            release_cleanup.set()
            writer.close()
            await writer.wait_closed()
            await server.close()
        assert child_loop is not None
        # test-contract: seam the owners expose no public closed-state check.
        assert child_loop._closed, "foreground child loop left open after disconnect"
        assert child_loop.tool_registry.background_tasks._closed, (
            "foreground child task registry left open after disconnect"
        )
    finally:
        await server.close()
        if child_loop is not None and not child_loop._closed:
            await child_loop.close(cancel_background=False)
