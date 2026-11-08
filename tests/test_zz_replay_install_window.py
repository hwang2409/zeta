from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import connect_replay, negotiate, replay_server
from tests.test_server_replay_table import _backend_factory, _frames_before_status

pytest_plugins = ("tests.support.socket_fixtures",)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_join_in_install_window_sees_effect_once(tmp_path: Path, transport: Any) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    release = threading.Event()
    try:
        await negotiate(controller, extra=("delivery_id", "user_message_event"))
        session_id = (await controller.request(2, "new_session", {"provider": "codex"}))["result"]["session"]["session_id"]
        await negotiate(observer, observer=True)
        store = server.runtime.opened.store
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        original = store._release_detached_log_state

        def gated(state: Any) -> None:
            loop.call_soon_threadsafe(entered.set)
            release.wait(10)
            original(state)

        # test-contract: seam hold the window between the row's install and the caller's resume
        store._release_detached_log_state = gated
        await controller.send(3, "send", {"text": "marker-text", "delivery_id": "d-1"})
        await await_or_fail(entered.wait(), what="delivery row installed")
        store._release_detached_log_state = original
        joined = await observer.request(4, "session_history_and_subscribe", {"session_id": session_id})
        history = [
            block.get("text")
            for row in joined["result"]["messages"]
            for block in row.get("content", [])
            if isinstance(block, dict)
        ]
        release.set()
        frames = await _frames_before_status(observer, 5)
        live = [f["params"].get("text") for f in frames if f.get("params", {}).get("event") == "user_message"]
        print("history:", history, "live:", live)
        assert (history + live).count("marker-text") == 1, f"history={history} live={live}"
    finally:
        release.set()
        await observer.close()
        await controller.close()
        await server.close()


def _assistant_backend(provider: str, model: str | None, home: Path) -> tuple[Any, str]:
    from tests.support.fake_backend import FakeBackend, ScriptedTurn
    from zeta.protocol.types import TextContent

    del provider, home
    return FakeBackend([ScriptedTurn(content=[TextContent("assistant-marker")])]), model or "offline"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_join_between_assistant_event_and_its_row_sees_it_once(tmp_path: Path, transport: Any) -> None:
    server = replay_server(tmp_path, transport, _assistant_backend)
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    release, entered, appended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        await negotiate(controller)
        session_id = (await controller.request(2, "new_session", {"provider": "codex"}))["result"]["session"]["session_id"]
        await negotiate(observer, observer=True)
        loop_owner = server.runtime.loop
        original = loop_owner._append_turn_message_with_approvals

        async def gated(message: Any, approvals: Any) -> Any:
            if getattr(message.role, "value", message.role) == "assistant":
                entered.set()
                await await_or_fail(release.wait(), what="release assistant row append")
                result = await original(message, approvals)
                appended.set()
                return result
            return await original(message, approvals)

        # test-contract: seam hold the assistant row append after its MESSAGE_END event was published
        loop_owner._append_turn_message_with_approvals = gated
        await controller.send(3, "send", {"text": "hello"})
        await await_or_fail(entered.wait(), what="assistant event published, row not installed")
        joined = await observer.request(4, "session_history_and_subscribe", {"session_id": session_id})
        history = [
            block.get("text")
            for row in joined["result"]["messages"]
            for block in row.get("content", [])
            if isinstance(block, dict)
        ]
        release.set()
        await await_or_fail(appended.wait(), what="assistant row installed")
        frames = await _frames_before_status(observer, 5)
        live = [
            block.get("text")
            for f in frames
            if f.get("params", {}).get("event") == "assistant_message"
            for block in f["params"]["message"].get("content", [])
        ]
        print("history:", history, "live:", live)
        assert (history + live).count("assistant-marker") == 1, f"history={history} live={live}"
    finally:
        release.set()
        await observer.close()
        await controller.close()
        await server.close()
