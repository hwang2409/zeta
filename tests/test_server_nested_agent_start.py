from __future__ import annotations

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
from zeta.protocol.types import TextContent, ToolCall

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
