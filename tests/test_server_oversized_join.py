from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.replay_transport import connect_replay, negotiate, replay_server
from zeta.server.protocol import MAX_FRAME_BYTES

pytest_plugins = ("tests.support.socket_fixtures",)


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_oversized_join_error_activates_no_stream(
    tmp_path: Path, transport: str
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    controller = await connect_replay(server, transport)
    observer = await connect_replay(server, transport)
    original = server.connections.finish_replay_join

    def oversized(*args: object) -> dict[str, object]:
        result = original(*args)
        return {**result, "padding": "x" * MAX_FRAME_BYTES}

    try:
        await negotiate(controller)
        await negotiate(observer, observer=True)
        created = await controller.request(2, "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        server.connections.finish_replay_join = oversized  # type: ignore[method-assign]
        failed = await observer.request(
            3, "session_history_and_subscribe", {"session_id": session_id}
        )
        assert failed["error"]["code"] == -32007
        server.connections.finish_replay_join = original  # type: ignore[method-assign]

        await controller.request(4, "send", {"text": "after failed join"})
        await observer.request(5, "status", {})
        assert not any(
            frame.get("params", {}).get("event") == "user_message"
            for frame in observer.buffered
        ), "the oversized join error activated a stream"
    finally:
        server.connections.finish_replay_join = original  # type: ignore[method-assign]
        await observer.close()
        await controller.close()
        await server.close()
