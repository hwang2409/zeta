from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_replay_table import _backend_factory


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_ordinary_server_negotiates_multi_controller(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    second = await connect_replay(server, transport)
    try:
        hello = await negotiate(first, replay=False, extra=("multi_controller",))
        assert "multi_controller" in hello["result"]["capabilities"]["features"]

        admitted = await negotiate(second, replay=False, extra=("multi_controller",))
        assert "multi_controller" in admitted["result"]["capabilities"]["features"]
    finally:
        await first.close()
        await second.close()
        await server.close()
