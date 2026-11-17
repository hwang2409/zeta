from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
    short_socket_path,
)
from tests.test_server_replay_table import _backend_factory
from zeta.server.server import ZetaServer


async def _close_server(server, clients) -> None:
    for client in clients:
        await client.close()
    await server.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_legacy_grace_rejects_multi_and_allows_legacy_reconnect(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    await negotiate(first, replay=False)
    await first.close()
    multi = await connect_replay(server, transport)
    legacy = await connect_replay(server, transport)
    try:
        refused = await negotiate(multi, replay=False, extra=("multi_controller",))
        assert refused["error"] == {
            "code": -32001,
            "message": "another client is already connected",
        }
        admitted = await negotiate(legacy, replay=False)
        assert "result" in admitted
    finally:
        await _close_server(server, (multi, legacy))


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_unattached_multi_member_blocks_legacy(tmp_path: Path, transport: Transport) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    second = await connect_replay(server, transport)
    try:
        await negotiate(first, replay=False, extra=("multi_controller",))
        refused = await negotiate(second, replay=False)
        assert refused["error"]["code"] == -32001
    finally:
        await _close_server(server, (first, second))


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_simultaneous_first_multi_hellos_have_one_serial_admission(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    clients = [await connect_replay(server, transport) for _ in range(2)]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    negotiate(client, replay=False, extra=("multi_controller",))
                    for client in clients
                )
            ),
            10,
        )
        assert all("result" in result for result in results)
        assert server.controllers.controller_mode == "multi"
    finally:
        await _close_server(server, clients)


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_multi_reconnect_requires_matching_session_id(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    await negotiate(first, replay=False, extra=("multi_controller",))
    created = await first.request(2, "new_session", {})
    session_id = created["result"]["session"]["session_id"]
    await first.close()
    wrong = await connect_replay(server, transport)
    right = await connect_replay(server, transport)
    try:
        refused = await wrong.request(
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["multi_controller"], "session_id": "wrong"},
        )
        assert refused["error"]["code"] == -32001
        admitted = await right.request(
            1,
            "hello",
            {
                "protocol_version": "1.1",
                "features": ["multi_controller"],
                "session_id": session_id,
            },
        )
        assert "result" in admitted
    finally:
        await _close_server(server, (wrong, right))


def test_production_sources_do_not_supply_private_feature_override() -> None:
    root = Path(__file__).parents[1] / "src" / "zeta"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "ZetaServer":
                continue
            assert all(
                keyword.arg != "_supported_features_for_test" for keyword in node.keywords
            ), path


@pytest.mark.asyncio
async def test_ordinary_server_keeps_multi_controller_dormant(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        provider="codex",
        backend_factory=_backend_factory,
    )
    await server.start()
    first = await connect_replay(server, "unix")
    second = await connect_replay(server, "unix")
    try:
        hello = await negotiate(first, replay=False, extra=("multi_controller",))
        assert "multi_controller" not in hello["result"]["capabilities"].get("features", [])
        refused = await negotiate(second, replay=False, extra=("multi_controller",))
        assert refused["error"] == {
            "code": -32001,
            "message": "another client is already connected",
        }
    finally:
        await _close_server(server, (first, second))
