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
async def test_unattached_multi_close_releases_mode_for_legacy(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    second = await connect_replay(server, transport)
    try:
        admitted = await negotiate(first, replay=False, extra=("multi_controller",))
        assert "result" in admitted
        refused = await negotiate(second, replay=False)
        assert refused["error"]["code"] == -32001
        await first.close()
        recovered_client = await connect_replay(server, transport)
        try:
            recovered = await negotiate(recovered_client, replay=False)
            assert "result" in recovered
        finally:
            await recovered_client.close()
    finally:
        await _close_server(server, (first, second))


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_unattached_multi_close_releases_mode_for_fresh_multi(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    second = await connect_replay(server, transport)
    try:
        admitted = await negotiate(first, replay=False, extra=("multi_controller",))
        assert "result" in admitted
        await first.close()
        recovered = await negotiate(second, replay=False, extra=("multi_controller",))
        assert "result" in recovered
    finally:
        await _close_server(server, (first, second))


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_simultaneous_first_legacy_and_multi_have_one_serial_admission(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    clients = [await connect_replay(server, transport) for _ in range(2)]
    admissions: list[tuple[str, str]] = []
    events: list[str] = []
    original = server.controllers.admit_hello

    def admit(connection, *, negotiated_multi, reconnect_session_id=None):
        mode = "multi" if negotiated_multi else "legacy"
        admissions.append((mode, str(id(connection))) )
        events.append("admit")
        return original(
            connection,
            negotiated_multi=negotiated_multi,
            reconnect_session_id=reconnect_session_id,
        )

    monkeypatch.setattr(server.controllers, "admit_hello", admit)
    original_negotiate_role = server.connections.negotiate_role

    def negotiate_role(*args, **kwargs):
        events.append("physical")
        return original_negotiate_role(*args, **kwargs)

    monkeypatch.setattr(server.connections, "negotiate_role", negotiate_role)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                negotiate(clients[0], replay=False),
                negotiate(clients[1], replay=False, extra=("multi_controller",)),
            ),
            10,
        )
        assert events[0] == "admit", (
            "first semantic admission must be the sole mode winner before physical hello response"
        )
        assert len(admissions) == 2
        assert {mode for mode, _ in admissions} == {"legacy", "multi"}
        assert sum("result" in result for result in results) == 1
        assert sum(result.get("error", {}).get("code") == -32001 for result in results) == 1
        winner = "legacy" if "result" in results[0] else "multi"
        assert server.controllers.controller_mode == winner
    finally:
        await _close_server(server, clients)


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_failed_mcp_bindings_abort_hello_admission(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    failed = await connect_replay(server, transport)
    legacy = await connect_replay(server, transport)
    try:
        result = await failed.request(
            1,
            "hello",
            {
                "protocol_version": "1.1",
                "features": ["multi_controller"],
                "mcp_bindings": {"invalid": True},
            },
        )
        assert result["error"]["code"] == -32602
        assert not server.controllers._controller_members
        admitted = await negotiate(legacy, replay=False)
        assert "result" in admitted
    finally:
        await _close_server(server, (failed, legacy))


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_post_claim_hello_failure_releases_semantic_claim(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    failed = await connect_replay(server, transport)
    legacy = await connect_replay(server, transport)
    control = server.controllers.attached().control
    original = control.controller_claimed

    async def fail_after_claim(*, reconnect, require_host_binding, deadline):
        result = await original(
            reconnect=reconnect,
            require_host_binding=require_host_binding,
            deadline=deadline,
        )
        assert result
        raise RuntimeError("ready gate failed")

    monkeypatch.setattr(control, "controller_claimed", fail_after_claim)
    try:
        result = await negotiate(failed, replay=False, extra=("multi_controller",))
        assert result["error"]["code"] == -32000
        assert not control._controller_attached
        assert not server.controllers._controller_members
        monkeypatch.setattr(control, "controller_claimed", original)
        admitted = await negotiate(legacy, replay=False)
        assert "result" in admitted
    finally:
        await _close_server(server, (failed, legacy))


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
