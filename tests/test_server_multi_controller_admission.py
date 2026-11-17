from __future__ import annotations

import ast
import asyncio
import contextlib
from pathlib import Path
from typing import Any

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

_CANCEL_ROWS = (
    "reconnect_mcp_bindings_before", "reconnect_mcp_bindings_after",
    "controller_claimed_before", "controller_claimed_after",
    "parking_claim_before", "parking_claim_after",
    "reattach_mcp_bindings_before", "reattach_mcp_bindings_after",
    "controller_ready_before", "controller_ready_after",
    "commit_hello_before", "commit_hello_after",
    "response_publication", "release_in_progress",
)


async def _cancel_at(original: Any, args: tuple[Any, ...], kwargs: dict[str, Any], *, after: bool) -> Any:
    if not after:
        raise asyncio.CancelledError
    await original(*args, **kwargs)
    raise asyncio.CancelledError


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", _CANCEL_ROWS)
@pytest.mark.asyncio
async def test_cancelled_hello_transaction_public_recovery(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cancel each real transaction boundary and recover through a new socket."""
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    failed = await connect_replay(server, transport)
    control = server.controllers.attached().control
    release_started = asyncio.Event()
    release_gate = asyncio.Event()
    original_release = control.begin_controller_release

    if row == "release_in_progress":
        def begin_release(*args: Any, **kwargs: Any) -> asyncio.Task[None]:
            task = original_release(*args, **kwargs)
            async def held() -> None:
                release_started.set()
                await asyncio.wait_for(release_gate.wait(), 10)
                await asyncio.wait_for(asyncio.shield(task), 10)
            return asyncio.create_task(held())
        monkeypatch.setattr(control, "begin_controller_release", begin_release)

    if row.startswith(("reconnect_mcp_bindings", "reattach_mcp_bindings")):
        from zeta.server.handlers import lifecycle
        async def fake_bindings(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return {}, {}
        monkeypatch.setattr(lifecycle, "reconnect_mcp_bindings", fake_bindings)
        if row.startswith("reconnect"):
            original = fake_bindings
            after = row.endswith("after")
            async def gated(*args: Any, **kwargs: Any) -> Any:
                return await _cancel_at(original, args, kwargs, after=after)
            monkeypatch.setattr(lifecycle, "reconnect_mcp_bindings", gated)
        else:
            original = control.reattach_mcp_bindings
            after = row.endswith("after")
            async def gated(*args: Any, **kwargs: Any) -> Any:
                return await _cancel_at(original, args, kwargs, after=after)
            monkeypatch.setattr(control, "reattach_mcp_bindings", gated)
    elif row.startswith("controller_claimed"):
        original = control.controller_claimed
        after = row.endswith("after")
        async def gated(*args: Any, **kwargs: Any) -> Any:
            return await _cancel_at(original, args, kwargs, after=after)
        monkeypatch.setattr(control, "controller_claimed", gated)
    elif row.startswith("parking_claim"):
        original = server.parking.claim
        after = row.endswith("after")
        async def gated(*args: Any, **kwargs: Any) -> Any:
            return await _cancel_at(original, args, kwargs, after=after)
        monkeypatch.setattr(server.parking, "claim", gated)
    elif row.startswith("controller_ready"):
        original = control.controller_ready
        after = row.endswith("after")
        async def gated(*args: Any, **kwargs: Any) -> Any:
            return await _cancel_at(original, args, kwargs, after=after)
        monkeypatch.setattr(control, "controller_ready", gated)
    elif row.startswith("commit_hello") or row == "release_in_progress":
        original = server.controllers.commit_hello
        after = True if row == "release_in_progress" else row.endswith("after")
        def gated(*args: Any, **kwargs: Any) -> None:
            if not after:
                raise asyncio.CancelledError
            original(*args, **kwargs)
            raise asyncio.CancelledError
        monkeypatch.setattr(server.controllers, "commit_hello", gated)
    elif row == "response_publication":
        from zeta.server.server import _Client
        original = _Client._write
        async def gated(client: Any, *args: Any, **kwargs: Any) -> Any:
            if client._request is not None and client._request[0] == "hello":
                return await _cancel_at(original, (client, *args), kwargs, after=False)
            return await original(client, *args, **kwargs)
        # test-contract: seam response-publication cancellation uses the existing writer.
        monkeypatch.setattr(_Client, "_write", gated)

    params: dict[str, Any] = {"protocol_version": "1.1", "features": ["multi_controller"]}
    if row.startswith(("reconnect_mcp_bindings", "reattach_mcp_bindings")):
        params.update(features=["multi_controller", "approval_reconnect"], mcp_bindings=[], session_id="cancelled-session")
    task = asyncio.create_task(failed.request(1, "hello", params))
    try:
        if row == "release_in_progress":
            await asyncio.wait_for(release_started.wait(), 10)
            release_gate.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 10)
        await asyncio.wait_for(failed.close(), 10)
        monkeypatch.undo()
        result = None
        for _ in range(20):
            recovered = await connect_replay(server, transport)
            try:
                candidate = await asyncio.wait_for(negotiate(recovered, replay=False), 10)
                if "result" in candidate:
                    result = candidate
                    break
            except (AssertionError, ConnectionError, TimeoutError, asyncio.CancelledError):
                continue
            finally:
                await recovered.close()
        assert result is not None, (
            f"cancelled hello did not release its public slot: "
            f"mode={server.controllers.controller_mode} members={len(server.controllers._controller_members)} "
            f"holder={server.connections.controller_metadata()}"
        )
    finally:
        release_gate.set()
        with contextlib.suppress(BaseException):
            await failed.close()
        await server.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_cancelled_old_hello_does_not_remove_newer_multi_token(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    failed = await connect_replay(server, transport)
    control = server.controllers.attached().control
    original = control.controller_claimed

    async def cancelled(*args: Any, **kwargs: Any) -> Any:
        await original(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(control, "controller_claimed", cancelled)
    task = asyncio.create_task(
        failed.request(1, "hello", {"protocol_version": "1.1", "features": ["multi_controller"]})
    )
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(task, 10)
    await asyncio.wait_for(failed.close(), 10)
    monkeypatch.undo()
    newer = await connect_replay(server, transport)
    try:
        result = await negotiate(newer, replay=False, extra=("multi_controller",))
        assert "result" in result
    finally:
        await newer.close()
        await server.close()
