from __future__ import annotations

import ast
import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
    short_socket_path,
)
from tests.test_server_replay_table import _backend_factory
from zeta.server.server import ZetaServer, _Client


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
    "reattach_mcp_bindings_before", "reattach_mcp_bindings_after",
    "controller_ready_before", "controller_ready_after",
    "commit_hello_before", "commit_hello_after",
    "response_publication", "release_in_progress",
)


async def _bound_reconnect_params(server, transport: Transport) -> tuple[dict[str, Any], Any]:
    seed = await connect_replay(server, transport)
    hello = await negotiate(
        seed,
        replay=False,
        extra=("multi_controller", "approval_reconnect", "session_mcp_bindings"),
    )
    assert "result" in hello
    created = await seed.request(2, "new_session", {})
    session_id = created["result"]["session"]["session_id"]
    record = next(
        record
        for record in server.controllers._records.values()
        if record.runtime.state is not None and record.runtime.session_id == session_id
    )
    control = record.control
    control._mcp_binding_required = True
    control._mcp_binding_servers = frozenset()
    control._active_mcp_bindings = {}
    await seed.close()
    return {
        "protocol_version": "1.1",
        "features": [
            "multi_controller",
            "approval_reconnect",
            "session_mcp_bindings",
        ],
        "mcp_bindings": [],
        "session_id": session_id,
    }, control


def _gate_async(monkeypatch, owner, name: str, entered: asyncio.Event, release: asyncio.Event, *, after: bool) -> None:
    original = getattr(owner, name)

    async def gated(*args: Any, **kwargs: Any) -> Any:
        if not after:
            entered.set()
            await await_or_fail(release.wait(), what=f"release {name} before")
        result = await original(*args, **kwargs)
        if after:
            entered.set()
            await await_or_fail(release.wait(), what=f"release {name} after")
        return result

    monkeypatch.setattr(owner, name, gated)


def _capture_real_hello_handler(
    monkeypatch: pytest.MonkeyPatch,
    entered: asyncio.Event,
    release: asyncio.Event,
    row: str,
) -> dict[str, Any]:
    original = _Client._route_hello
    captured: dict[str, Any] = {}

    async def routed(client: _Client, params: dict[str, Any]) -> tuple[bool, object]:
        if "task" not in captured:
            captured["client"] = client
            captured["task"] = asyncio.current_task()
        result = await original(client, params)
        if row == "commit_hello_after" and client is captured["client"]:
            entered.set()
            await await_or_fail(release.wait(), what="release after hello commit")
        return result

    # test-contract: seam capture the real server handler without a product hook.
    monkeypatch.setattr(_Client, "_route_hello", routed)
    return captured


async def _assert_failed_socket_closed_without_success(client) -> None:
    await await_or_fail(client.reader.read(), what="cancelled hello socket closes")
    assert client.reader.at_eof(), "cancelled hello socket stayed open"
    frames = [*client.wire_received, *client.buffered]
    assert not any(frame.get("id") == 1 and "result" in frame for frame in frames), (
        "cancelled hello published a successful response"
    )


async def _assert_public_recovery(server, transport: Transport, params: dict[str, Any]) -> None:
    session_id = params.get("session_id")
    if isinstance(session_id, str):
        recovered = await connect_replay(server, transport)
        try:
            result = await recovered.request(
                1,
                "hello",
                {
                    "protocol_version": "1.1",
                    "features": ["multi_controller"],
                    "session_id": session_id,
                },
            )
            assert "result" in result, "exact-session multi reconnect did not recover"
        finally:
            await recovered.close()
        return

    fresh_multi = await connect_replay(server, transport)
    try:
        result = await negotiate(
            fresh_multi, replay=False, extra=("multi_controller",)
        )
        assert "result" in result, "cancelled hello left a semantic reservation"
    finally:
        await fresh_multi.close()
    legacy = await connect_replay(server, transport)
    try:
        result = await negotiate(legacy, replay=False)
        assert "result" in result, "fresh multi close did not release dormant mode"
    finally:
        await legacy.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", _CANCEL_ROWS)
@pytest.mark.asyncio
async def test_cancelled_hello_transaction_public_recovery(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """externally cancel the real server handler at each hello transaction edge."""
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    params: dict[str, Any] = {
        "protocol_version": "1.1",
        "features": ["multi_controller"],
    }
    control = server.controllers.attached().control
    if row.startswith(("reconnect_mcp_bindings", "reattach_mcp_bindings")):
        params, control = await _bound_reconnect_params(server, transport)

    failed = await connect_replay(server, transport)
    entered = asyncio.Event()
    release = asyncio.Event()
    release_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    captured = _capture_real_hello_handler(monkeypatch, entered, release, row)

    if row.startswith("reconnect_mcp_bindings"):
        from zeta.server.handlers import lifecycle

        _gate_async(
            monkeypatch,
            lifecycle,
            "reconnect_mcp_bindings",
            entered,
            release,
            after=row.endswith("after"),
        )
    elif row.startswith("controller_claimed"):
        _gate_async(
            monkeypatch,
            control,
            "controller_claimed",
            entered,
            release,
            after=row.endswith("after"),
        )
    elif row.startswith("parking_claim"):
        _gate_async(
            monkeypatch,
            server.parking,
            "claim",
            entered,
            release,
            after=row.endswith("after"),
        )
    elif row.startswith("reattach_mcp_bindings"):
        _gate_async(
            monkeypatch,
            control,
            "reattach_mcp_bindings",
            entered,
            release,
            after=row.endswith("after"),
        )
    elif row.startswith("controller_ready"):
        _gate_async(
            monkeypatch,
            control,
            "controller_ready",
            entered,
            release,
            after=row.endswith("after"),
        )
    elif row == "commit_hello_before":
        _gate_async(
            monkeypatch,
            _Client,
            "_setup_controller_hello",
            entered,
            release,
            after=True,
        )
    elif row == "response_publication":
        original_write = _Client._write

        async def gated_write(client: _Client, *args: Any, **kwargs: Any) -> Any:
            if (
                client is captured.get("client")
                and client._request is not None
                and client._request[:2] == ("hello", 1)
            ):
                entered.set()
                await await_or_fail(release.wait(), what="release hello response write")
            return await original_write(client, *args, **kwargs)

        # test-contract: seam hold the real response writer at publication.
        monkeypatch.setattr(_Client, "_write", gated_write)
    elif row == "release_in_progress":
        _gate_async(
            monkeypatch,
            _Client,
            "_setup_controller_hello",
            entered,
            release,
            after=True,
        )
        original_release = control._controller_release

        def held_release(*args: Any, **kwargs: Any):
            operation = original_release(*args, **kwargs)

            async def finish() -> None:
                release_started.set()
                await await_or_fail(
                    release_cleanup.wait(), what="release held cleanup"
                )
                await await_or_fail(operation, what="real release operation")

            return finish()

        # test-contract: seam hold the real release cleanup after its state mutation.
        monkeypatch.setattr(control, "_controller_release", held_release)

    request = asyncio.create_task(failed.request(1, "hello", params))
    try:
        await await_or_fail(entered.wait(), what=f"{row} entered")
        handler = captured["task"]
        assert handler is not request, "test captured the client task, not the server handler"
        assert handler is captured["client"]._run_task
        handler.cancel()
        if row == "release_in_progress":
            await await_or_fail(
                release_started.wait(), what="real controller release started"
            )
            handler.cancel()
            release_cleanup.set()
        else:
            release.set()
        with contextlib.suppress(BaseException):
            await await_or_fail(request, what=f"{row} client request ends")
        await _assert_failed_socket_closed_without_success(failed)
        release_task = control._controller_release_task
        if release_task is not None:
            await await_or_fail(
                asyncio.shield(release_task), what=f"{row} release cleanup completes"
            )
        monkeypatch.undo()
        assert not server.controllers._controller_members
        await _assert_public_recovery(server, transport, params)
    finally:
        release.set()
        release_cleanup.set()
        if not request.done():
            request.cancel()
            with contextlib.suppress(BaseException):
                await await_or_fail(request, what="cancelled client request cleanup")
        with contextlib.suppress(BaseException):
            await failed.close()
        await server.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_cancelled_old_cleanup_does_not_remove_concurrent_newer_multi_token(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    old = await connect_replay(server, transport)
    entered = asyncio.Event()
    release = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    captured = _capture_real_hello_handler(monkeypatch, entered, release, "overlap")
    _gate_async(
        monkeypatch,
        _Client,
        "_setup_controller_hello",
        entered,
        release,
        after=True,
    )
    old_server_client = captured
    original_close = _Client.close

    async def held_close(client: _Client, *args: Any, **kwargs: Any) -> None:
        if client is old_server_client.get("client"):
            cleanup_started.set()
            await await_or_fail(cleanup_release.wait(), what="old cleanup release")
        await original_close(client, *args, **kwargs)

    monkeypatch.setattr(_Client, "close", held_close)
    old_request = asyncio.create_task(
        old.request(
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["multi_controller"]},
        )
    )
    newer = legacy = recovered = None
    try:
        await await_or_fail(entered.wait(), what="old hello reached post-setup gate")
        old_handler = captured["task"]
        assert old_handler is captured["client"]._run_task
        old_handler.cancel()
        await await_or_fail(cleanup_started.wait(), what="old release cleanup held")
        monkeypatch.undo()

        newer = await connect_replay(server, transport)
        admitted = await negotiate(newer, replay=False, extra=("multi_controller",))
        assert "result" in admitted, "newer multi token was not admitted during old cleanup"
        cleanup_release.set()
        with contextlib.suppress(BaseException):
            await await_or_fail(old_request, what="old cancelled request ends")
        await _assert_failed_socket_closed_without_success(old)
        usable = await newer.request(2, "status", {})
        assert "result" in usable, "old cleanup displaced the newer controller"
        legacy = await connect_replay(server, transport)
        refused = await negotiate(legacy, replay=False)
        assert refused["error"]["code"] == -32001, (
            "old cleanup cleared mode while the newer token still owned it"
        )
        await newer.close()
        recovered = await connect_replay(server, transport)
        admitted_legacy = await negotiate(recovered, replay=False)
        assert "result" in admitted_legacy
    finally:
        release.set()
        cleanup_release.set()
        if not old_request.done():
            old_request.cancel()
            with contextlib.suppress(BaseException):
                await await_or_fail(old_request, what="cancel old overlap request")
        await _close_server(
            server,
            tuple(client for client in (old, newer, legacy, recovered) if client is not None),
        )

@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_four_multi_controllers_keep_distinct_session_ownership(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    clients = [await connect_replay(server, transport) for _ in range(4)]
    try:
        for client in clients:
            hello = await negotiate(client, replay=False, extra=("multi_controller",))
            assert "result" in hello
        session_ids = []
        for request_id, client in enumerate(clients, 2):
            created = await client.request(request_id, "new_session", {})
            session_ids.append(created["result"]["session"]["session_id"])
        assert len(set(session_ids)) == 4
        for request_id, (client, session_id) in enumerate(zip(clients, session_ids), 10):
            status = await client.request(request_id, "status", {})
            assert status["result"]["session"]["session_id"] == session_id
    finally:
        await _close_server(server, clients)


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_same_session_refusal_is_structured_and_keeps_socket_open(
    tmp_path: Path, transport: Transport
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    owner = await connect_replay(server, transport)
    contender = await connect_replay(server, transport)
    try:
        await negotiate(owner, replay=False, extra=("multi_controller",))
        created = await owner.request(2, "new_session", {})
        session_id = created["result"]["session"]["session_id"]
        await negotiate(contender, replay=False, extra=("multi_controller", "ping"))
        refused = await contender.request(2, "resume", {"session_id": session_id})
        assert refused["error"] == {
            "code": -32014,
            "message": "session is controlled by another client",
            "data": {"code": "session_controlled", "session_id": session_id},
        }
        ping = await contender.request(3, "ping", {})
        assert "result" in ping, "session conflict closed the controller socket"
        replacement = await contender.request(4, "new_session", {})
        assert replacement["result"]["session"]["session_id"] != session_id
    finally:
        await _close_server(server, (owner, contender))

async def _make_persisted_target(server, transport: Transport) -> str:
    creator = await connect_replay(server, transport)
    await negotiate(creator, replay=False, extra=("multi_controller",))
    created = await creator.request(2, "new_session", {})
    session_id = created["result"]["session"]["session_id"]
    await creator.close()
    record = next(
        record
        for record in server.controllers._records.values()
        if record.runtime.state is not None and record.runtime.session_id == session_id
    )
    await await_or_fail(
        asyncio.shield(server.controllers.retire(record)),
        what="persist target before acquisition",
    )
    return session_id


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_failed_target_acquisition_keeps_old_session_and_releases_target(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.server.runtime import ServerRuntime

    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    owner = await connect_replay(server, transport)
    recovery = None
    try:
        await negotiate(owner, replay=False, extra=("multi_controller",))
        old = await owner.request(2, "new_session", {})
        old_id = old["result"]["session"]["session_id"]
        target_id = await _make_persisted_target(server, transport)
        original_resume = ServerRuntime.resume_session

        async def fail_target(runtime, session_id: str, *args: Any, **kwargs: Any):
            if session_id == target_id:
                raise RuntimeError("forced target setup failure")
            return await original_resume(runtime, session_id, *args, **kwargs)

        monkeypatch.setattr(ServerRuntime, "resume_session", fail_target)
        failed = await owner.request(3, "resume", {"session_id": target_id})
        assert failed["error"]["code"] == -32000
        still_old = await owner.request(4, "status", {})
        assert still_old["result"]["session"]["session_id"] == old_id
        monkeypatch.undo()

        recovery = await connect_replay(server, transport)
        await negotiate(recovery, replay=False, extra=("multi_controller",))
        resumed = await recovery.request(2, "resume", {"session_id": target_id})
        assert resumed["result"]["session"]["session_id"] == target_id
    finally:
        monkeypatch.undo()
        await _close_server(
            server,
            tuple(client for client in (owner, recovery) if client is not None),
        )


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_target_is_reserved_before_one_awaited_open(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.server.runtime import ServerRuntime

    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    first = await connect_replay(server, transport)
    second = await connect_replay(server, transport)
    entered = asyncio.Event()
    release = asyncio.Event()
    first_request = None
    try:
        await negotiate(first, replay=False, extra=("multi_controller",))
        await first.request(2, "new_session", {})
        target_id = await _make_persisted_target(server, transport)
        await negotiate(second, replay=False, extra=("multi_controller", "ping"))
        await second.request(2, "new_session", {})
        original_resume = ServerRuntime.resume_session
        opens = 0

        async def gated_resume(runtime, session_id: str, *args: Any, **kwargs: Any):
            nonlocal opens
            if session_id == target_id:
                opens += 1
                if opens == 1:
                    entered.set()
                    await await_or_fail(release.wait(), what="release target open")
            return await original_resume(runtime, session_id, *args, **kwargs)

        monkeypatch.setattr(ServerRuntime, "resume_session", gated_resume)
        first_request = asyncio.create_task(
            first.request(3, "resume", {"session_id": target_id})
        )
        await await_or_fail(entered.wait(), what="target open entered")
        refused = await second.request(3, "resume", {"session_id": target_id})
        assert opens == 1, "a conflicting controller opened the reserved target"
        assert refused["error"] == {
            "code": -32014,
            "message": "session is controlled by another client",
            "data": {"code": "session_controlled", "session_id": target_id},
        }
        ping = await second.request(4, "ping", {})
        assert "result" in ping
        release.set()
        acquired = await await_or_fail(first_request, what="first target acquisition")
        assert acquired["result"]["session"]["session_id"] == target_id
        assert opens == 1
    finally:
        release.set()
        if first_request is not None and not first_request.done():
            first_request.cancel()
            with contextlib.suppress(BaseException):
                await await_or_fail(first_request, what="cancel target acquisition")
        monkeypatch.undo()
        await _close_server(server, (first, second))


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize(
    "row",
    (
        "live_owner_refusal",
        "disconnected_resume",
        "parked_resume",
        "resident_setup_failure",
        "resident_setup_cancel",
        "unrelated_parked_hello",
    ),
)
@pytest.mark.asyncio
async def test_public_multi_controller_owner_state_table(
    tmp_path: Path,
    transport: Transport,
    row: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    owner = await connect_replay(server, transport)
    survivor = await connect_replay(server, transport)
    newcomer = None
    features = ("multi_controller", "session_park", "resume_attach")
    try:
        assert "result" in await negotiate(owner, replay=False, extra=features)
        assert "result" in await negotiate(survivor, replay=False, extra=features)
        target = await owner.request(2, "new_session", {})
        target_id = target["result"]["session"]["session_id"]
        old = await survivor.request(2, "new_session", {})
        old_id = old["result"]["session"]["session_id"]
        if row == "live_owner_refusal":
            refused = await survivor.request(3, "resume", {"session_id": target_id})
            owner_status = await owner.request(3, "status", {})
            survivor_status = await survivor.request(4, "status", {})
            assert refused["error"] == {
                "code": -32014,
                "message": "session is controlled by another client",
                "data": {"code": "session_controlled", "session_id": target_id},
            }
            assert owner_status["result"]["session"]["session_id"] == target_id
            assert survivor_status["result"]["session"]["session_id"] == old_id
            return
        await owner.close()

        if row == "disconnected_resume":
            resumed = await survivor.request(3, "resume", {"session_id": target_id})
            status = await survivor.request(4, "status", {})
            assert "result" in resumed, resumed
            assert resumed["result"]["session"]["session_id"] == target_id
            assert status["result"]["session"]["session_id"] == target_id
            assert target_id != old_id
        else:
            record = next(
                record
                for record in server.controllers.records()
                if record.runtime.opened is not None
                and record.runtime.session_id == target_id
            )
            server.controllers.park(record)
            if row == "resident_setup_failure":
                async def fail_publish() -> None:
                    raise RuntimeError("injected grant publication failure")

                monkeypatch.setattr(
                    record.control, "publish_session_grant_reset", fail_publish
                )
                failed = await survivor.request(3, "resume", {"session_id": target_id})
                status = await survivor.request(4, "status", {})
                assert failed["error"]["code"] == -32000
                assert status["result"]["session"]["session_id"] == old_id
                monkeypatch.undo()
                retried = await survivor.request(5, "resume", {"session_id": target_id})
                assert retried["result"]["session"]["session_id"] == target_id
                return
            if row == "resident_setup_cancel":
                entered = asyncio.Event()
                release = asyncio.Event()
                original_publish = record.control.publish_session_grant_reset

                async def hold_publish() -> None:
                    entered.set()
                    await await_or_fail(release.wait(), what="release grant publication")
                    await original_publish()

                monkeypatch.setattr(
                    record.control, "publish_session_grant_reset", hold_publish
                )
                await survivor.send(3, "resume", {"session_id": target_id})
                await await_or_fail(entered.wait(), what="grant publication entered")
                physical = server.connections.clients()[0]
                handler = physical._run_task
                assert handler is not None
                settled = asyncio.Event()
                handler.add_done_callback(lambda _task: settled.set())
                handler.cancel()
                release.set()
                await await_or_fail(settled.wait(), what="cancelled resume handler")
                await survivor.close()
                newcomer = await connect_replay(server, transport)
                recovered = await newcomer.request(
                    1,
                    "hello",
                    {
                        "protocol_version": "1.1",
                        "features": list(features),
                        "session_id": old_id,
                    },
                )
                assert "result" in recovered, recovered
                status = await newcomer.request(2, "status", {})
                assert status["result"]["session"]["session_id"] == old_id
                monkeypatch.undo()
                retried = await newcomer.request(
                    3, "resume", {"session_id": target_id}
                )
                assert retried["result"]["session"]["session_id"] == target_id
                return
            if row == "parked_resume":
                resumed = await survivor.request(3, "resume", {"session_id": target_id})
                status = await survivor.request(4, "status", {})
                assert resumed["result"]["session"]["session_id"] == target_id
                assert status["result"]["session"]["session_id"] == target_id
                assert target_id != old_id
                return
            newcomer = await connect_replay(server, transport)
            hello = await negotiate(newcomer, replay=False, extra=("multi_controller",))
            assert "result" in hello
            status = await survivor.request(3, "status", {})
            assert any(
                parked["session_id"] == target_id
                for parked in status["result"]["parked_sessions"]
            )
            assert not record.closing
    finally:
        await _close_server(
            server,
            tuple(client for client in (owner, survivor, newcomer) if client is not None),
        )


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize(
    "boundary",
    ("semantic", "flush", "workers", "cancel", "client", "controllers", "two_failures"),
)
@pytest.mark.asyncio
async def test_public_multi_controller_shutdown_state_table(
    tmp_path: Path, transport: Transport, boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.server.runtime import ServerRuntime

    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    clients = [await connect_replay(server, transport) for _ in range(5)]
    provisional_request = None
    release = asyncio.Event()
    try:
        features = ("multi_controller", "session_park", "resume_attach")
        for client in clients:
            assert "result" in await negotiate(client, replay=False, extra=features)
        for request_id, client in enumerate(clients[:4], 2):
            assert "result" in await client.request(request_id, "new_session", {})

        entered = asyncio.Event()
        original_create = ServerRuntime.create_session

        async def gated_create(runtime: ServerRuntime, *args: Any, **kwargs: Any):
            entered.set()
            await await_or_fail(release.wait(), what="release provisional create")
            return await original_create(runtime, *args, **kwargs)

        monkeypatch.setattr(ServerRuntime, "create_session", gated_create)
        provisional_request = asyncio.create_task(clients[4].request(9, "new_session", {}))
        await await_or_fail(entered.wait(), what="provisional create entered")
        physical_clients = server.connections.clients()
        records = server.controllers.records()
        first_client, second_client = physical_clients[:2]
        first_error = RuntimeError("first shutdown failure")
        second_error = RuntimeError("second shutdown failure")
        if boundary in {"semantic", "two_failures"}:
            def fail_release():
                raise first_error
            # test-contract: seam patches the exact client's semantic release.
            monkeypatch.setattr(first_client, "_begin_controller_release", fail_release)
        if boundary in {"flush", "two_failures"}:
            original_flush = server.connections.flush
            async def fail_flush(connection, *args, **kwargs):
                if connection is (second_client if boundary == "two_failures" else first_client):
                    raise second_error if boundary == "two_failures" else first_error
                return await original_flush(connection, *args, **kwargs)
            monkeypatch.setattr(server.connections, "flush", fail_flush)
        shutdown_gate = asyncio.Event()
        shutdown_entered = asyncio.Event()
        if boundary == "workers":
            original_workers = server.connections.stop_workers
            async def fail_workers():
                await original_workers()
                raise first_error
            monkeypatch.setattr(server.connections, "stop_workers", fail_workers)
        if boundary == "cancel":
            original_workers = server.connections.stop_workers
            async def gated_workers():
                shutdown_entered.set()
                await await_or_fail(shutdown_gate.wait(), what="shutdown worker gate")
                return await original_workers()
            monkeypatch.setattr(server.connections, "stop_workers", gated_workers)
        if boundary == "client":
            original_writer_close = first_client.writer.close
            def fail_writer_close():
                original_writer_close()
                raise first_error
            monkeypatch.setattr(first_client.writer, "close", fail_writer_close)
        if boundary == "controllers":
            original_controllers_close = server.controllers.close
            async def fail_controllers_close(*args, **kwargs):
                await original_controllers_close(*args, **kwargs)
                raise first_error
            monkeypatch.setattr(server.controllers, "close", fail_controllers_close)
        closes: dict[int, int] = {id(record.runtime): 0 for record in records}
        for record in records:
            original_close = record.runtime.close

            async def counted_close(
                *, _runtime=record.runtime, _close=original_close
            ) -> None:
                closes[id(_runtime)] += 1
                await _close()

            monkeypatch.setattr(record.runtime, "close", counted_close)

        if boundary == "cancel":
            close_task = asyncio.create_task(server.close())
            await await_or_fail(shutdown_entered.wait(), what="shutdown remains gated")
            close_task.cancel()
            shutdown_gate.set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(close_task, what="cancelled shutdown")
            await await_or_fail(server.close(), what="deferred shutdown")
        else:
            expected = pytest.raises((RuntimeError, asyncio.CancelledError)) if boundary != "cancel" else contextlib.nullcontext()
            with expected:
                await await_or_fail(server.close(), what="four-controller server shutdown")
        assert provisional_request.done()
        assert all(client._run_task is None or client._run_task.done() for client in physical_clients)
        assert server.connections is None
        assert server._server is None
        assert server.controllers.records() == ()
        assert set(closes.values()) == {1}
        await server.close()
    finally:
        release.set()
        if provisional_request is not None and not provisional_request.done():
            provisional_request.cancel()
        if provisional_request is not None:
            with contextlib.suppress(BaseException):
                await await_or_fail(provisional_request, what="provisional request settles")
        for client in clients:
            await client.close()
        if server.connections is not None:
            await server.close()
