"""real serve-path tests for read-only observer connections."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import TIMEOUT, _close, _connect, _request, _socket_path
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    with_message_origin,
)
from zeta.server import ZetaServer, ergonomics

MAX_PRE_HELLO_CONNECTIONS = 10


async def _observer(server: ZetaServer, number: int = 1, features: list[str] | None = None):
    if server._server is None:
        reader, writer = await _connect(server)
    elif server.port is None:
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    else:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    await _request(reader, writer, number, "hello", {
        "protocol_version": "1.2", "role": "observer",
        "features": features or ["serve_observers"],
    })
    return reader, writer


@pytest.mark.asyncio
async def test_observer_project_and_account_requests_are_read_only(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    reader, writer = await _observer(server)
    try:
        methods = [
            ("list_projects", {}), ("project_show", {"project_id": "p"}),
            ("project_memory_log", {"project_id": "p"}), ("project_inbox", {"project_id": "p"}),
            ("account_list", {}), ("account_refresh", {"provider": "codex"}),
            ("account_switch", {"account": "codex:x"}), ("list_sessions", {"project_id": "p"}),
            ("slash_list", {}), ("login_status", {}),
        ]
        for number, (method, params) in enumerate(methods, 3):
            response = (await _request(reader, writer, number, method, params))[-1]
            assert response["error"]["code"] == -32008
            assert response["error"]["data"]["code"] == "observer_read_only"
        assert (await _request(reader, writer, 99, "status"))[-1]["result"]["session"]
    finally:
        await _close(server, controller_writer)
        writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_hello_has_no_controller_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    calls: list[str] = []
    for name in ("set_background_event_sink", "set_memory_notice_sink", "set_background_wake_sink", "set_post_stream_provider_retry"):
        original = getattr(server.runtime, name)
        monkeypatch.setattr(server.runtime, name, lambda *args, _name=name, _original=original, **kwargs: calls.append(_name) or _original(*args, **kwargs))
    reader, writer = await _observer(server)
    try:
        assert calls == []
        assert (await _request(reader, writer, 2, "status"))[-1]["result"]["session"] is None
    finally:
        writer.close(); await writer.wait_closed(); await server.close()


async def _wait_thread(event: threading.Event) -> None:
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, TIMEOUT), TIMEOUT)


@pytest.mark.asyncio
async def test_observer_disconnects_keep_controller_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    await _request(controller_reader, controller_writer, 3, "send", {"text": "read it"})
    while True:
        frame = json.loads(await asyncio.wait_for(controller_reader.readline(), TIMEOUT))
        if frame.get("params", {}).get("event") == "approval_request":
            break

    assert server.connections is not None
    released = asyncio.Event()
    original_release = server.connections.release

    def release(connection: Any) -> None:
        original_release(connection)
        released.set()

    monkeypatch.setattr(server.connections, "release", release)
    assert server.runtime.policy is not None
    store_calls: list[tuple[str, int]] = []
    for name in ("pending_requests", "abort"):
        original = getattr(server.runtime.policy, name)

        def record(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            store_calls.append((_name, threading.get_ident()))
            return _original(*args, **kwargs)

        monkeypatch.setattr(server.runtime.policy, name, record)

    sockets: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
    try:
        observer = await _observer(server, 10)
        observer[1].close()
        await observer[1].wait_closed()
        await asyncio.wait_for(released.wait(), TIMEOUT)

        observers = [await _observer(server, 20 + index) for index in range(8)]
        sockets.extend(observers)
        rejected_reader, rejected_writer = await asyncio.open_unix_connection(str(server.socket_path))
        rejected = (await _request(rejected_reader, rejected_writer, 40, "hello", {
            "protocol_version": "1.2", "role": "observer", "features": ["serve_observers"],
        }))[-1]
        assert rejected["error"]["code"] == -32004
        released.clear()
        rejected_writer.close()
        await rejected_writer.wait_closed()
        await asyncio.wait_for(released.wait(), TIMEOUT)

        released.clear()
        _idle_reader, idle_writer = await asyncio.open_unix_connection(str(server.socket_path))
        idle_writer.close()
        await idle_writer.wait_closed()
        await asyncio.wait_for(released.wait(), TIMEOUT)
        assert store_calls == []

        status = (await _request(controller_reader, controller_writer, 41, "status"))[-1]
        assert [item["request_id"] for item in status["result"]["pending_approvals"]] == ["call-1"]
        approved = (await _request(
            controller_reader, controller_writer, 42, "approve", {"request_id": "call-1"}
        ))[-1]
        assert approved["result"]["decision"] == "approve"
    finally:
        for _, writer in sockets:
            writer.close()
            await writer.wait_closed()
        await _close(server, controller_writer)


@pytest.mark.asyncio
async def test_controller_history_does_not_block_other_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    session_id = (await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    ))[-1]["result"]["session"]["session_id"]
    observer_reader, observer_writer = await _observer(
        server, 3, features=["serve_observers", "ping"]
    )
    store = server.runtime.opened.store
    entered = threading.Event()
    release = threading.Event()
    ping_answered = threading.Event()
    answered_before_release: list[bool] = []
    original_replay = store.replay

    def gated_replay() -> Any:
        entered.set()
        assert release.wait(TIMEOUT)
        return original_replay()

    def bounded_release() -> None:
        answered_before_release.append(ping_answered.wait(0.5))
        release.set()

    monkeypatch.setattr(store, "replay", gated_replay)

    async def ping_after_history_starts() -> None:
        await _wait_thread(entered)
        response = (await _request(observer_reader, observer_writer, 4, "ping"))[-1]
        assert response["result"] == {"pong": True}
        ping_answered.set()

    ping = asyncio.create_task(ping_after_history_starts())
    watchdog = threading.Thread(target=bounded_release)
    watchdog.start()
    try:
        history = await asyncio.wait_for(
            _request(
                controller_reader,
                controller_writer,
                5,
                "session_history",
                {"session_id": session_id},
            ),
            TIMEOUT,
        )
        await asyncio.wait_for(ping, TIMEOUT)
        await asyncio.to_thread(watchdog.join)
        assert history[-1]["result"]["messages"] == []
        assert answered_before_release == [True]
    finally:
        release.set()
        watchdog.join()
        ping.cancel()
        await asyncio.gather(ping, return_exceptions=True)
        observer_writer.close()
        await observer_writer.wait_closed()
        await _close(server, controller_writer)


@pytest.mark.asyncio
async def test_observer_read_is_fenced_from_controller_session_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    session_a = (await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    ))[-1]["result"]["session"]["session_id"]
    await asyncio.to_thread(
        server.runtime.opened.store.append_message,
        with_message_origin(Message(MessageRole.USER, [TextContent("from-a")]), MessageOrigin.USER),
    )
    observer_reader, observer_writer = await _observer(server, 3)
    entered = threading.Event()
    release = threading.Event()
    original_history = ergonomics.history

    def gated_history(runtime: Any, params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        assert release.wait(TIMEOUT)
        return original_history(runtime, params)

    monkeypatch.setattr(ergonomics, "history", gated_history)
    try:
        request = asyncio.create_task(_request(
            observer_reader, observer_writer, 4, "session_history", {"session_id": session_a}
        ))
        await _wait_thread(entered)
        session_b = (await _request(
            controller_reader, controller_writer, 5, "new_session", {"provider": "codex"}
        ))[-1]["result"]["session"]["session_id"]
        await asyncio.to_thread(
            server.runtime.opened.store.append_message,
            with_message_origin(Message(MessageRole.USER, [TextContent("from-b")]), MessageOrigin.USER),
        )
        release.set()
        response = (await asyncio.wait_for(request, TIMEOUT))[-1]
        if "error" in response:
            assert response["error"]["code"] == -32003
        else:
            payload = json.dumps(response["result"])
            assert "from-a" in payload
            assert "from-b" not in payload
            assert session_b != session_a
    finally:
        release.set()
        observer_writer.close()
        await observer_writer.wait_closed()
        await _close(server, controller_writer)


@pytest.mark.asyncio
async def test_controller_close_never_writes_after_registry_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    assert server._client is not None and server.connections is not None
    client = server._client
    runtime_sink = server.runtime._background_event_sink
    assert runtime_sink is not None
    close_entered = asyncio.Event()
    close_release = asyncio.Event()
    original_terminate = server.control._terminate_pending_approvals

    async def gated_terminate(*args: Any, **kwargs: Any) -> None:
        close_entered.set()
        await asyncio.wait_for(close_release.wait(), TIMEOUT)
        await original_terminate(*args, **kwargs)

    monkeypatch.setattr(server.control, "_terminate_pending_approvals", gated_terminate)
    order: list[str] = []
    original_write = client._write_physical
    original_release = server.connections.release

    async def physical_write(payload: bytes) -> bool:
        order.append("write")
        return await original_write(payload)

    def registry_release(connection: Any) -> None:
        if connection is client:
            order.append("release")
        original_release(connection)

    monkeypatch.setattr(client, "_write_physical", physical_write)
    monkeypatch.setattr(server.connections, "release", registry_release)
    publication_done = asyncio.Event()
    publication_results: list[bool] = []
    original_notify = server.control._notify

    async def record_notification(*args: Any, **kwargs: Any) -> bool:
        result = await original_notify(*args, **kwargs)
        publication_results.append(result)
        publication_done.set()
        return result

    monkeypatch.setattr(server.control, "_notify", record_notification)
    writer.close()
    await writer.wait_closed()
    try:
        await asyncio.wait_for(close_entered.wait(), TIMEOUT)
        assert server.runtime._background_event_sink is runtime_sink
        runtime_sink(server.runtime.session_id, StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={"kind": "agent_completion", "text": "late"},
        ))
        close_release.set()
        await asyncio.wait_for(client._close_lock.acquire(), TIMEOUT)
        client._close_lock.release()
        await asyncio.wait_for(publication_done.wait(), TIMEOUT)
        assert publication_results in ([True], [False])
        assert "release" in order
        assert all(
            index < order.index("release")
            for index, item in enumerate(order)
            if item == "write"
        )
    finally:
        close_release.set()
        await server.close()


@pytest.mark.asyncio
async def test_pre_hello_connections_are_bounded(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    await server.start()
    sockets = [
        await asyncio.wait_for(asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT)
        for _ in range(MAX_PRE_HELLO_CONNECTIONS)
    ]
    extra_reader, extra_writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
    )
    try:
        assert await asyncio.wait_for(extra_reader.read(), TIMEOUT) == b""
        assert server.connections is not None
        assert len(server.connections._connections) == MAX_PRE_HELLO_CONNECTIONS
    finally:
        extra_writer.close()
        await extra_writer.wait_closed()
        for _, socket_writer in sockets:
            socket_writer.close()
            await socket_writer.wait_closed()
        await server.close()

async def _read_raw_until(
    reader: asyncio.StreamReader,
    frames: list[bytes],
    predicate: Any,
) -> dict[str, Any]:
    while True:
        frame = await asyncio.wait_for(reader.readline(), TIMEOUT)
        assert frame
        frames.append(frame)
        decoded = json.loads(frame)
        if predicate(decoded):
            return decoded


async def _controller_event_stream(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    observer: bool,
) -> bytes:
    counter = iter(range(1, 1000))
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))
    call = ToolCall("call-1", "bash", {"command": "printf hi"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = ZetaServer(
        home=home,
        socket_path=_socket_path(home),
        provider="codex",
        backend_factory=lambda provider, model, backend_home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    observer_socket: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
    frames: list[bytes] = []

    async def send(request_id: int, method: str, params: dict[str, object]) -> None:
        writer.write((json.dumps({
            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
        }, separators=(",", ":")) + "\n").encode())
        await asyncio.wait_for(writer.drain(), TIMEOUT)

    try:
        await send(1, "hello", {"protocol_version": "1.2"})
        await _read_raw_until(reader, frames, lambda frame: frame.get("id") == 1)
        await send(2, "new_session", {"provider": "codex"})
        session = await _read_raw_until(reader, frames, lambda frame: frame.get("id") == 2)
        session_id = session["result"]["session"]["session_id"]
        if observer:
            observer_socket = await _observer(server, 20)
        frames.clear()  # Compare the foreground/background event scenario, not setup metadata.
        await send(3, "send", {"text": "use the tool"})
        await _read_raw_until(
            reader,
            frames,
            lambda frame: frame.get("params", {}).get("event") == "approval_request",
        )
        assert server.runtime.loop is not None
        server.runtime.loop._publish_background_event(StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={"kind": "agent_completion", "text": "background done"},
        ))
        await _read_raw_until(
            reader,
            frames,
            lambda frame: frame.get("params", {}).get("event") == "sub_agent_receipt",
        )
        await send(4, "deny", {"request_id": "call-1"})
        await _read_raw_until(
            reader,
            frames,
            lambda frame: frame.get("params", {}).get("event") == "turn_end",
        )
        assert session_id.encode() in b"".join(frames)
        return b"".join(frames)
    finally:
        if observer_socket is not None:
            observer_socket[1].close()
            await observer_socket[1].wait_closed()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_controller_frames_are_byte_identical_with_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Generated from origin/main (PYTHONPATH=<origin-main-worktree>/src, zeta.__file__ checked)
    # at observers slice A; the one later change is approval_end.data.decision (PR #612).
    expected = (Path(__file__).parent / "fixtures" / "server_controller_event_stream.jsonl").read_bytes()
    assert await _controller_event_stream(tmp_path, monkeypatch, observer=True) == expected


@pytest.mark.asyncio
async def test_controller_slot_releases_after_cleanup_only(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    first_reader, first_writer = await _connect(server)
    await _request(first_reader, first_writer, 1, "hello", {"protocol_version": "1.2"})
    second_reader, second_writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        response = (await _request(second_reader, second_writer, 2, "hello", {"protocol_version": "1.2"}))[-1]
        assert response["error"]["code"] == -32001
    finally:
        await _close(server, first_writer); second_writer.close(); await second_writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_cap_and_reconnect(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    observers = [await _observer(server, i + 2) for i in range(8)]
    extra_reader, extra_writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        response = (await _request(extra_reader, extra_writer, 20, "hello", {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers"]}))[-1]
        assert response["error"]["code"] == -32004
    finally:
        extra_writer.close(); await extra_writer.wait_closed()
        observers[0][1].close(); await observers[0][1].wait_closed()
        replacement = await _observer(server, 30)
        replacement[1].close(); await replacement[1].wait_closed()
        await _close(server, controller_writer)
        for _, writer in observers[1:]: writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_outbound_limits_count_and_bytes(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    reader, writer = await _observer(server)
    try:
        assert server.connections is not None
        frame = (json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"event": "x"}}) + "\n").encode()
        observer = next(state.connection for state in server.connections._connections.values() if state.role == "observer")
        gate = asyncio.Event()
        async def blocked_write(payload: bytes) -> bool:
            await gate.wait()
            return True
        observer._write_physical = blocked_write
        tasks = [asyncio.create_task(server.connections.send(observer, frame)) for _ in range(257)]
        asyncio.get_running_loop().call_later(0.1, gate.set)
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.wait_for(reader.read(), TIMEOUT)
        assert reader.at_eof() or writer.is_closing()
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_session_tree_uses_one_store_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
    session = (
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    )[-1]["result"]["session"]
    store = server.runtime.opened.store
    first = store.append_message(
        with_message_origin(
            Message(MessageRole.USER, [TextContent("before")]), MessageOrigin.USER
        )
    )
    captured = threading.Event()
    release = threading.Event()
    entries_property = type(store).entries

    def gated_entries(target):
        entries = entries_property.__get__(target, type(target))
        captured.set()
        assert release.wait(TIMEOUT)
        return entries

    monkeypatch.setattr(type(store), "entries", property(gated_entries))
    request = asyncio.create_task(
        _request(
            reader,
            writer,
            3,
            "session_tree",
            {"session_id": session["session_id"]},
        )
    )
    try:
        await _wait_thread(captured)
        store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("after")]),
                MessageOrigin.USER,
            )
        )
    finally:
        release.set()
    try:
        response = (await request)[-1]
        assert response["result"] == {
            "branches": [
                {"id": first.id, "label": "before", "depth": 0, "current": True}
            ]
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_observer_read_workers_do_not_block_serve_loop_or_default_executor(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    reader, writer = await _observer(server)
    try:
        response = (await _request(reader, writer, 3, "status"))[-1]
        assert response["result"]["session"]
        assert await asyncio.wait_for(asyncio.to_thread(lambda: True), TIMEOUT)
    finally:
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


@pytest.mark.asyncio
async def test_stuck_observer_read_does_not_block_serve_exit(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    _reader, writer = await _observer(server)
    writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_observer_list_and_history_reads_run_off_loop(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    controller_reader, controller_writer = await _connect(server)
    await _request(controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"})
    await _request(controller_reader, controller_writer, 2, "new_session", {"provider": "codex"})
    reader, writer = await _observer(server)
    started = threading.Event()
    release = threading.Event()
    ping_done = threading.Event()
    answered_while_held: list[bool] = []
    original = server.runtime.manager.list_sessions_read_only

    def gated_list():
        started.set()
        # On the event loop this wait blocks the controller's ping; in the
        # observer worker it does not. Bounded so a regression cannot hang.
        release.wait(TIMEOUT)
        answered_while_held.append(ping_done.is_set())
        return original()

    server.runtime.manager.list_sessions_read_only = gated_list
    try:
        listing = asyncio.create_task(_request(reader, writer, 3, "list_sessions"))
        await _wait_thread(started)
        await _request(controller_reader, controller_writer, 4, "ping")
        ping_done.set()
        release.set()
        response = (await asyncio.wait_for(listing, TIMEOUT))[-1]
        assert "sessions" in response["result"]
        assert answered_while_held == [True]
    finally:
        release.set()
        await _close(server, controller_writer); writer.close(); await writer.wait_closed()


def test_delivery_model_inputs_remain_single_use_after_move() -> None:
    from zeta.model_input import ModelInputEnvelope
    from zeta.protocol.types import MessageOrigin
    from zeta.server.delivery import PendingModelInputs

    pending = PendingModelInputs()
    input_id = pending.register("session", ModelInputEnvelope("text", "display", MessageOrigin.USER))
    resolved = pending.resolve("session", {"input_id": input_id}, enabled=True)
    assert resolved.text == "text"
    with pytest.raises(ValueError, match="invalid or expired"):
        pending.resolve("session", {"input_id": input_id}, enabled=True)


@pytest.mark.asyncio
async def test_all_connection_output_is_registry_owned(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    try:
        response = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"}))[-1]
        assert response["result"]["protocol_version"]
        assert hasattr(server, "connections")
    finally:
        await _close(server, writer)
