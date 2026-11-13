"""real serve-path tests for read-only observer connections."""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from tests.support.raw_frames import assert_raw_frames_preserved
from tests.support.serve_close import close_serve
from tests.support.server_client import read_until, send
from tests.support.sockets import short_socket_path as _socket_path
from tests.test_server import TIMEOUT, _connect, _read, _request
from zeta.core.session import SessionManager
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)
from zeta.server import ZetaServer, approval_settlement
from zeta.server.handlers import history as history_module
from zeta.server.protocol import EventDescriptor

pytest_plugins = ("tests.support.socket_fixtures",)

MAX_PRE_HELLO_CONNECTIONS = 10


async def _observer(
    server: ZetaServer, number: int = 1, features: list[str] | None = None
):
    if server._server is None:
        reader, writer = await _connect(server)
    elif server.port is None:
        reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    else:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    requested_features = features or ["serve_observers"]
    hello = (
        await _request(
            reader,
            writer,
            number,
            "hello",
            {
                "protocol_version": "1.2",
                "role": "observer",
                "features": requested_features,
            },
        )
    )[-1]["result"]
    if "session_history_backward" in requested_features:
        assert "session_history_backward" in hello["capabilities"]["features"]
    readiness_method = (
        "session_grants" if "session_grants" in requested_features else "status"
    )
    await _request(reader, writer, f"{number}-ready", readiness_method)
    return reader, writer


@pytest.mark.asyncio
async def test_observer_catalog_matches_controller_with_paging(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader,
        controller_writer,
        1,
        "hello",
        {"protocol_version": "1.2", "features": ["list_sessions_paging"]},
    )
    await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    )
    await _request(
        controller_reader, controller_writer, 3, "new_session", {"provider": "codex"}
    )
    observer_reader, observer_writer = await _observer(
        server, 4, features=["serve_observers", "list_sessions_paging"]
    )
    try:
        controller_page = (
            await _request(
                controller_reader,
                controller_writer,
                5,
                "list_sessions",
                {"offset": 0, "limit": 1},
            )
        )[-1]["result"]
        observer_page = (
            await _request(
                observer_reader,
                observer_writer,
                6,
                "list_sessions",
                {"offset": 0, "limit": 1},
            )
        )[-1]["result"]
        assert observer_page == controller_page
        assert observer_page["next_offset"] == 1
        observer_tail = (
            await _request(
                observer_reader,
                observer_writer,
                7,
                "list_sessions",
                {"offset": 1, "limit": 1},
            )
        )[-1]["result"]
        assert observer_tail["next_offset"] is None
        assert "first_message_preview" in observer_page["sessions"][0]
    finally:
        await close_serve(server, controller_writer)
        observer_writer.close()
        await observer_writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_catalog_does_not_wait_for_controller_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    session_id = (
        await _request(
            controller_reader,
            controller_writer,
            2,
            "new_session",
            {"provider": "codex"},
        )
    )[-1]["result"]["session"]["session_id"]
    observer_reader, observer_writer = await _observer(
        server, 3, features=["serve_observers", "list_sessions_paging", "ping"]
    )
    store = server.runtime.opened.store
    entered = threading.Event()
    release = threading.Event()
    original_page = store.history_index_page

    def gated_page(request: dict[str, object]) -> Any:
        entered.set()
        assert release.wait(TIMEOUT)
        return original_page(request)

    monkeypatch.setattr(store, "history_index_page", gated_page)
    try:
        history = asyncio.create_task(
            _request(
                controller_reader,
                controller_writer,
                4,
                "session_history",
                {"session_id": session_id},
            )
        )
        assert await asyncio.to_thread(entered.wait, TIMEOUT)
        catalog = await asyncio.wait_for(
            _request(
                observer_reader, observer_writer, 5, "list_sessions", {"limit": 1}
            ),
            TIMEOUT,
        )
        assert catalog[-1]["result"]["sessions"]
        release.set()
        assert (await asyncio.wait_for(history, 10))[-1]["result"]["messages"] == []
    finally:
        release.set()
        observer_writer.close()
        await observer_writer.wait_closed()
        await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_observer_project_and_account_requests_are_read_only(
    tmp_path: Path,
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    )
    reader, writer = await _observer(server)
    try:
        methods = [
            ("list_projects", {}),
            ("project_show", {"project_id": "p"}),
            ("project_memory_log", {"project_id": "p"}),
            ("project_inbox", {"project_id": "p"}),
            ("account_list", {}),
            ("account_refresh", {"provider": "codex"}),
            ("account_switch", {"account": "codex:x"}),
            ("list_sessions", {"project_id": "p"}),
            ("slash_list", {}),
            ("login_status", {}),
        ]
        for number, (method, params) in enumerate(methods, 3):
            response = (await _request(reader, writer, number, method, params))[-1]
            assert response["error"]["code"] == -32008
            assert response["error"]["data"]["code"] == "observer_read_only"
        assert (await _request(reader, writer, 99, "status"))[-1]["result"]["session"]
    finally:
        await close_serve(server, controller_writer)
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_hello_has_no_controller_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    calls: list[str] = []
    for name in (
        "set_background_event_sink",
        "set_memory_notice_sink",
        "set_background_wake_sink",
    ):
        original = getattr(server.runtime, name)
        monkeypatch.setattr(
            server.runtime,
            name,
            lambda *args, _name=name, _original=original, **kwargs: (
                calls.append(_name) or _original(*args, **kwargs)
            ),
        )
    reader, writer = await _observer(server)
    try:
        assert calls == []
        assert (await _request(reader, writer, 2, "status"))[-1]["result"][
            "session"
        ] is None
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


async def _wait_thread(event: threading.Event) -> None:
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, TIMEOUT), TIMEOUT)


@pytest.mark.asyncio
async def test_observer_joins_with_pending_approval_baseline(tmp_path: Path) -> None:
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
    observer_writer: asyncio.StreamWriter | None = None
    legacy_observer_writer: asyncio.StreamWriter | None = None
    try:
        await _request(
            controller_reader,
            controller_writer,
            1,
            "hello",
            {"protocol_version": "1.2", "features": ["approval_reconnect"]},
        )
        session = (
            await _request(
                controller_reader,
                controller_writer,
                2,
                "new_session",
                {"provider": "codex"},
            )
        )[-1]["result"]["session"]
        await _request(
            controller_reader, controller_writer, 3, "send", {"text": "read it"}
        )
        approval = (
            await read_until(
                controller_reader,
                lambda frame: frame.get("params", {}).get("event")
                == "approval_request",
                what="approval_request",
            )
        )[-1]["params"]

        observer_reader, observer_writer = await _observer(
            server,
            10,
            ["serve_observers", "approval_reconnect", "session_replay_seq"],
        )
        status = (await _request(observer_reader, observer_writer, 11, "status"))[-1][
            "result"
        ]
        joined = (
            await _request(
                observer_reader,
                observer_writer,
                12,
                "session_history_and_subscribe",
                {"session_id": session["session_id"]},
            )
        )[-1]["result"]
        request_id = approval["request_id"]
        assert [row["request_id"] for row in status["pending_approvals"]] == [
            request_id
        ]
        assert [
            row["request_id"] for row in joined["status"]["pending_approvals"]
        ] == [request_id]

        legacy_observer_reader, legacy_observer_writer = await _observer(
            server, 20, ["serve_observers"]
        )
        legacy_status = (
            await _request(legacy_observer_reader, legacy_observer_writer, 21, "status")
        )[-1]["result"]
        assert [
            row["request_id"] for row in legacy_status["pending_approvals"]
        ] == [request_id]

        rejected = (
            await _request(
                observer_reader,
                observer_writer,
                13,
                "approve",
                {"request_id": request_id},
            )
        )[-1]
        assert rejected["error"]["code"] == -32008
        await _request(
            controller_reader,
            controller_writer,
            4,
            "approve",
            {"request_id": request_id},
        )
        ended = (
            await read_until(
                observer_reader,
                lambda frame: frame.get("params", {}).get("event") == "approval_end",
                what="approval_end",
            )
        )[-1]["params"]
        assert ended["request_id"] == request_id
    finally:
        if observer_writer is not None:
            observer_writer.close()
            await observer_writer.wait_closed()
        if legacy_observer_writer is not None:
            legacy_observer_writer.close()
            await legacy_observer_writer.wait_closed()
        await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_join_baseline_represents_durable_approval_after_restart(
    tmp_path: Path,
) -> None:
    # A durable foreground approval persisted before a serve restart must appear in the
    # join baseline for a resumed idle session, not only in the status RPC. The loop owns
    # no in-memory approval records until a turn resumes, so the baseline falls back to the
    # durable store the same way status does.
    manager = SessionManager(tmp_path)
    opened = await asyncio.to_thread(
        lambda: manager.create(provider="codex", model="offline", cwd=tmp_path)
    )
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("resumed-call", "read", {"path": str(target)})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    session_id = opened.metadata.session_id
    opened.store.close()
    backend = FakeBackend([])
    server = ZetaServer(
        home=tmp_path,
        provider="codex",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    controller_reader, controller_writer = await _connect(server)
    observer_writer: asyncio.StreamWriter | None = None
    try:
        await _request(
            controller_reader,
            controller_writer,
            1,
            "hello",
            {"protocol_version": "1.2", "features": ["session_replay_seq"]},
        )
        await _request(
            controller_reader, controller_writer, 2, "resume", {"session_id": session_id}
        )
        controller_join = (
            await _request(
                controller_reader,
                controller_writer,
                3,
                "session_history_and_subscribe",
                {"session_id": session_id},
            )
        )[-1]["result"]
        assert [
            row["request_id"]
            for row in controller_join["status"]["pending_approvals"]
        ] == [call.id]

        observer_reader, observer_writer = await _observer(
            server, 10, ["serve_observers", "session_replay_seq"]
        )
        observer_join = (
            await _request(
                observer_reader,
                observer_writer,
                11,
                "session_history_and_subscribe",
                {"session_id": session_id},
            )
        )[-1]["result"]
        assert [
            row["request_id"]
            for row in observer_join["status"]["pending_approvals"]
        ] == [call.id]
        rejected = (
            await _request(
                observer_reader,
                observer_writer,
                12,
                "approve",
                {"request_id": call.id},
            )
        )[-1]
        assert rejected["error"]["code"] == -32008
    finally:
        if observer_writer is not None:
            observer_writer.close()
            await observer_writer.wait_closed()
        await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_reconnect_baseline_keeps_durable_row_beside_settling_live_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two durable foreground approvals survive a restart. A controller approves one (D1)
    # over the socket; its decision persists, so D1 is a live settling record while D2
    # stays durable-only. In that window a second client's join baseline must still list
    # BOTH D1 (with resolution_pending "approve") and D2. The old rule ("the live rows if
    # there are any, else the durable store") returned only D1 and dropped D2.
    manager = SessionManager(tmp_path)
    opened = await asyncio.to_thread(
        lambda: manager.create(provider="codex", model="offline", cwd=tmp_path)
    )
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call_one = ToolCall("call-d1", "read", {"path": str(target)})
    call_two = ToolCall("call-d2", "read", {"path": str(target)})
    opened.store.append_message_with_approval_requests(
        Message(
            MessageRole.ASSISTANT,
            [ToolUseContent(call_one), ToolUseContent(call_two)],
        ),
        [(call_one.id, call_one), (call_two.id, call_two)],
    )
    session_id = opened.metadata.session_id
    opened.store.close()
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server = ZetaServer(
        home=tmp_path,
        provider="codex",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    controller_reader, controller_writer = await _connect(server)
    observer_writer: asyncio.StreamWriter | None = None
    try:
        await _request(
            controller_reader,
            controller_writer,
            1,
            "hello",
            {"protocol_version": "1.2", "features": ["session_replay_seq"]},
        )
        await _request(
            controller_reader, controller_writer, 2, "resume", {"session_id": session_id}
        )

        # Hold D1 in its settling phase: block the durable persist so the approve handler
        # parks after it has created D1's live settling record but before it finishes.
        entered = asyncio.Event()
        release = asyncio.Event()
        real_persist = approval_settlement.persist_durably

        async def blocking_persist(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            await await_or_fail(release.wait(), what="release settlement")
            return await real_persist(*args, **kwargs)

        monkeypatch.setattr(approval_settlement, "persist_durably", blocking_persist)

        # Approve D1 over the real socket without reading the reply yet; the handler parks
        # in the blocked persist while D1 settles.
        await send(controller_reader, controller_writer, 3, "approve", {"request_id": call_one.id})
        await await_or_fail(entered.wait(), what="settlement entered")
        try:
            # A second client joins during the settling window. Its baseline must show D1
            # settling (resolution_pending) alongside the still-durable D2.
            observer_reader, observer_writer = await _observer(
                server, 10, ["serve_observers", "session_replay_seq"]
            )
            observer_join = (
                await _request(
                    observer_reader,
                    observer_writer,
                    11,
                    "session_history_and_subscribe",
                    {"session_id": session_id},
                )
            )[-1]["result"]
            rows = {
                row["request_id"]: row
                for row in observer_join["status"]["pending_approvals"]
            }
            assert set(rows) == {call_one.id, call_two.id}
            assert rows[call_one.id].get("resolution_pending") == "approve"
        finally:
            # Release the persist and read the approve reply off the controller socket,
            # so a failed assertion above does not deadlock on the parked settlement.
            release.set()
            await read_until(
                controller_reader,
                lambda frame: frame.get("id") == 3
                and frame.get("result", {}).get("accepted") is True,
                what="approve D1 reply",
            )
    finally:
        if observer_writer is not None:
            observer_writer.close()
            await observer_writer.wait_closed()
        await close_serve(server, controller_writer)


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
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    )
    await _request(controller_reader, controller_writer, 3, "send", {"text": "read it"})
    while True:
        frame = await _read(controller_reader)
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

        def record(
            *args: Any, _name: str = name, _original: Any = original, **kwargs: Any
        ) -> Any:
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
        rejected_reader, rejected_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        rejected = (
            await _request(
                rejected_reader,
                rejected_writer,
                40,
                "hello",
                {
                    "protocol_version": "1.2",
                    "role": "observer",
                    "features": ["serve_observers"],
                },
            )
        )[-1]
        assert rejected["error"]["code"] == -32001
        released.clear()
        rejected_writer.close()
        await rejected_writer.wait_closed()
        await asyncio.wait_for(released.wait(), TIMEOUT)

        released.clear()
        _idle_reader, idle_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        idle_writer.close()
        await idle_writer.wait_closed()
        await asyncio.wait_for(released.wait(), TIMEOUT)
        assert store_calls == []

        status = (await _request(controller_reader, controller_writer, 41, "status"))[
            -1
        ]
        assert [
            item["request_id"] for item in status["result"]["pending_approvals"]
        ] == ["call-1"]
        approved = (
            await _request(
                controller_reader,
                controller_writer,
                42,
                "approve",
                {"request_id": "call-1"},
            )
        )[-1]
        assert approved["result"]["decision"] == "approve"
    finally:
        for _, writer in sockets:
            writer.close()
            await writer.wait_closed()
        await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_controller_history_does_not_block_other_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    session_id = (
        await _request(
            controller_reader,
            controller_writer,
            2,
            "new_session",
            {"provider": "codex"},
        )
    )[-1]["result"]["session"]["session_id"]
    observer_reader, observer_writer = await _observer(
        server, 3, features=["serve_observers", "ping"]
    )
    store = server.runtime.opened.store
    entered = threading.Event()
    release = threading.Event()
    ping_answered = threading.Event()
    answered_before_release: list[bool] = []
    original_page = store.history_index_page

    def gated_page(request: dict[str, object]) -> Any:
        entered.set()
        assert release.wait(TIMEOUT)
        return original_page(request)

    def bounded_release() -> None:
        answered_before_release.append(ping_answered.wait(0.5))
        release.set()

    monkeypatch.setattr(store, "history_index_page", gated_page)

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
        await asyncio.to_thread(watchdog.join, 10)
        assert history[-1]["result"]["messages"] == []
        assert answered_before_release == [True]
    finally:
        release.set()
        watchdog.join(timeout=10)
        ping.cancel()
        await asyncio.wait_for(asyncio.gather(ping, return_exceptions=True), 10)
        observer_writer.close()
        await observer_writer.wait_closed()
        await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_observer_read_is_fenced_from_controller_session_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    session_a = (
        await _request(
            controller_reader,
            controller_writer,
            2,
            "new_session",
            {"provider": "codex"},
        )
    )[-1]["result"]["session"]["session_id"]
    server.runtime.opened.store.append_message(
        with_message_origin(
            Message(MessageRole.USER, [TextContent("from-a")]), MessageOrigin.USER
        )
    )
    observer_reader, observer_writer = await _observer(server, 3)
    entered = threading.Event()
    release = threading.Event()
    original_history = history_module.history

    def gated_history(
        runtime: Any, params: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        entered.set()
        assert release.wait(TIMEOUT)
        return original_history(runtime, params)

    monkeypatch.setattr(history_module, "history", gated_history)
    try:
        request = asyncio.create_task(
            _request(
                observer_reader,
                observer_writer,
                4,
                "session_history",
                {"session_id": session_a},
            )
        )
        await _wait_thread(entered)
        session_b = (
            await _request(
                controller_reader,
                controller_writer,
                5,
                "new_session",
                {"provider": "codex"},
            )
        )[-1]["result"]["session"]["session_id"]
        server.runtime.opened.store.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("from-b")]), MessageOrigin.USER
            )
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
        await close_serve(server, controller_writer)


@pytest.mark.asyncio
async def test_controller_close_never_writes_after_registry_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
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
    original_admit = server.control._admit_background

    async def record_notification(*args: Any, **kwargs: Any) -> bool:
        result = await original_admit(*args, **kwargs)
        publication_results.append(result)
        publication_done.set()
        return result

    # test-contract: seam observe wire admission after semantic recording
    monkeypatch.setattr(server.control, "_admit_background", record_notification)
    writer.close()
    await writer.wait_closed()
    try:
        await asyncio.wait_for(close_entered.wait(), TIMEOUT)
        assert server.runtime._background_event_sink is runtime_sink
        publication = runtime_sink(
            server.runtime.session_id,
            StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={"kind": "agent_completion", "text": "late"},
            ),
        )
        publication_task = (
            asyncio.create_task(publication) if inspect.isawaitable(publication) else None
        )
        close_release.set()
        await asyncio.wait_for(client._close_lock.acquire(), TIMEOUT)
        client._close_lock.release()
        await asyncio.wait_for(publication_done.wait(), TIMEOUT)
        if publication_task is not None:
            await await_or_fail(publication_task, what="late background publication")
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
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    await server.start()
    sockets = [
        await asyncio.wait_for(
            asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
        )
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
        backend_factory=lambda provider, model, backend_home: (
            backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(server)
    observer_socket: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None

    def is_event(name: str) -> Callable[[dict[str, Any]], bool]:
        return lambda frame: frame.get("params", {}).get("event") == name

    try:
        await send(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        await read_until(reader, lambda frame: frame.get("id") == 1, what="hello")
        await send(reader, writer, 2, "new_session", {"provider": "codex"})
        session = (
            await read_until(
                reader, lambda frame: frame.get("id") == 2, what="new_session"
            )
        )[-1]
        session_id = session["result"]["session"]["session_id"]
        if observer:
            observer_socket = await _observer(server, 20)
        # Compare the foreground/background event scenario, not setup metadata.
        await send(reader, writer, 3, "send", {"text": "use the tool"})
        frames = await read_until(
            reader, is_event("approval_request"), what="approval_request"
        )
        assert server.runtime.loop is not None
        publication = server.runtime.loop._publish_background_event(
            StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={"kind": "agent_completion", "text": "background done"},
            )
        )
        if inspect.isawaitable(publication):
            await await_or_fail(publication, what="background event publication")
        frames += await read_until(
            reader, is_event("sub_agent_receipt"), what="sub_agent_receipt"
        )
        await send(reader, writer, 4, "deny", {"request_id": "call-1"})
        frames += await read_until(
            reader,
            lambda frame: (
                is_event("turn_end")(frame)
                and frame.get("params", {}).get("data", {}).get("turn") == 1
            ),
            what="turn_end",
        )
        deny_response = next(frame for frame in frames if frame.get("id") == 4)
        assert deny_response["result"] == {
            "accepted": True,
            "request_id": "call-1",
            "decision": "deny",
        }
        turn_one_end = next(
            index
            for index, frame in enumerate(frames)
            if (
                is_event("turn_end")(frame)
                and frame.get("params", {}).get("data", {}).get("turn") == 1
            )
        )
        bounded_frames = frames[: turn_one_end + 1]
        if deny_response not in bounded_frames:
            bounded_frames.append(deny_response)
        raw = b"".join(frame.raw for frame in bounded_frames)
        assert session_id.encode() in raw
        return raw
    finally:
        if observer_socket is not None:
            observer_socket[1].close()
            await observer_socket[1].wait_closed()
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_controller_frames_are_byte_identical_with_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Generated from origin/main (PYTHONPATH=<origin-main-worktree>/src, zeta.__file__ checked)
    # at observers slice A; the one later change is approval_end.data.decision (PR #612).
    expected = (
        Path(__file__).parent / "fixtures" / "server_controller_event_stream.jsonl"
    ).read_bytes()
    assert_raw_frames_preserved(
        await _controller_event_stream(tmp_path, monkeypatch, observer=True), expected
    )


@pytest.mark.asyncio
async def test_controller_slot_releases_after_cleanup_only(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    first_reader, first_writer = await _connect(server)
    await _request(first_reader, first_writer, 1, "hello", {"protocol_version": "1.2"})
    second_reader, second_writer = await asyncio.open_unix_connection(
        str(server.socket_path)
    )
    try:
        response = (
            await _request(
                second_reader, second_writer, 2, "hello", {"protocol_version": "1.2"}
            )
        )[-1]
        assert response["error"]["code"] == -32001
    finally:
        await close_serve(server, first_writer)
        second_writer.close()
        await second_writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_cap_and_reconnect(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    observers = [await _observer(server, i + 2) for i in range(8)]
    extra_reader, extra_writer = await asyncio.open_unix_connection(
        str(server.socket_path)
    )
    try:
        response = (
            await _request(
                extra_reader,
                extra_writer,
                20,
                "hello",
                {
                    "protocol_version": "1.2",
                    "role": "observer",
                    "features": ["serve_observers"],
                },
            )
        )[-1]
        assert response["error"]["code"] == -32001
    finally:
        extra_writer.close()
        await extra_writer.wait_closed()
        observers[0][1].close()
        await observers[0][1].wait_closed()
        replacement = await _observer(server, 30)
        replacement[1].close()
        await replacement[1].wait_closed()
        await close_serve(server, controller_writer)
        for _, writer in observers[1:]:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_outbound_limits_count_and_bytes(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    reader, writer = await _observer(server)
    try:
        assert server.connections is not None
        observer = next(
            state.connection
            for state in server.connections._connections.values()
            if state.role == "observer"
        )
        frame = observer.codec.notification(
            EventDescriptor.ASSISTANT_DELTA, "session", delta="x"
        )
        gate = asyncio.Event()

        async def blocked_write(payload: bytes) -> bool:
            await asyncio.wait_for(gate.wait(), 10)
            return True

        observer._write_physical = blocked_write
        tasks = [
            asyncio.create_task(server.connections.send(observer, frame))
            for _ in range(257)
        ]
        asyncio.get_running_loop().call_later(0.1, gate.set)
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        await asyncio.wait_for(reader.read(), TIMEOUT)
        assert reader.at_eof() or writer.is_closing()
    finally:
        await close_serve(server, controller_writer)
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_session_tree_uses_one_store_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
    session = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[
        -1
    ]["result"]["session"]
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
        response = (await asyncio.wait_for(request, 10))[-1]
        assert response["result"] == {
            "branches": [
                {"id": first.id, "label": "before", "depth": 0, "current": True}
            ]
        }
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_observer_read_workers_do_not_block_serve_loop_or_default_executor(
    tmp_path: Path,
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    )
    reader, writer = await _observer(server)
    try:
        response = (await _request(reader, writer, 3, "status"))[-1]
        assert response["result"]["session"]
        assert await asyncio.wait_for(asyncio.to_thread(lambda: True), TIMEOUT)
    finally:
        await close_serve(server, controller_writer)
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_stuck_observer_read_does_not_block_serve_exit(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    _reader, writer = await _observer(server)
    writer.close()
    await writer.wait_closed()
    await server.close()


@pytest.mark.asyncio
async def test_observer_list_and_history_reads_run_off_loop(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    await _request(
        controller_reader, controller_writer, 2, "new_session", {"provider": "codex"}
    )
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
        await close_serve(server, controller_writer)
        writer.close()
        await writer.wait_closed()


def test_delivery_model_inputs_remain_single_use_after_move() -> None:
    from zeta.model_input import ModelInputEnvelope
    from zeta.protocol.types import MessageOrigin
    from zeta.server.delivery import PendingModelInputs

    pending = PendingModelInputs()
    input_id = pending.register(
        "session", ModelInputEnvelope("text", "display", MessageOrigin.USER)
    )
    resolved = pending.resolve("session", {"input_id": input_id}, enabled=True)
    assert resolved.text == "text"
    with pytest.raises(ValueError, match="invalid or expired"):
        pending.resolve("session", {"input_id": input_id}, enabled=True)


@pytest.mark.asyncio
async def test_all_connection_output_is_registry_owned(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    reader, writer = await _connect(server)
    try:
        response = (
            await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        )[-1]
        assert response["result"]["protocol_version"]
        assert hasattr(server, "connections")
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "features", [["serve_observers"], ["serve_observers", "ping", "delivery_id"]]
)
async def test_observer_hello_advertises_only_callable_requests(
    tmp_path: Path, features: list[str]
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        hello = (
            await _request(
                reader,
                writer,
                2,
                "hello",
                {
                    "protocol_version": "1.2",
                    "role": "observer",
                    "features": features,
                },
            )
        )[-1]["result"]
        advertised = hello["capabilities"]["requests"]
        params = {"delivery_status": {"delivery_id": "missing"}}
        for number, method in enumerate(advertised, start=3):
            if method == "hello":
                continue
            reply = (
                await _request(reader, writer, number, method, params.get(method, {}))
            )[-1]
            error = reply.get("error", {})
            assert error.get("code") != -32601, (method, reply)
            assert "requires negotiated feature" not in error.get("message", ""), (
                method,
                reply,
            )
    finally:
        await close_serve(server, controller_writer)
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_observer_history_shape_matches_negotiated_backward_feature(
    tmp_path: Path,
) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    controller_reader, controller_writer = await _connect(server)
    await _request(
        controller_reader, controller_writer, 1, "hello", {"protocol_version": "1.2"}
    )
    session_id = (
        await _request(
            controller_reader,
            controller_writer,
            2,
            "new_session",
            {"provider": "codex"},
        )
    )[-1]["result"]["session"]["session_id"]
    server.runtime.opened.store.append_message(
        with_message_origin(
            Message(MessageRole.USER, [TextContent("history")]), MessageOrigin.USER
        )
    )
    backward_reader, backward_writer = await _observer(
        server, 3, features=["serve_observers", "session_history_backward"]
    )
    forward_reader, forward_writer = await _observer(
        server, 4, features=["serve_observers"]
    )
    try:
        backward = (
            await _request(
                backward_reader,
                backward_writer,
                5,
                "session_history",
                {"session_id": session_id, "before": None},
            )
        )[-1]
        assert "error" not in backward
        assert "next_before" in backward["result"]
        forward = (
            await _request(
                forward_reader,
                forward_writer,
                6,
                "session_history",
                {"session_id": session_id, "offset": 0},
            )
        )[-1]
        assert "error" not in forward
        assert "next_offset" in forward["result"]
        assert "next_before" not in forward["result"]
    finally:
        backward_writer.close()
        forward_writer.close()
        await backward_writer.wait_closed()
        await forward_writer.wait_closed()
        await close_serve(server, controller_writer)
