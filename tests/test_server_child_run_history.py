"""real transport checks for child run history negotiation and validation."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.replay_transport import connect_replay, negotiate, replay_server
from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.agent import pending_delivery as pending_delivery_module
from zeta.agent.background_owner import BackgroundAgentOwner
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore, ObserverSnapshotChanged
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
from zeta.server import ZetaServer


async def _open(tmp_path: Path, features: list[str]):
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        approval_reconnect_grace=0,
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    await request(
        reader,
        writer,
        1,
        "hello",
        {"protocol_version": "1.2", "features": features},
    )
    new_reply = (await request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]
    assert "result" in new_reply, new_reply
    session_id = new_reply["result"]["session"]["session_id"]
    return server, reader, writer, session_id


@pytest.mark.asyncio
async def test_child_history_is_available_to_negotiated_observer(tmp_path: Path) -> None:
    server, _controller_reader, controller_writer, session_id = await _open(
        tmp_path, ["child_run_history"]
    )
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("message")]))
    child.close()
    observer_reader, observer_writer = await asyncio.open_unix_connection(
        str(server.socket_path)
    )
    try:
        hello = (
            await request(
                observer_reader,
                observer_writer,
                3,
                "hello",
                {
                    "protocol_version": "1.2",
                    "role": "observer",
                    "features": ["serve_observers", "child_run_history"],
                },
            )
        )[-1]
        assert "child_run_history" in hello["result"]["capabilities"]["features"]
        reply = (
            await request(
                observer_reader,
                observer_writer,
                4,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1"},
            )
        )[-1]
        assert "result" in reply, reply
        assert reply["result"]["messages"][0]["content"][0]["text"] == "message"
        assert reply["result"]["next_before"] is None
        assert "history_validator" in reply["result"]
    finally:
        await close_serve(server, observer_writer, controller_writer)


@pytest.mark.asyncio
async def test_history_validator_contract_is_shared_by_all_serve_history_methods(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(
        tmp_path,
        [
            "child_run_history",
            "session_history_backward",
            "session_replay_seq",
        ],
    )
    server.runtime.opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("parent")])
    )
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("child")]))
    child.close()
    methods = (
        ("session_history", {"session_id": session_id, "before": None}),
        (
            "child_run_history",
            {"session_id": session_id, "run_id": f"{session_id}:1"},
        ),
        ("session_history_and_subscribe", {"session_id": session_id}),
    )
    request_id = 3
    try:
        for method, newest_params in methods:
            newest = (
                await request(reader, writer, request_id, method, newest_params)
            )[-1]
            request_id += 1
            assert "result" in newest, (method, newest)
            validator = newest["result"]["history_validator"]
            assert set(validator) == {"end", "entry_id", "repair_generation"}

            continuation = {
                **newest_params,
                "before": validator["end"],
                "history_validator": validator,
            }
            accepted = (
                await request(reader, writer, request_id, method, continuation)
            )[-1]
            request_id += 1
            assert "result" in accepted, (method, accepted)
            assert set(accepted["result"]["history_validator"]) == {
                "end",
                "entry_id",
                "repair_generation",
            }

            continuation["history_validator"] = {
                "end": validator["end"],
                "entry_id": validator["entry_id"],
            }
            stale = (
                await request(reader, writer, request_id, method, continuation)
            )[-1]
            request_id += 1
            assert stale["error"]["code"] == -32009, (method, stale)
            assert stale["error"]["data"]["code"] == "stale_history_snapshot"

            restarted = (
                await request(reader, writer, request_id, method, newest_params)
            )[-1]
            request_id += 1
            assert "result" in restarted, (method, restarted)
            assert set(restarted["result"]["history_validator"]) == {
                "end",
                "entry_id",
                "repair_generation",
            }
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_requires_negotiated_feature_and_valid_run_id(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, [])
    try:
        reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": "not-a-run"},
            )
        )[-1]
        assert reply["error"]["code"] == -32601
    finally:
        await close_serve(server, writer)

    server, reader, writer, session_id = await _open(tmp_path / "second", ["child_run_history"])
    try:
        reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": "../../etc"},
            )
        )[-1]
        assert reply["error"]["code"] == -32602
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_reads_newest_and_continuation_over_unix_socket(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(
        tmp_path, ["child_run_history", "session_history_fidelity"]
    )
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    for index in range(130):
        child.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"message-{index}")])
        )
    nested = ConversationStore(
        child.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    nested.start_agent_lifecycle(
        handle=f"{session_id}:1:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=2,
        agent_type="worker",
        description="nested",
    )
    nested.append_message(Message(MessageRole.ASSISTANT, [TextContent("nested")]))
    nested.close()
    child.close()
    try:
        newest_reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1"},
            )
        )[-1]
        assert "result" in newest_reply, newest_reply
        newest = newest_reply["result"]
        assert len(newest["messages"]) == 130
        assert newest["messages"][0]["content"][0]["text"] == "message-0"
        assert newest["messages"][-1]["content"][0]["text"] == "message-129"
        assert newest.get("next_before") is None
        nested_reply = (
            await request(
                reader,
                writer,
                4,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1:1"},
            )
        )[-1]
        assert "result" in nested_reply, nested_reply
        assert nested_reply["result"]["messages"][0]["content"][0]["text"] == "nested"
        continuation = (
            await request(
                reader,
                writer,
                4,
                "child_run_history",
                {
                    "session_id": session_id,
                    "run_id": f"{session_id}:1",
                    "before": 65,
                    "history_validator": newest["history_validator"],
                },
            )
        )[-1]["result"]
        assert continuation["messages"][0]["content"][0]["text"] == "message-0"
        assert continuation["messages"][-1]["content"][0]["text"] == "message-64"
        child_path = server.runtime.opened.store.session_dir / "agents" / "1"
        rows = (child_path / "conversation.jsonl").read_text().splitlines(True)
        replacement = child_path / "conversation.replacement"
        replacement.write_text("".join(rows[:-1]))
        os.replace(replacement, child_path / "conversation.jsonl")
        stale = (
            await request(
                reader,
                writer,
                5,
                "child_run_history",
                {
                    "session_id": session_id,
                    "run_id": f"{session_id}:1",
                    "before": 65,
                    "history_validator": newest["history_validator"],
                },
            )
        )[-1]
        assert stale["error"]["code"] == -32009
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_reads_closed_child_after_serve_restart(tmp_path: Path) -> None:
    server, _reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1", started_at="2026-01-01T00:00:00+00:00", depth=1,
        agent_type="worker", description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("after-restart")]))
    child.close()
    await close_serve(server, writer)
    resumed = ZetaServer(
        provider="codex", home=tmp_path, socket_path=short_socket_path(tmp_path), approval_reconnect_grace=0
    )
    await resumed.runtime.resume_session(session_id)
    await resumed.start()
    new_reader, new_writer = await asyncio.open_unix_connection(str(resumed.socket_path))
    try:
        await request(new_reader, new_writer, 1, "hello", {"protocol_version": "1.2", "features": ["child_run_history"]})
        reply = (await request(new_reader, new_writer, 2, "child_run_history", {"session_id": session_id, "run_id": f"{session_id}:1"}))[-1]
        assert reply["result"]["messages"][0]["content"][0]["text"] == "after-restart"
        page = reply["result"]
        continuation = (await request(new_reader, new_writer, 3, "child_run_history", {"session_id": session_id, "run_id": f"{session_id}:1", "before": 0, "history_validator": page["history_validator"]}))[-1]
        assert continuation["result"]["messages"] == []
    finally:
        await close_serve(resumed, new_writer)


@pytest.mark.asyncio
async def test_child_history_rejects_malformed_request_shape_over_unix_socket(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    invalid = (
        {},
        {"run_id": f"{session_id}:1"},
        {"session_id": session_id},
        {"session_id": 1, "run_id": f"{session_id}:1"},
        {"session_id": session_id, "run_id": 1},
        {"session_id": session_id, "run_id": f"{session_id}:1", "extra": True},
    )
    try:
        for request_id, params in enumerate(invalid, start=3):
            reply = (
                await request(
                    reader, writer, request_id, "child_run_history", params
                )
            )[-1]
            assert reply["error"]["code"] == -32602, (params, reply)
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_reports_cached_stable_torn_tail_with_repair_offer(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    child.close()
    params = {"session_id": session_id, "run_id": f"{session_id}:1"}
    try:
        first = (await request(reader, writer, 3, "child_run_history", params))[-1]
        assert "result" in first, first
        transcript = server.runtime.opened.store.session_dir / "agents" / "1" / "conversation.jsonl"
        with transcript.open("ab") as stream:
            stream.write(b'{"type":"message"')
        damaged = (await request(reader, writer, 4, "child_run_history", params))[-1]
        assert damaged.get("error", {}).get("data") == {
            "code": "corrupt_transcript",
            "kind": "torn_tail",
            "row": 3,
            "repairable": True,
            "session_id": session_id,
            "actions": [
                {
                    "method": "repair_transcript",
                    "session_id": session_id,
                    "child_run_id": f"{session_id}:1",
                    "requires_confirmation": True,
                }
            ],
        }
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_preserves_corrupt_child_diagnosis_over_unix_socket(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    child.close()
    transcript = server.runtime.opened.store.session_dir / "agents" / "1" / "conversation.jsonl"
    with transcript.open("ab") as stream:
        stream.write(b"{not json}\n")
    try:
        damaged = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1"},
            )
        )[-1]
        assert damaged["error"]["code"] == -32000, damaged
        assert damaged["error"]["data"]["code"] == "corrupt_transcript"
        assert damaged["error"]["data"]["session_id"] == session_id
        assert damaged["error"]["data"]["actions"][0]["child_run_id"] == f"{session_id}:1"
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_discards_obsolete_corruption_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    child.close()
    transcript = server.runtime.opened.store.session_dir / "agents" / "1" / "conversation.jsonl"
    valid = transcript.read_bytes()
    with transcript.open("ab") as stream:
        stream.write(b"{not json}\n")
    original_read_fd = ConversationStore._read_fd
    replaced = False

    def repair_after_read(fd: int, length: int) -> bytes:
        nonlocal replaced
        raw = original_read_fd(fd, length)
        if not replaced and os.fstat(fd).st_ino == transcript.stat().st_ino:
            replacement = transcript.with_suffix(".replacement")
            replacement.write_bytes(valid)
            os.replace(replacement, transcript)
            replaced = True
        return raw

    # test-contract: seam repairs the source between observer g0 and g1
    monkeypatch.setattr(ConversationStore, "_read_fd", staticmethod(repair_after_read))
    try:
        reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1"},
            )
        )[-1]
        assert "error" not in reply, reply
        assert reply["result"]["messages"][-1]["content"][0]["text"] == "complete"
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_bounds_persistent_source_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    child.close()
    transcript = server.runtime.opened.store.session_dir / "agents" / "1" / "conversation.jsonl"
    original_read_fd = ConversationStore._read_fd

    def replace_after_every_read(fd: int, length: int) -> bytes:
        raw = original_read_fd(fd, length)
        if os.fstat(fd).st_ino == transcript.stat().st_ino:
            replacement = transcript.with_suffix(".replacement")
            replacement.write_bytes(raw)
            os.replace(replacement, transcript)
        return raw

    # test-contract: seam forces every authoritative observer read to be replaced
    monkeypatch.setattr(ConversationStore, "_read_fd", staticmethod(replace_after_every_read))
    try:
        replies = await await_or_fail(
            request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1"},
            ),
            what="bounded child history source-change request",
        )
        assert replies[-1]["error"]["code"] == -32009, replies[-1]
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_warm_reader_serves_snapshot_when_every_refresh_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    child.close()
    params = {"session_id": session_id, "run_id": f"{session_id}:1"}
    try:
        # warm the single child reader slot with a successful cold read.
        first = (await request(reader, writer, 3, "child_run_history", params))[-1]
        assert "result" in first, first
        original_refresh = ConversationStore.refresh

        def refresh_always_changes(target: ConversationStore) -> None:
            if getattr(target, "_child_history_reader", False):
                raise ObserverSnapshotChanged("child transcript changed during read")
            original_refresh(target)

        # test-contract: seam makes every warm-reader refresh report a moved
        # source, so a newest-page read serves a snapshot only when the final
        # attempt skips the refresh (allow_stale); always refreshing yields -32009.
        monkeypatch.setattr(ConversationStore, "refresh", refresh_always_changes)
        during = (await request(reader, writer, 4, "child_run_history", params))[-1]
        assert "result" in during, during
        assert [
            row["content"][0]["text"]
            for row in during["result"]["messages"]
            if row["role"] == "assistant"
        ] == ["complete"]
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_rejects_replacement_during_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("first")]))
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("second")]))
    child.close()
    params = {"session_id": session_id, "run_id": f"{session_id}:1"}
    try:
        first = (await request(reader, writer, 3, "child_run_history", params))[-1]["result"]
        transcript = server.runtime.opened.store.session_dir / "agents" / "1" / "conversation.jsonl"
        original_read_fd = ConversationStore._read_fd
        replaced = False

        def replace_after_read(fd: int, length: int) -> bytes:
            nonlocal replaced
            raw = original_read_fd(fd, length)
            if not replaced and os.fstat(fd).st_ino == transcript.stat().st_ino:
                replacement = transcript.with_suffix(".replacement")
                replacement.write_bytes(b"".join(raw.splitlines(keepends=True)[:-1]))
                os.replace(replacement, transcript)
                replaced = True
            return raw

        # test-contract: seam replaces the source between observer g0 and g1
        monkeypatch.setattr(ConversationStore, "_read_fd", staticmethod(replace_after_read))
        continuation = (
            await request(
                reader,
                writer,
                4,
                "child_run_history",
                {
                    **params,
                    "before": 1,
                    "history_validator": first["history_validator"],
                },
            )
        )[-1]
        assert continuation.get("error", {}).get("code") == -32009, continuation
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_writer_after_idle_probe_does_not_offer_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    params = {"session_id": session_id, "run_id": f"{session_id}:1"}
    probe_complete = threading.Event()
    prefix_written = threading.Event()
    release_append = threading.Event()
    append_errors: list[BaseException] = []
    prefix_bytes: list[bytes] = []
    original_probe = ConversationStore._observer_append_active
    original_write_bytes = ConversationStore._write_bytes

    def probe_then_start_writer(target: ConversationStore) -> bool:
        active = original_probe(target)
        if not active and not probe_complete.is_set():
            probe_complete.set()
            assert prefix_written.wait(10)
        return active

    def split_write(target: ConversationStore, data: bytes) -> None:
        if target is not child or prefix_written.is_set():
            original_write_bytes(target, data)
            return
        original_write_bytes(target, data[:-1])
        prefix_bytes.append((target.session_dir / "conversation.jsonl").read_bytes())
        prefix_written.set()
        assert release_append.wait(10)
        original_write_bytes(target, data[-1:])

    def append_message() -> None:
        try:
            assert probe_complete.wait(10)
            child.append_message(Message(MessageRole.ASSISTANT, [TextContent("partial")]))
        except BaseException as exc:  # noqa: BLE001 - transfer thread failure
            append_errors.append(exc)

    try:
        first = (await request(reader, writer, 3, "child_run_history", params))[-1]
        assert "result" in first, first
        transcript = child.session_dir / "conversation.jsonl"
        with transcript.open("ab") as stream:
            stream.write(b'{"type":"message"')
        # test-contract: seam starts the real writer in the old post-probe gap
        monkeypatch.setattr(
            ConversationStore, "_observer_append_active", probe_then_start_writer
        )
        # test-contract: seam pauses production append after its strict row prefix
        monkeypatch.setattr(ConversationStore, "_write_bytes", split_write)
        append_thread = threading.Thread(target=append_message)
        append_thread.start()
        try:
            during = (await request(reader, writer, 4, "child_run_history", params))[-1]
            # a validatorless first page serves the last consistent snapshot
            # instead of -32009; the active writer's torn row is never repaired.
            assert "result" in during, during
            assert [
                row["content"][0]["text"]
                for row in during["result"]["messages"]
                if row["role"] == "assistant"
            ] == ["complete"]
            assert prefix_bytes and transcript.read_bytes() == prefix_bytes[0]
        finally:
            release_append.set()
            await await_or_fail(
                asyncio.to_thread(append_thread.join, 10),
                what="post-probe production child append completion",
            )
        assert not append_thread.is_alive()
        assert not append_errors, append_errors
    finally:
        release_append.set()
        child.close()
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_child_history_unix_ignores_production_partial_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, ["child_run_history"])
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("complete")]))
    params = {"session_id": session_id, "run_id": f"{session_id}:1"}
    append_started = threading.Event()
    release_append = threading.Event()
    append_errors: list[BaseException] = []
    original_write_bytes = ConversationStore._write_bytes

    def split_write(target: ConversationStore, data: bytes) -> None:
        if target is not child or append_started.is_set():
            original_write_bytes(target, data)
            return
        original_write_bytes(target, data[:-1])
        append_started.set()
        assert release_append.wait(10)
        original_write_bytes(target, data[-1:])

    def append_message() -> None:
        try:
            child.append_message(Message(MessageRole.ASSISTANT, [TextContent("partial")]))
        except BaseException as exc:  # noqa: BLE001 - transfer thread failure
            append_errors.append(exc)

    try:
        first = (await request(reader, writer, 3, "child_run_history", params))[-1]
        assert first["result"]["messages"][-1]["content"][0]["text"] == "complete"
        # test-contract: seam pauses one production append after its strict row prefix
        monkeypatch.setattr(ConversationStore, "_write_bytes", split_write)
        append_thread = threading.Thread(target=append_message)
        append_thread.start()
        try:
            assert await await_or_fail(
                asyncio.to_thread(append_started.wait, 10),
                what="production child append prefix",
            )
            during = (await request(reader, writer, 4, "child_run_history", params))[-1]
            assert during["result"]["messages"][-1]["content"][0]["text"] == "complete"
        finally:
            release_append.set()
            await await_or_fail(
                asyncio.to_thread(append_thread.join, 10),
                what="production child append completion",
            )
        assert not append_thread.is_alive()
        assert not append_errors, append_errors
        completed = (await request(reader, writer, 5, "child_run_history", params))[-1]
        assert completed["result"]["messages"][-1]["content"][0]["text"] == "partial"
    finally:
        release_append.set()
        child.close()
        await close_serve(server, writer)


_NOW = "2026-01-01T00:00:00+00:00"


def _count_event(messages: list[dict], event: str, run_id: str) -> int:
    return sum(
        1 for row in messages if row.get("event") == event and row.get("run_id") == run_id
    )


async def _seed_nested(root: ConversationStore, session_id: str, number: str, *, lease=None):
    """Durable state a child and its grandchild write when the grandchild spawns.

    Returns ``(child, grandchild, child_run_id, grandchild_run_id)``. The child
    records the grandchild's start row; the grandchild has its own lifecycle. A
    ``lease`` ties the stores to a live serve's storage lease.
    """
    shared = {"_storage_lease": lease} if lease is not None else {"cwd": root.cwd}
    child = await asyncio.to_thread(ConversationStore, root.session_dir / "agents", session_id=number, **shared)
    child.start_agent_lifecycle(
        handle=f"{session_id}:{number}", started_at=_NOW, depth=1,
        agent_type="worker", description="child",
    )
    gc_shared = {"_storage_lease": lease} if lease is not None else {"cwd": child.cwd}
    grandchild = await asyncio.to_thread(ConversationStore, child.session_dir / "agents", session_id="1", **gc_shared)
    grandchild.start_agent_lifecycle(
        handle=f"{session_id}:{number}:1", started_at=_NOW, depth=2,
        agent_type="worker", description="grandchild",
    )
    child_id = f"{session_id}:{number}"
    grandchild_id = f"{session_id}:{number}:1"
    child.register_agent_child(
        ToolCall("call-gc", "agent", {}), child_session_path=str(grandchild.session_dir),
        description="grandchild", background=True, child_instance_id=grandchild_id,
    )
    child.append_child_run_started(grandchild_id, child_id, "worker", "grandchild", _NOW)
    return child, grandchild, child_id, grandchild_id


async def _finalize_child(root: ConversationStore, child: ConversationStore) -> None:
    await child.finalize_background_agent(
        parent_store=root, terminal_state="completed", final_result="child done",
        turns_used=0, killed_task_ids=None, killed_task_count=None,
        killed_task_ids_truncated=False,
    )


def _completion_kwargs(grandchild: ConversationStore, grandchild_id: str) -> dict:
    return {
        "child_instance_id": grandchild_id, "marker_key": grandchild_id,
        "child_session_path": str(grandchild.session_dir), "description": "grandchild",
        "status": "completed", "text": "grandchild done", "stats": None,
        "killed_task_ids": None, "killed_task_count": None,
        "killed_task_ids_truncated": False,
    }


async def _route_adopt(root, child, grandchild, child_id, grandchild_id) -> None:
    """The child ends while the grandchild still runs; finalize adopts the
    grandchild to root, so its later completion receipt lands in root."""
    owner = BackgroundAgentOwner(root)
    channel = owner.conversation_channel
    channel.register_delivery_flow(None, pending_delivery_module.StepDeliveryFlow(root))
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    done.set_result(None)
    owner.register(grandchild_id, lambda: None, done, parent_store=child, parent_instance_id=child_id)
    await _finalize_child(root, child)
    owner.adopt(grandchild_id, root, None)
    grandchild.finish_agent_lifecycle("completed", final_result="grandchild done")
    await await_or_fail(
        channel.publish_completion(**_completion_kwargs(grandchild, grandchild_id)),
        what="adopted grandchild completion to root",
    )


async def _route_sealed(root, child, grandchild, child_id, grandchild_id) -> None:
    """The grandchild completion resolves to the child store and parks; the child
    is then finalized and drained (sealed), so the parked commit is refused and
    re-resolves to root (#1245)."""
    owner = BackgroundAgentOwner(root)
    channel = owner.conversation_channel
    channel.register_delivery_flow(None, pending_delivery_module.StepDeliveryFlow(root))
    channel.register_delivery_flow(child_id, pending_delivery_module.StepDeliveryFlow(child))
    channel.register_loop(child_id)
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    done.set_result(None)
    owner.register(grandchild_id, lambda: None, done, parent_store=child, parent_instance_id=child_id)
    grandchild.finish_agent_lifecycle("completed", final_result="grandchild done")
    original_commit = child.commit_delivery_input
    parked = asyncio.Event()
    release = asyncio.Event()

    async def gated_commit(item, *, return_committed_on_cancel=False):
        parked.set()
        await await_or_fail(release.wait(), what="release parked child commit")
        return await original_commit(item, return_committed_on_cancel=return_committed_on_cancel)

    child.commit_delivery_input = gated_commit
    publish = asyncio.ensure_future(
        channel.publish_completion(**_completion_kwargs(grandchild, grandchild_id))
    )
    try:
        await await_or_fail(parked.wait(), what="grandchild completion parks in child")
        await _finalize_child(root, child)
        owner.adopt(grandchild_id, root, None)
        await await_or_fail(channel.drain_terminal(child), what="drain child store")
        release.set()
        await await_or_fail(publish, what="grandchild completion re-resolves to root")
    finally:
        release.set()
        if not publish.done():
            await await_or_fail(publish, what="grandchild completion settles")


_ROUTES = {"adopt": _route_adopt, "sealed": _route_sealed, "warm": _route_adopt}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("adopt", "sealed", "warm"))
async def test_closed_child_history_ends_every_child_started(
    tmp_path: Path, route: str
) -> None:
    """A closed child's history ends every child_started, even when the
    grandchild's completion receipt landed in an ancestor store (adopt or sealed
    re-resolve), and a warm reader reflects a grandchild finishing later."""
    opened = await asyncio.to_thread(
        lambda: SessionManager(tmp_path).create(provider="codex", model="offline", cwd=tmp_path)
    )
    root = opened.store
    session_id = root.session_id
    child, grandchild, child_id, grandchild_id = await _seed_nested(root, session_id, "1")
    try:
        await _ROUTES[route](root, child, grandchild, child_id, grandchild_id)
        # the route left the grandchild's receipt in root, not the closed child
        assert not any(
            entry.data.get("kind") == "agent_completion"
            and entry.data.get("child_instance_id") == grandchild_id
            for entry in child.agent_notifications()
        ), "setup error: closed child still holds the grandchild receipt"
    finally:
        for store in (grandchild, child, root):
            store.close()

    resumed = replay_server(
        tmp_path, "unix", lambda provider, model, home: (FakeBackend([]), model or "offline")
    )
    await resumed.start()
    controller = await connect_replay(resumed, "unix")
    observer = await connect_replay(resumed, "unix")
    try:
        features = ("child_run_history", "child_run_events", "child_runs")
        await negotiate(controller, extra=features, protocol_version="1.2")
        resume = await controller.request(2, "resume", {"session_id": session_id})
        assert "error" not in resume, resume

        history = (
            await controller.request(
                3, "child_run_history", {"session_id": session_id, "run_id": child_id}
            )
        )["result"]["messages"]
        assert _count_event(history, "child_started", grandchild_id) == 1, history
        assert _count_event(history, "child_ended", grandchild_id) == 1, history

        await negotiate(observer, observer=True, extra=features, protocol_version="1.2")
        joined = (
            await observer.request(
                3, "child_run_events", {"session_id": session_id, "run_id": child_id}
            )
        )["result"]["messages"]
        assert _count_event(joined, "child_started", grandchild_id) == 1, joined
        assert _count_event(joined, "child_ended", grandchild_id) == 1, joined

        # cache gate: a closed child whose grandchild is still running shows no
        # child_ended; a warm reader still reflects the grandchild finishing even
        # though the closed store's generation never changes. The child is seeded
        # after resume so the live runtime's crash recovery does not finalize the
        # grandchild.
        lease = resumed.runtime.opened.store._storage_lease
        pend_child, pend_gc, pend_child_id, pend_gc_id = await _seed_nested(
            resumed.runtime.opened.store, session_id, "2", lease=lease
        )
        req_id = 4
        if route == "warm":
            # read the child while it is still live (grandchild running): this
            # warms the cache with a projection that has no child_ended. Closing
            # writes no log row, so the generation never changes; the cache must
            # still be rebuilt after the child closes, or the warm reader serves
            # that stale "running forever" projection.
            live = (
                await controller.request(
                    req_id, "child_run_history", {"session_id": session_id, "run_id": pend_child_id}
                )
            )["result"]["messages"]
            assert _count_event(live, "child_ended", pend_gc_id) == 0, live
            req_id += 1
        pend_child.finish_agent_lifecycle("completed", final_result="child done")
        pend_child.close()
        pend_gc.close()

        pending = (
            await controller.request(
                req_id, "child_run_history", {"session_id": session_id, "run_id": pend_child_id}
            )
        )["result"]["messages"]
        assert _count_event(pending, "child_started", pend_gc_id) == 1, pending
        assert _count_event(pending, "child_ended", pend_gc_id) == 0, pending
        req_id += 1

        finished = ConversationStore(
            resumed.runtime.opened.store.session_dir / "agents" / "2" / "agents",
            session_id="1",
            _storage_lease=lease,
        )
        finished.finish_agent_lifecycle("completed", final_result="grandchild done")
        finished.close()

        warmed = (
            await controller.request(
                req_id, "child_run_history", {"session_id": session_id, "run_id": pend_child_id}
            )
        )["result"]["messages"]
        assert _count_event(warmed, "child_ended", pend_gc_id) == 1, warmed
    finally:
        await controller.close()
        await observer.close()
        await resumed.close()
