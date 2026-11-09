"""real transport checks for child run history negotiation and validation."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent
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
    server, controller_reader, controller_writer, session_id = await _open(
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
            assert during.get("error", {}).get("data") == {
                "code": "stale_history_snapshot"
            }, during
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
