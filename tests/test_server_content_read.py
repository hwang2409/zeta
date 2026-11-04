from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tests.test_server import _close, _connect, _request
from zeta.protocol.types import (
    ImageContent,
    Message,
    MessageOrigin,
    MessageRole,
    with_message_origin,
)
from zeta.server import content, ergonomics
from zeta.server.protocol import ProtocolError

_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


def _store(entry_id: str = "11111111-1111-4111-8111-111111111111"):
    raw = b"0123456789"
    entry = SimpleNamespace(
        id=entry_id,
        type="message",
        data={
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "mimeType": "image/png",
                        "caption": "shot",
                        "data": base64.b64encode(raw).decode(),
                    }
                ],
            }
        },
    )
    store = SimpleNamespace()
    projection = SimpleNamespace(entry_index=MappingProxyType({entry_id: entry}))
    return store, projection, raw


def test_row_content_read_returns_bounded_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, projection, raw = _store()
    monkeypatch.setattr(content, "history_projection", lambda store: projection)
    handle = content.row_handle(
        next(iter(projection.entry_index)), "message.content", 0
    )
    reply = content.read(store, handle, offset=2, limit=4)
    assert base64.b64decode(reply["data"]) == raw[2:6]
    assert reply["next_offset"] == 6
    assert reply["eof"] is False


def test_malformed_and_foreign_handles_have_the_same_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, projection, _ = _store()
    monkeypatch.setattr(content, "history_projection", lambda store: projection)
    errors = []
    for handle in ("not-a-row", "r1.22222222-2222-4222-8222-222222222222.mc.0"):
        with pytest.raises(ProtocolError) as raised:
            content.read(store, handle)
        errors.append((raised.value.code, raised.value.data))
    assert errors[0] == errors[1] == (-32012, {"code": "content_not_found"})


def _error(reply: pytest.ExceptionInfo[ProtocolError]) -> tuple[object, ...]:
    error = reply.value
    return error.code, error.message, error.data


def test_forged_locators_have_one_unknown_handle_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    store, projection, _ = _store()
    monkeypatch.setattr(content, "history_projection", lambda store: projection)
    entry_id = next(iter(projection.entry_index))
    handles = (f"r1.{entry_id}.mc.9", f"r1.{entry_id}.tc.0", "r1.22222222-2222-4222-8222-222222222222.mc.0", f"r1.{entry_id}.mc.-1", f"r1.{entry_id}.mc.0.extra")
    errors = []
    for handle in handles:
        with pytest.raises(ProtocolError) as raised:
            content.read(store, handle)
        errors.append(_error(raised))
    assert all(error == errors[0] for error in errors)
    assert errors[0] == (-32012, "content not found", {"code": "content_not_found"})


def test_repeated_chunks_do_not_scan_a_large_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    store, projection, raw = _store()
    scanned = 0

    class NoScan:
        def __init__(self, entries: dict[str, object]) -> None:
            self._entries = entries

        def get(self, key: str, default: object = None) -> object:
            return self._entries.get(key, default)

        def items(self):
            nonlocal scanned
            scanned += 1
            raise AssertionError("content resolution scanned the projection")

    entries = {f"entry-{index}": object() for index in range(9999)}
    entries[next(iter(projection.entry_index))] = projection.entry_index[next(iter(projection.entry_index))]
    large = SimpleNamespace(entry_index=NoScan(entries))
    monkeypatch.setattr(content, "history_projection", lambda store: large)
    handle = content.row_handle(next(iter(projection.entry_index)), "message.content", 0)
    for offset in range(0, len(raw), 1):
        assert base64.b64decode(content.read(store, handle, offset=offset, limit=1)["data"]) == raw[offset : offset + 1]
    assert scanned == 0


def test_legacy_path_only_block_is_not_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_id = "11111111-1111-4111-8111-111111111111"
    entry = SimpleNamespace(id=entry_id, type="message", data={"message": {"role": "user", "content": [{"type": "image", "path": "/secret/image.png"}]}})
    monkeypatch.setattr(content, "history_projection", lambda store: SimpleNamespace(entry_index={entry_id: entry}))
    opened = False

    def fail_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("legacy image path was opened")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(ProtocolError) as raised:
        content.read(object(), content.row_handle(entry_id, "message.content", 0))
    assert _error(raised) == (-32012, "content not found", {"code": "content_not_found"})
    assert opened is False


async def _history_with_image(reader, writer, *, request_id: int = 2, session_id: str):
    result = (await _request(reader, writer, request_id, "session_history", {"session_id": session_id}))[-1]
    assert "result" in result
    rows = result["result"]["messages"]
    attachments = [block for row in rows for block in row.get("content", []) if block.get("type") == "attachment"]
    tool_blocks = [block for row in rows for block in (row.get("tool_result") or {}).get("content_blocks", [])]
    return result, attachments, tool_blocks


@pytest.mark.asyncio
async def test_handle_survives_disconnect_and_reattach(tmp_path: Path) -> None:
    from zeta.server import ZetaServer
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    raw = _PNG
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_history_fidelity", "content_read"]})
        session = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
        store = server.runtime.opened.store
        store.append_message(with_message_origin(Message(MessageRole.USER, [ImageContent(base64.b64encode(raw).decode(), "image/png")]), MessageOrigin.USER))
        _, attachments, _ = await _history_with_image(reader, writer, request_id=4, session_id=session["session_id"])
        handle = attachments[0]["handle"]
        writer.close(); await writer.wait_closed(); await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 3, "hello", {"protocol_version": "1.2", "features": ["session_history_fidelity", "content_read"]})
        reply = (await _request(reader, writer, 4, "content_read", {"session_id": session["session_id"], "handle": handle}))[-1]
        assert base64.b64decode(reply["result"]["data"]) == raw
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_handle_survives_server_restart(tmp_path: Path) -> None:
    from zeta.server import ZetaServer
    raw = _PNG
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_history_fidelity", "content_read"]})
    session = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
    store = server.runtime.opened.store
    store.append_message(with_message_origin(Message(MessageRole.USER, [ImageContent(base64.b64encode(raw).decode(), "image/png")]), MessageOrigin.USER))
    _, attachments, _ = await _history_with_image(reader, writer, request_id=4, session_id=session["session_id"])
    handle = attachments[0]["handle"]
    await _close(server, writer)
    restarted = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(restarted)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_history_fidelity", "content_read"]})
        await restarted.runtime.resume_session(session["session_id"])
        await _request(reader, writer, 2, "session_history", {"session_id": session["session_id"]})
        reply = (await _request(reader, writer, 3, "content_read", {"session_id": session["session_id"], "handle": handle}))[-1]
        assert base64.b64decode(reply["result"]["data"]) == raw
    finally:
        await _close(restarted, writer)


@pytest.mark.asyncio
async def test_handle_is_scoped_to_session_for_controller_and_observer(tmp_path: Path) -> None:
    from zeta.server import ZetaServer
    first = ZetaServer(home=tmp_path / "a", port=0, provider="codex")
    reader_a, writer_a = await _connect(first)
    raw = _PNG
    await _request(reader_a, writer_a, 1, "hello", {"protocol_version": "1.2", "features": ["session_history_fidelity", "content_read"]})
    session = (await _request(reader_a, writer_a, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
    store = first.runtime.opened.store
    store.append_message(with_message_origin(Message(MessageRole.USER, [ImageContent(base64.b64encode(raw).decode(), "image/png")]), MessageOrigin.USER))
    _, attachments, _ = await _history_with_image(reader_a, writer_a, request_id=4, session_id=session["session_id"])
    handle = attachments[0]["handle"]
    second = ZetaServer(home=tmp_path / "b", port=0, provider="codex")
    reader_b, writer_b = await _connect(second)
    try:
        await _request(reader_b, writer_b, 1, "hello", {"protocol_version": "1.2", "features": ["content_read"]})
        session_b = (await _request(reader_b, writer_b, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
        controller = (await _request(reader_b, writer_b, 3, "content_read", {"session_id": session_b["session_id"], "handle": handle}))[-1]["error"]
        writer_b.close(); await writer_b.wait_closed(); await asyncio.sleep(0.05)
        observer_reader, observer_writer = await asyncio.open_connection("127.0.0.1", second.port)
        try:
            await _request(observer_reader, observer_writer, 3, "hello", {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers", "content_read"]})
            observer = (await _request(observer_reader, observer_writer, 4, "content_read", {"session_id": session_b["session_id"], "handle": handle}))[-1]["error"]
        finally:
            observer_writer.close(); await observer_writer.wait_closed()
        assert controller == observer == {"code": -32012, "message": "content not found", "data": {"code": "content_not_found"}}
    finally:
        await _close(first, writer_a)
        await second.close()


def test_history_stubs_for_user_and_tool_images_read_original_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    user_raw = _PNG
    tool_raw = _PNG[::-1]  # legacy rows may carry any inline bytes; the resolver is byte exact.
    user_id = "11111111-1111-4111-8111-111111111111"
    tool_id = "22222222-2222-4222-8222-222222222222"
    entries = [
        SimpleNamespace(id=user_id, type="message", data={"message": {"role": "user", "content": [{"type": "image", "data": base64.b64encode(user_raw).decode(), "mimeType": "image/png"}]}}),
        SimpleNamespace(id=tool_id, type="message", data={"message": {"role": "assistant", "content": [], "tool_result": {"content": "", "content_blocks": [{"type": "image", "data": base64.b64encode(tool_raw).decode(), "mimeType": "image/png"}]}}}),
    ]
    projection = SimpleNamespace(entry_index={entry.id: entry for entry in entries})
    monkeypatch.setattr(content, "history_projection", lambda store: projection)
    for entry, field, expected in ((entries[0], "message.content", user_raw), (entries[1], "message.tool_result.content_blocks", tool_raw)):
        rows = ergonomics._history_entry_rows(entry, {}, fidelity=True, content_read=True)
        block = rows[0]["content"][0] if field == "message.content" else rows[0]["tool_result"]["content_blocks"][0]
        assert block["handle"]
        assert base64.b64decode(content.read(object(), block["handle"], limit=3)["data"]) == expected[:3]
        assert block.get("data_omitted", False) is (field != "message.content")
