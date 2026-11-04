from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.server_client import request
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.server import ZetaServer, ergonomics, protocol
from zeta.server.protocol import FrameCodec, ProtocolError


def _entry(number: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"message-{number}", type="message",
        data={"message": {"role": "user", "content": [{"type": "text", "text": "text"}]}},
    )


class _HistoryStore:
    def __init__(self, entries: list[object]) -> None:
        self.entries = entries
        self._log_offset = len(entries)
        self._entry_prefix_digest = bytes([len(entries)])

    def active_branch_snapshot(self) -> list[object]:
        return list(self.entries)


def _runtime(store: object) -> SimpleNamespace:
    return SimpleNamespace(opened=SimpleNamespace(store=store))


def _backward_history(store: _HistoryStore, params: dict[str, object]) -> dict[str, object]:
    return ergonomics.history(_runtime(store), params, backward_enabled=True)


@contextlib.asynccontextmanager
async def _real_history_server(tmp_path: Path, count: int = 220) -> AsyncIterator[tuple[object, object, str, ZetaServer]]:
    server = ZetaServer(
        home=tmp_path,
        socket_path=Path("/tmp") / f"zeta-history-{tmp_path.name}.sock",
        provider="codex",
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(
        str(server.socket_path), limit=2_000_000
    )
    try:
        await request(
            reader, writer, 1, "hello",
            {"protocol_version": "1.2", "features": ["session_history_backward"]},
        )
        session = (await request(
            reader, writer, 2, "new_session", {"provider": "codex"}
        ))[-1]["result"]["session"]
        for number in range(count):
            server.runtime.opened.store.append_message(
                with_message_origin(
                    Message(MessageRole.USER, [TextContent(f"message-{number}" + "x" * 100)]),
                    MessageOrigin.USER,
                )
            )
        yield reader, writer, session["session_id"], server
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


async def _history_request(reader: object, writer: object, request_id: int, session_id: str, params: dict[str, object]) -> dict[str, object]:
    response = (await request(
        reader, writer, request_id, "session_history", {"session_id": session_id, **params}
    ))[-1]
    assert "error" not in response, response
    return response["result"]


@pytest.mark.asyncio
async def test_real_socket_backward_newest_page_and_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    async with _real_history_server(tmp_path) as (reader, writer, session_id, _server):
        first = await _history_request(reader, writer, 3, session_id, {"before": None})
        first_ids = [row["content"][0]["text"] for row in first["messages"] if row["role"] != "system"]
        assert max(int(text.split("x", 1)[0].split("-", 1)[1]) for text in first_ids) == 219
        assert first["messages"]
        validator = first["history_validator"]
        pages = [first]
        cursor = first["next_before"]
        request_id = 4
        while cursor is not None:
            page = await _history_request(
                reader, writer, request_id, session_id,
                {"before": cursor, "history_validator": validator},
            )
            pages.append(page)
            cursor = page["next_before"]
            request_id += 1
        texts = [
            row["content"][0]["text"]
            for page in reversed(pages)
            for row in page["messages"]
            if row["role"] == "user"
        ]
        assert [text.split("x", 1)[0] for text in texts] == [f"message-{number}" for number in range(220)]


@pytest.mark.asyncio
async def test_real_socket_backward_continuation_survives_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    async with _real_history_server(tmp_path, count=220) as (reader, writer, session_id, server):
        first = await _history_request(reader, writer, 3, session_id, {"before": None})
        assert [row["id"] for row in first["messages"] if row["role"] != "system"]
        assert first["messages"][0]["content"][0]["text"].startswith("message-")
        assert first["next_before"] is not None
        server.runtime.opened.store.append_message(
            with_message_origin(Message(MessageRole.USER, [TextContent("new")]), MessageOrigin.USER)
        )
        page = await _history_request(
            reader, writer, 4, session_id,
            {"before": first["next_before"], "history_validator": first["history_validator"]},
        )
        assert "new" not in {row["content"][0]["text"] for row in page["messages"] if row["role"] == "user"}


@pytest.mark.asyncio
async def test_real_socket_backward_fork_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    async with _real_history_server(tmp_path, count=220) as (reader, writer, session_id, server):
        first = await _history_request(reader, writer, 3, session_id, {"before": None})
        assert first["messages"][0]["content"][0]["text"].startswith("message-")
        user_id = server.runtime.opened.store.active_branch_snapshot()[0].id
        fork = (await request(reader, writer, 4, "fork_message", {"session_id": session_id, "message_id": user_id}))[-1]
        assert "error" not in fork
        response = (await request(reader, writer, 5, "session_history", {
            "session_id": session_id, "before": first["next_before"],
            "history_validator": first["history_validator"],
        }))[-1]
        assert response["error"]["code"] == -32009


def test_newest_page_then_full_backward_walk_has_no_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    store = _HistoryStore([_entry(i) for i in range(220)])
    first = _backward_history(store, {"before": None})
    validator = first["history_validator"]
    seen = [row["id"] for row in first["messages"] if row["role"] != "system"]
    cursor = first["next_before"]
    while cursor is not None:
        page = _backward_history(
            store, {"before": cursor, "history_validator": validator}
        )
        seen = [row["id"] for row in page["messages"] if row["role"] != "system"] + seen
        cursor = page["next_before"]
    assert seen == [f"message-{i}" for i in range(220)]


def test_backward_continuation_survives_append(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    store = _HistoryStore([_entry(i) for i in range(220)])
    first = _backward_history(store, {"before": None})
    store.entries.append(_entry(220))
    store._log_offset += 1
    assert first["next_before"] is not None
    page = _backward_history(
        store,
        {"before": first["next_before"], "history_validator": first["history_validator"]},
    )
    assert "message-220" not in {row["id"] for row in page["messages"]}


def test_backward_continuation_is_stale_after_prefix_change(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    store = _HistoryStore([_entry(i) for i in range(220)])
    first = _backward_history(store, {"before": None})
    store.entries[:] = store.entries[:5] + [_entry(99)]
    store._log_offset += 1
    assert first["next_before"] is not None
    with pytest.raises(ProtocolError) as error:
        _backward_history(
            store,
            {"before": first["next_before"], "history_validator": first["history_validator"]},
        )
    assert error.value.code == -32009
    assert error.value.data == {"code": "stale_history_snapshot"}


def test_backward_params_require_negotiation() -> None:
    store = _HistoryStore([_entry(1)])
    with pytest.raises(ProtocolError) as error:
        ergonomics.history(_runtime(store), {"before": None})
    assert error.value.code == -32602


def test_backward_rejects_cursor_past_stable_end_and_terminal_append() -> None:
    store = _HistoryStore([_entry(i) for i in range(3)])
    first = _backward_history(store, {"before": None})
    with pytest.raises(ProtocolError) as error:
        _backward_history(
            store,
            {"before": first["history_validator"]["end"] + 1,
             "history_validator": first["history_validator"]},
        )
    assert error.value.code == -32602

    store.entries.append(SimpleNamespace(
        id="approval-end", type="approval_history", data={"event": "approval_end"}
    ))
    store._log_offset += 1
    page = _backward_history(
        store,
        {"before": 1, "history_validator": first["history_validator"]},
    )
    assert "approval-end" not in {row["id"] for row in page["messages"]}


def test_backward_zero_is_empty_and_malformed_params_are_rejected() -> None:
    store = _HistoryStore([_entry(i) for i in range(3)])
    first = _backward_history(store, {"before": None})
    empty = _backward_history(
        store,
        {"before": 0, "history_validator": first["history_validator"]},
    )
    assert empty == {
        "messages": [],
        "next_before": None,
        "history_validator": first["history_validator"],
    }
    for params in (
        {"before": -1},
        {"before": 1, "offset": 0},
        {"before": 1, "history_validator": {"end": 1}},
        {"history_validator": first["history_validator"]},
    ):
        with pytest.raises(ProtocolError) as error:
            _backward_history(store, params)
        assert error.value.code == -32602


def test_backward_frame_bound_and_oversized_group_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", 4_000)
    group = SimpleNamespace(
        id="approval",
        type="approval_history",
        data={"event": "approval_request", "tool_call": {"arguments": {f"k{i}": "x" * 8000 for i in range(20)}}},
    )
    store = _HistoryStore([_entry(0), group, _entry(1)])
    first = _backward_history(store, {"before": None})
    payload = FrameCodec().response("id", first)
    assert b'"error"' not in payload
    assert len(payload) <= protocol.MAX_FRAME_BYTES
    assert any(row.get("content", [{}])[0].get("text") == ergonomics.HISTORY_GROUP_TRUNCATED for row in first["messages"])
    if first["next_before"] is not None:
        next_page = _backward_history(store, {"before": first["next_before"], "history_validator": first["history_validator"]})
        assert not ({row.get("id") for row in first["messages"]} & {row.get("id") for row in next_page["messages"]})


def test_backward_near_limit_continuation_keeps_atomic_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 1_400)
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", 1_400)
    atomic = SimpleNamespace(
        id="atomic",
        type="message",
        data={
            "message": {"role": "assistant", "content": [{"type": "text", "text": "x" * 240}]},
            "approval_requests": [{
                "request_id": "request",
                "tool_call": {"id": "call", "name": "tool", "arguments": {"value": "y" * 240}},
            }],
        },
    )
    store = _HistoryStore([_entry(i) for i in range(45)] + [atomic])
    first = _backward_history(store, {"before": None})
    first_ids = {row["id"] for row in first["messages"]}
    assert "atomic" in first_ids
    assert all(row["id"] == "atomic" for row in first["messages"] if row["id"] == "atomic")
    continuation = _backward_history(
        store,
        {"before": first["next_before"], "history_validator": first["history_validator"]},
    )
    continuation_ids = {row["id"] for row in continuation["messages"]}
    assert not first_ids & continuation_ids
    assert first_ids | continuation_ids <= {f"message-{i}" for i in range(45)} | {"atomic"}
    assert min(int(row["id"].split("-", 1)[1]) for row in continuation["messages"] if row["id"].startswith("message-")) < min(int(row["id"].split("-", 1)[1]) for row in first["messages"] if row["id"].startswith("message-"))
    for page in (first, continuation):
        payload = FrameCodec().response("id", page)
        assert b'"error"' not in payload
        assert len(payload) <= protocol.MAX_FRAME_BYTES


def test_backward_does_not_replace_forward_cache_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 4_000)
    store = _HistoryStore([_entry(i) for i in range(220)])
    before = ergonomics.history(_runtime(store), {"offset": 0})
    backward = _backward_history(store, {"before": None})
    after = ergonomics.history(_runtime(store), {"offset": 0})
    assert before == after
    assert backward["history_validator"]["end"] == 220


@pytest.mark.asyncio
async def test_real_socket_backward_near_limit_groups_never_overflow_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a group that fits the forward page but not the larger backward envelope
    must come back as a marker, not as the raw forward projection."""
    limit = 8_000
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", limit)
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", limit)
    # group sizes step through the forward/backward envelope difference, so
    # some group fits forward raw but needs a marker backward.
    sizes = range(6_000, 7_600, 10)
    async with _real_history_server(tmp_path, count=0) as (reader, writer, session_id, server):
        for number, size in enumerate(sizes):
            server.runtime.opened.store.append_message(
                with_message_origin(
                    Message(MessageRole.USER, [TextContent(f"g{number:03d}" + "x" * size)]),
                    MessageOrigin.USER,
                )
            )
        params: dict[str, object] = {"before": None}
        seen: list[str] = []
        for page in range(len(sizes) + 2):
            # the largest legal request id: 128 utf-8 bytes, escaped on the wire
            request_id = "\x00" * 124 + f"{page:04d}"
            writer.write(
                (protocol.json.dumps({
                    "jsonrpc": "2.0", "id": request_id, "method": "session_history",
                    "params": {"session_id": session_id, **params},
                }) + "\n").encode()
            )
            await writer.drain()
            while True:
                line = await asyncio.wait_for(reader.readline(), 30)
                frame = protocol.json.loads(line)
                if frame.get("id") == request_id:
                    break
            assert "error" not in frame, frame["error"]
            assert len(line) <= limit + 1
            result = frame["result"]
            for row in result["messages"]:
                text = row["content"][0]["text"] if row.get("content") else ""
                seen.append("marker" if text == ergonomics.HISTORY_GROUP_TRUNCATED else text[:4])
            if result["next_before"] is None:
                break
            params = {"before": result["next_before"], "history_validator": result["history_validator"]}
        assert result["next_before"] is None
        assert len(seen) == len(sizes)
        assert "marker" in seen


async def _raw_request(reader: object, writer: object, request_id: object, method: str, params: dict[str, object]) -> dict[str, object]:
    writer.write((protocol.json.dumps({
        "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
    }) + "\n").encode())
    await writer.drain()
    while True:
        line = await asyncio.wait_for(reader.readline(), 30)
        frame = protocol.json.loads(line)
        if frame.get("id") == request_id:
            assert len(line) <= protocol.MAX_FRAME_BYTES + 1
            return frame


@pytest.mark.asyncio
async def test_real_socket_backward_after_forward_cache_never_drops_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a newest group already cached by a forward read is reprojected for the
    real backward cursor, so it comes back raw or as a marker, never dropped."""
    limit = 8_000
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", limit)
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", limit)
    fillers = 100
    big_request_id = "\x00" * protocol.MAX_REQUEST_ID_BYTES
    async with _real_history_server(tmp_path, count=0) as (reader, writer, _first, server):
        ids = iter(range(10_000, 1_000_000))

        async def walk(size: int) -> list[str]:
            session = (await _raw_request(
                reader, writer, next(ids), "new_session", {"provider": "codex"}
            ))["result"]["session"]["session_id"]
            store = server.runtime.opened.store
            for number in range(fillers):
                store.append_message(with_message_origin(
                    Message(MessageRole.USER, [TextContent("f")]), MessageOrigin.USER,
                ))
            store.append_message(with_message_origin(
                Message(MessageRole.USER, [TextContent("big:" + "x" * size)]), MessageOrigin.USER,
            ))
            # fill the forward cache with every group first
            offset: int | None = 0
            while offset is not None:
                forward = await _raw_request(
                    reader, writer, next(ids), "session_history",
                    {"session_id": session, "offset": offset},
                )
                assert "error" not in forward, forward
                assert forward["result"]["messages"] or forward["result"]["next_offset"] is None
                offset = forward["result"]["next_offset"]
            seen: list[str] = []
            params: dict[str, object] = {"before": None}
            for _page in range(fillers + 3):
                frame = await _raw_request(
                    reader, writer, big_request_id, "session_history", {"session_id": session, **params}
                )
                assert "error" not in frame, frame["error"]
                result = frame["result"]
                if not result["messages"]:
                    assert result["next_before"] is None, result
                for row in result["messages"]:
                    text = row["content"][0]["text"]
                    seen.append("marker" if text == ergonomics.HISTORY_GROUP_TRUNCATED else text[:4])
                if result["next_before"] is None:
                    return seen
                params = {"before": result["next_before"], "history_validator": result["history_validator"]}
            raise AssertionError("backward walk did not end")

        # find the largest newest group still returned raw, then step one
        # byte at a time across the envelope difference above it.
        low, high = 6_000, 7_800
        while low < high:
            middle = (low + high + 1) // 2
            if "big:" in await walk(middle):
                low = middle
            else:
                high = middle - 1
        for size in range(low - 2, low + 6):
            seen = await walk(size)
            assert len(seen) == fillers + 1, (size, seen[:3], len(seen))
            assert {"big:", "marker"} & set(seen), (size, seen[-3:])


def test_backward_after_forward_cache_reprojects_for_real_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a newest group cached raw by a forward read, a few bytes too large for
    the backward envelope, comes back as a marker, not as an empty page."""
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 8_000)
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", 8_000)
    for size in range(6_700, 7_300):
        big = SimpleNamespace(
            id="big", type="message",
            data={"message": {"role": "user", "content": [{"type": "text", "text": "x" * size}]}},
        )
        store = _HistoryStore([_entry(i) for i in range(100)] + [big])
        # fill the forward cache with every group; each forward page must
        # make progress and the walk must return every group.
        offset: int | None = 0
        forward_ids: list[str] = []
        while offset is not None:
            page = ergonomics.history(_runtime(store), {"offset": offset})
            assert page["messages"] or page["next_offset"] is None, (size, offset)
            forward_ids.extend(row["id"] for row in page["messages"])
            offset = page["next_offset"]
        assert len(forward_ids) == 101, (size, len(forward_ids))
        seen: list[str] = []
        params: dict[str, object] = {"before": None}
        for _page in range(110):
            page = _backward_history(store, params)
            assert page["messages"] or page["next_before"] is None, (size, page["next_before"])
            seen.extend(row["id"] for row in page["messages"])
            if page["next_before"] is None:
                break
            params = {"before": page["next_before"], "history_validator": page["history_validator"]}
        assert len(seen) == 101, (size, len(seen))


def test_forward_walk_progresses_when_last_group_is_near_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the last page's next_offset is null, wider than the cursor the group was
    projected for; the group must still come back instead of an empty page."""
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 8_000)
    monkeypatch.setattr(protocol, "MAX_FRAME_BYTES", 8_000)
    for size in range(6_700, 7_300):
        big = SimpleNamespace(
            id="big", type="message",
            data={"message": {"role": "user", "content": [{"type": "text", "text": "x" * size}]}},
        )
        store = _HistoryStore([_entry(i) for i in range(100)] + [big])
        offset: int | None = 0
        seen: list[str] = []
        while offset is not None:
            page = ergonomics.history(_runtime(store), {"offset": offset})
            assert page["messages"] or page["next_offset"] is None, (size, offset)
            payload = FrameCodec().response("\x00" * protocol.MAX_REQUEST_ID_BYTES, page)
            assert b'"error"' not in payload and len(payload) <= 8_000, size
            seen.extend(row["id"] for row in page["messages"])
            offset = page["next_offset"]
        assert seen[-1] == "big", size
