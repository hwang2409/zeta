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
    ToolUseContent,
    with_message_origin,
)
from zeta.protocol.types import ToolCall
from zeta.server import ZetaServer, ergonomics
from zeta.server.protocol import MAX_FRAME_BYTES, FrameCodec, ProtocolError


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
        hello = (await request(
            reader, writer, 1, "hello",
            {"protocol_version": "1.2", "features": ["session_history_backward"]},
        ))[-1]["result"]
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
    group = SimpleNamespace(
        id="approval",
        type="approval_history",
        data={"event": "approval_request", "tool_call": {"arguments": {f"k{i}": "x" * 8000 for i in range(20)}}},
    )
    store = _HistoryStore([_entry(0), group, _entry(1)])
    first = _backward_history(store, {"before": None})
    payload = FrameCodec().response("id", first)
    assert len(payload) <= MAX_FRAME_BYTES
    assert any(row.get("content", [{}])[0].get("text") == ergonomics.HISTORY_GROUP_TRUNCATED for row in first["messages"])
    if first["next_before"] is not None:
        next_page = _backward_history(store, {"before": first["next_before"], "history_validator": first["history_validator"]})
        assert not ({row.get("id") for row in first["messages"]} & {row.get("id") for row in next_page["messages"]})


def test_backward_near_limit_continuation_keeps_atomic_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ergonomics, "MAX_FRAME_BYTES", 1_400)
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
    assert len(FrameCodec().response("id", first)) <= MAX_FRAME_BYTES
    assert len(FrameCodec().response("id", continuation)) <= MAX_FRAME_BYTES


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
