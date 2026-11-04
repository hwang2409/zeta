from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from zeta.server import ergonomics
from zeta.server.connections import ConnectionRegistry
from zeta.server.protocol import MAX_FRAME_BYTES, MAX_REQUEST_ID_BYTES, FrameCodec


def _entry(number: int, text: str = "text") -> SimpleNamespace:
    return SimpleNamespace(
        id=f"message-{number}",
        type="message",
        data={
            "message": {
                "role": "user" if number % 2 else "assistant",
                "content": [{"type": "text", "text": text}],
            }
        },
    )


def _runtime(store: object) -> SimpleNamespace:
    return SimpleNamespace(opened=SimpleNamespace(store=store))


def test_session_history_full_paging_visits_each_entry_once() -> None:
    entries = tuple(_entry(i) for i in range(300))
    visits = 0

    class Snapshot(tuple):
        def __iter__(self):
            nonlocal visits
            for entry in super().__iter__():
                visits += 1
                yield entry

    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            return Snapshot(entries)

    runtime = _runtime(Store())
    offset = 0
    while True:
        result = ergonomics.history(runtime, {"offset": offset})
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert visits == len(entries)


def test_session_history_page_construction_is_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(_entry(i) for i in range(128))
    monkeypatch.setattr(ergonomics, "HISTORY_PAGE_ENTRIES", len(entries))
    serialized_rows = 0
    original = FrameCodec._json_bytes

    def counting_json_bytes(value: object) -> bytes:
        nonlocal serialized_rows
        if isinstance(value, list):
            serialized_rows += len(value)
        elif isinstance(value, dict):
            result = value.get("result")
            if isinstance(result, dict) and isinstance(result.get("messages"), list):
                serialized_rows += len(result["messages"])
        return original(value)

    monkeypatch.setattr(FrameCodec, "_json_bytes", staticmethod(counting_json_bytes))

    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            return entries

    result = ergonomics.history(_runtime(Store()), {})
    assert len(result["messages"]) == len(entries)
    assert serialized_rows <= len(entries)


def test_session_history_exact_size_and_real_frame_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = "\x00" * MAX_REQUEST_ID_BYTES
    codec = FrameCodec()
    fixtures = [
        [[{"id": "one", "text": "plain"}]],
        [[{"id": "日本語", "text": "naïve café"}], [{"id": "two", "text": "🙂"}]],
        [[{"id": "large", "text": "x" * 250_000}]],
    ]
    serialized_size = getattr(
        codec, "serialized_size", lambda value: len(codec._json_bytes(value))
    )
    response_size = getattr(
        codec,
        "response_size",
        lambda request_id, result: len(
            codec._json_bytes({"jsonrpc": "2.0", "id": request_id, "result": result})
        ),
    )
    frame_size = getattr(
        ergonomics,
        "_history_frame_size",
        lambda codec, request_id, *, messages_body_size, next_offset: response_size(
            request_id, {"messages": [], "next_offset": next_offset}
        )
        + messages_body_size,
    )
    for groups in fixtures:
        messages: list[dict[str, object]] = []
        messages_body_size = 0
        for group in groups:
            group_body_size = serialized_size(group) - len(b"[]\n")
            if messages and group:
                messages_body_size += len(b",")
            messages_body_size += group_body_size
            messages.extend(group)
        page = {"messages": messages, "next_offset": 123}
        computed = frame_size(
            codec,
            request_id,
            messages_body_size=messages_body_size,
            next_offset=123,
        )
        assert computed == len(
            codec._json_bytes(
                {"jsonrpc": "2.0", "id": request_id, "result": page}
            )
        )

    entries = tuple(_entry(i, "é" * 8_000) for i in range(1_200))

    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            return entries

    # The page builder must use incremental accounting, not whole-page probes.
    whole_page_probes = 0
    original_response_fits = FrameCodec.response_fits

    def count_response_fits(*args: object, **kwargs: object) -> bool:
        nonlocal whole_page_probes
        whole_page_probes += 1
        return original_response_fits(*args, **kwargs)

    monkeypatch.setattr(FrameCodec, "response_fits", count_response_fits)
    offset = 0
    runtime = _runtime(Store())
    while True:
        result = ergonomics.history(runtime, {"offset": offset})
        encoded_size = response_size(request_id, result)
        assert encoded_size <= MAX_FRAME_BYTES
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert whole_page_probes == 0


def test_session_history_matches_fixed_target_at_every_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(_entry(i, f"row {i} 日本語") for i in range(19))
    snapshot_calls = 0

    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return entries

    runtime = _runtime(Store())
    target = 8
    monkeypatch.setattr(ergonomics, "HISTORY_PAGE_ENTRIES", target)
    for offset in range(len(entries) + 1):
        result = ergonomics.history(runtime, {"offset": offset})
        stop = min(offset + target, len(entries))
        expected_rows = [
            row
            for entry in entries[offset:stop]
            for row in ergonomics._history_entry_rows(entry, {})
        ]
        expected = {
            "messages": expected_rows,
            "next_offset": stop if stop < len(entries) else None,
        }
        assert result == expected
    assert snapshot_calls == 1


@pytest.mark.asyncio
async def test_session_history_projection_constructed_once_off_event_loop() -> None:
    snapshot_threads: list[threading.Thread] = []

    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            snapshot_threads.append(threading.current_thread())
            return tuple(_entry(i) for i in range(300))

    runtime = _runtime(Store())
    registry = ConnectionRegistry(asyncio.get_running_loop())
    try:
        offset = 0
        while True:
            result = await registry.run_blocking(
                lambda offset=offset: ergonomics.history(runtime, {"offset": offset})
            )
            if result["next_offset"] is None:
                break
            offset = result["next_offset"]
    finally:
        await registry.aclose()

    assert len(snapshot_threads) == 1
    assert snapshot_threads[0] is not threading.main_thread()
