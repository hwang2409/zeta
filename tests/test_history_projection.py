from __future__ import annotations

import threading
from types import SimpleNamespace

from zeta.server import ergonomics
from zeta.server.protocol import MAX_REQUEST_ID_BYTES, FrameCodec


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


def test_session_history_pages_identical_to_previous_projection() -> None:
    entries = tuple(_entry(i) for i in range(7))

    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            return entries

    runtime = _runtime(Store())
    expected = []
    for entry in entries:
        expected.extend(ergonomics._history_entry_rows(entry, {}))
    actual = []
    offset = 0
    while True:
        result = ergonomics.history(runtime, {"offset": offset})
        actual.extend(result["messages"])
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert actual == expected


def test_session_history_cache_invalidates_on_append() -> None:
    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"
        def __init__(self):
            self.entries = [_entry(1)]

        def active_branch_snapshot(self):
            return tuple(self.entries)

    store = Store()
    runtime = _runtime(store)
    first = ergonomics.history(runtime, {})
    assert [row["id"] for row in first["messages"]] == ["message-1"]
    store.entries.append(_entry(2))
    store._log_offset = 2
    store._entry_prefix_digest = b"two"
    second = ergonomics.history(runtime, {"offset": 1})
    assert [row["id"] for row in second["messages"]] == ["message-2"]


def test_session_history_page_bounded_by_frame_bytes() -> None:
    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            return tuple(_entry(i, "x" * 8_000) for i in range(300))

    runtime = _runtime(Store())
    codec = FrameCodec()
    offset = 0
    seen = 0
    while True:
        result = ergonomics.history(runtime, {"offset": offset})
        assert codec.response_fits("\x00" * MAX_REQUEST_ID_BYTES, result)
        seen += len(result["messages"])
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert seen == 300


def test_session_history_projection_built_off_event_loop() -> None:
    class Store:
        _log_offset = 1
        _entry_prefix_digest = b"one"

        def active_branch_snapshot(self):
            assert threading.current_thread() is not threading.main_thread()
            return (_entry(1),)

    error: list[BaseException] = []

    def read() -> None:
        try:
            ergonomics.history(_runtime(Store()), {})
        except AssertionError as exc:  # pragma: no cover - assertion below reports it
            error.append(exc)

    worker = threading.Thread(target=read)
    worker.start()
    worker.join()
    assert not error
