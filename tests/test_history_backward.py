from __future__ import annotations

from types import SimpleNamespace

import pytest

from zeta.server import ergonomics
from zeta.server.protocol import ProtocolError


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
        {"before": first["next_before"], "history_validator": first["history_validator"]},
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
