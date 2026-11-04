from __future__ import annotations

import base64
from types import MappingProxyType, SimpleNamespace

import pytest

from zeta.server import content
from zeta.server.protocol import ProtocolError


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
