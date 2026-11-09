"""The optional delivery session_id guard over the real serve socket."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.server_client import event as _event
from tests.support.server_client import read_until as _read_until
from tests.support.server_client import send as _send
from tests.support.sockets import short_socket_path as _socket_path
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)
FEATURES = [
    "user_message_event",
    "delivery_id",
    "send_images_delivery_id",
    "delivery_session_guard",
]


class BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @owned_completion
    async def complete(self, messages, tool_schemas):
        del messages, tool_schemas
        self.started.set()
        await asyncio.wait_for(self.release.wait(), 10)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


async def _call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: int,
    method: str,
    params: dict[str, object],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the frames sent before the response, and the response."""
    await _send(reader, writer, request_id, method, params)
    frames = await _read_until(
        reader,
        lambda frame: frame.get("id") == request_id,
        what=f"response to {method}",
    )
    return frames[:-1], frames[-1]


def _echoes(frames: list[dict[str, Any]]) -> list[str]:
    return [
        frame["params"]["text"]
        for frame in frames
        if frame.get("params", {}).get("event") == "user_message"
    ]


def _params(method: str, session_id: str, delivery_id: str) -> dict[str, object]:
    params: dict[str, object] = {
        "text": "hello",
        "session_id": session_id,
        "delivery_id": delivery_id,
    }
    if method == "send_images":
        params["images"] = [
            {
                "name": "shot.png",
                "mime_type": "image/png",
                "data": base64.b64encode(PNG).decode(),
            }
        ]
    return params


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "steer", "send_images"])
async def test_mismatched_session_id_delivers_nothing_before_delivery_id_lookup(
    tmp_path: Path, method: str
) -> None:
    backend = BlockingBackend()
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    await asyncio.wait_for(server.start(), 10)
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), 10
    )
    try:
        _, hello = await _call(
            reader, writer, 1, "hello", {"protocol_version": "1.2", "features": FEATURES}
        )
        _, created = await _call(reader, writer, 2, "new_session", {"provider": "codex"})
        attached = created["result"]["session"]["session_id"]
        if method == "steer":
            await _call(reader, writer, 3, "send", {"text": "start"})
            await await_or_fail(backend.started.wait(), what="turn start")
        store = server.runtime.opened.store
        rows_before = [entry.id for entry in store.entries]
        mismatch = {
            "code": -32003,
            "message": "session is not active",
            "data": {
                "code": "session_mismatch",
                "attached_session_id": attached,
                "session_id": "other-session",
            },
        }

        # mismatch: exact error, no echo, no row, no delivery record
        before, rejected = await _call(
            reader, writer, 4, method, _params(method, "other-session", "g1")
        )
        assert rejected.get("error") == mismatch, rejected
        assert _echoes(before) == []
        assert [entry.id for entry in store.entries] == rows_before
        _, status = await _call(reader, writer, 5, "delivery_status", {"delivery_id": "g1"})
        assert status["result"] == {"delivery_id": "g1", "status": "unknown"}

        # match: delivered as a first delivery, not a duplicate of the rejection
        before, accepted = await _call(
            reader, writer, 6, method, _params(method, attached, "g1")
        )
        assert accepted["result"]["accepted"] is True
        assert "duplicate" not in accepted["result"]
        assert _echoes(before) == ["hello"]
        _, status = await _call(reader, writer, 7, "delivery_status", {"delivery_id": "g1"})
        expected = "queued" if method == "steer" else "delivered"
        assert status["result"]["status"] == expected

        # a retry of the delivered id aimed at another session is still rejected
        rows_after_match = [entry.id for entry in store.entries]
        before, retried = await _call(
            reader, writer, 8, method, _params(method, "other-session", "g1")
        )
        assert retried.get("error") == mismatch, retried
        assert _echoes(before) == []
        assert [entry.id for entry in store.entries] == rows_after_match

        backend.release.set()
        await _event(reader, "agent_end")
        # the guard is advertised so a client can tell it from an older server
        assert sorted(hello["result"]["capabilities"]["features"]) == sorted(FEATURES)
    finally:
        backend.release.set()
        await close_serve(server, writer)
