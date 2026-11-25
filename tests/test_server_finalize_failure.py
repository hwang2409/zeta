"""A raise during serve turn finalization must still end the turn for the client."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.server_client import frames_until_event as _frames_until_event
from tests.support.server_client import request as _request
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
from zeta.server.delivery import TurnOutcome


class _ReplyBackend:
    @owned_completion
    async def complete(self, messages, tool_schemas):
        del messages, tool_schemas
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


def _events(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["params"]["event"] for frame in frames if "event" in frame.get("params", {})]


@pytest.mark.asyncio
async def test_finalize_error_ends_turn_and_fails_late_steer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # achilles: a raise inside finalize_turn killed the turn task, so the held
    # turn_end/agent_end were never sent and an accepted steer stayed queued.
    backend = _ReplyBackend()
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-finalize-error"),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    await asyncio.wait_for(server.start(), 10)
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), 10
    )
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        session_id = server.controllers.attached().runtime.session_id
        store = server.controllers.attached().runtime.opened.store
        finalizing = asyncio.Event()
        fail = asyncio.Event()
        original_append = store.append_turn_end_async
        seen_turns: list[str] = []

        async def failing_append(turn_id: str, reason: Any) -> None:
            if seen_turns:
                await original_append(turn_id, reason)
                return
            seen_turns.append(turn_id)
            finalizing.set()
            await await_or_fail(fail.wait(), what="finalize failure release")
            raise OSError("disk full")

        store.append_turn_end_async = failing_append
        outcomes: list[TurnOutcome] = []
        original_finalize_or_fail = server.controllers.attached().control.deliveries.finalize_or_fail

        async def recording_finalize_or_fail(*args: Any, **kwargs: Any) -> TurnOutcome:
            result = await original_finalize_or_fail(*args, **kwargs)
            outcomes.append(result)
            return result

        server.controllers.attached().control.deliveries.finalize_or_fail = recording_finalize_or_fail
        caplog.set_level(logging.ERROR, logger="zeta.server")
        await _request(reader, writer, 3, "send", {"text": "first"})
        await await_or_fail(finalizing.wait(), what="turn finalization start")

        steer = await _request(
            reader, writer, 4, "steer", {"text": "late steer", "delivery_id": "late"}
        )
        assert steer[-1]["result"] == {"accepted": True}
        status = await _request(
            reader, writer, 5, "delivery_status", {"delivery_id": "late"}
        )
        assert status[-1]["result"]["status"] == "queued"

        fail.set()
        frames = await await_or_fail(
            _frames_until_event(reader, "agent_end"), what="agent_end after failed finalize"
        )
        events = _events(frames)
        assert "turn_end" in events, events
        assert events.index("error") < events.index("turn_end"), events
        error = next(f["params"] for f in frames if f["params"].get("event") == "error")
        assert error["error"]["code"] == "server_error"

        # the fallback ends the turn as FAILED and releases it, so the next
        # send starts a fresh turn instead of the server wedging on a held turn.
        assert outcomes == [TurnOutcome.FAILED], outcomes
        assert server.controllers.attached().control._turn_task is None
        assert server.controllers.attached().runtime.state.status == "idle"

        status = await _request(
            reader, writer, 6, "delivery_status", {"delivery_id": "late"}
        )
        assert status[-1]["result"]["status"] == "dropped"
        assert store.client_delivery("late").reason == "failed"
        logged = [r for r in caplog.records if "finalization failed" in r.getMessage()]
        assert logged and session_id in logged[0].getMessage()
        assert seen_turns[0] in logged[0].getMessage()

        await _request(reader, writer, 7, "send", {"text": "next"})
        frames = await await_or_fail(
            _frames_until_event(reader, "agent_end"), what="next turn agent_end"
        )
        assert "turn_end" in _events(frames)
        assert "error" not in _events(frames)
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_finalize_and_drop_error_still_release_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # if the fallback's own steering drop also raises, _release_turn must still
    # run so the turn is freed and the next send is not wedged.
    backend = _ReplyBackend()
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-finalize-drop-error"),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    await asyncio.wait_for(server.start(), 10)
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), 10
    )
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        store = server.controllers.attached().runtime.opened.store
        finalizing = asyncio.Event()
        fail = asyncio.Event()
        original_append = store.append_turn_end_async
        original_drop = store.drop_client_steering_async
        seen_turns: list[str] = []
        dropped: list[bool] = []

        async def failing_append(turn_id: str, reason: Any) -> None:
            if seen_turns:
                await original_append(turn_id, reason)
                return
            seen_turns.append(turn_id)
            finalizing.set()
            await await_or_fail(fail.wait(), what="finalize failure release")
            raise OSError("disk full")

        async def failing_drop(reason: Any) -> Any:
            if dropped:
                return await original_drop(reason)
            dropped.append(True)
            raise OSError("drop failed")

        store.append_turn_end_async = failing_append
        store.drop_client_steering_async = failing_drop
        caplog.set_level(logging.ERROR, logger="zeta.server")
        await _request(reader, writer, 3, "send", {"text": "first"})
        await await_or_fail(finalizing.wait(), what="turn finalization start")

        steer = await _request(
            reader, writer, 4, "steer", {"text": "late steer", "delivery_id": "late"}
        )
        assert steer[-1]["result"] == {"accepted": True}

        fail.set()
        frames = await await_or_fail(
            _frames_until_event(reader, "agent_end"),
            what="agent_end after failed finalize and drop",
        )
        assert "turn_end" in _events(frames), _events(frames)
        assert dropped == [True]
        assert server.controllers.attached().control._turn_task is None
        assert server.controllers.attached().runtime.state.status == "idle"

        await _request(reader, writer, 5, "send", {"text": "next"})
        frames = await await_or_fail(
            _frames_until_event(reader, "agent_end"), what="next turn agent_end"
        )
        assert "turn_end" in _events(frames)
    finally:
        await close_serve(server, writer)
