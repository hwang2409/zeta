"""Multi-controller `resume` table: each row resumes through the real serve
socket and checks the session keeps one resident record and runs its next turn."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.branch_reads import branch_messages
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_in_flight_replay import HOLD, GatedBackend
from tests.test_server_session_park import _until
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer

MULTI = ("multi_controller", "new_session_settings")
ATTACH = (*MULTI, "resume_attach")
SETTINGS = {"provider": "codex", "model": "gpt-5.6-sol"}


def _reply(text: str) -> tuple[list[object], list[object]]:
    return [TextContent(text)], [TextContent(text)]


def _records(server: ZetaServer, session_id: str) -> list[object]:
    return [
        record for record in server.controllers.records()
        if not record.closing and record.runtime.opened is not None
        and record.runtime.session_id == session_id
    ]


async def _turn(server: ZetaServer, client: ReplayClient, request_id: int, session_id: str) -> None:
    """Send one turn; it must end and its user message must be stored."""
    sent = await client.request(request_id, "send", {"text": f"after resume {request_id}"})
    assert sent.get("result", {}).get("accepted") is True, sent
    while (end := await await_or_fail(client.event(match={"session_id": session_id}), what="turn end"))[
        "event"
    ] not in ("turn_end", "error"):
        pass
    assert end["event"] == "turn_end", f"the turn after resume failed: {end}"
    (record,) = _records(server, session_id)
    texts = [
        block.text for message in branch_messages(record.runtime.opened.store)
        for block in message.content if isinstance(block, TextContent)
    ]
    assert f"after resume {request_id}" in texts, "the sent message must be stored"


async def _legacy_resident(server: ZetaServer, transport: Transport, *, unhealthy: bool = False,
                           monkeypatch: pytest.MonkeyPatch) -> str:
    """Leave an idle legacy session resident (optionally unhealthy) and the mode empty."""
    legacy = await connect_replay(server, transport)
    try:
        assert "result" in await negotiate(legacy, replay=False, extra=("new_session_settings",))
        mode = "ask" if unhealthy else "allow"
        created = await legacy.request(2, "new_session", {**SETTINGS, "approval_mode": mode})
        session_id = created["result"]["session"]["session_id"]
        if unhealthy:
            control = server.controllers.attached().control
            assert "result" in await legacy.request(3, "send", {"text": "write"})
            await _until(control.has_pending_approvals, "the write approval")
            status = await legacy.request(4, "status", {})
            request_id = status["result"]["pending_approvals"][0]["request_id"]
            with monkeypatch.context() as patch:
                patch.setattr(
                    ConversationStore, "resolve_approval",
                    lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
                )
                failed = await legacy.request(5, "approve", {"request_id": request_id})
            assert failed["error"]["code"] == -32000, failed
            assert control.unhealthy()
            await _until(lambda: not control.state_change_busy(), "the failed turn ends")
    finally:
        await legacy.close()
    await _until(lambda: server.controllers.controller_mode == "empty", "legacy leaves")
    return session_id


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize(
    "row",
    ("disk", "disk_attach", "legacy_idle", "legacy_idle_attach", "own_busy", "legacy_unhealthy"),
)
@pytest.mark.asyncio
async def test_multi_resume_keeps_one_record_and_runs_turns(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = ToolUseContent(ToolCall("w-1", "write", {"path": str(tmp_path / "out.txt"), "content": "x"}))
    replies = {
        "own_busy": [([TextContent("held"), HOLD], [TextContent("held")]), _reply("next")],
        "legacy_unhealthy": [([TextContent("writing")], [TextContent("writing"), write]), _reply("next")],
    }.get(row, [_reply("first"), _reply("next")])
    backend = GatedBackend(replies)
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start resume server")
    client = None
    try:
        legacy_row = row.startswith("legacy")
        if legacy_row:
            target = await _legacy_resident(
                server, transport, unhealthy=row == "legacy_unhealthy", monkeypatch=monkeypatch
            )
        client = await connect_replay(server, transport)
        features = ATTACH if row.endswith("attach") else MULTI
        assert "result" in await negotiate(client, replay=False, extra=features)
        if not legacy_row:
            target = (await client.request(2, "new_session", SETTINGS))["result"]["session"]["session_id"]
            if row == "own_busy":
                assert "result" in await client.request(3, "send", {"text": "hold"})
                await await_or_fail(backend.held[0].wait(), what="the held turn")
            if row != "own_busy":
                assert "result" in await client.request(4, "new_session", SETTINGS)
                await server.controllers.wait_closed(target)

        resumed = await client.request(5, "resume", {"session_id": target})

        if row == "own_busy":
            # without resume_attach, resume of the busy attached session is refused, not reopened
            assert resumed.get("error", {}).get("code") == -32004, resumed
            assert len(_records(server, target)) == 1, "a refused resume must not open a record"
            backend.open_all()
            return
        assert resumed["result"]["session"]["session_id"] == target, resumed
        assert len(_records(server, target)) == 1, "one resident record per session id"
        if row == "legacy_unhealthy":
            (record,) = _records(server, target)
            assert not record.control.unhealthy(), "resume must reopen the unhealthy session"
        await _turn(server, client, 6, target)
    finally:
        backend.open_all()
        if client is not None:
            await client.close()
        await await_or_fail(server.close(), what="close resume server")
