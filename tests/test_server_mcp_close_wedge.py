"""real serve-socket coverage: a resume that re-binds a bound MCP server whose
client close() ignores cancellation must still reply, and must not pin the
controller slot (RCA resume-stuck, achilles CI run 43862802409)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import zeta.mcp.mount as mount_module
import zeta.mcp.server_actor as server_actor_module
from tests.support.owned import await_or_fail
from tests.support.server_client import TIMEOUT
from tests.test_server import _connect, _event, _request
from tests.test_server_mcp_bindings import (
    LocalMCP,
    _wait_for_controller_release,
    allow_local_http,
)
from tests.test_server_turn_reconnect import HeldBackend, _hello, _server
from zeta.protocol.types import TextContent

FEATURES = [
    "approval_reconnect",
    "turn_reconnect",
    "served_turn_end",
    "delivery_id",
    "new_session_settings",
    "session_mcp_bindings",
    "resume_attach",
]
BINDING = {"server": "page", "headers": {"Authorization": "Bearer page"}}


def _wrap_uncancellable_close(
    monkeypatch: pytest.MonkeyPatch, close_gate: asyncio.Event
) -> None:
    """Every bound client gets a close() that blocks on close_gate and ignores
    cancellation until it is set -- the anyio/httpx aclose against a restarting
    peer that the RCA names."""
    real_build = mount_module._build_client

    def build(config):
        client = real_build(config)
        original_close = client.close

        async def blocked_close() -> None:
            while not close_gate.is_set():
                try:
                    await close_gate.wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None:
                        task.uncancel()
            await original_close()

        client.close = blocked_close
        return client

    monkeypatch.setattr(mount_module, "_build_client", build)


@pytest.mark.asyncio
async def test_resume_rebind_replies_though_bound_close_wedges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    # a small, injected close bound: the fix must hold it even though the inner
    # safe_close wait_for cannot, because the client close ignores cancellation.
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0.2)
    close_gate = asyncio.Event()
    _wrap_uncancellable_close(monkeypatch, close_gate)

    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "page": {
                        "transport": "streamable-http",
                        "url": f"http://127.0.0.1:{port}",
                    }
                }
            }
        )
    )
    backend = HeldBackend([([TextContent("warm")], False)])
    server = _server(tmp_path, backend)

    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _hello(reader, writer, 1, FEATURES)
        created = (
            await _request(
                reader,
                writer,
                2,
                "new_session",
                {
                    "provider": "codex",
                    "model": "gpt-5.6-sol",
                    "approval_mode": "allow",
                    "mcp_bindings": [BINDING],
                },
            )
        )[-1]["result"]
        session_id = created["session"]["session_id"]
        # one turn so the MCP mount is live and the actor holds a bound client.
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "agent_end")

        # the second resume re-binds page: it suspends and closes the old bound
        # client. on base, _handle_suspend awaits that close inline, the actor
        # loop wedges, and this resume never replies (fails at the wait below).
        resume = await await_or_fail(
            _request(
                reader,
                writer,
                4,
                "resume",
                {"session_id": session_id, "mcp_bindings": [BINDING]},
            ),
            what="client A second resume reply while the bound close is wedged",
        )
        assert "result" in resume[-1], resume[-1]
        assert not close_gate.is_set()

        # A disconnects; a fresh controller must attach, not get -32001.
        writer.close()
        await writer.wait_closed()
        await await_or_fail(
            _wait_for_controller_release(server),
            what="controller slot release after A disconnects",
        )
        reader2, writer2 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        hello2 = await await_or_fail(
            _request(
                reader2,
                writer2,
                1,
                "hello",
                {"protocol_version": "1.2", "features": FEATURES},
            ),
            what="fresh controller hello after the wedged close",
        )
        assert "result" in hello2[-1], hello2[-1]
        assert not close_gate.is_set()
    finally:
        close_gate.set()
        for connection in (writer2, writer):
            if connection is not None and not connection.is_closing():
                connection.close()
                await connection.wait_closed()
        await server.close()
        await mcp.close()
