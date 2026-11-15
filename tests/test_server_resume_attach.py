"""real serve-socket coverage for `resume` attaching to the active busy session."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import TIMEOUT, _connect, _event, _request
from tests.test_server_mcp_bindings import (
    LocalMCP,
    _wait_for_controller_release,
    allow_local_http,
)
from zeta.protocol.types import TextContent, ToolCall
from zeta.server import ZetaServer

BUSY = -32004
INVALID_PARAMS = -32602


async def _open(server: ZetaServer, request_id: int, features: list[str]):
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
    )
    hello = await _request(reader, writer, request_id, "hello", {
        "protocol_version": "1.2", "features": features
    })
    assert "error" not in hello[-1]
    return reader, writer


async def _close(*writers) -> None:
    for writer in writers:
        if writer is not None and not writer.is_closing():
            writer.close()
            await writer.wait_closed()


def _error_code(frames: list[dict]) -> int | None:
    return frames[-1].get("error", {}).get("code")


@pytest.mark.asyncio
async def test_resume_attach_keeps_approval_and_rebinds_mcp_for_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    config = {"transport": "streamable-http", "url": f"http://127.0.0.1:{port}"}
    (tmp_path / "mcp.json").write_text(json.dumps({
        "servers": {"page": config, "other": config}
    }))
    backend = FakeBackend([
        ScriptedTurn([TextContent("mounted")]),
        ScriptedTurn(tool_calls=[ToolCall("page-1", "page__echo", {})]),
        ScriptedTurn([TextContent("done")]),
        ScriptedTurn(tool_calls=[ToolCall("page-2", "page__echo", {})]),
        ScriptedTurn([TextContent("done again")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    features = [
        "approval_reconnect", "new_session_settings", "session_mcp_bindings",
        "resume_attach",
    ]
    old = {"server": "page", "headers": {"Authorization": "Bearer old-secret"}}
    new = {"server": "page", "headers": {"Authorization": "Bearer new-secret"}}
    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        other_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
        }))[-1]["result"]["session"]["session_id"]
        session_id = (await _request(reader, writer, 3, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
            "approval_mode": "ask", "mcp_bindings": [old],
        }))[-1]["result"]["session"]["session_id"]
        await _request(reader, writer, 4, "send", {"text": "warm"})
        await _event(reader, "agent_end")
        await _request(reader, writer, 5, "send", {"text": "use page"})
        approval = await _event(reader, "approval_request")

        # Same connection ("new chat, then back"): equal headers are a no-op attach.
        same = await _request(reader, writer, 6, "resume", {
            "session_id": session_id, "mcp_bindings": [old]
        })
        assert same[-1].get("result", {}).get("attached") == "live_turn", same[-1]
        changed = await _request(reader, writer, 7, "resume", {
            "session_id": session_id, "mcp_bindings": [new]
        })
        assert _error_code(changed) == BUSY
        assert server.control._active_mcp_bindings == {
            "page": {"Authorization": "Bearer old-secret"}
        }

        await _close(writer)
        await _wait_for_controller_release(server)
        reader2, writer2 = await _open(server, 8, features)
        elsewhere = await _request(reader2, writer2, 9, "resume", {
            "session_id": other_id
        })
        assert _error_code(elsewhere) == BUSY
        mismatched = await _request(reader2, writer2, 10, "resume", {
            "session_id": session_id,
            "mcp_bindings": [{"server": "other", "headers": {"X-Key": "k"}}],
        })
        assert _error_code(mismatched) == INVALID_PARAMS
        attached = await _request(reader2, writer2, 11, "resume", {
            "session_id": session_id, "mcp_bindings": [new]
        })
        result = attached[-1]["result"]
        assert result["attached"] == "live_turn"
        assert result["session"]["session_id"] == session_id

        status = (await _request(reader2, writer2, 12, "status", {}))[-1]["result"]
        assert [row["request_id"] for row in status["pending_approvals"]] == [
            approval["request_id"]
        ]
        allowed = await _request(reader2, writer2, 13, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader2, "agent_end")
        await _request(reader2, writer2, 14, "send", {"text": "use page again"})
        second = await _event(reader2, "approval_request")
        await _request(reader2, writer2, 15, "approve", {
            "request_id": second["request_id"]
        })
        await _event(reader2, "agent_end")

        calls = [
            header
            for method, header in zip(mcp.methods, mcp.headers, strict=False)
            if method == "tools/call"
        ]
        assert calls == ["Bearer new-secret", "Bearer new-secret"]
    finally:
        await _close(writer2, writer)
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_resume_attach_resupplies_host_commands_for_next_turn(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([
        ScriptedTurn(tool_calls=[ToolCall("probe-1", "probe", {})]),
        ScriptedTurn([TextContent("probed")]),
        ScriptedTurn([TextContent("next turn")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    features = [
        "approval_reconnect", "new_session_settings", "session_host_commands",
        "resume_attach",
    ]
    host_commands = [{
        "name": "gamma",
        "executable": "/usr/bin/env",
        "summary": "call the gamma session API",
        "environment": {"GAMMA_TEST_TOKEN": "token"},
    }]
    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        session_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
            "approval_mode": "ask", "host_commands": host_commands,
        }))[-1]["result"]["session"]["session_id"]

        async def probe(arguments: dict[str, object], abort_signal: object) -> str:
            return "ok"

        assert server.runtime.loop is not None
        server.runtime.loop.tool_registry.register("probe", probe)
        await _request(reader, writer, 3, "send", {"text": "probe"})
        approval = await _event(reader, "approval_request")
        await _close(writer)
        await _wait_for_controller_release(server)

        reader2, writer2 = await _open(server, 4, features)
        attached = await _request(reader2, writer2, 5, "resume", {
            "session_id": session_id, "host_commands": host_commands
        })
        assert attached[-1].get("result", {}).get("attached") == "live_turn", (
            attached[-1]
        )
        allowed = await _request(reader2, writer2, 6, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader2, "agent_end")
        sent = await _request(reader2, writer2, 7, "send", {"text": "next"})
        assert "error" not in sent[-1], sent[-1]
        await _event(reader2, "agent_end")
    finally:
        await _close(writer2, writer)
        await server.close()


@pytest.mark.asyncio
async def test_resume_attach_rejects_mcp_bindings_for_unbound_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": "http://127.0.0.1:9"
    }}}))
    backend = FakeBackend([
        ScriptedTurn(tool_calls=[ToolCall("probe-1", "probe", {})]),
        ScriptedTurn([TextContent("probed")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    features = [
        "approval_reconnect", "new_session_settings", "session_mcp_bindings",
        "resume_attach",
    ]
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        session_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol", "approval_mode": "ask",
        }))[-1]["result"]["session"]["session_id"]

        async def probe(arguments: dict[str, object], abort_signal: object) -> str:
            return "ok"

        assert server.runtime.loop is not None
        server.runtime.loop.tool_registry.register("probe", probe)
        await _request(reader, writer, 3, "send", {"text": "probe"})
        approval = await _event(reader, "approval_request")
        rejected = await _request(reader, writer, 4, "resume", {
            "session_id": session_id,
            "mcp_bindings": [{"server": "page", "headers": {"X-Key": "k"}}],
        })
        assert _error_code(rejected) == INVALID_PARAMS, rejected[-1]
        assert server.control._active_mcp_bindings == {}
        allowed = await _request(reader, writer, 5, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader, "agent_end")
    finally:
        await _close(writer)
        await server.close()


@pytest.mark.asyncio
async def test_resume_without_attach_feature_stays_busy(tmp_path: Path) -> None:
    backend = FakeBackend([
        ScriptedTurn(tool_calls=[ToolCall("probe-1", "probe", {})]),
        ScriptedTurn([TextContent("probed")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2",
            "features": ["approval_reconnect", "new_session_settings"],
        })
        session_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol", "approval_mode": "ask",
        }))[-1]["result"]["session"]["session_id"]

        async def probe(arguments: dict[str, object], abort_signal: object) -> str:
            return "ok"

        assert server.runtime.loop is not None
        server.runtime.loop.tool_registry.register("probe", probe)
        await _request(reader, writer, 3, "send", {"text": "probe"})
        approval = await _event(reader, "approval_request")
        resumed = await _request(reader, writer, 4, "resume", {
            "session_id": session_id
        })
        assert _error_code(resumed) == BUSY
        allowed = await _request(reader, writer, 5, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader, "agent_end")
    finally:
        await _close(writer)
        await server.close()


@pytest.mark.asyncio
async def test_reload_resume_attaches_to_wake_in_setup_before_turn_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.owned import await_or_fail
    from tests.test_server_turn_reconnect import RECONNECT, _drop, _reconnect

    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    backend = FakeBackend([
        ScriptedTurn([TextContent("warm")]),
        ScriptedTurn([TextContent("woke")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    features = [*RECONNECT, "new_session_settings", "session_mcp_bindings", "resume_attach"]
    page = {"server": "page", "headers": {"Authorization": "Bearer page"}}
    reader, writer = await _connect(server)
    writer2 = None
    release_setup = asyncio.Event()
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        other_id = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
        }))[-1]["result"]["session"]["session_id"]
        session_id = (await _request(reader, writer, 3, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol", "mcp_bindings": [page],
        }))[-1]["result"]["session"]["session_id"]
        await _request(reader, writer, 4, "send", {"text": "warm"})
        await _event(reader, "agent_end")

        loop = server.runtime.loop
        in_setup = asyncio.Event()
        original_ensure = loop._ensure_mcp_servers

        async def held_ensure() -> None:
            # hold only the wake's call; the attach's own MCP rebind passes.
            if not in_setup.is_set():
                in_setup.set()
                await await_or_fail(release_setup.wait(), what="wake setup release")
            await original_ensure()

        # test-contract: seam MCP startup is the slow step a wake runs before turn_start
        monkeypatch.setattr(loop, "_ensure_mcp_servers", held_ensure)
        server.runtime.opened.store.append_agent_notification(
            "child", child_session_path="/tmp/child", description="child",
            status="completed", text="child done",
        )
        server.runtime._background_wake_sink(session_id)
        await await_or_fail(in_setup.wait(), what="wake in setup")

        # a page reload during the wake's setup: reconnect and resume the open session.
        await _drop(writer, server)
        reader2, writer2 = await _reconnect(server, 5, features)
        elsewhere = await _request(reader2, writer2, 6, "resume", {"session_id": other_id})
        assert _error_code(elsewhere) == BUSY
        attached = await _request(reader2, writer2, 7, "resume", {
            "session_id": session_id, "mcp_bindings": [page]
        })
        result = attached[-1].get("result", {})
        assert result.get("attached") == "live_turn", attached[-1]
        assert result["session"]["session_id"] == session_id

        release_setup.set()
        await _event(reader2, "turn_start")
        end = await _event(reader2, "agent_end")
        assert end["session_id"] == session_id
        assert server.runtime.session_id == session_id
    finally:
        release_setup.set()
        await _close(writer2, writer)
        await server.close()
        await mcp.close()
