"""A tool a server adds mid-run must reach the next model call in that run."""

import asyncio
import importlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.mcp import MCPMount, MCPServerConfig, MCPTool
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

mount_module = importlib.import_module("zeta.mcp.mount")
mcp_session_module = importlib.import_module("zeta.runtime.loop.mcp_session")

ECHO = MCPTool("echo", "", {"type": "object"})
ADDED = MCPTool("added", "", {"type": "object"})


class _AddingServer:
    """Fake MCP server: its echo call adds a tool and sends list_changed before it replies."""

    def __init__(self, config: MCPServerConfig, *, notify: bool = True) -> None:
        self.config = config
        self.protocol_version = "2025-06-18"
        self.capabilities = {"tools": {"listChanged": True}}
        self.notify = notify
        self.tools = [ECHO]
        self.refresh_started = asyncio.Event()
        self.refresh_release = asyncio.Event()
        self.refresh_started_at: float | None = None
        self._notification_sink: Callable[[str], None] | None = None
        self._list_calls = 0

    def set_failure_sink(self, sink: Callable[[str], None] | None) -> None:
        del sink

    def set_notification_sink(self, sink: Callable[[str], None] | None) -> None:
        self._notification_sink = sink

    async def connect(self) -> None:
        return None

    async def list_tools(self) -> list[MCPTool]:
        self._list_calls += 1
        if self._list_calls > 1:
            self.refresh_started_at = time.monotonic()
            self.refresh_started.set()
            await asyncio.wait_for(self.refresh_release.wait(), 30)
        return list(self.tools)

    async def call_tool(
        self, name: str, arguments: dict[str, object], abort_signal: AbortSignal
    ):
        del name, arguments, abort_signal
        if self.notify:
            self.tools.append(ADDED)
            assert self._notification_sink is not None
            self._notification_sink("notifications/tools/list_changed")
        return {"content": [], "isError": False, "structuredContent": None}

    async def close(self) -> None:
        self.refresh_release.set()


def _loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: _AddingServer
) -> tuple[AgentLoop, FakeBackend, list[float]]:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "home"))
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"servers": {"server": {"transport": "stdio", "command": "unused"}}})
    )
    monkeypatch.setenv("ZETA_MCP_CONFIG", str(config_path))
    # test-contract: seam the mount client factory is the fake MCP server boundary
    monkeypatch.setattr(mount_module, "_build_client", lambda _config: server)
    request_times: list[float] = []

    def serialize(messages: Sequence[Message], tools: Sequence[ToolSchema]) -> bytes:
        del messages, tools
        request_times.append(time.monotonic())
        return b""

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "server__echo", {})]),
            ScriptedTurn([TextContent("done")]),
        ],
        request_serializer=serialize,
    )
    loop = AgentLoop(
        backend, ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()
    )
    return loop, backend, request_times


async def _close(loop: AgentLoop) -> None:
    try:
        await loop.close()
    finally:
        loop.store.close()


def _names(backend: FakeBackend, call: int) -> set[str]:
    return {schema["name"] for schema in backend.calls[call][1]}


async def _run(loop: AgentLoop) -> list[StreamEventType]:
    return [
        event.type
        async for event in loop.run_turn("go", origin=MessageOrigin.USER)
    ]


def _spy_refresh_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[float], asyncio.Event]:
    waits: list[float] = []
    waiting = asyncio.Event()
    original = MCPMount.wait_for_tool_refreshes

    async def spy(self: MCPMount) -> None:
        waits.append(time.monotonic())
        waiting.set()
        await original(self)

    # test-contract: seam counts the only await a pending refresh adds to a step
    monkeypatch.setattr(MCPMount, "wait_for_tool_refreshes", spy)
    return waits, waiting


def _refresh_wait_tasks() -> list[asyncio.Task[object]]:
    names = ("spy", "wait")
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and getattr(task.get_coro(), "__name__", "") in names
    ]


@pytest.mark.asyncio
async def test_tool_added_mid_run_reaches_the_next_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _AddingServer(MCPServerConfig("server", "stdio", "unused"))
    loop, backend, _ = _loop(tmp_path, monkeypatch, server)

    async def release_refresh_late() -> None:
        await await_or_fail(server.refresh_started.wait(), what="tool refresh start")
        # test-contract: latency the refresh must still run while the next step is built
        await asyncio.sleep(0.2)
        server.refresh_release.set()

    releaser = asyncio.create_task(release_refresh_late())
    try:
        events = await await_or_fail(_run(loop), what="agent run")
        await await_or_fail(releaser, what="refresh release")
        assert events[-1] is StreamEventType.AGENT_END
        assert "server__added" not in _names(backend, 0)
        assert "server__added" in _names(backend, 1), "new tool missing from next call"
    finally:
        releaser.cancel()
        await await_or_fail(
            asyncio.gather(releaser, return_exceptions=True), what="releaser stop"
        )
        await _close(loop)


@pytest.mark.asyncio
async def test_stuck_refresh_costs_one_capped_wait_and_keeps_old_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 0.3
    monkeypatch.setattr(mcp_session_module, "MCP_REFRESH_TURN_WAIT_SECONDS", cap)
    server = _AddingServer(MCPServerConfig("server", "stdio", "unused"))
    loop, backend, request_times = _loop(tmp_path, monkeypatch, server)
    try:
        events = await await_or_fail(_run(loop), what="agent run")
        assert events[-1] is StreamEventType.AGENT_END
        assert server.refresh_started_at is not None
        step_wait = request_times[1] - server.refresh_started_at
        # test-contract: latency the capped wait is the product contract
        assert cap * 0.8 <= step_wait < cap + 0.5, step_wait
        assert "server__echo" in _names(backend, 1)
        assert "server__added" not in _names(backend, 1)
    finally:
        await _close(loop)


@pytest.mark.asyncio
async def test_cancel_during_refresh_wait_ends_the_run_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_session_module, "MCP_REFRESH_TURN_WAIT_SECONDS", 30.0)
    _, waiting = _spy_refresh_waits(monkeypatch)
    server = _AddingServer(MCPServerConfig("server", "stdio", "unused"))
    loop, backend, _ = _loop(tmp_path, monkeypatch, server)
    run = asyncio.create_task(_run(loop))
    waiter = asyncio.create_task(asyncio.wait_for(waiting.wait(), 10))
    try:
        done, _ = await asyncio.wait(
            (run, waiter), timeout=10, return_when=asyncio.FIRST_COMPLETED
        )
        assert waiter in done and not run.done(), "step did not wait for the refresh"
        cancelled_at = time.monotonic()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(run, what="cancelled run")
        # test-contract: latency cancel must not sit out the refresh wait
        assert time.monotonic() - cancelled_at < 0.5
        assert len(backend.calls) == 1
        assert _refresh_wait_tasks() == []
    finally:
        run.cancel()
        waiter.cancel()
        await await_or_fail(
            asyncio.gather(run, waiter, return_exceptions=True), what="run stop"
        )
        await _close(loop)


@pytest.mark.asyncio
async def test_step_without_pending_refresh_does_not_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits, _ = _spy_refresh_waits(monkeypatch)
    server = _AddingServer(MCPServerConfig("server", "stdio", "unused"), notify=False)
    loop, backend, _ = _loop(tmp_path, monkeypatch, server)
    try:
        events = await await_or_fail(_run(loop), what="agent run")
        assert events[-1] is StreamEventType.AGENT_END
        assert len(backend.calls) == 2
        assert waits == []
    finally:
        await _close(loop)
