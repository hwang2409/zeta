"""table coverage of the bounded MCP close helper through the actor interface:
an uncancellable client close must never wedge suspend or shutdown, while a
normal close still runs and errors are swallowed (RCA resume-stuck)."""
from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path

import pytest

import zeta.mcp.server_actor as server_actor_module
from tests.support.owned import await_or_fail
from tests.test_mcp import _LifecycleClient
from zeta.mcp.client import MCPTool, MCPToolRefreshError
from zeta.mcp.config import MCPServerConfig
from zeta.mcp.server_actor import MCPServerActor

CONFIG = MCPServerConfig("page", "stdio", "unused")


async def _connect_and_list(client: _LifecycleClient) -> list[MCPTool]:
    await client.connect()
    return await client.list_tools()


def _actor(client: _LifecycleClient, tmp_path: Path) -> MCPServerActor:
    return MCPServerActor(
        CONFIG,
        source=tmp_path / "mcp.json",
        registry=None,
        publish=lambda *_args: None,
        build_client=lambda _config: client,
        connect_and_list=_connect_and_list,
    )


class _SuccessClient(_LifecycleClient):
    pass


class _UncancellableCloseClient(_LifecycleClient):
    """close() blocks on a gate and ignores cancellation until it is set."""

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self.close_started = asyncio.Event()
        self.gate = asyncio.Event()

    async def close(self) -> None:
        self.close_started.set()
        while not self.gate.is_set():
            try:
                # test-contract: durability models a close that ignores cancellation
                await self.gate.wait()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
        self.closed = True


class _RaisingCloseClient(_LifecycleClient):
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError("close failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    ["success", "uncancellable", "raising"],
)
async def test_suspend_and_close_are_bounded_across_close_behaviors(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0.2)
    client: _LifecycleClient = {
        "success": _SuccessClient,
        "uncancellable": _UncancellableCloseClient,
        "raising": _RaisingCloseClient,
    }[kind](CONFIG)
    actor = _actor(client, tmp_path)
    actor.start()
    try:
        status = await await_or_fail(actor.wait_started(), what="mount")
        assert status.state == "mounted"
        # suspend schedules the bound client's close; an uncancellable close
        # must not wedge the actor loop here.
        await await_or_fail(actor.suspend(CONFIG, tmp_path / "mcp.json"), what="suspend")
        if kind == "uncancellable":
            await await_or_fail(client.close_started.wait(), what="close attempted")
        # the actor stays usable: a later close completes, bounded.
        await await_or_fail(actor.close(), what="close")
        if kind == "success":
            assert client.closed
    finally:
        if isinstance(client, _UncancellableCloseClient):
            client.gate.set()
        await await_or_fail(actor.close(), what="final close")


@pytest.mark.asyncio
async def test_suspend_caller_cancel_at_the_close_wait_keeps_actor_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 30.0)
    client = _UncancellableCloseClient(CONFIG)
    actor = _actor(client, tmp_path)
    actor.start()
    try:
        await await_or_fail(actor.wait_started(), what="mount")
        suspend = asyncio.create_task(actor.suspend(CONFIG, tmp_path / "mcp.json"))
        await await_or_fail(client.close_started.wait(), what="close wait reached")
        suspend.cancel()
        # the suspend call is shielded; cancelling the caller must not corrupt
        # the actor. a later close still completes.
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(suspend, what="cancelled suspend caller")
        client.gate.set()
        await await_or_fail(actor.close(), what="close after caller cancel")
    finally:
        client.gate.set()
        await await_or_fail(actor.close(), what="final close")


@pytest.mark.asyncio
async def test_schedule_close_dedups_a_client_already_closing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 30.0)
    client = _UncancellableCloseClient(CONFIG)
    actor = _actor(client, tmp_path)
    actor.start()
    try:
        await await_or_fail(actor.wait_started(), what="mount")
        # schedule the bound client's close (it blocks on the gate, not yet
        # abandoned because the drain timeout is long).
        first = actor._schedule_close(client)
        await await_or_fail(client.close_started.wait(), what="close attempted")
        # scheduling the same still-closing client returns the same task: one
        # close, never two, so a later close cannot double-close it.
        again = actor._schedule_close(client)
        assert again is first
        client.gate.set()
    finally:
        client.gate.set()
        await await_or_fail(actor.close(), what="final close")


class _RefreshBlockingClient(_LifecycleClient):
    """list_tools() returns once for mount, then blocks on a gate and ignores
    cancellation, modelling a tool refresh that never ends on a dead peer."""

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self.refresh_started = asyncio.Event()
        self.gate = asyncio.Event()
        self._calls = 0

    async def list_tools(self) -> list[MCPTool]:
        self._calls += 1
        if self._calls == 1:
            return [MCPTool("echo", "", {"type": "object"})]
        self.refresh_started.set()
        while not self.gate.is_set():
            try:
                # test-contract: durability models a refresh that ignores cancellation
                await self.gate.wait()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
        return [MCPTool("echo", "", {"type": "object"})]


@pytest.mark.asyncio
async def test_repeated_rebinds_keep_abandoned_closes_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each detach closes the bound client; an uncancellable close is abandoned.
    Repeated re-binds must not accumulate those abandoned closes: at most one
    stays live per server, and setting the gates drains them back to zero."""
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0.2)
    clients: list[_UncancellableCloseClient] = []

    def build(_config: MCPServerConfig) -> _UncancellableCloseClient:
        client = _UncancellableCloseClient(CONFIG)
        clients.append(client)
        return client

    created: list[asyncio.Task[object]] = []
    original_schedule = MCPServerActor._schedule_close

    def record_schedule(
        self: MCPServerActor, client: object, *, detach: bool = True
    ) -> asyncio.Task[object]:
        task = original_schedule(self, client, detach=detach)
        created.append(task)
        return task

    # test-contract: seam record each scheduled close so the test can await them
    monkeypatch.setattr(MCPServerActor, "_schedule_close", record_schedule)

    actor = MCPServerActor(
        CONFIG,
        source=tmp_path / "mcp.json",
        registry=None,
        publish=lambda *_args: None,
        build_client=build,
        connect_and_list=_connect_and_list,
    )
    actor.start()
    try:
        status = await await_or_fail(actor.wait_started(), what="mount")
        assert status.state == "mounted"
        rebinds = 5
        for _ in range(rebinds):
            await await_or_fail(
                actor.suspend(CONFIG, tmp_path / "mcp.json"), what="suspend"
            )
            # the owned set never grows with abandoned closes; on the old code
            # each suspend left its stuck close here and this reached `rebinds`.
            assert len(actor._scheduled_closes) <= 1
            status = await await_or_fail(actor.reconnect(), what="reconnect")
            assert status.state == "mounted"
        await await_or_fail(
            actor.suspend(CONFIG, tmp_path / "mcp.json"), what="final suspend"
        )
        assert len(actor._scheduled_closes) <= 1
        # at most one abandoned close stays live per server, process-wide.
        assert len(actor._abandoned_closes) <= 1
        assert server_actor_module.live_abandoned_closes() <= 1
        tracked = weakref.ref(next(iter(actor._abandoned_closes)))
    finally:
        for client in clients:
            client.gate.set()
        await await_or_fail(
            asyncio.gather(*created, return_exceptions=True), what="closes drain"
        )
        await await_or_fail(actor.close(), what="final close")
    # once the closes end, nothing stays abandoned and the tracked task is
    # released: no strong reference to it is kept.
    assert server_actor_module.live_abandoned_closes() == 0
    assert not actor._abandoned_closes
    created.clear()
    gc.collect()
    assert tracked() is None


@pytest.mark.asyncio
async def test_tool_refresh_close_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool refresh whose list_tools() ignores cancellation must not wedge
    suspend: the refresh close is routed through the same bounded drain."""
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0.2)
    client = _RefreshBlockingClient(CONFIG)
    actor = _actor(client, tmp_path)
    actor.start()
    try:
        await await_or_fail(actor.wait_started(), what="mount")
        refresh = asyncio.create_task(actor.refresh_tools())
        await await_or_fail(client.refresh_started.wait(), what="refresh started")
        # suspend closes the stuck refresh; on the old gather() this never
        # returned and the suspend deadlocked.
        await await_or_fail(
            actor.suspend(CONFIG, tmp_path / "mcp.json"), what="suspend"
        )
        with pytest.raises(MCPToolRefreshError):
            await await_or_fail(refresh, what="refresh rejected")
        assert server_actor_module.live_abandoned_closes() <= 1
    finally:
        client.gate.set()
        await await_or_fail(actor.close(), what="final close")
    assert server_actor_module.live_abandoned_closes() == 0
