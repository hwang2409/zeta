"""table coverage of the bounded MCP close helper through the actor interface:
an uncancellable client close must never wedge suspend or shutdown, while a
normal close still runs and errors are swallowed (RCA resume-stuck)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import zeta.mcp.server_actor as server_actor_module
from tests.support.owned import await_or_fail
from tests.test_mcp import _LifecycleClient
from zeta.mcp.client import MCPTool
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
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0.2)
    client = _UncancellableCloseClient(CONFIG)
    actor = _actor(client, tmp_path)
    actor.start()
    try:
        await await_or_fail(actor.wait_started(), what="mount")
        # suspend schedules the bound client's close (it blocks on the gate).
        await await_or_fail(actor.suspend(CONFIG, tmp_path / "mcp.json"), what="suspend")
        await await_or_fail(client.close_started.wait(), what="close attempted")
        first = actor._scheduled_closes[id(client)][1]
        # scheduling the same still-closing client returns the same task: one
        # close, never two, so a later close cannot double-close it.
        again = actor._schedule_close(client)
        assert again is first
        client.gate.set()
    finally:
        client.gate.set()
        await await_or_fail(actor.close(), what="final close")
