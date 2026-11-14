import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from zeta.mcp import (
    MCPConfig,
    MCPMount,
    MCPServerConfig,
    MCPTool,
    MCPToolRefreshError,
    mount_mcp_servers,
)
from zeta.mcp import mount as mount_module
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


class _HeldRefreshClient:
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        fail_refresh: bool = False,
        connect_release: asyncio.Event | None = None,
    ) -> None:
        self.config = config
        self.protocol_version = "2025-06-18"
        self.capabilities = {"tools": {"listChanged": True}}
        self.fail_refresh = fail_refresh
        self.connect_release = connect_release
        self.refresh_started = asyncio.Event()
        self.refresh_release = asyncio.Event()
        self._failure_sink: Callable[[str], None] | None = None
        self._notification_sink: Callable[[str], None] | None = None
        self._list_calls = 0

    def set_failure_sink(self, sink: Callable[[str], None] | None) -> None:
        self._failure_sink = sink

    def set_notification_sink(self, sink: Callable[[str], None] | None) -> None:
        self._notification_sink = sink

    async def connect(self) -> None:
        if self.connect_release is not None:
            await asyncio.wait_for(self.connect_release.wait(), 10)

    async def list_tools(self) -> list[MCPTool]:
        self._list_calls += 1
        if self._list_calls == 1:
            return [MCPTool("echo", "", {"type": "object"})]
        self.refresh_started.set()
        await asyncio.wait_for(self.refresh_release.wait(), 10)
        if self.fail_refresh:
            raise RuntimeError("refresh failed")
        return [MCPTool("changed", "", {"type": "object"})]

    def notify_list_changed(self) -> None:
        assert self._notification_sink is not None
        self._notification_sink("notifications/tools/list_changed")

    def fail_transport(self) -> None:
        assert self._failure_sink is not None
        self._failure_sink("transport dropped")

    async def close(self) -> None:
        return None


async def _mounted_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: _HeldRefreshClient,
    *later_clients: _HeldRefreshClient,
) -> tuple[ToolRegistry, MCPMount]:
    clients = iter((client, *later_clients))
    # test-contract: seam the mount client factory forces lifecycle ordering
    monkeypatch.setattr(mount_module, "_build_client", lambda _config: next(clients))
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    config = MCPServerConfig("server", "stdio", "unused")
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {"server": config})
    )
    client.notify_list_changed()
    await await_or_fail(client.refresh_started.wait(), what="held tool refresh")
    return registry, mount


async def _explicit_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_refresh: bool = False,
) -> tuple[_HeldRefreshClient, ToolRegistry, MCPMount]:
    client = _HeldRefreshClient(
        MCPServerConfig("server", "stdio", "unused"),
        fail_refresh=fail_refresh,
    )
    clients = iter((client,))
    monkeypatch.setattr(mount_module, "_build_client", lambda _config: next(clients))
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    config = MCPServerConfig("server", "stdio", "unused")
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {"server": config})
    )
    return client, registry, mount


async def _wait_catalog(
    mount: MCPMount, registry: ToolRegistry
) -> tuple[str, set[str]]:
    await mount.wait_for_tool_refreshes()
    status = mount.statuses.get("server")
    return (
        "removed" if status is None else status.state,
        {schema["name"] for schema in registry.schemas},
    )


async def _assert_following_turn_does_not_wait(mount: MCPMount) -> None:
    assert not mount.tool_refresh_pending
    await await_or_fail(
        mount.wait_for_tool_refreshes(), what="following tool refresh wait"
    )


@pytest.mark.asyncio
async def test_explicit_tool_refresh_coalesces_concurrent_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _HeldRefreshClient(MCPServerConfig("server", "stdio", "unused"))
    registry, mount = await _mounted_refresh(tmp_path, monkeypatch, client)
    refresh = mount._actors["server"]._tool_refresh
    requests_coalesced = asyncio.Event()
    original_request = type(refresh).request

    def observe_request(self, message) -> None:
        original_request(self, message)
        if len(self._request_waiters) == 2:
            requests_coalesced.set()

    monkeypatch.setattr(type(refresh), "request", observe_request)
    first = asyncio.create_task(mount.refresh_tools("server"))
    second = asyncio.create_task(mount.refresh_tools("server"))
    try:
        await await_or_fail(requests_coalesced.wait(), what="coalesced refresh requests")
        assert client._list_calls == 2
        client.refresh_release.set()
        assert await await_or_fail(first, what="first explicit refresh") == ("server",)
        assert await await_or_fail(second, what="coalesced explicit refresh") == (
            "server",
        )
        assert client._list_calls == 2, "concurrent requests started parallel tools/list"
    finally:
        client.refresh_release.set()
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_failed_explicit_tool_refresh_keeps_published_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, registry, mount = await _explicit_refresh(
        tmp_path, monkeypatch, fail_refresh=True
    )
    refresh = asyncio.create_task(mount.refresh_tools("server"))
    try:
        await await_or_fail(client.refresh_started.wait(), what="failing refresh start")
        client.refresh_release.set()
        with pytest.raises(MCPToolRefreshError, match="refresh failed"):
            await await_or_fail(refresh, what="failing explicit refresh")
        assert {schema["name"] for schema in registry.schemas} == {"server__echo"}
        assert mount.statuses["server"].state == "mounted"
    finally:
        client.refresh_release.set()
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_cancelled_explicit_tool_refresh_does_not_block_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, registry, mount = await _explicit_refresh(tmp_path, monkeypatch)
    refresh = asyncio.create_task(mount.refresh_tools("server"))
    try:
        await await_or_fail(client.refresh_started.wait(), what="cancelled refresh start")
        refresh.cancel()
        with pytest.raises(asyncio.CancelledError):
            await refresh
        client.refresh_release.set()
        await await_or_fail(
            mount.wait_for_tool_refreshes(), what="cancelled caller refresh settlement"
        )
        assert {schema["name"] for schema in registry.schemas} == {"server__changed"}
        assert await mount.refresh_tools("server") == ("server",)
    finally:
        client.refresh_release.set()
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_failed_held_refresh_publishes_degraded_catalog_before_settling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _HeldRefreshClient(
        MCPServerConfig("server", "stdio", "unused"), fail_refresh=True
    )
    registry, mount = await _mounted_refresh(tmp_path, monkeypatch, client)
    actor = mount._actors["server"]
    original_publish = actor._publish_callback
    pending_at_degraded_publish: list[bool] = []

    def observe_publish(actor, status, published_client) -> None:
        original_publish(actor, status, published_client)
        if status.state == "degraded":
            pending_at_degraded_publish.append(actor._tool_refresh.pending)

    actor._publish_callback = observe_publish
    waiter = asyncio.create_task(_wait_catalog(mount, registry))
    try:
        client.refresh_release.set()
        assert await await_or_fail(waiter, what="failed refresh settlement") == (
            "degraded",
            set(),
        )
        assert pending_at_degraded_publish == [True]
        await _assert_following_turn_does_not_wait(mount)
    finally:
        client.refresh_release.set()
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_close_publishes_removed_catalog_before_settling_held_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _HeldRefreshClient(MCPServerConfig("server", "stdio", "unused"))
    registry, mount = await _mounted_refresh(tmp_path, monkeypatch, client)
    actor = mount._actors["server"]
    waiter = asyncio.create_task(_wait_catalog(mount, registry))
    close = asyncio.create_task(actor.close())
    try:
        assert await await_or_fail(waiter, what="close refresh settlement") == (
            "failed",
            set(),
        )
        await _assert_following_turn_does_not_wait(mount)
        await await_or_fail(close, what="server actor close")
    finally:
        client.refresh_release.set()
        if not close.done():
            close.cancel()
            await asyncio.wait_for(
                asyncio.gather(close, return_exceptions=True), 10
            )
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_replacement_publishes_failed_catalog_before_settling_held_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("server", "stdio", "unused")
    client = _HeldRefreshClient(config)
    replacement_connect_release = asyncio.Event()
    replacement = _HeldRefreshClient(
        config, connect_release=replacement_connect_release
    )
    registry, mount = await _mounted_refresh(
        tmp_path, monkeypatch, client, replacement
    )
    waiter = asyncio.create_task(_wait_catalog(mount, registry))
    replace = asyncio.create_task(
        mount.replace_server(config, source=tmp_path / "replacement.json")
    )
    try:
        assert await await_or_fail(waiter, what="replacement refresh settlement") == (
            "failed",
            set(),
        )
        await _assert_following_turn_does_not_wait(mount)
    finally:
        client.refresh_release.set()
        replacement_connect_release.set()
        await await_or_fail(replace, what="server replacement")
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_disconnect_publishes_degraded_catalog_before_settling_held_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _HeldRefreshClient(MCPServerConfig("server", "stdio", "unused"))
    registry, mount = await _mounted_refresh(tmp_path, monkeypatch, client)
    waiter = asyncio.create_task(_wait_catalog(mount, registry))
    try:
        client.fail_transport()
        assert await await_or_fail(waiter, what="disconnect refresh settlement") == (
            "degraded",
            set(),
        )
        await _assert_following_turn_does_not_wait(mount)
    finally:
        client.refresh_release.set()
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_suspend_publishes_degraded_catalog_before_settling_held_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("server", "stdio", "unused")
    client = _HeldRefreshClient(config)
    registry, mount = await _mounted_refresh(tmp_path, monkeypatch, client)
    waiter = asyncio.create_task(_wait_catalog(mount, registry))
    suspend = asyncio.create_task(
        mount.suspend_server(config, source=tmp_path / "mcp.json")
    )
    try:
        assert await await_or_fail(waiter, what="suspend refresh settlement") == (
            "degraded",
            set(),
        )
        await _assert_following_turn_does_not_wait(mount)
        await await_or_fail(suspend, what="server suspension")
    finally:
        client.refresh_release.set()
        if not suspend.done():
            suspend.cancel()
            await asyncio.wait_for(
                asyncio.gather(suspend, return_exceptions=True), 10
            )
        await mount.close()
        await registry.close()
