"""Tool notification and refresh lifecycle for one MCP server actor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..client import (
    MCPClient,
    MCPTool,
    MCPToolRefreshError,
    degrades_server,
    error_text,
    notice,
)
from ..config import mcp_log_path

if TYPE_CHECKING:
    from ..server_actor import MCPServerActor


@dataclass(frozen=True, slots=True)
class ToolsListChanged:
    client: MCPClient


@dataclass(frozen=True, slots=True)
class ToolsListRefreshed:
    client: MCPClient
    task: asyncio.Task[list[MCPTool]]


@dataclass(frozen=True, slots=True)
class ToolsListRefreshRequested:
    result: asyncio.Future[None]


class ToolRefreshActor:
    """Own notification buffering and the active catalog refresh task."""

    def __init__(self, owner: MCPServerActor) -> None:
        self._owner = owner
        self._setup_client: MCPClient | None = None
        self._task: asyncio.Task[list[MCPTool]] | None = None
        self._pending = False
        self._settled = asyncio.Event()
        self._settled.set()
        self._setup_notification_client: MCPClient | None = None
        self._request_waiters: list[asyncio.Future[None]] = []

    def start_setup(self, client: MCPClient) -> None:
        self._setup_client = client
        set_notification_sink = getattr(client, "set_notification_sink", None)
        if set_notification_sink is not None:
            set_notification_sink(
                lambda method, client=client: self.notify(client, method)
            )

    def finish_setup(self, client: MCPClient) -> None:
        if self._setup_client is not client:
            return
        self._setup_client = None
        if self._setup_notification_client is client:
            self._setup_notification_client = None
            self.handle_changed(ToolsListChanged(client))

    def abort_setup(self, client: MCPClient | None) -> None:
        if self._setup_client is client:
            self._setup_client = None
        if self._setup_notification_client is client:
            self._setup_notification_client = None
            self.settle()

    def notify(self, client: MCPClient, method: str) -> None:
        if method != "notifications/tools/list_changed":
            return
        owner = self._owner
        if client is not owner._client and client is not self._setup_client:
            return
        self._settled.clear()
        if owner._status.state != "mounted":
            self._setup_notification_client = client
        else:
            owner._queue.put_nowait(ToolsListChanged(client))

    @property
    def pending(self) -> bool:
        return not self._settled.is_set()

    async def wait(self) -> None:
        await self._settled.wait()

    def request(self, message: ToolsListRefreshRequested) -> None:
        owner = self._owner
        client = owner._client
        if client is None or owner._status.state != "mounted":
            if not message.result.done():
                message.result.set_exception(
                    MCPToolRefreshError(f"MCP server {owner.name} is not mounted")
                )
            return
        self._request_waiters.append(message.result)
        self._settled.clear()
        if self._task is None:
            self.handle_changed(ToolsListChanged(client))

    def handle_changed(self, message: ToolsListChanged) -> None:
        owner = self._owner
        if message.client is not owner._client or owner._status.state != "mounted":
            return
        if self._task is not None:
            self._pending = True
            return
        task = asyncio.create_task(self._list_tools(message.client))
        self._task = task
        owner._children.add(task)
        task.add_done_callback(
            lambda done, client=message.client: owner._queue.put_nowait(
                ToolsListRefreshed(client, done)
            )
        )

    async def _list_tools(self, client: MCPClient) -> list[MCPTool]:
        return await asyncio.wait_for(client.list_tools(), self._owner._setup_timeout)

    def handle_refreshed(self, message: ToolsListRefreshed) -> None:
        from ..server_actor import MCPServerStatus

        owner = self._owner
        owner._children.discard(message.task)
        if self._task is message.task:
            self._task = None
        if message.client is not owner._client or owner._status.state != "mounted":
            self._pending = False
            return
        waiters = self._request_waiters
        try:
            tools = owner._filter_tools(message.task.result(), owner._notice_sink)
        except Exception as exc:  # noqa: BLE001 - classify refresh failure
            self._resolve_requests(waiters, exc)
            self._request_waiters = []
            # An explicit refresh reports the error and retains the last published
            # catalog. Notification-only transport failures keep existing behavior.
            if waiters:
                notice(
                    owner._notice_sink,
                    f"mcp · {owner.name} tool refresh failed: {error_text(exc)}",
                )
                self._continue_or_settle(message.client)
            elif degrades_server(exc):
                self._pending = False
                owner._degrade_current(error_text(exc))
            else:
                self._pending = False
                notice(
                    owner._notice_sink,
                    f"mcp · {owner.name} tool refresh failed: {error_text(exc)}",
                )
                self.settle()
            return
        owner._unregister_tools()
        owner._tools = tuple(tools)
        owner._set_status(
            MCPServerStatus(
                owner.name,
                owner.config.transport,
                "mounted",
                tool_count=len(owner._tools),
                stderr_log_path=str(mcp_log_path(owner.name)),
            )
        )
        owner._republish_definitions()
        owner._publish_callback(owner, owner._status, owner._client)
        if not tools and (
            owner.config.allowed_tools is not None or owner.config.disallowed_tools
        ):
            notice(owner._notice_sink, f"mcp · {owner.name} mounted with zero tools")
        self._resolve_requests(waiters)
        self._request_waiters = []
        self._continue_or_settle(message.client)

    def _continue_or_settle(self, client: MCPClient) -> None:
        if self._pending:
            self._pending = False
            self.handle_changed(ToolsListChanged(client))
        else:
            self.settle()

    @staticmethod
    def _resolve_requests(
        waiters: list[asyncio.Future[None]], error: Exception | None = None
    ) -> None:
        for waiter in waiters:
            if waiter.done():
                continue
            if error is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(MCPToolRefreshError(error_text(error)))

    def cancel(self) -> asyncio.Task[list[MCPTool]] | None:
        task = self._task
        self._task = None
        self._pending = False
        if task is not None:
            task.cancel()
        self._resolve_requests(
            self._request_waiters,
            MCPToolRefreshError("MCP tool refresh was cancelled"),
        )
        self._request_waiters = []
        return task

    def settle(self) -> None:
        self._pending = False
        if not self._settled.is_set():
            self._settled.set()

    async def close(self) -> None:
        task = self.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
