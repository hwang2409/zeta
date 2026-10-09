"""Tool notification and refresh lifecycle for one MCP server actor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..client import MCPClient, MCPTool, error_text, notice
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


class ToolRefreshActor:
    """Own notification buffering and the active catalog refresh task."""

    def __init__(self, owner: MCPServerActor) -> None:
        self._owner = owner
        self._setup_client: MCPClient | None = None
        self._task: asyncio.Task[list[MCPTool]] | None = None
        self._pending = False
        self._setup_notification_pending = False

    def start_setup(self, client: MCPClient) -> None:
        self._setup_client = client
        set_notification_sink = getattr(client, "set_notification_sink", None)
        if set_notification_sink is not None:
            set_notification_sink(
                lambda method, client=client: self.notify(client, method)
            )

    def finish_setup(self, client: MCPClient) -> None:
        self._setup_client = None
        if self._setup_notification_pending:
            self._setup_notification_pending = False
            self.handle_changed(ToolsListChanged(client))

    def notify(self, client: MCPClient, method: str) -> None:
        if method != "notifications/tools/list_changed":
            return
        owner = self._owner
        if (
            (client is owner._client or client is self._setup_client)
            and owner._status.state != "mounted"
        ):
            self._setup_notification_pending = True
        else:
            owner._queue.put_nowait(ToolsListChanged(client))

    def handle_changed(self, message: ToolsListChanged) -> None:
        owner = self._owner
        if message.client is not owner._client or owner._status.state != "mounted":
            return
        if self._task is not None:
            self._pending = True
            return
        task = asyncio.create_task(
            asyncio.wait_for(message.client.list_tools(), owner._setup_timeout)
        )
        self._task = task
        owner._children.add(task)
        task.add_done_callback(
            lambda done, client=message.client: owner._queue.put_nowait(
                ToolsListRefreshed(client, done)
            )
        )

    def handle_refreshed(self, message: ToolsListRefreshed) -> None:
        from ..server_actor import MCPServerStatus

        owner = self._owner
        owner._children.discard(message.task)
        if self._task is message.task:
            self._task = None
        if message.client is not owner._client or owner._status.state != "mounted":
            self._pending = False
            return
        try:
            tools = owner._filter_tools(message.task.result(), owner._notice_sink)
        except Exception as exc:  # noqa: BLE001 - refresh failure degrades the server
            self._pending = False
            owner._degrade_current(error_text(exc))
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
        if self._pending:
            self._pending = False
            self.handle_changed(ToolsListChanged(message.client))

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
