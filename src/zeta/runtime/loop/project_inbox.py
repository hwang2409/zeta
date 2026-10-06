"""Project inbox polling and durable notices for top-level loops."""

from __future__ import annotations

import asyncio
from typing import Any

from ...project_inbox import InboxError, ProjectInbox


class ProjectInboxNotificationMixin:
    """Notice idle sessions without a daemon or a generated model turn per file."""

    _inbox_message_ids: tuple[str, ...] = ()
    tool_registry: Any
    store: Any
    agent_depth: int

    async def _activate_project_inbox(self) -> None:
        if "inbox" not in self.tool_registry.registered_names or self.agent_depth > 0:
            return
        await self._check_project_inbox()
        watcher = asyncio.create_task(self._watch_project_inbox())
        self._tracked_tasks.add(watcher)
        watcher.add_done_callback(self._tracked_tasks.discard)

    async def _watch_project_inbox(self) -> None:
        while True:
            await asyncio.sleep(2)
            await self._check_project_inbox()

    async def _check_project_inbox(self) -> None:
        """Add one durable user/model notice when the set of new messages changes."""

        projects = self.tool_registry.project_registry
        project_id = self.tool_registry.project_id
        if (
            "inbox" not in self.tool_registry.registered_names
            or projects is None
            or project_id is None
            or self.agent_depth > 0
        ):
            return

        def scan() -> tuple[str, ...]:
            inbox = ProjectInbox(
                projects, sessions_root=projects.root.parent / "sessions"
            )
            return tuple(item["id"] for item in inbox.list(project_id)["new"])

        try:
            message_ids = await asyncio.to_thread(scan)
        except (InboxError, OSError):
            return
        if message_ids == self._inbox_message_ids:
            return
        self._inbox_message_ids = message_ids
        if not message_ids:
            return
        _entry, appended = self.store.append_inbox_notification_if_absent(
            project_id, list(message_ids)
        )
        if appended:
            self.notify_background_persisted()
