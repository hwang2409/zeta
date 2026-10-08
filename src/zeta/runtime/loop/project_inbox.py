"""Project inbox polling and durable notices for top-level loops."""

from __future__ import annotations

import asyncio
from typing import Any

from ...project_inbox import InboxError, ProjectInboxScanner


class ProjectInboxNotificationMixin:
    """Notice top-level sessions without a daemon or persistent ownership."""

    _inbox_scanner: ProjectInboxScanner | None = None
    tool_registry: Any
    store: Any
    agent_depth: int
    _turn_active: bool

    async def _activate_project_inbox(self) -> None:
        if "inbox" not in self.tool_registry.registered_names or self.agent_depth > 0:
            return
        self._inbox_scanner = self._new_project_inbox_scanner()
        await self._check_project_inbox()
        watcher = asyncio.create_task(self._watch_project_inbox())
        self._tracked_tasks.add(watcher)
        watcher.add_done_callback(self._tracked_tasks.discard)

    def _new_project_inbox_scanner(self) -> ProjectInboxScanner:
        projects = self.tool_registry.project_registry
        project_id = self.tool_registry.project_id
        if projects is None or project_id is None:
            raise InboxError(
                "inbox is unavailable outside a registered project session"
            )
        return ProjectInboxScanner(
            projects,
            project_id,
            sessions_root=projects.root.parent / "sessions",
            session_id=self.store.session_id,
        )

    async def _watch_project_inbox(self) -> None:
        while True:
            await asyncio.sleep(2)
            await self._check_project_inbox(periodic=True)

    async def _check_project_inbox(self, *, periodic: bool = False) -> None:
        """Coalesce one passive notice and let one peer claim the model wake."""

        projects = self.tool_registry.project_registry
        project_id = self.tool_registry.project_id
        if (
            "inbox" not in self.tool_registry.registered_names
            or projects is None
            or project_id is None
            or self.agent_depth > 0
            or (periodic and self._turn_active)
        ):
            return
        scanner = self._inbox_scanner
        if scanner is None:
            scanner = self._new_project_inbox_scanner()
            self._inbox_scanner = scanner
        try:
            message_ids = await asyncio.to_thread(scanner.scan)
        except (InboxError, OSError):
            return
        if message_ids is None or not message_ids:
            return
        _entry, changed = self.store.append_inbox_notification_if_absent(
            project_id, list(message_ids)
        )
        if changed and await asyncio.to_thread(
            scanner.inbox.claim_wake, project_id, message_ids
        ):
            self.notify_background_persisted()
