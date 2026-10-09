"""Failure-safe cleanup for one TUI runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ..runtime.cleanup import close_session
from .bootstrap import surface_shutdown_notifications


class RuntimeCloseMixin:
    """Close every resource once, preserving the first cleanup error."""

    def _init_runtime_close(self) -> None:
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        if self._closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_runtime())
        await asyncio.shield(self._close_task)

    async def _close_runtime(self) -> None:
        first_error: BaseException | None = None

        def attempt(action: Callable[[], object]) -> object | None:
            nonlocal first_error
            try:
                return action()
            except BaseException as exc:  # noqa: BLE001 - cleanup must continue
                if first_error is None:
                    first_error = exc
                return None

        async def attempt_async(
            action: Callable[[], Awaitable[object]],
        ) -> None:
            nonlocal first_error
            try:
                await action()
            except BaseException as exc:  # noqa: BLE001 - cleanup must continue
                if first_error is None:
                    first_error = exc

        try:
            attempt(self.stop_decisions_poll)
            attempt(self._stop_decisions_refresh)
            await attempt_async(self._close_finder_workers)
            pending = attempt(
                lambda: (
                    {entry.id for entry in self.loop.store.agent_notifications()}
                    if self._terminal_restored
                    else set()
                )
            )
            pending_before = pending if isinstance(pending, set) else set()
            attempt(self._agent_navigation.unbind_layout)
            await attempt_async(self._cancel_mcp_wizard)
            await attempt_async(self._submissions.close)
            attempt(self._draft.detach)
            await attempt_async(
                lambda: close_session(
                    self.loop,
                    self._workspace_snapshot_store,
                    before_store_close=(
                        lambda: (
                            surface_shutdown_notifications(self, pending_before)
                            if self._terminal_restored
                            else None
                        )
                    ),
                )
            )
        finally:
            self._workspace_snapshot_store = None
            attempt(lambda: self.loop.set_background_event_sink(None))
            attempt(lambda: self.loop.set_background_wake_callback(None))
            attempt(lambda: self.loop.set_mcp_notice_sink(None))
            attempt(lambda: self.loop.set_mcp_prompt_refresh(None))
            attempt(
                lambda: self.loop.tool_registry.background_tasks.set_notice_sink(None)
            )
            if self._hooks is not None:
                attempt(lambda: setattr(self._hooks, "notice_sink", None))
            self._active_session = None
            self._draft_session = None
            self._session = None
            self._closed = True
        if first_error is not None:
            raise first_error


__all__ = ["RuntimeCloseMixin"]
