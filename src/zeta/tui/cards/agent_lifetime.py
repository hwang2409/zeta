from __future__ import annotations

import asyncio
import time
from pathlib import Path

from .agent_shutdown import SHUTDOWN_BUDGET_SECONDS, finish_cleanup_or_handoff
from .agent_sync import AgentTranscriptSource


class AgentTranscriptLifetime:
    """Own the worker-backed transcript source through asynchronous shutdown."""

    _source: AgentTranscriptSource | None
    _sync_task: asyncio.Task[bool] | None

    def _initialize_transcript_lifetime(self) -> None:
        self._source = None
        self._replacement_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def _replace_tracked(self, path: Path) -> None:
        replacement = asyncio.create_task(self._replace(path))
        self._replacement_task = replacement
        try:
            await replacement
        finally:
            if self._replacement_task is replacement:
                self._replacement_task = None

    async def _replace(self, path: Path) -> None:
        previous, self._source = self._source, None
        source = await asyncio.to_thread(self._replace_source, path, previous)
        if self._closed:
            await asyncio.to_thread(source.close)
        else:
            self._source = source

    @staticmethod
    def _replace_source(
        path: Path, previous: AgentTranscriptSource | None
    ) -> AgentTranscriptSource:
        if previous is not None:
            previous.close()
        source = AgentTranscriptSource(path)
        try:
            source.refresh()
        except BaseException:
            source.close()
            raise
        return source

    async def close(self, *, deadline: float | None = None) -> None:
        if self._close_task is None:
            if deadline is None:
                deadline = time.monotonic() + SHUTDOWN_BUDGET_SECONDS
            self._close_task = asyncio.create_task(
                self._close_transcript_lifetime(deadline)
            )
        await asyncio.shield(self._close_task)

    async def _close_transcript_lifetime(self, deadline: float) -> None:
        self._closed = True
        self._pending_path: Path | None = None
        candidates = self._sync_task, self._replacement_task
        tasks = {task for task in candidates if task is not None and not task.done()}
        source, self._source = self._source, None
        await finish_cleanup_or_handoff(
            tasks,
            source.close if source is not None else None,
            deadline=deadline,
            label="agent transcript",
        )


__all__ = ["AgentTranscriptLifetime"]
