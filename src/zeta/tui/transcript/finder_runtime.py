"""Async worker lifecycle for full-screen finders."""

from __future__ import annotations

import asyncio
from typing import Any

from prompt_toolkit.buffer import Buffer


class FinderRuntimeMixin:
    """Keep finder preparation responsive and fuzzy ranking off the UI loop."""

    def _finder_target(self) -> Any:
        return getattr(self, "_active_finder", self._transcript)

    def _finder_open(self) -> None:
        self._active_finder = self._transcript
        request = self._transcript.open_finder()
        self._start_finder_preparation(request)

    def _history_finder_open(self, buffer: Buffer) -> None:
        self._active_finder = self._history_finder
        request = self._history_finder.open_finder(buffer)
        self._start_finder_preparation(request)

    def _start_finder_preparation(self, request: Any) -> None:
        if request is None:
            return
        finder = self._finder_target()
        self._finder_prepare_task = asyncio.create_task(
            self._prepare_finder_candidates(finder, request)
        )
        self._invalidate_prompt()

    async def _prepare_finder_candidates(self, finder: Any, request: Any) -> None:
        """Extract bounded candidates without blocking the UI loop."""

        task = asyncio.current_task()
        try:
            candidates = await finder.build_finder_candidates(request)
            if finder is self._finder_target() and finder.finder_publish_candidates(
                request, candidates
            ):
                self._invalidate_prompt()
                state = finder.finder_state()
                if state is not None and not state.complete:
                    self._start_finder_ranking()
        finally:
            if self._finder_prepare_task is task:
                self._finder_prepare_task = None

    def _finder_input(self, query: str) -> None:
        finder = self._finder_target()
        finder.finder_set_query(query)
        state = finder.finder_state()
        if state is None or state.complete:
            self._invalidate_prompt()
            return
        self._start_finder_ranking()

    def _start_finder_ranking(self) -> None:
        """Start one latest-query worker; stale generations are recomputed in order."""

        if self._finder_rank_task is None or self._finder_rank_task.done():
            self._finder_rank_task = asyncio.create_task(self._rank_finder_queries())

    async def _rank_finder_queries(self) -> None:
        task = asyncio.current_task()
        finder = self._finder_target()
        try:
            while finder is self._finder_target() and finder.finder_active:
                result = await asyncio.to_thread(finder.finder_rank)
                if result is None:
                    return
                if finder.finder_publish(result):
                    self._invalidate_prompt()
                    return
        finally:
            if self._finder_rank_task is task:
                self._finder_rank_task = None

    def _cancel_finder_workers(self) -> None:
        for task in (self._finder_prepare_task, self._finder_rank_task):
            if task is not None and not task.done():
                task.cancel()
        self._finder_prepare_task = None
        self._finder_rank_task = None

    async def _close_finder_workers(self) -> None:
        tasks = tuple(
            task
            for task in (self._finder_prepare_task, self._finder_rank_task)
            if task is not None and not task.done()
        )
        self._cancel_finder_workers()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _finder_move(self, delta: int) -> None:
        self._finder_target().finder_move(delta)
        self._invalidate_prompt()

    def _finder_accept(self) -> None:
        finder = self._finder_target()
        if finder.finder_accept():
            self._cancel_finder_workers()
            self._active_finder = self._transcript
        self._invalidate_prompt()

    def _finder_cancel(self) -> None:
        self._finder_target().finder_cancel()
        self._cancel_finder_workers()
        self._active_finder = self._transcript
        self._invalidate_prompt()

    async def _prepare_ui_session_change(self) -> None:
        """Close work scoped to a UI session before replacing or removing it."""

        finder = getattr(self, "_active_finder", None)
        if finder is not None and finder.finder_active:
            self._finder_cancel()
            session = self._active_session
            if session is not None:
                self._composer_restore_document = session.default_buffer.document
        paste_runtime = getattr(self, "_paste_runtime", None)
        if paste_runtime is not None:
            await paste_runtime.reset()

    def _finder_toggle_preview(self) -> None:
        self._finder_target().finder_toggle_preview()
        self._invalidate_prompt()


__all__ = ["FinderRuntimeMixin"]
