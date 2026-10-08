"""Async worker lifecycle for the full-screen transcript finder."""

from __future__ import annotations

import asyncio
from typing import Any


class FinderRuntimeMixin:
    """Keep finder preparation responsive and fuzzy ranking off the UI loop."""

    def _finder_open(self) -> None:
        request = self._transcript.open_finder()
        if request is None:
            return
        self._finder_prepare_task = asyncio.create_task(
            self._prepare_finder_candidates(request)
        )
        self._invalidate_prompt()

    async def _prepare_finder_candidates(self, request: Any) -> None:
        """Extract bounded transcript candidates without blocking the UI loop."""

        task = asyncio.current_task()
        try:
            candidates = await self._transcript.build_finder_candidates(request)
            if self._transcript.finder_publish_candidates(request, candidates):
                self._invalidate_prompt()
                state = self._transcript.finder_state()
                if state is not None and not state.complete:
                    self._start_finder_ranking()
        finally:
            if self._finder_prepare_task is task:
                self._finder_prepare_task = None

    def _finder_input(self, query: str) -> None:
        self._transcript.finder_set_query(query)
        state = self._transcript.finder_state()
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
        try:
            while self._transcript.finder_active:
                result = await asyncio.to_thread(self._transcript.finder_rank)
                if result is None:
                    return
                if self._transcript.finder_publish(result):
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
        self._transcript.finder_move(delta)
        self._invalidate_prompt()

    def _finder_accept(self) -> None:
        if self._transcript.finder_accept():
            self._cancel_finder_workers()
        self._invalidate_prompt()

    def _finder_cancel(self) -> None:
        self._transcript.finder_cancel()
        self._cancel_finder_workers()
        self._invalidate_prompt()

    def _finder_toggle_preview(self) -> None:
        self._transcript.finder_toggle_preview()
        self._invalidate_prompt()


__all__ = ["FinderRuntimeMixin"]
