"""Async worker lifecycle for full-screen finders."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

from prompt_toolkit.buffer import Buffer

_RANK_BATCH_SIZE = 16


class FinderRuntimeTarget(Protocol):
    """Finder operations used by the shared runtime."""

    @property
    def finder_active(self) -> bool: ...

    def open_finder(self, *args: Any) -> Any: ...

    async def build_finder_candidate_batches(
        self, request: Any
    ) -> AsyncIterator[tuple[Any, bool]]: ...

    def finder_publish_candidate_batch(
        self, request: Any, candidates: Any, *, complete: bool
    ) -> bool: ...

    def finder_set_query(self, query: str) -> int | None: ...

    def finder_needs_ranking(self) -> bool: ...

    def finder_rank(self, *, candidate_limit: int | None = None) -> Any: ...

    def finder_publish(self, result: Any) -> bool: ...

    def finder_state(self) -> Any: ...

    def finder_move(self, delta: int) -> None: ...

    def finder_toggle_preview(self) -> None: ...

    def finder_accept(self) -> bool: ...

    def finder_cancel(self) -> None: ...


class FinderRuntimeMixin:
    """Keep finder preparation responsive and fuzzy ranking off the UI loop."""

    def _finder_target(self) -> FinderRuntimeTarget:
        finder = getattr(self, "_active_finder", None)
        return finder if finder is not None else self._viewed_transcript()

    def _finder_open(self) -> None:
        self._active_finder = self._viewed_transcript()
        finder = self._active_finder
        session = getattr(self, "_active_session", None)
        field = getattr(self, "_finder_field", None)
        if session is not None and field is not None:
            try:
                session.layout.focus(field.window)
            except ValueError:
                pass
        request = finder.open_finder()
        self._start_finder_preparation(request)

    def _history_finder_open(self, buffer: Buffer) -> None:
        self._active_finder = self._history_finder
        finder = self._history_finder
        session = getattr(self, "_active_session", None)
        field = getattr(self, "_finder_field", None)
        if session is not None and field is not None:
            try:
                session.layout.focus(field.window)
            except ValueError:
                pass
            field.buffer.set_document(buffer.document, bypass_readonly=True)
        request = finder.open_finder(buffer)
        self._start_finder_preparation(request)

    def _start_finder_preparation(self, request: Any) -> None:
        if request is None:
            return
        finder = self._finder_target()
        self._finder_prepare_task = asyncio.create_task(
            self._prepare_finder_candidates(finder, request)
        )
        self._invalidate_prompt()

    async def _prepare_finder_candidates(
        self, finder: FinderRuntimeTarget, request: Any
    ) -> None:
        """Extract bounded candidates without blocking the UI loop."""

        task = asyncio.current_task()
        try:
            async for candidates, complete in finder.build_finder_candidate_batches(
                request
            ):
                if finder is not self._finder_target():
                    return
                if not finder.finder_publish_candidate_batch(
                    request, candidates, complete=complete
                ):
                    return
                self._invalidate_prompt()
                if finder.finder_needs_ranking():
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
                result = await asyncio.to_thread(
                    finder.finder_rank, candidate_limit=_RANK_BATCH_SIZE
                )
                if result is None:
                    return
                if finder.finder_publish(result):
                    self._invalidate_prompt()
                    if not finder.finder_needs_ranking():
                        return
        finally:
            if self._finder_rank_task is task:
                self._finder_rank_task = None
                finder = self._finder_target()
                if finder.finder_active and finder.finder_needs_ranking():
                    self._start_finder_ranking()

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
            self._active_finder = self._viewed_transcript()
        self._restore_composer_focus()
        self._invalidate_prompt()

    def _finder_cancel(self) -> None:
        self._finder_target().finder_cancel()
        self._cancel_finder_workers()
        self._active_finder = self._viewed_transcript()
        self._restore_composer_focus()
        self._invalidate_prompt()

    def _restore_composer_focus(self) -> None:
        session = getattr(self, "_active_session", None)
        if session is not None:
            session.layout.focus(session.default_buffer)

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


__all__ = ["FinderRuntimeMixin", "FinderRuntimeTarget"]
