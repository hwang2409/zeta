"""Async worker lifecycle for full-screen finders."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any, Protocol

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.shortcuts import PromptSession

from ..overlay_text_field import OverlayTextField
from .finder_overlay import FinderControl

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

    def _init_finder_runtime(self) -> None:
        app_ref = weakref.ref(self)
        self._finder_field = OverlayTextField(
            "TRANSCRIPT_FINDER",
            on_text_changed=lambda query: (
                target._finder_input(query)
                if (target := app_ref()) is not None
                else None
            ),
        )
        self._finder_control = FinderControl(
            lambda: app_ref()._finder_drawn_state(), self._finder_field
        )
        self._finder_prepare_task: asyncio.Task[None] | None = None
        self._finder_rank_task: asyncio.Task[None] | None = None
        self._finder_accept_pending = False
        self._finder_vi_mode: InputMode | None = None

    def _finder_drawn_state(self) -> Any:
        """The open finder's state, still searching while any result source runs."""

        state = self._finder_target().finder_state()
        if state is not None and state.complete and self._finder_loading():
            return replace(state, complete=False)
        return state

    def _focus_finder_field(self) -> PromptSession[str] | None:
        session = getattr(self, "_active_session", None)
        field = getattr(self, "_finder_field", None)
        if session is None or field is None:
            return None
        # Esc in the finder leaves vi insert mode; closing it gives this back.
        self._finder_vi_mode = session.app.vi_state.input_mode
        field.focus(session.layout)
        return session

    def _finder_target(self) -> FinderRuntimeTarget:
        finder = getattr(self, "_active_finder", None)
        return finder if finder is not None else self._viewed_transcript()

    def _finder_open(self) -> None:
        transcript = self._viewed_transcript()
        self._open_finder_overlay(transcript, transcript.open_finder)

    def _history_finder_open(
        self, buffer: Buffer, load: Callable[[], Awaitable[Any]] | None = None
    ) -> None:
        self._open_finder_overlay(
            self._history_finder,
            lambda: self._history_finder.open_finder(buffer),
            buffer.document,
            load,
        )

    def _open_finder_overlay(
        self,
        finder: FinderRuntimeTarget,
        open_finder: Callable[[], Any],
        document: Document | None = None,
        load: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        def show() -> bool:
            self._finder_accept_pending = False
            if document is None:
                self._finder_field.reset(notify=False)
            self._active_finder = finder
            if self._focus_finder_field() is not None and document is not None:
                self._finder_field.set_document(document, notify=False)
            self._start_finder_preparation(open_finder())
            return True

        self._open_overlay(show, load, on_loaded=self._finder_accept_when_ready)

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
                else:
                    self._finder_accept_when_ready()
        finally:
            if self._finder_prepare_task is task:
                self._finder_prepare_task = None
                # An Enter held while this task ran runs now that the rows stand.
                self._finder_accept_when_ready()

    def _finder_input(self, query: str) -> None:
        # A queued Enter belongs to the query it was typed for, not a later one.
        self._finder_accept_pending = False
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
                        self._finder_accept_when_ready()
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

    def _finder_loading(self) -> bool:
        """True while the open finder's rows can still change."""

        state = self._finder_target().finder_state()
        return state is not None and (
            not state.complete
            or self.overlay_loading
            or self._finder_prepare_task is not None
        )

    def _finder_accept_when_ready(self) -> None:
        # Run after the calling worker returns: accepting cancels the workers.
        if self._finder_accept_pending:
            asyncio.get_running_loop().call_soon(self._finder_accept_if_ready)

    def _finder_accept_if_ready(self) -> None:
        if self._finder_accept_pending and not self._finder_loading():
            self._finder_accept()

    def _finder_accept(self) -> None:
        # Enter typed before the rows are ready waits for them instead of dropping.
        self._finder_accept_pending = self._finder_loading()
        if self._finder_accept_pending:
            return
        finder = self._finder_target()
        if finder.finder_accept():
            self._cancel_finder_workers()
            self._active_finder = self._viewed_transcript()
            self._finder_field.reset(notify=False)
            self._restore_composer_focus()
        self._invalidate_prompt()

    def _finder_cancel(self) -> None:
        self._finder_accept_pending = False
        self._finder_target().finder_cancel()
        self._cancel_finder_workers()
        self._active_finder = self._viewed_transcript()
        self._finder_field.reset(notify=False)
        self._restore_composer_focus()
        self._invalidate_prompt()

    def _restore_composer_focus(self) -> None:
        session = getattr(self, "_active_session", None)
        if session is not None:
            session.layout.focus(session.default_buffer)
            if self._finder_vi_mode is not None:
                session.app.vi_state.input_mode = self._finder_vi_mode

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
