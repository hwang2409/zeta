"""Compatibility exports for the focused TUI card package."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any

from prompt_toolkit.application.current import get_app
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from rich.console import Console
from rich.text import Text

from ..core.checkpoints import ConversationIntegrityError
from ..core.session_files import SessionError
from ..core.store import ConversationStore
from ..protocol.types import Message, ToolCall
from . import theme
from .agent_view_state import (
    _TERMINAL_AGENT_STATES,
    MAX_AGENT_SCAN_BYTES,
    MAX_AGENT_VIEW_LINES,
    AgentEntry,
    _agent_metadata,
    agent_navigation_style_rules,
    capture_inactive_view,
    read_agent_transcript,
)
from .agent_view_state import InactiveViewState as _InactiveViewState
from .agent_view_state import NavigationSnapshot as _NavigationSnapshot
from .agent_view_state import PreparedNavigation as _PreparedNavigation
from .cards.agent import (
    AgentCard,
    AgentRunCommandMixin,
    _read_lifecycle,
    render_agent_expanded,
    render_agent_progress,
    render_agent_receipt,
)
from .cards.agent_list import (
    AGENT_LIST_PAGE_SIZE,
    MAX_AGENT_LIST_ROWS,
    AgentListControl,
)
from .cards.agent_shutdown import close_agent_owners
from .cards.agent_sync import (
    AgentTodoControl,
    agent_child_paths,
    agent_path_signature,
    agent_tree_signature,
    candidate_paths,
    registered_child_markers,
)
from .cards.shared import (
    MAX_CARD_COLUMNS,
    MAX_CARD_LINES,
    infer_language,
)
from .cards.shared import (
    BoundedToolOutput as _BoundedToolOutput,
)
from .cards.shared import (
    scan_tool_output as _scan_tool_output,
)
from .cards.tool import TOOL_CARD_REGISTRY, register_tool_card
from .checkpoints import render_replayed_message
from .todo import TodoSnapshot
from .user import displayed_user_text, user_message

__all__ = [
    "MAX_AGENT_SCAN_BYTES",
    "MAX_AGENT_VIEW_LINES",
    "MAX_CARD_COLUMNS",
    "MAX_CARD_LINES",
    "TOOL_CARD_REGISTRY",
    "AgentCard",
    "AgentEntry",
    "AgentNavigation",
    "AgentRunCommandMixin",
    "_BoundedToolOutput",
    "_read_lifecycle",
    "_scan_tool_output",
    "agent_navigation_style_rules",
    "infer_language",
    "read_agent_transcript",
    "register_tool_card",
    "render_agent_expanded",
    "render_agent_progress",
    "render_agent_receipt",
    "time",
]

_AGENT_REFRESH_INTERVAL_SECONDS = 1.0
_AGENT_IDLE_REFRESH_INTERVAL_SECONDS = 4.0

class AgentTranscriptControl(UIControl):
    """Render the shared pager for the focused child view."""
    def __init__(self, history_pager: Any = None) -> None:
        from .transcript import TranscriptPresenter, TranscriptWidget
        from .transcript.paged_history import PagedTranscriptHistory
        self._history_pager = (
            history_pager if history_pager is not None else PagedTranscriptHistory(None)
        )
        self.transcript = TranscriptWidget(max_lines=None)
        self.presenter = TranscriptPresenter(
            self.transcript,
            Console(),
            lambda: True,
            self.transcript.append,
        )
        self._tool_calls: dict[str, ToolCall] = {}
        self._path: Path | None = None
        self._source: Any = None
        self._load_task: asyncio.Task[None] | None = None
        self._load_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._pending_path: Path | None = None
        self._activation_generation = 0
        self.search_task: asyncio.Task[None] | None = None
        self.search_results: tuple[Any, ...] = ()
        self.search_index = 0
        self.view_path: Path | None = None
        self._root_path: Path | None = None
        self._root_source: Any = None
        self._root_transcript: Any = None
        self._root_view_state: _InactiveViewState | None = None
        self._child_view_states: OrderedDict[Path, _InactiveViewState] = OrderedDict()
        self._prompt_history: Any = None
        self._composer_buffer: Any = None
        self._current_path: Callable[[], Path] | None = None
        self._prepare_path: Callable[[Path], _PreparedNavigation] | None = None
        self._begin_path: Callable[[int], None] | None = None
        self._commit_path: Callable[[Path, _PreparedNavigation, int], bool] | None = None
        self._rollback_path: Callable[[int], None] | None = None
        self._finish_path: Callable[[int], None] | None = None
        self._render_active: Callable[[], None] | None = None
        self._root_endpoint: Callable[[], str | None] | None = None
        self._invalidate: Callable[[], None] | None = None
        self.on_history_edge: Callable[[str], None] | None = None
    @property
    def history_pager(self) -> Any:
        return self._history_pager

    def transition_pending(self) -> bool:
        """Return whether a view change still owns unpublished work."""

        return self._load_task is not None and not self._load_task.done()

    @property
    def prompt_history(self) -> Any:
        return self._prompt_history
    @property
    def offset(self) -> int:
        return self.transcript.scroll_offset
    def is_focusable(self) -> bool:
        return True
    def configure_history(
        self,
        *,
        root_path: Path,
        root_source: Any,
        root_transcript: Any,
        prompt_history: Any,
        current_path: Callable[[], Path],
        prepare_path: Callable[[Path], _PreparedNavigation],
        begin_path: Callable[[int], None],
        commit_path: Callable[[Path, _PreparedNavigation, int], bool],
        rollback_path: Callable[[int], None],
        finish_path: Callable[[int], None],
        render_active: Callable[[], None],
        root_endpoint: Callable[[], str | None],
        invalidate: Callable[[], None],
    ) -> None:
        self._root_path = root_path
        self.view_path = root_path
        self._root_source = root_source
        self._root_transcript = root_transcript
        self._prompt_history = prompt_history
        self._current_path = current_path
        self._prepare_path = prepare_path
        self._begin_path = begin_path
        self._commit_path = commit_path
        self._rollback_path = rollback_path
        self._finish_path = finish_path
        self._render_active = render_active
        self._root_endpoint = root_endpoint
        self._invalidate = invalidate
    def start_search(
        self,
        query: str,
        transcript: Any,
        publish: Callable[[Any, str], None],
    ) -> None:
        self.search_results = ()
        self.search_index = 0
        if self.search_task is not None:
            self.search_task.cancel()
        if not query:
            return
        async def reveal_match() -> None:
            matches = await self._history_pager.search(query)
            if not matches or not transcript.finder_active:
                return
            self.search_results = matches
            self.search_index = 0
            target = matches[0]
            resident = {
                unit.entry_id
                for page in self._history_pager.pages
                for unit in page.units
            }
            if target.entry_id not in resident:
                await self._history_pager.reveal(target.entry_id)
            self.render_pages(self._history_pager.pages)
            transcript.restore_durable_scroll_anchor(
                (target.entry_id, target.unit_ordinal, 0)
            )
            publish(target, query)
        self.search_task = asyncio.create_task(reveal_match())

    def navigate_search(
        self,
        delta: int,
        transcript: Any,
        publish: Callable[[Any, str], None],
    ) -> bool:
        if not self.search_results:
            return False
        self.search_index = (self.search_index + delta) % len(self.search_results)
        target = self.search_results[self.search_index]

        async def reveal() -> None:
            await self._history_pager.reveal(target.entry_id)
            if not transcript.finder_active:
                return
            query = transcript.search_query
            self.render_pages(self._history_pager.pages)
            transcript.restore_durable_scroll_anchor(
                (target.entry_id, target.unit_ordinal, 0)
            )
            publish(target, query)

        if self.search_task is not None:
            self.search_task.cancel()
        self.search_task = asyncio.create_task(reveal())
        return True
    def end_search(self) -> None:
        self.search_results = ()
        self.search_index = 0

    async def close_search(self) -> None:
        task, self.search_task = self.search_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.end_search()

    def _active_transcript(self) -> Any:
        return self._root_transcript if self._path == self._root_path else self.transcript

    async def _refresh_active(
        self, path: Path, prepared: _PreparedNavigation, generation: int
    ) -> None:
        """Publish fresh navigation and child rows without moving the viewport."""

        if self._commit_path is not None and not self._commit_path(
            path, prepared, generation
        ):
            return
        if self._finish_path is not None:
            self._finish_path(generation)
        source = self._source
        if path != self._root_path and source is not None:
            anchor = self.transcript.durable_scroll_anchor
            page = await (
                source.reveal(anchor[0])
                if anchor is not None
                else source.page(None, direction="older")
            )
            latest = (
                await source.page(None, direction="older") if anchor is not None else None
            )
            fresh = (page,) if latest is None or latest == page else (page, latest)
            # A scroll made during the reads wins; the next refresh catches up.
            if (
                generation == self._activation_generation
                and self.transcript.durable_scroll_anchor == anchor
                and fresh != self._history_pager.pages
            ):
                self._history_pager.reset()
                for index, fresh_page in enumerate(fresh):
                    self._history_pager.adopt(
                        fresh_page, direction="newer" if index else "older"
                    )
                self.render_pages(self._history_pager.pages)
                if anchor is not None:
                    # A short revealed window reads as the tail; keep the anchor.
                    self._history_pager.set_anchor(*anchor)
                    self._restore_anchor(self.transcript, anchor)
        if self._invalidate is not None:
            self._invalidate()

    def _saved_view(self, path: Path) -> _InactiveViewState | None:
        if path == self._root_path:
            return self._root_view_state
        return self._child_view_states.get(path)

    def _park_active_view(self) -> bool:
        path = self._path
        if path is None:
            return True
        transcript = self._root_transcript if path == self._root_path else self.transcript
        anchor = transcript.durable_scroll_anchor if transcript is not None else None
        state = capture_inactive_view(
            buffer=self._composer_buffer,
            prompt_history=self._prompt_history,
            transcript=transcript,
            anchor=anchor,
            search_index=self.search_index,
        )
        if state is None:
            return False
        if path == self._root_path:
            self._root_view_state = state
        else:
            self._child_view_states[path] = state
            self._child_view_states.move_to_end(path)
            while len(self._child_view_states) > 8:
                self._child_view_states.popitem(last=False)
        return True

    def _cancel_transition_work(self) -> tuple[asyncio.Task[Any], ...]:
        tasks: list[asyncio.Task[Any]] = []
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            tasks.append(self._load_task)
        if self.search_task is not None and not self.search_task.done():
            self.search_task.cancel()
            tasks.append(self.search_task)
        self.search_task = None
        tasks.extend(self._history_pager.cancel_pending())
        if self._prompt_history is not None:
            tasks.extend(self._prompt_history.cancel_pending())
        return tuple(dict.fromkeys(tasks))

    def _start_transition(
        self, path: Path, *, refresh: bool
    ) -> asyncio.Task[None] | None:
        self._activation_generation += 1
        generation = self._activation_generation
        self._pending_path = path
        if refresh:
            # A refresh replaces no runtime: page, prompt, and finder reads stay.
            cancelled = tuple(
                task
                for task in (self._load_task,)
                if task is not None and not task.done()
            )
            for task in cancelled:
                task.cancel()
        else:
            cancelled = self._cancel_transition_work()
        if self._begin_path is not None:
            self._begin_path(generation)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._transition_without_loop(path, generation, refresh, cancelled))
            self._load_task = None
            return None
        task = loop.create_task(
            self._transition(path, generation, refresh, cancelled)
        )
        self._load_task = task
        self._load_tasks.add(task)
        task.add_done_callback(self._load_tasks.discard)
        return task

    def activate(self, path: Path) -> asyncio.Task[None] | None:
        """Start the single owned transition for a child activation."""

        if self._closed or path == self._pending_path:
            return None
        if path == self._path:
            return self._start_transition(path, refresh=True)
        if not self._park_active_view():
            return None
        return self._start_transition(path, refresh=False)

    async def _transition_without_loop(self, *args: Any) -> None:
        # The commit starts follow-up work (the todo refresh) on this private
        # loop; finish it here, or asyncio.run cancels it with its store open.
        await self._transition(*args)
        current = asyncio.current_task()
        while pending := asyncio.all_tasks() - {current}:
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _restore_anchor(transcript: Any, anchor: tuple[str, int, int]) -> None:
        if not transcript.restore_durable_scroll_anchor(anchor):
            transcript.restore_durable_scroll_anchor((anchor[0], 0, 0))

    async def _transition(
        self,
        path: Path,
        generation: int,
        refresh: bool,
        cancelled: tuple[asyncio.Task[Any], ...],
    ) -> None:
        from .transcript.child_history_source import ChildLogHistorySource

        prepared_source: Any = None
        committed = False
        try:
            if cancelled:
                await asyncio.gather(*cancelled, return_exceptions=True)
            if generation != self._activation_generation:
                return
            prepared_navigation = (
                await asyncio.to_thread(self._prepare_path, path)
                if self._prepare_path is not None
                else _PreparedNavigation(())
            )
            if generation != self._activation_generation:
                return
            if refresh:
                await self._refresh_active(path, prepared_navigation, generation)
                return

            state = self._saved_view(path)
            if path == self._root_path:
                prepared_source = self._root_source
            else:
                worker = asyncio.create_task(asyncio.to_thread(ChildLogHistorySource, path))
                try:
                    prepared_source = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    prepared_source = await worker
                    await prepared_source.close()
                    raise
            if generation != self._activation_generation:
                if prepared_source not in {None, self._root_source}:
                    await prepared_source.close()
                return

            prepared_page = None
            prepared_newer = None
            if prepared_source is not None:
                if state is not None and state.anchor is not None:
                    prepared_page = await prepared_source.reveal(state.anchor[0])
                    # Rows after the anchor let the viewport start exactly at it.
                    if prepared_page.newer is not None:
                        prepared_newer = await prepared_source.page(
                            prepared_page.newer, direction="newer"
                        )
                else:
                    prepared_page = await prepared_source.page(None, direction="older")
            if generation != self._activation_generation:
                if prepared_source not in {None, self._root_source}:
                    await prepared_source.close()
                return
            endpoint = (
                self._root_endpoint()
                if prepared_source is self._root_source and self._root_endpoint
                else None
            )
            if (
                prepared_page is not None
                and endpoint is not None
                and not prepared_page.endpoint_id
            ):
                prepared_page = replace(prepared_page, endpoint_id=endpoint)

            committed = (
                self._commit_path(path, prepared_navigation, generation)
                if self._commit_path is not None
                else True
            )
            if not committed:
                if prepared_source not in {None, self._root_source}:
                    await prepared_source.close()
                return

            previous = self._source
            previous_path = self._path
            if previous_path == self._root_path and self._root_transcript is not None:
                self._root_transcript.clear()
            elif previous_path is not None:
                self.presenter.clear()
                self._tool_calls.clear()

            self._path = path
            self._source = None if prepared_source is self._root_source else prepared_source
            self._history_pager.source = prepared_source
            self._history_pager.reset(endpoint_id=endpoint)
            if prepared_page is not None:
                self._history_pager.adopt(prepared_page)
            if prepared_newer is not None:
                self._history_pager.adopt(prepared_newer, direction="newer")
            if state is not None and state.anchor is not None:
                self._history_pager.set_anchor(*state.anchor)
            restored = state or _InactiveViewState(None, "", 0, None, None, "", 0)
            if self._prompt_history is not None:
                self._prompt_history.bind(path, prepared_source, endpoint)
                self._prompt_history.restore_navigation(
                    restored.prompt_cursor,
                    restored.prompt_text,
                    Document(restored.draft, cursor_position=restored.draft_cursor),
                )
            if path != self._root_path:
                self._child_view_states.pop(path, None)
            buffer = self._composer_buffer
            if buffer is not None:
                displayed = restored.prompt_text or restored.draft
                cursor = (
                    len(displayed)
                    if restored.prompt_text is not None
                    else restored.draft_cursor
                )
                document = Document(displayed, cursor_position=cursor)
                if self._prompt_history is not None:
                    self._prompt_history.apply(buffer, document)
                else:
                    buffer.set_document(document, bypass_readonly=True)
            self.search_index = restored.finder_cursor
            if self._render_active is not None:
                self._render_active()
            else:
                self.render_pages(self._history_pager.pages)
            transcript = self._active_transcript()
            if transcript is not None and restored.finder_query:
                transcript.open_finder()
                transcript.finder_set_query(restored.finder_query)
            if state is not None and state.anchor is not None and transcript is not None:
                self._restore_anchor(transcript, state.anchor)
                asyncio.get_running_loop().call_soon(
                    self._restore_anchor, transcript, state.anchor
                )
            if self._finish_path is not None:
                self._finish_path(generation)
            committed = False
            if self._invalidate is not None:
                self._invalidate()
            if previous is not None and previous is not prepared_source:
                await previous.close()
        except asyncio.CancelledError:
            if committed and self._rollback_path is not None:
                self._rollback_path(generation)
            if prepared_source not in {None, self._root_source, self._source}:
                await prepared_source.close()
            raise
        except (ConversationIntegrityError, OSError, SessionError, ValueError):
            if self._rollback_path is not None:
                self._rollback_path(generation)
            if prepared_source not in {None, self._root_source, self._source}:
                await prepared_source.close()
            if self._invalidate is not None:
                self._invalidate()
        except Exception:
            if self._rollback_path is not None:
                self._rollback_path(generation)
            if prepared_source not in {None, self._root_source, self._source}:
                await prepared_source.close()
            raise
        finally:
            if generation == self._activation_generation and path == self._pending_path:
                self._pending_path = None

    def request_sync(
        self, path: Path, invalidate: Callable[[], None] | None = None
    ) -> bool:
        active_search = self.search_task is not None and not self.search_task.done()
        prompt_busy = bool(
            self._prompt_history is not None and self._prompt_history.busy
        )
        if self._closed or (
            path == self._path
            and (self._pending_path is not None or active_search or prompt_busy)
        ):
            return False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        task = (
            self.activate(path)
            if path != self._path
            else self._start_transition(path, refresh=True)
        )
        if task is not None and invalidate is not None:
            task.add_done_callback(lambda _task: invalidate())
        return task is not None
    def render_pages(self, pages: tuple[Any, ...]) -> None:
        anchor = self.transcript.durable_scroll_anchor
        follow_tail = self.transcript.follow_tail
        self.presenter.clear()
        self._tool_calls.clear()
        provider = self._source.replay_provider if self._source is not None else None
        for page in pages:
            for unit in page.units:
                entry = unit.value
                if getattr(entry, "type", None) != "message" or entry.message is None:
                    continue
                self.transcript.begin_history_entry(unit.entry_id)
                try:
                    render_replayed_message(
                        entry.message,
                        presenter=self.presenter,
                        print_user=self._print_user,
                        print_unit=self.presenter.print_unit,
                        tool_calls=self._tool_calls,
                        include_thoughts=True,
                        replay_tool_results=True,
                        replay_tool_starts=True,
                        provider=provider,
                    )
                finally:
                    self.transcript.end_history_entry()
        if not follow_tail and anchor is not None:
            self.transcript.restore_durable_scroll_anchor(anchor)

    def _print_user(self, message: Message) -> None:
        self.presenter.print_user(
            user_message(Text(displayed_user_text(message), style=theme.BODY))
        )

    def pending_prompts(self, path: Path) -> tuple[str, ...]:
        if path != self._path or self._source is None:
            return ()
        return self._source.pending_prompts
    def capture_root_view(self) -> None:
        """Park only the root scalars before navigation changes layout focus."""

        if self._path == self._root_path:
            self._park_active_view()

    async def cancel_activation(self) -> None:
        generation = self._activation_generation
        self._activation_generation += 1
        tasks = self._cancel_transition_work()
        if self._rollback_path is not None:
            self._rollback_path(generation)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        remaining = tuple(self._load_tasks)
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)
        self._load_tasks.clear()
        self._load_task = None
        self._pending_path = None
    async def close(self, deadline: float | None = None) -> None:
        del deadline
        self._closed = True
        await self.cancel_activation()
        self._current_path = None
        self._prepare_path = None
        self._begin_path = None
        self._commit_path = None
        self._rollback_path = None
        self._finish_path = None
        self._render_active = None
        self._root_endpoint = None
        self._invalidate = None
        self.on_history_edge = None
        source, self._source = self._source, None
        if source is not None:
            if inspect.iscoroutinefunction(source.close):
                await source.close()
            else:
                await asyncio.to_thread(source.close)

    def scroll(self, amount: int) -> None:
        self.transcript.scroll_lines(amount)
        if amount < 0 and self.transcript.scroll_offset == 0 and self.on_history_edge:
            self.on_history_edge("older")
        elif amount > 0 and self.transcript.follow_tail and self.on_history_edge:
            self.on_history_edge("newer")

    def half_page(self, amount: int) -> None:
        self.scroll(amount * max(1, self.transcript._viewport_height // 2))

    def top(self) -> None:
        self.transcript.scroll_to_top()
        if self.on_history_edge:
            self.on_history_edge("older")

    def bottom(self) -> None:
        self.transcript.scroll_to_bottom()
        if self.on_history_edge:
            self.on_history_edge("newer")

    def toggle_latest_agent(self) -> bool:
        return self.transcript.toggle_latest_agent()
    def create_content(self, width: int, height: int | None) -> UIContent:
        return self.transcript.create_content(width, height)
    def vertical_scroll(self, window: Window) -> int:
        return self.transcript.vertical_scroll(window)
    def mouse_handler(self, mouse_event: MouseEvent):
        result = self.transcript.mouse_handler(mouse_event)
        if result is NotImplemented and mouse_event.event_type is MouseEventType.MOUSE_DOWN:
            get_app().layout.focus(self)
            return None
        return result
class AgentNavigation:
    """Own the current session path, list selection, and child transcript."""

    def __init__(self, store: ConversationStore, history_pager: Any = None) -> None:
        self.store = store
        self.root_path = store.session_dir
        self.current_path = self.root_path
        self._path_stack: list[Path] = [self.root_path]
        self._breadcrumb_labels = ["main"]
        self.entries: list[AgentEntry] = []
        self.selected_index = 0
        self._fork_title: str | None = None
        self._fork_main_path: Path | None = None
        self._on_return_to_main: Callable[[], None] | None = None
        self.list_control = AgentListControl(self)
        self.transcript_control = AgentTranscriptControl(history_pager)
        self.transcript_control._root_path = self.transcript_control._path = self.root_path
        self.transcript_control._prepare_path = self._prepare_view_path
        self.transcript_control._begin_path = self._begin_view_transition
        self.transcript_control._commit_path = self._commit_view_path
        self.transcript_control._rollback_path = self._rollback_view_transition
        self.transcript_control._finish_path = self._finish_view_transition
        self.list_window = Window(
            content=self.list_control,
            height=Dimension(min=1, max=MAX_AGENT_LIST_ROWS),
            wrap_lines=False,
        )
        self.transcript_window = Window(
            content=self.transcript_control,
            wrap_lines=False,
            get_vertical_scroll=self.transcript_control.vertical_scroll,
        )
        self.breadcrumb_window = Window(
            content=FormattedTextControl(self._breadcrumb_fragments),
            height=1,
            wrap_lines=False,
        )
        self.view_container = HSplit([self.breadcrumb_window, self.transcript_window])
        self._layout: Any = None
        self._composer_buffer: Any = None
        self._transcript_layout: Any = None
        self._main_transcript: Any = None
        self._main_transcript_widget: Any = None
        self._main_transcript_parent: Any = None
        self._main_transcript_index: int | None = None
        self._external_view_container: Any = None
        self._external_view_transcript: Any = None
        self._todo_control = AgentTodoControl(cwd=self.store.cwd)
        self._metadata_cache: dict[Path, tuple[tuple[int, ...], dict[str, Any]]] = {}
        self._refresh_signature: tuple[object, ...] | None = None
        self._last_refresh_at = float("-inf")
        self._refresh_handle: asyncio.TimerHandle | None = None
        self._invalidate: Callable[[], None] | None = None
        self._navigation_generation: int | None = None
        self._navigation_snapshot: _NavigationSnapshot | None = None
        self.refresh(force=True)

    def _breadcrumb_fragments(self) -> list[tuple[str, str]]:
        """Style the breadcrumb so the main agent and its subagents differ.

        The first crumb is the main agent (blue); every deeper crumb is a
        subagent (green). Separators stay in the dim chrome colour.
        """

        fragments: list[tuple[str, str]] = []
        for index, label in enumerate(self._breadcrumb_labels):
            if index:
                fragments.append(("class:agent-breadcrumb", " > "))
            role = (
                "class:agent-breadcrumb.main"
                if index == 0
                else "class:agent-breadcrumb.child"
            )
            fragments.append((role, label))
        return fragments

    @property
    def child_view_active(self) -> bool:
        return self.current_path != self.root_path

    def child_instance_id(self) -> str | None:
        if not self.child_view_active:
            return None
        metadata = self._cached_agent_metadata(self.current_path)
        handle = metadata.get("handle")
        return handle if type(handle) is str and handle else None

    def child_queued_messages(self) -> tuple[str, tuple[str, ...]] | None:
        if not self.child_view_active:
            return None
        label = self.child_instance_id() or self.current_path.name
        return label, self.transcript_control.pending_prompts(self.current_path)

    async def wait_for_publication(self) -> None:
        """Wait for the current transcript and todo view to publish."""

        while True:
            task = self.transcript_control._load_task
            if task is not None:
                await asyncio.shield(task)
            if task is self.transcript_control._load_task:
                break
        await self._todo_control.settle()

    def todo_snapshot(self) -> TodoSnapshot | None:
        if not self.child_view_active:
            return TodoSnapshot(tuple(self.store.todo_items()), self.store.todo_dismissed)
        return self._todo_control.snapshot(self.current_path)

    async def close(self) -> None:
        await close_agent_owners(self.transcript_control.close, self._todo_control.close)

    def child_view_focused(self) -> bool:
        return self._layout is not None and self._layout.has_focus(self.transcript_window)

    @property
    def list_visible(self) -> bool:
        self.refresh()
        return bool(self.entries)

    @property
    def page_index(self) -> int:
        return self.selected_index // AGENT_LIST_PAGE_SIZE

    @property
    def page_count(self) -> int:
        return max(
            1,
            (len(self.entries) + AGENT_LIST_PAGE_SIZE - 1) // AGENT_LIST_PAGE_SIZE,
        )

    @property
    def list_height(self) -> int:
        if not self.entries:
            return 0
        page_start = self.page_index * AGENT_LIST_PAGE_SIZE
        page_rows = min(AGENT_LIST_PAGE_SIZE, len(self.entries) - page_start)
        return page_rows + int(len(self.entries) > AGENT_LIST_PAGE_SIZE)

    def bind_layout(
        self,
        layout: Any,
        composer_buffer: Any,
        invalidate: Callable[[], None] | None = None,
    ) -> None:
        self._layout = layout
        self._composer_buffer = composer_buffer
        self.transcript_control._composer_buffer = composer_buffer
        self._invalidate = invalidate
        self._schedule_refresh(_AGENT_REFRESH_INTERVAL_SECONDS)

    def unbind_layout(self) -> None:
        if self._refresh_handle is not None:
            self._refresh_handle.cancel()
            self._refresh_handle = None
        self._invalidate = None
        self._layout = None
        self._composer_buffer = None
        self.transcript_control._composer_buffer = None

    def bind_transcript_layout(self, layout: Any, main_transcript: Any) -> None:
        self._transcript_layout = layout
        self._main_transcript = main_transcript
        self._main_transcript_parent = None
        self._main_transcript_index = None
        layout_children = getattr(layout, "children", [])
        if not layout_children:
            return
        parent = layout_children[0]
        for index, child in enumerate(getattr(parent, "children", [])):
            if child is main_transcript:
                self._main_transcript_parent = parent
                self._main_transcript_index = index
                return

    @property
    def viewed_transcript(self) -> Any:
        if self._external_view_transcript is not None:
            return self._external_view_transcript
        if self.child_view_active:
            return self.transcript_control.transcript
        return self._main_transcript_widget

    def show_external_view(self, container: Any, transcript: Any) -> None:
        self._external_view_container = container
        self._external_view_transcript = transcript
        self._switch_transcript()

    def clear_external_view(self) -> None:
        self._external_view_container = None
        self._external_view_transcript = None
        self._switch_transcript()

    def _switch_transcript(self) -> None:
        if on_view_change := getattr(self, "_on_view_change", None):
            on_view_change()
        if self._transcript_layout is None:
            return
        replacement = (
            self._external_view_container
            if self._external_view_container is not None
            else self.view_container if self.child_view_active else self._main_transcript
        )
        if self._main_transcript_parent is not None and self._main_transcript_index is not None:
            self._main_transcript_parent.children[self._main_transcript_index] = replacement
        else:
            self._transcript_layout.children[0] = replacement

    _path_signature = staticmethod(agent_path_signature)

    def _agent_tree_signature(self) -> tuple[object, ...]:
        running = (
            entry.path for entry in self.entries
            if entry.state not in _TERMINAL_AGENT_STATES
            and (entry.path != self.root_path or self.child_view_active)
        )
        return agent_tree_signature(
            self.current_path, self.selected_index, running, self._path_signature
        )

    def _schedule_refresh(self, delay: float) -> None:
        if self._invalidate is None or self._refresh_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refresh_handle = loop.call_later(delay, self._refresh_after_debounce)

    def request_refresh(self) -> None:
        """Re-arm discovery promptly after an event that can create a child."""

        if self._refresh_handle is not None:
            self._refresh_handle.cancel()
            self._refresh_handle = None
        self._schedule_refresh(0)

    def _refresh_after_debounce(self) -> None:
        self._refresh_handle = None
        if self._invalidate is None:
            return
        if self.refresh():
            self._invalidate()
        delay = (
            _AGENT_REFRESH_INTERVAL_SECONDS
            if any(
                entry.state not in _TERMINAL_AGENT_STATES
                for entry in self.entries
            )
            else _AGENT_IDLE_REFRESH_INTERVAL_SECONDS
        )
        self._schedule_refresh(delay)

    def refresh(self, *, force: bool = False) -> bool:
        """Refresh at most once per second unless fresh state is required."""

        now = _monotonic()
        elapsed = now - self._last_refresh_at
        if not force and elapsed < _AGENT_REFRESH_INTERVAL_SECONDS:
            self._schedule_refresh(_AGENT_REFRESH_INTERVAL_SECONDS - elapsed)
            self._resize_list_window()
            return False
        self._last_refresh_at = now
        owned_refresh = self.transcript_control._root_source is not None
        transcript_changed = (
            self.transcript_control.request_sync(self.current_path, self._invalidate)
            if owned_refresh
            else False
        )
        todo_changed = (
            self._todo_control.request(self.current_path, self._invalidate)
            if self.child_view_active
            else False
        )
        if owned_refresh:
            self._resize_list_window()
            return transcript_changed or todo_changed
        signature = self._agent_tree_signature()
        if not force and signature == self._refresh_signature:
            return transcript_changed or todo_changed
        was_list_focused = bool(self.entries) and self.list_focused()
        selected_path = self.entries[self.selected_index].path if self.entries else None
        previous_index = self.selected_index
        fallback = registered_child_markers(self.current_path)
        known = {
            entry.path
            for entry in self.entries
            if entry.path.parent == self.current_path / "agents"
        }
        candidates = candidate_paths(
            signature, self._refresh_signature, fallback, known, force
        )
        children = self._children(self.current_path, fallback, candidates=candidates)
        at_root = self.current_path == self.root_path
        main_entry = (
            [self._main_entry()] if self._fork_main_path is not None and at_root else []
        )
        if at_root and not children and not main_entry:
            self.entries = []
        else:
            self.entries = [self._root_entry(), *main_entry, *children]
        if selected_path is not None:
            self.selected_index = next(
                (index for index, entry in enumerate(self.entries) if entry.path == selected_path),
                min(previous_index, max(0, len(self.entries) - 1)),
            )
        else:
            default_index = 1 if children else 0
            self.selected_index = min(
                max(previous_index, default_index),
                max(0, len(self.entries) - 1),
            )
        self._resize_list_window()
        # Keep the pre-read signature: mutations during _children() must be
        # visible to the next refresh rather than being paired with stale data.
        self._refresh_signature = signature
        if was_list_focused and not self.entries:
            self.focus_composer()
        return True

    def _resize_list_window(self) -> None:
        list_height = min(self.list_height, MAX_AGENT_LIST_ROWS)
        self.list_window.height = Dimension(
            min=list_height,
            preferred=list_height,
            max=MAX_AGENT_LIST_ROWS,
        )

    def _children(
        self, path: Path, fallback: dict[Path, dict[str, Any]], *,
        candidates: set[Path] | None = None,
    ) -> list[AgentEntry]:
        children = agent_child_paths(path, candidates)
        entries: list[AgentEntry] = []
        for child in children:
            metadata = self._cached_agent_metadata(child, fallback.get(child))
            entries.append(
                AgentEntry(
                    child,
                    str(metadata.get("description") or f"child {child.name}"),
                    str(metadata.get("agent_type") or "child"),
                    str(metadata.get("state") or "running"),
                )
            )
        candidate_set = set(children)
        self._metadata_cache = {
            child: cached
            for child, cached in self._metadata_cache.items()
            if child == self.root_path or child in candidate_set
        }
        return [
            entry for entry in entries if entry.state not in _TERMINAL_AGENT_STATES
        ]

    def _cached_agent_metadata(
        self, path: Path, fallback: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        lifecycle_path = path / "agent_lifecycle.json"
        try:
            stat = lifecycle_path.stat()
            key = (1, stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            try:
                stat = (path / "session_state.json").stat()
                key = (0, stat.st_mtime_ns, stat.st_size, stat.st_ino)
            except OSError:
                key = (0, 0, 0, 0)
        cached = self._metadata_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        metadata = _agent_metadata(path, fallback)
        self._metadata_cache[path] = (key, metadata)
        return metadata

    def _root_entry(self) -> AgentEntry:
        metadata = self._cached_agent_metadata(self.root_path)
        forked = self._fork_title is not None
        label = f"discussion: {self._fork_title}" if forked else "main"
        return AgentEntry(
            self.root_path,
            label,
            str(metadata.get("agent_type") or ("discussion" if forked else "main")),
            str(metadata.get("state") or "running"),
        )

    def _main_entry(self) -> AgentEntry:
        assert self._fork_main_path is not None
        return AgentEntry(self._fork_main_path, "(main)", "orchestrator", "running")

    def set_fork_context(
        self, *, title: str, main_path: Path, on_return: Callable[[], None]
    ) -> None:
        """Teach the view that this session is a discussion fork of ``main_path``."""
        self._fork_title = title
        self._fork_main_path = main_path
        self._on_return_to_main = on_return
        self.refresh(force=True)

    def focus_composer(self) -> None:
        if self._layout is not None and self._composer_buffer is not None:
            self._layout.focus(self._composer_buffer)

    def list_focused(self) -> bool:
        return self._layout is not None and self._layout.has_focus(self.list_window)

    def focus_list(self) -> None:
        self.transcript_control.capture_root_view()
        if self.list_visible and self._layout is not None:
            self._layout.focus(self.list_window)
        else:
            self.focus_composer()
        self.refresh(force=True)

    def list_back(self) -> None:
        self.back_to_parent()

    def exit_navigation(self, *, preselect_path: Path | None = None) -> None:
        del preselect_path
        if self.current_path == self.root_path:
            self.focus_composer()
            return
        self.transcript_control.activate(self.root_path)

    def focus_child_list(self) -> None:
        if self.list_visible and self._layout is not None:
            self._layout.focus(self.list_window)
        self.refresh(force=True)

    def move_selection(self, amount: int) -> None:
        if not self.entries:
            return
        self.selected_index = (self.selected_index + amount) % len(self.entries)

    def open_selected(self) -> asyncio.Task[None] | None:
        if not self.entries:
            return None
        entry = self.entries[self.selected_index]
        if (
            self._fork_main_path is not None
            and entry.path == self._fork_main_path
            and self._on_return_to_main is not None
        ):
            self._on_return_to_main()
            return None
        if entry.path == self.root_path:
            if self.child_view_active:
                self.exit_navigation(preselect_path=self.current_path)
            return None
        if entry.path == self.current_path:
            return None
        task = self.transcript_control.activate(entry.path)
        if task is None:
            self._todo_control.request(self.current_path, self._invalidate)
        return task

    def back_to_parent(self) -> None:
        if not self.child_view_active:
            self.focus_composer()
            return
        self._leave_current_view()

    def _leave_current_view(self) -> None:
        self.transcript_control.activate(self._path_stack[-2])

    def _begin_view_transition(self, generation: int) -> None:
        focused = self._layout.current_window if self._layout is not None else None
        self._navigation_generation = generation
        self._navigation_snapshot = _NavigationSnapshot(
            self.current_path,
            tuple(self._path_stack),
            tuple(self._breadcrumb_labels),
            tuple(self.entries),
            self.selected_index,
            focused,
        )

    def _rollback_view_transition(self, generation: int) -> None:
        if self._navigation_generation != generation:
            return
        snapshot = self._navigation_snapshot
        self._navigation_generation = None
        self._navigation_snapshot = None
        if snapshot is None:
            return
        self.current_path = snapshot.current_path
        self._path_stack = list(snapshot.path_stack)
        self._breadcrumb_labels = list(snapshot.breadcrumb_labels)
        self.entries = list(snapshot.entries)
        self.selected_index = snapshot.selected_index
        self._resize_list_window()
        self._switch_transcript()
        if self._layout is not None and snapshot.focused is not None:
            self._layout.focus(snapshot.focused)

    def _finish_view_transition(self, generation: int) -> None:
        if self._navigation_generation != generation:
            return
        snapshot = self._navigation_snapshot
        self._navigation_generation = None
        self._navigation_snapshot = None
        if snapshot is not None and snapshot.current_path != self.current_path:
            self._todo_control.clear(self._invalidate)
            if self.child_view_active:
                self._todo_control.request(self.current_path, self._invalidate)

    def _prepare_view_path(self, path: Path) -> _PreparedNavigation:
        """Read one target's child discovery and metadata off the TUI loop."""

        fallback = registered_child_markers(path)
        children = []
        for child in agent_child_paths(path, None):
            metadata = _agent_metadata(child, fallback.get(child))
            entry = AgentEntry(
                child,
                str(metadata.get("description") or f"child {child.name}"),
                str(metadata.get("agent_type") or "child"),
                str(metadata.get("state") or "running"),
            )
            if entry.state not in _TERMINAL_AGENT_STATES:
                children.append(entry)
        root_metadata = _agent_metadata(self.root_path)
        forked = self._fork_title is not None
        entries = [
            AgentEntry(
                self.root_path,
                f"discussion: {self._fork_title}" if forked else "main",
                str(
                    root_metadata.get("agent_type")
                    or ("discussion" if forked else "main")
                ),
                str(root_metadata.get("state") or "running"),
            )
        ]
        if path == self.root_path and self._fork_main_path is not None:
            entries.append(self._main_entry())
        entries.extend(children)
        if path == self.root_path and not children and len(entries) == 1:
            entries.clear()
        return _PreparedNavigation(tuple(entries))

    def _commit_view_path(
        self, path: Path, prepared: _PreparedNavigation, generation: int
    ) -> bool:
        """Claim and publish prepared navigation state without filesystem access."""

        if self._navigation_generation != generation:
            return False
        previous = self.current_path
        previous_index = self.selected_index
        if path in self._path_stack:
            index = self._path_stack.index(path)
            self._path_stack = self._path_stack[: index + 1]
            self._breadcrumb_labels = self._breadcrumb_labels[: index + 1]
        else:
            label = next(
                (entry.label for entry in self.entries if entry.path == path),
                path.name,
            )
            self._path_stack.append(path)
            self._breadcrumb_labels.append(label)
        # A refresh keeps the selected row; a return to root selects the child left.
        selected = (
            self.entries[previous_index].path
            if path == previous and previous_index < len(self.entries)
            else previous
        )
        self.current_path = path
        self.entries = list(prepared.entries)
        if path == self.root_path or path == previous:
            self.selected_index = next(
                (
                    index
                    for index, entry in enumerate(self.entries)
                    if entry.path == selected
                ),
                min(previous_index, max(0, len(self.entries) - 1))
                if path == previous
                else 0,
            )
        else:
            self.selected_index = min(
                previous_index, max(0, len(self.entries) - 1)
            )
        self._resize_list_window()
        self._refresh_signature = None
        self._switch_transcript()
        if path == previous:
            # A refresh of the same view keeps the user's focus.
            if self.list_focused() and not self.entries:
                self.focus_composer()
        elif self.child_view_active and self._layout is not None:
            self._layout.focus(self.transcript_window)
        else:
            self.focus_composer()
        return True

    def toggle_latest_agent(self) -> bool:
        if self.child_view_active:
            return self.transcript_control.toggle_latest_agent()
        return False

    def child_scroll(self, amount: int) -> None:
        self.transcript_control.scroll(amount)

    def child_half_page(self, amount: int) -> None:
        self.transcript_control.half_page(amount)

    def child_top(self) -> None:
        self.transcript_control.top()

    def child_bottom(self) -> None:
        self.transcript_control.bottom()
