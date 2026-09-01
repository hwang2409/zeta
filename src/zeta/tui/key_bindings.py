"""Prompt-toolkit sessions and key bindings for the TUI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cursor_shapes import CursorShape, CursorShapeConfig
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, is_searching, vi_insert_mode
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.vi import load_vi_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys

SHIFT_ENTER_SEQUENCES = frozenset(
    {
        "\x1b[27;2;13~",
        "\x1b[27;5;13~",
        "\x1b[27;6;13~",
    }
)


class FullScreenPromptSession(PromptSession[str]):
    """Prompt session that owns the alternate screen for the whole app."""

    def _create_application(
        self, editing_mode: EditingMode, erase_when_done: bool
    ) -> Application[str]:
        application = super()._create_application(editing_mode, erase_when_done)
        application.ttimeoutlen, application.timeoutlen, application.cursor = (
            0.02,
            0.5,
            VimCursorShapeConfig(),
        )
        (
            application.full_screen,
            application.renderer.full_screen,
            application.erase_when_done,
        ) = True, True, False
        return application

    def restore_terminal(self) -> None:
        """Restore the shell viewport after prompt-toolkit exits."""

        import os
        import sys

        self.app.output.quit_alternate_screen()
        self.app.output.show_cursor()
        self.app.output.flush()
        stdout = sys.__stdout__
        if stdout.isatty():
            try:
                os.write(stdout.fileno(), b"\x1b[?1049l\x1b[?25h")
            except OSError:
                pass


class VimCursorShapeConfig(CursorShapeConfig):
    """Use a beam in insert mode and a block in every other vi mode."""

    def get_cursor_shape(self, application: Application[Any]) -> CursorShape:
        if getattr(application, "editing_mode", None) is not EditingMode.VI:
            return CursorShape._NEVER_CHANGE
        if getattr(application.vi_state, "input_mode", None) in {
            InputMode.INSERT,
            InputMode.INSERT_MULTIPLE,
        }:
            return CursorShape.BEAM
        return CursorShape.BLOCK


def build_key_bindings(
    *,
    on_interrupt: Callable[[], None],
    on_exit: Callable[[], None],
    on_submit: Callable[[str], None] | None = None,
    on_paste: Callable[[KeyPressEvent], None] | None = None,
    on_page_up: Callable[[], None] | None = None,
    on_page_down: Callable[[], None] | None = None,
    on_search_start: Callable[[], None] | None = None,
    search_active: Callable[[], bool] | None = None,
    on_search_input: Callable[[str], None] | None = None,
    on_search_backspace: Callable[[], None] | None = None,
    on_search_next: Callable[[], None] | None = None,
    on_search_previous: Callable[[], None] | None = None,
    on_search_end: Callable[[], None] | None = None,
    on_previous_user: Callable[[], None] | None = None,
    on_next_user: Callable[[], None] | None = None,
    on_toggle_agent: Callable[[], None] | None = None,
    on_retry: Callable[[], None] | None = None,
    retry_available: Callable[[], bool] | None = None,
    on_undo: Callable[[], None] | None = None,
    append_history: bool = True,
    on_approve: Callable[[], None] | None = None,
    on_deny: Callable[[], None] | None = None,
    approval_active: Callable[[], bool] | None = None,
    on_scroll_up: Callable[[], None] | None = None,
    on_scroll_down: Callable[[], None] | None = None,
) -> KeyBindings:
    """Build the small key map used by the full-screen composer."""

    bindings = KeyBindings()
    escape_chord_pending = False
    escape_chord_cursor_position: int | None = None
    history_navigation_active = False
    history_navigation_buffer: Buffer | None = None
    suppress_history_detach = False
    search_buffer = Buffer(name="TRANSCRIPT_SEARCH")
    search_input_active = False

    def track_history_buffer(buffer: Buffer) -> None:
        nonlocal history_navigation_active, history_navigation_buffer
        if history_navigation_buffer is buffer:
            return
        history_navigation_buffer = buffer

        def detach_history_navigation(_buffer: Buffer) -> None:
            nonlocal history_navigation_active
            if not suppress_history_detach:
                history_navigation_active = False

        buffer.on_text_changed += detach_history_navigation

    @Condition
    def vi_insert_history_navigation() -> bool:
        app = get_app()
        buffer = app.current_buffer
        track_history_buffer(buffer)
        return vi_insert_mode()

    @Condition
    def emacs_history_navigation() -> bool:
        app = get_app()
        track_history_buffer(app.current_buffer)
        return app.editing_mode is EditingMode.EMACS

    @Condition
    def full_screen_mode() -> bool:
        return get_app().full_screen

    @Condition
    def retry_ready() -> bool:
        return (
            on_retry is not None
            and (retry_available is None or retry_available())
        )

    @Condition
    def transcript_search_mode() -> bool:
        return (
            full_screen_mode()
            and search_active is not None
            and search_active()
            and not is_searching()
        )

    @Condition
    def transcript_search_input_mode() -> bool:
        return transcript_search_mode() and search_input_active

    @Condition
    def transcript_search_navigation_mode() -> bool:
        return transcript_search_mode() and not search_input_active

    def insert_newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
        nonlocal escape_chord_cursor_position, escape_chord_pending
        if escape_chord_pending:
            escape_chord_pending = False
            if escape_chord_cursor_position is not None:
                event.current_buffer.cursor_position = escape_chord_cursor_position
            escape_chord_cursor_position = None
            event.app.vi_state.input_mode = InputMode.INSERT
            insert_newline(event)
            return
        if event.data in SHIFT_ENTER_SEQUENCES:
            insert_newline(event)
            return
        if on_submit is not None:
            if append_history:
                event.current_buffer.append_to_history()
            on_submit(event.current_buffer.text)
            event.current_buffer.reset()
        else:
            event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def newline(event: KeyPressEvent) -> None:
        insert_newline(event)

    @bindings.add("enter", filter=is_searching, eager=True)
    def accept_history_search(event: KeyPressEvent) -> None:
        del event
        from prompt_toolkit.search import accept_search

        accept_search()

    if on_retry is not None:

        @bindings.add("c-y", filter=retry_ready, eager=True)
        def retry(event: KeyPressEvent) -> None:
            del event
            on_retry()

    if on_undo is not None:

        @bindings.add("c-u", eager=True)
        def undo(event: KeyPressEvent) -> None:
            del event
            on_undo()

    if on_paste is not None:

        @bindings.add("c-v")
        def paste(event: KeyPressEvent) -> None:
            on_paste(event)

    @bindings.add("escape", "enter", filter=~full_screen_mode)
    def alt_enter(event: KeyPressEvent) -> None:
        insert_newline(event)

    native_escape = next(
        binding
        for binding in load_vi_bindings().bindings
        if binding.keys == (Keys.Escape,)
    )

    bindings.add(Keys.Escape, filter=native_escape.filter & ~full_screen_mode)(native_escape)

    @bindings.add(Keys.Escape, filter=native_escape.filter & full_screen_mode, eager=True)
    def escape(event: KeyPressEvent) -> None:
        nonlocal escape_chord_cursor_position, escape_chord_pending, search_input_active
        if transcript_search_mode():
            if on_search_end is not None:
                search_buffer.reset()
                search_input_active = False
                on_search_end()
            return
        escape_chord_cursor_position = event.current_buffer.cursor_position
        native_escape.call(event)
        next_key = next(iter(event.key_processor.input_queue), None)
        escape_chord_pending = next_key is not None and next_key.key == Keys.Enter
        if not escape_chord_pending:
            escape_chord_cursor_position = None

    @bindings.add(
        "up", filter=vi_insert_history_navigation | emacs_history_navigation
    )
    def history_up(event: KeyPressEvent) -> None:
        nonlocal history_navigation_active, suppress_history_detach
        buffer = event.current_buffer
        if not history_navigation_active and buffer.text:
            if buffer.document.cursor_position_row > 0:
                buffer.auto_up()
            return
        suppress_history_detach = True
        try:
            buffer.history_backward()
        finally:
            suppress_history_detach = False
        history_navigation_active = buffer.text != ""

    @bindings.add(
        "down", filter=vi_insert_history_navigation | emacs_history_navigation
    )
    def history_down(event: KeyPressEvent) -> None:
        nonlocal history_navigation_active, suppress_history_detach
        buffer = event.current_buffer
        if not history_navigation_active:
            if buffer.document.cursor_position_row < buffer.document.line_count - 1:
                buffer.auto_down()
            return
        suppress_history_detach = True
        try:
            buffer.history_forward()
            buffer.cursor_position = len(buffer.text)
        finally:
            suppress_history_detach = False
        history_navigation_active = buffer.text != ""

    @bindings.add("c-c")
    def interrupt(event: KeyPressEvent) -> None:
        on_interrupt()
        event.current_buffer.reset()

    @bindings.add("c-d")
    def exit_prompt(event: KeyPressEvent) -> None:
        on_exit()
        event.app.exit(exception=EOFError())

    if on_page_up is not None:

        @bindings.add("pageup")
        def page_up(event: KeyPressEvent) -> None:
            del event
            on_page_up()

    if on_page_down is not None:

        @bindings.add("pagedown")
        def page_down(event: KeyPressEvent) -> None:
            del event
            on_page_down()

    if on_search_start is not None:

        @bindings.add("c-f", filter=full_screen_mode & ~transcript_search_mode, eager=True)
        def start_transcript_search(event: KeyPressEvent) -> None:
            nonlocal search_input_active
            del event
            search_buffer.reset()
            search_input_active = True
            on_search_start()

    if on_search_end is not None:

        @bindings.add(Keys.Escape, filter=transcript_search_mode, eager=True)
        def end_transcript_search(event: KeyPressEvent) -> None:
            nonlocal search_input_active
            del event
            search_buffer.reset()
            search_input_active = False
            on_search_end()

    if on_search_input is not None:

        @bindings.add(Keys.Any, filter=transcript_search_input_mode, eager=True)
        def transcript_search_input(event: KeyPressEvent) -> None:
            if event.data:
                search_buffer.insert_text(event.data)
                on_search_input(search_buffer.text)

    if on_search_backspace is not None:

        @bindings.add("backspace", filter=transcript_search_input_mode, eager=True)
        def transcript_search_backspace(event: KeyPressEvent) -> None:
            del event
            search_buffer.delete_before_cursor()
            on_search_backspace()

        @bindings.add("c-h", filter=transcript_search_input_mode, eager=True)
        def transcript_search_backspace_ctrl_h(event: KeyPressEvent) -> None:
            del event
            search_buffer.delete_before_cursor()
            on_search_backspace()

    if on_search_next is not None:

        @bindings.add("enter", filter=transcript_search_input_mode, eager=True)
        def commit_transcript_search(event: KeyPressEvent) -> None:
            nonlocal search_input_active
            del event
            search_input_active = False
            on_search_next()

        @bindings.add("n", filter=transcript_search_navigation_mode, eager=True)
        def next_transcript_match(event: KeyPressEvent) -> None:
            del event
            on_search_next()

        @bindings.add("enter", filter=transcript_search_navigation_mode, eager=True)
        def next_transcript_match_enter(event: KeyPressEvent) -> None:
            del event
            on_search_next()

    if on_search_previous is not None:

        @bindings.add("N", filter=transcript_search_navigation_mode, eager=True)
        def previous_transcript_match(event: KeyPressEvent) -> None:
            del event
            on_search_previous()

    if on_previous_user is not None:

        @bindings.add(Keys.ControlUp, filter=full_screen_mode, eager=True)
        def previous_user(event: KeyPressEvent) -> None:
            del event
            on_previous_user()

    if on_next_user is not None:

        @bindings.add(Keys.ControlDown, filter=full_screen_mode, eager=True)
        def next_user(event: KeyPressEvent) -> None:
            del event
            on_next_user()

    if on_toggle_agent is not None:

        @bindings.add("c-x", "c-o")
        def toggle_agent(event: KeyPressEvent) -> None:
            del event
            on_toggle_agent()

    if on_approve is not None and on_deny is not None:

        @Condition
        def approval_pending() -> bool:
            # Only while the composer is empty: otherwise the "n" and "y" in a
            # typed "deny 3" would resolve requests instead of reaching the
            # buffer, and the second keystroke would answer the next request.
            return (
                approval_active is not None
                and approval_active()
                and not get_app().current_buffer.text
            )

        # Transcript search owns n and N while it is open, and history search
        # takes the keyboard whole, so the shortcut stands down for both.
        answering = approval_pending & ~transcript_search_mode & ~is_searching

        @bindings.add("y", filter=answering, eager=True)
        def approve(event: KeyPressEvent) -> None:
            del event
            on_approve()

        @bindings.add("n", filter=answering, eager=True)
        def deny(event: KeyPressEvent) -> None:
            del event
            on_deny()

    if on_scroll_up is not None:

        @bindings.add(Keys.ScrollUp)
        def scroll_up(event: KeyPressEvent) -> None:
            del event
            on_scroll_up()

    if on_scroll_down is not None:

        @bindings.add(Keys.ScrollDown)
        def scroll_down(event: KeyPressEvent) -> None:
            del event
            on_scroll_down()

    return bindings
