"""Prompt-toolkit sessions and key bindings for the TUI.

The action-name → key remap layer lives at the top of this file (users author
it via the ``[keybindings]`` table in ``settings.toml`` — see ZETA-73). It is
kept here rather than a sibling module so the ``tui/`` package stays within
its per-directory file cap (see :mod:`tests.test_module_limits`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Final

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cursor_shapes import CursorShape, CursorShapeConfig
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import (
    Condition,
    has_completions,
    is_searching,
    vi_insert_mode,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.vi import load_vi_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import ALL_KEYS, Keys
from prompt_toolkit.output import Output

# --- keybinding remap layer ------------------------------------------------


class KeybindingError(Exception):
    """Raised when a remap file names an unknown action or an unparseable key."""


# Canonical action names, in the order they appear in ``build_key_bindings``.
# The value is the default key sequence prompt-toolkit expects for the action;
# the same map doubles as the "valid actions" set for error messages.
DEFAULTS: Final[Mapping[str, tuple[str, ...]]] = {
    "submit": ("enter",),
    "insert-newline": ("c-j",),
    "interrupt": ("c-c",),
    "exit": ("c-d",),
    "retry": ("c-y",),
    "undo": ("c-u",),
    "paste": ("c-v",),
    "page-up": ("pageup",),
    "page-down": ("pagedown",),
    "search-start": ("c-f",),
    "transcript-previous-user": ("c-up",),
    "transcript-next-user": ("c-down",),
    "toggle-agent": ("c-x", "c-o"),
    "plan-mode-toggle": ("s-tab",),
    # Bash-style external-editor chord: opens $EDITOR (fallback vi) on the
    # composer buffer via prompt-toolkit's ``Buffer.open_in_editor``, which
    # handles terminal state around the round-trip. Works in both vi and
    # emacs editing modes; the vi ``v``-in-normal shortcut still applies.
    "open-editor": ("c-x", "c-e"),
}

ACTIONS: Final[frozenset[str]] = frozenset(DEFAULTS)


_MODIFIER_ALIASES: Final[Mapping[str, str]] = {
    "ctrl": "c",
    "control": "c",
    "c": "c",
    "shift": "s",
    "s": "s",
}


def parse_key_spec(spec: str) -> tuple[str, ...]:
    """Turn ``spec`` into a prompt-toolkit key tuple.

    Accepts ``c-r`` / ``ctrl-r`` / ``ctrl+r`` / ``Ctrl-R``, chords like
    ``c-x c-o``, and named keys such as ``pageup`` / ``f5`` / ``escape``.
    Raises :class:`KeybindingError` when any segment does not resolve.
    """

    if not isinstance(spec, str) or not spec.strip():
        raise KeybindingError("empty key spec")
    normalized: list[str] = []
    for segment in spec.split():
        canonical = _normalize_key_segment(segment)
        if canonical not in ALL_KEYS:
            raise KeybindingError(f"unknown key {segment!r}")
        normalized.append(canonical)
    return tuple(normalized)


def _normalize_key_segment(segment: str) -> str:
    lowered = segment.lower().replace("+", "-").replace("_", "-")
    if not lowered:
        raise KeybindingError("empty key segment")
    parts = lowered.split("-")
    if any(part == "" for part in parts):
        raise KeybindingError(f"invalid key spec {segment!r}")
    if len(parts) == 1:
        return parts[0]
    canonical: list[str] = []
    for modifier in parts[:-1]:
        if modifier not in _MODIFIER_ALIASES:
            raise KeybindingError(
                f"unknown modifier {modifier!r} in {segment!r}"
            )
        canonical.append(_MODIFIER_ALIASES[modifier])
    canonical.append(parts[-1])
    return "-".join(canonical)


def resolve_keybindings(
    user_map: Mapping[str, str] | None,
) -> dict[str, tuple[str, ...]]:
    """Return the final ``action -> key tuple`` map.

    Defaults from :data:`DEFAULTS` apply first; each user entry replaces the
    default for that action. Unknown action names or unparseable key specs
    raise :class:`KeybindingError` naming the valid actions so a typo does
    not silently disable a shortcut.
    """

    resolved: dict[str, tuple[str, ...]] = dict(DEFAULTS)
    if user_map:
        for action, spec in user_map.items():
            if action not in ACTIONS:
                valid = ", ".join(sorted(ACTIONS))
                raise KeybindingError(
                    f"keybindings: unknown action {action!r}; valid: {valid}"
                )
            try:
                resolved[action] = parse_key_spec(spec)
            except KeybindingError as exc:
                raise KeybindingError(
                    f"keybindings: {action!r} = {spec!r}: {exc}"
                ) from exc
    # Collision detection: a remap that lands on a key already owned by
    # another action means prompt-toolkit registers two handlers on the same
    # key and fires both — silent shadowing the user has no way to spot.
    # Fail loudly so a one-line typo does not turn `ctrl-y` into a coin flip
    # between submit and retry.
    reverse: dict[tuple[str, ...], list[str]] = {}
    for action, key_tuple in resolved.items():
        reverse.setdefault(key_tuple, []).append(action)
    for key_tuple, actions in reverse.items():
        if len(actions) > 1:
            a, b = sorted(actions)[:2]
            key_display = " ".join(key_tuple)
            raise KeybindingError(
                f"keybindings: {key_display!r} is bound to both "
                f"{a!r} and {b!r}; each key may only bind one action"
            )
    return resolved


# --- prompt-toolkit session/key wiring -------------------------------------

SHIFT_ENTER_SEQUENCES = frozenset(
    {
        "\x1b[27;2;13~",
        "\x1b[27;5;13~",
        "\x1b[27;6;13~",
    }
)


# Mouse reporting modes prompt-toolkit turns on, cleared again by hand so a
# hard exit cannot leave the shell swallowing clicks and selections.
MOUSE_OFF = b"\x1b[?1000l\x1b[?1003l\x1b[?1015l\x1b[?1006l"


def _enable_wheel_reporting(output: Output) -> None:
    """Report clicks and wheel ticks, but not every pointer move.

    prompt-toolkit's own `enable_mouse_support` also turns on ?1003h, which
    streams an event for every step of pointer motion across the terminal.
    The transcript only needs the wheel, so that traffic is pure overhead.
    """

    output.write_raw("\x1b[?1000h")  # click and wheel reporting
    output.write_raw("\x1b[?1015h")  # urxvt extended coordinates
    output.write_raw("\x1b[?1006h")  # SGR extended coordinates


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
        application.output.enable_mouse_support = partial(
            _enable_wheel_reporting, application.output
        )
        return application

    def restore_terminal(self) -> None:
        """Restore the shell viewport after prompt-toolkit exits."""

        import os
        import sys

        self.app.output.disable_mouse_support()
        self.app.output.quit_alternate_screen()
        self.app.output.show_cursor()
        self.app.output.flush()
        stdout = sys.__stdout__
        if stdout.isatty():
            try:
                os.write(stdout.fileno(), MOUSE_OFF + b"\x1b[?1049l\x1b[?25h")
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
    on_plan_toggle: Callable[[], None] | None = None,
    on_scroll_up: Callable[[], None] | None = None,
    on_scroll_down: Callable[[], None] | None = None,
    on_picker_move: Callable[[int], None] | None = None,
    on_picker_select: Callable[[], None] | None = None,
    on_picker_cancel: Callable[[], None] | None = None,
    picker_active: Callable[[], bool] | None = None,
    key_remap: Mapping[str, str] | None = None,
) -> KeyBindings:
    """Build the small key map used by the full-screen composer."""

    resolved_keys = resolve_keybindings(key_remap)
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

    @bindings.add(*resolved_keys["submit"])
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

    @bindings.add(*resolved_keys["insert-newline"])
    def newline(event: KeyPressEvent) -> None:
        insert_newline(event)

    @bindings.add("enter", filter=is_searching, eager=True)
    def accept_history_search(event: KeyPressEvent) -> None:
        del event
        from prompt_toolkit.search import accept_search

        accept_search()

    if on_retry is not None:

        @bindings.add(*resolved_keys["retry"], filter=retry_ready, eager=True)
        def retry(event: KeyPressEvent) -> None:
            del event
            on_retry()

    if on_undo is not None:

        @bindings.add(*resolved_keys["undo"], eager=True)
        def undo(event: KeyPressEvent) -> None:
            del event
            on_undo()

    if on_paste is not None:

        @bindings.add(*resolved_keys["paste"])
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

    history_navigation_filter = (
        vi_insert_history_navigation | emacs_history_navigation
    ) & ~has_completions

    @bindings.add("up", filter=history_navigation_filter)
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

    @bindings.add("down", filter=history_navigation_filter)
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

    @bindings.add(*resolved_keys["interrupt"])
    def interrupt(event: KeyPressEvent) -> None:
        on_interrupt()
        event.current_buffer.reset()

    @bindings.add(*resolved_keys["exit"])
    def exit_prompt(event: KeyPressEvent) -> None:
        on_exit()
        event.app.exit(exception=EOFError())

    if on_page_up is not None:

        @bindings.add(*resolved_keys["page-up"])
        def page_up(event: KeyPressEvent) -> None:
            del event
            on_page_up()

    if on_page_down is not None:

        @bindings.add(*resolved_keys["page-down"])
        def page_down(event: KeyPressEvent) -> None:
            del event
            on_page_down()

    if on_search_start is not None:

        @bindings.add(*resolved_keys["search-start"], filter=full_screen_mode & ~transcript_search_mode, eager=True)
        def start_transcript_search(event: KeyPressEvent) -> None:
            nonlocal search_input_active
            event.current_buffer.cancel_completion()
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

        @bindings.add(*resolved_keys["transcript-previous-user"], filter=full_screen_mode, eager=True)
        def previous_user(event: KeyPressEvent) -> None:
            del event
            on_previous_user()

    if on_next_user is not None:

        @bindings.add(*resolved_keys["transcript-next-user"], filter=full_screen_mode, eager=True)
        def next_user(event: KeyPressEvent) -> None:
            del event
            on_next_user()

    if on_toggle_agent is not None:

        @bindings.add(*resolved_keys["toggle-agent"])
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

    if (
        on_picker_move is not None
        and on_picker_select is not None
        and on_picker_cancel is not None
    ):

        @Condition
        def picker_open() -> bool:
            # Only while the composer is empty: a typed message or a full
            # "/model <name>" keeps its own enter and arrow keys.
            return (
                picker_active is not None
                and picker_active()
                and not get_app().current_buffer.text
            )

        picking = (
            picker_open & ~transcript_search_mode & ~is_searching & ~has_completions
        )

        @bindings.add("up", filter=picking, eager=True)
        def picker_up(event: KeyPressEvent) -> None:
            del event
            on_picker_move(-1)

        @bindings.add("down", filter=picking, eager=True)
        def picker_down(event: KeyPressEvent) -> None:
            del event
            on_picker_move(1)

        @bindings.add("enter", filter=picking, eager=True)
        def picker_select(event: KeyPressEvent) -> None:
            del event
            on_picker_select()

        @bindings.add(Keys.Escape, filter=picking, eager=True)
        def picker_cancel(event: KeyPressEvent) -> None:
            del event
            on_picker_cancel()

    @bindings.add(*resolved_keys["open-editor"], eager=True)
    def open_external_editor(event: KeyPressEvent) -> None:
        event.current_buffer.open_in_editor()

    if on_plan_toggle is not None:
        # Shift+Tab cycles modes in the harnesses people arrive from, so it
        # toggles plan mode here. It stands down while the completion menu is
        # open, where the terminal's own back-tab walks the list, and during
        # either search, which takes the keyboard whole. (``s-tab`` is
        # prompt-toolkit's alias for ``BackTab``.)
        @bindings.add(
            *resolved_keys["plan-mode-toggle"],
            filter=~has_completions & ~transcript_search_mode & ~is_searching,
            eager=True,
        )
        def toggle_plan_mode(event: KeyPressEvent) -> None:
            del event
            on_plan_toggle()

    # Terminals report the wheel as mouse events, which the transcript window
    # handles itself; these keys only exist for the few that send \x1b[62~ and
    # \x1b[63~ instead. They still earn their place: prompt-toolkit's own
    # binding for them feeds `up`/`down` back in, which the composer would
    # take as history navigation.
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
