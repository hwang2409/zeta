"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.cursor_shapes import CursorShape, CursorShapeConfig
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, vi_insert_mode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.vi import load_vi_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from rich.text import Text


SHIFT_ENTER_SEQUENCES = frozenset(
    {
        "\x1b[27;2;13~",
        "\x1b[27;5;13~",
        "\x1b[27;6;13~",
    }
)


class VimCursorShapeConfig(CursorShapeConfig):
    """Use a beam in insert mode and a block in every other vi mode."""

    def get_cursor_shape(self, application: object) -> CursorShape:
        if getattr(application, "editing_mode", None) is not EditingMode.VI:
            return CursorShape._NEVER_CHANGE
        if getattr(application.vi_state, "input_mode", None) in {
            InputMode.INSERT,
            InputMode.INSERT_MULTIPLE,
        }:
            return CursorShape.BEAM
        return CursorShape.BLOCK


def vim_state_label(vim_mode: bool) -> str | None:
    """Return the native prompt-toolkit vi state for the footer."""

    if not vim_mode:
        return None
    try:
        app = get_app()
    except RuntimeError:
        return "INSERT"
    if getattr(app, "editing_mode", EditingMode.VI) is not EditingMode.VI:
        return None
    buffer = getattr(app, "current_buffer", None)
    if buffer is not None and buffer.selection_state is not None:
        return "VISUAL"
    mode = getattr(getattr(app, "vi_state", None), "input_mode", None)
    return "NORMAL" if mode is InputMode.NAVIGATION else "INSERT"


def status_formatted_text(status: Text) -> FormattedText:
    """Convert Rich status spans into prompt-toolkit fragments."""

    fragments: list[tuple[str, str]] = []
    boundaries = {0, len(status.plain)}
    for span in status.spans:
        boundaries.update((span.start, span.end))
    ordered_boundaries = sorted(boundaries)
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        styles = ["class:status-bar"]
        if status.style:
            styles.append(str(status.style))
        styles.extend(
            str(span.style)
            for span in status.spans
            if span.start <= start and end <= span.end
        )
        fragments.append((" ".join(styles), status.plain[start:end]))
    return FormattedText(fragments)


def parse_input(value: str) -> str | None:
    """Return a usable user turn, or None for blank input."""

    stripped = value.strip()
    return stripped or None


def build_key_bindings(
    *,
    on_interrupt: Callable[[], None],
    on_exit: Callable[[], None],
    on_submit: Callable[[str], None] | None = None,
    on_page_up: Callable[[], None] | None = None,
    on_page_down: Callable[[], None] | None = None,
) -> KeyBindings:
    """Build the small key map used by the full-screen composer."""

    bindings = KeyBindings()
    escape_chord_pending = False

    @Condition
    def vi_insert_history_navigation() -> bool:
        app = get_app()
        buffer = app.current_buffer
        return vi_insert_mode() and (
            not buffer.text
            or buffer.working_index < len(buffer._working_lines) - 1
        )

    def insert_newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
        nonlocal escape_chord_pending
        if escape_chord_pending:
            escape_chord_pending = False
            insert_newline(event)
            return
        if event.data in SHIFT_ENTER_SEQUENCES:
            insert_newline(event)
            return
        if on_submit is not None:
            event.current_buffer.append_to_history()
            on_submit(event.current_buffer.text)
            event.current_buffer.reset()
        else:
            event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def newline(event: KeyPressEvent) -> None:
        insert_newline(event)

    native_escape = next(
        binding
        for binding in load_vi_bindings().bindings
        if binding.keys == (Keys.Escape,)
    )

    def clear_escape_chord() -> None:
        nonlocal escape_chord_pending
        escape_chord_pending = False

    @bindings.add(Keys.Escape, filter=native_escape.filter, eager=True)
    def escape(event: KeyPressEvent) -> None:
        nonlocal escape_chord_pending
        native_escape.call(event)
        escape_chord_pending = True
        loop = event.app.loop
        if loop is not None:
            loop.call_later(0.1, clear_escape_chord)

    @bindings.add("up", filter=vi_insert_history_navigation)
    def history_up(event: KeyPressEvent) -> None:
        event.current_buffer.auto_up()

    @bindings.add("down", filter=vi_insert_history_navigation)
    def history_down(event: KeyPressEvent) -> None:
        event.current_buffer.auto_down()

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

    return bindings


def history_for(path: str | Path) -> FileHistory:
    """Create a persistent history object and its parent directory."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(history_path))
