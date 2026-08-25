"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.application import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, vi_insert_mode
from prompt_toolkit.history import FileHistory
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

    @Condition
    def vi_insert_empty() -> bool:
        app = get_app()
        return vi_insert_mode() and not app.current_buffer.text

    def insert_newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
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

    @bindings.add("escape", "enter")
    def alt_enter(event: KeyPressEvent) -> None:
        insert_newline(event)

    native_escape = next(
        binding
        for binding in load_vi_bindings().bindings
        if binding.keys == (Keys.Escape,)
    )
    bindings.add(Keys.Escape)(native_escape)

    @bindings.add("up", filter=vi_insert_empty)
    def history_up(event: KeyPressEvent) -> None:
        event.current_buffer.auto_up()

    @bindings.add("down", filter=vi_insert_empty)
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
