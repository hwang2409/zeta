"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from rich.cells import cell_len
from rich.text import Text


def _fit(value: str, width: int) -> str:
    if width <= 0:
        return ""
    text = Text(value, no_wrap=True, overflow="ellipsis")
    text.truncate(width, overflow="ellipsis")
    return text.plain

def parse_input(value: str) -> str | None:
    """Return a usable user turn, or None for blank input."""

    stripped = value.strip()
    return stripped or None


def format_composer_info(
    provider: str,
    model: str,
    *,
    width: int | None = None,
) -> FormattedText:
    """Build the dim identity row below the input line."""

    left = "zeta"
    right = f"{provider} · {model}"
    if width is None:
        return FormattedText(
            [
                ("class:composer-info", left),
                ("class:composer-info", "  "),
                ("class:composer-info", right),
            ]
        )
    if width <= 0:
        return FormattedText()
    if cell_len(left) + 2 + cell_len(right) <= width:
        spaces = width - cell_len(left) - cell_len(right)
        return FormattedText(
            [
                ("class:composer-info", left),
                ("class:composer-info", " " * spaces),
                ("class:composer-info", right),
            ]
        )
    if width <= cell_len(left) + 2:
        return FormattedText([("class:composer-info", _fit(left, width))])
    right = _fit(right, width - cell_len(left) - 2).rstrip(" ·")
    value = f"{left}  {right}" if right else _fit(left, width)
    return FormattedText(
        [("class:composer-info", _fit(value, width))]
    )


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

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
        if on_submit is not None:
            event.current_buffer.append_to_history()
            on_submit(event.current_buffer.text)
            event.current_buffer.reset()
        else:
            event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

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
