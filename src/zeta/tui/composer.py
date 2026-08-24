"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

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
    spaces = max(2, width - len(left) - len(right))
    return FormattedText(
        [
            ("class:composer-info", left),
            ("class:composer-info", " " * spaces),
            ("class:composer-info", right),
        ]
    )


def build_key_bindings(
    *,
    on_interrupt: Callable[[], None],
    on_exit: Callable[[], None],
) -> KeyBindings:
    """Build the small key map used by the inline composer."""

    bindings = KeyBindings()

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
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

    return bindings


def history_for(path: str | Path) -> FileHistory:
    """Create a persistent history object and its parent directory."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(history_path))
