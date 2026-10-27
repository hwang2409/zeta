from __future__ import annotations

from types import SimpleNamespace

from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition


def render_full_screen(session, width: int, height: int = 24) -> Screen:
    session.app.output = SimpleNamespace(
        get_size=lambda: Size(rows=height, columns=width),
    )
    screen = Screen(initial_width=width, initial_height=height)
    with set_app(session.app):
        session.layout.update_parents_relations()
        session.layout.container.write_to_screen(
            screen,
            MouseHandlers(),
            WritePosition(0, 0, width, height),
            "",
            False,
            None,
        )
    return screen
