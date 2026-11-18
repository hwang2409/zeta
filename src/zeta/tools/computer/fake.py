"""In-memory ``ComputerBackend`` for tests: a real frame and an action log."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO

from .backend import Button, ScreenFrame, ScreenState


def _solid_png(width: int, height: int) -> bytes:
    """A real PNG of the given size, so normalization sees the true dimensions."""

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height), (32, 96, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class FakeBackend:
    """Records every op and returns a fixed frame. Tests inspect ``actions``.

    The frame image is a real PNG matching ``width``/``height`` so the tool's
    downscaling and coordinate mapping run against true pixel dimensions.
    """

    width: int = 100
    height: int = 100
    scale: float = 1.0
    origin_x: int = 0
    origin_y: int = 0
    media_type: str = "image/png"
    image: bytes | None = None
    cursor: tuple[int, int] = (0, 0)
    locked: bool = False
    secure_input: bool = False
    frontmost_bundle_id: str | None = "com.example.app"
    actions: list[tuple] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.image is None:
            self.image = _solid_png(self.width, self.height)

    async def screenshot(self, display: int | None = None) -> ScreenFrame:
        self.actions.append(("screenshot", display))
        return ScreenFrame(
            self.image,
            self.media_type,
            self.width,
            self.height,
            self.scale,
            self.origin_x,
            self.origin_y,
        )

    async def move(self, x: int, y: int) -> None:
        self.actions.append(("move", x, y))
        self.cursor = (x, y)

    async def click(self, x: int | None, y: int | None, button: Button, count: int, modifiers: tuple[str, ...] = ()) -> None:
        if x is None or y is None:
            x, y = self.cursor
        self.actions.append(("click", x, y, button, count) if not modifiers else ("click", x, y, button, count, modifiers))
        self.cursor = (x, y)

    async def mouse_down(self, x: int, y: int, button: Button) -> None:
        self.actions.append(("mouse_down", x, y, button))
        self.cursor = (x, y)

    async def mouse_up(self, x: int, y: int, button: Button) -> None:
        self.actions.append(("mouse_up", x, y, button))
        self.cursor = (x, y)

    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        self.actions.append(("drag", list(path)))
        if path:
            self.cursor = tuple(path[-1])

    async def type_text(self, text: str) -> None:
        self.actions.append(("type_text", text))

    async def key(self, combo: str, repeat: int = 1, hold: bool = False, release: bool = False) -> None:
        self.actions.append(("key", combo) if (repeat, hold, release) == (1, False, False) else ("key", combo, repeat, hold, release))

    async def scroll(self, x: int | None, y: int | None, dx: int, dy: int, modifiers: tuple[str, ...] = ()) -> None:
        if x is None or y is None:
            x, y = self.cursor
        self.actions.append(("scroll", x, y, dx, dy) if not modifiers else ("scroll", x, y, dx, dy, modifiers))
        self.cursor = (x, y)

    async def cursor_position(self) -> tuple[int, int]:
        self.actions.append(("cursor_position",))
        return self.cursor

    async def state(self) -> ScreenState:
        self.actions.append(("state",))
        return ScreenState(self.locked, self.secure_input, self.frontmost_bundle_id)
