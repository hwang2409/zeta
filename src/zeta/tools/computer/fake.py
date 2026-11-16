"""In-memory ``ComputerBackend`` for tests: a fixed frame and an action log."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .backend import Button, ScreenFrame, ScreenState

# A complete 1x1 PNG, so screenshot results pass real image validation and the
# in-process normalization fast path (no worker subprocess).
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)


@dataclass
class FakeBackend:
    """Records every op and returns a fixed frame. Tests inspect ``actions``."""

    image: bytes = _PNG_1x1
    media_type: str = "image/png"
    width: int = 1
    height: int = 1
    scale: float = 1.0
    origin_x: int = 0
    origin_y: int = 0
    cursor: tuple[int, int] = (0, 0)
    locked: bool = False
    secure_input: bool = False
    frontmost_bundle_id: str | None = "com.example.app"
    actions: list[tuple] = field(default_factory=list)

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

    async def click(self, x: int, y: int, button: Button, count: int) -> None:
        self.actions.append(("click", x, y, button, count))
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

    async def key(self, combo: str) -> None:
        self.actions.append(("key", combo))

    async def scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self.actions.append(("scroll", x, y, dx, dy))
        self.cursor = (x, y)

    async def cursor_position(self) -> tuple[int, int]:
        self.actions.append(("cursor_position",))
        return self.cursor

    async def state(self) -> ScreenState:
        self.actions.append(("state",))
        return ScreenState(self.locked, self.secure_input, self.frontmost_bundle_id)
