"""The one seam between the ``computer`` tool and the operating system.

A ``ComputerBackend`` performs screen capture and input on a real desktop. The
macOS helper client (slice 2) and the test ``FakeBackend`` both implement it.
The methods mirror the helper's JSON ops in the lane shared-interface section,
so the real client stays a thin translation and coordinates line up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Button = Literal["left", "right", "middle"]


@dataclass(frozen=True, slots=True)
class ScreenFrame:
    """One captured frame: encoded image bytes plus the geometry slice 3 maps.

    ``scale`` is backing-store pixels per point; ``origin_x``/``origin_y`` are
    the display origin in the global point space. They are recorded now and
    unused until slice 3 maps model coordinates back to the backend.
    """

    image: bytes
    media_type: str  # "image/png" or "image/jpeg"
    width: int
    height: int
    scale: float
    origin_x: int
    origin_y: int


@dataclass(frozen=True, slots=True)
class ScreenState:
    locked: bool
    secure_input: bool
    frontmost_bundle_id: str | None


class ComputerBackend(Protocol):
    """Capture and input on one desktop. Coordinates are backend points."""

    async def screenshot(self, display: int | None = None) -> ScreenFrame: ...

    async def move(self, x: int, y: int) -> None: ...

    async def click(self, x: int, y: int, button: Button, count: int) -> None: ...

    async def mouse_down(self, x: int, y: int, button: Button) -> None: ...

    async def mouse_up(self, x: int, y: int, button: Button) -> None: ...

    async def drag(self, path: Sequence[tuple[int, int]]) -> None: ...

    async def type_text(self, text: str) -> None: ...

    async def key(self, combo: str) -> None: ...

    async def scroll(self, x: int, y: int, dx: int, dy: int) -> None: ...

    async def cursor_position(self) -> tuple[int, int]: ...

    async def state(self) -> ScreenState: ...
