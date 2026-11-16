"""The real macOS :class:`ComputerBackend`, backed by the signed Swift helper.

:class:`MacBackend` is a thin async translation of the helper's JSON ops (see
``helper_client.py``). It spawns the helper lazily on first use so a session that
never calls the tool never launches it, bridges the sync client with
``asyncio.to_thread``, and turns helper error codes into :class:`ComputerError`
so the tool can report a clear per-action message.

``make_mac_backend`` returns ``None`` off macOS or when the helper binary is not
present, which leaves ``registry.computer_backend`` unset and the enabled tool
reporting that no backend is available.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .backend import Button, ComputerError, ScreenFrame, ScreenState
from .helper_client import HelperClient, HelperError, HelperProtocolError, HelperTimeout

# default build output of scripts/build-computer-helper.sh, used when the env var is unset.
_DEFAULT_HELPER = (
    Path(__file__).resolve().parents[4]
    / "helpers"
    / "macos-computer"
    / "build"
    / "zeta-computer-helper"
)


def helper_path() -> Path | None:
    """The helper binary to run: ``ZETA_COMPUTER_HELPER`` if set, else the build
    default. Returns ``None`` when the resolved path does not exist."""
    override = os.environ.get("ZETA_COMPUTER_HELPER")
    path = Path(override) if override else _DEFAULT_HELPER
    return path if path.is_file() else None


def make_mac_backend() -> MacBackend | None:
    if sys.platform != "darwin":
        return None
    path = helper_path()
    if path is None:
        return None
    return MacBackend([str(path)])


class MacBackend:
    """Implements :class:`ComputerBackend` by talking to the Swift helper."""

    def __init__(self, argv: list[str]) -> None:
        self._client = HelperClient(argv)
        self._started = False
        self._start_lock = asyncio.Lock()

    async def _call(self, op: str, **args: object) -> dict:
        if not self._started:
            async with self._start_lock:
                if not self._started:
                    await asyncio.to_thread(self._client.start)
                    self._started = True
        try:
            return await asyncio.to_thread(self._client.request, op, **args)
        except HelperError as exc:
            raise ComputerError(exc.code, exc.message) from exc
        except (HelperTimeout, HelperProtocolError) as exc:
            raise ComputerError("internal", str(exc)) from exc

    async def screenshot(self, display: int | None = None) -> ScreenFrame:
        args = {} if display is None else {"display": display}
        reply = await self._call("screenshot", **args)
        return ScreenFrame(
            image=base64.b64decode(reply["png_b64"]),
            media_type="image/png",
            width=int(reply["width"]),
            height=int(reply["height"]),
            scale=float(reply["scale"]),
            origin_x=int(reply["origin_x"]),
            origin_y=int(reply["origin_y"]),
        )

    async def move(self, x: int, y: int) -> None:
        await self._call("move", x=x, y=y)

    async def click(self, x: int, y: int, button: Button, count: int) -> None:
        await self._call("click", x=x, y=y, button=button, count=count)

    async def mouse_down(self, x: int, y: int, button: Button) -> None:
        await self._call("mouse_down", x=x, y=y, button=button)

    async def mouse_up(self, x: int, y: int, button: Button) -> None:
        await self._call("mouse_up", x=x, y=y, button=button)

    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        await self._call("drag", path=[[x, y] for x, y in path])

    async def type_text(self, text: str) -> None:
        await self._call("type", text=text)

    async def key(self, combo: str) -> None:
        await self._call("key", combo=combo)

    async def scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        await self._call("scroll", x=x, y=y, dx=dx, dy=dy)

    async def cursor_position(self) -> tuple[int, int]:
        reply = await self._call("cursor_position")
        return (int(reply["x"]), int(reply["y"]))

    async def state(self) -> ScreenState:
        reply = await self._call("state")
        bundle = reply.get("frontmost_bundle_id") or None
        return ScreenState(bool(reply["locked"]), bool(reply["secure_input"]), bundle)

    def close(self) -> None:
        self._client.close()
