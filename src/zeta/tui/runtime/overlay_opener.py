"""The one way a key opens an overlay."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class OverlayOpenerMixin:
    """An overlay owns the keys from the call that opens it.

    Keys typed right after an opener arrive before the overlay's first paint,
    often in the same read. If the overlay became active only after an await,
    those keys would reach the composer and Enter would send them as a prompt.
    So ``show`` makes the overlay active and focused synchronously, and slow work
    (reading history, recovering decisions) runs inside the open overlay, which
    holds its action keys while ``overlay_loading`` is true.
    """

    _overlay_load: asyncio.Future[Any] | None = None

    def _open_overlay(
        self,
        show: Callable[[], bool],
        load: Callable[[], Awaitable[Any]] | None = None,
        *,
        on_loaded: Callable[[], None] | None = None,
    ) -> asyncio.Future[Any] | None:
        """Open with ``show`` (active + focus, no await), then run ``load``."""

        # A new overlay does not wait for the load of one already closed.
        self._overlay_load = None
        if not show():
            return None
        self._invalidate_prompt()
        return None if load is None else self._hold_overlay_keys(load(), on_loaded)

    def _hold_overlay_keys(
        self, load: Awaitable[Any], on_loaded: Callable[[], None] | None = None
    ) -> asyncio.Future[Any]:
        """Hold the open overlay's action keys until ``load`` ends."""

        task = asyncio.ensure_future(load)
        self._overlay_load = task

        def ended(done: asyncio.Future[Any]) -> None:
            if self._overlay_load is done:
                self._overlay_load = None
            self._invalidate_prompt()
            if on_loaded is not None:
                on_loaded()

        task.add_done_callback(ended)
        return task

    @property
    def overlay_loading(self) -> bool:
        return self._overlay_load is not None and not self._overlay_load.done()


__all__ = ["OverlayOpenerMixin"]
