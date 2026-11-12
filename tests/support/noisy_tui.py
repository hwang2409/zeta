"""Launch the scripted TUI with a backend that makes every kind of stray stderr noise."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import warnings

from zeta.providers.completion import owned_completion
from zeta.tui import app as tui_app

from .tui_backend import FakeInteractiveBackend

_PENDING: list[asyncio.Task[None]] = []


class _Unraisable:
    def __del__(self) -> None:
        raise RuntimeError("ZNOISE-unraisable")


def _thread_error() -> None:
    raise RuntimeError("ZNOISE-thread")


async def _fails_on_teardown() -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        raise RuntimeError("ZNOISE-teardown") from None


def make_noise() -> None:
    warnings.warn("ZNOISE-warning", RuntimeWarning, stacklevel=1)
    logging.getLogger("zeta.noise").error("ZNOISE-logging")
    _Unraisable()
    thread = threading.Thread(target=_thread_error)
    thread.start()
    thread.join()
    subprocess.run(["sh", "-c", "echo ZNOISE-child >&2"], check=True)
    # Raises only when asyncio.run cancels it after the TUI exits.
    _PENDING.append(asyncio.get_running_loop().create_task(_fails_on_teardown()))


class _NoisyBackend(FakeInteractiveBackend):
    @owned_completion
    async def complete(self, messages, tool_schemas):  # type: ignore[override]
        make_noise()
        async for event in FakeInteractiveBackend.complete.__wrapped__(
            self, messages, tool_schemas
        ):
            yield event


def _build_backend(provider: str, model: str | None, **_kwargs: object):
    return _NoisyBackend(delay=0, model=model or "gpt-5.6-luna"), model or "gpt-5.6-luna"


def main() -> int:
    tui_app.build_backend = _build_backend
    return tui_app.main()


if __name__ == "__main__":
    raise SystemExit(main())
