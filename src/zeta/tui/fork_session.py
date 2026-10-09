"""Run attention discussion forks in the same process as the main TUI.

The main runtime stays live behind the TUI while a discussion fork is shown.
Opening a discussion suspends the main *UI* only: its turns, background child
agents, background tasks, notifications, and inbox keep running on the one
event loop. The fork app runs on top, showing one runtime at a time. Returning
closes the fork and re-shows the suspended main UI with its state intact. Only
a genuine exit (Ctrl-D, ``/new``, or a real resume request) closes the main
runtime.

The controller only ever drives the app lifecycle through the small
:class:`ForkHost` interface, so it is exercised with fakes without a terminal.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol


class ForkHost(Protocol):
    """Lifecycle a TUI app exposes to the fork controller.

    ``run`` closes the app on a real exit and keeps it live when it suspends to
    open a fork; the controller never closes an app itself.
    """

    async def run(self) -> None:
        """Present this runtime's UI until it exits, returns, or opens a fork."""

    async def leave_fork(self) -> None:
        """Release the discussion fork binding; a no-op for a non-fork runtime."""

    def take_open_fork(self) -> str | None:
        """Consume and return a requested fork session id, or ``None``."""

    def attach_main(self, main: ForkHost) -> None:
        """Teach a fork which runtime stays live behind it (for its status bar)."""

    async def close(self) -> None:
        """Close this runtime, including its background work."""


async def run_fork_stack(
    main_app: ForkHost,
    build_fork: Callable[[str], Awaitable[ForkHost]],
) -> None:
    """Drive the main app and any discussion forks stacked on top of it.

    Each app's own ``run`` closes it on a real exit and leaves it live when it
    suspends to open a fork, so this controller never closes an app itself: it
    only releases a left fork's binding and re-shows its parent. The main app is
    closed by its own ``run`` when it finally exits, after this returns.
    """

    stack: list[ForkHost] = [main_app]
    try:
        while stack:
            current = stack[-1]
            await current.run()
            target = current.take_open_fork()
            if target is not None:
                fork = await build_fork(target)
                fork.attach_main(current)
                stack.append(fork)
                continue
            if len(stack) == 1:
                # The main runtime exited (it closed itself); the caller resumes
                # with a new session or a real resume as requested.
                return
            # A fork left (returned to its parent) and closed itself; release
            # its binding and re-show its parent.
            await current.leave_fork()
            stack.pop()
    except BaseException:
        # A process-level exit must close suspended parents too, newest first.
        for runtime in reversed(stack):
            await runtime.close()
        raise


__all__ = ["ForkHost", "run_fork_stack"]
