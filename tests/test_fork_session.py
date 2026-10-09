from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from zeta.tui.fork_session import run_fork_stack


@dataclass
class _Resource:
    """A live piece of main-runtime work that a discussion must not disturb."""

    alive: bool = True

    def stop(self) -> None:
        self.alive = False


@dataclass
class _FakeApp:
    name: str
    # Scripted UI results: each run() consumes one. A str opens that fork id;
    # None means the app exited (return-to-parent / real exit).
    script: list[str | None]
    is_fork: bool = False
    # Main-runtime work that stays live across a discussion.
    child: _Resource = field(default_factory=_Resource)
    task: _Resource = field(default_factory=_Resource)
    turn: _Resource = field(default_factory=_Resource)
    notifications: int = 0
    runs: int = 0
    closed: bool = False
    left: bool = False
    main: _FakeApp | None = None
    baseline: int = 0
    _pending_fork: str | None = None

    async def run(self) -> None:
        self.runs += 1
        self._pending_fork = self.script.pop(0) if self.script else None
        # A real exit or a return closes the app itself; suspending to open a
        # fork leaves it live (and the controller re-runs it later).
        if self._pending_fork is None:
            self.closed = True

    async def leave_fork(self) -> None:
        self.left = True

    def take_open_fork(self) -> str | None:
        target, self._pending_fork = self._pending_fork, None
        return target

    def attach_main(self, main: _FakeApp) -> None:
        self.main = main
        self.baseline = main.notifications

    @property
    def main_activity_pending(self) -> bool:
        return self.main is not None and self.main.notifications > self.baseline


def test_opening_discussion_keeps_main_runtime_live_and_untouched() -> None:
    main = _FakeApp("main", script=["fork-1", None])
    fork = _FakeApp("fork", script=[None], is_fork=True)

    built: list[str] = []

    async def build_fork(target: str) -> _FakeApp:
        built.append(target)
        # The discussion opens while main has a running child, task, and turn.
        assert (main.child.alive, main.task.alive, main.turn.alive) == (True,) * 3
        # The main runtime is suspended, not closed, to show the fork.
        assert main.closed is False
        return fork

    asyncio.run(run_fork_stack(main, build_fork))

    assert built == ["fork-1"]
    # Main kept its live work through the whole discussion and after return.
    assert (main.child.alive, main.task.alive, main.turn.alive) == (True,) * 3
    # Main ran twice: before the discussion and again after returning.
    assert main.runs == 2
    # The fork was released and closed itself on return. Main closed only when
    # it finally exited (its second run), never to show the fork.
    assert (fork.left, fork.closed) == (True, True)
    assert main.left is False
    assert main.closed is True


def test_main_notification_during_discussion_is_seen_by_the_fork() -> None:
    main = _FakeApp("main", script=["fork-1", None])
    fork = _FakeApp("fork", script=[None], is_fork=True)

    async def build_fork(target: str) -> _FakeApp:
        return fork

    async def driver() -> None:
        task = asyncio.ensure_future(run_fork_stack(main, build_fork))
        # Let the stack open the fork, then deliver a main-runtime notification
        # while the discussion is shown.
        await asyncio.sleep(0)
        main.notifications += 1
        assert fork.main_activity_pending is True
        await task

    asyncio.run(driver())


def test_forks_can_stack_and_unwind_newest_first() -> None:
    main = _FakeApp("main", script=["fork-1", None])
    inner = _FakeApp("inner", script=["fork-2", None], is_fork=True)
    deepest = _FakeApp("deepest", script=[None], is_fork=True)
    built = {"fork-1": inner, "fork-2": deepest}

    async def build_fork(target: str) -> _FakeApp:
        return built[target]

    asyncio.run(run_fork_stack(main, build_fork))

    assert (inner.left, inner.closed) == (True, True)
    assert (deepest.left, deepest.closed) == (True, True)
    assert main.closed is True
    # main runs 2x, inner runs 2x (open deepest, then after return), deepest 1x.
    assert (main.runs, inner.runs, deepest.runs) == (2, 2, 1)
