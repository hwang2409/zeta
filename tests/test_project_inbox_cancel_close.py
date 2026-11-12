from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.core.session import SessionManager
from zeta.project_inbox import InboxError, ProjectInboxScanner
from zeta.project_registry import ProjectRegistry
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


async def _yield_until(predicate: Callable[[], bool], *, what: str) -> None:
    """Yield the event loop until predicate holds, without sleeping."""
    loop = asyncio.get_running_loop()
    for _ in range(10_000):
        if predicate():
            return
        waker = loop.create_future()
        loop.call_soon(waker.set_result, None)
        await waker
    raise AssertionError(f"{what} did not happen")


@pytest.mark.asyncio
async def test_close_finishes_when_inbox_poll_fails_during_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poll that fails while close cancels it must not strand close().

    Reproduces the p03a supervisor hang: the cancel-join re-awaited the poll
    worker and let its InboxError replace the CancelledError, so the supervisor
    looped forever and AgentLoop.close() never returned.
    """
    def _setup() -> tuple[ProjectRegistry, str, object]:
        home = tmp_path / ".zeta"
        registry = ProjectRegistry(home / "projects")
        project = registry.create_project("alpha", "alpha work")
        (home / "storage-format.json").write_text('{"writer_epoch":7}\n')
        session = SessionManager(home).create(
            provider="codex",
            model="fake",
            cwd=home,
            project_id=project.project_id,
            auto_project=False,
        )
        return registry, project.project_id, session

    registry, project_id, session = await asyncio.to_thread(_setup)

    poll_started = threading.Event()
    release = threading.Event()

    def gated_poll(self: ProjectInboxScanner) -> tuple[object, ...]:
        poll_started.set()
        release.wait(30)
        raise InboxError("project not found")

    monkeypatch.setattr(ProjectInboxScanner, "poll", gated_poll)

    loop = AgentLoop(
        FakeBackend([]),
        session.store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=project_id,
        project_registry=registry,
    )
    try:
        await loop.activate()
        supervisor = loop._project_inbox_poll_task
        # the poll thread is in flight and held, so the cancel arrives mid-poll.
        await await_or_fail(
            asyncio.to_thread(poll_started.wait),
            what="inbox poll thread starts",
        )
        close_task = asyncio.create_task(loop.close())
        # let close reach the point where it has asked the supervisor to cancel,
        # while the poll thread is still held.
        await _yield_until(
            lambda: supervisor.cancelling() > 0,
            what="close requests supervisor cancel",
        )
        # the held poll now raises project-missing inside the cancel handler.
        release.set()
        await await_or_fail(close_task, what="loop.close finishes")
        assert supervisor.done()
    finally:
        release.set()
        session.store.close()
