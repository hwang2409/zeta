from __future__ import annotations

import asyncio
import errno
import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tools._shared.shell_cwd import bind_shell_cwd_async
from zeta.tools.registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["normal", "removed", "replaced", "explicit"])
async def test_bind_shell_cwd_cases(tmp_path: Path, case: str) -> None:
    session = tmp_path / "session"
    session.mkdir()
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    registry = ToolRegistry(session, skill_catalog=SkillCatalog.empty())
    try:
        if case == "removed":
            session.rmdir()
        elif case == "replaced":
            session.rmdir()
            session.mkdir()
        requested = str(explicit) if case == "explicit" else None
        if case == "replaced":
            with pytest.raises(ValueError, match="replaced"):
                await bind_shell_cwd_async(registry, requested)
            return
        start, fd = await bind_shell_cwd_async(registry, requested)
        try:
            assert Path(start).is_dir()
            assert os.fstat(fd).st_ino == os.stat(start).st_ino
            if case == "removed":
                assert start == str(tmp_path)
                assert registry.bash_cwd == str(tmp_path)
                assert registry.take_cwd_recovery_notices() == (
                    (str(session), str(tmp_path)),
                )
            else:
                assert registry.take_cwd_recovery_notices() == ()
        finally:
            os.close(fd)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_bind_shell_cwd_worker_failure_and_duplicate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = ToolRegistry(session, skill_catalog=SkillCatalog.empty())
    import zeta.tools._shared.shell_cwd as module

    # test-contract: seam force the worker failure path through the public async binder.
    monkeypatch.setattr(
        module,
        "_bind_shell_cwd",
        lambda *args: (_ for _ in ()).throw(ValueError("worker failure")),
    )
    with pytest.raises(ValueError, match="worker failure"):
        await bind_shell_cwd_async(registry, None)
    assert registry.take_cwd_recovery_notices() == ()
    monkeypatch.undo()
    first = await bind_shell_cwd_async(registry, None)
    second = await bind_shell_cwd_async(registry, None)
    os.close(first[1])
    os.close(second[1])
    assert registry.take_cwd_recovery_notices() == ()
    await registry.close()


@pytest.mark.asyncio
async def test_bind_shell_cwd_worker_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = ToolRegistry(session, skill_catalog=SkillCatalog.empty())
    import zeta.tools._shared.shell_cwd as module

    release = asyncio.Event()
    original = module._bind_shell_cwd

    async def readiness() -> None:
        await asyncio.sleep(0)
        readiness.ran = True

    readiness.ran = False

    def blocked(*args: object) -> tuple[str, int, tuple[str, str] | None]:
        while not release.is_set():
            pass
        return original(*args)

    # test-contract: seam hold the worker to prove the loop remains responsive.
    monkeypatch.setattr(module, "_bind_shell_cwd", blocked)
    task = asyncio.create_task(bind_shell_cwd_async(registry, None))
    await asyncio.wait_for(readiness(), 10)
    assert readiness.ran
    release.set()
    _, fd = await asyncio.wait_for(task, 10)
    os.close(fd)
    await registry.close()


@pytest.mark.asyncio
async def test_bind_shell_cwd_cancellation_drains_and_does_not_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = ToolRegistry(session, skill_catalog=SkillCatalog.empty())
    started = asyncio.Event()
    release = asyncio.Event()
    import zeta.tools._shared.shell_cwd as module

    original = module._bind_shell_cwd
    original_thread_call = module.run_owned_thread_call
    close_done = threading.Event()
    returned_fds: list[int] = []

    def blocked(
        *args: object, **kwargs: object
    ) -> tuple[str, int, tuple[str, str] | None]:
        started.set()
        while not release.is_set():
            pass
        result = original(*args, **kwargs)
        returned_fds.append(result[1])
        return result

    async def observed_thread_call(
        acquire: Callable[[], tuple[str, int, tuple[str, str] | None]],
        close: Callable[[tuple[str, int, tuple[str, str] | None]], None],
        **kwargs: bool,
    ) -> tuple[str, int, tuple[str, str] | None]:
        def observed_close(result: tuple[str, int, tuple[str, str] | None]) -> None:
            close(result)
            close_done.set()

        return await original_thread_call(acquire, observed_close, **kwargs)

    # test-contract: seam hold the worker to prove cancellation drains its fd.
    monkeypatch.setattr(module, "_bind_shell_cwd", blocked)
    monkeypatch.setattr(module, "run_owned_thread_call", observed_thread_call)
    task = asyncio.create_task(bind_shell_cwd_async(registry, None))
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)
    assert await asyncio.to_thread(close_done.wait, 10)
    assert registry.bash_cwd == str(session)
    assert registry.take_cwd_recovery_notices() == ()
    assert returned_fds
    with pytest.raises(OSError) as error:
        os.fstat(returned_fds[0])
    assert error.value.errno == errno.EBADF
    await registry.close()


@pytest.mark.asyncio
async def test_child_registry_keeps_its_own_cwd_recovery_notice(tmp_path: Path) -> None:
    """A child's removed cwd is reported in the child, not by the parent's next result."""
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)
    store = ConversationStore(
        tmp_path / "sessions",
        session_id="child",
        cwd=str(child_dir),
        bash_cwd=str(child_dir),
    )
    root = ToolRegistry(parent_dir, skill_catalog=SkillCatalog.empty())
    child = root.clone_for_session(store, cwd=child_dir)
    try:
        child_dir.rmdir()
        _start, fd = await bind_shell_cwd_async(child, None)
        os.close(fd)

        assert root.take_cwd_recovery_notices() == ()
        assert child.take_cwd_recovery_notices() == ((str(child_dir), str(parent_dir)),)
    finally:
        await child.close()
        await root.close()
        store.close()
