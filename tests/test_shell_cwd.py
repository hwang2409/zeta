import asyncio
import os
import threading
from pathlib import Path

import pytest

import zeta.tools._shared.shell_cwd as shell_cwd
from tests.support.owned import owned_app
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


async def _wait_for_bind(
    registry: ToolRegistry,
    tool_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[asyncio.Task[object], threading.Event, threading.Event, list[int]]:
    entered = threading.Event()
    release = threading.Event()
    fd_holder: list[int] = []
    original = shell_cwd.bind_shell_cwd

    def blocked(*args: object, **kwargs: object) -> tuple[str, int]:
        entered.set()
        assert release.wait(10)
        result = original(*args, **kwargs)
        fd_holder.append(result[1])
        return result

    monkeypatch.setattr(shell_cwd, "bind_shell_cwd", blocked)
    arguments = {"command": "true"}
    task = asyncio.create_task(registry.execute(ToolCall("cwd-test", tool_name, arguments)))
    assert await asyncio.to_thread(entered.wait, 10)
    return task, entered, release, fd_holder


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["bash", "run_background"])
async def test_shell_bind_does_not_block_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    release = threading.Event()
    try:
        task, _entered, release, _fd_holder = await _wait_for_bind(
            registry, tool_name, monkeypatch
        )
        ready = asyncio.Event()
        asyncio.get_running_loop().call_soon(ready.set)
        await asyncio.wait_for(ready.wait(), timeout=10)
        release.set()
        result = await asyncio.wait_for(task, timeout=10)
        assert result["isError"] is False
    finally:
        release.set()
        await registry.close()


@pytest.mark.asyncio
async def test_canceled_shell_bind_closes_fd_without_state_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    baseline = registry.bash_cwd
    release = threading.Event()
    try:
        task, _entered, release, fd_holder = await _wait_for_bind(
            registry, "bash", monkeypatch
        )
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert await asyncio.wait_for(asyncio.to_thread(lambda: bool(fd_holder)), 10)
        assert registry.bash_cwd == baseline
        with pytest.raises(OSError) as error:
            os.fstat(fd_holder[0])
        assert error.value.errno == 9
    finally:
        release.set()
        await registry.close()
