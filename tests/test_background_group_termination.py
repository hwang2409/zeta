from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading

import pytest

from tests.support.processes import wait_pid_exited
from zeta.tools._shared import background_spawn


@pytest.mark.asyncio
async def test_termination_waits_for_term_ignoring_group_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_read, ready_write = os.pipe()
    os.set_inheritable(ready_write, True)
    child_code = (
        "import os,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "os.write(int(sys.argv[1]), str(os.getpid()).encode()); "
        "os.close(int(sys.argv[1])); time.sleep(30)"
    )
    leader_code = (
        "import os,subprocess,sys; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}, "
        "sys.argv[1]], pass_fds=(int(sys.argv[1]),)); "
        "os.close(int(sys.argv[1])); child.wait()"
    )
    process: asyncio.subprocess.Process | None = None
    child_pid: int | None = None
    terminating: asyncio.Task[None] | None = None
    kill_delivery: asyncio.Task[None] | None = None
    kill_held = threading.Event()
    release_kill = asyncio.Event()
    held_groups: list[tuple[asyncio.subprocess.Process, int]] = []
    held_descendants: list[tuple[dict[int, str], int]] = []
    original_signal_group = background_spawn._signal_group
    original_signal_pids = background_spawn._signal_pids

    def hold_group_kill(
        target: asyncio.subprocess.Process, signal_number: int
    ) -> None:
        if signal_number == signal.SIGKILL:
            held_groups.append((target, signal_number))
            return
        original_signal_group(target, signal_number)

    def hold_descendant_kill(pids: dict[int, str], signal_number: int) -> None:
        if signal_number == signal.SIGKILL:
            held_descendants.append((pids, signal_number))
            if pids:
                kill_held.set()
            return
        original_signal_pids(pids, signal_number)

    async def deliver_held_kills() -> None:
        await asyncio.wait_for(release_kill.wait(), 10)
        for target, signal_number in held_groups:
            original_signal_group(target, signal_number)
        for pids, signal_number in held_descendants:
            original_signal_pids(pids, signal_number)

    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                leader_code,
                str(ready_write),
                pass_fds=(ready_write,),
                start_new_session=True,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            ),
            timeout=10,
        )
        os.close(ready_write)
        ready_write = -1
        ready = await asyncio.wait_for(asyncio.to_thread(os.read, ready_read, 32), 10)
        child_pid = int(ready)
        # test-contract: seam hold real group kill delivery at the termination seam
        monkeypatch.setattr(background_spawn, "_signal_group", hold_group_kill)
        # test-contract: seam hold real descendant kill delivery at the termination seam
        monkeypatch.setattr(background_spawn, "_signal_pids", hold_descendant_kill)

        kill_delivery = asyncio.create_task(deliver_held_kills())
        terminating = asyncio.create_task(
            background_spawn._terminate_process(process, term_grace=0)
        )
        assert await asyncio.wait_for(asyncio.to_thread(kill_held.wait, 10), 10)
        assert held_descendants[0][0]
        assert not terminating.done()

        release_kill.set()
        await asyncio.wait_for(kill_delivery, 10)
        await asyncio.wait_for(terminating, 10)
        assert not background_spawn._group_exists(process.pid)
    finally:
        release_kill.set()
        if ready_read >= 0:
            os.close(ready_read)
        if ready_write >= 0:
            os.close(ready_write)
        if process is not None and background_spawn._group_exists(process.pid):
            original_signal_group(process, signal.SIGKILL)
        if child_pid is not None:
            await asyncio.wait_for(wait_pid_exited(child_pid), 10)
        if process is not None:
            await asyncio.wait_for(process.wait(), 10)
        if terminating is not None and not terminating.done():
            terminating.cancel()
        pending = [
            task
            for task in (terminating, kill_delivery)
            if task is not None
        ]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 10)
