from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from zeta.tools._shared import tool_spawn
from zeta.tools._shared.tool_spawn import create_tool_subprocess_shell

_SENTINEL = "ATFORK_ESCAPE"


@pytest.mark.parametrize("spawn_path", ["bash", "run_background", "approved_cwd"])
def test_shell_spawn_does_not_leak_atfork_stderr_to_parent_fd2(
    tmp_path: Path, spawn_path: str
) -> None:
    helper = textwrap.dedent(
        f"""
        import asyncio
        import json
        import os
        from pathlib import Path

        from zeta.protocol.types import ToolCall
        from zeta.skills import SkillCatalog
        from zeta.tools import ToolRegistry
        from zeta.tools._shared.background_spawn import create_subprocess_shell_in_fd

        os.register_at_fork(
            after_in_child=lambda: os.write(2, b"{_SENTINEL}\\n")
        )

        async def main():
            root = Path({str(tmp_path)!r})
            if {spawn_path!r} == "approved_cwd":
                cwd_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    process = await create_subprocess_shell_in_fd(
                        "printf child-output",
                        cwd_fd,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=os.environ.copy(),
                        niceness=10,
                    )
                    stdout, stderr = await process.communicate()
                    output = stdout.decode()
                    child_stderr = stderr.decode()
                finally:
                    os.close(cwd_fd)
            else:
                registry = ToolRegistry(
                    root, skill_catalog=SkillCatalog.empty()
                )
                try:
                    result = await registry.execute(
                        ToolCall("spawn", {spawn_path!r}, {{"command": "printf child-output"}})
                    )
                    if {spawn_path!r} == "bash":
                        output = result["structuredContent"]["stdout"]
                    else:
                        task_id = result["structuredContent"]["task_id"]
                        await registry.background_tasks.wait(task_id)
                        result = await registry.execute(
                            ToolCall("output", "task_output", {{"task_id": task_id}})
                        )
                        output = result["structuredContent"]["output"]
                    child_stderr = ""
                finally:
                    await registry.close()
            print(json.dumps({{"output": output, "stderr": child_stderr}}))

        asyncio.run(main())
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", helper],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert result == {"output": "child-output", "stderr": ""}


@pytest.mark.asyncio
async def test_watcher_registration_failure_leaves_no_process_zombie_or_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    watcher = loop._watcher
    child_pid: int | None = None
    baseline_fds = len(os.listdir("/dev/fd"))

    def fail_registration(pid: int, callback: object, transport: object) -> None:
        nonlocal child_pid
        child_pid = pid
        raise RuntimeError("registration failed")

    monkeypatch.setattr(watcher, "add_child_handler", fail_registration)
    with pytest.raises(RuntimeError, match="registration failed"):
        await create_tool_subprocess_shell(
            "sleep 30", stdin=asyncio.subprocess.DEVNULL
        )

    assert child_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    with pytest.raises(ChildProcessError):
        await asyncio.to_thread(os.waitpid, child_pid, os.WNOHANG)
    assert len(os.listdir("/dev/fd")) == baseline_fds


async def _assert_launcher_failure_is_reaped(
    monkeypatch: pytest.MonkeyPatch, launcher: str
) -> None:
    child_pid: int | None = None
    await asyncio.to_thread(lambda: None)
    baseline_fds = len(os.listdir("/dev/fd"))
    original_spawn = tool_spawn._posix_spawn

    def capture_spawn(*args: object, **kwargs: object) -> tool_spawn._SpawnedProcess:
        nonlocal child_pid
        process = original_spawn(*args, **kwargs)  # type: ignore[arg-type]
        child_pid = process.pid
        return process

    monkeypatch.setattr(tool_spawn, "_LAUNCHER", launcher)
    monkeypatch.setattr(tool_spawn, "_posix_spawn", capture_spawn)
    with pytest.raises(RuntimeError, match="tool subprocess launcher failed to start"):
        await create_tool_subprocess_shell("printf unreachable")

    assert child_pid is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(child_pid, 0)
    with pytest.raises(ChildProcessError):
        await asyncio.to_thread(os.waitpid, child_pid, os.WNOHANG)
    assert len(os.listdir("/dev/fd")) == baseline_fds


@pytest.mark.asyncio
async def test_launcher_eof_before_ready_is_a_spawn_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_launcher_failure_is_reaped(monkeypatch, "raise SystemExit(1)")


@pytest.mark.asyncio
async def test_launcher_wrong_ready_byte_is_a_spawn_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_launcher_failure_is_reaped(
        monkeypatch,
        "import os, sys; os.write(int(sys.argv[2]), b'0')",
    )


@pytest.mark.asyncio
async def test_launcher_hang_before_ready_is_bounded_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool_spawn, "_LAUNCHER_START_TIMEOUT_SECONDS", 0.1)
    started = asyncio.get_running_loop().time()

    await _assert_launcher_failure_is_reaped(
        monkeypatch,
        "import signal; signal.pause()",
    )

    assert asyncio.get_running_loop().time() - started < 2
