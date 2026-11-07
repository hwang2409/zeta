from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.support.processes import assert_pid_gone
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
        await create_tool_subprocess_shell("sleep 30", stdin=asyncio.subprocess.DEVNULL)

    assert child_pid is not None
    await assert_pid_gone(child_pid)
    with pytest.raises(ChildProcessError):
        await asyncio.to_thread(os.waitpid, child_pid, os.WNOHANG)
    assert len(os.listdir("/dev/fd")) == baseline_fds


@pytest.mark.parametrize(
    ("spawn_path", "scenario"),
    [
        (spawn_path, scenario)
        for spawn_path in ("native_setsid", "launcher_setsid")
        for scenario in ("success", "launcher_failure", "readiness_cancellation")
    ],
)
@pytest.mark.asyncio
async def test_tool_spawn_paths_preserve_process_contract(
    monkeypatch: pytest.MonkeyPatch, spawn_path: str, scenario: str
) -> None:
    await asyncio.to_thread(lambda: None)
    baseline_fds = len(os.listdir("/dev/fd"))
    spawned_pids: list[int] = []
    spawn_attempts: list[bool] = []
    real_posix_spawn = os.posix_spawn

    def controlled_posix_spawn(
        path: str,
        argv: list[str] | tuple[str, ...],
        env: dict[str, str] | Mapping[str, str],
        **kwargs: object,
    ) -> int:
        uses_setsid = kwargs.get("setsid") is True
        spawn_attempts.append(uses_setsid)
        if spawn_path == "launcher_setsid" and uses_setsid:
            raise NotImplementedError(
                "posix_spawn: setsid unavailable on this platform"
            )
        try:
            pid = real_posix_spawn(path, argv, env, **kwargs)  # type: ignore[arg-type]
        except NotImplementedError:
            assert spawn_path == "native_setsid" and uses_setsid
            kwargs.pop("setsid")
            pid = real_posix_spawn(path, argv, env, **kwargs)  # type: ignore[arg-type]
        spawned_pids.append(pid)
        return pid

    # test-contract: seam reset the process-wide capability decision between rows
    monkeypatch.setattr(
        tool_spawn, "_POSIX_SPAWN_SETSID_SUPPORTED", None, raising=False
    )
    monkeypatch.setattr(os, "posix_spawn", controlled_posix_spawn)

    if scenario == "success":
        inherited_read, inherited_write = os.pipe()
        try:
            probe_command = shlex.join(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import fcntl, json, os, signal
                        open_fds = []
                        for fd in range(3, 256):
                            try:
                                fcntl.fcntl(fd, fcntl.F_GETFD)
                            except OSError:
                                continue
                            open_fds.append(fd)
                        print(json.dumps({
                            "is_session_leader": os.getsid(0) == os.getpid(),
                            "open_fds": open_fds,
                            "sigpipe_default": int(os.environ["ZETA_TEST_SIGPIPE_STATUS"]) == 128 + signal.SIGPIPE,
                            "sigxfsz_default": not hasattr(signal, "SIGXFSZ") or int(os.environ["ZETA_TEST_SIGXFSZ_STATUS"]) == 128 + signal.SIGXFSZ,
                        }))
                        """
                    ),
                ]
            )
            command = (
                "sh -c 'kill -s PIPE $$'; ZETA_TEST_SIGPIPE_STATUS=$?; "
                "sh -c 'kill -s XFSZ $$'; ZETA_TEST_SIGXFSZ_STATUS=$?; "
                "export ZETA_TEST_SIGPIPE_STATUS ZETA_TEST_SIGXFSZ_STATUS; "
                f"exec {probe_command}"
            )
            process: asyncio.subprocess.Process | None = None
            spawn_error: BaseException | None = None
            try:
                process = await asyncio.wait_for(
                    create_tool_subprocess_shell(command, pass_fds=(inherited_read,)),
                    10,
                )
            except (NotImplementedError, RuntimeError, TimeoutError) as exc:
                spawn_error = exc
            assert spawn_error is None, "spawn must accept the session-leader request"
            assert process is not None
            stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0, stderr.decode()
            assert json.loads(stdout) == {
                "is_session_leader": True,
                "open_fds": [inherited_read],
                "sigpipe_default": True,
                "sigxfsz_default": True,
            }
        finally:
            os.close(inherited_read)
            os.close(inherited_write)
    elif scenario == "launcher_failure":
        # test-contract: seam force failure before the private readiness byte
        monkeypatch.setattr(tool_spawn, "_LAUNCHER", "raise SystemExit(1)")
        spawn_error: BaseException | None = None
        try:
            await asyncio.wait_for(
                create_tool_subprocess_shell("printf unreachable"), 10
            )
        except (NotImplementedError, RuntimeError, TimeoutError) as exc:
            spawn_error = exc
        assert isinstance(spawn_error, RuntimeError), (
            "launcher failure must be reported after the child starts"
        )
        assert str(spawn_error) == "tool subprocess launcher failed to start"
    else:
        marker_read, marker_write = os.pipe()
        operation: asyncio.Task[asyncio.subprocess.Process] | None = None
        marker: asyncio.Task[bytes] | None = None
        # test-contract: seam hold the private launcher at its readiness boundary
        monkeypatch.setattr(
            tool_spawn,
            "_LAUNCHER",
            """
import os, signal, sys
source, _target = map(int, sys.argv[5].split(':'))
os.write(source, b'1')
signal.pause()
""",
        )
        try:
            operation = asyncio.create_task(
                create_tool_subprocess_shell(
                    "printf unreachable", pass_fds=(marker_write,)
                )
            )
            marker = asyncio.create_task(asyncio.to_thread(os.read, marker_read, 1))
            done, _pending = await asyncio.wait(
                {operation, marker}, timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            assert marker in done, "spawn must reach launcher readiness"
            assert marker.result() == b"1"
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operation, 10)
        finally:
            if operation is not None and not operation.done():
                operation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(operation, 10)
            if marker is not None and not marker.done():
                os.close(marker_write)
                await asyncio.wait_for(marker, 10)
            os.close(marker_read)
            try:
                os.close(marker_write)
            except OSError:
                pass

    assert spawned_pids
    child_pid = spawned_pids[-1]
    if scenario != "success":
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        with pytest.raises(ChildProcessError):
            await asyncio.wait_for(
                asyncio.to_thread(os.waitpid, child_pid, os.WNOHANG), 10
            )
    assert spawn_attempts == (
        [True] if spawn_path == "native_setsid" else [True, False]
    )
    assert len(os.listdir("/dev/fd")) == baseline_fds


@pytest.mark.asyncio
async def test_launcher_exit_before_failure_preserves_spawn_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadyPipe:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Process:
        ready = ReadyPipe()

    class Transport:
        instance: Transport

        def __init__(
            self,
            *args: object,
            waiter: asyncio.Future[None],
            **kwargs: object,
        ) -> None:
            del args, kwargs
            self._proc: Process | None = Process()
            self.closed = False
            self.waited = False
            Transport.instance = self
            waiter.set_result(None)

        def get_pid(self) -> int:
            return 1

        def _process_exited(self, returncode: int) -> None:
            assert returncode == 1
            self._proc = None

        def close(self) -> None:
            self.closed = True

        async def _wait(self) -> int:
            self.waited = True
            return 1

    async def fail_after_process_exit(*, process: Process) -> None:
        assert Transport.instance._proc is process
        Transport.instance._process_exited(1)
        raise RuntimeError("tool subprocess launcher failed to start")

    loop = asyncio.get_running_loop()
    watcher = getattr(loop, "_watcher", None)
    if watcher is None:
        with tool_spawn.warnings.catch_warnings():
            tool_spawn.warnings.simplefilter("ignore", DeprecationWarning)
            watcher = tool_spawn.events.get_child_watcher()
    monkeypatch.setattr(watcher, "add_child_handler", lambda *args: None)
    # test-contract: seam no public hook can force the transport's process-exit race
    monkeypatch.setattr(tool_spawn, "_ToolSubprocessTransport", Transport)
    # test-contract: seam no public hook orders launcher failure after process exit
    monkeypatch.setattr(tool_spawn, "_wait_for_launcher", fail_after_process_exit)

    with pytest.raises(RuntimeError, match="tool subprocess launcher failed to start"):
        await create_tool_subprocess_shell("printf unreachable")

    assert Process.ready.closed
    assert Transport.instance.closed
    assert Transport.instance.waited
