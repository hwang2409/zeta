"""Spawn tool shells without running code in a forked Python child."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import warnings
from asyncio import events, unix_events
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

_LAUNCHER = """
import os, sys
cwd_fd = int(sys.argv[1])
niceness = int(sys.argv[2])
try:
    current = os.getpriority(os.PRIO_PROCESS, 0)
    os.setpriority(os.PRIO_PROCESS, 0, min(19, current + niceness))
except (AttributeError, OSError):
    pass
os.fchdir(cwd_fd)
os.close(cwd_fd)
for inherited_fd in sys.argv[4:]:
    source, target = map(int, inherited_fd.split(':'))
    os.dup2(source, target)
    os.close(source)
os.execvpe('/bin/sh', ['sh', '-c', sys.argv[3]], os.environ)
"""


class _SpawnedProcess:
    def __init__(
        self,
        pid: int,
        *,
        stdin: IO[bytes] | None,
        stdout: IO[bytes],
        stderr: IO[bytes] | None,
    ) -> None:
        self.pid = pid
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        os.kill(self.pid, signal.SIGKILL)


class _ToolSubprocessTransport(unix_events._UnixSubprocessTransport):  # type: ignore[attr-defined]
    def _start(
        self,
        args: Sequence[str],
        shell: bool,
        stdin: int,
        stdout: int,
        stderr: int,
        bufsize: int,
        **kwargs: Any,
    ) -> None:
        del shell, bufsize
        self._proc = _posix_spawn(args, stdin, stdout, stderr, **kwargs)


async def create_tool_subprocess_shell(
    command: str,
    *,
    cwd: str | Path | None = None,
    cwd_fd: int | None = None,
    stdin: int = asyncio.subprocess.DEVNULL,
    stdout: int = asyncio.subprocess.PIPE,
    stderr: int = asyncio.subprocess.PIPE,
    pass_fds: Sequence[int] = (),
    env: Mapping[str, str] | None = None,
    niceness: int = 0,
) -> asyncio.subprocess.Process:
    """Start a session-owned shell after its stdio and inherited fds are installed."""

    if cwd_fd is None:
        directory = os.fspath(cwd if cwd is not None else os.getcwd())
        owned_cwd_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        cwd_fd = owned_cwd_fd
    else:
        owned_cwd_fd = None
    try:
        loop = asyncio.get_running_loop()
        protocol = asyncio.subprocess.SubprocessStreamProtocol(
            asyncio.streams._DEFAULT_LIMIT, loop
        )
        waiter = loop.create_future()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            watcher = events.get_child_watcher()
        with watcher:
            if not watcher.is_active():
                raise RuntimeError("asyncio child watcher is not active")
            transport = _ToolSubprocessTransport(
                loop,
                protocol,
                (sys.executable, "-c", _LAUNCHER, str(niceness), command),
                False,
                stdin,
                stdout,
                stderr,
                0,
                waiter=waiter,
                cwd_fd=cwd_fd,
                pass_fds=tuple(pass_fds),
                env=dict(env) if env is not None else os.environ.copy(),
            )
            watcher.add_child_handler(
                transport.get_pid(), loop._child_watcher_callback, transport
            )
            try:
                await waiter
            except BaseException:
                transport.close()
                await transport._wait()
                raise
        return asyncio.subprocess.Process(transport, protocol, loop)
    finally:
        if owned_cwd_fd is not None:
            os.close(owned_cwd_fd)


def _posix_spawn(
    args: Sequence[str],
    stdin: int,
    stdout: int,
    stderr: int,
    *,
    cwd_fd: int,
    pass_fds: Sequence[int],
    env: Mapping[str, str],
) -> _SpawnedProcess:
    if stdin not in (subprocess.PIPE, subprocess.DEVNULL):
        raise ValueError("tool subprocess stdin must be PIPE or DEVNULL")
    if stdout != subprocess.PIPE:
        raise ValueError("tool subprocess stdout must be PIPE")
    if stderr not in (subprocess.PIPE, subprocess.STDOUT):
        raise ValueError("tool subprocess stderr must be PIPE or STDOUT")

    opened: list[int] = []
    parent_stdin: IO[bytes] | None = None
    parent_stdout: IO[bytes] | None = None
    parent_stderr: IO[bytes] | None = None
    try:
        if stdin == subprocess.PIPE:
            child_stdin, parent_stdin_fd = os.pipe()
            opened.extend((child_stdin, parent_stdin_fd))
            parent_stdin = os.fdopen(parent_stdin_fd, "wb", buffering=0)
            opened.remove(parent_stdin_fd)
        else:
            child_stdin = os.open(os.devnull, os.O_RDONLY)
            opened.append(child_stdin)

        parent_stdout_fd, child_stdout = os.pipe()
        opened.extend((parent_stdout_fd, child_stdout))
        parent_stdout = os.fdopen(parent_stdout_fd, "rb", buffering=0)
        opened.remove(parent_stdout_fd)

        if stderr == subprocess.PIPE:
            parent_stderr_fd, child_stderr = os.pipe()
            opened.extend((parent_stderr_fd, child_stderr))
            parent_stderr = os.fdopen(parent_stderr_fd, "rb", buffering=0)
            opened.remove(parent_stderr_fd)
        else:
            child_stderr = child_stdout

        inherited = (cwd_fd, *pass_fds)
        unavailable = {0, 1, 2, *opened, *inherited}
        child_fds = tuple(
            fd
            for fd in range(3, 3 + len(unavailable) + len(inherited))
            if fd not in unavailable
        )[: len(inherited)]
        cwd_child_fd, *passed_child_fds = child_fds
        spawn_args = (
            *args[:3],
            str(cwd_child_fd),
            *args[3:],
            *(
                f"{source}:{target}"
                for source, target in zip(passed_child_fds, pass_fds, strict=True)
            ),
        )
        file_actions = [
            (os.POSIX_SPAWN_DUP2, child_stdin, 0),
            (os.POSIX_SPAWN_DUP2, child_stdout, 1),
            (os.POSIX_SPAWN_DUP2, child_stderr, 2),
            *(
                (os.POSIX_SPAWN_DUP2, source, target)
                for source, target in zip(inherited, child_fds, strict=True)
            ),
        ]
        for fd in set(opened) - {child_stdin, child_stdout, child_stderr}:
            file_actions.append((os.POSIX_SPAWN_CLOSE, fd))
        pid = os.posix_spawn(
            spawn_args[0],
            spawn_args,
            env,
            file_actions=file_actions,
            setsid=True,
            setsigdef=tuple(
                signal_number
                for signal_number in (
                    signal.SIGPIPE,
                    getattr(signal, "SIGXFZ", None),
                    getattr(signal, "SIGXFSZ", None),
                )
                if signal_number is not None
            ),
        )
        return _SpawnedProcess(
            pid, stdin=parent_stdin, stdout=parent_stdout, stderr=parent_stderr
        )
    except BaseException:
        for stream in (parent_stdin, parent_stdout, parent_stderr):
            if stream is not None:
                stream.close()
        raise
    finally:
        for fd in opened:
            try:
                os.close(fd)
            except OSError:
                pass
