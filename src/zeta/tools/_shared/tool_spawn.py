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
ready_fd = int(sys.argv[2])
niceness = int(sys.argv[3])
try:
    current = os.getpriority(os.PRIO_PROCESS, 0)
    os.setpriority(os.PRIO_PROCESS, 0, min(19, current + niceness))
except (AttributeError, OSError):
    pass
os.fchdir(cwd_fd)
os.close(cwd_fd)
for inherited_fd in sys.argv[5:]:
    source, target = map(int, inherited_fd.split(':'))
    os.dup2(source, target)
    os.close(source)
os.write(ready_fd, b'1')
os.close(ready_fd)
os.execvpe('/bin/sh', ['sh', '-c', sys.argv[4]], os.environ)
"""


class _SpawnedProcess:
    def __init__(
        self,
        pid: int,
        *,
        stdin: IO[bytes] | None,
        stdout: IO[bytes],
        stderr: IO[bytes] | None,
        ready: IO[bytes],
    ) -> None:
        self.pid = pid
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.ready = ready
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        os.kill(self.pid, signal.SIGKILL)


async def _cleanup_unwatched_transport(
    transport: _ToolSubprocessTransport, waiter: asyncio.Future[None]
) -> None:
    process = transport._proc
    await asyncio.sleep(0)
    transport.close()
    await asyncio.sleep(0)
    if waiter.done():
        waiter.exception()
    for stream in (process.stdin, process.stdout, process.stderr, process.ready):
        if stream is not None:
            stream.close()
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    reap = asyncio.create_task(asyncio.to_thread(os.waitpid, process.pid, 0))
    try:
        await asyncio.shield(reap)
    except BaseException:
        await asyncio.shield(reap)
        raise


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
        watcher = getattr(loop, "_watcher", None)
        if watcher is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                watcher = events.get_child_watcher()
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
        try:
            watcher.add_child_handler(
                transport.get_pid(), loop._child_watcher_callback, transport
            )
        except BaseException:
            cleanup = asyncio.create_task(
                _cleanup_unwatched_transport(transport, waiter)
            )
            try:
                await asyncio.shield(cleanup)
            except BaseException:
                await asyncio.shield(cleanup)
                raise
            raise
        try:
            await waiter
            await _wait_for_launcher(process=transport._proc)
        except BaseException:
            transport._proc.ready.close()
            transport.close()
            await transport._wait()
            raise
        return asyncio.subprocess.Process(transport, protocol, loop)
    finally:
        if owned_cwd_fd is not None:
            os.close(owned_cwd_fd)


async def _wait_for_launcher(*, process: _SpawnedProcess) -> None:
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    fd = process.ready.fileno()

    def consume_ready_byte() -> None:
        if ready.done():
            return
        loop.remove_reader(fd)
        os.read(fd, 1)
        ready.set_result(None)

    loop.add_reader(fd, consume_ready_byte)
    try:
        await ready
    finally:
        loop.remove_reader(fd)
        process.ready.close()


class _PipeOwner:
    def __init__(self, stdin: int, stderr: int) -> None:
        self.opened: set[int] = set()
        self.parent_stdin: IO[bytes] | None = None
        self.parent_stdout: IO[bytes]
        self.parent_stderr: IO[bytes] | None = None
        parent_ready, self.child_ready = os.pipe()
        self.opened.update((parent_ready, self.child_ready))
        self.parent_ready = os.fdopen(parent_ready, "rb", buffering=0)
        self.opened.remove(parent_ready)
        if stdin == subprocess.PIPE:
            child_stdin, parent_stdin = os.pipe()
            self.opened.update((child_stdin, parent_stdin))
            self.parent_stdin = os.fdopen(parent_stdin, "wb", buffering=0)
            self.opened.remove(parent_stdin)
        else:
            child_stdin = os.open(os.devnull, os.O_RDONLY)
            self.opened.add(child_stdin)
        self.child_stdin = child_stdin

        parent_stdout, self.child_stdout = os.pipe()
        self.opened.update((parent_stdout, self.child_stdout))
        self.parent_stdout = os.fdopen(parent_stdout, "rb", buffering=0)
        self.opened.remove(parent_stdout)
        if stderr == subprocess.PIPE:
            parent_stderr, self.child_stderr = os.pipe()
            self.opened.update((parent_stderr, self.child_stderr))
            self.parent_stderr = os.fdopen(parent_stderr, "rb", buffering=0)
            self.opened.remove(parent_stderr)
        else:
            self.child_stderr = self.child_stdout

    def transfer(self, pid: int) -> _SpawnedProcess:
        process = _SpawnedProcess(
            pid,
            stdin=self.parent_stdin,
            stdout=self.parent_stdout,
            stderr=self.parent_stderr,
            ready=self.parent_ready,
        )
        self.parent_stdin = self.parent_stdout = self.parent_stderr = None  # type: ignore[assignment]
        self.parent_ready = None  # type: ignore[assignment]
        self.close()
        return process

    def close(self) -> None:
        for stream in (
            self.parent_stdin,
            self.parent_stdout,
            self.parent_stderr,
            self.parent_ready,
        ):
            if stream is not None:
                stream.close()
        for fd in self.opened:
            try:
                os.close(fd)
            except OSError:
                pass
        self.opened.clear()


def _file_actions(
    owner: _PipeOwner, inherited: tuple[int, ...], child_fds: tuple[int, ...]
) -> list[tuple[int, int, int]]:
    actions = [
        (os.POSIX_SPAWN_DUP2, owner.child_stdin, 0),
        (os.POSIX_SPAWN_DUP2, owner.child_stdout, 1),
        (os.POSIX_SPAWN_DUP2, owner.child_stderr, 2),
        *(
            (os.POSIX_SPAWN_DUP2, source, target)
            for source, target in zip(inherited, child_fds, strict=True)
        ),
    ]
    actions.extend(
        (os.POSIX_SPAWN_CLOSE, fd)
        for fd in owner.opened - {owner.child_stdin, owner.child_stdout, owner.child_stderr}
    )
    return actions


def _posix_spawn(
    args: Sequence[str], stdin: int, stdout: int, stderr: int, *, cwd_fd: int,
    pass_fds: Sequence[int], env: Mapping[str, str],
) -> _SpawnedProcess:
    if stdin not in (subprocess.PIPE, subprocess.DEVNULL):
        raise ValueError("tool subprocess stdin must be PIPE or DEVNULL")
    if stdout != subprocess.PIPE:
        raise ValueError("tool subprocess stdout must be PIPE")
    if stderr not in (subprocess.PIPE, subprocess.STDOUT):
        raise ValueError("tool subprocess stderr must be PIPE or STDOUT")

    owner: _PipeOwner | None = None
    try:
        owner = _PipeOwner(stdin, stderr)
        inherited = (cwd_fd, owner.child_ready, *pass_fds)
        unavailable = {0, 1, 2, *owner.opened, *inherited}
        child_fds = tuple(
            fd for fd in range(3, 3 + len(unavailable) + len(inherited))
            if fd not in unavailable
        )[: len(inherited)]
        spawn_args = (
            *args[:3],
            str(child_fds[0]),
            str(child_fds[1]),
            *args[3:],
            *(
                f"{source}:{target}"
                for source, target in zip(
                    child_fds[2:], pass_fds, strict=True
                )
            ),
        )
        pid = os.posix_spawn(
            spawn_args[0], spawn_args, env,
            file_actions=_file_actions(owner, inherited, child_fds), setsid=True,
            setsigdef=tuple(signum for signum in (signal.SIGPIPE, getattr(signal, "SIGXFZ", None), getattr(signal, "SIGXFSZ", None)) if signum is not None),
        )
        return owner.transfer(pid)
    except BaseException:
        if owner is not None:
            owner.close()
        raise
