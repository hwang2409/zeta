from __future__ import annotations

import asyncio
import fcntl
import functools
import multiprocessing
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from zeta.core.session import SessionManager


def _epoch(store: Any) -> str | None:
    return getattr(getattr(store, "execution_lease", None), "epoch", None)


def _hold_session(home: str, session_id: str, ready: Any, release: Any) -> None:
    manager = SessionManager(home)
    with manager.open(session_id).store as store:
        ready.send(("owned", _epoch(store)))
        release.recv()


def _try_open(home: str, session_id: str, result: Any) -> None:
    manager = SessionManager(home)
    try:
        with manager.open(session_id).store as store:
            result.send(("opened", _epoch(store)))
    except Exception as exc:  # noqa: BLE001 - report across the process boundary
        result.send(("error", type(exc).__name__, str(exc)))


def _spawn(target: Any, *args: Any) -> multiprocessing.Process:
    process = multiprocessing.get_context("spawn").Process(target=target, args=args)
    process.start()
    return process


def _created_session(home: Path, cwd: Path) -> str:
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="offline", cwd=cwd)
    session_id = opened.metadata.session_id
    opened.store.close()
    manager._release_storage_lease()
    return session_id


def _create_session_process(home: str, cwd: str, result: Any) -> None:
    result.send(_created_session(Path(home), Path(cwd)))


def _production_spawn_owner(
    home: str,
    session_id: str,
    cwd: str,
    ready_fifo: str,
    release_fifo: str,
    done_fifo: str,
    ready: Any,
    release: Any,
) -> None:
    from zeta.tools._shared.tool_spawn import create_tool_subprocess_shell

    script = (
        "import os; "
        f"p={ready_fifo!r}; r={release_fifo!r}; d={done_fifo!r}; "
        "open(p, 'w').write(str(os.getpid())); "
        "open(r, 'rb').read(1); "
        "open(d, 'w').write('done')"
    )
    command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    async def start() -> int:
        process = await create_tool_subprocess_shell(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return process.pid

    with SessionManager(home).open(session_id).store as store:
        child_pid = asyncio.run(start())
        ready.send((child_pid, _epoch(store)))
        release.recv()


def _open_with_format_lock_probe(home: str, session_id: str, result: Any) -> None:
    from zeta import storage_format

    real_flock = storage_format.fcntl.flock
    reported = False

    def probe(fd: int, operation: int) -> None:
        nonlocal reported
        if operation & fcntl.LOCK_SH and not reported:
            reported = True
            result.send("attempting")
        real_flock(fd, operation)

    storage_format.fcntl.flock = probe
    try:
        with SessionManager(home).open(session_id).store as store:
            result.send(("opened", _epoch(store)))
    except Exception as exc:  # noqa: BLE001 - report across the process boundary
        result.send(("error", type(exc).__name__, str(exc)))


def test_second_process_cannot_open_owned_session(tmp_path: Path) -> None:
    """The base fails here because both processes return writable stores."""

    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    owner_ready, owner_send = multiprocessing.Pipe(duplex=False)
    release_recv, release_send = multiprocessing.Pipe(duplex=False)
    owner = _spawn(_hold_session, str(home), session_id, owner_send, release_recv)
    assert owner_ready.recv()[0] == "owned"

    result_recv, result_send = multiprocessing.Pipe(duplex=False)
    contender = _spawn(_try_open, str(home), session_id, result_send)
    result = result_recv.recv()
    contender.join()
    release_send.send(None)
    owner.join()

    assert result[0:2] == ("error", "SessionOwnedError")
    assert owner.exitcode == contender.exitcode == 0


def test_owner_death_allows_immediate_new_epoch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    ready, ready_send = multiprocessing.Pipe(duplex=False)
    release_recv, _release_send = multiprocessing.Pipe(duplex=False)
    owner = _spawn(_hold_session, str(home), session_id, ready_send, release_recv)
    first_epoch = ready.recv()[1]

    os.kill(owner.pid, signal.SIGKILL)
    owner.join()
    assert owner.exitcode == -signal.SIGKILL

    result_recv, result_send = multiprocessing.Pipe(duplex=False)
    successor = _spawn(_try_open, str(home), session_id, result_send)
    result = result_recv.recv()
    successor.join()
    assert result[0] == "opened"
    assert first_epoch is not None and result[1] != first_epoch
    assert successor.exitcode == 0


def test_suspend_does_not_release_owner(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    ready, ready_send = multiprocessing.Pipe(duplex=False)
    release_recv, release_send = multiprocessing.Pipe(duplex=False)
    owner = _spawn(_hold_session, str(home), session_id, ready_send, release_recv)
    ready.recv()

    os.kill(owner.pid, signal.SIGSTOP)
    waited_pid, status = os.waitpid(owner.pid, os.WUNTRACED)
    assert waited_pid == owner.pid and os.WIFSTOPPED(status)
    result_recv, result_send = multiprocessing.Pipe(duplex=False)
    contender = _spawn(_try_open, str(home), session_id, result_send)
    result = result_recv.recv()
    contender.join()

    os.kill(owner.pid, signal.SIGCONT)
    release_send.send(None)
    owner.join()
    assert result[0:2] == ("error", "SessionOwnedError")
    assert owner.exitcode == contender.exitcode == 0


def test_read_only_store_is_lease_free(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    manager = SessionManager(home)
    with (
        manager.open(session_id).store as owner,
        manager.open(session_id, _read_only=True).store as observer,
    ):
        assert observer.messages() == owner.messages()
        assert observer._runtime_lease is None
        with pytest.raises(Exception, match="read-only"):
            observer.append_pending_prompt("no")
        with pytest.raises(Exception) as excinfo:
            manager.open(session_id)
        assert type(excinfo.value).__name__ == "SessionOwnedError"


def test_closed_capability_is_stale(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    store = SessionManager(home).open(session_id).store
    capability = getattr(store, "execution_lease", None)
    store.close()

    assert capability is not None
    with pytest.raises(Exception) as excinfo:
        capability.assert_owned()
    assert type(excinfo.value).__name__ == "StaleSessionOwnerError"

    with SessionManager(home).open(session_id).store as successor:
        assert _epoch(successor) != capability.epoch


def test_stale_store_and_metadata_capabilities_cannot_publish(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    manager = SessionManager(home)
    opened = manager.open(session_id)
    stale = getattr(opened.store, "execution_lease", None)
    assert stale is not None
    stale.close()

    with pytest.raises(Exception) as store_error:
        opened.store.append_pending_prompt("blocked")
    assert type(store_error.value).__name__ == "StaleSessionOwnerError"
    with pytest.raises(Exception) as metadata_error:
        manager._mutate(
            session_id,
            lambda metadata: metadata,
            execution_lease=stale,
        )
    assert type(metadata_error.value).__name__ == "StaleSessionOwnerError"
    opened.store.close()

    with manager.open(session_id).store as successor:
        assert successor.pending_prompts() == []


def test_execution_descriptor_is_not_inheritable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    with SessionManager(home).open(session_id).store as store:
        lease = getattr(store, "execution_lease", None)
        assert lease is not None
        assert os.get_inheritable(lease._owned.fd) is False


def test_detached_writer_retains_the_resident_epoch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    with SessionManager(home).open(session_id).store as resident:
        detached = resident._new_persistent_durable_writer()
        try:
            assert _epoch(resident) is not None
            assert _epoch(detached) == _epoch(resident)
        finally:
            detached.close()


def test_tui_real_render_refuses_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zeta.cli.main import _create_tui_app, build_parser

    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    session_id = _created_session(home, tmp_path)
    parser = build_parser()
    args = parser.parse_args(["--provider", "codex", "--resume", session_id])

    with SessionManager(home).open(session_id).store:
        try:
            app = _create_tui_app(args, parser)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            app.loop.store.close()
            pytest.fail("the TUI rendered a second writable session")

    rendered = capsys.readouterr()
    assert "active in another zeta process" in rendered.err


def test_headless_refuses_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from zeta.cli.main import build_parser
    from zeta.runtime.headless import run_headless

    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    session_id = _created_session(home, tmp_path)
    args = build_parser().parse_args(
        ["--provider", "codex", "--resume", session_id, "-p", "blocked"]
    )

    with SessionManager(home).open(session_id).store:
        code = run_headless(args, args.prompt)

    assert code == 1
    assert "active in another zeta process" in capsys.readouterr().err


def test_production_spawn_does_not_inherit_execution_owner(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    ready_fifo = tmp_path / "ready.fifo"
    release_fifo = tmp_path / "release.fifo"
    done_fifo = tmp_path / "done.fifo"
    for fifo in (ready_fifo, release_fifo, done_fifo):
        os.mkfifo(fifo)

    owner_ready, owner_send = multiprocessing.Pipe(duplex=False)
    owner_release_recv, _owner_release_send = multiprocessing.Pipe(duplex=False)
    owner = _spawn(
        _production_spawn_owner,
        str(home),
        session_id,
        str(tmp_path),
        str(ready_fifo),
        str(release_fifo),
        str(done_fifo),
        owner_send,
        owner_release_recv,
    )
    spawned_pid, first_epoch = owner_ready.recv()
    with ready_fifo.open() as stream:
        child_pid = int(stream.read())
    assert child_pid == spawned_pid

    contender_recv, contender_send = multiprocessing.Pipe(duplex=False)
    contender = _spawn(_try_open, str(home), session_id, contender_send)
    refusal = contender_recv.recv()
    contender.join()

    os.kill(owner.pid, signal.SIGKILL)
    owner.join()
    assert owner.exitcode == -signal.SIGKILL
    os.kill(child_pid, 0)

    with SessionManager(home).open(session_id).store as successor:
        successor_epoch = _epoch(successor)
    with release_fifo.open("wb", buffering=0) as release:
        release.write(b"1")
    with done_fifo.open() as done:
        assert done.read() == "done"
    assert first_epoch is not None and successor_epoch != first_epoch
    assert refusal[0:2] == ("error", "SessionOwnedError")


def test_takeover_waits_for_detached_write_to_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.core.store import ConversationStore
    from zeta.protocol.types import Message, MessageRole, TextContent

    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    store = SessionManager(home).open(session_id).store
    write_started = threading.Event()
    release_write = threading.Event()
    real_append = ConversationStore.append_message

    @functools.wraps(real_append)
    def blocked_append(self: Any, *args: Any, **kwargs: Any) -> Any:
        write_started.set()
        release_write.wait()
        return real_append(self, *args, **kwargs)

    monkeypatch.setattr(ConversationStore, "append_message", blocked_append)

    async def exercise() -> None:
        write = asyncio.create_task(
            store.append_message_async(
                Message(MessageRole.ASSISTANT, [TextContent("persisted")])
            )
        )
        await asyncio.to_thread(write_started.wait)
        loop = asyncio.get_running_loop()
        close_waiting = loop.create_future()
        condition = store._durable_write_condition
        real_wait = condition.wait

        def report_close_wait(*args: Any, **kwargs: Any) -> Any:
            if not close_waiting.done():
                loop.call_soon_threadsafe(close_waiting.set_result, None)
            return real_wait(*args, **kwargs)

        condition.wait = report_close_wait
        close = asyncio.create_task(asyncio.to_thread(store.close))
        await close_waiting
        assert store._closing

        try:
            result_recv, result_send = multiprocessing.Pipe(duplex=False)
            contender = _spawn(_try_open, str(home), session_id, result_send)
            result = result_recv.recv()
            contender.join()
            assert result[0:2] == ("error", "SessionOwnedError")
            assert contender.exitcode == 0
        finally:
            release_write.set()
            await write
            await close

    asyncio.run(exercise())
    with SessionManager(home).open(session_id).store as successor:
        assert _epoch(successor) is not None


def test_format_lock_and_execution_lock_order_across_processes(tmp_path: Path) -> None:
    from zeta.storage_format import StorageFormatError, advance, verify

    home = tmp_path / "home"
    home.mkdir()
    (home / "storage-format.json").write_text('{"writer_epoch":1}\n')
    session_id = _created_session(home, tmp_path)
    owner_ready, owner_send = multiprocessing.Pipe(duplex=False)
    owner_release_recv, owner_release_send = multiprocessing.Pipe(duplex=False)
    owner = _spawn(_hold_session, str(home), session_id, owner_send, owner_release_recv)
    assert owner_ready.recv()[0] == "owned"
    try:
        advance(home, expected_epoch=1, target_epoch=2)
    except StorageFormatError as exc:
        advance_error = str(exc)
    else:
        advance_error = ""
    owner_release_send.send(None)
    owner.join()
    assert "active writers" in advance_error
    assert owner.exitcode == 0

    advanced_home = tmp_path / "advanced-home"
    advanced_home.mkdir()
    (advanced_home / "storage-format.json").write_text('{"writer_epoch":2}\n')
    session_recv, session_send = multiprocessing.Pipe(duplex=False)
    creator = _spawn(
        _create_session_process, str(advanced_home), str(tmp_path), session_send
    )
    advanced_session_id = session_recv.recv()
    creator.join()
    assert creator.exitcode == 0
    format_fd = os.open(advanced_home / ".storage-format.lock", os.O_RDWR)
    fcntl.flock(format_fd, fcntl.LOCK_EX)
    try:
        result_recv, result_send = multiprocessing.Pipe(duplex=False)
        contender = _spawn(
            _open_with_format_lock_probe,
            str(advanced_home),
            advanced_session_id,
            result_send,
        )
        assert result_recv.recv() == "attempting"
        assert contender.is_alive()
    finally:
        fcntl.flock(format_fd, fcntl.LOCK_UN)
        os.close(format_fd)
    assert result_recv.recv()[0] == "opened"
    contender.join()
    assert contender.exitcode == 0
    assert verify(advanced_home).writer_epoch == 2
    for connection in (
        owner_ready,
        owner_send,
        owner_release_recv,
        owner_release_send,
        session_recv,
        session_send,
        result_recv,
        result_send,
    ):
        connection.close()
    owner.close()
    creator.close()
    contender.close()
