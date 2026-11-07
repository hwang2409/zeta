from __future__ import annotations

import multiprocessing
import os
import signal
from pathlib import Path
from typing import Any

import pytest

from zeta.core.session import SessionManager
from zeta.core.store.runtime_lease import SessionOwnedError, StaleSessionOwnerError


def _hold_session(home: str, session_id: str, ready: Any, release: Any) -> None:
    manager = SessionManager(home, owner_surface="serve")
    with manager.open(session_id).store as store:
        ready.send(("owned", store.execution_lease.epoch))
        release.recv()


def _try_open(home: str, session_id: str, result: Any) -> None:
    manager = SessionManager(home, owner_surface="tui")
    try:
        with manager.open(session_id).store as store:
            result.send(("opened", store.execution_lease.epoch))
    except SessionOwnedError as exc:
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
    return session_id


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

    assert result[0:2] == ("error", "SessionOwnedError")
    release_send.send(None)
    owner.join()
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
    assert result[1] != first_epoch
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
    assert result_recv.recv()[0:2] == ("error", "SessionOwnedError")
    contender.join()

    os.kill(owner.pid, signal.SIGCONT)
    release_send.send(None)
    owner.join()
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
        with pytest.raises(Exception, match="read-only"):
            observer.append_pending_prompt("no")


def test_closed_capability_is_stale(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    store = SessionManager(home).open(session_id).store
    capability = store.execution_lease
    store.close()

    with pytest.raises(StaleSessionOwnerError):
        capability.assert_owned()

    with SessionManager(home).open(session_id).store as successor:
        assert successor.execution_lease.epoch != capability.epoch


def test_stale_store_and_metadata_capabilities_cannot_publish(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    manager = SessionManager(home)
    opened = manager.open(session_id)
    stale = opened.store.execution_lease
    stale.close()

    with pytest.raises(StaleSessionOwnerError):
        opened.store.append_pending_prompt("blocked")
    with pytest.raises(StaleSessionOwnerError):
        manager._mutate(
            session_id,
            lambda metadata: metadata,
            execution_lease=stale,
        )
    opened.store.close()

    with manager.open(session_id).store as successor:
        assert successor.pending_prompts() == []


def test_execution_descriptor_is_not_inheritable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    with SessionManager(home).open(session_id).store as store:
        assert os.get_inheritable(store.execution_lease._owned.fd) is False


def test_detached_writer_retains_the_resident_epoch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session_id = _created_session(home, tmp_path)
    with SessionManager(home).open(session_id).store as resident:
        detached = resident._new_persistent_durable_writer()
        try:
            assert detached.execution_lease.epoch == resident.execution_lease.epoch
        finally:
            detached.close()
