from __future__ import annotations

import ast
import fcntl
import gc
import os
import threading
import weakref
from collections.abc import Callable
from pathlib import Path

import pytest

from zeta.core import session_execution
from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.session import SessionManager
from zeta.storage_format import (
    WRITER_EPOCH,
    RegistryLock,
    drain_releases,
    open_writer,
    release_later,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "zeta"
_DEADLINE_SECONDS = 10


def _finalizer_violations() -> list[str]:
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            where = f"{path.relative_to(SRC.parent)}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.FunctionDef) and node.name == "__del__":
                violations.append(f"{where} defines __del__")
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "weakref"
                and any(alias.name == "finalize" for alias in node.names)
            ):
                violations.append(f"{where} imports finalize unqualified")
            if not isinstance(node, ast.Call) or ast.unparse(node.func) != (
                "weakref.finalize"
            ):
                continue
            callback = ast.unparse(node.args[1]) if len(node.args) > 1 else ""
            if callback not in {"os.close", "release_later"}:
                violations.append(f"{where} finalizes with {callback or '?'}")
    return violations


def test_finalizers_only_close_fds_or_defer_lease_release() -> None:
    violations = _finalizer_violations()
    assert not violations, (
        "a finalizer can run inside a held registry lock; route lease release "
        "through zeta.storage_format.release_later (fd-only finalizers may use "
        "os.close):\n" + "\n".join(violations)
    )


def _run_bounded(call: Callable[[], object]) -> None:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread
            errors.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_DEADLINE_SECONDS)
    if worker.is_alive():
        # The hung thread still holds the registry lock; any cleanup would hang too.
        pytest.fail("a lease finalizer run by GC under a held registry lock deadlocked")
    if errors:
        raise errors[0]


def _exclusive_flock_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


class _CollectOnCompare(str):
    """An epoch whose comparison inside assert_owned's locked check runs the GC."""

    __hash__ = str.__hash__

    def __ne__(self, other: object) -> bool:
        gc.collect()
        return str.__ne__(self, other)


def test_gc_finalized_store_under_assert_owned_releases_without_deadlock(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    manager = SessionManager(home)
    created = [
        manager.create(provider="codex", model="offline", cwd=tmp_path)
        for _ in range(2)
    ]
    dropped_id = created[0].metadata.session_id
    created[1].store.close()
    survivor = manager.open(created[1].metadata.session_id).store
    gc.disable()
    try:
        dropped = created[0].store
        dropped_epoch = dropped.execution_lease.epoch
        dropped._cycle = dropped  # only the collector can free it now
        dropped_ref = weakref.ref(dropped)
        del created, dropped
        assert dropped_ref() is not None

        lease = survivor.execution_lease
        _run_bounded(lambda: lease.assert_owned(_CollectOnCompare(lease.epoch)))

        assert dropped_ref() is None, "the collector did not finalize the store"
        lease_path = home / "sessions" / dropped_id / "runtime.lease"
        assert _exclusive_flock_free(lease_path), "the finalized lease was not released"
        with manager.open(dropped_id).store as reopened:
            assert reopened.execution_lease.epoch != dropped_epoch
    finally:
        gc.enable()
    survivor.close()
    manager._release_storage_lease()


class _CollectOnCapability(int):
    """A capability whose comparison inside open_writer's locked lookup runs the GC."""

    def __lt__(self, other: object) -> bool:
        gc.collect()
        return int.__lt__(self, other)


def _untracked(cls: type, *args: object) -> object:
    # The repository conftest wraps owner constructors to keep owners (and an
    # extra home writer lease) alive for cleanup; this test needs the collector
    # to free the only owner of the home lease.
    owner = cls.__new__(cls)
    getattr(cls.__init__, "__wrapped__", cls.__init__)(owner, *args)
    return owner


def _manager_owner(home: Path) -> object:
    return _untracked(SessionManager, home)


def _workspace_owner(home: Path) -> object:
    return _untracked(WorkspaceSnapshotStore, home / "sessions" / "owner", "owner")


@pytest.mark.parametrize("make_owner", [_manager_owner, _workspace_owner])
def test_gc_finalized_writer_owner_during_open_writer_keeps_home_locked(
    tmp_path: Path, make_owner: Callable[[Path], object]
) -> None:
    home = tmp_path / "home"
    open_writer(home).close()  # bootstrap the manifest
    gc.disable()
    try:
        owner = make_owner(home)
        owner._cycle = owner  # type: ignore[attr-defined]
        owner_ref = weakref.ref(owner)
        del owner
        leases = []
        _run_bounded(
            lambda: leases.append(
                open_writer(home, capability=_CollectOnCapability(WRITER_EPOCH))
            )
        )
        assert owner_ref() is None, "the collector did not finalize the owner"
    finally:
        gc.enable()
    lock_path = home / ".storage-format.lock"
    try:
        assert not _exclusive_flock_free(lock_path), (
            "a live writer lease must keep the home lock held"
        )
    finally:
        leases[0].close()
    assert _exclusive_flock_free(lock_path)


def test_gc_finalizer_on_another_thread_queues_while_a_registry_lock_is_held(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    lease_path = home / "sessions" / opened.metadata.session_id / "runtime.lease"
    holding = threading.Event()
    finalizer_returned = threading.Event()
    waited: list[bool] = []

    def hold_registry_until_finalizer_returns() -> None:
        with session_execution._REGISTRY_LOCK:
            holding.set()
            waited.append(finalizer_returned.wait(_DEADLINE_SECONDS))

    holder = threading.Thread(target=hold_registry_until_finalizer_returns, daemon=True)
    gc.disable()
    try:
        dropped = opened.store
        dropped._cycle = dropped  # only the collector can free it now
        dropped_ref = weakref.ref(dropped)
        del opened, dropped
        holder.start()
        assert holding.wait(_DEADLINE_SECONDS), "the holder never took the lock"
        gc.collect()  # runs the store finalizer on this thread
        finalizer_returned.set()
        holder.join(_DEADLINE_SECONDS)
    finally:
        finalizer_returned.set()
        gc.enable()
        manager._release_storage_lease()
    assert not holder.is_alive(), "the registry holder never let go"
    assert dropped_ref() is None, "the collector did not finalize the store"
    assert waited == [True], (
        "the finalizer blocked on a registry lock held by another thread"
    )
    assert _exclusive_flock_free(lease_path), (
        "the queued release did not run after the holder let go"
    )


def test_gc_during_a_drain_queues_without_recursion_and_all_closes_run() -> None:
    ran: list[tuple[str, int]] = []
    depth = [0]
    registry = RegistryLock()

    class Owner:
        pass

    def drained(call: Callable[..., None]) -> Callable[..., None]:
        def run(*args: object) -> None:
            depth[0] += 1
            try:
                call(*args)
            finally:
                depth[0] -= 1

        return run

    def close(name: str) -> None:
        with registry:  # this exit is inside the drain and must not drain again
            ran.append((name, depth[0]))

    def close_that_collects() -> None:
        owner = Owner()
        owner.cycle = owner  # type: ignore[attr-defined]
        weakref.finalize(owner, release_later, drained(close), "collected")
        del owner
        release_later(drained(close), "queued")
        gc.collect()
        close("first")

    def queue_under_lock() -> None:
        with registry:
            release_later(drained(close_that_collects))

    _run_bounded(queue_under_lock)
    assert ran == [("first", 1), ("queued", 1), ("collected", 1)], (
        "closes queued during a drain must run after it, in order, never nested "
        "inside another queued close"
    )


def test_explicit_close_frees_its_locks_while_another_thread_is_draining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_close = session_execution.close_store_leases
    drainer_running = threading.Event()
    open_drainer = threading.Event()
    drainer_moved = threading.Event()
    release_on_drainer = threading.Event()
    drainer_ident: list[int] = []

    def close_store_leases(*args: object) -> None:
        if threading.get_ident() in drainer_ident:
            drainer_moved.set()
            release_on_drainer.wait(_DEADLINE_SECONDS)
        real_close(*args)

    def gated_close() -> None:
        # Holding a registry lock keeps other threads' lock exits from draining.
        with RegistryLock():
            drainer_running.set()
            open_drainer.wait(_DEADLINE_SECONDS)

    def drain_on_another_thread() -> None:
        drainer_ident.append(threading.get_ident())
        release_later(gated_close)
        drain_releases()
        drainer_moved.set()

    def hand_over_to_drainer() -> None:
        open_drainer.set()
        drainer_moved.wait(_DEADLINE_SECONDS)

    monkeypatch.setattr(session_execution, "close_store_leases", close_store_leases)
    home = tmp_path / "home"
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    store = opened.store
    session_dir = home / "sessions" / opened.metadata.session_id
    drainer = threading.Thread(target=drain_on_another_thread, daemon=True)
    try:
        drainer.start()
        assert drainer_running.wait(_DEADLINE_SECONDS), "the drainer never started"
        release_later(hand_over_to_drainer)  # queued ahead of the store's release
        store.close()
        fd = os.open(session_dir, os.O_RDONLY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pytest.fail("explicit close returned before its own lease was released")
        finally:
            os.close(fd)
    finally:
        open_drainer.set()
        release_on_drainer.set()
        drainer.join(_DEADLINE_SECONDS)
        manager._release_storage_lease()
    assert not drainer.is_alive(), "the drainer never finished"


def test_next_lease_acquire_runs_releases_queued_with_no_lock_held(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    session_id = opened.metadata.session_id
    gc.disable()
    try:
        dropped = opened.store
        dropped_epoch = dropped.execution_lease.epoch
        dropped._cycle = dropped
        del opened, dropped
        gc.collect()  # no registry lock is held: the release only queues
    finally:
        gc.enable()
    directory_fd = os.open(home / "sessions" / session_id, os.O_RDONLY)
    try:
        lease = session_execution.SessionExecutionLease.acquire(
            directory_fd,
            session_id=session_id,
            owner=session_execution.OwnerHint("headless"),
        )
        try:
            assert lease.epoch != dropped_epoch, (
                "acquire reused the dropped owner instead of releasing it first"
            )
        finally:
            lease.close()
    finally:
        os.close(directory_fd)
        manager._release_storage_lease()
