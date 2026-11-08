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

from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.session import SessionManager
from zeta.storage_format import WRITER_EPOCH, open_writer

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
        "through zeta.lease_release.release_later (fd-only finalizers may use "
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
