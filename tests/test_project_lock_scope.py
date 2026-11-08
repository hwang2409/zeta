from __future__ import annotations

import fcntl
import json
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from zeta.core.session import SessionManager
from zeta.core.session_links import (
    PENDING_CHILD_LINKS_DIRNAME,
    persist_pending_child_link,
    reconcile_child_links,
)
from zeta.project_registry import ProjectRegistry
from zeta.storage_format import WRITER_EPOCH, advance


def _send(channel: socket.socket, value: str) -> None:
    channel.sendall(value.encode("ascii") + b"\n")


def _receive(channel: socket.socket) -> str:
    channel.settimeout(10)
    return channel.recv(64).decode("ascii").strip()


def _hold_memory_publication(home: str, project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    original = registry._acquire_lock

    def gated(fd: int, mode: int) -> None:
        original(fd, mode)
        if mode == fcntl.LOCK_EX:
            _send(channel, "acquired")
            if _receive(channel) != "release":
                raise AssertionError("release signal was not sent")

    registry._acquire_lock = gated  # type: ignore[method-assign]
    state = registry.memory_state(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=state.digest,
        updates={"brief.md": "project a\n"},
        provenance={"session_id": "a" * 32, "seq_start": 1, "seq_end": 1},
    )
    _send(channel, "a-done")


def _publish_other_project(home: str, project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    state = registry.memory_state(project_id)
    registry.compare_and_swap_memory(
        project_id,
        expected_digest=state.digest,
        updates={"brief.md": "project b\n"},
        provenance={"session_id": "b" * 32, "seq_start": 1, "seq_end": 1},
    )
    _send(channel, "b-done")


def _record_session(home: str, project_id: str, channel: socket.socket) -> None:
    opened = SessionManager(home).create(
        provider="codex",
        model="test",
        cwd=home,
        project_id=project_id,
        auto_project=False,
    )
    opened.store.close()
    linked = any(
        row["session_id"] == opened.metadata.session_id
        for row in SessionManager(home).project_registry.list_session_links(project_id)
    )
    _send(channel, "session-done" if linked else "session-pending")


def _hold_sessions_publication(
    home: str, project_id: str, channel: socket.socket
) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    project_lock = getattr(registry, "_project_lock", None)
    authority = (
        project_lock(project_id, "sessions", exclusive=True)
        if project_lock is not None
        else registry._locked(write=True)
    )
    with authority:
        _send(channel, "acquired")
        if _receive(channel) != "release":
            raise AssertionError("release signal was not sent")
    _send(channel, "sessions-done")


def _reconcile_child_worker(
    home: str, value: str, channel: socket.socket
) -> None:
    project_id, root_id = value.split("|", 1)
    manager = SessionManager(home)
    reconcile_child_links(manager.project_registry, manager.sessions_dir, root_id)
    if any(
        row["session_id"] == f"{root_id}:child"
        for row in manager.project_registry.list_session_links(project_id)
    ):
        _send(channel, "child-done")
    else:
        _send(channel, "child-missing")


def _hold_identity_publication(home: str, _project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    original = registry._acquire_lock

    def gated(fd: int, mode: int) -> None:
        original(fd, mode)
        if mode == fcntl.LOCK_EX:
            _send(channel, "acquired")
            if _receive(channel) != "release":
                raise AssertionError("release signal was not sent")

    registry._acquire_lock = gated  # type: ignore[method-assign]
    registry.create_project("new", "new")
    _send(channel, "identity-done")


_WORKERS = {
    "hold-memory": _hold_memory_publication,
    "hold-identity": _hold_identity_publication,
    "hold-sessions": _hold_sessions_publication,
    "publish-memory": _publish_other_project,
    "record-session": _record_session,
    "reconcile-child": _reconcile_child_worker,
}


def _start_worker(kind: str, home: Path, project_id: str) -> tuple[subprocess.Popen[bytes], socket.socket]:
    parent, child = socket.socketpair()
    process = subprocess.Popen(
        [sys.executable, __file__, kind, str(home), project_id, str(child.fileno())],
        pass_fds=(child.fileno(),),
    )
    child.close()
    return process, parent


def _run_while_lock_is_held(
    home: Path,
    *,
    holder_kind: str,
    holder_project: str,
    other_kind: str,
    other_project: str,
    expected: str,
) -> None:
    holder, holder_channel = _start_worker(holder_kind, home, holder_project)
    other: subprocess.Popen[bytes] | None = None
    other_channel: socket.socket | None = None
    try:
        assert _receive(holder_channel) == "acquired", (
            "held mutation did not acquire its lock"
        )
        other, other_channel = _start_worker(other_kind, home, other_project)
        assert _receive(other_channel) == expected, (
            "unrelated public mutation remained serialized"
        )
        _send(holder_channel, "release")
        assert _receive(holder_channel).endswith("done")
        assert holder.wait(timeout=10) == 0
        assert other.wait(timeout=10) == 0
    finally:
        for process in (holder, other):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        holder_channel.close()
        if other_channel is not None:
            other_channel.close()


if __name__ == "__main__":
    worker = _WORKERS[sys.argv[1]]
    channel = socket.socket(fileno=int(sys.argv[4]))
    try:
        worker(sys.argv[2], sys.argv[3], channel)
    finally:
        channel.close()

def test_p06_1_unrelated_projects_publish_without_root_lock_serialization(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project_a = registry.create_project("a", "a")
    project_b = registry.create_project("b", "b")
    registry.initialize_memory(project_a.project_id)
    registry.initialize_memory(project_b.project_id)

    _run_while_lock_is_held(
        home,
        holder_kind="hold-memory",
        holder_project=project_a.project_id,
        other_kind="publish-memory",
        other_project=project_b.project_id,
        expected="b-done",
    )


def test_p06_2_memory_and_sessions_domains_do_not_serialize(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo")
    registry.initialize_memory(project.project_id)

    _run_while_lock_is_held(
        home,
        holder_kind="hold-memory",
        holder_project=project.project_id,
        other_kind="record-session",
        other_project=project.project_id,
        expected="session-done",
    )


def test_p06_3_identity_and_domain_publications_do_not_serialize(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo")
    registry.initialize_memory(project.project_id)

    _run_while_lock_is_held(
        home,
        holder_kind="hold-identity",
        holder_project=project.project_id,
        other_kind="publish-memory",
        other_project=project.project_id,
        expected="b-done",
    )

def test_p06_4_epoch_selects_legacy_then_named_authority(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    for epoch in range(WRITER_EPOCH - 1):
        advance(home, expected_epoch=epoch, target_epoch=epoch + 1)
    legacy = ProjectRegistry(home / "projects")
    old_project = legacy.create_project("old", "old")
    legacy.initialize_memory(old_project.project_id)
    assert (home / "projects" / ".lock").is_file()
    assert not (home / "projects" / old_project.project_id / ".memory.lock").exists()

    del legacy
    advance(home, expected_epoch=WRITER_EPOCH - 1, target_epoch=WRITER_EPOCH)
    current = ProjectRegistry(home / "projects")
    current.load_memory(old_project.project_id)
    assert (home / "projects" / old_project.project_id / ".memory.lock").is_file()


def _project_session_row(
    registry: ProjectRegistry, project_id: str, session_id: str
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in registry.list_session_links(project_id)
            if row["session_id"] == session_id
        ),
        None,
    )


def _replace_pending(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_p06_5_root_reconciliation_revalidates_replaced_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    manager = SessionManager(home)
    registry = manager.project_registry
    old = registry.create_project("old", "old")
    stale = registry.create_project("stale", "stale")
    latest = registry.create_project("latest", "latest")
    opened = manager.create(
        provider="codex",
        model="test",
        cwd=tmp_path,
        project_id=old.project_id,
        auto_project=False,
    )
    session_id = opened.metadata.session_id
    pending_path = manager.sessions_dir / session_id / "project_link_pending.json"
    stale_intent = {
        "project_id": stale.project_id,
        "previous_project_id": old.project_id,
        "role": "worker",
        "parent_session_id": "stale-parent",
        "transcript_path": str(manager.sessions_dir / session_id),
    }
    latest_intent = {
        **stale_intent,
        "project_id": latest.project_id,
        "role": "orchestrator",
        "parent_session_id": "latest-parent",
    }
    _replace_pending(pending_path, stale_intent)

    reached_publication = threading.Event()
    release_publication = threading.Event()
    if hasattr(registry, "_project_locks"):
        original = registry._project_locks
        first = True

        @contextmanager
        def gated(*args: object, **kwargs: object):
            nonlocal first
            if first:
                first = False
                reached_publication.set()
                assert release_publication.wait(10), "replacement gate was not released"
            with original(*args, **kwargs) as value:
                yield value

        # test-contract: seam replace the root intent during authority acquisition
        monkeypatch.setattr(registry, "_project_locks", gated)
    else:
        original_record = registry.record_session  # type: ignore[attr-defined]

        def gated_record(*args: object, **kwargs: object) -> None:
            reached_publication.set()
            assert release_publication.wait(10), "replacement gate was not released"
            original_record(*args, **kwargs)

        monkeypatch.setattr(registry, "record_session", gated_record)

    failure: list[BaseException] = []

    def reconcile() -> None:
        try:
            manager.list_sessions()
        except BaseException as exc:  # noqa: BLE001 - thread reports every failure
            failure.append(exc)

    worker = threading.Thread(target=reconcile)
    worker.start()
    try:
        assert reached_publication.wait(10), "reconciler did not reach publication"
        _replace_pending(pending_path, latest_intent)
        release_publication.set()
        worker.join(10)
        assert not worker.is_alive(), "root reconciliation did not finish"
    finally:
        release_publication.set()
        worker.join(10)
        opened.store.close()
    assert failure == []
    rows = {
        project.project_id: _project_session_row(registry, project.project_id, session_id)
        for project in (old, stale, latest)
    }
    assert rows[stale.project_id] is None, "replaced root intent published a stale row"
    assert rows[old.project_id] is None
    latest_row = rows[latest.project_id]
    assert latest_row is not None
    assert latest_row["role"] == "orchestrator"
    assert latest_row["parent_session_id"] == "latest-parent"
    assert not pending_path.exists()
    assert sum(row is not None for row in rows.values()) == 1


@pytest.mark.parametrize(
    "boundary",
    ("before-old-removal", "between-removal-upsert", "after-upsert-before-cleanup"),
)
def test_p06_5_root_reconciliation_recovers_each_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    home = tmp_path / boundary
    advance(home, expected_epoch=0)
    manager = SessionManager(home)
    registry = manager.project_registry
    old = registry.create_project("old", "old")
    new = registry.create_project("new", "new")
    opened = manager.create(
        provider="codex",
        model="test",
        cwd=tmp_path,
        project_id=old.project_id,
        auto_project=False,
    )
    session_id = opened.metadata.session_id
    pending_path = manager.sessions_dir / session_id / "project_link_pending.json"

    calls = 0
    original_update = getattr(registry, "_update_sessions_locked", None)
    if original_update is not None and boundary != "after-upsert-before-cleanup":
        def crash_update(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if boundary == "before-old-removal" or calls == 2:
                raise OSError(f"crash {boundary}")
            original_update(*args, **kwargs)

        # test-contract: durability inject each session-publication crash boundary
        monkeypatch.setattr(registry, "_update_sessions_locked", crash_update)
    elif original_update is not None:
        import zeta.core.session as session_module

        original_unlink = session_module.os.unlink
        crashed = False

        def crash_cleanup(path: object, *args: object, **kwargs: object) -> None:
            nonlocal crashed
            if path == "project_link_pending.json" and not crashed:
                crashed = True
                raise OSError("crash after upsert")
            original_unlink(path, *args, **kwargs)

        # test-contract: durability fail after upsert and before intent cleanup
        monkeypatch.setattr(session_module.os, "unlink", crash_cleanup)

    manager.associate_project(opened.metadata, new.project_id)
    old_row = _project_session_row(registry, old.project_id, session_id)
    new_row = _project_session_row(registry, new.project_id, session_id)
    assert not (old_row is not None and new_row is not None), (
        "a completed crash-boundary observation exposed two active project rows"
    )
    monkeypatch.undo()
    retry_failures: list[BaseException] = []

    def retry() -> None:
        try:
            manager.list_sessions()
        except BaseException as exc:  # noqa: BLE001 - thread reports every failure
            retry_failures.append(exc)

    reconcilers = [threading.Thread(target=retry) for _ in range(2)]
    for reconciler in reconcilers:
        reconciler.start()
    for reconciler in reconcilers:
        reconciler.join(10)
        assert not reconciler.is_alive(), "root reconciler did not make progress"
    assert retry_failures == []
    assert _project_session_row(registry, old.project_id, session_id) is None
    repaired = _project_session_row(registry, new.project_id, session_id)
    assert repaired is not None
    assert repaired["transcript_path"] == str(manager.sessions_dir / session_id)
    assert not pending_path.exists()
    opened.store.close()


def test_p06_5_child_reconciliation_revalidates_replaced_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    manager = SessionManager(home)
    registry = manager.project_registry
    project = registry.create_project("demo", "demo")
    root = manager.create(
        provider="codex", model="test", cwd=tmp_path, auto_project=False
    )
    root_id = root.metadata.session_id
    child_id = f"{root_id}:child"
    stale = {
        "project_id": project.project_id,
        "session_id": child_id,
        "role": "worker",
        "parent_session_id": root_id,
        "transcript_path": "/stale",
    }
    latest = {**stale, "role": "orchestrator", "transcript_path": "/latest"}
    persist_pending_child_link(
        manager.sessions_dir / root_id, stale, manager._storage_lease
    )
    pending_dir = manager.sessions_dir / root_id / PENDING_CHILD_LINKS_DIRNAME
    pending_path = next(pending_dir.iterdir())

    reached_publication = threading.Event()
    release_publication = threading.Event()
    if hasattr(registry, "_project_locks"):
        original = registry._project_locks

        @contextmanager
        def gated(*args: object, **kwargs: object):
            reached_publication.set()
            assert release_publication.wait(10), "replacement gate was not released"
            with original(*args, **kwargs) as value:
                yield value

        # test-contract: seam replace the child intent during authority acquisition
        monkeypatch.setattr(registry, "_project_locks", gated)
    else:
        import zeta.core.session_links as session_links_module

        original_batch = session_links_module._record_child_lineage_batch

        def gated_batch(*args: object, **kwargs: object) -> None:
            reached_publication.set()
            assert release_publication.wait(10), "replacement gate was not released"
            original_batch(*args, **kwargs)

        # test-contract: seam replace the child intent during batch publication
        monkeypatch.setattr(
            session_links_module, "_record_child_lineage_batch", gated_batch
        )

    worker = threading.Thread(
        target=reconcile_child_links,
        args=(registry, manager.sessions_dir, root_id),
    )
    worker.start()
    try:
        assert reached_publication.wait(10), "child reconciler did not reach publication"
        _replace_pending(pending_path, latest)
        release_publication.set()
        worker.join(10)
        assert not worker.is_alive(), "child reconciliation did not finish"
    finally:
        release_publication.set()
        worker.join(10)
    row = _project_session_row(registry, project.project_id, child_id)
    assert row is None, "changed full-value child intent was published by stale reconciler"
    assert pending_path.exists(), "stale reconciler cleaned the replacement child intent"
    monkeypatch.undo()
    reconcile_child_links(registry, manager.sessions_dir, root_id)
    row = _project_session_row(registry, project.project_id, child_id)
    assert row is not None
    assert row["role"] == "orchestrator"
    assert row["transcript_path"] == "/latest"
    assert not pending_path.exists()
    root.store.close()


def test_p06_8_root_reconciliation_process_progress(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    held = registry.create_project("held", "held")
    target = registry.create_project("target", "target")

    _run_while_lock_is_held(
        home,
        holder_kind="hold-sessions",
        holder_project=held.project_id,
        other_kind="record-session",
        other_project=target.project_id,
        expected="session-done",
    )


def test_p06_8_child_reconciliation_process_progress(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    held = registry.create_project("held", "held")
    target = registry.create_project("target", "target")
    manager = SessionManager(home)
    root = manager.create(
        provider="codex", model="test", cwd=tmp_path, auto_project=False
    )
    root_id = root.metadata.session_id
    persist_pending_child_link(
        manager.sessions_dir / root_id,
        {
            "project_id": target.project_id,
            "session_id": f"{root_id}:child",
            "role": "worker",
            "parent_session_id": root_id,
            "transcript_path": "/child",
        },
        manager._storage_lease,
    )
    root.store.close()

    _run_while_lock_is_held(
        home,
        holder_kind="hold-sessions",
        holder_project=held.project_id,
        other_kind="reconcile-child",
        other_project=f"{target.project_id}|{root_id}",
        expected="child-done",
    )


def test_storage_d1_project_lock_obeys_global_rank_order(tmp_path: Path) -> None:
    from zeta.locking import execution_lock, shape_lock

    home = tmp_path / ".zeta"
    advance(home, expected_epoch=0)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo")

    with execution_lock(), registry._project_lock(
        project.project_id, "memory", exclusive=False
    ) as project_fd:
        assert project_fd >= 0

    with (
        shape_lock(),
        pytest.raises(RuntimeError, match="filesystem lock order inversion"),
        registry._project_lock(project.project_id, "memory", exclusive=False),
    ):
        pass


def test_p06_8_rank_guard_rejects_inversions_and_unsorted_domains() -> None:
    try:
        from zeta.locking import (
            domain_lock,
            execution_lock,
            format_lock,
            identity_lock,
            shape_lock,
        )
    except ModuleNotFoundError:
        pytest.skip("rank guard does not exist before p06")

    with (
        shape_lock(),
        pytest.raises(RuntimeError, match="order inversion"),
        identity_lock(),
    ):
        pass
    with execution_lock(), domain_lock("p_a", "memory"):
        pass
    with (
        shape_lock(),
        pytest.raises(RuntimeError, match="order inversion"),
        domain_lock("p_a", "memory"),
    ):
        pass
    with (
        domain_lock("p_a", "memory"),
        pytest.raises(RuntimeError, match="order inversion"),
        format_lock(),
    ):
        pass
    with (
        domain_lock("p_a", "sessions"),
        pytest.raises(RuntimeError, match="sorted order"),
        domain_lock("p_a", "memory"),
    ):
        pass
