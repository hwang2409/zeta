from __future__ import annotations

import json
import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path

from zeta.core.session import SessionManager, SessionMetadata
from zeta.core.session_files import session_root
from zeta.session_retention import (
    RETENTION_PIN_FILENAME,
    RetentionBounds,
    RetentionPlanner,
    is_retention_pinned,
    remove_retention_pin,
    write_retention_pin,
)
from zeta.storage_format import open_writer


def _directory(path):
    path.mkdir()
    return session_root(path)


def test_pin_round_trip_and_idempotent_unpin(tmp_path):
    with _directory(tmp_path / "session") as directory_fd:
        assert not is_retention_pinned(directory_fd)
        write_retention_pin(directory_fd)
        assert is_retention_pinned(directory_fd)
        remove_retention_pin(directory_fd)
        remove_retention_pin(directory_fd)
        assert not is_retention_pinned(directory_fd)


def test_pin_write_is_atomic_and_directory_durable(tmp_path, monkeypatch):
    syncs = []
    real_fsync = os.fsync

    def record_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            syncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("zeta.core.session_files.os.fsync", record_fsync)
    with _directory(tmp_path / "session") as directory_fd:
        write_retention_pin(directory_fd)
        assert syncs
        syncs.clear()
        remove_retention_pin(directory_fd)
        assert syncs


def test_leftover_temp_file_is_pin(tmp_path):
    with _directory(tmp_path / "session") as directory_fd:
        (tmp_path / "session" / f".{RETENTION_PIN_FILENAME}.leftover.tmp").write_text(
            '{"schema_version":1,"pinned":true}'
        )
        assert is_retention_pinned(directory_fd)


def test_invalid_pin_files_fail_closed(tmp_path):
    cases = {
        "malformed": b"not json",
        "oversized": b"x" * 257,
        "wrong-version": b'{"schema_version":2,"pinned":true}',
        "false": b'{"schema_version":1,"pinned":false}',
        "empty": b"{}",
    }
    for name, content in cases.items():
        session = tmp_path / name
        with _directory(session) as directory_fd:
            pin = session / RETENTION_PIN_FILENAME
            pin.write_bytes(content)
            pin.chmod(0o600)
            assert is_retention_pinned(directory_fd)


def test_symlink_multi_link_and_nonregular_pin_files_fail_closed(tmp_path):
    session = tmp_path / "session"
    with _directory(session) as directory_fd:
        (session / "target").write_bytes(b'{"schema_version":1,"pinned":true}')
        (session / RETENTION_PIN_FILENAME).symlink_to("target")
        assert is_retention_pinned(directory_fd)
    session = tmp_path / "hardlink"
    with _directory(session) as directory_fd:
        pin = session / RETENTION_PIN_FILENAME
        pin.write_bytes(b'{"schema_version":1,"pinned":true}')
        os.link(pin, session / "other")
        assert is_retention_pinned(directory_fd)
    session = tmp_path / "directory"
    with _directory(session) as directory_fd:
        (session / RETENTION_PIN_FILENAME).mkdir()
        assert is_retention_pinned(directory_fd)
    session = tmp_path / "wrong-mode"
    with _directory(session) as directory_fd:
        pin = session / RETENTION_PIN_FILENAME
        pin.write_bytes(b'{"schema_version":1,"pinned":true}')
        pin.chmod(0o644)
        assert is_retention_pinned(directory_fd)


def test_metadata_writes_do_not_change_pin(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    manager = SessionManager(tmp_path)
    metadata = SessionMetadata.new(
        session_id="session",
        provider="codex",
        model="gpt-5.6-sol",
        cwd=str(tmp_path),
        retained_tail=10,
        compaction_budget=10,
    )
    with session_root(session) as directory_fd:
        write_retention_pin(directory_fd)
        manager._write_unlocked(metadata, directory_fd=directory_fd)
        assert is_retention_pinned(directory_fd)
        assert (session / RETENTION_PIN_FILENAME).read_bytes()


_DAY_NS = 86_400 * 1_000_000_000
_OLD = "2020-01-01T00:00:00+00:00"


def _manager(home):
    return SessionManager(home)


def _create_closed(manager, stack, **kwargs):
    opened = manager.create(
        provider="codex", model="gpt-5.6-sol", cwd=str(manager.home),
        auto_project=False, **kwargs,
    )
    opened.store.close()
    return opened.metadata.session_id


def _true_tree_bytes(session_dir) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(session_dir):
        for name in files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            total += os.path.getsize(path)
    return total


def test_closure_reasons_table(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        # One root per blocker in design §3.2 (1-7) plus age, unsafe, reserved,
        # and one genuinely closed, old, unreferenced candidate.
        held = manager.create(
            provider="codex", model="gpt-5.6-sol", cwd=str(tmp_path), auto_project=False
        )
        stack.callback(held.store.close)
        held_id = held.metadata.session_id

        live_child = _create_closed(manager, stack)
        child_dir = tmp_path / "sessions" / live_child / "agents" / "1"
        child_dir.mkdir(parents=True)
        (child_dir / "agent_lifecycle.json").write_text(
            json.dumps({"state": "running", "started_at": _OLD, "finished_at": None})
        )
        (child_dir / ".lock").write_text("")

        pinned = _create_closed(manager, stack)
        with session_root(tmp_path / "sessions" / pinned) as fd:
            write_retention_pin(fd)

        pending = _create_closed(manager, stack)
        (tmp_path / "sessions" / pending / "project_link_pending.json").write_text(
            json.dumps({"project_id": "p", "role": "session"})
        )

        parent = _create_closed(manager, stack)
        dependent = _create_closed(manager, stack, parent_session_id=parent)

        inbox = _create_closed(manager, stack)
        new_dir = tmp_path / "projects" / "p_x" / "inbox" / "new"
        new_dir.mkdir(parents=True)
        (new_dir / "m.json").write_text(
            json.dumps({"from": {"project": "p", "session": inbox}, "to_session": None})
        )

        reserved = _create_closed(manager, stack)
        reservations = tmp_path / "session-trash" / "reservations"
        reservations.mkdir(parents=True)
        (reservations / f"{reserved}.json").write_text("{}")

        unsafe = _create_closed(manager, stack)
        os.symlink("meta.json", tmp_path / "sessions" / unsafe / "danger")

        candidate = _create_closed(manager, stack)
        candidate_bytes = _true_tree_bytes(tmp_path / "sessions" / candidate)

        lease = stack.enter_context(_opened_lease(tmp_path))
        future = stack.enter_context(RetentionPlanner(
            tmp_path, lease, bounds=RetentionBounds(roots_advanced=32),
            now_ns=lambda: _future_ns(),
        ))
        result = future.run_pass(days=30)
        reasons = {row.root_id: (row.closed, row.reason) for row in result.reasons}

    assert reasons[held_id] == (False, "session is open or in use")
    assert reasons[live_child] == (False, "live or recoverable child")
    assert reasons[pinned] == (False, "pinned")
    assert reasons[pending] == (False, "pending project link")
    assert reasons[parent] == (False, "published top-level dependent")
    assert reasons[inbox] == (False, "pending inbox reference")
    assert reasons[reserved] == (False, "session is being deleted")
    assert reasons[unsafe] == (False, "unsafe entry in tree")
    assert reasons[dependent] == (True, None)
    assert reasons[candidate] == (True, None)
    # The candidate's exact retained bytes are reported in the preview.
    assert result.complete is True
    assert candidate in result.preview.root_ids
    assert result.preview.bytes >= candidate_bytes


def test_not_old_blocks_under_real_clock(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        fresh = _create_closed(manager, stack)
        lease = stack.enter_context(_opened_lease(tmp_path))
        import time

        planner = stack.enter_context(RetentionPlanner(tmp_path, lease, now_ns=time.time_ns))
        result = planner.run_pass(days=30)
        reasons = {row.root_id: (row.closed, row.reason) for row in result.reasons}
    assert reasons[fresh] == (False, "not old enough")


def test_resume_completes_over_passes_with_exact_bytes(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        # A terminal child store adds directories so the walk spans passes.
        child = tmp_path / "sessions" / root / "agents" / "1"
        child.mkdir(parents=True)
        (child / "agent_lifecycle.json").write_text(
            json.dumps({"state": "completed", "started_at": _OLD, "finished_at": _OLD})
        )
        (child / "conversation.jsonl").write_text("x" * 100)
        (child / ".lock").write_text("")
        expected = _true_tree_bytes(tmp_path / "sessions" / root)

        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = stack.enter_context(RetentionPlanner(
            tmp_path, lease, bounds=RetentionBounds(entries_per_root=1),
            now_ns=lambda: _future_ns(),
        ))
        passes = 0
        result = planner.run_pass(days=30)
        while not result.complete:
            passes += 1
            assert passes < 50
            result = planner.run_pass(days=30)
    assert passes >= 1  # the walk did not finish in a single pass
    assert root in result.preview.root_ids
    assert result.preview.bytes == expected


def test_touched_completed_directory_is_rewalked(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        child = tmp_path / "sessions" / root / "agents" / "1"
        child.mkdir(parents=True)
        (child / "agent_lifecycle.json").write_text(
            json.dumps({"state": "completed", "started_at": _OLD, "finished_at": _OLD})
        )
        (child / ".lock").write_text("")

        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = stack.enter_context(RetentionPlanner(
            tmp_path, lease, bounds=RetentionBounds(entries_per_root=1),
            now_ns=lambda: _future_ns(),
        ))
        first = planner.run_pass(days=30)
        assert first.complete is False  # root directory completed, walk suspended
        # A new file lands in the already-walked root directory.
        (tmp_path / "sessions" / root / "late.txt").write_text("y" * 321)
        expected = _true_tree_bytes(tmp_path / "sessions" / root)
        result = planner.run_pass(days=30)
        while not result.complete:
            result = planner.run_pass(days=30)
    # The stale completed directory was invalidated and re-walked, so the late
    # file is counted; a resume that trusted the fingerprint would undercount.
    assert result.preview.bytes == expected


def test_restart_rewalks_a_file_changed_between_planners(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        root_dir = tmp_path / "sessions" / root
        lease = stack.enter_context(_opened_lease(tmp_path))
        bounds = RetentionBounds(entries_per_root=1)
        planner_a = stack.enter_context(
            RetentionPlanner(tmp_path, lease, bounds=bounds, now_ns=_future_ns)
        )
        assert planner_a.run_pass(days=30).complete is False
        planner_a.close()  # process exit: the walked root directory is not rewalked
        # A file in the already-listed root directory changes; the directory
        # mtime does not, so only a fresh walk can see it.
        walked = sorted(p for p in root_dir.iterdir() if p.is_file())[0]
        with walked.open("ab") as stream:
            stream.write(b"z" * 17)
        recent = _future_ns() - _DAY_NS
        os.utime(walked, ns=(recent, recent))

        planner_b = stack.enter_context(
            RetentionPlanner(tmp_path, lease, bounds=bounds, now_ns=_future_ns)
        )
        result, reasons = _run_sweep(planner_b)
    assert reasons[root] == (False, "not old enough")
    assert root not in result.preview.root_ids


def test_below_feature_epoch_writes_nothing(tmp_path):
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":7}\n')
    lease = open_writer(tmp_path)
    try:
        planner = RetentionPlanner(tmp_path, lease)
        result = planner.run_pass(days=30)
    finally:
        lease.close()
    assert result.enabled is False
    assert result.complete is False
    assert not (tmp_path / "retention-state.json").exists()


def test_planner_entry_points_have_no_production_callers():
    root = Path(__file__).resolve().parents[1] / "src" / "zeta"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "session_retention.py":
            continue  # the module that defines the planner
        text = path.read_text(encoding="utf-8")
        if "RetentionPlanner" in text or ".run_pass(" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == [], f"planner must stay dormant, used in: {offenders}"


def _future_ns() -> int:
    import time

    return time.time_ns() + 60 * _DAY_NS


def _run_sweep(planner, days=30):
    """Run passes to the end of one sweep; return the result and every reason."""
    reasons = {}
    for _ in range(500):
        result = planner.run_pass(days=days)
        reasons.update({row.root_id: (row.closed, row.reason) for row in result.reasons})
        if result.complete:
            return result, reasons
    raise AssertionError("sweep did not complete in 500 passes")


@contextmanager
def _opened_lease(home):
    lease = open_writer(home)
    try:
        yield lease
    finally:
        lease.close()
