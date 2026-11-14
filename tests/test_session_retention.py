from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import stat
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from zeta import session_retention
from zeta.attention_records import AttentionStore
from zeta.core.session import SessionManager, SessionMetadata
from zeta.core.session_files import SessionError, session_root
from zeta.core.transcript_recovery import EvidenceManifest, RecoveryRecord, RepairTarget
from zeta.project_inbox import ProjectInbox
from zeta.project_inbox_sent import SentMessageTracker
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.session_retention import RetentionBounds, RetentionPlanner
from zeta.session_retention.pin import (
    RETENTION_PIN_FILENAME,
    is_retention_pinned,
    remove_retention_pin,
    write_retention_pin,
)
from zeta.storage_format import open_writer
from zeta.transcript_search.index import TranscriptIndex, TranscriptSource


def _directory(path):
    path.mkdir()
    return session_root(path)


def test_retention_history_trims_oldest_row(tmp_path):
    from zeta.session_retention import append_retention_history, retention_result

    with _opened_lease(tmp_path) as lease:
        for index in range(129):
            append_retention_history(
                tmp_path,
                lease,
                retention_result(timestamp=str(index), mode="preview", days=30),
            )
    rows = session_retention.read_retention_history(tmp_path)
    assert len(rows) == 128
    assert rows[0]["timestamp"] == "1"
    assert rows[-1]["timestamp"] == "128"


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


def _planner(stack, *args, **kwargs):
    planner = RetentionPlanner(*args, **kwargs)
    stack.callback(getattr(planner, "close", lambda: None))
    return planner


def test_not_old_blocks_under_real_clock(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        fresh = _create_closed(manager, stack)
        aged = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / aged)
        recent_meta = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / recent_meta, keep_metadata=True)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=time.time_ns)
        _, reasons = _run_sweep(planner)
    assert reasons[fresh] == (False, "not old enough")
    assert reasons[aged] == (True, None)
    # Metadata timestamps are activity even when every mtime is old.
    assert reasons[recent_meta] == (False, "not old enough")


def test_resume_completes_over_passes_with_exact_bytes(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        # A terminal child store adds directories so the walk spans passes.
        child = tmp_path / "sessions" / root / "agents" / "1"
        child.mkdir(parents=True)
        (child / "agent_lifecycle.json").write_text(
            json.dumps({"handle": "child", "state": "completed", "started_at": _OLD, "finished_at": _OLD})
        )
        (child / "conversation.jsonl").write_text("x" * 100)
        (child / ".lock").write_text("")
        expected = _true_tree_bytes(tmp_path / "sessions" / root)

        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(
            stack, tmp_path, lease, bounds=RetentionBounds(entries_per_root=1),
            now_ns=lambda: _future_ns(),
        )
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
            json.dumps({"handle": "child", "state": "completed", "started_at": _OLD, "finished_at": _OLD})
        )
        (child / ".lock").write_text("")

        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(
            stack, tmp_path, lease, bounds=RetentionBounds(entries_per_root=1),
            now_ns=lambda: _future_ns(),
        )
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
        _age_tree(root_dir)
        (root_dir / "bulk").mkdir()
        for index in range(20):
            (root_dir / "bulk" / f"{index}.txt").write_text("x")
        lease = stack.enter_context(_opened_lease(tmp_path))
        bounds = RetentionBounds(entries_per_root=1)
        planner_a = _planner(
            stack, tmp_path, lease, bounds=bounds, now_ns=_future_ns
        )
        planner_a.run_pass(days=30)
        getattr(planner_a, "close", lambda: None)()  # process exit: the walked root directory is not rewalked
        # A file in the already-listed root directory changes; the directory
        # mtime does not, so only a fresh walk can see it.
        walked = min(p for p in root_dir.iterdir() if p.is_file())
        with walked.open("ab") as stream:
            stream.write(b"z" * 17)
        recent = _future_ns() - _DAY_NS
        os.utime(walked, ns=(recent, recent))

        planner_b = _planner(
            stack, tmp_path, lease, bounds=bounds, now_ns=_future_ns
        )
        result, reasons = _run_sweep(planner_b)
    assert reasons[root] == (False, "not old enough")
    assert root not in result.preview.root_ids


def test_every_root_is_swept_and_the_next_sweep_starts_fresh(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        ids = sorted(_create_closed(manager, stack) for _ in range(40))
        # The dependent sorts last, past every per-pass candidate bound.
        parent, dependent = ids[0], ids[-1]
        meta = tmp_path / "sessions" / dependent / "meta.json"
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "parent_session_id": parent}))
        expected = sum(_true_tree_bytes(tmp_path / "sessions" / i) for i in ids[1:])
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        first, reasons = _run_sweep(planner)
        second, _ = _run_sweep(planner)
    assert reasons[parent] == (False, "published top-level dependent")
    assert sorted(first.preview.root_ids) == ids[1:]
    assert first.preview.bytes == expected
    # A finished sweep is not resumed: the next one recounts from scratch.
    assert second.preview.preview_id != first.preview.preview_id
    assert sorted(second.preview.root_ids) == ids[1:]
    assert second.preview.bytes == expected


def test_in_flight_walk_keeps_conforming_writers_out(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_with_message(manager, "first")
        child = tmp_path / "sessions" / root / "agents" / "1"
        child.mkdir(parents=True)
        (child / "agent_lifecycle.json").write_text(
            json.dumps({"handle": "child", "state": "completed", "started_at": _OLD, "finished_at": _OLD})
        )
        _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, bounds=RetentionBounds(entries_per_root=1))
        planner.run_pass(days=30)  # the root directory is walked; the walk is suspended
        # Walk progress stays authoritative only because no conforming writer
        # can append between passes (a stale age would select an active root).
        with pytest.raises(SessionError):
            manager.open(root).store.close()
        result, reasons = _run_sweep(planner)
        manager.open(root).store.close()  # the finished sweep holds no lease
    assert reasons[root] == (True, None)
    assert result.preview.root_ids == (root,)


def test_inherited_pin_blocks_descendant(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        parent = _create_closed(manager, stack)
        child = _create_closed(manager, stack, parent_session_id=parent)
        with session_root(tmp_path / "sessions" / parent) as parent_fd:
            write_retention_pin(parent_fd)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        result, reasons = _run_sweep(planner)
    assert reasons[child] == (False, "pinned")
    assert child not in result.preview.root_ids


def test_unreadable_entry_blocks_only_its_root(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        ids = sorted(_create_closed(manager, stack) for _ in range(2))
        blocked = tmp_path / "sessions" / ids[0] / "spill"
        blocked.mkdir()
        blocked.chmod(0)
        stack.callback(blocked.chmod, 0o700)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        result, reasons = _run_sweep_without_error(planner)
    assert result is not None
    assert reasons[ids[0]] == (False, "unreadable entry: Permission denied")
    assert reasons[ids[1]] == (True, None)
    assert result.preview.root_ids == (ids[1],)


def test_bounds_are_checked_per_entry_and_per_record(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        big = _create_closed(manager, stack)
        spill = tmp_path / "sessions" / big / "spill"
        spill.mkdir()
        for index in range(10):
            (spill / f"{index}.txt").write_text("s" * (index + 1))
        expected = _true_tree_bytes(tmp_path / "sessions" / big)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(
            stack, tmp_path, lease, bounds=RetentionBounds(entries_per_root=4), now_ns=_future_ns
        )
        planner.run_pass(days=30)
        walked = json.loads((tmp_path / "retention-state.json").read_text())["entry_count"]
        result, reasons = _run_sweep(planner)

        inbox = tmp_path / "projects" / "p_x" / "inbox" / "new"
        inbox.mkdir(parents=True)
        for index in range(3):
            (inbox / f"{index}.json").write_text(json.dumps({"to_session": "other"}))
        bounded = _planner(
            stack, tmp_path, lease, bounds=RetentionBounds(reference_records_per_root=2),
            now_ns=_future_ns,
        )
        _, bounded_reasons = _run_sweep(bounded)
    assert walked <= 4
    assert reasons[big] == (True, None)
    assert result.preview.bytes == expected
    assert bounded_reasons[big] == (False, "references not fully scanned")


def test_malformed_metadata_is_never_closed(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        ids = sorted(_create_closed(manager, stack) for _ in range(2))
        (tmp_path / "sessions" / ids[0] / "meta.json").write_text("{not json")
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        result, reasons = _run_sweep(planner)
    # The broken root is ineligible, and it may hide a parent link, so the
    # other root's dependent scan cannot pass either.
    assert reasons[ids[0]][0] is False
    assert reasons[ids[1]][0] is False
    assert result.preview.root_ids == ()


def test_below_feature_epoch_writes_nothing(tmp_path):
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":7}\n')
    lease = open_writer(tmp_path)
    try:
        with RetentionPlanner(tmp_path, lease) as planner:
            result = planner.run_pass(days=30)
    finally:
        lease.close()
    assert result.enabled is False
    assert result.complete is False
    assert not (tmp_path / "retention-state.json").exists()


def _future_ns() -> int:
    return time.time_ns() + 60 * _DAY_NS


def _reply(text):
    return Message(MessageRole.ASSISTANT, [TextContent(text)])


def _create_with_message(manager, text):
    opened = manager.create(
        provider="codex", model="gpt-5.6-sol", cwd=str(manager.home), auto_project=False,
    )
    try:
        opened.store.append_message(_reply(text))
    finally:
        opened.store.close()
    return opened.metadata.session_id


def _age_tree(session_dir, *, days=100, keep_metadata=False):
    """Make a session look ``days`` old: metadata timestamps and every mtime."""
    if not keep_metadata:
        meta = session_dir / "meta.json"
        value = json.loads(meta.read_text())
        value["created_at"] = value["updated_at"] = _OLD
        meta.write_text(json.dumps(value))
    old = time.time_ns() - days * _DAY_NS
    for dirpath, _dirs, files in os.walk(session_dir, topdown=False):
        for name in files:
            os.utime(os.path.join(dirpath, name), ns=(old, old), follow_symlinks=False)
        os.utime(dirpath, ns=(old, old))


def _write_sent_ledger(home, session_id, reported):
    """A v1 sent ledger, the shape ``SentMessageTracker`` itself loads."""
    record = {
        "id": "a" * 32, "target_project": "p_x",
        "created_at": "2020-01-01T00:00:00+00:00", "reported": reported,
    }
    path = home / "sessions" / session_id / "project-inbox-sent.json"
    path.write_text(json.dumps({"schema_version": 1, "records": [record]}))
    path.chmod(0o600)


def _run_sweep(planner, days=30):
    """Run passes to the end of one sweep; return the result and every reason."""
    reasons = {}
    for _ in range(500):
        result = planner.run_pass(days=days)
        reasons.update({row.root_id: (row.closed, row.reason) for row in result.reasons})
        if result.complete:
            return result, reasons
    raise AssertionError("sweep did not complete in 500 passes")


def _run_sweep_without_error(planner, days=30):
    try:
        return _run_sweep(planner, days)
    except OSError:
        return None, {}


@contextmanager
def _opened_lease(home):
    lease = open_writer(home)
    try:
        yield lease
    finally:
        lease.close()


def test_isolated_all_roots_behavior(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        roots = [_create_closed(manager, stack) for _ in range(40)]
        for root in roots:
            _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        result, _ = _run_sweep(_planner(stack, tmp_path, lease, now_ns=_future_ns))
    assert result.complete
    assert set(result.preview.root_ids) == set(roots)


def test_isolated_completed_sweeps_do_not_double_count(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        first, _ = _run_sweep(planner)
        first_bytes = first.preview.bytes
        first_id = first.preview.preview_id
        second, _ = _run_sweep(planner)
    assert second.complete
    assert second.preview.preview_id != first_id
    assert second.preview.bytes == first_bytes


def test_isolated_partial_second_sweep_keeps_complete_preview(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(
            stack, tmp_path, lease, bounds=RetentionBounds(entries_per_root=1), now_ns=_future_ns
        )
        complete, _ = _run_sweep(planner)
        preview_id = complete.preview.preview_id
        (tmp_path / "sessions" / root / "new-old-file").write_bytes(b"x" * 100)
        planner.run_pass(days=30)
        state = json.loads((tmp_path / "retention-state.json").read_text())
    assert state["preview"]["preview_id"] == preview_id


@pytest.mark.parametrize(
    ("reported", "expected"),
    [(["claimed", "done"], True), (["claimed"], False), ("done", False)],
)
def test_isolated_sent_ledger_states(tmp_path, reported, expected):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        _write_sent_ledger(tmp_path, root, reported)
        lease = stack.enter_context(_opened_lease(tmp_path))
        _, reasons = _run_sweep(_planner(stack, tmp_path, lease, now_ns=_future_ns))
    assert reasons[root][0] is expected


_MATRIX_KINDS = (
    "reservation", "session_state", "child_lifecycle", "pending_link",
    "pending_child_link", "sent_root", "inbox", "dependent_meta",
    "dependent_fork", "source_attention", "child_sent", "child_inbox",
    "pin", "age", "generation", "discovery", "walk",
    "transcript_recovery", "child_transcript_recovery",
    "memory_reconciliation", "child_memory_reconciliation",
)
_MATRIX_MODES = ("absent", "present", "malformed")
_MATRIX_ROWS = [(kind, mode) for kind in _MATRIX_KINDS for mode in _MATRIX_MODES]


def _valid_inbox(session_id):
    return {
        "schema_version": 1, "id": "b" * 32,
        "from": {"project": "p_x", "session": "c" * 32},
        "to_project": "p_x", "to_session": session_id, "kind": "question",
        "title": "title", "body": "", "in_reply_to": None, "created_at": _OLD,
    }


def _valid_attention(session_id, fork_session_id=None):
    return {
        "id": "a" * 32, "created_at": _OLD, "session_id": session_id,
        "project_id": None, "entry_id": None, "entry_seq": None,
        "lane": "default", "title": "title", "why": "why", "options": [],
        "recommendation": None, "status": "open", "fork_session_id": fork_session_id,
    }


def _valid_pending(session_id=None):
    value = {
        "project_id": "p_x", "role": "session", "transcript_path": "conversation.jsonl",
        "parent_session_id": None, "previous_project_id": None,
    }
    if session_id is not None:
        value["session_id"] = session_id
    return value


def _child_store(root_dir, state="completed"):
    child = root_dir / "agents" / "1"
    child.mkdir(parents=True)
    lifecycle = {
        "handle": "child", "state": state, "started_at": _OLD,
        "finished_at": _OLD if state in {"completed", "canceled", "failed"} else None,
    }
    (child / "agent_lifecycle.json").write_text(json.dumps(lifecycle))
    (child / ".lock").write_text("")
    return child


def _write_inbox(home, payload, *, name=None):
    inbox = home / "projects" / "p_x" / "inbox" / "new"
    inbox.mkdir(parents=True, exist_ok=True)
    message_name = name or f"{payload.get('id', 'record')}.json"
    (inbox / message_name).write_text(json.dumps(payload))


def _setup_reservation(home, root, mode):
    reservations = home / "session-trash" / "reservations"
    reservations.mkdir(parents=True)
    target = root if mode != "absent" else "d" * 32
    value = {
        "schema_version": 1, "session_id": target, "transaction_id": "tx",
        "source_path": f"sessions/{target}", "trash_path": f"session-trash/tx/{target}",
        "device": 1, "inode": 1,
    }
    if mode == "malformed":
        value = {"schema_version": 1, "session_id": root}
    (reservations / f"{target}.json").write_text(json.dumps(value))


def _setup_session_state(root_dir, mode):
    path = root_dir / "session_state.json"
    if mode == "absent":
        return
    if mode == "malformed":
        path.write_text("{}")
        return
    value = json.loads(path.read_text())
    value["agent_children"] = {
        "call": {
            "tool_call": {"id": "call", "name": "agent", "arguments": {}},
            "child_session_path": "/child", "description": "child",
        }
    }
    path.write_text(json.dumps(value))


def _setup_pending(root_dir, root, kind, mode):
    if kind == "pending_link":
        value = _valid_pending() if mode != "malformed" else {"project_id": "p_x"}
        if mode != "absent":
            (root_dir / "project_link_pending.json").write_text(json.dumps(value))
        return
    pending = root_dir / "pending_child_links"
    pending.mkdir()
    if mode == "absent":
        return
    value = _valid_pending(f"{root}:1") if mode != "malformed" else {"session_id": f"{root}:1"}
    (pending / "child.json").write_text(json.dumps(value))


def _setup_dependent(manager, stack, home, root, kind, mode):
    other = _create_closed(manager, stack)
    other_dir = home / "sessions" / other
    _age_tree(other_dir)
    if kind == "dependent_meta":
        meta = json.loads((other_dir / "meta.json").read_text())
        meta["parent_session_id"] = root if mode == "present" else None
        if mode == "malformed":
            meta["parent_session_id"] = 7
        (other_dir / "meta.json").write_text(json.dumps(meta))
    elif kind == "dependent_fork":
        value = {
            "forked_from_session": root if mode == "present" else "e" * 32,
            "forked_at_entry": "entry", "attention_id": "a" * 32,
        }
        if mode == "malformed":
            value = {"forked_from_session": root}
        (other_dir / "attention_fork.json").write_text(json.dumps(value))
    else:
        attention = other_dir / "attention"
        attention.mkdir()
        value = _valid_attention(other, root if mode == "present" else None)
        name = "a" * 32 + ".json"
        if mode == "malformed":
            name = "b" * 32 + ".json"
        (attention / name).write_text(json.dumps(value))


def _write_recovery_state(directory, target, mode):
    quarantine_id = "a" * 32
    if mode == "malformed":
        (directory / f"conversation.recovery.{quarantine_id}.json").write_text("{")
        return
    payload = b"quarantined transcript\n"
    source_digest = hashlib.sha256(payload).hexdigest()
    repaired_digest = "b" * 64
    manifest = EvidenceManifest(
        quarantine_id, target, source_digest, len(payload), 2, 0, 0, 0,
        repaired_digest, _OLD, source_digest, len(payload),
    )
    record = RecoveryRecord(
        quarantine_id, target, "committed", source_digest, repaired_digest, 0,
        _OLD, "review_pending" if mode == "present" else "not_required",
    )
    (directory / f"conversation.quarantine.{quarantine_id}.jsonl").write_bytes(payload)
    (directory / f"conversation.quarantine.{quarantine_id}.json").write_text(
        json.dumps(manifest.to_dict())
    )
    (directory / f"conversation.recovery.{quarantine_id}.json").write_text(
        json.dumps(record.to_dict())
    )


def _write_reconciliation_state(directory, mode):
    cursor = {} if mode in {"absent", "malformed_receipt"} else {
        "recovery_paused": True
    }
    (directory / "memory-reconcile.json").write_text(
        "{" if mode == "malformed" else json.dumps(cursor)
    )
    receipt = {
        "key": "terminal", "seq_start": 1, "seq_end": 1,
        "attempt_count": 3, "validation_summary": "failed", "occurred_at": _OLD,
    }
    (directory / "memory-reconcile-receipts.jsonl").write_text(
        "{\n" if mode == "malformed_receipt" else json.dumps(receipt) + "\n"
    )


def _setup_child_authority(home, root, root_dir, kind, mode):
    state = "running" if kind == "child_lifecycle" and mode == "present" else "completed"
    child = _child_store(root_dir, state)
    if kind == "child_transcript_recovery":
        _write_recovery_state(child, RepairTarget(root, f"{root}:1"), mode)
        return
    if kind == "child_memory_reconciliation":
        _write_reconciliation_state(child, mode)
        return
    if kind == "child_lifecycle":
        if mode == "malformed":
            (child / "agent_lifecycle.json").write_text('{"state":"completed"}')
        return
    if kind == "child_sent":
        _write_sent_ledger(home, root, ["claimed"] if mode == "present" else ["done"])
        ledger = home / "sessions" / root / "project-inbox-sent.json"
        (child / "project-inbox-sent.json").write_text(ledger.read_text())
        ledger.unlink()
        if mode == "malformed":
            (child / "project-inbox-sent.json").write_text(
                '{"records":[{"id":"x","reported":["done"]}]}'
            )
        return
    payload = _valid_inbox(f"{root}:1" if mode == "present" else "f" * 32)
    if mode == "malformed":
        payload = {"title": "x"}
    _write_inbox(home, payload)


def _setup_authority(tmp_path, manager, stack, root, kind, mode):
    root_dir = tmp_path / "sessions" / root
    if kind == "reservation":
        _setup_reservation(tmp_path, root, mode)
    elif kind == "session_state":
        _setup_session_state(root_dir, mode)
    elif kind in {
        "child_lifecycle", "child_sent", "child_inbox",
        "child_transcript_recovery", "child_memory_reconciliation",
    }:
        _setup_child_authority(tmp_path, root, root_dir, kind, mode)
    elif kind in {"pending_link", "pending_child_link"}:
        _setup_pending(root_dir, root, kind, mode)
    elif kind == "transcript_recovery":
        _write_recovery_state(root_dir, RepairTarget(root), mode)
    elif kind == "memory_reconciliation":
        _write_reconciliation_state(root_dir, mode)
    elif kind == "sent_root":
        reported = ["done"] if mode == "absent" else ["claimed"]
        _write_sent_ledger(tmp_path, root, reported)
        if mode == "malformed":
            (root_dir / "project-inbox-sent.json").write_text(
                '{"records":[{"id":"x","reported":["done"]}]}'
            )
    elif kind == "inbox":
        if mode.startswith("non_json_"):
            status = mode.removeprefix("non_json_")
            inbox = tmp_path / "projects" / "p_x" / "inbox" / status
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / "interrupted-record").write_text("partial")
        else:
            payload = _valid_inbox(root if mode == "present" else "f" * 32)
            if mode == "malformed":
                payload = {"title": "x"}
            _write_inbox(tmp_path, payload)
    else:
        _setup_authority_tail(tmp_path, manager, stack, root, kind, mode)


def _setup_authority_tail(tmp_path, manager, stack, root, kind, mode):
    root_dir = tmp_path / "sessions" / root
    if kind in {"dependent_meta", "dependent_fork", "source_attention"}:
        _setup_dependent(manager, stack, tmp_path, root, kind, mode)
    elif kind == "pin" and mode != "absent":
        if mode == "present":
            with session_root(root_dir) as fd:
                write_retention_pin(fd)
        elif mode == "missing_parent":
            meta = json.loads((root_dir / "meta.json").read_text())
            meta["parent_session_id"] = "f" * 32
            (root_dir / "meta.json").write_text(json.dumps(meta))
        else:
            (root_dir / RETENTION_PIN_FILENAME).write_text('{"pinned":false}')
    elif kind == "age":
        if mode == "present":
            recent = _future_ns() - _DAY_NS
            os.utime(root_dir / "meta.json", ns=(recent, recent))
        elif mode == "malformed":
            meta = json.loads((root_dir / "meta.json").read_text())
            meta["updated_at"] = "not-a-time"
            (root_dir / "meta.json").write_text(json.dumps(meta))
    elif kind == "generation":
        (tmp_path / "session-reference-generation").write_text(
            "broken" if mode == "malformed" else "0\n"
        )
    elif kind == "discovery" and mode == "malformed":
        (tmp_path / "sessions" / "unsafe-root").symlink_to(root_dir)
    elif kind == "walk" and mode != "absent":
        if mode == "present":
            (root_dir / "unsafe").symlink_to("meta.json")
        else:
            os.link(root_dir / "meta.json", root_dir / "hard-linked-meta")


def _generation_reader(monkeypatch):
    calls = 0

    def read(_home):
        nonlocal calls
        calls += 1
        return 0 if calls % 2 else 1

    monkeypatch.setattr("zeta.session_retention._helpers.read_reference_generation", read)


def _check_authority_case(tmp_path, monkeypatch, kind, mode):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        root_dir = tmp_path / "sessions" / root
        _age_tree(root_dir)
        _setup_authority(tmp_path, manager, stack, root, kind, mode)
        if kind == "generation" and mode == "present":
            _generation_reader(monkeypatch)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        blocked_evidence = kind == "reservation" or (
            kind == "discovery" and mode == "malformed"
        )
        result, reasons = (
            (planner.run_pass(days=30), {})
            if blocked_evidence else _run_sweep(planner)
        )
    if blocked_evidence:
        assert not result.complete
        assert (tmp_path / "sessions" / root).exists()
        return
    expected_clear = mode == "absent" or (kind == "discovery" and mode == "present")
    assert reasons[root][0] is expected_clear
    assert (root in result.preview.root_ids) is expected_clear

_OVER_BOUND_KINDS = (
    "child_stores", "sent_root", "inbox",
    "dependent_meta", "source_attention", "child_sent", "discovery", "walk",
)


def _many_sent_records(path, count):
    records = [
        {
            "id": f"{index:032x}", "target_project": "p_x", "created_at": _OLD,
            "reported": ["done"],
        }
        for index in range(count)
    ]
    path.write_text(json.dumps({"schema_version": 1, "records": records}))


def _setup_over_bound(home, manager, stack, root, kind, count=24):
    root_dir = home / "sessions" / root
    if kind == "child_stores":
        for index in range(count):
            child = root_dir / "agents" / str(index)
            child.mkdir(parents=True)
            (child / "agent_lifecycle.json").write_text(json.dumps({
                "handle": f"child-{index}", "state": "completed",
                "started_at": _OLD, "finished_at": _OLD,
            }))
            (child / ".lock").write_text("")
    elif kind == "pending_child_link":
        pending = root_dir / "pending_child_links"
        pending.mkdir()
        for index in range(count):
            value = _valid_pending(f"{root}:{index}")
            (pending / f"{index}.json").write_text(json.dumps(value))
    elif kind in {"sent_root", "child_sent"}:
        directory = root_dir if kind == "sent_root" else _child_store(root_dir)
        _many_sent_records(directory / "project-inbox-sent.json", count)
    elif kind == "inbox":
        for index in range(count):
            payload = _valid_inbox("f" * 32)
            payload["id"] = f"{index:032x}"
            _write_inbox(home, payload)
    else:
        _setup_over_bound_tail(home, manager, stack, root, kind, count)


def _setup_over_bound_tail(home, manager, stack, root, kind, count):
    if kind in {"dependent_meta", "discovery"}:
        for _ in range(count):
            other = _create_closed(manager, stack)
            _age_tree(home / "sessions" / other)
    elif kind == "source_attention":
        other = _create_closed(manager, stack)
        _age_tree(home / "sessions" / other)
        attention = home / "sessions" / other / "attention"
        attention.mkdir()
        for index in range(count):
            value = _valid_attention(other)
            value["id"] = f"{index:032x}"
            (attention / f"{index:032x}.json").write_text(json.dumps(value))
    elif kind == "walk":
        spill = home / "sessions" / root / "spill"
        spill.mkdir()
        for index in range(count):
            (spill / str(index)).write_text("x")


def _check_over_bound_case(tmp_path, kind):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        _setup_over_bound(tmp_path, manager, stack, root, kind)
        lease = stack.enter_context(_opened_lease(tmp_path))
        if kind == "discovery":
            bounds = RetentionBounds(entries_total=2)
        elif kind == "walk":
            bounds = RetentionBounds(entries_per_root=2)
        else:
            bounds = RetentionBounds(reference_records_per_root=20)
        planner = _planner(stack, tmp_path, lease, bounds=bounds, now_ns=_future_ns)
        if kind in {"discovery", "walk"}:
            result = planner.run_pass(days=30)
            reasons = {}
        else:
            result, reasons = _run_sweep(planner)
    if kind in {"discovery", "walk"}:
        assert not result.complete
        assert result.preview is None
    else:
        assert reasons[root] == (False, "references not fully scanned")
        assert root not in result.preview.root_ids

_DEADLINE_TARGETS = {
    "reservation": ("_recover_one", None),
    "session_state": ("_has_child_marker", "session_state.json"),
    "child_lifecycle": ("_authority_store_reason", "agent_lifecycle.json"),
    "pending_link": ("_authority_store_reason", "project_link_pending.json"),
    "pending_child_link": ("_pending_child_link", None),
    "sent_root": ("_sent_file_reason", "project-inbox-sent.json"),
    "inbox": ("_authority_inbox_reason", None),
    "dependent_meta": ("_authority_dependents", "meta.json"),
    "dependent_fork": ("_authority_dependents", "attention_fork.json"),
    "source_attention": ("_attention_reference", None),
    "child_sent": ("_sent_file_reason", "project-inbox-sent.json"),
    "child_inbox": ("_authority_inbox_reason", None),
    "pin": ("is_retention_pinned", None),
    "age": ("_authority_store_reason", "meta.json"),
    "generation": ("_generation_value", None),
    "discovery": ("_next_root", None),
    "walk": ("_walk", None),
    "transcript_recovery": ("_names_by_store", None),
    "child_transcript_recovery": ("_names_by_store", None),
    "memory_reconciliation": ("_names_by_store", None),
    "child_memory_reconciliation": ("_names_by_store", None),
}


class _DeadlineClock:
    def __init__(self, target):
        self.function, self.filename = _DEADLINE_TARGETS[target]
        self.hit = False

    def __call__(self):
        frames = inspect.stack()
        functions = {frame.function for frame in frames}
        names = {frame.frame.f_locals.get("name") for frame in frames}
        matched = self.function in functions and (
            self.filename is None or self.filename in names
        )
        if matched:
            self.hit = True
            return _future_ns() + 60_000_000_000
        return _future_ns()


def _check_deadline_case(tmp_path, kind):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        mode = "present" if kind in {
            "reservation", "pending_link", "pending_child_link"
        } else "absent"
        _setup_authority(tmp_path, manager, stack, root, kind, mode)
        clock = _DeadlineClock(kind)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=clock)
        result = planner.run_pass(days=30)
        planner.close()
    assert clock.hit, f"deadline did not cross inside {kind}"
    assert not result.complete
    assert result.preview is None
    assert not any(row.root_id == root and row.closed for row in result.reasons)

_RUN_PASS_ROWS = [
    *_MATRIX_ROWS,
    ("pin", "missing_parent"),
    ("inbox", "non_json_new"),
    ("inbox", "non_json_claimed"),
    ("memory_reconciliation", "malformed_receipt"),
    ("child_memory_reconciliation", "malformed_receipt"),
    *((kind, "deadline") for kind in _MATRIX_KINDS),
    *((kind, "over_bound") for kind in _OVER_BOUND_KINDS),
]


@pytest.mark.parametrize(("kind", "case"), _RUN_PASS_ROWS)
def test_closure_authority_run_pass_matrix(tmp_path, monkeypatch, kind, case):
    if case == "deadline":
        _check_deadline_case(tmp_path, kind)
    elif case == "over_bound":
        _check_over_bound_case(tmp_path, kind)
    else:
        _check_authority_case(tmp_path, monkeypatch, kind, case)


def _charge_fault_fixture(home):
    now = 2_000_000_000_000_000_000
    manager = _manager(home)
    with ExitStack() as stack:
        roots = [_create_closed(manager, stack) for _ in range(3)]
    recent = home / "sessions" / roots[1] / "deep" / "subtree" / "recent"
    recent.parent.mkdir(parents=True)
    recent.write_text("recent")
    for root in roots:
        _age_tree(home / "sessions" / root)
    os.utime(recent, ns=(now, now))
    root_dir = home / "sessions" / roots[0]
    _write_sent_ledger(home, roots[0], ["done"])
    _write_recovery_state(root_dir, RepairTarget(roots[0]), "absent")
    _write_reconciliation_state(root_dir, "absent")
    _write_inbox(home, _valid_inbox("f" * 32))
    manager._storage_lease.close()
    return roots


def _run_charge_fault_sweep(home, roots, monkeypatch, fail_at=None):
    now = 2_000_000_000_000_000_000
    with ExitStack() as stack:
        budget_charge = session_retention._Budget.charge_entry
        closure_entry = session_retention._ClosureBudget.charge_entry
        closure_charge = session_retention._ClosureBudget.charge
        charges = 0
        charging_entry = False

        def fault():
            nonlocal charges
            charges += 1
            if charges == fail_at:
                raise session_retention._Deferred

        def charged_budget_entry(self, size=0):
            if not charging_entry:
                fault()
            return budget_charge(self, size)

        def charged_closure_entry(self, size):
            nonlocal charging_entry
            fault()
            charging_entry = True
            try:
                return closure_entry(self, size)
            finally:
                charging_entry = False

        def charged_record(self, size):
            if not charging_entry:
                fault()
            return closure_charge(self, size)

        monkeypatch.setattr(session_retention._Budget, "charge_entry", charged_budget_entry)
        monkeypatch.setattr(session_retention._ClosureBudget, "charge_entry", charged_closure_entry)
        monkeypatch.setattr(session_retention._ClosureBudget, "charge", charged_record)
        monkeypatch.setattr(
            session_retention.uuid, "uuid4", lambda: type("FixedUUID", (), {"hex": "preview"})()
        )
        lease = stack.enter_context(_opened_lease(home))
        planner = _planner(stack, home, lease, now_ns=lambda: now)
        reasons = {}
        passes = 0
        while passes < 20:
            passes += 1
            result = planner.run_pass(days=30)
            reasons.update({
                row.root_id: (row.closed, row.reason, row.bytes) for row in result.reasons
            })
            if result.complete:
                break
        else:
            raise AssertionError("faulted sweep did not complete in 20 passes")

    assert set(reasons) == set(roots), "completed sweep omitted candidate roots"
    classification = tuple(reasons[root] for root in roots)
    selected = tuple(index for index, root in enumerate(roots) if root in result.preview.root_ids)
    preview = (
        result.preview.days, result.preview.cutoff, selected,
        result.preview.count, result.preview.bytes, result.preview.created_at,
    )
    return charges, passes, classification, preview


def test_every_charge_deferral_preserves_final_sweep(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    roots = _charge_fault_fixture(fixture)
    baseline = tmp_path / "case-000"
    shutil.copytree(fixture, baseline)
    with monkeypatch.context() as patch:
        total, _, expected_classification, expected_preview = _run_charge_fault_sweep(
            baseline, roots, patch
        )
    assert 10 <= total < 250, f"focused fixture made {total} charges"

    for charge in range(1, total + 1):
        case = tmp_path / f"case-{charge:03}"
        shutil.copytree(fixture, case)
        with monkeypatch.context() as patch:
            actual, passes, classification, preview = _run_charge_fault_sweep(
                case, roots, patch, fail_at=charge
            )
        assert passes > 1, (
            f"charge {charge} did not cross a pass boundary (run made {actual} charges)"
        )
        assert classification == expected_classification, f"charge {charge} changed root decisions"
        assert preview == expected_preview, f"charge {charge} changed the completed preview"


def test_armed_completed_preview_quarantines_eligible_root(tmp_path):
    """An arm for the last complete preview makes the next pass quarantine it."""
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        preview, reasons = _run_sweep(planner)
        assert reasons[root] == (True, None)
        arm_path = tmp_path / "retention-arm.json"
        arm_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_version": 1,
                    "armed_by_zeta_version": "test",
                    "armed_at": _OLD,
                    "preview_id": preview.preview.preview_id,
                }
            )
            + "\n"
        )
        arm_path.chmod(0o600)
        planner.close()
        resumed = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        resumed.run_pass(days=30)

    assert not (tmp_path / "sessions" / root).exists(), (
        "an armed root from the completed preview remained published"
    )
    reservation = tmp_path / "session-trash" / "reservations" / f"{root}.json"
    assert reservation.exists(), "quarantine did not publish its durable reservation"


def test_quarantine_restart_resumes_to_durable_removal(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        first = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        preview, _ = _run_sweep(first)
        arm = tmp_path / "retention-arm.json"
        arm.write_text(json.dumps({
            "schema_version": 1, "gate_version": 1,
            "armed_by_zeta_version": "test", "armed_at": _OLD,
            "preview_id": preview.preview.preview_id,
        }) + "\n")
        arm.chmod(0o600)
        first.close()
        for _ in range(100):
            resumed = RetentionPlanner(tmp_path, lease, now_ns=_future_ns)
            try:
                resumed.run_pass(days=30)
            finally:
                resumed.close()
            if not (tmp_path / "session-trash" / "reservations" / f"{root}.json").exists():
                break
        else:
            raise AssertionError("quarantine did not finish within 100 restart steps")

    assert not (tmp_path / "sessions" / root).exists()
    assert not list((tmp_path / "session-trash" / "transactions").iterdir())


def _write_test_arm(home, preview_id):
    arm = home / "retention-arm.json"
    arm.write_text(json.dumps({
        "schema_version": 1, "gate_version": 1,
        "armed_by_zeta_version": "test", "armed_at": _OLD,
        "preview_id": preview_id,
    }) + "\n")
    arm.chmod(0o600)
    return arm


def _drive_until_one_root_remains(planner, home, roots):
    for _ in range(200):
        planner.run_pass(days=30)
        remaining = [root for root in roots if (home / "sessions" / root).exists()]
        reservations = list((home / "session-trash" / "reservations").glob("*.json"))
        if len(remaining) == 1 and not reservations:
            return
    raise AssertionError("first armed transaction did not finish in 200 public passes")


def _drive_until_roots_and_reservations_are_gone(planner, home, roots):
    for _ in range(200):
        planner.run_pass(days=30)
        if (
            not any((home / "sessions" / root).exists() for root in roots)
            and not list((home / "session-trash" / "reservations").glob("*.json"))
        ):
            return
    raise AssertionError("armed transactions did not finish in 200 public passes")


def test_one_arm_authorizes_fresh_proofs_across_transactions_and_restart(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        roots = sorted(_create_closed(manager, stack) for _ in range(2))
        for root in roots:
            _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        preview, reasons = _run_sweep(planner)
        assert all(reasons[root] == (True, None) for root in roots)
        arm = _write_test_arm(tmp_path, preview.preview.preview_id)
        armed_bytes = arm.read_bytes()

        _drive_until_one_root_remains(planner, tmp_path, roots)
        planner.close()
        resumed = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        _drive_until_roots_and_reservations_are_gone(resumed, tmp_path, roots)

    assert arm.read_bytes() == armed_bytes, "a later transaction rewrote the durable arm"


def _drive_to_manifest_stage(planner, home, stage):
    for _ in range(200):
        planner.run_pass(days=30)
        manifests = list(
            (home / "session-trash" / "transactions").glob("*/manifest.json")
        )
        if manifests and json.loads(manifests[0].read_text())["stage"] == stage:
            return
    raise AssertionError(f"transaction did not reach {stage}")


def _finish_with_tree_before_evidence(planner, home, root):
    tree_removed_with_evidence = False
    for _ in range(200):
        planner.run_pass(days=30)
        manifests = list(
            (home / "session-trash" / "transactions").glob("*/manifest.json")
        )
        trees = list((home / "session-trash" / "transactions").glob("*/tree"))
        reservation = home / "session-trash" / "reservations" / f"{root}.json"
        if manifests and json.loads(manifests[0].read_text())["stage"] == "removed":
            assert not trees, "transaction 2 advanced before removing its owned tree"
        if not trees and manifests and reservation.exists():
            tree_removed_with_evidence = True
        if not reservation.exists():
            return tree_removed_with_evidence
    raise AssertionError("second same-id transaction did not finish")


def test_same_id_transaction_rejects_stale_removal_frontier(tmp_path):
    from zeta.remote_sync import LocalTransport, push_session

    manager = _manager(tmp_path)
    import_home = tmp_path.parent / f"{tmp_path.name}-same-id"
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        shutil.copytree(tmp_path, import_home)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        preview, _ = _run_sweep(planner)
        _write_test_arm(tmp_path, preview.preview.preview_id)
        _drive_until_roots_and_reservations_are_gone(
            planner, tmp_path, (root,)
        )
        stale_frontier = (tmp_path / "retention-state.json").read_bytes()

        push_session(import_home, LocalTransport(tmp_path), session_id=root, force=True)
        _age_tree(tmp_path / "sessions" / root)
        planner.close()
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        preview, reasons = _run_sweep(planner)
        assert reasons[root] == (True, None)
        assert preview.preview.root_ids == (root,)
        _write_test_arm(tmp_path, preview.preview.preview_id)
        _drive_to_manifest_stage(planner, tmp_path, "removing")
        (tmp_path / "retention-state.json").write_bytes(stale_frontier)
        tree_removed_with_evidence = _finish_with_tree_before_evidence(
            planner, tmp_path, root
        )

    assert tree_removed_with_evidence, (
        "transaction 2 removed its manifest or reservation before its tree"
    )


def _retention_manifest(home: Path) -> tuple[Path, dict]:
    manifests = list((home / "session-trash" / "transactions").glob("*/manifest.json"))
    assert len(manifests) == 1
    return manifests[0], json.loads(manifests[0].read_text())


def _arm_and_quarantine(home, stack, root, lease):
    _age_tree(home / "sessions" / root)
    planner = _planner(stack, home, lease, now_ns=_future_ns)
    preview, reasons = _run_sweep(planner)
    assert reasons[root] == (True, None)
    arm = home / "retention-arm.json"
    arm.write_text(json.dumps({
        "schema_version": 1,
        "gate_version": 1,
        "armed_by_zeta_version": "test",
        "armed_at": _OLD,
        "preview_id": preview.preview.preview_id,
    }) + "\n")
    arm.chmod(0o600)
    planner.close()
    resumed = _planner(stack, home, lease, now_ns=_future_ns)
    resumed.run_pass(days=30)
    assert _retention_manifest(home)[1]["stage"] == "quarantined"
    return resumed


def _registry(stack, home):
    registry = ProjectRegistry(home / "projects")
    release = getattr(registry, "_release_storage_lease", None)
    if release is not None:
        stack.callback(release)
    return registry


def _external_reference(stage, home, manager, stack, root, lease):
    registry = _registry(stack, home)
    project = registry.create_project(f"retention-{stage}", "test")
    if stage in {"projects", "indexes"}:
        registry.record_session(
            project.project_id,
            session_id=root,
            transcript_path=str(home / "sessions" / root),
        )
        if stage == "indexes":
            source_dir = home / "indexed-source"
            source_dir.mkdir()
            rows = [
                {
                    "seq": 1,
                    "id": "user",
                    "parent_id": None,
                    "lane": "main",
                    "type": "message",
                    "data": {"message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "prompt"}],
                        "metadata": {"zeta.origin": "user", "origin": "human"},
                    }},
                },
                {
                    "seq": 2,
                    "id": "assistant",
                    "parent_id": "user",
                    "lane": "main",
                    "type": "message",
                    "data": {"message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "indexed"}],
                        "metadata": {"response_state": "complete"},
                    }},
                },
            ]
            (source_dir / "conversation.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            index = TranscriptIndex(registry.root / project.project_id, project.project_id)
            status = index.append(TranscriptSource(root, source_dir))
            assert status.session_count == 1
            return registry, project, index
        return registry, project, None
    if stage == "inbox":
        sender = _create_closed(manager, stack)
        tracker = SentMessageTracker(
            home / "sessions", sender, observed_epoch=lease.observed_epoch
        )
        inbox = ProjectInbox(
            registry, sessions_root=home / "sessions", sent_tracker=tracker
        )
        message_id = inbox.send_user(
            from_project=project.project_id,
            from_session=sender,
            to_project=project.project_id,
            to_session=root,
            kind="question",
            title="terminal reference",
            body="done",
        )
        assert inbox.claim(project.project_id, message_id, root) is not None
        inbox.done(project.project_id, message_id, root, "done")
        tracker.mark_reported(((message_id, "done"),))
        return inbox, tracker, message_id
    child = home / "sessions" / root / "agents" / "1"
    child.mkdir(parents=True)
    (child / "agent_lifecycle.json").write_text(json.dumps({
        "handle": "child",
        "state": "completed",
        "started_at": _OLD,
        "finished_at": _OLD,
    }))
    (child / "conversation.jsonl").write_text("")
    (child / ".lock").write_text("")
    source = _create_closed(manager, stack)
    with session_root(home / "sessions" / source) as source_fd:
        write_retention_pin(source_fd)
    store = AttentionStore(
        home / "sessions" / source, writer_epoch=lease.observed_epoch
    )
    record = store.request(
        session_id=source,
        project_id=project.project_id,
        entry_id=None,
        entry_seq=None,
        title="retention fork",
        why="test external fork link",
    )
    store.replace(
        replace(
            record,
            status="resolved",
            resolved_at=_OLD,
            fork_session_id=f"{root}:1",
            decision="done",
        )
    )
    return store, record.id, None


@pytest.mark.parametrize(
    ("stage", "start_stage", "advance_count"),
    [
        ("projects", "quarantined", 0),
        ("inbox", "projects", 1),
        ("attention", "inbox", 2),
        ("indexes", "attention", 3),
    ],
)
def test_quarantine_cleanup_uses_durable_owners(
    tmp_path, stage, start_stage, advance_count
):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        lease = stack.enter_context(_opened_lease(tmp_path))
        owner = _external_reference(stage, tmp_path, manager, stack, root, lease)
        planner = _arm_and_quarantine(tmp_path, stack, root, lease)
        prior = None
        for _ in range(advance_count):
            prior = planner.run_pass(days=30)
        assert _retention_manifest(tmp_path)[1]["stage"] == start_stage, prior
        planner.run_pass(days=30)

        if stage == "projects":
            registry, project, _ = owner
            assert all(
                link["session_id"] != root
                for link in registry.list_session_links(project.project_id)
            ), "project cleanup left the quarantined session linked"
        elif stage == "inbox":
            _inbox, tracker, message_id = owner
            assert all(
                row["id"] != message_id for row in tracker.records()
            ), "inbox cleanup left the external sender tracker"
        elif stage == "attention":
            store, attention_id, _ = owner
            assert store.get(attention_id).fork_session_id is None, (
                "attention cleanup left the quarantined fork linked"
            )
        else:
            _registry_owner, _project, index = owner
            assert index.status().session_count == 0, (
                "index cleanup left the quarantined session indexed"
            )


@pytest.mark.parametrize(
    ("stage", "advance_count"),
    [
        ("projects", 0),
        ("inbox", 1),
        ("attention", 2),
        ("indexes", 3),
    ],
)
def test_quarantine_cleanup_malformed_owner_record_does_not_advance(
    tmp_path, stage, advance_count
):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        lease = stack.enter_context(_opened_lease(tmp_path))
        owner = _external_reference(stage, tmp_path, manager, stack, root, lease)
        planner = _arm_and_quarantine(tmp_path, stack, root, lease)
        for _ in range(advance_count):
            planner.run_pass(days=30)
        manifest_path, before = _retention_manifest(tmp_path)
        if stage == "projects":
            registry, project, _ = owner
            malformed = registry.root / project.project_id / "sessions.jsonl"
        elif stage == "inbox":
            inbox, _tracker, message_id = owner
            malformed = (
                inbox.registry.root
                / inbox.registry.list_projects()[0].project_id
                / "inbox"
                / "done"
                / f"{message_id}.json"
            )
        elif stage == "attention":
            store, attention_id, _ = owner
            malformed = store.directory / f"{attention_id}.json"
        else:
            _registry_owner, _project, index = owner
            malformed = index.path
        malformed.write_text("not a durable owner record\n")
        result = planner.run_pass(days=30)

    assert json.loads(manifest_path.read_text())["stage"] == before["stage"], (
        "cleanup advanced after its durable owner refused a malformed record"
    )
    assert result.reasons[0].reason is not None


def test_quarantine_cleanup_stops_when_id_is_republished(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _arm_and_quarantine(tmp_path, stack, root, lease)
        (tmp_path / "sessions" / root).mkdir()
        result = planner.run_pass(days=30)

    assert result.reasons[0].reason == "id_republished"
    assert _retention_manifest(tmp_path)[1]["stage"] == "id_republished"


class _SimulatedProcessDeath(BaseException):
    pass


def _arm_fault_fixture(home):
    details = {}
    with ExitStack() as stack:
        manager = _manager(home)
        stack.callback(manager._release_storage_lease)
        root = _create_closed(manager, stack)
        root_dir = home / "sessions" / root
        (root_dir / "nested" / "leaf").mkdir(parents=True)
        (root_dir / "nested" / "one.txt").write_text("one")
        (root_dir / "nested" / "leaf" / "two.txt").write_text("two")
        lease = stack.enter_context(_opened_lease(home))
        _registry_owner, project, _ = _external_reference(
            "projects", home, manager, stack, root, lease
        )
        inbox, tracker, message_id = _external_reference(
            "inbox", home, manager, stack, root, lease
        )
        inbox_project_id = next(
            project.project_id
            for project in inbox.registry.list_projects()
            if any(
                row["id"] == message_id
                for row in inbox.list(project.project_id)["done"]
            )
        )
        attention, attention_id, _ = _external_reference(
            "attention", home, manager, stack, root, lease
        )
        _index_registry, index_project, _index = _external_reference(
            "indexes", home, manager, stack, root, lease
        )
        details.update(
            root=root,
            project_id=project.project_id,
            inbox_project_id=inbox_project_id,
            sender_id=tracker.session_id,
            message_id=message_id,
            attention_session=attention.session_dir.name,
            attention_id=attention_id,
            index_project_id=index_project.project_id,
        )
        for session_dir in (home / "sessions").iterdir():
            if session_dir.name != root:
                with session_root(session_dir) as directory_fd:
                    write_retention_pin(directory_fd)
        _age_tree(root_dir)
        planner = _planner(stack, home, lease, now_ns=_future_ns)
        preview, reasons = _run_sweep(planner)
        assert reasons[root] == (True, None)
        arm = home / "retention-arm.json"
        arm.write_text(json.dumps({
            "schema_version": 1,
            "gate_version": 1,
            "armed_by_zeta_version": "test",
            "armed_at": _OLD,
            "preview_id": preview.preview.preview_id,
        }) + "\n")
        arm.chmod(0o600)
    return details


def _external_reference_state(home, details):
    with ExitStack() as stack:
        registry = _registry(stack, home)
        project_link = any(
            row["session_id"] == details["root"]
            for row in registry.list_session_links(details["project_id"])
        )
        inbox = ProjectInbox(registry, sessions_root=home / "sessions")
        inbox_link = any(
            row["id"] == details["message_id"]
            for row in inbox.list(details["inbox_project_id"])["done"]
        )
        tracker = SentMessageTracker(
            home / "sessions",
            details["sender_id"],
            observed_epoch=registry._storage_lease.observed_epoch,
        )
        sent_link = any(
            row["id"] == details["message_id"] for row in tracker.records()
        )
        attention = AttentionStore(
            home / "sessions" / details["attention_session"],
            writer_epoch=registry._storage_lease.observed_epoch,
        ).get(details["attention_id"])
        index = TranscriptIndex(
            registry.root / details["index_project_id"],
            details["index_project_id"],
        )
        return (
            project_link,
            inbox_link,
            sent_link,
            attention.fork_session_id is not None,
            index.status().session_count != 0,
        )


def _manifest_stage(home):
    manifests = list((home / "session-trash" / "transactions").glob("*/manifest.json"))
    if not manifests:
        return None
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text())["stage"]


def _drive_fault_transaction(home, root):
    crashed = False
    with _opened_lease(home) as lease:
        for _ in range(200):
            planner = RetentionPlanner(home, lease, now_ns=_future_ns)
            try:
                planner.run_pass(days=30)
            except BaseException as exc:
                if not isinstance(
                    exc,
                    (_SimulatedProcessDeath, __import__("asyncio").CancelledError),
                ):
                    raise
                crashed = True
            finally:
                planner.close()
            if crashed or (
                not (home / "session-trash" / "reservations" / f"{root}.json").exists()
                and not (home / "sessions" / root).exists()
            ):
                return crashed
    raise AssertionError("transaction did not finish in 200 public passes")


def _run_fault_transaction(home, details, monkeypatch, *, fail_at=None, cancel=False):
    from zeta.session_retention import transaction
    from zeta.storage_format import WriterLease

    steps = []
    active = False
    failure_stage = None
    real_advance = transaction._QuarantineTransaction.advance
    real_publish = transaction._QuarantineTransaction._publish
    real_os = transaction.os
    real_delete = WriterLease.delete

    def step(name):
        nonlocal failure_stage
        if not active:
            return
        steps.append(name)
        if fail_at == len(steps):
            failure_stage = _manifest_stage(home)
            if cancel:
                import asyncio
                raise asyncio.CancelledError
            raise _SimulatedProcessDeath

    def advance(owner, **kwargs):
        nonlocal active
        active = True
        try:
            return real_advance(owner, **kwargs)
        finally:
            active = False

    def publish(owner, directory_fd, name, payload, validate):
        step(f"publish:{name}")
        return real_publish(owner, directory_fd, name, payload, validate)

    class TransactionOs:
        def __getattr__(self, name):
            return getattr(real_os, name)

        def fsync(self, fd):
            step("fsync")
            return real_os.fsync(fd)

        def rename(self, *args, **kwargs):
            step("rename")
            return real_os.rename(*args, **kwargs)

        def unlink(self, *args, **kwargs):
            step("unlink")
            return real_os.unlink(*args, **kwargs)

        def rmdir(self, *args, **kwargs):
            step("rmdir")
            return real_os.rmdir(*args, **kwargs)

    def delete(lease, directory_fd, name):
        step(f"delete:{name}")
        return real_delete(lease, directory_fd, name)

    monkeypatch.setattr(transaction._QuarantineTransaction, "advance", advance)
    # test-contract: seam observes every durable publication without product hooks
    monkeypatch.setattr(transaction._QuarantineTransaction, "_publish", publish)
    monkeypatch.setattr(transaction, "os", TransactionOs())
    monkeypatch.setattr(WriterLease, "delete", delete)

    crashed = _drive_fault_transaction(home, details["root"])
    return steps, crashed, failure_stage


def _finish_fault_recovery(home, details):
    stages = []
    with _opened_lease(home) as lease:
        # Two independent owners replay the same durable input before ordinary
        # recovery continues, which covers restart and duplicate delivery.
        for _ in range(2):
            planner = RetentionPlanner(home, lease, now_ns=_future_ns)
            try:
                planner.run_pass(days=30)
                stages.append(_manifest_stage(home))
            finally:
                planner.close()
        for _ in range(200):
            reservation = (
                home / "session-trash" / "reservations" / f"{details['root']}.json"
            )
            if (
                not reservation.exists()
                and not (home / "sessions" / details["root"]).exists()
            ):
                return stages
            planner = RetentionPlanner(home, lease, now_ns=_future_ns)
            try:
                planner.run_pass(days=30)
                stages.append(_manifest_stage(home))
            finally:
                planner.close()
    raise AssertionError("recovery did not finish in 200 public passes")


def test_quarantine_every_durable_fault_recovers_monotonically(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    details = _arm_fault_fixture(fixture)
    assert _external_reference_state(fixture, details) == (True,) * 5

    uninterrupted = tmp_path / "uninterrupted"
    shutil.copytree(fixture, uninterrupted)
    with monkeypatch.context() as patch:
        steps, crashed, _ = _run_fault_transaction(
            uninterrupted, details, patch
        )
    assert not crashed
    assert 10 <= len(steps) <= 120, f"fixture made {len(steps)} durable steps"
    rename_step = steps.index("rename")
    assert steps[rename_step + 1:rename_step + 3] == ["fsync", "fsync"], (
        "quarantine rename did not fsync both parent directories"
    )
    assert _external_reference_state(uninterrupted, details) == (False,) * 5

    for fail_at in range(1, len(steps) + 1):
        home = tmp_path / f"fault-{fail_at:03}"
        shutil.copytree(fixture, home)
        with monkeypatch.context() as patch:
            actual, crashed, failure_stage = _run_fault_transaction(
                home, details, patch, fail_at=fail_at
            )
        assert crashed and len(actual) == fail_at, f"durable fault {fail_at} was not reached"
        assert _manifest_stage(home) == failure_stage, (
            f"durable fault {fail_at} advanced its stage after failure"
        )
        root_exists = (home / "sessions" / details["root"]).exists()
        references = _external_reference_state(home, details)
        if root_exists:
            assert references == (True,) * 5, (
                f"durable fault {fail_at} cleaned a still-published root"
            )
        reservation = (
            home / "session-trash" / "reservations" / f"{details['root']}.json"
        )
        if reservation.exists():
            with ExitStack() as stack:
                registry = _registry(stack, home)
                project = registry.create_project(f"blocked-{fail_at}", "test")
                with pytest.raises(SessionError, match="reserved|quarantined|being deleted"):
                    registry.record_session(
                        project.project_id,
                        session_id=details["root"],
                        transcript_path=str(home / "sessions" / details["root"]),
                    )
        recovered_stages = _finish_fault_recovery(home, details)
        ranks = {
            None: -1,
            "quarantined": 0,
            "projects": 1,
            "inbox": 2,
            "attention": 3,
            "indexes": 4,
            "references-clean": 5,
            "removing": 6,
            "removed": 7,
        }
        visible = [
            ranks[stage]
            for stage in recovered_stages
            if stage is not None and stage in ranks
        ]
        assert visible == sorted(visible), f"durable fault {fail_at} replayed backwards"
        assert not (home / "sessions" / details["root"]).exists()
        assert _external_reference_state(home, details) == (False,) * 5, (
            f"durable fault {fail_at} left an external reference"
        )

    cancelled = tmp_path / "cancelled"
    shutil.copytree(fixture, cancelled)
    cancel_at = steps.index("rename") + 1
    with monkeypatch.context() as patch:
        _actual, crashed, failure_stage = _run_fault_transaction(
            cancelled, details, patch, fail_at=cancel_at, cancel=True
        )
    assert crashed
    assert _manifest_stage(cancelled) == failure_stage
    _finish_fault_recovery(cancelled, details)
    assert _external_reference_state(cancelled, details) == (False,) * 5


@pytest.mark.parametrize(
    "case",
    [
        "malformed_reservation",
        "mismatched_manifest",
        "recovery_bound",
        "recovery_deadline",
        "removal_bound",
        "removal_deadline",
    ],
)
def test_quarantine_recovery_refusal_table(tmp_path, case):
    details = _arm_fault_fixture(tmp_path)
    root = details["root"]
    reservation = tmp_path / "session-trash" / "reservations" / f"{root}.json"
    with ExitStack() as stack:
        lease = stack.enter_context(_opened_lease(tmp_path))
        if case == "malformed_reservation":
            reservation.parent.mkdir(parents=True)
            reservation.write_text("{}\n")
            reservation.chmod(0o600)
            before = None
        else:
            planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
            planner.run_pass(days=30)
            before = _manifest_stage(tmp_path)
            assert before == "quarantined"
            if case == "mismatched_manifest":
                manifest_path, manifest = _retention_manifest(tmp_path)
                manifest["transaction_id"] = "f" * 32
                manifest_path.write_text(json.dumps(manifest) + "\n")
                manifest_path.chmod(0o600)
            elif case.startswith("removal"):
                for _ in range(6):
                    planner.run_pass(days=30)
                before = _manifest_stage(tmp_path)
                assert before == "removing"
            planner.close()

        if case.endswith("bound"):
            bounds = RetentionBounds(entries_total=1)
            clock = _future_ns
        elif case.endswith("deadline"):
            bounds = RetentionBounds()
            clock = _DeadlineClock("reservation")
        else:
            bounds = RetentionBounds()
            clock = _future_ns
        refused = _planner(
            stack, tmp_path, lease, bounds=bounds, now_ns=clock
        ).run_pass(days=30)

    assert refused.reasons and refused.reasons[0].reason is not None
    assert _manifest_stage(tmp_path) == before, f"{case} advanced after refusing evidence"
    if case == "malformed_reservation":
        assert (tmp_path / "sessions" / root).is_dir()
        assert _external_reference_state(tmp_path, details) == (True,) * 5


def test_quarantine_reservation_blocks_open_import_reference_and_pin(tmp_path):
    from zeta.remote_sync import LocalTransport, push_session

    details = _arm_fault_fixture(tmp_path)
    root = details["root"]
    import_home = tmp_path.parent / f"{tmp_path.name}-import"
    shutil.copytree(tmp_path, import_home)
    shutil.rmtree(import_home / "sessions" / root / "agents")
    with ExitStack() as stack:
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = _planner(stack, tmp_path, lease, now_ns=_future_ns)
        planner.run_pass(days=30)
        assert (tmp_path / "session-trash" / "reservations" / f"{root}.json").is_file()

        manager = _manager(tmp_path)
        stack.callback(manager._release_storage_lease)
        with pytest.raises((FileNotFoundError, SessionError)):
            manager.open(root)
        registry = _registry(stack, tmp_path)
        project = registry.create_project("blocked-reference", "test")
        with pytest.raises(SessionError, match="reserved|quarantined|being deleted"):
            registry.record_session(
                project.project_id,
                session_id=root,
                transcript_path=str(tmp_path / "sessions" / root),
            )
        with pytest.raises(SessionError, match="reserved|quarantined|being deleted"):
            push_session(
                import_home,
                LocalTransport(tmp_path),
                session_id=root,
                force=True,
            )
        with (
            pytest.raises(FileNotFoundError),
            session_root(tmp_path / "sessions" / root) as directory_fd,
        ):
            write_retention_pin(directory_fd)


def test_pin_committed_before_final_authority_blocks_quarantine(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from zeta.core import session_files

    details = _arm_fault_fixture(tmp_path)
    root = details["root"]
    requested = Event()
    release = Event()
    real_exclusive = session_files.SessionReferenceGate.exclusive

    @contextmanager
    def paused_exclusive(gate):
        requested.set()
        assert release.wait(10), "test did not release retention authority"
        with real_exclusive(gate) as authority:
            yield authority

    monkeypatch.setattr(
        session_files.SessionReferenceGate, "exclusive", paused_exclusive
    )

    def run_pass():
        with (
            _opened_lease(tmp_path) as lease,
            RetentionPlanner(tmp_path, lease, now_ns=_future_ns) as planner,
        ):
            return planner.run_pass(days=30)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_pass)
        assert requested.wait(10), "retention never requested exclusive authority"
        with session_root(tmp_path / "sessions" / root) as directory_fd:
            write_retention_pin(directory_fd)
        manager = _manager(tmp_path)
        try:
            opened = manager.open(root)
            opened.store.close()
        finally:
            manager._release_storage_lease()
        release.set()
        result = future.result(timeout=10)

    assert (tmp_path / "sessions" / root).is_dir()
    assert not (tmp_path / "session-trash" / "reservations" / f"{root}.json").exists()
    assert result.reasons[0].reason == "pinned"
