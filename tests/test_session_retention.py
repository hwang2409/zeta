from __future__ import annotations

import json
import os
import stat
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

from zeta.core.session import SessionManager, SessionMetadata
from zeta.core.session_files import SessionError, session_root
from zeta.protocol.types import Message, MessageRole, TextContent
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
        planner = RetentionPlanner(tmp_path, lease, now_ns=time.time_ns)
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
        planner = RetentionPlanner(tmp_path, lease, now_ns=_future_ns)
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
        planner = RetentionPlanner(tmp_path, lease, bounds=RetentionBounds(entries_per_root=1))
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
        planner = RetentionPlanner(tmp_path, lease, now_ns=_future_ns)
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
        planner = RetentionPlanner(tmp_path, lease, now_ns=_future_ns)
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
        result, _ = _run_sweep(RetentionPlanner(tmp_path, lease, now_ns=_future_ns))
    assert result.complete
    assert set(result.preview.root_ids) == set(roots)


def test_isolated_completed_sweeps_do_not_double_count(tmp_path):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        _age_tree(tmp_path / "sessions" / root)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = RetentionPlanner(tmp_path, lease, now_ns=_future_ns)
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
        planner = RetentionPlanner(
            tmp_path, lease, bounds=RetentionBounds(entries_per_root=1), now_ns=_future_ns
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
        _, reasons = _run_sweep(RetentionPlanner(tmp_path, lease, now_ns=_future_ns))
    assert reasons[root][0] is expected


_MATRIX_KINDS = (
    "reservation", "session_state", "child_lifecycle", "pending_link",
    "pending_child_link", "sent_root", "inbox", "dependent_meta",
    "dependent_fork", "source_attention", "child_sent", "child_inbox",
    "pin", "age", "generation", "discovery", "walk",
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


def _setup_child_authority(home, root, root_dir, kind, mode):
    state = "running" if kind == "child_lifecycle" and mode == "present" else "completed"
    child = _child_store(root_dir, state)
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
    elif kind in {"child_lifecycle", "child_sent", "child_inbox"}:
        _setup_child_authority(tmp_path, root, root_dir, kind, mode)
    elif kind in {"pending_link", "pending_child_link"}:
        _setup_pending(root_dir, root, kind, mode)
    elif kind == "sent_root":
        reported = ["done"] if mode == "absent" else ["claimed"]
        _write_sent_ledger(tmp_path, root, reported)
        if mode == "malformed":
            (root_dir / "project-inbox-sent.json").write_text(
                '{"records":[{"id":"x","reported":["done"]}]}'
            )
    elif kind == "inbox":
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

    monkeypatch.setattr("zeta.session_retention.read_reference_generation", read)


@pytest.mark.parametrize(("kind", "mode"), _MATRIX_ROWS)
def test_closure_authority_matrix(tmp_path, monkeypatch, kind, mode):
    manager = _manager(tmp_path)
    with ExitStack() as stack:
        root = _create_closed(manager, stack)
        root_dir = tmp_path / "sessions" / root
        _age_tree(root_dir)
        _setup_authority(tmp_path, manager, stack, root, kind, mode)
        if kind == "generation" and mode == "present":
            _generation_reader(monkeypatch)
        lease = stack.enter_context(_opened_lease(tmp_path))
        planner = RetentionPlanner(tmp_path, lease, now_ns=_future_ns)
        result, reasons = _run_sweep(planner) if not (
            kind == "discovery" and mode == "malformed"
        ) else (planner.run_pass(days=30), {})
    if kind == "discovery" and mode == "malformed":
        assert not result.complete
        assert result.preview is None
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


@pytest.mark.parametrize("kind", _OVER_BOUND_KINDS)
def test_closure_authority_over_bound_matrix(tmp_path, kind):
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
        planner = RetentionPlanner(tmp_path, lease, bounds=bounds, now_ns=_future_ns)
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
