"""Retention age is transcript time only (design s4c F2)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from tests.support.retention_owner import (
    _DAY_NS,
    _age_tree,
    _manager,
    _opened_lease,
    _planner,
    _run_sweep,
)
from zeta.cli.main import main as cli_main
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.remote_sync import LocalTransport, pull_session, push_session


def _closed_session(home, *, child: bool) -> str:
    """A closed root with one message and, optionally, one finished child store."""
    opened = _manager(home).create(
        provider="codex", model="gpt-5.6-sol", cwd=str(home), auto_project=False
    )
    try:
        opened.store.append_message(Message(MessageRole.ASSISTANT, [TextContent("root")]))
        if child:
            store = ConversationStore(
                opened.store.session_dir / "agents", session_id="1", cwd=str(home)
            )
            try:
                store.append_message(Message(MessageRole.ASSISTANT, [TextContent("child")]))
            finally:
                store.close()
            (opened.store.session_dir / "agents" / "1" / "agent_lifecycle.json").write_text(
                json.dumps({
                    "handle": "child", "state": "completed",
                    "started_at": "2020-01-01T00:00:00+00:00",
                    "finished_at": "2020-01-01T00:00:00+00:00",
                })
            )
    finally:
        opened.store.close()
    return opened.metadata.session_id


def _set_mtime(path, days_ago: float) -> None:
    moment = time.time_ns() - int(days_ago * _DAY_NS)
    os.utime(path, ns=(moment, moment))


def _touch_aux_files_now(session_dir) -> None:
    """The 2026-11-13 shape: lock, index and meta written today; directories touched."""
    (session_dir / ".project-links.lock").write_bytes(b"")
    (session_dir / "history-index.sqlite3").write_bytes(b"index")
    os.utime(session_dir / "meta.json")
    for directory in (session_dir, *session_dir.glob("agents"), *session_dir.glob("agents/*")):
        os.utime(directory)


def _transcripts(home) -> dict[str, tuple[int, int, int]]:
    return {
        path.relative_to(home).as_posix(): (
            path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns
        )
        for path in sorted((home / "sessions").rglob("conversation.jsonl"))
    }


def _preview(home):
    with ExitStack() as stack:
        lease = stack.enter_context(_opened_lease(home))
        planner = _planner(stack, home, lease, now_ns=time.time_ns)
        return _run_sweep(planner)


def test_new_aux_files_do_not_make_old_transcripts_young(tmp_path):
    old = _closed_session(tmp_path, child=True)
    young_child = _closed_session(tmp_path, child=True)
    for root in (old, young_child):
        _age_tree(tmp_path / "sessions" / root, days=40)
    _set_mtime(tmp_path / "sessions" / young_child / "agents" / "1" / "conversation.jsonl", 5)
    for root in (old, young_child):
        _touch_aux_files_now(tmp_path / "sessions" / root)

    result, reasons = _preview(tmp_path)

    assert reasons[old] == (True, None)
    assert result.preview.root_ids == (old,)
    # A child's transcript is activity of its root.
    assert reasons[young_child] == (False, "not old enough")


def _epoch5_home_with_sessions(home) -> tuple[set[str], str]:
    """Write the sessions in another process that exits, like the old zeta before a cutover."""
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from tests.test_session_retention_age import _closed_session\n"
        "from zeta.storage_format import advance\n"
        "home = Path(sys.argv[1])\n"
        "advance(home, expected_epoch=0, target_epoch=5)\n"
        "old = [_closed_session(home, child=True), _closed_session(home, child=False)]\n"
        "print(json.dumps([old, _closed_session(home, child=False)]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(home)],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=60, check=True,
    )
    old, recent = json.loads(done.stdout)
    return set(old), recent


def test_cutover_keeps_transcript_age(tmp_path, monkeypatch):
    old, recent = _epoch5_home_with_sessions(tmp_path)
    for root in old:
        _age_tree(tmp_path / "sessions" / root, days=40)
    _age_tree(tmp_path / "sessions" / recent, days=5)
    for root in (*old, recent):
        _touch_aux_files_now(tmp_path / "sessions" / root)
    before = _transcripts(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    assert cli_main(["storage-format", "advance", "--expected-epoch", "5", "--target-epoch", "11"]) == 0

    assert _transcripts(tmp_path) == before
    result, reasons = _preview(tmp_path)
    assert set(result.preview.root_ids) == old
    assert reasons[recent] == (False, "not old enough")


def test_push_and_pull_keep_transcript_mtimes(tmp_path):
    source, remote, destination = (tmp_path / name for name in ("source", "remote", "destination"))
    session_id = _closed_session(source, child=True)
    _age_tree(source / "sessions" / session_id, days=40)
    expected = {
        path.relative_to(source / "sessions").as_posix(): path.stat().st_mtime_ns
        for path in (source / "sessions").rglob("conversation.jsonl")
    }
    transport = LocalTransport(remote)

    push_session(source, transport, session_id=session_id)
    pull_session(destination, transport, session_id=session_id)

    for home in (remote, destination):
        copied = {
            path.relative_to(home / "sessions").as_posix(): path.stat().st_mtime_ns
            for path in (home / "sessions").rglob("conversation.jsonl")
        }
        assert copied == expected, home.name
