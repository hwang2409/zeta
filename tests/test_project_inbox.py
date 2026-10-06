from __future__ import annotations

import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.cli.inbox import run as run_inbox_cli
from zeta.core.store import ConversationStore
from zeta.project_inbox import InboxError, ProjectInbox, ProjectInboxScanner
from zeta.project_registry import ProjectRegistry
from zeta.runtime.loop import AgentLoop


def _projects(tmp_path: Path):
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    project_a = registry.create_project("alpha", "alpha work")
    project_b = registry.create_project("beta", "beta work")
    return home, registry, project_a, project_b


def test_send_is_idempotent_and_large_body_spills(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    body = "large body\n" * 10_000

    first = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project="beta",
        kind="bug_report",
        title="A bug",
        body=body,
        message_id="b" * 32,
    )
    second = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project="beta",
        kind="bug_report",
        title="A bug",
        body=body,
        message_id="b" * 32,
    )

    new_dir = registry.root / project_b.project_id / "inbox" / "new"
    assert first == second
    assert [path.name for path in new_dir.iterdir()] == [f"{first}.json"]
    raw = json.loads((new_dir / f"{first}.json").read_text())
    assert raw["schema_version"] == 1
    assert raw["body"] == {"file": f"bodies/{first}.txt"}
    listed = inbox.list(project_b.project_id)
    assert listed["new"][0]["body"] == body


def test_two_sessions_race_to_claim_and_dead_claim_returns_to_new(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="question",
        title="Race",
        body="Who gets this?",
    )
    live_session = "c" * 32
    live_dir = home / "sessions" / live_session
    live_dir.mkdir(parents=True)
    live_fd = os.open(live_dir, os.O_RDONLY | os.O_DIRECTORY)
    import fcntl

    fcntl.flock(live_fd, fcntl.LOCK_SH)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda session: inbox.claim(
                        project_b.project_id, message_id, session
                    ),
                    (live_session, "d" * 32),
                )
            )
        assert sum(result is not None for result in results) == 1
    finally:
        fcntl.flock(live_fd, fcntl.LOCK_UN)
        os.close(live_fd)

    # Whichever session won is now dead because no session lease remains.
    state = inbox.list(project_b.project_id)
    assert [item["id"] for item in state["new"]] == [message_id]
    assert state["new"][0]["recovery_note"].startswith("Returned from stale claim")
    assert state["claimed"] == []


def test_done_reply_failure_is_retryable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="change_request",
        title="Reply safely",
        body="body",
    )
    session_id = "c" * 32
    assert inbox.claim(project_b.project_id, message_id, session_id) is not None
    real_send = inbox.send
    attempts = 0

    def flaky_send(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected reply publication failure")
        return real_send(**kwargs)

    monkeypatch.setattr(inbox, "send", flaky_send)
    with pytest.raises(OSError, match="injected reply"):
        inbox.done(
            project_b.project_id,
            message_id,
            session_id,
            "complete",
            reply="finished",
        )

    completed = inbox.done(
        project_b.project_id,
        message_id,
        session_id,
        "complete",
        reply="finished",
    )
    repeated = inbox.done(
        project_b.project_id,
        message_id,
        session_id,
        "complete",
        reply="finished",
    )

    replies = inbox.list(project_a.project_id)["new"]
    assert completed["id"] == repeated["id"] == message_id
    assert len(replies) == 1
    assert replies[0]["body"] == "finished"
    assert replies[0]["in_reply_to"] == message_id


def test_done_records_outcome_and_reply_reaches_sender(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    body = "Details\n" * 10_000
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="change_request",
        title="Please change this",
        body=body,
    )
    claimer = "c" * 32
    session_dir = home / "sessions" / claimer
    session_dir.mkdir(parents=True)
    fd = os.open(session_dir, os.O_RDONLY | os.O_DIRECTORY)
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        claimed = inbox.claim(project_b.project_id, message_id, claimer)
        assert claimed is not None
        assert claimed["body"] == body
        claimed_path = (
            registry.root / project_b.project_id / "inbox" / "claimed" / f"{message_id}.json"
        )
        assert json.loads(claimed_path.read_text())["body"] == {
            "file": f"bodies/{message_id}.txt"
        }
        completed = inbox.done(
            project_b.project_id,
            message_id,
            claimer,
            "fixed in PR #12",
            reply="The fix is ready.",
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert completed["outcome"] == "fixed in PR #12"
    assert completed["body"] == body
    done_path = registry.root / project_b.project_id / "inbox" / "done" / f"{message_id}.json"
    assert json.loads(done_path.read_text())["body"] == {
        "file": f"bodies/{message_id}.txt"
    }
    assert inbox.list(project_b.project_id)["done"][0]["id"] == message_id
    replies = inbox.list(project_a.project_id)["new"]
    assert len(replies) == 1
    assert replies[0]["kind"] == "reply"
    assert replies[0]["in_reply_to"] == message_id
    assert replies[0]["body"] == "The fix is ready."


def _notice_loop(home, registry, project_id, session_id, wakes):
    store = ConversationStore(home / "sessions", session_id=session_id)
    return SimpleNamespace(
        tool_registry=SimpleNamespace(
            registered_names=frozenset({"inbox"}),
            project_registry=registry,
            project_id=project_id,
        ),
        agent_depth=0,
        _turn_active=False,
        _inbox_message_ids=(),
        _inbox_scanner=ProjectInboxScanner(
            registry, project_id, sessions_root=home / "sessions"
        ),
        store=store,
        notify_background_persisted=lambda: wakes.append(session_id),
    )


@pytest.mark.asyncio
async def test_one_message_wakes_exactly_one_of_two_peer_sessions(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="one wake",
        body="body",
    )
    wakes = []
    loops = [
        _notice_loop(home, registry, project_b.project_id, "b" * 32, wakes),
        _notice_loop(home, registry, project_b.project_id, "c" * 32, wakes),
    ]
    try:
        for loop in loops:
            await AgentLoop._check_project_inbox(loop)
        assert len(wakes) == 1
        assert all(len(loop.store.agent_notifications()) == 1 for loop in loops)
    finally:
        for loop in loops:
            loop.store.close()


@pytest.mark.asyncio
async def test_arrivals_coalesce_into_one_pending_notice(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    wakes = []
    loop = _notice_loop(home, registry, project_b.project_id, "b" * 32, wakes)
    ids = []
    try:
        for index in range(3):
            ids.append(
                inbox.send(
                    from_project=project_a.project_id,
                    from_session="a" * 32,
                    to_project=project_b.project_id,
                    kind="info",
                    title=f"arrival {index}",
                    body="body",
                )
            )
            await AgentLoop._check_project_inbox(loop)
        notices = loop.store.agent_notifications()
        assert len(notices) == 1
        assert notices[0].data["message_ids"] == sorted(ids)
    finally:
        loop.store.close()


def test_unchanged_scanner_uses_no_locks_or_json_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="stable",
        body="body",
    )
    scanner = ProjectInboxScanner(
        registry, project_b.project_id, sessions_root=home / "sessions"
    )
    assert scanner.scan() is not None
    locks = 0
    parses = 0
    real_locked = registry._locked
    real_read = scanner.inbox._read_record

    def counted_locked(*args, **kwargs):
        nonlocal locks
        locks += 1
        return real_locked(*args, **kwargs)

    def counted_read(*args, **kwargs):
        nonlocal parses
        parses += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(registry, "_locked", counted_locked)
    monkeypatch.setattr(scanner.inbox, "_read_record", counted_read)
    assert [scanner.scan() for _ in range(5)] == [None] * 5
    assert locks == 0
    assert parses == 0


@pytest.mark.asyncio
async def test_active_turn_skips_periodic_inbox_scan(tmp_path: Path) -> None:
    _home, registry, _project_a, project_b = _projects(tmp_path)
    scans = []
    loop = SimpleNamespace(
        tool_registry=SimpleNamespace(
            registered_names=frozenset({"inbox"}),
            project_registry=registry,
            project_id=project_b.project_id,
        ),
        agent_depth=0,
        _turn_active=True,
        _inbox_scanner=SimpleNamespace(scan=lambda: scans.append(True)),
    )
    await AgentLoop._check_project_inbox(loop, periodic=True)
    assert scans == []


@pytest.mark.asyncio
async def test_session_notices_new_message_at_idle_or_tool_boundary(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="Notice me",
        body="body",
    )
    store = ConversationStore(home / "sessions", session_id="b" * 32)
    wakes: list[bool] = []
    loop = SimpleNamespace(
        tool_registry=SimpleNamespace(
            registered_names=frozenset({"inbox"}),
            project_registry=registry,
            project_id=project_b.project_id,
        ),
        agent_depth=0,
        _turn_active=False,
        _inbox_message_ids=(),
        _inbox_scanner=ProjectInboxScanner(
            registry, project_b.project_id, sessions_root=home / "sessions"
        ),
        store=store,
        notify_background_persisted=lambda: wakes.append(True),
    )
    try:
        await AgentLoop._check_project_inbox(loop)
        notices = store.agent_notifications()
        assert wakes == [True]
        assert notices[0].data["kind"] == "project_inbox"
        assert notices[0].data["message_ids"] == [message_id]
        await AgentLoop._check_project_inbox(loop)
        assert len(store.agent_notifications()) == 1
    finally:
        store.close()


def test_inbox_cli_lists_a_named_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    ProjectInbox(registry, sessions_root=home / "sessions").send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="question",
        title="CLI",
        body="visible",
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    output = io.StringIO()
    code = run_inbox_cli(argparse.Namespace(project="beta"), stdout=output)

    assert code == 0
    value = json.loads(output.getvalue())
    assert value["project"]["name"] == "beta"
    assert value["new"][0]["title"] == "CLI"


@pytest.mark.parametrize("unsafe", ["malformed", "symlink", "hardlink"])
def test_unsafe_message_files_are_rejected(tmp_path: Path, unsafe: str) -> None:
    home, registry, _project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.list(project_b.project_id)
    path = registry.root / project_b.project_id / "inbox" / "new" / ("e" * 32 + ".json")
    external = tmp_path / "external"
    external.write_text("not json", encoding="utf-8")
    if unsafe == "malformed":
        path.write_text("{not json", encoding="utf-8")
    elif unsafe == "symlink":
        path.symlink_to(external)
    else:
        path.hardlink_to(external)

    with pytest.raises(InboxError):
        inbox.list(project_b.project_id)
