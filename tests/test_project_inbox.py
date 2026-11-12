from __future__ import annotations

import argparse
import asyncio
import io
import json
import multiprocessing
import multiprocessing.resource_tracker
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import zeta.agent.pending_delivery as pending_delivery_module
import zeta.project_inbox as project_inbox_module
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.cli.inbox import run as run_inbox_cli
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.project_inbox import (
    SENT_PAGE_LIMIT,
    InboxError,
    ProjectInbox,
    ProjectInboxScanner,
    SentMessageTracker,
)
from zeta.project_registry import ProjectRegistry, ProjectRegistryError
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    TextContent,
)
from zeta.runtime.loop import AgentLoop
from zeta.server.connections import ConnectionRegistry
from zeta.server.control import NotificationPresentation
from zeta.server.event_contract import notification_wire_event
from zeta.server.protocol import EventDescriptor, FrameCodec
from zeta.skills import SkillCatalog
from zeta.storage_format import open_writer
from zeta.tools.registry import ToolRegistry
from zeta.tui import theme
from zeta.tui.transcript.transcript import TranscriptWidget
from zeta.tui.transcript.transcript_presenter import TranscriptPresenter


@pytest.fixture(autouse=True)
def _retain_async_test_writer(tmp_path: Path, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("asyncio") is None:
        yield
        return
    with open_writer(tmp_path / ".zeta"):
        yield


def _stale_tracker_writer(
    sessions_root: str,
    session_id: str,
    target_project: str,
    connection,
) -> None:
    tracker = SentMessageTracker(Path(sessions_root), session_id)
    connection.send("loaded")
    connection.recv()
    try:
        tracker.record("d" * 32, target_project)
        tracker.mark_reported((("a" * 32, "done"),))
    except Exception as exc:  # noqa: BLE001 - process result is asserted by parent
        connection.send(f"{type(exc).__name__}: {exc}")
    else:
        connection.send(tracker.health().health.state)
    finally:
        connection.close()


def _projects(tmp_path: Path):
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    project_a = registry.create_project("alpha", "alpha work")
    project_b = registry.create_project("beta", "beta work")
    return home, registry, project_a, project_b


def test_exact_delegation_scopes_sender_and_kind(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    (home / "storage-format.json").write_text('{"writer_epoch":7}\n')
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.set_delegation(project_b.project_id, project_a.project_id, "question", enabled=True)
    question_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="question",
        title="question",
        body="body",
    )
    bug_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="bug_report",
        title="bug",
        body="body",
    )
    assert inbox.claim(project_b.project_id, question_id, "b" * 32) is not None
    assert inbox.claim(project_b.project_id, bug_id, "b" * 32) is None


def test_model_origin_defaults_to_information_and_cannot_claim(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="question",
        title="title",
        body="body",
    )

    listed = inbox.list(project_b.project_id)["new"]
    assert listed[0]["origin"] == "agent"
    assert listed[0]["author"] == "agent"
    assert listed[0]["authority"] == "information"
    assert inbox.claim(project_b.project_id, message_id, "b" * 32) is None


def test_missing_origin_is_information(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="question",
        title="legacy",
        body="body",
    )
    path = home / "projects" / project_b.project_id / "inbox" / "new" / f"{message_id}.json"
    value = json.loads(path.read_text())
    value.pop("origin", None)
    path.write_text(json.dumps(value))
    listed = inbox.list(project_b.project_id)["new"][0]
    assert listed.get("authority") == "information"


def test_information_message_claim_is_refused(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="question",
        title="question",
        body="body",
    )
    assert inbox.claim(project_b.project_id, message_id, "b" * 32) is None


def test_claim_syncs_both_inbox_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stat

    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="bug_report",
        title="test",
        body="test",
    )
    synced: list[int] = []
    original_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.fstat(fd).st_ino)
        original_fsync(fd)

    monkeypatch.setattr(project_inbox_module.os, "fsync", record_fsync)
    assert inbox.claim(project_b.project_id, message_id, "b" * 32) is not None

    assert len(set(synced)) >= 2


def _sent_status_message(
    message_id: str,
    status: str = "done",
    *,
    origin: MessageOrigin = MessageOrigin.HARNESS_NUDGE,
) -> Message:
    return Message(
        MessageRole.USER,
        [TextContent("status")],
        metadata={
            "zeta_event": "project_inbox_sent_status",
            "zeta.origin": origin.value,
            "sent_statuses": [{"message_id": message_id, "status": status}],
        },
    )


def test_inbox_receipt_index_includes_abandoned_branches(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    root = store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("root")],
            metadata={"zeta.origin": MessageOrigin.USER.value},
        )
    )
    store.append_message(_sent_status_message("a" * 32))
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("other branch")],
            metadata={"zeta.origin": MessageOrigin.USER.value},
        ),
        parent_id=root.id,
    )
    session_id = store.session_id
    store.close()
    reopened = ConversationStore(tmp_path, session_id=session_id)

    assert reopened.sent_message_receipts() == (("a" * 32, "done"),)


def test_inbox_receipt_index_ignores_lookalikes(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(_sent_status_message("not-an-id"))
    store.append_message(_sent_status_message("b" * 32, "unknown"))
    store.append_message(
        _sent_status_message("c" * 32, origin=MessageOrigin.USER)
    )
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("non-passive")],
            metadata={
                "zeta.origin": MessageOrigin.USER.value,
                "zeta_event": "project_inbox_sent_status",
                "sent_statuses": [{"message_id": "d" * 32, "status": "done"}],
            },
        )
    )

    assert store.sent_message_receipts() == ()


def test_inbox_receipt_index_updates_on_append(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    assert store.sent_message_receipts() == ()
    store.append_message(_sent_status_message("e" * 32, "claimed"))
    assert store.sent_message_receipts() == (("e" * 32, "claimed"),)


def test_send_is_idempotent_and_large_body_spills(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    body = "large body\n" * 10_000

    first = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project="beta",
        kind="bug_report",
        title="A bug",
        body=body,
        message_id="b" * 32,
    )
    second = inbox.send_user(
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


def test_read_missing_inbox_is_empty_without_creating_storage(tmp_path: Path) -> None:
    home, registry, _project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox_path = registry.root / project_b.project_id / "inbox"

    assert {path.name for path in inbox_path.iterdir()} == {".lock"}
    assert inbox.read(project_b.project_id) == {
        "new": [],
        "claimed": [],
        "done": [],
        "invalid": [],
    }
    assert {path.name for path in inbox_path.iterdir()} == {".lock"}


def test_read_rejects_missing_required_inbox_directory(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="message",
        body="body",
    )
    (registry.root / project_b.project_id / "inbox" / "done").rmdir()

    with pytest.raises(InboxError, match="inbox storage is unsafe or unavailable"):
        inbox.read(project_b.project_id)


def test_two_sessions_race_to_claim_and_dead_claim_returns_to_new(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
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
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="change_request",
        title="Reply safely",
        body="body",
    )
    session_id = "c" * 32
    assert inbox.claim(project_b.project_id, message_id, session_id) is not None
    real_send = inbox.send_with_outcome
    attempts = 0

    def flaky_send(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected reply publication failure")
        return real_send(**kwargs)

    monkeypatch.setattr(inbox, "send_with_outcome", flaky_send)
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
    message_id = inbox.send_user(
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
    assert replies[0].get("origin") == "agent"


def _notice_loop(home, registry, project_id, session_id, wakes):
    store = ConversationStore(home / "sessions", session_id=session_id)
    flow = pending_delivery_module.StepDeliveryFlow(
        store, wake_sink=lambda _source: wakes.append(session_id)
    )

    async def ensure_step_delivery():
        return flow

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
        _ensure_step_delivery=ensure_step_delivery,
    )


@pytest.mark.asyncio
async def test_one_message_wakes_exactly_one_of_two_peer_sessions(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    await asyncio.to_thread(inbox.send_user,
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
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    wakes = []
    loop = _notice_loop(home, registry, project_b.project_id, "b" * 32, wakes)
    ids = []
    try:
        for index in range(3):
            ids.append(
                await asyncio.to_thread(inbox.send_user,
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
    inbox.send_user(
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
    real_locked = registry._project_lock
    real_read = scanner.inbox._read_record

    def counted_locked(*args, **kwargs):
        nonlocal locks
        locks += 1
        return real_locked(*args, **kwargs)

    def counted_read(*args, **kwargs):
        nonlocal parses
        parses += 1
        return real_read(*args, **kwargs)

    # test-contract: seam count lock entries behind the scanner interface
    monkeypatch.setattr(registry, "_project_lock", counted_locked)
    monkeypatch.setattr(scanner.inbox, "_read_record", counted_read)
    assert [scanner.scan() for _ in range(5)] == [None] * 5
    assert locks == 0
    assert parses == 0


@pytest.mark.asyncio
async def test_targeted_inbox_notification_stays_with_target_session(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    (home / "storage-format.json").write_text('{"writer_epoch":7}\n')
    manager = SessionManager(home)
    session_a, session_b = await asyncio.to_thread(
        lambda: (
            manager.create(
                provider="codex",
                model="fake",
                cwd=home,
                project_id=project_b.project_id,
                auto_project=False,
            ),
            manager.create(
                provider="codex",
                model="fake",
                cwd=home,
                project_id=project_b.project_id,
                auto_project=False,
            ),
        )
    )
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        to_session=session_a.store.session_id,
        kind="info",
        title="only for a",
        body="private body for session a",
    )
    other_id = await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        to_session=session_b.store.session_id,
        kind="info",
        title="only for b",
        body="private body for session b",
    )
    loop_a = AgentLoop(
        FakeBackend([]),
        session_a.store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=project_b.project_id,
        project_registry=registry,
    )
    loop_b = AgentLoop(
        FakeBackend([]),
        session_b.store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=project_b.project_id,
        project_registry=registry,
    )
    try:
        await loop_a.activate()
        await loop_b.activate()
        notices_a = await _await_notices(
            session_a.store, what="session a inbox delivery"
        )
        notices_b = await _await_notices(
            session_b.store, what="session b inbox delivery"
        )
        assert notices_a and notices_a[0].data["message_ids"] == [message_id]
        assert "messages" not in notices_a[0].data
        assert notices_b and notices_b[0].data["message_ids"] == [other_id]
        assert "messages" not in notices_b[0].data
    finally:
        await loop_a.close()
        await loop_b.close()
        session_a.store.close()
        session_b.store.close()


async def test_session_notices_new_message_at_idle_or_tool_boundary(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="Notice me",
        body="body",
    )
    store = ConversationStore(home / "sessions", session_id="b" * 32)
    wakes: list[bool] = []
    flow = pending_delivery_module.StepDeliveryFlow(
        store, wake_sink=lambda _source: wakes.append(True)
    )

    async def ensure_step_delivery():
        return flow

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
        _ensure_step_delivery=ensure_step_delivery,
    )
    try:
        await AgentLoop._check_project_inbox(loop)
        notices = store.agent_notifications()
        assert wakes == [True]
        assert notices[0].data["kind"] == "project_inbox"
        assert notices[0].data["message_ids"] == [message_id]
        assert "Project inbox wake" in notices[0].data["text"]
        await AgentLoop._check_project_inbox(loop)
        assert len(store.agent_notifications()) == 1
    finally:
        store.close()


def test_inbox_cli_lists_a_named_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    ProjectInbox(registry, sessions_root=home / "sessions").send_user(
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


def test_unknown_optional_fields_survive_claim_and_done(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="newer writer",
        body="body",
    )
    new_path = registry.root / project_b.project_id / "inbox" / "new" / f"{message_id}.json"
    record = json.loads(new_path.read_text())
    record["future_optional"] = {"needed": True}
    record["from"]["future_sender_metadata"] = "preserve me"
    record["to_session"] = "f" * 32
    new_path.write_text(json.dumps(record) + "\n")

    state = inbox.list(project_b.project_id)
    assert [item["id"] for item in state["new"]] == [message_id]
    assert state["invalid"] == []

    session_id = "f" * 32
    claimed = inbox.claim(project_b.project_id, message_id, session_id)
    assert claimed is not None
    assert claimed["future_optional"] == {"needed": True}
    assert claimed["from"]["future_sender_metadata"] == "preserve me"
    assert claimed["to_session"] == "f" * 32
    completed = inbox.done(project_b.project_id, message_id, session_id, "complete")
    assert completed["future_optional"] == {"needed": True}
    assert completed["from"]["future_sender_metadata"] == "preserve me"
    done_path = registry.root / project_b.project_id / "inbox" / "done" / f"{message_id}.json"
    stored = json.loads(done_path.read_text())
    assert stored["future_optional"] == {"needed": True}
    assert stored["from"]["future_sender_metadata"] == "preserve me"
    assert stored["to_session"] == "f" * 32


def test_loader_isolates_unexpected_decode_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="valid before injected failure",
        body="body",
    )
    monkeypatch.setattr(
        project_inbox_module.json,
        "loads",
        lambda _data: (_ for _ in ()).throw(TypeError("injected decoder failure")),
    )

    state = inbox.list(project_b.project_id)

    assert state["new"] == []
    assert state["invalid"][0]["filename"] == f"{message_id}.json"
    assert len(state["invalid"][0]["reason"].encode()) <= 512


def test_to_session_requires_a_session_id(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="invalid target session",
        body="body",
    )
    path = registry.root / project_b.project_id / "inbox" / "new" / f"{message_id}.json"
    record = json.loads(path.read_text())
    record["to_session"] = "not-a-session"
    path.write_text(json.dumps(record))

    state = inbox.list(project_b.project_id)

    assert state["new"] == []
    assert state["invalid"][0]["reason"] == "invalid target session"


def test_large_integer_json_is_isolated(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    valid_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="valid",
        body="body",
    )
    invalid_id = "d" * 32
    path = registry.root / project_b.project_id / "inbox" / "new" / f"{invalid_id}.json"
    path.write_text('{"schema_version":1,"id":' + "9" * 5_000 + "}")

    state = inbox.list(project_b.project_id)

    assert [item["id"] for item in state["new"]] == [valid_id]
    assert state["invalid"] == [
        {
            "filename": path.name,
            "reason": f"malformed inbox message: {path.name}",
            "status": "new",
        }
    ]


def test_corrupt_and_higher_schema_messages_are_isolated(tmp_path: Path) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    valid_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="valid",
        body="body",
    )
    new_dir = registry.root / project_b.project_id / "inbox" / "new"
    corrupt_name = f"{'d' * 32}.json"
    (new_dir / corrupt_name).write_text("{not json")
    higher_id = "e" * 32
    higher = json.loads((new_dir / f"{valid_id}.json").read_text())
    higher["id"] = higher_id
    higher["schema_version"] = 2
    (new_dir / f"{higher_id}.json").write_text(json.dumps(higher) + "\n")

    state = inbox.list(project_b.project_id)

    assert [item["id"] for item in state["new"]] == [valid_id]
    assert {item["filename"] for item in state["invalid"]} == {
        corrupt_name,
        f"{higher_id}.json",
    }
    assert all(item["status"] == "new" for item in state["invalid"])
    assert (new_dir / corrupt_name).exists()
    assert (new_dir / f"{higher_id}.json").exists()
    assert inbox.claim(project_b.project_id, higher_id, "c" * 32) is None


def test_done_history_pruning_never_deletes_invalid_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.list(project_b.project_id)
    done_dir = registry.root / project_b.project_id / "inbox" / "done"
    invalid = done_dir / ("e" * 32 + ".json")
    invalid.write_text("{not json")
    monkeypatch.setattr(project_inbox_module, "DONE_HISTORY_LIMIT", 1)

    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="valid",
        body="body",
    )
    session_id = "c" * 32
    assert inbox.claim(project_b.project_id, message_id, session_id) is not None
    inbox.done(project_b.project_id, message_id, session_id, "complete")

    assert invalid.exists()
    state = inbox.list(project_b.project_id)
    assert state["invalid"][0]["filename"] == invalid.name


def test_invalid_message_is_logged_once_across_readers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    home, registry, _project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.list(project_b.project_id)
    invalid = registry.root / project_b.project_id / "inbox" / "new" / ("e" * 32 + ".json")
    invalid.write_text("{not json")

    with caplog.at_level("WARNING", logger="zeta.project_inbox"):
        inbox.list(project_b.project_id)
        ProjectInbox(registry, sessions_root=home / "sessions").list(project_b.project_id)

    messages = [record.message for record in caplog.records]
    assert sum("Skipping invalid project inbox message" in item for item in messages) == 1


@pytest.mark.asyncio
async def test_inbox_watcher_ignores_stable_invalid_message(tmp_path: Path) -> None:
    home, registry, _project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    await asyncio.to_thread(inbox.list, project_b.project_id)
    invalid = registry.root / project_b.project_id / "inbox" / "new" / ("e" * 32 + ".json")
    invalid.write_text("{not json")
    wakes: list[str] = []
    loop = _notice_loop(home, registry, project_b.project_id, "b" * 32, wakes)
    try:
        await AgentLoop._check_project_inbox(loop)
        await AgentLoop._check_project_inbox(loop)
        assert wakes == []
        assert loop.store.agent_notifications() == []
    finally:
        loop.store.close()


@pytest.mark.parametrize("unsafe", ["malformed", "symlink", "hardlink"])
def test_unsafe_message_files_are_isolated(tmp_path: Path, unsafe: str) -> None:
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

    state = inbox.list(project_b.project_id)
    assert state["new"] == []
    assert state["invalid"][0]["filename"] == path.name
    assert state["invalid"][0]["status"] == "new"


def test_all_invalid_messages_log_once_when_report_is_bounded(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, registry, _project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.list(project_b.project_id)
    new_dir = registry.root / project_b.project_id / "inbox" / "new"
    monkeypatch.setattr(
        project_inbox_module,
        "_LOGGED_INVALID",
        project_inbox_module.OrderedDict(),
    )
    for index in range(105):
        (new_dir / f"{index:032x}.json").write_text("{not json")

    with caplog.at_level("WARNING", logger="zeta.project_inbox"):
        state = inbox.list(project_b.project_id)
    assert len(state["invalid"]) == 100
    assert (
        sum(
            "Skipping invalid project inbox message" in record.message
            for record in caplog.records
        )
        == 105
    )

    caplog.clear()
    with caplog.at_level("WARNING", logger="zeta.project_inbox"):
        inbox.list(project_b.project_id)
    assert not any(
        "Skipping invalid project inbox message" in record.message
        for record in caplog.records
    )


def test_invalid_log_dedup_evicts_before_logging_new_files(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home, registry, _project_a, project_b = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    inbox.list(project_b.project_id)
    new_dir = registry.root / project_b.project_id / "inbox" / "new"
    monkeypatch.setattr(project_inbox_module, "_MAX_LOGGED_INVALID", 2)
    monkeypatch.setattr(
        project_inbox_module,
        "_LOGGED_INVALID",
        project_inbox_module.OrderedDict(),
    )
    for index in range(2):
        (new_dir / f"{index:032x}.json").write_text("{not json")
    with caplog.at_level("WARNING", logger="zeta.project_inbox"):
        inbox.list(project_b.project_id)

    caplog.clear()
    new_name = f"{2:032x}.json"
    (new_dir / new_name).write_text("{not json")
    with caplog.at_level("WARNING", logger="zeta.project_inbox"):
        inbox.list(project_b.project_id)
    assert any(new_name in record.message for record in caplog.records)

async def _collect_turn(loop: AgentLoop, text: str) -> None:
    async for _event in loop.run_turn(text, origin=MessageOrigin.USER):
        pass


def _sender_loop(
    home: Path,
    registry: ProjectRegistry,
    project_id: str,
    session_id: str,
    turns: int = 1,
):
    store = ConversationStore(home / "sessions", session_id=session_id, cwd=home)
    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("ok")]) for _ in range(turns)]
    )
    loop = AgentLoop(
        backend,
        store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=project_id,
        project_registry=registry,
    )
    return loop, backend, store


def _sent_status_notes(backend: FakeBackend) -> list:
    return [
        message
        for call, _tools in backend.calls
        for message in call
        if message.metadata.get("zeta_event") == "project_inbox_sent_status"
    ]


@pytest.mark.asyncio
async def test_claim_injects_note_on_senders_next_turn_without_waking(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="question",
        title="Can you review this?",
        body="Please review it.",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await loop._check_project_inbox()
        assert backend.calls == []
        assert store.agent_notifications() == []
        await _collect_turn(loop, "What changed?")
        notes = _sent_status_notes(backend)
        assert len(notes) == 1
        assert backend.calls[0][0][-1] is notes[0]
        assert notes[0].role is MessageRole.SYSTEM
        assert notes[0].content[0].text.startswith("[zeta] inbox status:")
        claimed = await asyncio.wait_for(
            asyncio.to_thread(inbox.read, project_b.project_id), 10
        )
        assert (
            'inbox: your message "Can you review this?" to beta was claimed '
            f"by session {'b' * 32} at {claimed['claimed'][0]['claimed_at']}"
            in notes[0].content[0].text
        )
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_claim_note_not_redelivered_after_fork_before_note(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id, turns=3
    )
    await _collect_turn(loop, "before send")
    forkpoint = next(
        entry
        for entry in store.replay()
        if entry.type == "message"
        and Message.from_dict(entry.data["message"]).role is MessageRole.USER
    )
    inbox = ProjectInbox(
        registry,
        sessions_root=home / "sessions",
        sent_tracker=loop.tool_registry.sent_message_tracker,
    )
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="question",
        title="Only once across branches",
        body="body",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    await loop._check_project_inbox()
    await _collect_turn(loop, "deliver status")
    assert len(_sent_status_notes(backend)) == 1
    store.append_message_fork(forkpoint.id)
    await loop.close()
    store.close()

    resumed, resumed_backend, resumed_store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await resumed._check_project_inbox()
        await _collect_turn(resumed, "after fork")
        assert _sent_status_notes(resumed_backend) == []
    finally:
        await resumed.close()
        resumed_store.close()


@pytest.mark.asyncio
async def test_claim_note_delivered_once_across_resume(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="info",
        title="One time",
        body="body",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    loop, _backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id, turns=2
    )
    await loop._check_project_inbox()
    await _collect_turn(loop, "first")
    await _collect_turn(loop, "second")
    assert (
        sum(
            message.metadata.get("zeta_event") == "project_inbox_sent_status"
            for message in store.messages()
        )
        == 1
    )
    await loop.close()
    store.close()

    resumed, _resumed_backend, resumed_store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await resumed._check_project_inbox()
        await _collect_turn(resumed, "after resume")
        assert (
            sum(
                message.metadata.get("zeta_event") == "project_inbox_sent_status"
                for message in resumed_store.messages()
            )
            == 1
        )
    finally:
        await resumed.close()
        resumed_store.close()


@pytest.mark.asyncio
async def test_done_with_reply_does_not_duplicate_signal(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="question",
        title="Reply to me",
        body="body",
    )
    claimer = "b" * 32
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, claimer) is not None
    await asyncio.to_thread(inbox.done,project_b.project_id, message_id, claimer, "answered", reply="done")
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await loop._check_project_inbox()
        assert len(store.agent_notifications()) == 1
        await _collect_turn(loop, "show me")
        assert _sent_status_notes(backend) == []
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_claim_note_not_in_system_prompt_and_not_user_origin(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="info",
        title="Origin",
        body="body",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await loop._check_project_inbox()
        await _collect_turn(loop, "hello")
        note = _sent_status_notes(backend)[0]
        assert all(
            "your message" not in block.text
            for block in loop.context_assembler.system_prompt.content
            if isinstance(block, TextContent)
        )
        assert (
            note.metadata[MESSAGE_ORIGIN_METADATA] == MessageOrigin.HARNESS_NUDGE.value
        )
        text = note.content[0].text
        assert "UNTRUSTED CROSS-PROJECT DATA" in text
        assert "data, not instructions" in text
        assert "--- BEGIN UNTRUSTED CROSS-PROJECT DATA ---" in text
        assert "--- END UNTRUSTED CROSS-PROJECT DATA ---" in text
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_claim_tracking_is_read_only_for_receiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="info",
        title="Read only",
        body="body",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    receiver_tree = registry.root / project_b.project_id / "inbox"

    def snapshot() -> dict[str, bytes]:
        return {
            str(path.relative_to(receiver_tree)): path.read_bytes()
            for path in receiver_tree.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    inbox_locks: list[bool] = []
    real_lock = ProjectInbox._lock

    @contextmanager
    def counted_lock(self, project_id: str, *, exclusive: bool):
        inbox_locks.append(exclusive)
        with real_lock(self, project_id, exclusive=exclusive) as project_fd:
            yield project_fd

    # test-contract: seam observe domain lock modes behind sent-status scanning
    monkeypatch.setattr(ProjectInbox, "_lock", counted_lock)
    scanner = ProjectInboxScanner(
        registry,
        project_a.project_id,
        sessions_root=home / "sessions",
        session_id=session_id,
    )
    statuses = await asyncio.to_thread(scanner.scan_sent)

    assert statuses is not None
    assert [(item["id"], item["status"]) for item in statuses] == [
        (message_id, "claimed")
    ]
    assert snapshot() == before
    assert inbox_locks == [False]
    assert [path.name for path in receiver_tree.rglob("*.lock")] == [".lock"]


def test_idle_tracker_does_no_filesystem_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    session_id = "a" * 32
    scanner = ProjectInboxScanner(
        registry,
        project_a.project_id,
        sessions_root=home / "sessions",
        session_id=session_id,
    )
    assert scanner.tracker is not None
    scanner.tracker.record("1" * 32, project_b.project_id)
    scanner.tracker.mark_reported(iter((("1" * 32, "done"),)))
    filesystem_calls = 0

    def unexpected_filesystem_call(*args, **kwargs):
        nonlocal filesystem_calls
        filesystem_calls += 1
        raise AssertionError("idle sent tracker accessed the filesystem")

    monkeypatch.setattr(project_inbox_module.os, "listdir", unexpected_filesystem_call)
    tracker_module = sys.modules[SentMessageTracker.__module__]
    monkeypatch.setattr(
        tracker_module, "read_session_file", unexpected_filesystem_call
    )

    assert scanner.scan_sent() == ()
    assert filesystem_calls == 0


def test_passive_scanner_io_is_bounded_by_sender_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="info",
        title="tracked",
        body="body",
    )
    reads: list[tuple[str, str]] = []
    real_read_tracked = ProjectInbox._read_tracked_message

    def counted_read_tracked(
        self, source_project, source_session, target_project, tracked_message_id
    ):
        reads.append((target_project, tracked_message_id))
        return real_read_tracked(
            self,
            source_project,
            source_session,
            target_project,
            tracked_message_id,
        )

    monkeypatch.setattr(ProjectInbox, "_read_tracked_message", counted_read_tracked)
    scanner = ProjectInboxScanner(
        registry,
        project_a.project_id,
        sessions_root=home / "sessions",
        session_id=session_id,
    )

    statuses = scanner.scan_sent()

    assert statuses is not None
    assert [(item["id"], item["status"]) for item in statuses] == [
        (message_id, "new")
    ]
    assert reads == [(project_b.project_id, message_id)]


def test_newest_record_with_lowest_id_survives_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _registry, _project_a, project_b = _projects(tmp_path)
    timestamps = iter(
        (
            "2026-01-01T00:00:00.000001Z",
            "2026-01-01T00:00:00.000002Z",
            "2026-01-01T00:00:00.000003Z",
            "2026-01-01T00:00:00.000004Z",
        )
    )
    tracker_module = sys.modules[SentMessageTracker.__module__]
    monkeypatch.setattr(tracker_module, "SENT_HISTORY_LIMIT", 2)
    # test-contract: seam make pruning order deterministic
    monkeypatch.setattr(tracker_module, "_now", lambda: next(timestamps))
    tracker = SentMessageTracker(home / "sessions", "a" * 32)

    tracker.record("f" * 32, project_b.project_id)
    tracker.record("e" * 32, project_b.project_id)
    tracker.record("0" * 32, project_b.project_id)

    # unresolved rows are retained; p03a prunes terminal rows only.
    assert {record["id"] for record in tracker.records()} == {
        "f" * 32,
        "e" * 32,
        "0" * 32,
    }

    tracker.mark_reported(iter((("e" * 32, "done"),)))
    tracker.record("1" * 32, project_b.project_id)

    assert {record["id"] for record in tracker.records()} == {
        "f" * 32,
        "0" * 32,
        "1" * 32,
    }


@pytest.mark.asyncio
async def test_cancel_after_status_note_persisted_does_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="info",
        title="Cancellation-safe receipt",
        body="body",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id, turns=1
    )
    original_durable_write = store._run_committed_mutation
    append_committed = threading.Event()
    release_append = threading.Event()

    def pause_after_status_commit(command, expected_generation, cancelled_before_write):
        result = original_durable_write(
            command, expected_generation, cancelled_before_write
        )
        message = (
            command.message
            if type(command).__name__ == "_AppendMessageCommand"
            else None
        )
        if (
            message is not None
            and message.metadata.get("zeta_event") == "project_inbox_sent_status"
        ):
            append_committed.set()
            assert release_append.wait(timeout=5)
        return result

    # test-contract: seam pauses after durable inbox status commit
    monkeypatch.setattr(store, "_run_committed_mutation", pause_after_status_commit)
    try:
        await loop._check_project_inbox()
        first_turn = asyncio.create_task(_collect_turn(loop, "first"))
        assert await asyncio.to_thread(append_committed.wait, 2)
        first_turn.cancel()
        release_append.set()
        with pytest.raises(asyncio.CancelledError):
            await await_or_fail(first_turn, what="test await")
        # test-contract: seam releases the next turn from the injected pause
        monkeypatch.setattr(store, "_run_committed_mutation", original_durable_write)
        await _collect_turn(loop, "second")

        persisted = [
            message
            for message in store.messages()
            if message.metadata.get("zeta_event") == "project_inbox_sent_status"
        ]
        assert len(persisted) == 1
        assert len(_sent_status_notes(backend)) == 1
    finally:
        release_append.set()
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_crash_between_note_append_and_receipt_does_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="info",
        title="Crash-safe receipt",
        body="body",
    )
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, "b" * 32) is not None
    loop, _backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    await loop._check_project_inbox()
    assert loop._inbox_scanner is not None

    def crash_before_receipt(*args, **kwargs):
        raise RuntimeError("simulated crash after durable note append")

    monkeypatch.setattr(loop._inbox_scanner, "mark_reported", crash_before_receipt)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await _collect_turn(loop, "first")
    assert sum(
        message.metadata.get("zeta_event") == "project_inbox_sent_status"
        for message in store.entries
        if message.type == "message"
        for message in [Message.from_dict(message.data["message"])]
    ) == 1
    await loop.close()
    store.close()

    resumed, resumed_backend, resumed_store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await resumed._check_project_inbox()
        await _collect_turn(resumed, "after crash")
        assert len(_sent_status_notes(resumed_backend)) == 1
    finally:
        await resumed.close()
        resumed_store.close()


@pytest.mark.asyncio
async def test_done_without_reply_injects_completed_status(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(inbox.send_user,
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="change_request",
        title="Finish this",
        body="body",
    )
    claimer = "b" * 32
    assert await asyncio.to_thread(inbox.claim,project_b.project_id, message_id, claimer) is not None
    completed = await asyncio.to_thread(inbox.done,
        project_b.project_id, message_id, claimer, "fixed without a reply"
    )
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id
    )
    try:
        await loop._check_project_inbox()
        assert backend.calls == []
        assert store.agent_notifications() == []
        await _collect_turn(loop, "next")
        notes = _sent_status_notes(backend)
        assert len(notes) == 1
        assert (
            'inbox: your message "Finish this" to beta was completed '
            f"at {completed['done_at']} (outcome: fixed without a reply)"
            in notes[0].content[0].text
        )
    finally:
        await loop.close()
        store.close()


def test_broken_sent_ledger_projects_authoritative_delivery_without_writing(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    session_id = "a" * 32
    session_dir = home / "sessions" / session_id
    session_dir.mkdir(parents=True)
    ledger = session_dir / "project-inbox-sent.json"
    original = b'{"schema_version":1,"records":[\n'
    tracker = SentMessageTracker(home / "sessions", session_id)
    ledger.write_bytes(original)
    inbox = ProjectInbox(
        registry, sessions_root=home / "sessions", sent_tracker=tracker
    )
    try:
        message_id = inbox.send_user(
            from_project=project_a.project_id,
            from_session=session_id,
            to_project=project_b.project_id,
            kind="bug_report",
            title="broken ledger",
            body="delivered",
            message_id="f" * 32,
        )
    except InboxError:
        message_id = None

    assert message_id == "f" * 32
    assert ledger.read_bytes() == original
    result = inbox.sent(project_a.project_id, session_id=session_id)
    assert result["history_complete"] is False
    assert result["state"] == "broken"
    assert [item["id"] for item in result["messages"]] == [message_id]
    assert result["messages"][0]["reported"] == []


def test_broken_projection_keeps_identity_when_spilled_body_is_undecodable(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    session_id = "a" * 32
    tracker = SentMessageTracker(home / "sessions", session_id)
    inbox = ProjectInbox(
        registry,
        sessions_root=home / "sessions",
        sent_tracker=tracker,
        receipts=(("0" * 32, "done"),),
    )
    message_id = inbox.send_user(
        from_project=project_a.project_id,
        from_session=session_id,
        to_project=project_b.project_id,
        kind="bug_report",
        title="body unavailable",
        body="x" * 70_000,
        message_id="1" * 32,
    )
    ledger = home / "sessions" / session_id / "project-inbox-sent.json"
    ledger.write_bytes(b"{")
    body = registry.root / project_b.project_id / "inbox" / "bodies" / f"{message_id}.txt"
    body.write_bytes(b"\\xff")

    result = inbox.sent(project_a.project_id, session_id=session_id)

    assert [row["id"] for row in result["messages"]] == [message_id]
    assert result["messages"][0]["reported"] == []
    assert result["history_complete"] is False


def test_broken_sent_projection_prunes_completed_row_with_both_receipts(
    tmp_path: Path,
) -> None:
    home, registry, project_a, project_b = _projects(tmp_path)
    session_id = "a" * 32
    session_dir = home / "sessions" / session_id
    session_dir.mkdir(parents=True)
    ledger = session_dir / "project-inbox-sent.json"
    completed_id = f"{0:032x}"
    tracker = SentMessageTracker(home / "sessions", session_id)
    inbox = ProjectInbox(
        registry,
        sessions_root=home / "sessions",
        sent_tracker=tracker,
    )
    expected_ids = []
    send_error = None
    for index in range(513):
        message_id = f"{index:032x}"
        try:
            inbox.send_user(
                from_project=project_a.project_id,
                from_session=session_id,
                to_project=project_b.project_id,
                kind="info",
                title=f"message {index}",
                body="body",
                message_id=message_id,
            )
        except InboxError as exc:
            send_error = exc
            break
        if message_id == completed_id:
            tracker.mark_reported(
                ((completed_id, "claimed"), (completed_id, "done"))
            )
        else:
            expected_ids.append(message_id)

    assert send_error is None
    ledger.write_bytes(b"{")
    projection_error = None
    try:
        broken_tracker = SentMessageTracker(home / "sessions", session_id)
        broken_tracker.mark_reported(
            ((completed_id, "claimed"), (completed_id, "done"))
        )
        inbox = ProjectInbox(
            registry,
            sessions_root=home / "sessions",
            sent_tracker=broken_tracker,
        )
    except InboxError as exc:
        projection_error = exc
    assert projection_error is None
    actual_ids = []
    offset = 0
    while True:
        page = inbox.sent(
            project_a.project_id,
            session_id=session_id,
            offset=offset,
            limit=SENT_PAGE_LIMIT,
        )
        actual_ids.extend(item["id"] for item in page["messages"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]

    assert page["total"] == 512
    assert set(actual_ids) == set(expected_ids)
    assert completed_id not in actual_ids
    assert ledger.read_bytes() == b"{"


def test_broken_sent_ledger_transitions_to_loaded_and_missing(tmp_path: Path) -> None:
    home, _registry, project_a, _project_b = _projects(tmp_path)
    session_id = "a" * 32
    ledger = home / "sessions" / session_id / "project-inbox-sent.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(b"{")
    tracker = SentMessageTracker(home / "sessions", session_id)
    assert "unless the repaired ledger includes their IDs" in tracker.health().warning
    retained = "2" * 32
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "id": retained,
                        "target_project": project_a.project_id,
                        "created_at": "2026-01-01T00:00:00.000000Z",
                        "reported": [],
                    }
                ],
            }
        )
    )

    assert tracker.health().health.state == "healthy"
    assert tracker.records()[0]["id"] == retained

    ledger.unlink()
    assert tracker.health().health.state == "healthy"
    assert tracker.records() == ()


def test_stale_loaded_process_adopts_broken_ledger_without_republishing(
    tmp_path: Path,
) -> None:
    home, _registry, project_a, project_b = _projects(tmp_path)
    session_id = "a" * 32
    ledger = home / "sessions" / session_id / "project-inbox-sent.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "id": "a" * 32,
                        "target_project": project_a.project_id,
                        "created_at": "2026-01-01T00:00:00.000000Z",
                        "reported": [],
                    }
                ],
            }
        )
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_stale_tracker_writer,
        args=(str(home / "sessions"), session_id, project_b.project_id, child),
    )
    process.start()
    child.close()
    corrupt = b'{"schema_version":1,"records":"broken"}'
    try:
        assert parent.poll(10), "stale writer did not load the healthy ledger"
        assert parent.recv() == "loaded"
        ledger.write_bytes(corrupt)
        parent.send("write")
        assert parent.poll(10), "stale writer did not finish"
        result = parent.recv()
        process.join(10)
        assert process.exitcode == 0
        assert result == "broken"
        assert ledger.read_bytes() == corrupt
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
        process.close()
        multiprocessing.resource_tracker._resource_tracker._stop()


@pytest.mark.asyncio
async def test_inbox_activation_reads_broken_ledger_off_owner_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, projects, project_a, _project_b = await asyncio.to_thread(_projects, tmp_path)
    store = ConversationStore(
        home / "sessions", session_id="a" * 32, cwd=tmp_path
    )
    ledger = store.session_dir / "project-inbox-sent.json"
    ledger.write_bytes(b"{")
    registry = ToolRegistry(
        tmp_path,
        skill_catalog=SkillCatalog.empty(),
        project_id=project_a.project_id,
        project_registry=projects,
        inbox_enabled=True,
    )
    loop = AgentLoop(
        FakeBackend([]),
        store,
        registry=registry,
        skill_catalog=SkillCatalog.empty(),
        skip_mcp_mount=True,
    )
    tracker_module = sys.modules[SentMessageTracker.__module__]
    real_read = tracker_module.read_session_file
    read_entered = threading.Event()
    release_read = threading.Event()
    owner_thread = threading.get_ident()

    def gated_read(*args, **kwargs):
        assert threading.get_ident() != owner_thread
        read_entered.set()
        assert release_read.wait(timeout=10)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(tracker_module, "read_session_file", gated_read)

    activation = asyncio.create_task(loop.activate())
    try:
        assert await asyncio.wait_for(asyncio.to_thread(read_entered.wait), 10)
        assert not release_read.is_set()
        owner_progress = asyncio.Event()
        asyncio.get_running_loop().call_soon(owner_progress.set)
        await asyncio.wait_for(owner_progress.wait(), 10)
        assert owner_progress.is_set()
        assert not release_read.is_set()
        release_read.set()
        await asyncio.wait_for(activation, 10)
        assert loop._inbox_scanner is not None
        health = await asyncio.to_thread(loop._inbox_scanner.tracker.health)
        assert health.health.state == "broken"
    finally:
        release_read.set()
        if not activation.done():
            activation.cancel()
            await asyncio.wait_for(
                asyncio.gather(activation, return_exceptions=True), 10
            )
        await loop.close()
        store.close()


def _raise_once_scanner_factory(loop, exc):
    """Wrap the real scanner so its first full scan raises, then it recovers."""
    real_new = loop._new_project_inbox_scanner
    state = {"n": 0}

    def new_scanner():
        scanner = real_new()
        real_scan = scanner.scan

        def wrapped_scan():
            state["n"] += 1
            if state["n"] == 1:
                raise exc
            return real_scan()

        scanner.scan = wrapped_scan
        return scanner

    return new_scanner


async def _await_notices(store, *, what: str):
    """Wait for the supervisor to deliver at least one durable inbox notice."""
    async with asyncio.timeout(10):
        while not store.agent_notifications():
            # test-contract: latency poll the delivered durable inbox notice
            await asyncio.sleep(0.02)
    return store.agent_notifications()


def _health_states(events):
    return [
        event.data["state"]
        for event in events
        if isinstance(event, StreamEvent)
        and event.data.get("kind") == "project_inbox_health"
    ]


def _health_signal(events, *, throw_on_degraded: bool = False):
    """Wire the background sink to record health events and fire per state.

    Returns (sink, degraded, healthy). The supervisor only publishes on a
    transition, so `healthy` fires on recovery, not on the first clean poll.
    """
    degraded = asyncio.Event()
    healthy = asyncio.Event()

    def sink(event):
        events.append(event)
        if not (
            isinstance(event, StreamEvent)
            and event.data.get("kind") == "project_inbox_health"
        ):
            return
        if event.data["state"] == "degraded":
            degraded.set()
            if throw_on_degraded:
                raise RuntimeError("health sink down")
        else:
            healthy.set()

    return sink, degraded, healthy


@pytest.mark.asyncio
async def test_p03a_failed_commit_retries_same_poll_before_rescan(
    tmp_path, monkeypatch
):
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="retry me",
        body="body",
    )
    loop, _backend, store = _sender_loop(home, registry, project_b.project_id, "b" * 32)

    scan_calls = {"n": 0}
    real_new = loop._new_project_inbox_scanner

    def counting_scanner():
        scanner = real_new()
        real_scan = scanner.scan

        def counting():
            scan_calls["n"] += 1
            return real_scan()

        scanner.scan = counting
        return scanner

    # test-contract: seam inject a scanner that counts real scans at the runtime seam
    monkeypatch.setattr(loop, "_new_project_inbox_scanner", counting_scanner)

    attempts = {"n": 0}
    holder: dict[str, bool] = {}
    real_ensure = loop._ensure_step_delivery

    async def ensure():
        flow = await real_ensure()
        if not holder:
            holder["wrapped"] = True
            real_produce = flow.produce

            async def produce(item):
                if item.kind == "inbox" and attempts["n"] == 0:
                    attempts["n"] += 1
                    raise RuntimeError("commit failed")
                return await real_produce(item)

            flow.produce = produce
        return flow

    # test-contract: seam fail the first inbox commit at the real delivery seam
    monkeypatch.setattr(loop, "_ensure_step_delivery", ensure)
    events: list[StreamEvent] = []
    sink, degraded, healthy = _health_signal(events)
    loop.set_background_event_sink(sink)
    try:
        # activation signals the supervisor: the first poll scans, then its
        # delivery commit fails, so the supervisor degrades and keeps the poll.
        await loop.activate()
        await await_or_fail(degraded.wait(), what="first inbox commit fails")
        # one more signal: the supervisor retries the retained poll (no rescan)
        # and this time the commit succeeds, so it recovers and delivers.
        await loop._check_project_inbox()
        await await_or_fail(healthy.wait(), what="retry delivers the retained poll")
        notices = store.agent_notifications()
        assert len(notices) == 1
        assert notices[0].data["message_ids"] == [message_id]
        # the identical poll is reused: the scanner never scans a second time.
        assert scan_calls["n"] == 1
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_p03a_repeated_delivery_failure_does_not_flap_health(
    tmp_path, monkeypatch
):
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="flap",
        body="body",
    )
    loop, _backend, store = _sender_loop(home, registry, project_b.project_id, "b" * 32)

    attempts = {"n": 0}
    attempt_seen = asyncio.Event()
    holder: dict[str, bool] = {}
    real_ensure = loop._ensure_step_delivery

    async def ensure():
        flow = await real_ensure()
        if not holder:
            holder["wrapped"] = True
            real_produce = flow.produce

            async def produce(item):
                if item.kind == "inbox":
                    attempts["n"] += 1
                    attempt_seen.set()
                    if attempts["n"] <= 2:
                        raise RuntimeError("commit failed")
                return await real_produce(item)

            flow.produce = produce
        return flow

    # test-contract: seam fail the first two inbox commits at the real delivery seam
    monkeypatch.setattr(loop, "_ensure_step_delivery", ensure)
    events: list[StreamEvent] = []
    sink, degraded, healthy = _health_signal(events)
    loop.set_background_event_sink(sink)
    try:
        # the first delivery fails and the owner degrades.
        await loop.activate()
        await await_or_fail(degraded.wait(), what="first delivery fails")
        # the second delivery also fails: the base publishes healthy before every
        # retry and flaps, while this branch stays degraded until a real success.
        attempt_seen.clear()
        await loop._check_project_inbox()
        await await_or_fail(attempt_seen.wait(), what="second delivery attempt")
        # the third delivery succeeds and the owner heals exactly once.
        await loop._check_project_inbox()
        await await_or_fail(healthy.wait(), what="delivery finally succeeds")
        assert _health_states(events) == ["degraded", "healthy"]
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_p03a_close_waits_for_blocked_scanner_worker(tmp_path, monkeypatch):
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="block",
        body="body",
    )
    loop, _backend, store = _sender_loop(home, registry, project_b.project_id, "b" * 32)

    gate = threading.Event()
    scanner_started = threading.Event()
    scanner_finished = threading.Event()
    scan_calls = {"n": 0}
    real_new = loop._new_project_inbox_scanner

    def blocking_scanner():
        scanner = real_new()
        real_scan = scanner.scan

        def blocking():
            scan_calls["n"] += 1
            result = real_scan()
            scanner_started.set()
            gate.wait(timeout=10)
            scanner_finished.set()
            return result

        scanner.scan = blocking
        return scanner

    # test-contract: seam inject a scanner whose real scan blocks the poll worker
    monkeypatch.setattr(loop, "_new_project_inbox_scanner", blocking_scanner)

    release_event = asyncio.Event()
    real_close_delivery = loop._close_delivery_resources

    async def wrapped_close_delivery(**kwargs):
        release_event.set()
        return await real_close_delivery(**kwargs)

    # test-contract: seam observe when close releases storage at the real release seam
    monkeypatch.setattr(loop, "_close_delivery_resources", wrapped_close_delivery)

    try:
        # activation signals the supervisor; its first poll blocks in the worker.
        await loop.activate()
        await await_or_fail(
            asyncio.to_thread(scanner_started.wait), what="scan blocks the worker"
        )
        closing = asyncio.create_task(loop.close())
        # base releases storage while the scanner thread is still blocked;
        # this branch parks close on the shielded worker and never releases yet.
        released_while_blocked = False
        for _ in range(2000):
            if release_event.is_set():
                released_while_blocked = not scanner_finished.is_set()
                break
            if closing.done():
                break
            await asyncio.sleep(0)
        gate.set()
        await await_or_fail(closing, what="close drains scanner")
        assert not released_while_blocked
    finally:
        gate.set()
        store.close()


@pytest.mark.asyncio
async def test_p03a_health_sink_failure_cannot_kill_the_owner(tmp_path, monkeypatch):
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="survive",
        body="body",
    )
    loop, _backend, store = _sender_loop(home, registry, project_b.project_id, "b" * 32)
    # test-contract: seam inject a scanner whose first real scan fails, then recovers
    monkeypatch.setattr(
        loop,
        "_new_project_inbox_scanner",
        _raise_once_scanner_factory(loop, ProjectRegistryError("registry")),
    )

    events: list[StreamEvent] = []
    sink, degraded, healthy = _health_signal(events, throw_on_degraded=True)
    loop.set_background_event_sink(sink)
    try:
        # the first poll hits a registry error and degrades; the sink throws on
        # that degraded event, which must not kill the supervisor.
        await loop.activate()
        await await_or_fail(degraded.wait(), what="first scan degrades")
        # the next signal recovers the same owner, which delivers and heals.
        await loop._check_project_inbox()
        await await_or_fail(healthy.wait(), what="owner recovers after a dead sink")
        assert _health_states(events) == ["degraded", "healthy"]
        assert len(store.agent_notifications()) == 1
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_p03a_status_backlog_survives_without_a_count_cap(tmp_path: Path) -> None:
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    session_id = "a" * 32
    loop, backend, store = _sender_loop(
        home, registry, project_a.project_id, session_id, turns=2
    )
    # the first check builds the sender's tracker so the sends are recorded.
    await loop._check_project_inbox()
    tracker = loop.tool_registry.sent_message_tracker
    assert tracker is not None
    inbox = ProjectInbox(
        registry, sessions_root=home / "sessions", sent_tracker=tracker
    )
    ids = []
    for index in range(101):
        message_id = await asyncio.to_thread(
            inbox.send_user,
            from_project=project_a.project_id,
            from_session=session_id,
            to_project=project_b.project_id,
            kind="info",
            title=f"backlog {index}",
            body="body",
        )
        assert (
            await asyncio.to_thread(
                inbox.claim, project_b.project_id, message_id, "b" * 32
            )
            is not None
        )
        ids.append(message_id)
    try:
        await loop._check_project_inbox()
        reported: list[tuple[str, str]] = []
        real_mark = loop._inbox_scanner.mark_reported

        def capture_mark(statuses):
            items = tuple(statuses)
            reported.extend(items)
            return real_mark(iter(items))

        # test-contract: seam capture the statuses the turn reports to the tracker
        loop._inbox_scanner.mark_reported = capture_mark
        await _collect_turn(loop, "deliver the backlog")
        notes = _sent_status_notes(backend)
        assert len(notes) == 1
        # every claimed status is carried; the base pruned the pending set to 100.
        assert len(notes[0].metadata["sent_statuses"]) == 101
        assert {entry["message_id"] for entry in notes[0].metadata["sent_statuses"]} == set(ids)
        # the tracker records the whole backlog as reported, not a capped subset.
        assert {message_id for message_id, _status in reported} == set(ids)
    finally:
        await loop.close()
        store.close()


@pytest.mark.asyncio
async def test_p03a_runtime_health_event_reaches_tui_and_serve(tmp_path, monkeypatch):
    home, registry, project_a, project_b = await asyncio.to_thread(_projects, tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    await asyncio.to_thread(
        inbox.send_user,
        from_project=project_a.project_id,
        from_session="a" * 32,
        to_project=project_b.project_id,
        kind="info",
        title="health",
        body="body",
    )
    loop, _backend, store = _sender_loop(home, registry, project_b.project_id, "b" * 32)
    # test-contract: seam inject a scanner whose first real scan fails, then recovers
    monkeypatch.setattr(
        loop,
        "_new_project_inbox_scanner",
        _raise_once_scanner_factory(loop, ProjectRegistryError("broken registry")),
    )

    events: list[StreamEvent] = []
    sink, degraded, _healthy = _health_signal(events)
    loop.set_background_event_sink(sink)
    try:
        # activation signals the supervisor; the first poll hits a broken
        # registry and the runtime emits a degraded health event for the owner.
        await loop.activate()
        await await_or_fail(degraded.wait(), what="runtime emits degraded health")
        health = next(
            event
            for event in events
            if event.data.get("kind") == "project_inbox_health"
        )
        assert health.data["state"] == "degraded"
        assert health.data["session_id"] == store.session_id
    finally:
        await loop.close()
        store.close()

    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        Console(file=io.StringIO(), width=80),
        lambda: False,
        lambda renderable: None,
    )
    rendered = presenter.render_event(health)
    assert rendered is not None
    assert "project inbox degraded" in str(rendered)
    assert rendered.style is theme.WARNING

    # the serve wire goes through the real notification presentation: the
    # runtime event's own kind drives notification_wire_event, and a 1.2
    # controller receives the frame only because EVENT_FEATURES names the
    # descriptor. this is the runtime-event -> wire-event -> admission chain.
    class _HealthController:
        def __init__(self) -> None:
            self.codec = FrameCodec()
            self.frames: list[bytes] = []
            self._write_lock = asyncio.Lock()
            self.server = SimpleNamespace(runtime=SimpleNamespace())

        async def _write_physical(self, frame: bytes) -> bool:
            async with self._write_lock:
                self.frames.append(frame)
                return True

        async def close(self) -> None:
            pass

    connections = ConnectionRegistry(asyncio.get_running_loop())
    connection = _HealthController()
    connections.register(connection)
    connections.negotiate_role(connection, "controller", "1.2", frozenset())

    async def _send(event, session_id, **fields):
        return await connections.publish_controller(event, session_id, **fields)

    presentation = NotificationPresentation()
    try:
        hello = connection.codec.notification(
            EventDescriptor.ASSISTANT_DELTA, health.data["session_id"], delta="hello"
        )
        assert await connections.send(connection, hello)
        connection.frames.clear()
        await presentation.publish(
            health.data["session_id"], health.data, "1.2", _send
        )
        params = [json.loads(frame)["params"] for frame in connection.frames]
        health_params = [p for p in params if p.get("event") == "project_inbox_health"]
        assert len(health_params) == 1
        assert health_params[0]["session_id"] == store.session_id
        assert health_params[0]["data"]["state"] == "degraded"
        assert len(health_params[0]["data"].get("reason", "")) <= 200
    finally:
        connections.release(connection)
        await connections.aclose()
    # the legacy protocol keeps the health event on the shared notification wire.
    assert (
        notification_wire_event("project_inbox_health", "1.0")
        is not EventDescriptor.PROJECT_INBOX_HEALTH
    )
