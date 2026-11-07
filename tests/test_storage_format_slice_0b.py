from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import multiprocessing
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from zeta.accounts import AccountError, AccountId, AccountStore
from zeta.core.session_files import atomic_publish_file
from zeta.mcp.oauth_store import MCPOAuthToken, save_token, token_store_path
from zeta.memory.reconciliation_state import ReconciliationState
from zeta.project_inbox import (
    SENT_HISTORY_LIMIT,
    InboxError,
    ProjectInbox,
    SentMessageTracker,
)
from zeta.project_registry import ProjectRegistry, ProjectRegistryError
from zeta.runtime.loop.project_inbox import ProjectInboxNotificationMixin
from zeta.server.slash_commands import ServerSlashSession
from zeta.storage_format import (
    BROAD_OWNED_SUBTREES,
    CLOSED_VARIANTS,
    SetValue,
    StorageFormatError,
    open_writer,
    require_closed_variant,
)
from zeta.tui.slash_handlers import SlashHandlerMixin


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _publish_shape_field(
    home: str,
    directory: str,
    field: str,
    value: object,
    entered: multiprocessing.synchronize.Event | None = None,
    release: multiprocessing.synchronize.Event | None = None,
) -> None:
    with open_writer(home) as lease:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:

            def publish(fd: int, name: str, payload: bytes) -> None:
                if entered is not None:
                    entered.set()
                if release is not None:
                    release.wait()
                atomic_publish_file(fd, name, payload, sync_directory=True)

            lease.publish_json(
                directory_fd,
                "state.json",
                (SetValue((field,), value),),
                publish,
            )
        finally:
            os.close(directory_fd)


def test_project_memory_pointer_preserves_unknown_fields(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("alpha", "test project")
    registry.update_memory(project.project_id, {"state.md": "one"})
    pointer = home / "projects" / project.project_id / "memory-current.json"
    value = _read(pointer)
    value["future-field"] = {"kept": True}
    _write(pointer, value)

    registry.update_memory(project.project_id, {"state.md": "two"})

    assert _read(pointer)["future-field"] == {"kept": True}


def test_project_memory_prune_keeps_unknown_version_records(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("alpha", "test project")
    registry.update_memory(project.project_id, {"state.md": "one"})
    memory_root = home / "projects" / project.project_id
    pointer = memory_root / "memory-current.json"
    unknown_version = "f" * 32
    unknown_blob = "e" * 64
    value = _read(pointer)
    value["future_version"] = unknown_version
    _write(pointer, value)
    versions = memory_root / "memory-versions" / "versions"
    blobs = memory_root / "memory-versions" / "blobs"
    _write(
        versions / f"{unknown_version}.json",
        {"version": unknown_version, "snapshot": {"future.md": unknown_blob}},
    )
    (blobs / unknown_blob).write_text("future")

    registry.update_memory(project.project_id, {"state.md": "two"})

    assert _read(pointer)["future_version"] == unknown_version
    assert (versions / f"{unknown_version}.json").exists()
    assert (blobs / unknown_blob).read_text() == "future"


def test_sent_ledger_preserves_unknown_record_fields_and_prunes_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zeta.project_inbox as inbox

    assert SENT_HISTORY_LIMIT > 2
    monkeypatch.setattr(inbox, "SENT_HISTORY_LIMIT", 2)
    sessions = tmp_path / "sessions"
    session_id = "1" * 32
    messages = ("2" * 32, "3" * 32, "4" * 32)
    project_id = "5" * 32
    tracker = SentMessageTracker(sessions, session_id)
    tracker.record(messages[0], project_id)
    ledger = sessions / session_id / "project-inbox-sent.json"
    value = _read(ledger)
    records = value["records"]
    assert isinstance(records, list)
    records[0]["future-field"] = "kept"
    _write(ledger, value)

    tracker = SentMessageTracker(sessions, session_id)
    tracker.mark_reported(((messages[0], "done"),))
    tracker.record(messages[1], project_id)
    value = _read(ledger)
    records = value["records"]
    assert isinstance(records, list)
    records[1]["future-field"] = "kept"
    _write(ledger, value)
    tracker = SentMessageTracker(sessions, session_id)
    tracker.record(messages[2], project_id)

    records = _read(ledger)["records"]
    assert isinstance(records, list)
    assert [record["id"] for record in records] == [messages[1], messages[2]]
    assert records[0]["future-field"] == "kept"
    assert all(record["id"] != messages[0] for record in records)


def test_sent_ledger_preserves_unknown_record_ids(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session_id = "1" * 32
    tracker = SentMessageTracker(sessions, session_id)
    tracker.record("2" * 32, "3" * 32)
    ledger = sessions / session_id / "project-inbox-sent.json"
    value = _read(ledger)
    records = value["records"]
    assert isinstance(records, list)
    records.append(
        {
            "id": "4" * 32,
            "target_project": "5" * 32,
            "created_at": "9999-01-01T00:00:00.000000Z",
            "reported": [],
            "future-field": True,
        }
    )
    _write(ledger, value)

    SentMessageTracker(sessions, session_id).record("6" * 32, "7" * 32)

    records = _read(ledger)["records"]
    assert isinstance(records, list)
    future = next(record for record in records if record["id"] == "4" * 32)
    assert future["future-field"] is True


def test_reconciliation_keyed_record_preserves_unknown_fields(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    path = home / "sessions" / "session-1" / "memory-reconcile.json"
    path.parent.mkdir(parents=True)
    kwargs = {
        "project_id": "p_alpha",
        "session_id": "session-1",
        "diagnostics_path": home / "logs" / "memory.jsonl",
    }
    with open_writer(home) as lease:
        if "storage_lease" in inspect.signature(ReconciliationState).parameters:
            kwargs["storage_lease"] = lease
        state = ReconciliationState(path, **kwargs)
        state.record_failure(
            seq_start=1,
            seq_end=2,
            validation_summary="failed",
            reason="test",
            usage={},
            retry_backoff_seconds=1,
            now=1,
            occurred_at="2026-01-01T00:00:00Z",
        )
        value = _read(path)
        pending = value["pending_failures"]
        assert isinstance(pending, list)
        pending[0]["future-field"] = {"kept": True}
        _write(path, value)
        state = ReconciliationState(path, **kwargs)
        state.record_failure(
            seq_start=1,
            seq_end=2,
            validation_summary="failed again",
            reason="test",
            usage={},
            retry_backoff_seconds=1,
            now=2,
            occurred_at="2026-01-01T00:00:01Z",
        )

    pending = _read(path)["pending_failures"]
    assert isinstance(pending, list)
    assert pending[0]["future-field"] == {"kept": True}


def test_account_registry_preserves_unknown_root_fields(tmp_path: Path) -> None:
    store = AccountStore(tmp_path)
    first = AccountId("anthropic", "first")
    second = AccountId("anthropic", "second")
    with store._home() as home_fd:
        store._register_account(home_fd, first)
    registry = tmp_path / "accounts.json"
    value = _read(registry)
    value["future-field"] = {"kept": True}
    _write(registry, value)

    with store._home() as home_fd:
        store._register_account(home_fd, second)

    assert _read(registry)["future-field"] == {"kept": True}


def test_mcp_oauth_token_preserves_unknown_fields(tmp_path: Path) -> None:
    token = MCPOAuthToken(
        access_token="first",
        refresh_token="refresh",
        expires_at=None,
        token_type="Bearer",
        scope=None,
        authorization_server="https://example.test",
        token_endpoint="https://example.test/token",
        authorization_endpoint="https://example.test/auth",
        client_id="client",
        client_secret=None,
        redirect_uri="http://127.0.0.1/callback",
        resource="https://example.test/resource",
    )
    save_token("server", token, home=tmp_path)
    path = token_store_path("server", tmp_path)
    value = _read(path)
    value["future-field"] = {"kept": True}
    _write(path, value)

    save_token("server", token, home=tmp_path)

    assert _read(path)["future-field"] == {"kept": True}


def test_project_and_account_keyed_records_preserve_future_records(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    first = registry.create_project("first", "scope")
    project_record = home / "projects" / first.project_id / "project.json"
    value = _read(project_record)
    value["lanes"] = {"future-lane": {"future-field": True}}
    _write(project_record, value)

    registry.create_project("second", "scope")

    assert _read(project_record)["lanes"] == {"future-lane": {"future-field": True}}

    store = AccountStore(home / "accounts-home")
    known = AccountId("anthropic", "known")
    with store._home() as home_fd:
        store._register_account(home_fd, known)
    accounts_path = home / "accounts-home" / "accounts.json"
    accounts = _read(accounts_path)
    account_records = accounts["accounts"]
    assert isinstance(account_records, dict)
    account_records["anthropic:future"] = {
        "provider": "anthropic",
        "name": "future",
        "future-field": True,
    }
    _write(accounts_path, accounts)
    with store._home() as home_fd:
        store._register_account(home_fd, AccountId("anthropic", "second"))

    future = _read(accounts_path)["accounts"]["anthropic:future"]
    assert future["future-field"] is True


def test_inbox_mutation_preserves_unknown_record_fields(tmp_path: Path) -> None:
    registry = ProjectRegistry(tmp_path / "projects")
    sender = registry.create_project("sender", "scope")
    target = registry.create_project("target", "scope")
    inbox = ProjectInbox(registry, sessions_root=tmp_path / "sessions")
    message_id = "1" * 32
    session_id = "2" * 32
    inbox.send(
        from_project=sender.project_id,
        from_session=session_id,
        to_project=target.project_id,
        kind="question",
        title="title",
        body="body",
        message_id=message_id,
    )
    path = (
        tmp_path
        / "projects"
        / target.project_id
        / "inbox"
        / "new"
        / f"{message_id}.json"
    )
    value = _read(path)
    value["future-field"] = {"kept": True}
    _write(path, value)

    claimed = inbox.claim(target.project_id, message_id, session_id)

    assert claimed is not None
    assert claimed["future-field"] == {"kept": True}
    claimed_path = path.parent.parent / "claimed" / path.name
    assert _read(claimed_path)["future-field"] == {"kept": True}


def test_shape_lock_serializes_two_process_publishers(tmp_path: Path) -> None:
    home = tmp_path / ".zeta"
    directory = tmp_path / "records"
    directory.mkdir()
    with open_writer(home):
        pass
    _write(directory / "state.json", {"known": 1})
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    writer_a = context.Process(
        target=_publish_shape_field,
        args=(str(home), str(directory), "future-field", True, entered, release),
    )
    writer_a.start()
    entered.wait()
    lock_path = home / ".storage-format-shapes.lock"
    lock_held = lock_path.exists()
    if lock_held:
        info = directory.stat()
        digest = hashlib.sha256(
            f"{info.st_dev}:{info.st_ino}:state.json".encode()
        ).digest()
        offset = int.from_bytes(digest[:8]) & ((1 << 63) - 1)
        lock_fd = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
        finally:
            os.close(lock_fd)
    writer_b = context.Process(
        target=_publish_shape_field,
        args=(str(home), str(directory), "known", 2),
    )
    writer_b.start()
    release.set()
    writer_a.join()
    writer_b.join()
    assert writer_a.exitcode == 0
    assert writer_b.exitcode == 0
    writer_a.close()
    writer_b.close()
    assert lock_held, "writer did not hold a cross-process shape lock"
    assert _read(directory / "state.json") == {"known": 2, "future-field": True}


@pytest.mark.asyncio
async def test_memory_slash_and_completed_reply_mutations_run_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread = threading.get_ident()
    calls: list[int] = []

    def command(*_args: object) -> str:
        calls.append(threading.get_ident())
        return "ok"

    monkeypatch.setattr("zeta.tui.slash_handlers.run_memory_command", command)
    monkeypatch.setattr("zeta.server.slash_commands.run_memory_command", command)
    registry = object()
    tui = SimpleNamespace(
        loop=SimpleNamespace(
            project_registry=registry,
            session_metadata=SimpleNamespace(project_id="p"),
            memory_reconciler=None,
        )
    )
    runtime = SimpleNamespace(
        metadata=SimpleNamespace(project_id="p"),
        manager=SimpleNamespace(project_registry=registry),
        loop=SimpleNamespace(memory_reconciler=None),
    )
    tui_result = SlashHandlerMixin.slash_memory(tui, "undo")
    if inspect.isawaitable(tui_result):
        tui_result = await tui_result
    assert tui_result == "ok"
    serve_result = ServerSlashSession(runtime).slash_memory("undo")
    if inspect.isawaitable(serve_result):
        serve_result = await serve_result
    assert serve_result == "ok"

    class Scanner:
        def mark_reported(self, _statuses: object) -> None:
            calls.append(threading.get_ident())

    owner = SimpleNamespace(_inbox_scanner=Scanner())
    mark_result = ProjectInboxNotificationMixin._record_sent_statuses(
        owner,
        ({"id": "1" * 32, "status": "done", "reply_id": "2" * 32},),
    )
    if inspect.isawaitable(mark_result):
        await mark_result

    assert len(calls) == 3
    assert all(thread_id != loop_thread for thread_id in calls)


def test_closed_variant_readers_reject_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / ".zeta"
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("alpha", "scope")
    project_path = home / "projects" / project.project_id / "project.json"
    project_value = _read(project_path)
    project_value["repo_agents"] = "future"
    _write(project_path, project_value)
    before = project_path.read_bytes()
    with pytest.raises(ProjectRegistryError):
        registry.show_project(project.project_id)
    assert project_path.read_bytes() == before
    project_value["repo_agents"] = "include"
    _write(project_path, project_value)

    sessions_path = home / "projects" / project.project_id / "sessions.jsonl"
    sessions_path.write_text(
        json.dumps(
            {
                "session_id": "1" * 32,
                "transcript_path": "session.jsonl",
                "role": "future",
                "parent_session_id": None,
                "recorded_at": "2026-01-01T00:00:00.000000Z",
            }
        )
        + "\n"
    )
    sessions_path.chmod(0o600)
    before = sessions_path.read_bytes()
    with pytest.raises(ProjectRegistryError):
        registry.list_session_links(project.project_id)
    assert sessions_path.read_bytes() == before

    account_home = home / "accounts"
    account_home.mkdir()
    account_path = account_home / "accounts.json"
    _write(
        account_path,
        {
            "version": 1,
            "accounts": {"future:account": {"provider": "future", "name": "account"}},
            "defaults": {},
            "labels": {},
        },
    )
    before = account_path.read_bytes()
    with pytest.raises(AccountError):
        AccountStore(account_home).list_catalog()
    assert account_path.read_bytes() == before

    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = "8" * 32
    inbox.send(
        from_project=project.project_id,
        from_session="9" * 32,
        to_project=project.project_id,
        kind="question",
        title="title",
        body="body",
        message_id=message_id,
    )
    message_path = (
        home / "projects" / project.project_id / "inbox" / "new" / f"{message_id}.json"
    )
    message = _read(message_path)
    message["kind"] = "future"
    _write(message_path, message)
    before = message_path.read_bytes()
    listed = inbox.list(project.project_id)
    assert listed["new"] == []
    assert len(listed["invalid"]) == 1
    assert message_path.read_bytes() == before

    ledger = home / "sessions" / ("2" * 32) / "project-inbox-sent.json"
    ledger.parent.mkdir(parents=True)
    _write(
        ledger,
        {
            "schema_version": 1,
            "records": [
                {
                    "id": "3" * 32,
                    "target_project": project.project_id,
                    "created_at": "2026-01-01T00:00:00.000000Z",
                    "reported": ["future"],
                }
            ],
        },
    )
    before = ledger.read_bytes()
    with pytest.raises(InboxError):
        SentMessageTracker(home / "sessions", "2" * 32)
    assert ledger.read_bytes() == before

    monkeypatch.setitem(
        CLOSED_VARIANTS,
        "project.session_role",
        CLOSED_VARIANTS["project.session_role"] | {"future"},
    )
    try:
        links = registry.list_session_links(project.project_id)
    except ProjectRegistryError:
        links = []
    assert links and links[0]["role"] == "future"


def test_slice_0b_registers_closed_variants_and_broad_subtrees() -> None:
    assert CLOSED_VARIANTS["project.schema_version"] == frozenset({1})
    assert CLOSED_VARIANTS["project.session_role"] == frozenset(
        {"session", "orchestrator", "worker"}
    )
    assert CLOSED_VARIANTS["project.repo_agents"] == frozenset({"include", "ignore"})
    assert CLOSED_VARIANTS["inbox.schema_version"] == frozenset({1})
    assert CLOSED_VARIANTS["inbox.kind"] == frozenset(
        {"bug_report", "change_request", "question", "info", "reply"}
    )
    assert CLOSED_VARIANTS["sent_message_ledger.schema_version"] == frozenset({1})
    assert CLOSED_VARIANTS["account_registry.version"] == frozenset({1})
    assert CLOSED_VARIANTS["account.provider"] == frozenset({"anthropic", "codex"})
    assert CLOSED_VARIANTS["sent_message_ledger.reported_status"] == frozenset(
        {"claimed", "done"}
    )
    assert CLOSED_VARIANTS["memory.format"] == frozenset({1, 2})
    assert BROAD_OWNED_SUBTREES["memory-current.json"] == frozenset({("history",)})
    with pytest.raises(StorageFormatError, match="unsupported stored variant"):
        require_closed_variant("inbox.schema_version", 2)
