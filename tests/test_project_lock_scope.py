from __future__ import annotations

import asyncio
import fcntl
import json
import socket
import subprocess
import sys
import threading
from collections.abc import Awaitable
from contextlib import asynccontextmanager, contextmanager
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from zeta.attention_records import AttentionStore, observed_epoch_for_session
from zeta.core.session import SessionManager
from zeta.core.session_links import (
    PENDING_CHILD_LINKS_DIRNAME,
    persist_pending_child_link,
    reconcile_child_links,
)
from zeta.core.store import ConversationStore
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.skills import SkillCatalog
from zeta.storage_format import (
    FEATURE_EPOCHS,
    P06_FEATURE_EPOCH,
    StorageFeature,
    advance,
)
from zeta.tui.app import TUIApp


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


def _send_inbox_message(home: str, registry: ProjectRegistry, project_id: str) -> str:
    return ProjectInbox(registry, sessions_root=Path(home) / "sessions").send(
        from_project=project_id,
        from_session="c" * 32,
        to_project=project_id,
        kind="info",
        title="p06-1",
        body="unrelated inbox publication",
    )


def _hold_inbox_publication(home: str, project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    original = registry._acquire_lock

    def gated(fd: int, mode: int) -> None:
        original(fd, mode)
        if mode == fcntl.LOCK_EX:
            _send(channel, "acquired")
            if _receive(channel) != "release":
                raise AssertionError("release signal was not sent")

    registry._acquire_lock = gated  # type: ignore[method-assign]
    _send_inbox_message(home, registry, project_id)
    _send(channel, "a-done")


def _publish_other_inbox(home: str, project_id: str, channel: socket.socket) -> None:
    registry = ProjectRegistry(Path(home) / "projects")
    message_id = _send_inbox_message(home, registry, project_id)
    listed = ProjectInbox(registry, sessions_root=Path(home) / "sessions").list(
        project_id
    )
    published = any(item["id"] == message_id for item in listed["new"])
    _send(channel, "b-done" if published else "b-missing")


_WORKERS = {
    "hold-inbox": _hold_inbox_publication,
    "publish-inbox": _publish_other_inbox,
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

@pytest.mark.parametrize(
    ("holder_kind", "other_kind"),
    [("hold-memory", "publish-memory"), ("hold-inbox", "publish-inbox")],
    ids=["memory", "inbox"],
)
def test_p06_1_unrelated_projects_publish_without_root_lock_serialization(
    tmp_path: Path, holder_kind: str, other_kind: str
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
        holder_kind=holder_kind,
        holder_project=project_a.project_id,
        other_kind=other_kind,
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
    for epoch in range(P06_FEATURE_EPOCH - 1):
        advance(home, expected_epoch=epoch, target_epoch=epoch + 1)
    legacy = ProjectRegistry(home / "projects")
    old_project = legacy.create_project("old", "old")
    legacy.initialize_memory(old_project.project_id)
    assert (home / "projects" / ".lock").is_file()
    assert not (home / "projects" / old_project.project_id / ".memory.lock").exists()

    del legacy
    advance(home, expected_epoch=P06_FEATURE_EPOCH - 1, target_epoch=P06_FEATURE_EPOCH)
    current = ProjectRegistry(home / "projects")
    current.load_memory(old_project.project_id)
    assert (home / "projects" / old_project.project_id / ".memory.lock").is_file()


@pytest.mark.asyncio
async def test_p06_below_epoch_real_tui_serve_journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal TUI, serve, inbox, and attention use stays on an epoch-5 home."""
    from tests.support.fake_backend import FakeBackend
    from tests.support.owned import await_or_fail, owned_app
    from tests.support.serve_close import close_serve
    from tests.test_server import _ready_extensions, _request
    from zeta.cli.main import main as cli_main

    def serve_result(
        response_frames: list[dict[str, object]], method: str, request_id: str
    ) -> dict[str, object]:
        response = response_frames[-1]
        assert "result" in response, f"{method} {request_id}: {response!r}"
        return response["result"]  # type: ignore[return-value]

    home = tmp_path / "home"
    for epoch in range(5):
        advance(home, expected_epoch=epoch, target_epoch=epoch + 1)
    project_dir = tmp_path / "project"
    await asyncio.to_thread(
        subprocess.run,
        ("git", "init", "--quiet", str(project_dir)),
        check=True,
        capture_output=True,
    )
    server = await asyncio.to_thread(
        ZetaServer, home=home, cwd=project_dir, port=0, provider="codex"
    )
    reader = writer = None
    opened = None
    app = None
    try:
        monkeypatch.setenv("ZETA_HOME", str(home))
        monkeypatch.chdir(project_dir)
        assert await asyncio.to_thread(
            cli_main, ["project", "init", str(project_dir)]
        ) == 0
        reader, writer, serve_session_id = await _ready_extensions(server)
        project_result = serve_result(
            await _request(
                reader,
                writer,
                "project-init",
                "slash_run",
                {"session_id": serve_session_id, "text": "/project"},
            ),
            "slash_run",
            "project-init",
        )
        assert "project" in project_result["text"].lower()
        registry = server.runtime.manager.project_registry
        assert registry is not None
        project = (await asyncio.to_thread(registry.list_projects))[0]
        assert project.name == project_dir.name
        # epoch 5 homes must keep the legacy root authority until cutover.
        assert (home / "projects" / ".lock").is_file()
        assert not (home / "projects" / project.project_id / ".memory.lock").exists()

        manager = SessionManager(home)
        opened = await asyncio.to_thread(
            manager.create,
            provider="codex",
            model="offline",
            cwd=project_dir,
            project_id=project.project_id,
            auto_project=False,
        )
        app = TUIApp(
            AgentLoop(
                FakeBackend([]),
                opened.store,
                skill_catalog=SkillCatalog.empty(),
                project_registry=manager.project_registry,
                root_project_id=project.project_id,
            ),
            provider="codex",
            model="offline",
            console=Console(file=StringIO(), force_terminal=True),
        )
        session = app._make_session()
        app._active_session = session
        app._install_full_screen_layout(session)
        async with owned_app(app):
            await await_or_fail(app.loop.activate(), what="below-epoch TUI activation")
            sent = await await_or_fail(
                app.loop.tool_registry.execute(
                    ToolCall(
                        "below-epoch-send",
                        "inbox",
                        {
                            "action": "send",
                            "project": project.project_id,
                            "kind": "question",
                            "title": "below epoch",
                            "body": "normal journey",
                        },
                    ),
                    _skip_approval=True,
                ),
                what="below-epoch inbox send",
            )
            assert sent.get("isError") is not True
            message_id = sent["structuredContent"]["id"]
            listed = await await_or_fail(
                app.loop.tool_registry.execute(
                    ToolCall("below-epoch-list", "inbox", {"action": "list"})
                ),
                what="below-epoch inbox list",
            )
            assert listed.get("isError") is not True
            assert any(
                message["id"] == message_id
                for message in listed["structuredContent"]["messages"]
            )

            record = await asyncio.to_thread(
                AttentionStore(
                    home / "sessions" / serve_session_id,
                    writer_epoch=observed_epoch_for_session(
                        home / "sessions" / serve_session_id
                    ),
                ).request,
                session_id=serve_session_id,
                project_id=project.project_id,
                entry_id=None,
                entry_seq=None,
                title="below epoch reply",
                why="normal journey",
                options=("yes",),
            )
            reply = serve_result(
                await _request(
                    reader,
                    writer,
                    "attention-reply",
                    "slash_run",
                    {
                        "session_id": serve_session_id,
                        "text": f"/decisions reply {record.id} yes",
                    },
                ),
                "slash_run",
                "attention-reply",
            )
            assert reply["text"] == f"replied to {record.id}"
            assert (
                await asyncio.to_thread(
                    lambda: AttentionStore(
                        home / "sessions" / serve_session_id,
                        writer_epoch=observed_epoch_for_session(
                            home / "sessions" / serve_session_id
                        ),
                    ).get(record.id).status
                )
                == "resolved"
            )
    finally:
        if opened is not None and app is None:
            opened.store.close()
        if opened is not None:
            opened.store.close()
        if writer is not None:
            await close_serve(server, writer)
        else:
            await close_serve(server)
    assert json.loads((home / "storage-format.json").read_text())["writer_epoch"] == 5


def _advance_to_project_locks(home: Path) -> None:
    for epoch in range(FEATURE_EPOCHS[StorageFeature.PROJECT_LOCKS]):
        advance(home, expected_epoch=epoch, target_epoch=epoch + 1)


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
    _advance_to_project_locks(home)
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
    _advance_to_project_locks(home)
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


async def test_p06_8_tui_workspace_lifetime_closes_and_deletes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # guard: the cached workspace store pins its session with a lifetime lease,
    # so it leaves no session-shape rank on the event-loop thread while it lives
    # (a short-shape pin makes the second checkpoint raise a rank inversion).
    from zeta.cli.main import build_parser
    from zeta.tui.app import create_app

    home = tmp_path / "home"
    await asyncio.to_thread(
        subprocess.run, ["git", "init", "-q", str(tmp_path)], check=True
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    app = await asyncio.to_thread(
        create_app, build_parser().parse_args(["--provider", "codex"])
    )
    session_id = app.loop.store.session_id
    try:
        # The first checkpoint caches the store; the second runs while it is alive.
        for label in ("first", "second"):
            result = await app._slash_commands.dispatch_async(
                app, f"/checkpoint {label}"
            )
            assert result is not None, label
            assert result.startswith(f"checkpoint '{label}'"), result
            assert "skipped" not in result, result
    finally:
        await app.close()
    SessionManager(home).delete(session_id)


def test_p06_installed_callbacks_drain_in_append_order_after_rank_release(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="installed-drain-order")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_returned = threading.Event()
    delivered: list[tuple[str, str]] = []

    def append(text: str) -> None:
        store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(text)])
        )

    def installed(delta: object) -> None:
        assert delta != "reset"
        for entry in delta:  # type: ignore[attr-defined]
            text = entry.data["message"]["content"][0]["text"]
            delivered.append((text, entry.id))
            if text == "a":
                first_entered.set()
                release_first.wait(timeout=10)
            elif text == "b":
                # Re-appending takes the execution and shape ranks again, so it
                # fails with an order inversion unless the drain runs unlocked.
                append("c")

    def append_b() -> None:
        append("b")
        second_returned.set()

    store.on_entries_installed = installed
    thread_a = threading.Thread(target=append, args=("a",))
    thread_b = threading.Thread(target=append_b)
    try:
        thread_a.start()
        assert first_entered.wait(timeout=10)
        thread_b.start()
        # B may return before its delta is delivered: A's drainer owns delivery.
        assert second_returned.wait(timeout=10)
        assert [text for text, _ in delivered] == ["a"], delivered
        release_first.set()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)
        assert not thread_a.is_alive() and not thread_b.is_alive()

        durable = [
            (row.data["message"]["content"][0]["text"], row.id)
            for row in store._entries
            if row.type == "message"
        ]
        assert delivered == durable
        assert [text for text, _ in delivered] == ["a", "b", "c"]
        assert len({entry_id for _, entry_id in delivered}) == 3
    finally:
        release_first.set()
        thread_a.join(timeout=10)
        if thread_b.ident is not None:
            thread_b.join(timeout=10)
        store.close()


def _held_authority(registry: ProjectRegistry, project_id: str, domain: str):
    if domain == "identity":
        identity_write = getattr(registry, "_identity_write", None)
        if identity_write is not None:
            return identity_write()
    else:
        project_lock = getattr(registry, "_project_lock", None)
        if project_lock is not None:
            return project_lock(project_id, domain, exclusive=True)
    return registry._locked(write=True)


@asynccontextmanager
async def _held_production_lock(
    monkeypatch: pytest.MonkeyPatch, home: Path, project_id: str, domain: str
):
    """Hold one real project authority in a thread until a loop sentinel runs.

    The sentinel is scheduled on the event loop when a public operation first
    finds the held lock contended. It records whether that lock wait was still
    pending, then lets the holder release.
    """
    loop = asyncio.get_running_loop()
    acquired = threading.Event()
    contended = threading.Event()
    wait_ended = threading.Event()
    sentinel_ran = threading.Event()
    observed: list[str] = []

    def sentinel() -> None:
        observed.append(
            "sentinel" if not wait_ended.is_set() else "sentinel after the lock wait"
        )
        sentinel_ran.set()

    def hold() -> None:
        try:
            registry = ProjectRegistry(home / "projects")
            with _held_authority(registry, project_id, domain):
                acquired.set()
                if contended.wait(10):
                    loop.call_soon_threadsafe(sentinel)
                    sentinel_ran.wait(10)
                observed.append("released")
        except Exception as exc:  # noqa: BLE001 - reported through observed
            observed.append(f"holder failed: {exc!r}")
        finally:
            acquired.set()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert await asyncio.to_thread(acquired.wait, 10), "holder did not start"
        assert observed == [], observed
        original = ProjectRegistry._acquire_lock

        def gated(lock_fd: int, mode: int) -> None:
            try:
                fcntl.flock(lock_fd, mode | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                contended.set()
            try:
                original(lock_fd, mode)
            finally:
                wait_ended.set()

        # test-contract: seam observe the first contended wait on the held lock
        monkeypatch.setattr(ProjectRegistry, "_acquire_lock", staticmethod(gated))
        yield observed
    finally:
        contended.set()
        sentinel_ran.set()
        await asyncio.to_thread(holder.join, 10)
        assert not holder.is_alive(), "lock holder did not exit"


async def _settle[T](operation: Awaitable[T]) -> T | BaseException:
    try:
        return await asyncio.wait_for(operation, 30)
    except Exception as exc:  # noqa: BLE001 - the sentinel order is asserted first
        return exc


_LOOP_BLOCKED = "loop sentinel did not run before the held production lock was released"


def _git_project(tmp_path: Path, home: Path) -> tuple[Path, str]:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("demo", "demo", repository)
    registry.initialize_memory(project.project_id)
    return repository, project.project_id


async def test_p06_2_child_lineage_stays_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.fake_backend import FakeBackend
    from zeta.core.abort import AbortGenerationRegistry
    from zeta.protocol.types import ToolCall
    from zeta.runtime.loop import AgentLoop
    from zeta.skills import SkillCatalog

    home = tmp_path / ".zeta"
    repository, project_id = await asyncio.to_thread(_git_project, tmp_path, home)
    registry = ProjectRegistry(home / "projects")
    opened = await asyncio.to_thread(
        SessionManager(home).create, provider="codex", model="fake", cwd=repository
    )
    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=project_id,
        project_registry=registry,
    )

    async def stop_after_publication(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("stop after lineage publication")

    # test-contract: seam stop the real spawn right after lineage publication
    monkeypatch.setattr(
        loop.store, "append_child_run_started_async", stop_after_publication
    )
    call = ToolCall("child", "agent", {"prompt": "work", "description": "child"})
    try:
        async with _held_production_lock(
            monkeypatch, home, project_id, "sessions"
        ) as observed:
            outcome = await _settle(
                loop._run_agent_tool(
                    call, call.arguments, AbortGenerationRegistry().new_generation(), None
                )
            )
        assert observed == ["sentinel", "released"], _LOOP_BLOCKED
        assert isinstance(outcome, RuntimeError), repr(outcome)
        links = await asyncio.to_thread(registry.list_session_links, project_id)
        assert f"{opened.store.session_id}:1" in {
            row["session_id"] for row in links
        }
    finally:
        try:
            await asyncio.wait_for(loop.close(), 10)
        finally:
            opened.store.close()


def _tool_registry(home: Path, project_id: str):
    from zeta.skills import SkillCatalog
    from zeta.tools.registry import ToolRegistry

    store = ConversationStore(home / "sessions", session_id="a" * 32, cwd=home)
    tools = ToolRegistry(
        home,
        skill_catalog=SkillCatalog.empty(),
        project_id=project_id,
        project_registry=ProjectRegistry(home / "projects"),
        inbox_enabled=True,
    )
    tools.bind_session_store(store)
    return tools, store


async def _tool_stays_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    arguments: dict[str, object],
    domain: str,
) -> None:
    from zeta.protocol.types import ToolCall

    home = tmp_path / ".zeta"
    _, project_id = await asyncio.to_thread(_git_project, tmp_path, home)
    tools, store = await asyncio.to_thread(_tool_registry, home, project_id)
    try:
        async with _held_production_lock(
            monkeypatch, home, project_id, domain
        ) as observed:
            result = await _settle(tools.execute(ToolCall("call", tool, arguments)))
        assert observed == ["sentinel", "released"], _LOOP_BLOCKED
        assert isinstance(result, dict), repr(result)
        assert result["isError"] is False, result
        return result["structuredContent"]
    finally:
        await tools.close()
        store.close()


async def test_p06_2_project_tool_stays_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = await _tool_stays_off_event_loop(
        tmp_path, monkeypatch, "project", {"action": "inspect"}, "memory"
    )
    assert content["project"]["name"] == "demo"
    assert "memory" in content


async def test_p06_2_inbox_tool_stays_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = await _tool_stays_off_event_loop(
        tmp_path, monkeypatch, "inbox", {"action": "list"}, "inbox"
    )
    assert content["status"] == "new"
    assert content["messages"] == []


_SLASH_CASES = {
    "/project": "identity",
    "/memory": "memory",
    "/inbox": "inbox",
}


def _assert_slash_output(command: str, output: object) -> None:
    assert isinstance(output, str), repr(output)
    if command == "/project":
        assert "demo" in output, output
    elif command == "/memory":
        assert not output.startswith("memory:"), output
    else:
        assert json.loads(output)["new"] == [], output


@pytest.mark.parametrize("command", list(_SLASH_CASES))
async def test_p06_2_tui_project_surfaces_stay_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    from zeta.cli.main import build_parser
    from zeta.tui.app import create_app

    home = tmp_path / ".zeta"
    repository, project_id = await asyncio.to_thread(_git_project, tmp_path, home)
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(repository)
    app = await asyncio.to_thread(
        create_app, build_parser().parse_args(["--provider", "codex"])
    )
    try:
        assert app.loop.session_metadata.project_id == project_id
        async with _held_production_lock(
            monkeypatch, home, project_id, _SLASH_CASES[command]
        ) as observed:
            output = await _settle(app._slash_commands.dispatch_async(app, command))
        assert observed == ["sentinel", "released"], _LOOP_BLOCKED
        _assert_slash_output(command, output)
    finally:
        await asyncio.wait_for(app.close(), 30)


@pytest.mark.parametrize("command", ["/memory", "/inbox"])
async def test_p06_2_serve_project_surfaces_stay_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    from tests.support.serve_close import close_serve
    from tests.support.server_client import request
    from zeta.server import ZetaServer

    home = tmp_path / "home"
    repository, project_id = await asyncio.to_thread(_git_project, tmp_path, home)
    server = ZetaServer(home=home, cwd=repository, port=0, provider="codex")
    await asyncio.wait_for(server.start(), 30)
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", server.port), 30
    )
    try:
        await request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.0", "client_version": "1.1"},
        )
        session = (
            await request(reader, writer, 2, "new_session", {"provider": "codex"})
        )[-1]["result"]["session"]
        assert session["project_id"] == project_id, session
        async with _held_production_lock(
            monkeypatch, home, project_id, _SLASH_CASES[command]
        ) as observed:
            frames = await _settle(
                request(
                    reader,
                    writer,
                    3,
                    "slash_run",
                    {"session_id": session["session_id"], "text": command},
                )
            )
        assert observed == ["sentinel", "released"], _LOOP_BLOCKED
        assert isinstance(frames, list), repr(frames)
        result = frames[-1]["result"]
        assert result["kind"] == "output", result
        _assert_slash_output(command, result["text"])
    finally:
        await close_serve(server, writer)
