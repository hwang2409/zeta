from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import zeta.core.session as session_module
import zeta.core.session_files as session_files_module
import zeta.core.session_links as session_links_module
import zeta.project_inbox as project_inbox_module
import zeta.project_registry as project_registry_module
import zeta.remote_sync as remote_sync_module
from tests.support.owned import await_or_fail
from zeta.attention_forks import create_discussion_fork
from zeta.attention_records import AttentionStore
from zeta.core.session import SessionManager
from zeta.core.session_files import SessionReferenceGate, SessionReferenceUnavailable
from zeta.core.session_links import (
    persist_pending_child_link,
    persist_pending_root_link,
    reconcile_child_links,
)
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.remote_sync import LocalTransport, RemoteSyncError, pull_session, push_session


@dataclass
class Operation:
    home: Path
    run: Callable[[], Any]
    unavailable_id: str
    on_loop_na: str | None = "public operation is offloaded before lock admission"
    close: Callable[[], None] = lambda: None


def _set_epoch(home: Path, epoch: int) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "storage-format.json").write_text(
        f'{{"writer_epoch":{epoch}}}\n', encoding="utf-8"
    )


def _create(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    manager = SessionManager(home)

    def run() -> str:
        opened = manager.create(
            provider="codex", model="fake", cwd=home, parent_session_id="p" * 32
        )
        try:
            return opened.store.session_id
        finally:
            opened.store.close()

    return Operation(home, run, "p" * 32)


def _open(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="fake", cwd=home)
    session_id = opened.store.session_id
    opened.store.close()

    def run() -> str:
        reopened = manager.open(session_id)
        try:
            return reopened.store.session_id
        finally:
            reopened.store.close()

    return Operation(home, run, session_id)


def _associate(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("route", "route test")
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="fake", cwd=home)
    metadata = opened.metadata
    return Operation(
        home,
        lambda: manager.associate_project(metadata, project.project_id).project_id,
        metadata.session_id,
        close=opened.store.close,
    )


def _inbox_send(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    source = registry.create_project("source", "source")
    target = registry.create_project("target", "target")
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    return Operation(
        home,
        lambda: inbox.send_with_outcome(
            from_project=source.project_id,
            from_session="a" * 32,
            to_project=target.project_id,
            kind="info",
            title="route",
            body="body",
        )[0],
        "a" * 32,
    )


def _inbox_claim(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    source = registry.create_project("source", "source")
    target = registry.create_project("target", "target")
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send(
        from_project=source.project_id,
        from_session="a" * 32,
        to_project=target.project_id,
        kind="info",
        title="route",
        body="body",
    )
    return Operation(
        home,
        lambda: inbox.claim(target.project_id, message_id, "b" * 32),
        "b" * 32,
    )


def _registry(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("route", "route test")
    return Operation(
        home,
        lambda: registry.record_session(
            project.project_id,
            session_id="a" * 32,
            transcript_path="sessions/" + "a" * 32,
        ),
        "a" * 32,
        on_loop_na=None,
    )


def _root_reconciliation(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("route", "route test")
    manager = SessionManager(home)
    opened = manager.create(provider="codex", model="fake", cwd=home)
    session_id = opened.store.session_id
    opened.store.close()
    persist_pending_root_link(
        manager.sessions_dir,
        session_id,
        project.project_id,
        "session",
        None,
        manager._storage_lease,
    )

    def run() -> str:
        reopened = manager.open(session_id)
        try:
            return reopened.store.session_id
        finally:
            reopened.store.close()

    return Operation(home, run, session_id)


def _child_reconciliation(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("route", "route test")
    manager = SessionManager(home)
    opened = manager.create(
        provider="codex", model="fake", cwd=home, project_id=project.project_id
    )
    root_id = opened.store.session_id
    opened.store.close()
    persist_pending_child_link(
        manager.sessions_dir / root_id,
        {
            "schema_version": 1,
            "session_id": "child:1",
            "project_id": project.project_id,
            "role": "worker",
            "parent_session_id": root_id,
            "transcript_path": f"sessions/{root_id}/agents/1/conversation.jsonl",
        },
        manager._storage_lease,
    )
    return Operation(
        home,
        lambda: reconcile_child_links(registry, manager.sessions_dir, root_id),
        root_id,
        on_loop_na=None,
    )


def _discussion_fork(home: Path, epoch: int) -> Operation:
    _set_epoch(home, epoch)
    registry = ProjectRegistry(home / "projects")
    project = registry.create_project("route", "route test")
    manager = SessionManager(home)
    opened = manager.create(
        provider="codex", model="fake", cwd=home, project_id=project.project_id
    )
    anchor = opened.store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("choose")])
    )
    attention = AttentionStore(opened.store.session_dir, writer_epoch=epoch).request(
        session_id=opened.store.session_id,
        project_id=project.project_id,
        entry_id=anchor.id,
        entry_seq=anchor.seq,
        title="route",
        why="route test",
    )
    session_id = opened.store.session_id
    opened.store.close()
    return Operation(
        home,
        lambda: create_discussion_fork(home, session_id, attention.id),
        session_id,
    )


def _push(home: Path, epoch: int) -> Operation:
    source = home.parent / f"{home.name}-source"
    _set_epoch(source, epoch)
    _set_epoch(home, epoch)
    opened = SessionManager(source).create(provider="codex", model="fake", cwd=source)
    session_id = opened.store.session_id
    opened.store.close()
    transport = LocalTransport(home)
    return Operation(
        home,
        lambda: push_session(source, transport, session_id=session_id).session_id,
        session_id,
    )


def _pull(home: Path, epoch: int) -> Operation:
    source = home.parent / f"{home.name}-source"
    remote = home.parent / f"{home.name}-remote"
    _set_epoch(source, epoch)
    _set_epoch(remote, epoch)
    _set_epoch(home, epoch)
    opened = SessionManager(source).create(provider="codex", model="fake", cwd=source)
    session_id = opened.store.session_id
    opened.store.close()
    push_session(source, LocalTransport(remote), session_id=session_id)
    return Operation(
        home,
        lambda: pull_session(home, LocalTransport(remote), session_id=session_id).session_id,
        session_id,
    )


_ROUTES = (
    ("create", _create),
    ("writable-open", _open),
    ("associate-project", _associate),
    ("inbox-send", _inbox_send),
    ("inbox-claim", _inbox_claim),
    ("discussion-fork", _discussion_fork),
    ("local-publish", _push),
    ("local-pull", _pull),
    ("project-registry", _registry),
    ("root-reconciliation", _root_reconciliation),
    ("child-reconciliation", _child_reconciliation),
)


def _patch_reference_publication(monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
    real = session_files_module.reference_publication

    @contextmanager
    def counted(*args: Any, **kwargs: Any) -> Iterator[Any]:
        calls.append(1)
        assert len(calls) == 1, "public operation entered reference publication more than once"
        with real(*args, **kwargs) as authority:
            yield authority

    for module in (
        session_files_module,
        session_module,
        session_links_module,
        project_inbox_module,
        project_registry_module,
        remote_sync_module,
    ):
        monkeypatch.setattr(module, "reference_publication", counted)


def _published_files(home: Path) -> dict[str, bytes]:
    ignored = {
        ".fork-allocation.lock",
        "runtime.lease",
        "writer.lease",
        "session-reference-generation",
    }
    return {
        str(path.relative_to(home)): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
        and path.name not in ignored
        and ".domain-locks" not in path.parts
    }


@pytest.mark.parametrize(("name", "setup"), _ROUTES, ids=[row[0] for row in _ROUTES])
def test_public_reference_writers_enter_once_refuse_without_partial_publication_and_stay_dormant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    setup: Callable[[Path, int], Operation],
) -> None:
    refused = setup(tmp_path / f"{name}-refused", 8)
    try:
        reservation = (
            refused.home
            / "session-trash"
            / "reservations"
            / f"{refused.unavailable_id}.json"
        )
        with SessionReferenceGate(
            refused.home, SessionManager(refused.home)._storage_lease
        ).exclusive(session_id=name):
            reservation.parent.mkdir(parents=True, exist_ok=True)
            reservation.write_text("{}\n", encoding="utf-8")
        before = _published_files(refused.home)
        calls = []
        with monkeypatch.context() as patch:
            _patch_reference_publication(patch, calls)
            with pytest.raises(
                (SessionReferenceUnavailable, RemoteSyncError),
                match="session reference unavailable",
            ):
                refused.run()
        assert len(calls) == 1
        assert _published_files(refused.home) == before, f"{name} published after refusal"
    finally:
        refused.close()

    success = setup(tmp_path / f"{name}-success", 8)
    try:
        calls = []
        with monkeypatch.context() as patch:
            _patch_reference_publication(patch, calls)
            success.run()
        assert len(calls) == 1, f"{name} entered reference publication {len(calls)} times"
    finally:
        success.close()

    if refused.on_loop_na is None:
        contended = setup(tmp_path / f"{name}-contended", 8)
        acquired = threading.Event()
        release = threading.Event()

        def hold_exclusive() -> None:
            manager = SessionManager(contended.home)
            with SessionReferenceGate(
                contended.home, manager._storage_lease
            ).exclusive(session_id=name):
                acquired.set()
                assert release.wait(10), "reference holder was not released"

        holder = threading.Thread(target=hold_exclusive)
        holder.start()
        try:
            assert acquired.wait(10), "reference holder did not acquire the gate"

            async def run_on_loop() -> None:
                with pytest.raises(SessionReferenceUnavailable):
                    contended.run()

            async def check_on_loop() -> None:
                await await_or_fail(
                    run_on_loop(), what=f"{name} on-loop reference refusal"
                )

            asyncio.run(check_on_loop())
        finally:
            release.set()
            holder.join(10)
            assert not holder.is_alive(), "reference holder did not stop"
            contended.close()

    dormant = setup(tmp_path / f"{name}-dormant", 7)
    try:
        dormant.run()
        assert not (dormant.home / "session-reference-generation").exists()
        assert not (dormant.home / "session-trash").exists()
    finally:
        dormant.close()
