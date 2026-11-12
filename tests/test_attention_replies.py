from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta import attention_replies
from zeta.attention_records import AttentionRecord, AttentionStore, Broken
from zeta.attention_replies import AttentionReplies, ReplyOutcome
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.project_inbox import ProjectInbox, SentMessageTracker
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.project_inbox import ProjectInboxNotificationMixin
from zeta.server import ZetaServer
from zeta.skills import SkillCatalog


@dataclass
class ReplyFixture:
    home: Path
    session_id: str
    project_id: str
    answerer_session_id: str
    records: list[AttentionRecord]

    @property
    def replies(self) -> Any:
        return AttentionReplies(self.home, self.session_id)

    @property
    def attention_dir(self) -> Path:
        return self.home / "sessions" / self.session_id / "attention"

    def inbox_messages(self) -> list[dict[str, Any]]:
        inbox = ProjectInbox(
            SessionManager(self.home).project_registry,
            sessions_root=self.home / "sessions",
        )
        return inbox.list(self.project_id, session_id=self.session_id)["new"]


def _fixture(home: Path, count: int = 3) -> ReplyFixture:
    manager = SessionManager(home)
    project = manager.project_registry.create_project("alpha", "Alpha")
    session = manager.create(
        provider="codex",
        model="fake",
        cwd=home,
        project_id=project.project_id,
        auto_project=False,
    )
    answerer = manager.create(
        provider="codex",
        model="fake",
        cwd=home,
        project_id=project.project_id,
        auto_project=False,
    )
    store = AttentionStore(session.store.session_dir, writer_epoch=5)
    records = [
        store.request(
            session_id=session.store.session_id,
            project_id=project.project_id,
            entry_id=None,
            entry_seq=None,
            title=f"choice {index}",
            why="test the durable reply path",
        )
        for index in range(count)
    ]
    session.store.close()
    answerer.store.close()
    return ReplyFixture(
        home,
        session.store.session_id,
        project.project_id,
        answerer.store.session_id,
        records,
    )


def test_invalid_attention_name_is_listed_without_reading_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=1)
    path = fixture.attention_dir / "junk.json"
    path.write_bytes(b"must not be read")
    long_path = fixture.attention_dir / ("x" * 200 + ".json")
    long_path.write_bytes(b"also must not be read")

    listed = AttentionStore(tmp_path / "sessions" / fixture.session_id, writer_epoch=5).list()

    broken = next(item for item in listed if isinstance(item, Broken))
    assert broken.id == "junk.json"
    assert broken.path == "attention/junk.json"
    assert broken.reason == "invalid_name"
    assert broken.bytes == b""
    long_broken = next(
        item
        for item in listed
        if isinstance(item, Broken)
        and item.path
        == long_path.relative_to(
            fixture.home / "sessions" / fixture.session_id
        ).as_posix()
    )
    assert isinstance(long_broken, Broken)
    assert len(long_broken.id) == 160
    assert long_broken.id.endswith("...")
    assert path.read_bytes() == b"must not be read"
    assert long_path.read_bytes() == b"also must not be read"
    assert any(
        isinstance(item, AttentionRecord) and item.id == fixture.records[0].id
        for item in listed
    )


def test_broken_attention_is_listed_and_cannot_be_replied_or_deleted(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, count=1)
    path = fixture.attention_dir / f"{fixture.records[0].id}.json"
    original = path.read_bytes()
    path.write_bytes(b"{")

    listed = AttentionStore(tmp_path / "sessions" / fixture.session_id, writer_epoch=5).list()
    assert len(listed) == 1
    broken = listed[0]
    assert isinstance(broken, Broken)
    assert broken.id == fixture.records[0].id
    assert broken.path == f"attention/{broken.id}.json"
    assert broken.reason == "invalid_json"
    assert broken.bytes == b"{"
    with pytest.raises(attention_replies.ReplyConflict, match="invalid_json"):
        fixture.replies.dismiss(broken.id)
    assert path.read_bytes() == b"{"
    assert original != path.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["agent", "serve"])
async def test_broken_sealed_journal_does_not_discard_loaded_session(
    surface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = await asyncio.to_thread(_fixture, tmp_path, 1)
    journal = (
        await asyncio.to_thread(
            _leave_sealed_journals,
            fixture,
            fixture.replies,
            monkeypatch,
            [(fixture.records[0], "answer")],
        )
    )[0]
    record_path = fixture.attention_dir / f"{fixture.records[0].id}.json"
    await asyncio.to_thread(record_path.write_bytes, b"{")
    failure: attention_replies.ReplyConflict | None = None

    if surface == "agent":
        store = await asyncio.to_thread(
            ConversationStore,
            fixture.home / "sessions",
            session_id=fixture.session_id,
        )
        loop = AgentLoop(
            FakeBackend([]),
            store,
            skill_catalog=SkillCatalog.empty(),
            skip_mcp_mount=True,
        )
        try:
            try:
                await loop.activate()
            # Capture the old failure so fail-first reaches the behavior assertion.
            except attention_replies.ReplyConflict as exc:
                failure = exc
            assert failure is None, f"agent activation discarded session: {failure!r}"
            assert loop.store.session_id == fixture.session_id
        finally:
            await loop.close()
            store.close()
    else:
        server = ZetaServer(
            provider="codex",
            home=fixture.home,
            cwd=fixture.home,
            socket_path=tmp_path / "zeta.sock",
            backend_factory=lambda provider, model, home: (
                FakeBackend([]),
                model or "offline",
            ),
        )
        try:
            try:
                await server.runtime.resume_session(fixture.session_id)
            # Capture the old failure so fail-first reaches the behavior assertion.
            except attention_replies.ReplyConflict as exc:
                failure = exc
            assert failure is None, f"serve load discarded session: {failure!r}"
            assert server.runtime.opened is not None
            assert server.runtime.opened.metadata.session_id == fixture.session_id
        finally:
            await server.close()

    listed = AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5).list()
    assert any(
        isinstance(record, Broken) and record.id == fixture.records[0].id
        for record in listed
    )
    assert journal.is_dir()
    assert record_path.read_bytes() == b"{"


def _make_information_item(fixture: ReplyFixture, record: AttentionRecord) -> None:
    path = fixture.attention_dir / f"{record.id}.json"
    value = path.read_text(encoding="utf-8").replace(
        '"status":"open"', '"kind":"answer","status":"open"'
    )
    path.write_text(value, encoding="utf-8")


def _stage_all(fixture: ReplyFixture, replies: list[str] | None = None) -> list[str]:
    values = replies or [f"answer {index}" for index in range(len(fixture.records))]
    for record, reply in zip(fixture.records, values, strict=True):
        fixture.replies.stage(record.id, reply)
    return values


def _assert_resolved(fixture: ReplyFixture, replies: list[str]) -> None:
    store = AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
    assert [store.get(record.id).decision for record in fixture.records] == replies
    assert all(store.get(record.id).status == "resolved" for record in fixture.records)


def _assert_no_live_journal(fixture: ReplyFixture) -> None:
    assert not list(fixture.attention_dir.glob("sending-*-*"))
    assert not list(fixture.attention_dir.glob("done-*-*"))
    assert not list(fixture.attention_dir.glob(".batch-*"))


def _leave_sealed_journals(
    fixture: ReplyFixture,
    replies: AttentionReplies,
    monkeypatch: pytest.MonkeyPatch,
    pairs: list[tuple[AttentionRecord, str]],
) -> tuple[Path, ...]:
    held: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def hold_seal(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        destination_path = Path(destination)
        if destination_path.name.startswith("sending-"):
            held_path = fixture.attention_dir / f"held-{len(held) + 1}"
            real_replace(source, held_path, *args, **kwargs)
            held.append((held_path, destination_path))
            raise RuntimeError("leave journal sealed")
        real_replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", hold_seal)
        for record, reply in pairs:
            with pytest.raises(RuntimeError, match="leave journal sealed"):
                replies.stage_and_flush(
                    [(record.id, reply)], fixture.answerer_session_id
                )
    for held_path, destination in held:
        held_path.rename(destination)
    return tuple(destination for _held_path, destination in held)


def _fail_once(kind: str) -> BaseException:
    if kind == "cancel":
        return asyncio.CancelledError()
    return RuntimeError("injected crash")


def test_popup_stage_syncs_contents_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, count=1)
    events: list[tuple[str, str]] = []
    real_sync = AttentionReplies._sync_file
    real_replace = os.replace

    def sync(path: Path) -> None:
        events.append(("sync", path.name))
        real_sync(path)

    def replace(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        if Path(destination).parent.name == "staged":
            events.append(("rename", Path(destination).name))
        real_replace(source, destination, *args, **kwargs)

    # test-contract: seam record the file barrier order around atomic staging
    monkeypatch.setattr(AttentionReplies, "_sync_file", staticmethod(sync))
    monkeypatch.setattr(os, "replace", replace)
    fixture.replies.stage(fixture.records[0].id, "durable answer")
    rename_index = next(
        index for index, event in enumerate(events) if event[0] == "rename"
    )
    # test-contract: durability verify the private file barrier precedes atomic staging rename
    assert any(kind == "sync" for kind, _name in events[:rename_index])


@pytest.mark.parametrize("case", ["batch", "same-id-conflict"])
def test_attention_reply_behavior_on_pre_batch_surface(
    case: str, tmp_path: Path
) -> None:
    """Keep behavior-level fail-first checks runnable before the batch module exists."""
    fixture = _fixture(tmp_path, count=2)
    if case == "batch":
        _stage_all(fixture, ["first", "second"])
        fixture.replies.flush(fixture.session_id)
        assert len(fixture.inbox_messages()) == 1
        return

    record = fixture.records[0]
    fixture.replies.stage(record.id, "popup answer")
    with pytest.raises(ValueError, match="TUI decisions popup"):
        fixture.replies.reply(record.id, "different answer", fixture.session_id)


_OPERATIONS = (
    "stage",
    "flush",
    "reply",
    "dismiss",
    "stage_and_flush",
    "resolve_local",
    "recover",
)
_OPERATION_ROWS = [
    f"operation-{operation}-{scenario}"
    for operation in _OPERATIONS
    for scenario in ("success", "duplicate")
]
_CONCURRENT_ROWS = [
    f"concurrent-seal-{operation}-{selection}"
    for operation in _OPERATIONS
    for selection in ("unrelated", "selected")
]
_PUBLICATION_ROWS = (
    "publication-success",
    "publication-exception-after",
    "publication-cancel-after",
    "publication-duplicate",
    "publication-recovery",
    "publication-none",
    "publication-callback-error",
)


@pytest.mark.asyncio
async def test_publication_callback_only_signals_watcher_on_event_loop() -> None:
    class Owner(ProjectInboxNotificationMixin):
        def __init__(self) -> None:
            self._project_inbox_owner_loop = asyncio.get_running_loop()
            self._project_inbox_publication_event = asyncio.Event()
            # mid-turn, so the 2-second idle tick is suppressed and only the
            # threadsafe signal drives a poll; the test then fails if the
            # callback stops signalling the watcher.
            self._turn_active = True
            self.calls: list[int] = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def _poll_project_inbox_once(self) -> None:
            self.calls.append(threading.get_ident())
            self.started.set()
            await asyncio.wait_for(self.release.wait(), 10)
            self.finished.set()

    owner = Owner()
    watcher = asyncio.create_task(owner._project_inbox_supervisor())
    notifier_thread = 0

    def publish_burst() -> None:
        nonlocal notifier_thread
        notifier_thread = threading.get_ident()
        owner.notify_project_inbox_publication()

    try:
        await await_or_fail(asyncio.to_thread(publish_burst), what="signal publication")
        await await_or_fail(owner.started.wait(), what="watcher inbox check start")
        assert notifier_thread != threading.get_ident()
        assert owner.calls == [threading.get_ident()]
        owner.release.set()
        await await_or_fail(owner.finished.wait(), what="watcher inbox check finish")
        await asyncio.sleep(0)
        assert len(owner.calls) == 1
    finally:
        owner.release.set()
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(watcher, 10)


_FAULT_ROWS = [
    f"{step}-{kind}"
    for step in (
        "stage",
        "seal-before",
        "seal-after",
        "send-before",
        "send-after",
        "resolve-first",
        "resolve-between",
        "cleanup-rename-before",
        "cleanup-rename-after",
        "cleanup-remove",
    )
    for kind in ("exception", "cancel")
]


@pytest.mark.parametrize(
    "case",
    [
        "success",
        *_PUBLICATION_ROWS,
        *_OPERATION_ROWS,
        *_CONCURRENT_ROWS,
        *_FAULT_ROWS,
        "publish-body",
        "publish-record",
        "publish-tracker",
        "duplicates",
        "same-id-conflicts",
        "multiple-journals",
        "broken-journal-recovery",
        "dismiss-information",
    ],
)
def test_attention_replies_public_table(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise durable reply outcomes only through the public AttentionReplies API."""
    fixture = _fixture(tmp_path, count=3)
    publications: list[bool] = []
    replies = AttentionReplies(
        fixture.home,
        fixture.session_id,
        on_inbox_visible=lambda: publications.append(True),
    )

    if case.startswith("publication-"):
        target = fixture.records[0]
        if case == "publication-none":
            replies.stage(target.id, "answer")
            assert publications == []
            return
        if case == "publication-recovery":
            sealer = AttentionReplies(fixture.home, fixture.session_id)

            def stop_before_send(_journal: Path) -> None:
                raise RuntimeError("leave journal sealed")

            # test-contract: seam leave sealed work for public recovery
            monkeypatch.setattr(sealer, "_send_journal", stop_before_send)
            with pytest.raises(RuntimeError, match="leave journal sealed"):
                sealer.reply(target.id, "answer", fixture.answerer_session_id)
            replies.recover()
            assert len(publications) == 1
            return
        if case == "publication-callback-error":
            failing = AttentionReplies(
                fixture.home,
                fixture.session_id,
                on_inbox_visible=lambda: (_ for _ in ()).throw(
                    RuntimeError("wake unavailable")
                ),
            )
            outcome = failing.reply(target.id, "answer", fixture.answerer_session_id)
            assert outcome.message_id is not None
            assert len(fixture.inbox_messages()) == 1
            return
        if case in {"publication-exception-after", "publication-cancel-after"}:
            failure = (
                asyncio.CancelledError()
                if case == "publication-cancel-after"
                else RuntimeError("tracker failed")
            )

            def fail_tracker(_tracker: SentMessageTracker, *_args: Any) -> None:
                raise failure

            # test-contract: seam fail after the inbox record becomes visible
            monkeypatch.setattr(SentMessageTracker, "record", fail_tracker)
            with pytest.raises((RuntimeError, asyncio.CancelledError)):
                replies.reply(target.id, "answer", fixture.answerer_session_id)
            assert len(publications) == 1
            assert len(fixture.inbox_messages()) == 1
            return
        if case == "publication-duplicate":
            real_finish = replies._finish_journal_locked
            failed = False

            def fail_after_send(journal: Path) -> None:
                nonlocal failed
                if not failed:
                    failed = True
                    raise RuntimeError("finish failed")
                real_finish(journal)

            # test-contract: seam retain a journal after its first durable send
            monkeypatch.setattr(replies, "_finish_journal_locked", fail_after_send)
            with pytest.raises(RuntimeError, match="finish failed"):
                replies.reply(target.id, "answer", fixture.answerer_session_id)
            replies.recover()
            assert len(publications) == 2
            assert len(fixture.inbox_messages()) == 1
            return
        assert case == "publication-success"
        replies.reply(target.id, "answer", fixture.answerer_session_id)
        assert len(publications) == 1
        assert len(fixture.inbox_messages()) == 1
        return

    if case.startswith("operation-"):
        operation, scenario = case.removeprefix("operation-").rsplit("-", 1)
        target = fixture.records[1]
        if operation == "flush":
            replies.stage(target.id, "target")
        elif operation == "dismiss":
            _make_information_item(fixture, target)
        elif operation == "recover":
            sealer = AttentionReplies(fixture.home, fixture.session_id)

            def stop_after_seal(_journal: Path) -> None:
                raise RuntimeError("leave journal sealed")

            sealer._send_journal = stop_after_seal  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match="leave journal sealed"):
                sealer.reply(target.id, "target", fixture.answerer_session_id)

        def run_operation() -> Any:
            if operation == "stage":
                return replies.stage(target.id, "target")
            if operation == "flush":
                return replies.flush(fixture.answerer_session_id)
            if operation == "reply":
                return replies.reply(target.id, "target", fixture.answerer_session_id)
            if operation == "dismiss":
                return replies.dismiss(target.id)
            if operation == "stage_and_flush":
                return replies.stage_and_flush(
                    [(target.id, "target")], fixture.answerer_session_id
                )
            if operation == "resolve_local":
                return replies.resolve_local(target.id, "target")
            assert operation == "recover"
            return replies.recover()

        outcome = run_operation()
        publishes = operation in {"flush", "reply", "stage_and_flush", "recover"}
        assert len(publications) == int(publishes)
        if scenario == "duplicate":
            duplicate = run_operation()
            assert len(publications) == int(publishes)
            if duplicate.items:
                assert all(
                    item.status in {"duplicate", "already-resolved"}
                    for item in duplicate.items
                )
        return

    if case.startswith("concurrent-seal-"):
        operation, selection = case.removeprefix("concurrent-seal-").rsplit("-", 1)
        sealed = fixture.records[0]
        target = sealed if selection == "selected" else fixture.records[1]
        if operation == "flush":
            target = fixture.records[1]
            replies.stage(target.id, "target")
        if operation == "dismiss":
            _make_information_item(fixture, target)
        real_lock = replies._lock
        injected = False

        @contextmanager
        def seal_before_lock() -> Any:
            nonlocal injected
            if not injected:
                injected = True
                sealer = AttentionReplies(fixture.home, fixture.session_id)

                def stop_after_seal(_journal: Path) -> None:
                    raise RuntimeError("leave journal sealed")

                sealer._send_journal = stop_after_seal  # type: ignore[method-assign]
                with pytest.raises(RuntimeError, match="leave journal sealed"):
                    sealer.reply(sealed.id, "sealed", fixture.answerer_session_id)
            with real_lock():
                yield

        replies._lock = seal_before_lock  # type: ignore[method-assign]
        if operation == "stage":
            outcome = replies.stage(
                target.id, "sealed" if target is sealed else "target"
            )
        elif operation == "flush":
            outcome = replies.flush(fixture.answerer_session_id)
        elif operation == "reply":
            outcome = replies.reply(
                target.id,
                "sealed" if target is sealed else "target",
                fixture.answerer_session_id,
            )
        elif operation == "dismiss":
            outcome = replies.dismiss(target.id)
        elif operation == "stage_and_flush":
            outcome = replies.stage_and_flush(
                [(target.id, "sealed" if target is sealed else "target")],
                fixture.answerer_session_id,
            )
        elif operation == "resolve_local":
            outcome = replies.resolve_local(
                target.id, "sealed" if target is sealed else "target"
            )
        else:
            assert operation == "recover"
            outcome = replies.recover()
        assert outcome is not None, "terminal operation lost its recovery result"
        publishes_target = operation == "flush" or (
            operation in {"reply", "stage_and_flush"} and selection == "unrelated"
        )
        assert len(publications) == 1 + int(publishes_target), (
            "terminal operation lost its recovery publication"
        )
        return

    if case == "success":
        values = _stage_all(fixture)
        replies.flush(fixture.session_id)
        assert len(publications) == 1
        messages = fixture.inbox_messages()
        assert len(messages) == 1
        assert all(value in messages[0]["body"] for value in values)
        _assert_resolved(fixture, values)
        assert not replies.recover().staged
        _assert_no_live_journal(fixture)
        return

    if case.startswith("stage-"):
        fault_kind = case.rsplit("-", 1)[1]
        real_replace = os.replace

        def fail_stage(
            source: object, destination: object, *args: object, **kwargs: object
        ) -> None:
            if Path(destination).parent.name == "staged":
                raise _fail_once(fault_kind)
            real_replace(source, destination, *args, **kwargs)

        # test-contract: seam stop the atomic popup-stage publication at its named step
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_stage)
            with pytest.raises((RuntimeError, asyncio.CancelledError)):
                replies.stage(fixture.records[0].id, "answer")
        restarted = fixture.replies
        assert restarted.recover().staged == ()
        assert fixture.inbox_messages() == []
        assert (
            AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
            .get(fixture.records[0].id)
            .status
            == "open"
        )
        return

    if case.startswith("seal-"):
        values = _stage_all(fixture)
        after = case.startswith("seal-after")
        fault_kind = case.rsplit("-", 1)[1]
        real_replace = os.replace

        def fail_seal(
            source: object, destination: object, *args: object, **kwargs: object
        ) -> None:
            if Path(destination).name.startswith("sending-"):
                if after:
                    real_replace(source, destination, *args, **kwargs)
                raise _fail_once(fault_kind)
            real_replace(source, destination, *args, **kwargs)

        # test-contract: seam stop the atomic journal seal immediately before or after rename
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_seal)
            with pytest.raises((RuntimeError, asyncio.CancelledError)):
                replies.flush(fixture.session_id)
        restarted = fixture.replies
        recovery = restarted.recover()
        if after:
            assert len(fixture.inbox_messages()) == 1
            _assert_resolved(fixture, values)
            assert recovery.staged == ()
            _assert_no_live_journal(fixture)
        else:
            assert fixture.inbox_messages() == []
            assert recovery.staged == tuple(
                (record.id, value)
                for record, value in sorted(
                    zip(fixture.records, values, strict=True),
                    key=lambda item: item[0].id,
                )
            )
        return

    if case.startswith(("send-", "resolve-", "cleanup-")):
        values = _stage_all(fixture)
        fault_kind = case.rsplit("-", 1)[1]
        if case.startswith("send-"):
            after = case.startswith("send-after")
            real_send = ProjectInbox.send

            def fail_send(inbox: ProjectInbox, **kwargs: Any) -> str:
                if after:
                    real_send(inbox, **kwargs)
                raise _fail_once(fault_kind)

            # test-contract: seam stop inbox publication immediately before or after its public send
            monkeypatch.setattr(ProjectInbox, "send", fail_send)
        elif case.startswith("resolve-"):
            real_record_replace = AttentionStore.replace
            resolved_count = 0
            fail_at = 0 if case.startswith("resolve-first") else 1

            def fail_resolve(
                store: AttentionStore, record: AttentionRecord, **kwargs: Any
            ) -> None:
                nonlocal resolved_count
                if record.status == "resolved":
                    if resolved_count == fail_at:
                        raise _fail_once(fault_kind)
                    resolved_count += 1
                real_record_replace(store, record, **kwargs)

            # test-contract: seam stop the real record replacement at the resolution commit
            monkeypatch.setattr(AttentionStore, "replace", fail_resolve)
        elif case.startswith("cleanup-rename"):
            after = case.startswith("cleanup-rename-after")
            real_replace = os.replace

            def fail_done(
                source: object, destination: object, *args: object, **kwargs: object
            ) -> None:
                if Path(source).name.startswith("sending-") and Path(
                    destination
                ).name.startswith("done-"):
                    if after:
                        real_replace(source, destination, *args, **kwargs)
                    raise _fail_once(fault_kind)
                real_replace(source, destination, *args, **kwargs)

            # test-contract: seam stop journal cleanup immediately before or after its done rename
            monkeypatch.setattr(os, "replace", fail_done)
        else:
            real_unlink = Path.unlink
            failed = False

            def fail_remove(path: Path, *args: object, **kwargs: object) -> None:
                nonlocal failed
                if path.parent.name.startswith("done-") and not failed:
                    failed = True
                    raise _fail_once(fault_kind)
                real_unlink(path, *args, **kwargs)

            # test-contract: seam stop durable journal removal at its file unlink
            monkeypatch.setattr(Path, "unlink", fail_remove)

        with pytest.raises((RuntimeError, asyncio.CancelledError)):
            replies.flush(fixture.session_id)
        monkeypatch.undo()
        restarted = fixture.replies
        restarted.recover()
        messages = fixture.inbox_messages()
        assert len(messages) == 1
        assert len({message["id"] for message in messages}) == 1
        _assert_resolved(fixture, values)
        _assert_no_live_journal(fixture)
        return

    if case.startswith("publish-"):
        values = (
            ["x" * 30_000 for _ in fixture.records]
            if case == "publish-body"
            else _stage_all(fixture)
        )
        if case == "publish-body":
            _stage_all(fixture, values)
            real_publish = ProjectInbox._publish_once

            def fail_body(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("injected body publication failure")

            # test-contract: seam fail the optional large-body atomic publication
            monkeypatch.setattr(ProjectInbox, "_publish_once", staticmethod(fail_body))
        elif case == "publish-record":
            real_publish = ProjectInbox._publish_record

            def fail_record(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("injected inbox record publication failure")

            # test-contract: seam fail the inbox record atomic publication
            monkeypatch.setattr(ProjectInbox, "_publish_record", fail_record)
        else:
            real_publish = SentMessageTracker.record

            def fail_tracker(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("injected sent tracker publication failure")

            # test-contract: seam fail sent-tracker publication after the inbox record exists
            monkeypatch.setattr(SentMessageTracker, "record", fail_tracker)
        with pytest.raises(RuntimeError):
            replies.flush(fixture.answerer_session_id)
        monkeypatch.undo()
        fixture.replies.recover()
        messages = fixture.inbox_messages()
        assert len(messages) == 1
        assert messages[0]["from"]["session"] == fixture.answerer_session_id
        _assert_resolved(fixture, values)
        _assert_no_live_journal(fixture)
        assert real_publish is not None
        return

    if case == "duplicates":
        record = fixture.records[0]
        replies.stage(record.id, "same")
        replies.stage(record.id, "same")
        with pytest.raises(ValueError, match="TUI decisions popup"):
            replies.stage(record.id, "different")
        replies.flush(fixture.session_id)
        first = fixture.replies.recover()
        second = fixture.replies.recover()
        outcome = fixture.replies.reply(record.id, "same", fixture.session_id)
        assert outcome.already_resolved
        assert outcome.message_id == fixture.inbox_messages()[0]["id"]
        assert first.staged == second.staged == ()
        assert len(fixture.inbox_messages()) == 1
        return

    if case == "same-id-conflicts":
        popup, immediate, dismissed = fixture.records
        _make_information_item(fixture, dismissed)
        replies.stage(popup.id, "typed popup text")
        staged_dir = fixture.attention_dir / "staged"
        before = {path.name: path.read_bytes() for path in staged_dir.iterdir()}
        for operation in (
            lambda: replies.reply(popup.id, "other", fixture.session_id),
            lambda: replies.dismiss(popup.id),
            lambda: replies.stage_and_flush([(popup.id, "other")], fixture.session_id),
        ):
            with pytest.raises(
                attention_replies.ReplyConflict, match="TUI decisions popup"
            ):
                operation()
            assert publications == []
        replies.reply(immediate.id, "unrelated", fixture.session_id)
        replies.dismiss(dismissed.id)
        after = {path.name: path.read_bytes() for path in staged_dir.iterdir()}
        assert after == before
        assert replies.recover().staged == ((popup.id, "typed popup text"),)
        assert (
            AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
            .get(popup.id)
            .status
            == "open"
        )
        assert len(fixture.inbox_messages()) == 1
        return

    if case == "broken-journal-recovery":
        broken, unrelated = fixture.records[:2]
        journals = _leave_sealed_journals(
            fixture,
            replies,
            monkeypatch,
            [(broken, "broken answer"), (unrelated, "unrelated answer")],
        )
        broken_path = fixture.attention_dir / f"{broken.id}.json"
        readable_bytes = broken_path.read_bytes()
        broken_path.write_bytes(b"{")
        retained_bytes = {
            path.relative_to(journals[0]): path.read_bytes()
            for path in journals[0].iterdir()
            if path.is_file()
        }

        try:
            outcome: ReplyOutcome | attention_replies.ReplyConflict = replies.recover()
        except attention_replies.ReplyConflict as exc:  # old behavior asserted below
            outcome = exc
        assert not isinstance(outcome, attention_replies.ReplyConflict), (
            f"broken journal stopped unrelated recovery: {outcome!r}"
        )
        assert [item.attention_id for item in outcome.items] == [unrelated.id]
        assert journals[0].is_dir()
        assert not journals[1].exists()
        assert broken_path.read_bytes() == b"{"
        assert {
            path.relative_to(journals[0]): path.read_bytes()
            for path in journals[0].iterdir()
            if path.is_file()
        } == retained_bytes
        assert (
            AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
            .get(unrelated.id)
            .decision
            == "unrelated answer"
        )
        assert len(fixture.inbox_messages()) == 1

        assert replies.recover().items == ()
        assert journals[0].is_dir()
        assert len(fixture.inbox_messages()) == 1

        broken_path.write_bytes(readable_bytes)
        repaired = replies.recover()
        assert [item.attention_id for item in repaired.items] == [broken.id]
        assert not journals[0].exists()
        assert len(fixture.inbox_messages()) == 2
        return

    if case == "multiple-journals":
        held: list[Path] = []
        real_replace = os.replace

        def hold_seal(
            source: object, destination: object, *args: object, **kwargs: object
        ) -> None:
            destination_path = Path(destination)
            if destination_path.name.startswith("sending-"):
                held_path = fixture.attention_dir / f"held-{len(held) + 1}"
                real_replace(source, held_path, *args, **kwargs)
                held.append(held_path)
                raise RuntimeError("crash after seal")
            real_replace(source, destination, *args, **kwargs)

        # test-contract: seam retain two journals produced by public calls for ordered recovery
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", hold_seal)
            for record, value in zip(
                fixture.records[:2], ("first", "second"), strict=True
            ):
                with pytest.raises(RuntimeError, match="crash after seal"):
                    replies.stage_and_flush([(record.id, value)], fixture.session_id)
        journal_ids = []
        for sequence, held_path in enumerate(held, 1):
            item = next(
                path for path in held_path.iterdir() if not path.name.startswith(".")
            )
            message_id = hashlib.sha256(
                json.dumps(
                    [(item.name, item.read_text(encoding="utf-8"))],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()[:32]
            destination = (
                fixture.attention_dir / f"sending-{sequence:020d}-{message_id}"
            )
            held_path.rename(destination)
            journal_ids.append(message_id)
        real_send = ProjectInbox.send
        sends: list[str] = []

        def stop_between(inbox: ProjectInbox, **kwargs: Any) -> str:
            sends.append(kwargs["message_id"])
            if len(sends) == 2:
                raise RuntimeError("crash between journals")
            return real_send(inbox, **kwargs)

        # test-contract: seam crash after the first sealed journal drains, before the second publishes
        with monkeypatch.context() as patch:
            patch.setattr(ProjectInbox, "send", stop_between)
            with pytest.raises(RuntimeError, match="between journals"):
                fixture.replies.recover()
        fixture.replies.recover()
        assert sends == journal_ids
        assert {message["id"] for message in fixture.inbox_messages()} == set(
            journal_ids
        )
        store = AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
        assert store.get(fixture.records[0].id).decision == "first"
        assert store.get(fixture.records[1].id).decision == "second"
        _assert_no_live_journal(fixture)
        return

    assert case == "dismiss-information"
    record = fixture.records[0]
    _make_information_item(fixture, record)
    first = replies.dismiss(record.id)
    second = replies.dismiss(record.id)
    assert first.items[0].status == "applied"
    assert second.already_resolved
    resolved = AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5).get(
        record.id
    )
    assert resolved.status == "resolved" and resolved.decision is None
    assert fixture.inbox_messages() == []



_RELAY_PREAMBLE = (
    "User relayed from session {sender}. "
    "Check whether the situation has changed before acting."
)


def _section(record: AttentionRecord, value: str) -> str:
    return (
        f"Question: {record.title} (id {record.id}, asked {record.created_at})\n"
        f"Decision: {value}"
    )


def test_batch_reply_message_names_each_question(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=2)
    values = _stage_all(fixture, ["yes", "no"])
    fixture.replies.flush(fixture.answerer_session_id)
    [message] = fixture.inbox_messages()
    # sections follow the journal order, which sorts by item id
    ordered = sorted(
        zip(fixture.records, values, strict=True), key=lambda pair: pair[0].id
    )
    assert message["body"] == "\n\n".join(
        [
            _RELAY_PREAMBLE.format(sender=fixture.answerer_session_id),
            *(_section(record, value) for record, value in ordered),
        ]
    )
    assert message["title"] == "Decision batch (2)"


def test_single_reply_message_names_its_question(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=1)
    [record] = fixture.records
    fixture.replies.stage(record.id, "Use SQLite.")
    fixture.replies.flush(fixture.answerer_session_id)
    [message] = fixture.inbox_messages()
    assert message["body"] == (
        _RELAY_PREAMBLE.format(sender=fixture.answerer_session_id)
        + "\n\n"
        + _section(record, "Use SQLite.")
    )
    assert message["title"] == f"Decision: {record.title}"


def _attention_files(fixture: ReplyFixture) -> dict[str, bytes]:
    return {
        path.relative_to(fixture.attention_dir).as_posix(): path.read_bytes()
        for path in fixture.attention_dir.rglob("*")
        if path.is_file() and path.name != ".resolution.lock"
    }


def test_flush_sends_good_staged_reply_and_keeps_broken_one(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=2)
    good, bad = fixture.records
    fixture.replies.stage(good.id, "good answer")
    fixture.replies.stage(bad.id, "bad answer")
    bad_path = fixture.attention_dir / f"{bad.id}.json"
    bad_path.write_bytes(b"{")

    try:
        outcome = fixture.replies.flush(fixture.answerer_session_id)
    # Capture the old failure so fail-first reaches the behavior assertion.
    except attention_replies.ReplyConflict as exc:
        outcome = ReplyOutcome(items=(), message_id=repr(exc))

    messages = fixture.inbox_messages()
    assert len(messages) == 1, f"broken staged record blocked the batch: {outcome}"
    assert messages[0]["body"] == (
        _RELAY_PREAMBLE.format(sender=fixture.answerer_session_id)
        + "\n\n"
        + _section(good, "good answer")
    )
    store = AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
    assert store.get(good.id).status == "resolved"
    assert bad_path.read_bytes() == b"{"
    assert (fixture.attention_dir / "staged" / bad.id).read_text() == "bad answer"
    assert not (fixture.attention_dir / "staged" / good.id).exists()
    statuses = {item.attention_id: item.status for item in outcome.items}
    assert statuses == {good.id: "applied", bad.id: "broken"}
    assert outcome.staged == ((bad.id, "bad answer"),)
    _assert_no_live_journal(fixture)


@pytest.mark.parametrize(
    "operation",
    [
        "stage",
        "stage-flush",
        "flush",
        "reply",
        "dismiss",
        "resolve_local",
        "stage_and_flush",
        "recover",
    ],
)
def test_record_broken_at_call_time_never_blocks_good_items(
    operation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, count=2)
    good, bad = fixture.records
    replies = fixture.replies
    sender = fixture.answerer_session_id
    if operation in {"stage-flush", "flush"}:
        replies.stage(bad.id, "bad answer")
    if operation == "flush":
        replies.stage(good.id, "good answer")
    if operation == "recover":
        _leave_sealed_journals(
            fixture, replies, monkeypatch, [(good, "good answer"), (bad, "bad answer")]
        )
    bad_path = fixture.attention_dir / f"{bad.id}.json"
    bad_path.write_bytes(b"{")
    before = _attention_files(fixture)
    calls = {
        "stage": lambda: replies.stage(bad.id, "bad answer"),
        "stage-flush": lambda: replies.stage(
            good.id, "good answer", flush_ids={good.id}, from_session=sender
        ),
        "flush": lambda: replies.flush(sender),
        "reply": lambda: replies.reply(bad.id, "bad answer", sender),
        "dismiss": lambda: replies.dismiss(bad.id),
        "resolve_local": lambda: replies.resolve_local(bad.id, "bad answer"),
        "stage_and_flush": lambda: replies.stage_and_flush(
            [(good.id, "good answer"), (bad.id, "bad answer")], sender
        ),
        "recover": replies.recover,
    }
    refusal: attention_replies.ReplyConflict | None = None
    outcome = ReplyOutcome()
    try:
        outcome = calls[operation]()
    except attention_replies.ReplyConflict as exc:
        refusal = exc

    assert bad_path.read_bytes() == b"{"
    store = AttentionStore(fixture.home / "sessions" / fixture.session_id, writer_epoch=5)
    if operation in {"stage", "reply", "dismiss", "resolve_local"}:
        assert refusal is not None and "invalid_json" in str(refusal)
        assert _attention_files(fixture) == before, "refusal left a partial write"
        assert fixture.inbox_messages() == []
        return
    assert refusal is None, f"{operation} let a broken record block good items"
    [message] = fixture.inbox_messages()
    assert message["body"] == (
        _RELAY_PREAMBLE.format(sender=sender) + "\n\n" + _section(good, "good answer")
    )
    assert store.get(good.id).status == "resolved"
    if operation == "recover":
        # The broken record's journal is kept until the record is repaired.
        [journal] = fixture.attention_dir.glob("sending-*-*")
        assert (journal / bad.id).read_text() == "bad answer"
        return
    broken = [item for item in outcome.items if item.attention_id == bad.id]
    assert [(item.status, item.detail) for item in broken] == [
        (
            "broken",
            (
                f"attention record {bad.id} is broken (invalid_json); "
                f"repair attention/{bad.id}.json before replying"
            ),
        )
    ]
    staged = fixture.attention_dir / "staged" / bad.id
    assert staged.exists() == (operation != "stage_and_flush")
    _assert_no_live_journal(fixture)
