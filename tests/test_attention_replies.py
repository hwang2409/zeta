from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from zeta.attention_records import AttentionRecord, AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox, SentMessageTracker


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
    store = AttentionStore(session.store.session_dir)
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
    store = AttentionStore(fixture.home / "sessions" / fixture.session_id)
    assert [store.get(record.id).decision for record in fixture.records] == replies
    assert all(store.get(record.id).status == "resolved" for record in fixture.records)


def _assert_no_live_journal(fixture: ReplyFixture) -> None:
    assert not list(fixture.attention_dir.glob("sending-*-*"))
    assert not list(fixture.attention_dir.glob("done-*-*"))
    assert not list(fixture.attention_dir.glob(".batch-*"))


def _fail_once(kind: str) -> BaseException:
    if kind == "cancel":
        return asyncio.CancelledError()
    return RuntimeError("injected crash")


def test_popup_stage_syncs_contents_before_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, count=1)
    events: list[tuple[str, str]] = []
    real_sync = AttentionReplies._sync_file
    real_replace = os.replace

    def sync(path: Path) -> None:
        events.append(("sync", path.name))
        real_sync(path)

    def replace(source: object, destination: object, *args: object, **kwargs: object) -> None:
        if Path(destination).parent.name == "staged":
            events.append(("rename", Path(destination).name))
        real_replace(source, destination, *args, **kwargs)

    # test-contract: seam record the file barrier order around atomic staging
    monkeypatch.setattr(AttentionReplies, "_sync_file", staticmethod(sync))
    monkeypatch.setattr(os, "replace", replace)
    fixture.replies.stage(fixture.records[0].id, "durable answer")
    rename_index = next(index for index, event in enumerate(events) if event[0] == "rename")
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
        *_FAULT_ROWS,
        "publish-body",
        "publish-record",
        "publish-tracker",
        "duplicates",
        "same-id-conflicts",
        "multiple-journals",
        "dismiss-information",
    ],
)
def test_attention_replies_public_table(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise durable reply outcomes only through the public AttentionReplies API."""
    fixture = _fixture(tmp_path, count=3)
    replies = fixture.replies

    if case == "success":
        values = _stage_all(fixture)
        result = replies.flush(fixture.session_id)
        assert result.published
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
            AttentionStore(fixture.home / "sessions" / fixture.session_id)
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
        message_id, already = fixture.replies.reply(
            record.id, "same", fixture.session_id
        )
        assert already and message_id == fixture.inbox_messages()[0]["id"]
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
            with pytest.raises(ValueError, match="TUI decisions popup"):
                operation()
        replies.reply(immediate.id, "unrelated", fixture.session_id)
        replies.dismiss(dismissed.id)
        after = {path.name: path.read_bytes() for path in staged_dir.iterdir()}
        assert after == before
        assert replies.recover().staged == ((popup.id, "typed popup text"),)
        assert (
            AttentionStore(fixture.home / "sessions" / fixture.session_id)
            .get(popup.id)
            .status
            == "open"
        )
        assert len(fixture.inbox_messages()) == 1
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
        store = AttentionStore(fixture.home / "sessions" / fixture.session_id)
        assert store.get(fixture.records[0].id).decision == "first"
        assert store.get(fixture.records[1].id).decision == "second"
        _assert_no_live_journal(fixture)
        return

    assert case == "dismiss-information"
    record = fixture.records[0]
    _make_information_item(fixture, record)
    assert not replies.dismiss(record.id)
    assert replies.dismiss(record.id)
    resolved = AttentionStore(fixture.home / "sessions" / fixture.session_id).get(
        record.id
    )
    assert resolved.status == "resolved" and resolved.decision is None
    assert fixture.inbox_messages() == []
