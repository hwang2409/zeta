from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.test_server import (
    _close,
    _ready_extensions,
    _request,
    _seed_slash_fixtures,
)
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.project_inbox import ProjectInbox
from zeta.server import ZetaServer
from zeta.server.slash_commands import ServerSlashSession


@pytest.mark.asyncio
async def test_serve_decisions_batch_is_one_message_and_rejects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    try:
        opened = server.runtime.opened
        assert opened is not None
        registry = server.runtime.manager.project_registry
        assert registry is not None
        project_id = (
            await asyncio.to_thread(registry.create_project, "serve attention", "test")
        ).project_id
        store = AttentionStore(opened.store.session_dir)
        records = await await_or_fail(
            asyncio.gather(
                *(
                    asyncio.to_thread(
                        store.request,
                        session_id=session_id,
                        project_id=project_id,
                        entry_id=None,
                        entry_seq=None,
                        title=f"pick {index}",
                        why="choose",
                        options=("yes",),
                    )
                    for index in range(3)
                )
            ),
            what="create attention records",
        )
        payload = json.dumps([{"id": record.id, "reply": "yes"} for record in records])
        response = (
            await _request(
                reader,
                writer,
                "batch",
                "slash_run",
                {"session_id": session_id, "text": f"/decisions batch {payload}"},
            )
        )[-1]["result"]
        assert response["text"] == "replied to 3 attention item(s)"
        assert all(store.get(record.id).status == "resolved" for record in records)
        inbox = ProjectInbox(registry, sessions_root=tmp_path / "home" / "sessions")
        assert len(inbox.new_ids(project_id, session_id=session_id)) == 1

        conflict = await asyncio.to_thread(
            store.request,
            session_id=session_id,
            project_id=project_id,
            entry_id=None,
            entry_seq=None,
            title="conflict",
            why="choose",
            options=("yes",),
        )
        await asyncio.to_thread(
            AttentionReplies(tmp_path / "home", session_id).stage, conflict.id, "staged"
        )
        conflict_payload = json.dumps(
            [{"id": conflict.id, "reply": "other"}], separators=(",", ":")
        )
        error = (
            await _request(
                reader,
                writer,
                "conflict",
                "slash_run",
                {
                    "session_id": session_id,
                    "text": f"/decisions batch {conflict_payload}",
                },
            )
        )[-1]["error"]
        assert "staged reply in a TUI decisions popup" in error["message"]

        malformed = (
            await _request(
                reader,
                writer,
                "malformed",
                "slash_run",
                {"session_id": session_id, "text": "/decisions batch nope"},
            )
        )[-1]["result"]
        assert (
            malformed["text"]
            == "batch must be a JSON array of objects with id and reply"
        )

        sealed = await asyncio.to_thread(
            store.request,
            session_id=session_id,
            project_id=project_id,
            entry_id=None,
            entry_seq=None,
            title="sealed reply",
            why="choose",
            options=("yes",),
        )
        dismissible = await asyncio.to_thread(
            store.request,
            session_id=session_id,
            project_id=project_id,
            entry_id=None,
            entry_seq=None,
            title="status",
            why="inform",
            kind="answer",
        )
        real_dismiss = AttentionReplies.dismiss
        injected = False

        def dismiss_after_concurrent_seal(replies: AttentionReplies, attention_id: str):
            nonlocal injected
            if not injected:
                injected = True
                sealer = AttentionReplies(tmp_path / "home", session_id)

                def stop_after_seal(_journal: Path) -> None:
                    raise RuntimeError("leave journal sealed")

                sealer._send_journal = stop_after_seal  # type: ignore[method-assign]
                with pytest.raises(RuntimeError, match="leave journal sealed"):
                    sealer.reply(sealed.id, "yes", session_id)
            return real_dismiss(replies, attention_id)

        loop = server.runtime.loop
        assert loop is not None
        checks = 0

        async def count_check() -> None:
            nonlocal checks
            checks += 1

        # test-contract: seam seal a second journal immediately before serve dismiss locks
        monkeypatch.setattr(AttentionReplies, "dismiss", dismiss_after_concurrent_seal)
        # test-contract: seam count the real serve adapter's inbox checks
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        dismissed = (
            await _request(
                reader,
                writer,
                "dismiss",
                "slash_run",
                {
                    "session_id": session_id,
                    "text": f"/decisions dismiss {dismissible.id}",
                },
            )
        )[-1]["result"]
        assert dismissed["text"] == f"dismissed {dismissible.id}"
        assert checks == 1
    finally:
        await _close(server, writer)


async def _seed_attention_record(
    store: AttentionStore, session_id: str, project_id: str, title: str
):
    return await asyncio.to_thread(
        store.request,
        session_id=session_id,
        project_id=project_id,
        entry_id=None,
        entry_seq=None,
        title=title,
        why="choose",
        options=("yes",),
    )


def _leave_sealed_reply(
    home: Path, session_id: str, attention_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies = AttentionReplies(home, session_id)

    def stop_after_seal(_journal: Path) -> None:
        raise RuntimeError("leave journal sealed")

    monkeypatch.setattr(replies, "_send_journal", stop_after_seal)
    with pytest.raises(Exception, match="leave journal sealed"):
        replies.reply(attention_id, "yes", session_id)


@pytest.mark.asyncio
async def test_serve_invalid_batch_does_not_recover_older_sealed_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    try:
        opened = server.runtime.opened
        assert opened is not None
        registry = server.runtime.manager.project_registry
        assert registry is not None
        project_id = (
            await asyncio.to_thread(registry.create_project, "invalid batch", "test")
        ).project_id
        store = AttentionStore(opened.store.session_dir)
        sealed = await _seed_attention_record(store, session_id, project_id, "sealed")
        invalid = await _seed_attention_record(store, session_id, project_id, "invalid")
        _leave_sealed_reply(tmp_path / "home", session_id, sealed.id, monkeypatch)

        loop = server.runtime.loop
        assert loop is not None
        checks = 0

        async def count_check() -> None:
            nonlocal checks
            checks += 1

        # test-contract: seam count inbox admission through the real serve adapter
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        payload = json.dumps([{"id": invalid.id, "reply": "   "}])
        error = (
            await _request(
                reader,
                writer,
                "invalid-batch",
                "slash_run",
                {"session_id": session_id, "text": f"/decisions batch {payload}"},
            )
        )[-1]["error"]

        assert "reply must be nonempty and within the size limit" in error["message"]
        assert store.get(sealed.id).status == "open"
        assert checks == 0
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_checks_recovery_outcome_before_unexpected_operation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    try:
        opened = server.runtime.opened
        assert opened is not None
        registry = server.runtime.manager.project_registry
        assert registry is not None
        project_id = (
            await asyncio.to_thread(registry.create_project, "operation error", "test")
        ).project_id
        store = AttentionStore(opened.store.session_dir)
        sealed = await _seed_attention_record(store, session_id, project_id, "sealed")
        target = await _seed_attention_record(store, session_id, project_id, "target")
        _leave_sealed_reply(tmp_path / "home", session_id, sealed.id, monkeypatch)

        real_record = AttentionReplies._record

        def fail_target(replies: AttentionReplies, attention_id: str):
            if attention_id == target.id:
                raise RuntimeError("unexpected operation failure")
            return real_record(replies, attention_id)

        loop = server.runtime.loop
        assert loop is not None
        checks = 0
        outcomes = []
        real_check = ServerSlashSession._check_attention_outcome

        async def count_check() -> None:
            nonlocal checks
            checks += 1

        async def capture_outcome(session: ServerSlashSession, outcome) -> None:
            outcomes.append(outcome)
            await real_check(session, outcome)

        # test-contract: seam fail after recovery publishes but before the action completes
        monkeypatch.setattr(AttentionReplies, "_record", fail_target)
        # test-contract: seam inspect the outcome passed through the real serve adapter
        monkeypatch.setattr(ServerSlashSession, "_check_attention_outcome", capture_outcome)
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        error = (
            await _request(
                reader,
                writer,
                "operation-error",
                "slash_run",
                {
                    "session_id": session_id,
                    "text": f"/decisions dismiss {target.id}",
                },
            )
        )[-1]["error"]

        assert "unexpected operation failure" in error["message"]
        assert len(outcomes) == 1
        assert outcomes[0].published
        assert checks == 1
    finally:
        await _close(server, writer)
