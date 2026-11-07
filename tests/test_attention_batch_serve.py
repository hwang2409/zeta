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
from zeta.project_inbox import ProjectInbox
from zeta.server import ZetaServer


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
        from zeta.attention_replies import AttentionReplies

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
        from zeta.attention_replies import AttentionReplies

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
