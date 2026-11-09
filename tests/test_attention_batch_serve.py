from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.test_server import (
    _ready_extensions,
    _request,
    _seed_slash_fixtures,
)
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.project_inbox import ProjectInbox
from zeta.server import ZetaServer
from zeta.server.slash_commands import ServerSlashSession


def _start_publication_watcher(loop) -> asyncio.Task[None]:
    loop._project_inbox_owner_loop = asyncio.get_running_loop()
    loop._project_inbox_publication_event = asyncio.Event()
    return asyncio.create_task(loop._watch_project_inbox())


async def _stop_publication_watcher(watcher: asyncio.Task[None]) -> None:
    watcher.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(watcher, 10)


@pytest.mark.asyncio
async def test_serve_decisions_lists_broken_record_without_body(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    try:
        opened = server.runtime.opened
        assert opened is not None
        registry = server.runtime.manager.project_registry
        assert registry is not None
        project_id = await asyncio.to_thread(registry.create_project, "broken", "test")
        record = await _seed_attention_record(
            AttentionStore(opened.store.session_dir),
            session_id,
            project_id.project_id,
            "secret request",
        )
        path = opened.store.session_dir / "attention" / f"{record.id}.json"
        path.write_bytes(b'{"why":"secret request body"')
        raw_response = await _request(
            reader,
            writer,
            "broken-list",
            "slash_run",
            {"session_id": session_id, "text": "/decisions"},
        )
        response = raw_response[-1]["result"]["text"]
        assert record.id in response
        assert "attention/" in response
        assert "invalid_json" in response
        assert "secret request body" not in response
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_serve_decisions_batch_is_one_message_and_rejects_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    watcher = None
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
        new_ids = await asyncio.to_thread(inbox.new_ids, project_id, session_id=session_id)
        assert len(new_ids) == 1

        broken_id = "e" * 32
        broken_path = opened.store.session_dir / "attention" / f"{broken_id}.json"
        broken_bytes = b'{"why":"do not expose"'
        broken_path.write_bytes(broken_bytes)
        refusal = (
            f"attention record {broken_id} is broken (invalid_json); "
            f"repair attention/{broken_id}.json before replying"
        )
        for command in ("reply", "dismiss"):
            response = (
                await _request(
                    reader,
                    writer,
                    f"broken-{command}",
                    "slash_run",
                    {
                        "session_id": session_id,
                        "text": f"/decisions {command} {broken_id} yes".rstrip(),
                    },
                )
            )[-1]["result"]
            assert response["text"] == refusal
        broken_payload = json.dumps([{"id": broken_id, "reply": "yes"}])
        response = (
            await _request(
                reader,
                writer,
                "broken-batch",
                "slash_run",
                {
                    "session_id": session_id,
                    "text": f"/decisions batch {broken_payload}",
                },
            )
        )[-1]["result"]
        assert response["text"] == refusal
        assert broken_path.read_bytes() == broken_bytes

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
        checked = asyncio.Event()

        async def count_check(*, periodic: bool = False) -> None:
            nonlocal checks
            checks += 1
            checked.set()

        # test-contract: seam seal a second journal immediately before serve dismiss locks
        monkeypatch.setattr(AttentionReplies, "dismiss", dismiss_after_concurrent_seal)
        # test-contract: seam count the real serve adapter's inbox checks
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        watcher = _start_publication_watcher(loop)
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
        await await_or_fail(checked.wait(), what="serve owner inbox check")
        assert checks == 1
    finally:
        if watcher is not None:
            await _stop_publication_watcher(watcher)
        await close_serve(server, writer)


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

    # test-contract: seam leave a sealed journal before the public adapter runs
    monkeypatch.setattr(replies, "_send_journal", stop_after_seal)
    with pytest.raises(Exception, match="leave journal sealed"):
        replies.reply(attention_id, "yes", session_id)


@pytest.mark.asyncio
async def test_serve_batch_sends_good_items_and_refuses_broken_one(
    tmp_path: Path,
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
        good, bad = [
            await asyncio.to_thread(
                store.request,
                session_id=session_id,
                project_id=project_id,
                entry_id=None,
                entry_seq=None,
                title=f"pick {index}",
                why="choose",
                options=("yes",),
            )
            for index in range(2)
        ]
        bad_path = opened.store.session_dir / "attention" / f"{bad.id}.json"
        bad_path.write_bytes(b"{")
        payload = json.dumps(
            [{"id": good.id, "reply": "yes"}, {"id": bad.id, "reply": "yes"}]
        )
        response = (
            await _request(
                reader,
                writer,
                "mixed",
                "slash_run",
                {"session_id": session_id, "text": f"/decisions batch {payload}"},
            )
        )[-1]["result"]
        assert response["text"] == (
            "replied to 1 attention item(s)\n"
            f"attention record {bad.id} is broken (invalid_json); "
            f"repair attention/{bad.id}.json before replying"
        )
        assert store.get(good.id).status == "resolved"
        assert bad_path.read_bytes() == b"{"
        inbox = ProjectInbox(registry, sessions_root=tmp_path / "home" / "sessions")
        assert len(
            await asyncio.to_thread(inbox.new_ids, project_id, session_id=session_id)
        ) == 1
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_serve_invalid_batch_does_not_recover_older_sealed_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    watcher = None
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

        async def count_check(*, periodic: bool = False) -> None:
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
        if watcher is not None:
            await _stop_publication_watcher(watcher)
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_serve_checks_recovery_outcome_before_unexpected_operation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    watcher = None
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
        checked = asyncio.Event()

        async def count_check(*, periodic: bool = False) -> None:
            nonlocal checks
            checks += 1
            checked.set()

        # test-contract: seam fail after recovery publishes but before the action completes
        monkeypatch.setattr(AttentionReplies, "_record", fail_target)
        # test-contract: seam count publisher-driven admission after the operation error
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        watcher = _start_publication_watcher(loop)
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
        await await_or_fail(checked.wait(), what="serve recovery inbox check")
        assert checks == 1
    finally:
        if watcher is not None:
            await _stop_publication_watcher(watcher)
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_cancelled_serve_adapter_still_checks_late_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    _reader, writer, session_id = await _ready_extensions(server)
    release = threading.Event()
    worker_started = threading.Event()
    worker_done = threading.Event()
    watcher = None
    try:
        opened = server.runtime.opened
        loop = server.runtime.loop
        assert opened is not None and loop is not None
        registry = server.runtime.manager.project_registry
        project_id = (
            await asyncio.to_thread(registry.create_project, "cancel serve", "test")
        ).project_id
        record = await _seed_attention_record(
            AttentionStore(opened.store.session_dir), session_id, project_id, "late"
        )
        real_send = ProjectInbox.send

        def blocked_send(inbox: ProjectInbox, **kwargs):
            worker_started.set()
            assert release.wait(10), "serve publication gate was not released"
            try:
                return real_send(inbox, **kwargs)
            finally:
                worker_done.set()

        checks: list[bool] = []
        real_check = loop._check_project_inbox

        async def count_check(*, periodic: bool = False) -> None:
            checks.append(periodic)
            await real_check(periodic=periodic)

        # test-contract: seam hold the real serve worker before durable publication
        monkeypatch.setattr(ProjectInbox, "send", blocked_send)
        # test-contract: seam count admission owned by the project-inbox watcher
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        watcher = _start_publication_watcher(loop)
        adapter = ServerSlashSession(server.runtime)
        replies = adapter._attention_replies(session_id)
        task = asyncio.create_task(
            adapter._run_attention_operation(
                replies.reply, record.id, "yes", session_id
            )
        )
        assert await await_or_fail(
            asyncio.to_thread(worker_started.wait, 10), what="serve worker start"
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert checks == []
        release.set()
        assert await await_or_fail(
            asyncio.to_thread(worker_done.wait, 10), what="serve worker finish"
        )
        for _ in range(10):
            if checks:
                break
            await asyncio.sleep(0)
        assert checks.count(False) == 1, (
            "cancelled serve adapter lost its late publication check"
        )
    finally:
        release.set()
        if watcher is not None:
            await _stop_publication_watcher(watcher)
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_cross_process_style_send_uses_existing_two_second_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    _reader, writer, session_id = await _ready_extensions(server)
    watcher = None
    try:
        loop = server.runtime.loop
        assert loop is not None
        registry = server.runtime.manager.project_registry
        project_id = (
            await asyncio.to_thread(registry.create_project, "poll send", "test")
        ).project_id
        loop.tool_registry.project_id = project_id
        periodic_check = asyncio.Event()
        calls: list[bool] = []
        real_check = loop._check_project_inbox

        async def count_check(*, periodic: bool = False) -> None:
            calls.append(periodic)
            await real_check(periodic=periodic)
            if periodic:
                periodic_check.set()

        # test-contract: seam identify the existing idle watcher's periodic check
        monkeypatch.setattr(loop, "_check_project_inbox", count_check)
        watcher = _start_publication_watcher(loop)
        separate_inbox = ProjectInbox(
            registry, sessions_root=tmp_path / "home" / "sessions"
        )
        await asyncio.to_thread(
            separate_inbox.send,
            from_project=project_id,
            from_session="f" * 32,
            to_project=project_id,
            to_session=session_id,
            kind="info",
            title="cross-process style",
            body="no in-process callback",
        )
        await await_or_fail(periodic_check.wait(), what="two-second inbox poll")
        assert calls == [True]
    finally:
        if watcher is not None:
            await _stop_publication_watcher(watcher)
        await close_serve(server, writer)
