from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail, owned_app
from tests.support.serve_close import close_serve
from tests.test_attention_batch_ui import (
    _make_app,
    _press,
    _rendered_text,
    _session,
    _wait_for_records,
)
from tests.test_attention_batch_ui import _request as _seed_attention
from tests.test_server import (
    _connect,
    _ready_extensions,
    _request,
    _seed_slash_fixtures,
)
from zeta.attention_records import AttentionStore
from zeta.attention_replies import AttentionReplies
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.server.slash_commands import ServerSlashSession
from zeta.skills import SkillCatalog
from zeta.tools.list_attention import register as register_list_attention
from zeta.tools.registry import ToolRegistry


def _start_publication_watcher(loop, *, turn_active: bool = False) -> asyncio.Task[None]:
    loop._project_inbox_owner_loop = asyncio.get_running_loop()
    loop._project_inbox_publication_event = asyncio.Event()
    # test-contract: a real owner is mid-turn when an in-process publication
    # arrives, so the supervisor's 2-second idle tick is suppressed; only the
    # wake delivers. this lets the wake tests fail if the wake is removed.
    loop._turn_active = turn_active
    return asyncio.create_task(loop._project_inbox_supervisor())


async def _stop_publication_watcher(watcher: asyncio.Task[None]) -> None:
    watcher.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(watcher, 10)


async def _await_inbox_notice(store, *, what: str, message_id: str | None = None) -> list:
    """Wait for the supervisor to deliver a matching durable inbox notice."""
    async with asyncio.timeout(10):
        while not (notices := [
            notice for notice in store.agent_notifications()
            if message_id is None or message_id in notice.data.get("message_ids", ())
        ]):
            # test-contract: latency poll the delivered durable inbox notice
            await asyncio.sleep(0.02)
    return notices


def _seed_inbox(registry, tmp_path: Path, project_id: str, session_id: str) -> str:
    """Put one durable inbox message in the owner's own project inbox."""
    inbox = ProjectInbox(registry, sessions_root=tmp_path / "home" / "sessions")
    return inbox.send(
        from_project=project_id,
        from_session="f" * 32,
        to_project=project_id,
        to_session=session_id,
        kind="info",
        title="seeded",
        body="deliver on the next poll",
    )


@pytest.mark.asyncio
async def test_project_attention_queue_matches_across_surfaces(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    home = tmp_path / "home"
    server = ZetaServer(home=home, cwd=project, port=0, provider="codex")
    reader, writer = await _connect(server)
    predecessors = []
    tools = None
    try:
        async def rpc(method, params):
            response = (await _request(reader, writer, method, method, params))[-1]
            assert "result" in response, response
            return response["result"]

        await rpc("hello", {"protocol_version": "1.0", "client_version": "1.1", "features": ["state_read_v1"]})
        runtime = server.controllers.attached().runtime
        await asyncio.to_thread(runtime.manager.project_registry.create_project, "local", "test", project)
        session_id = (await rpc("new_session", {"provider": "codex"}))["session"]["session_id"]
        opened = runtime.opened
        project_id = opened.metadata.project_id
        foreign = await asyncio.to_thread(runtime.manager.project_registry.create_project, "foreign", "test")
        for owner_project in (project_id, foreign.project_id):
            predecessors.append(await asyncio.to_thread(_session, home, owner_project))
        dead, other = predecessors
        rows = []
        for owner, kind, title, expected in (
            (opened, "decision", "Pick a database", "action"),
            (dead, "deferred", "Revisit the budget", "action"),
            (dead, "answer", "Build finished", "fyi"),
            (dead, "decided", "Use the blue theme", "fyi"),
            (dead, "decided", "Already dismissed", None),
            (other, "decision", "Foreign project choice", None),
        ):
            record = await asyncio.to_thread(_seed_attention, owner, owner.metadata.project_id, title, kind=kind)
            rows.append((record, expected))
            if title == "Already dismissed":
                store = AttentionStore(owner.store.session_dir, writer_epoch=5)
                await asyncio.to_thread(store.replace, replace(record, status="resolved", resolved_at=record.created_at))
        dead.store.close()
        tools = ToolRegistry(project, session_store=opened.store, skill_catalog=SkillCatalog.empty(), project_id=project_id)
        register_list_attention(tools)
        listing = tools.definitions_by_name["list_attention"].handler
        app = await _make_app(home, opened, session_id)
        async with owned_app(app):
            rendered = _rendered_text(app)
            assert "● 2 attention items" in rendered, rendered
            for view in ("action", "fyi"):
                if view == "fyi":
                    _press(app, "f")
                rendered = _rendered_text(app)
                slash = (await rpc("slash_run", {"session_id": session_id, "text": f"/decisions {view}"}))["text"]
                page = await rpc("state_attention", {"project_id": project_id, "list": view})
                result = (await listing({"list": view}))["structuredContent"]
                expected_ids = {record.id for record, expected in rows if expected == view}
                assert {row["attention_id"] for row in page["items"]} == expected_ids
                assert {row["id"] for row in result["items"]} == expected_ids
                assert (result["action_required_total"], result["info_total"]) == (2, 2)
                for record, expected in rows:
                    assert (record.title in rendered) == (expected == view), rendered
                    assert (record.title in slash) == (expected == view), slash
            assert "● 2 attention items" in rendered
            selected = app._decisions_panel.selected.record
            _press(app, "d")
            await await_or_fail(
                asyncio.to_thread(_wait_for_records, home, dead.store.session_id, {selected.id}),
                what="dismiss FYI from a dead predecessor in the TUI",
            )
            await app._decisions_tick()
            assert selected.title not in _rendered_text(app)
            page = await rpc("state_attention", {"project_id": project_id, "status": "open", "list": "fyi"})
            remaining = next(record for record, expected in rows if expected == "fyi" and record.id != selected.id)
            assert [row["attention_id"] for row in page["items"]] == [remaining.id]
            await rpc("slash_run", {"session_id": session_id, "text": f"/decisions dismiss {remaining.id}"})
            result = (await listing({"list": "fyi"}))["structuredContent"]
            assert result["items"] == [] and result["action_required_total"] == 2
    finally:
        if tools is not None:
            await tools.close()
        for predecessor in predecessors:
            predecessor.store.close()
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_serve_decisions_lists_broken_record_without_body(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    try:
        opened = server.controllers.attached().runtime.opened
        assert opened is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        assert registry is not None
        project_id = await asyncio.to_thread(registry.create_project, "broken", "test")
        record = await _seed_attention_record(
            AttentionStore(opened.store.session_dir, writer_epoch=5),
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
    async def defer_inbox_activation(_loop):
        pass

    # test-contract: seam start the real supervisor only at the dismiss below, not for the earlier batch.
    monkeypatch.setattr(AgentLoop, "_activate_project_inbox", defer_inbox_activation)
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    registry = ProjectRegistry(tmp_path / "home" / "projects")
    try:
        await asyncio.to_thread(registry.create_project, "serve attention", "test", project)
    finally:
        registry._storage_lease.close()
    reader, writer, session_id = await _ready_extensions(server)
    watcher = None
    try:
        opened = server.controllers.attached().runtime.opened
        assert opened is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        assert registry is not None
        project_id = opened.metadata.project_id
        store = AttentionStore(opened.store.session_dir, writer_epoch=5)
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

        loop = server.controllers.attached().runtime.loop
        assert loop is not None
        loop.tool_registry.project_id = project_id
        seeded = await asyncio.to_thread(
            _seed_inbox, registry, tmp_path, project_id, session_id
        )

        # test-contract: seam seal a second journal immediately before serve dismiss locks
        monkeypatch.setattr(AttentionReplies, "dismiss", dismiss_after_concurrent_seal)
        watcher = _start_publication_watcher(loop, turn_active=True)
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
        # the dismiss publishes in-process, waking the owner's supervisor to
        # deliver the pending inbox message as one durable notification.
        notices = await _await_inbox_notice(
            opened.store, what="serve owner inbox delivery", message_id=seeded
        )
        assert any(seeded in notice.data["message_ids"] for notice in notices)
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
        opened = server.controllers.attached().runtime.opened
        assert opened is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        assert registry is not None
        project_id = (
            await asyncio.to_thread(registry.create_project, "serve attention", "test")
        ).project_id
        store = AttentionStore(opened.store.session_dir, writer_epoch=5)
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
        opened = server.controllers.attached().runtime.opened
        assert opened is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        assert registry is not None
        project_id = (
            await asyncio.to_thread(registry.create_project, "invalid batch", "test")
        ).project_id
        store = AttentionStore(opened.store.session_dir, writer_epoch=5)
        sealed = await _seed_attention_record(store, session_id, project_id, "sealed")
        invalid = await _seed_attention_record(store, session_id, project_id, "invalid")
        _leave_sealed_reply(tmp_path / "home", session_id, sealed.id, monkeypatch)

        loop = server.controllers.attached().runtime.loop
        assert loop is not None
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
    registry = ProjectRegistry(tmp_path / "home" / "projects")
    try:
        await asyncio.to_thread(registry.create_project, "operation error", "test", project)
    finally:
        registry._storage_lease.close()
    reader, writer, session_id = await _ready_extensions(server)
    watcher = None
    try:
        opened = server.controllers.attached().runtime.opened
        assert opened is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        assert registry is not None
        project_id = opened.metadata.project_id
        store = AttentionStore(opened.store.session_dir, writer_epoch=5)
        sealed = await _seed_attention_record(store, session_id, project_id, "sealed")
        target = await _seed_attention_record(store, session_id, project_id, "target")
        _leave_sealed_reply(tmp_path / "home", session_id, sealed.id, monkeypatch)

        real_record = AttentionReplies._record

        def fail_target(replies: AttentionReplies, attention_id: str):
            if attention_id == target.id:
                raise RuntimeError("unexpected operation failure")
            return real_record(replies, attention_id)

        loop = server.controllers.attached().runtime.loop
        assert loop is not None
        loop.tool_registry.project_id = project_id
        seeded = await asyncio.to_thread(
            _seed_inbox, registry, tmp_path, project_id, session_id
        )

        # test-contract: seam fail after recovery publishes but before the action completes
        monkeypatch.setattr(AttentionReplies, "_record", fail_target)
        watcher = _start_publication_watcher(loop, turn_active=True)
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
        # the sealed-journal recovery still publishes in-process, so the owner's
        # supervisor wakes and delivers the pending inbox even though the op failed.
        notices = await _await_inbox_notice(
            opened.store, what="serve recovery inbox delivery"
        )
        assert any(seeded in notice.data["message_ids"] for notice in notices)
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
        opened = server.controllers.attached().runtime.opened
        loop = server.controllers.attached().runtime.loop
        assert opened is not None and loop is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        project_id = (
            await asyncio.to_thread(registry.create_project, "cancel serve", "test")
        ).project_id
        record = await _seed_attention_record(
            AttentionStore(opened.store.session_dir, writer_epoch=5), session_id, project_id, "late"
        )
        real_send = ProjectInbox.send

        def blocked_send(inbox: ProjectInbox, **kwargs):
            worker_started.set()
            assert release.wait(10), "serve publication gate was not released"
            try:
                return real_send(inbox, **kwargs)
            finally:
                worker_done.set()

        loop.tool_registry.project_id = project_id
        seeded = await asyncio.to_thread(
            _seed_inbox, registry, tmp_path, project_id, session_id
        )

        # test-contract: seam hold the real serve worker before durable publication
        monkeypatch.setattr(ProjectInbox, "send", blocked_send)
        watcher = _start_publication_watcher(loop, turn_active=True)
        adapter = ServerSlashSession(server.controllers.attached().runtime)
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
        # the publication has not completed, so nothing has woken the supervisor.
        assert opened.store.agent_notifications() == []
        release.set()
        assert await await_or_fail(
            asyncio.to_thread(worker_done.wait, 10), what="serve worker finish"
        )
        # the late durable publication still wakes the owner's supervisor, which
        # delivers the pending inbox even though the adapter task was cancelled.
        notices = await _await_inbox_notice(
            opened.store, what="cancelled serve adapter late delivery"
        )
        assert any(seeded in notice.data["message_ids"] for notice in notices)
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
        loop = server.controllers.attached().runtime.loop
        assert loop is not None
        registry = server.controllers.attached().runtime.manager.project_registry
        project_id = (
            await asyncio.to_thread(registry.create_project, "poll send", "test")
        ).project_id
        loop.tool_registry.project_id = project_id
        watcher = _start_publication_watcher(loop)
        separate_inbox = ProjectInbox(
            registry, sessions_root=tmp_path / "home" / "sessions"
        )
        seeded = await asyncio.to_thread(
            separate_inbox.send,
            from_project=project_id,
            from_session="f" * 32,
            to_project=project_id,
            to_session=session_id,
            kind="info",
            title="cross-process style",
            body="no in-process callback",
        )
        # no in-process callback fires, so only the supervisor's 2-second timeout
        # poll finds the cross-process message and delivers it.
        notices = await _await_inbox_notice(
            loop.store, what="two-second inbox poll"
        )
        assert any(seeded in notice.data["message_ids"] for notice in notices)
    finally:
        if watcher is not None:
            await _stop_publication_watcher(watcher)
        await close_serve(server, writer)
