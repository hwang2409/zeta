from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.serve_close import close_serve
from tests.test_project_inbox import _projects
from tests.test_server import _connect, _request
from tests.test_server_projects import _hello, _server
from zeta.agent.pending_delivery import DeliveryPublications
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import (
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
    harness_notice,
)
from zeta.runtime.loop import AgentLoop
from zeta.server.control import NotificationPresentation
from zeta.server.slash_commands import ServerSlashSession
from zeta.skills import SkillCatalog
from zeta.tui.render import render_event
from zeta.tui.slash_handlers import SlashHandlerMixin


async def _serve_payload(data: dict[str, object]) -> dict[str, object]:
    sent: list[dict[str, object]] = []

    async def send(
        _event: object,
        _session: object,
        *,
        data: dict[str, object],
        replay_row: object = None,
    ) -> bool:
        sent.append(data)
        return True

    await NotificationPresentation().publish("session", data, "1.2", send)
    return sent[0]


def _rendered_text(data: dict[str, object]) -> str:
    rendered = render_event(StreamEvent(StreamEventType.AGENT_NOTIFICATION, data=data))
    assert rendered is not None
    return rendered.plain


async def _run_model_route(
    manager: SessionManager,
    session_id: str,
    recipient_project_id: str,
    registry: ProjectRegistry,
    epoch: int,
    case: str,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    resumed = await asyncio.to_thread(manager.open, session_id)
    resumed_loop = AgentLoop(
        backend, resumed.store, skill_catalog=SkillCatalog.empty(),
        root_project_id=recipient_project_id, project_registry=registry,
    )
    try:
        await resumed_loop.activate()
        events = [event async for event in resumed_loop.run_turn(
            "resume", origin=MessageOrigin.USER
        )]
        notification = next(event for event in events if event.type is StreamEventType.AGENT_NOTIFICATION)
        delivered = next(
            message_ for message_ in backend.calls[0][0]
            if "zeta.step_delivery" in message_.metadata
        )
        model_text = "\n".join(
            block.text for message_ in backend.calls[0][0]
            for block in message_.content if isinstance(block, TextContent)
        )
        text = notification.data["text"]
        expected = "Requests are work to do" if epoch < 7 else "Project inbox wake"
        assert expected in text
        assert text in model_text
        if epoch >= 7:
            assert "trusted_assignment" not in model_text
            if case in {"delegated", "revoked"}:
                metadata_text = json.dumps(delivered.metadata["notifications"])
                assert all(
                    marker not in metadata_text
                    for marker in ('"author":', '"authority":', "trusted_assignment")
                )
    finally:
        await resumed_loop.close()
        resumed.store.close()


async def _run_live_route(
    manager: SessionManager,
    session_id: str,
    recipient_project_id: str,
    registry: ProjectRegistry,
    entry: object,
    epoch: int,
) -> None:
    resumed = await asyncio.to_thread(manager.open, session_id)
    live_loop = AgentLoop(
        FakeBackend([]), resumed.store, skill_catalog=SkillCatalog.empty(),
        root_project_id=recipient_project_id, project_registry=registry,
    )
    try:
        await live_loop.activate()
        event = live_loop._delivery_publication_events(
            DeliveryPublications((entry,), None)
        )[0]
        rendered = _rendered_text(event.data)
        served = await _serve_payload(event.data)
        assert served["text"] == rendered
        expected = "Requests are work to do" if epoch < 7 else "Project inbox wake"
        assert expected in rendered
        if epoch >= 7:
            assert "trusted_assignment" not in rendered
    finally:
        await live_loop.close()
        resumed.store.close()


async def _run_rpc_route(
    tmp_path: Path,
    home: Path,
    registry: ProjectRegistry,
    manager: SessionManager,
    recipient_project_id: str,
    message_id: str,
    session_id: str,
    case: str,
    epoch: int,
) -> None:
    registry._storage_lease.close()
    manager._storage_lease.close()
    (home / "storage-format.json").write_text(json.dumps({"writer_epoch": epoch}) + "\n")
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        result = (await _request(
            reader, writer, 2, "project_inbox",
            {"project_id": recipient_project_id},
        ))[-1]["result"]
        row = next(item for item in result["messages"] if item["id"] == message_id)
        if epoch < 7:
            assert "authority" not in row
            assert "author" not in row
            assert result["untrusted"] is False
        else:
            expected_authority = "trusted_assignment" if case in {"delegated", "legacy"} else "information"
            assert row["authority"] == expected_authority
            assert row["author"] == ("user" if case == "legacy" else "agent")
            assert result["untrusted"] is (expected_authority != "trusted_assignment")
        if epoch >= 7 and case in {"delegated", "revoked"}:
            live_inbox = ProjectInbox(
                server.runtime.manager.project_registry, sessions_root=home / "sessions"
            )
            claimed = await asyncio.to_thread(
                live_inbox.claim, recipient_project_id, message_id, session_id
            )
            assert (claimed is not None) is (case == "delegated")
    finally:
        await close_serve(server, writer)


async def _create_fixtures(
    tmp_path: Path, epoch: int
) -> tuple[Path, ProjectRegistry, object, object, SessionManager, object, ProjectInbox]:
    home = tmp_path / "home"
    home.mkdir()
    (home / "storage-format.json").write_text(json.dumps({"writer_epoch": epoch}) + "\n")
    registry = ProjectRegistry(home / "projects")
    sender, recipient = await asyncio.to_thread(
        lambda: (
            registry.create_project("sender", "sender work"),
            registry.create_project("recipient", "recipient work"),
        )
    )
    manager = SessionManager(home)
    opened = await asyncio.to_thread(
        manager.create, provider="codex", model="fake", cwd=home,
        project_id=recipient.project_id, auto_project=False,
    )
    for lease in (manager._storage_lease, opened.store._storage_lease, registry._storage_lease):
        lease._entry.epoch = epoch
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    return home, registry, sender, recipient, manager, opened, inbox


async def _seed_record(
    inbox: ProjectInbox, case: str, epoch: int, sender: object, recipient: object, opened: object
) -> tuple[str, dict[str, object]]:
    if case == "legacy":
        message_id = await asyncio.to_thread(
            inbox.send_user, from_project=sender.project_id, from_session="a" * 32,
            to_project=recipient.project_id, to_session=opened.store.session_id,
            kind="question", title="legacy", body="body",
        )
        return message_id, {
            "kind": "project_inbox", "notification_id": message_id,
            "project_id": recipient.project_id, "message_ids": [message_id],
            "text": "Requests are work to do: claim, do, and mark done. You do not need to confirm the sender with the user.",
        }
    delegated = case in {"delegated", "revoked"}
    if delegated and epoch >= 7:
        await asyncio.to_thread(
            inbox.set_delegation, recipient.project_id, sender.project_id,
            "question", enabled=True,
        )
    message_id = await asyncio.to_thread(
        inbox.send, from_project=sender.project_id, from_session="a" * 32,
        to_project=recipient.project_id, to_session=opened.store.session_id,
        kind="question", title=case, body="assignment",
    )
    message = next(
        item for item in (await asyncio.to_thread(
            inbox.list, recipient.project_id, session_id=opened.store.session_id
        ))["new"] if item["id"] == message_id
    )
    if case == "revoked" and epoch >= 7:
        await asyncio.to_thread(
            inbox.set_delegation, recipient.project_id, sender.project_id,
            "question", enabled=False,
        )
    return message_id, {
        "kind": "project_inbox", "notification_id": message_id,
        "project_id": recipient.project_id, "message_ids": [message_id],
        "messages": [message], "text": "Requests are work to do",
    }


async def _stage_delivery(
    opened: object, recipient: object, registry: ProjectRegistry,
    message_id: str, entry_data: dict[str, object],
) -> object:
    loop = AgentLoop(
        FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty(),
        root_project_id=recipient.project_id, project_registry=registry,
    )
    await loop.activate()
    entry, _appended = opened.store.append_inbox_notification_if_absent(
        recipient.project_id, [message_id]
    )
    if entry is None:
        entry = next(
            item for item in opened.store.agent_notifications()
            if item.data.get("kind") == "project_inbox"
        )
    opened.store.append_message(harness_notice(
        "background update", entry_data["text"], event="agent_notifications",
        metadata={"notifications": [entry_data], "zeta.step_delivery": {
            "version": 1, "batch_id": "legacy", "notification_ids": [entry.id]
        }}, visible=False,
    ))
    await loop.close()
    return entry


@pytest.mark.parametrize("case", ["legacy", "agent", "delegated", "revoked"])
@pytest.mark.parametrize("epoch", [6, 7])
@pytest.mark.parametrize("route", ["model", "live", "rpc"])
@pytest.mark.asyncio
async def test_inbox_provenance_routes(
    tmp_path: Path, case: str, epoch: int, route: str
) -> None:
    """Every public route projects the same current inbox authority."""
    home, registry, sender, recipient, manager, opened, inbox = await _create_fixtures(
        tmp_path, epoch
    )
    message_id, entry_data = await _seed_record(inbox, case, epoch, sender, recipient, opened)
    entry = await _stage_delivery(opened, recipient, registry, message_id, entry_data)
    session_id = opened.store.session_id
    opened.store.close()

    if route == "model":
        await _run_model_route(
            manager, session_id, recipient.project_id, registry, epoch, case
        )
        return
    if route == "live":
        await _run_live_route(
            manager, session_id, recipient.project_id, registry, entry, epoch
        )
        return
    await _run_rpc_route(
        tmp_path, home, registry, manager, recipient.project_id, message_id,
        session_id, case, epoch,
    )


def test_legacy_and_missing_origin_fail_closed(tmp_path: Path) -> None:
    home, registry, sender, recipient = _projects(tmp_path)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = inbox.send_user(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=recipient.project_id,
        kind="question",
        title="legacy",
        body="body",
    )
    path = home / "projects" / recipient.project_id / "inbox" / "new" / f"{message_id}.json"
    value = json.loads(path.read_text())
    value.pop("origin")
    path.write_text(json.dumps(value))
    listed = inbox.list(recipient.project_id)["new"][0]
    assert listed["author"] == "agent"
    assert listed["authority"] == "information"

    value["origin"] = "local"
    path.write_text(json.dumps(value))
    listed = inbox.list(recipient.project_id)["new"][0]
    assert listed["author"] == "agent"
    assert listed["authority"] == "information"


@pytest.mark.asyncio
async def test_epoch_five_refuses_delegation_through_tui_and_serve_slash_paths(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".zeta"
    home.mkdir()
    (home / "storage-format.json").write_text('{"writer_epoch":5}\n')
    def create_projects() -> tuple[ProjectRegistry, object, object]:
        registry = ProjectRegistry(home / "projects")
        return registry, registry.create_project("alpha", "alpha work"), registry.create_project("beta", "beta work")

    registry, sender, recipient = await asyncio.to_thread(create_projects)
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(
        inbox.send,
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=recipient.project_id,
        kind="question",
        title="old home",
        body="body",
    )
    tui = SimpleNamespace(
        loop=SimpleNamespace(
            project_registry=registry,
            session_metadata=SimpleNamespace(project_id=recipient.project_id, session_id="b" * 32),
        )
    )
    serve = SimpleNamespace(
        manager=SimpleNamespace(project_registry=registry),
        opened=SimpleNamespace(metadata=SimpleNamespace(project_id=recipient.project_id, session_id="b" * 32)),
        loop=SimpleNamespace(project_registry=registry),
    )
    tui_result = await SlashHandlerMixin.slash_inbox(
        tui, f"delegate {sender.project_id} question"
    )
    serve_result = await ServerSlashSession(serve).slash_inbox(
        f"delegate {sender.project_id} question"
    )
    assert "require stored-format epoch 7" in tui_result
    assert "require stored-format epoch 7" in serve_result
    assert not (home / "projects" / recipient.project_id / "inbox" / "delegations.json").exists()
    listed = (await asyncio.to_thread(inbox.list, recipient.project_id))["new"][0]
    assert listed["id"] == message_id
    stored = json.loads(
        (home / "projects" / recipient.project_id / "inbox" / "new" / f"{message_id}.json").read_text()
    )
    assert "origin" not in stored
    claimed = await asyncio.to_thread(
        inbox.claim, recipient.project_id, message_id, "b" * 32
    )
    assert claimed is not None
    assert "authority" not in claimed
