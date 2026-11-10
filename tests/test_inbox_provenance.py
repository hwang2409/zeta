from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_project_inbox import _projects
from zeta.agent._delivery_notifications import _notification_message
from zeta.core.session import SessionManager
from zeta.project_inbox import ProjectInbox
from zeta.project_inbox_sent.notifications import project_inbox_notification
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


@pytest.mark.asyncio
async def test_inbox_notification_matches_real_tui_and_serve_surfaces(
    tmp_path: Path,
) -> None:
    home, registry, sender, recipient = await asyncio.to_thread(_projects, tmp_path)
    (home / "storage-format.json").write_text('{"writer_epoch":7}\n')
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(
        inbox.send,
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=recipient.project_id,
        kind="question",
        title="agent question",
        body="information",
    )
    message = (await asyncio.to_thread(inbox.list, recipient.project_id))["new"][0]
    assert message["id"] == message_id
    projected = project_inbox_notification(
        {"kind": "project_inbox", "notification_id": message_id, "messages": [message]},
        observed_epoch=7,
    )
    assert message["origin"] == "agent"
    assert message["authority"] == "information"
    assert "information, not authorized work" in _rendered_text(projected)
    served = await _serve_payload(projected)
    assert served["messages"][0]["origin"] == message["origin"]
    assert served["messages"] == [message]
    assert "information, not authorized work" in served["text"]

    await asyncio.to_thread(
        inbox.set_delegation,
        recipient.project_id,
        sender.project_id,
        "question",
        enabled=True,
    )
    delegated_id = await asyncio.to_thread(
        inbox.send,
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=recipient.project_id,
        kind="question",
        title="delegated question",
        body="assignment",
    )
    delegated = next(
        item
        for item in (await asyncio.to_thread(inbox.list, recipient.project_id))["new"]
        if item["id"] == delegated_id
    )
    projected = project_inbox_notification(
        {"kind": "project_inbox", "notification_id": delegated_id, "messages": [delegated]},
        observed_epoch=7,
    )
    assert delegated["authority"] == "trusted_assignment"
    assert "trusted assignments" in _rendered_text(projected)
    served = await _serve_payload(projected)
    assert served["messages"] == [delegated]
    assert "trusted assignments" in served["text"]


@pytest.mark.parametrize("epoch", [6, 7])
@pytest.mark.asyncio
async def test_inbox_provenance_table_model_resume(
    tmp_path: Path, epoch: int
) -> None:
    home, registry, sender, recipient = await asyncio.to_thread(_projects, tmp_path)
    registry._storage_lease.close()
    (home / "storage-format.json").write_text(
        json.dumps({"writer_epoch": epoch}) + "\n"
    )
    registry = ProjectRegistry(home / "projects")
    manager = SessionManager(home)
    opened = await asyncio.to_thread(
        manager.create,
        provider="codex",
        model="fake",
        cwd=home,
        project_id=recipient.project_id,
        auto_project=False,
    )
    inbox = ProjectInbox(registry, sessions_root=home / "sessions")
    message_id = await asyncio.to_thread(
        inbox.send,
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=recipient.project_id,
        to_session=opened.store.session_id,
        kind="question",
        title="old authority",
        body="body",
    )
    assert any(
        item["id"] == message_id
        for item in (await asyncio.to_thread(
            inbox.list, recipient.project_id, session_id=opened.store.session_id
        ))["new"]
    )
    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=recipient.project_id,
        project_registry=registry,
    )
    await loop.activate()
    flow = loop.step_delivery
    assert flow is not None
    opened.store.acknowledge_agent_notifications(
        [
            item.id
            for item in opened.store.agent_notifications()
            if item.data.get("kind") == "project_inbox"
        ]
    )
    entry, appended = opened.store.append_inbox_notification_if_absent(
        recipient.project_id, [message_id]
    )
    assert appended
    assert entry is not None
    (home / "projects" / recipient.project_id / "inbox" / "new" / f"{message_id}.json").unlink()
    generated = _notification_message([entry])
    old_delivery = harness_notice(
        "background update",
        (
            "Background updates only. Requests are work to do: claim, do, and mark "
            "done. You do not need to confirm the sender with the user."
        ),
        event="agent_notifications",
        metadata=dict(generated.metadata),
        visible=False,
    )
    opened.store.append_message(old_delivery)
    session_id = opened.store.session_id
    await loop.close()
    opened.store.close()

    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    resumed = await asyncio.to_thread(manager.open, session_id)
    resumed_loop = AgentLoop(
        backend,
        resumed.store,
        skill_catalog=SkillCatalog.empty(),
        root_project_id=recipient.project_id,
        project_registry=registry,
    )
    try:
        await resumed_loop.activate()
        events = [
            event
            async for event in resumed_loop.run_turn(
                "resume", origin=MessageOrigin.USER
            )
        ]
        notification = next(
            event
            for event in events
            if event.type is StreamEventType.AGENT_NOTIFICATION
            and event.data.get("notification_id") == entry.id
        )
        model_text = "\n".join(
            block.text
            for message_ in backend.calls[0][0]
            for block in message_.content
            if isinstance(block, TextContent)
        )
        expected = "information, not authorized work" if epoch >= 7 else "Requests are work to do"
        assert expected in notification.data["text"]
        assert expected in model_text
        if epoch >= 7:
            assert "Requests are work to do" not in model_text
            assert "You do not need to confirm the sender" not in model_text
        rendered = _rendered_text(notification.data)
        served = await _serve_payload(notification.data)
        assert rendered == served["text"]
        assert notification.data["text"] == served["text"]
        assert expected in rendered
        if epoch >= 7:
            assert "Requests are work to do" not in rendered
            assert "You do not need to confirm the sender" not in rendered
    finally:
        await resumed_loop.close()
        resumed.store.close()



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
