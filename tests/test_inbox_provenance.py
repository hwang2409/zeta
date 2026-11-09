from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_project_inbox import _projects
from zeta.core.project_inbox_notifications import project_inbox_notification
from zeta.project_inbox import ProjectInbox
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.server.control import NotificationPresentation
from zeta.server.slash_commands import ServerSlashSession
from zeta.tui.render import render_event
from zeta.tui.slash_handlers import SlashHandlerMixin


async def _serve_payload(data: dict[str, object]) -> dict[str, object]:
    sent: list[dict[str, object]] = []

    async def send(_event: object, _session: object, *, data: dict[str, object]) -> bool:
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
    home = tmp_path / ".zeta"
    home.mkdir()
    (home / "storage-format.json").write_text('{"writer_epoch":7}\n')
    home, registry, sender, recipient = await asyncio.to_thread(_projects, tmp_path)
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
        {"kind": "project_inbox", "notification_id": message_id, "messages": [message]}
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
        {"kind": "project_inbox", "notification_id": delegated_id, "messages": [delegated]}
    )
    assert delegated["authority"] == "trusted_assignment"
    assert "trusted assignments" in _rendered_text(projected)
    served = await _serve_payload(projected)
    assert served["messages"] == [delegated]
    assert "trusted assignments" in served["text"]


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
    registry = ProjectRegistry(home / "projects")
    sender, recipient = await asyncio.to_thread(
        lambda: (
            registry.create_project("alpha", "alpha work"),
            registry.create_project("beta", "beta work"),
        )
    )
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
    assert listed["authority"] == "information"
