from __future__ import annotations

import json
from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from zeta.automations.runner import _receipt
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.slash import create_slash_registry
from zeta.core.store import ConversationStore
from zeta.mcp.prompt_commands import SlashModelInput
from zeta.memory.reconciler import Transcript, prepare_request
from zeta.protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
)
from zeta.runtime.driver import drive_turn
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.empty_turn import build_nudge_message
from zeta.skills import SkillCatalog, discover_session_skills
from zeta.tui._attachments import build_user_message

SESSION_ID = "a" * 32


def _rendered(row: dict[str, object]) -> dict[str, object]:
    request = prepare_request(
        Transcript(SESSION_ID, (row,)), {}, as_of=date(2026, 10, 7)
    )
    rows = json.loads(request.prompt.split("Completed transcript rows:\n", 1)[1])
    return rows[0]


def _label(row: dict[str, object]) -> str:
    return str(_rendered(row)["authorship"])


def _message_row(message: Message) -> dict[str, object]:
    return {"seq": 1, "type": "message", "data": {"message": message.to_dict()}}


def test_typed_tui_message_is_labeled_user(tmp_path: Path) -> None:
    message = build_user_message("typed in TUI", tmp_path)

    assert message.metadata[MESSAGE_ORIGIN_METADATA] == MessageOrigin.USER
    assert _label(_message_row(message)) == "user"


@pytest.mark.asyncio
async def test_serve_user_message_is_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    async for _event in loop.run_turn("sent through serve"):
        pass

    assert _label(store.entries[0].to_dict()) == "user"
    await loop.close()


@pytest.mark.asyncio
async def test_headless_prompt_is_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    code = await drive_turn(
        loop,
        "sent with -p",
        format="text",
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    assert _label(store.entries[0].to_dict()) == "user"
    await loop.close()


@pytest.mark.asyncio
async def test_automation_prompt_is_not_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(
        FakeBackend([ScriptedTurn([TextContent("done")])]),
        store,
        skill_catalog=SkillCatalog.empty(),
    )

    code = await drive_turn(
        loop,
        "saved automation prompt",
        format="text",
        stdout=StringIO(),
        stderr=StringIO(),
        origin=MessageOrigin.AUTOMATION_PROMPT,
    )

    assert code == 0
    assert _label(store.entries[0].to_dict()) == "automation_prompt"
    await loop.close()


def test_skill_expansion_is_not_labeled_user(tmp_path: Path) -> None:
    skill = tmp_path / ".zeta" / "skills" / "review.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: review\ndescription: Review code\n---\nReview carefully.",
        encoding="utf-8",
    )
    registry = create_slash_registry(
        project_dir=tmp_path,
        skill_catalog=discover_session_skills(project_dir=tmp_path),
    )

    expansion = registry.dispatch(object(), "$review this")

    assert isinstance(expansion, SlashModelInput)
    message = Message(
        MessageRole.USER,
        [TextContent(expansion.text)],
        metadata={
            MESSAGE_ORIGIN_METADATA: expansion.origin.value,
            "zeta.user_display_text": expansion.display_text,
        },
    )
    rendered = _rendered(_message_row(message))
    assert rendered["authorship"] == "skill_expansion"
    assert rendered["user_authored_input"] == {
        "authorship": "user",
        "text": "$review this",
    }


def test_slash_expansion_is_not_labeled_user() -> None:
    expansion = SlashModelInput("expanded slash prompt")
    message = Message(
        MessageRole.USER,
        [TextContent(expansion.text)],
        metadata={MESSAGE_ORIGIN_METADATA: expansion.origin.value},
    )

    assert _label(_message_row(message)) == "slash_expansion"


def test_agent_send_pending_prompt_is_not_labeled_user(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    entry = store.pending_prompt_queue.append(
        "agent follow-up", origin=MessageOrigin.AGENT_SEND
    )

    assert _label(entry.to_dict()) == "agent_send"


def test_empty_turn_nudge_is_not_labeled_user() -> None:
    assert _label(_message_row(build_nudge_message())) == "harness_nudge"


def test_agent_completion_notification_is_not_labeled_user(tmp_path: Path) -> None:
    entry = ConversationStore(tmp_path).append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="review",
        status="completed",
        text="done",
    )

    assert entry.data["origin"] == MessageOrigin.NOTIFICATION
    assert _label(entry.to_dict()) == "harness_notification"


def test_automation_receipt_is_labeled_harness(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    _receipt(store, "delivered")

    assert _label(store.entries[-1].to_dict()) == "harness"


def test_new_unmarked_user_message_persists_unknown_origin(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    entry = store.append_message(
        Message(MessageRole.USER, [TextContent("unattributed text")])
    )

    assert entry.data["message"]["metadata"][MESSAGE_ORIGIN_METADATA] == "unknown"
    assert _label(entry.to_dict()) == "harness_unknown"


def test_historical_unmarked_user_message_fails_closed() -> None:
    message = Message(MessageRole.USER, [TextContent("historical text")])

    assert _label(_message_row(message)) == "harness_unknown"
