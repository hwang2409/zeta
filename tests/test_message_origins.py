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
from zeta.mcp import MCPPrompt, MCPPromptArgument
from zeta.memory.reconciler import Transcript, prepare_request
from zeta.model_input import ModelInputEnvelope
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
from zeta.tui.slash_handlers import SlashHandlerMixin

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

    async for _event in loop.run_turn("sent through serve", origin=MessageOrigin.USER):
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
        "sent with -p", origin=MessageOrigin.USER,
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

    assert isinstance(expansion, ModelInputEnvelope)
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


def test_slash_skill_preserves_user_authored_input(tmp_path: Path) -> None:
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

    expansion = registry.dispatch(object(), "/review this branch")

    assert isinstance(expansion, ModelInputEnvelope)
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
        "text": "/review this branch",
    }


def _render_envelope(envelope: ModelInputEnvelope) -> dict[str, object]:
    return _rendered(
        _message_row(
            Message(
                MessageRole.USER,
                [TextContent(envelope.text)],
                metadata={
                    MESSAGE_ORIGIN_METADATA: envelope.origin.value,
                    "zeta.user_display_text": envelope.display_text,
                },
            )
        )
    )


def test_plan_preserves_exact_typed_input_as_nested_user_evidence() -> None:
    class Session(SlashHandlerMixin):
        active = False
        pending_approvals: tuple[object, ...] = ()
        plan_mode = False

        def __init__(self) -> None:
            self.loop = self

        def set_plan_mode(self, enabled: bool) -> None:
            self.plan_mode = enabled

        def _invalidate_prompt(self) -> None:
            pass

    typed = "/plan   inspect this exact branch"
    envelope = create_slash_registry(
        skill_catalog=SkillCatalog.empty()
    ).dispatch(Session(), typed)

    assert isinstance(envelope, ModelInputEnvelope)
    assert envelope.display_text == typed
    assert envelope.origin is MessageOrigin.SLASH_EXPANSION
    assert _render_envelope(envelope)["user_authored_input"] == {
        "authorship": "user",
        "text": typed,
    }


@pytest.mark.asyncio
async def test_mcp_prompt_preserves_exact_typed_input_as_nested_user_evidence() -> None:
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())
    registry.set_mcp_prompts(
        [
            (
                "server:review",
                "server",
                MCPPrompt(
                    "review",
                    "review code",
                    (MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )

    class Session:
        async def slash_mcp_prompt(
            self, name: str, arguments: dict[str, str]
        ) -> str:
            return f"resolved {name} {arguments['topic']}"

    typed = "/server:review   exact topic"
    envelope = await registry.dispatch_async(Session(), typed)

    assert isinstance(envelope, ModelInputEnvelope)
    assert envelope.display_text == typed
    assert envelope.origin is MessageOrigin.SLASH_EXPANSION
    assert _render_envelope(envelope)["user_authored_input"] == {
        "authorship": "user",
        "text": typed,
    }


def test_slash_expansion_is_not_labeled_user() -> None:
    expansion = ModelInputEnvelope(
        "expanded slash prompt",
        "/generated",
        MessageOrigin.SLASH_EXPANSION,
    )
    message = Message(
        MessageRole.USER,
        [TextContent(expansion.text)],
        metadata={MESSAGE_ORIGIN_METADATA: expansion.origin.value},
    )

    assert _label(_message_row(message)) == "slash_expansion"


_MODEL_INPUT_AUDIT = (
    ("plain text", MessageOrigin.USER, "plain text"),
    ("$skill", MessageOrigin.SKILL_EXPANSION, "$review this"),
    ("/skill", MessageOrigin.SKILL_EXPANSION, "/review this"),
    ("custom command", MessageOrigin.SLASH_EXPANSION, "/custom this"),
    ("custom inline shell", MessageOrigin.SLASH_EXPANSION, "/custom-shell"),
    ("/plan", MessageOrigin.SLASH_EXPANSION, "/plan inspect"),
    ("/init", MessageOrigin.SLASH_EXPANSION, "/init"),
    ("/implement", MessageOrigin.SLASH_EXPANSION, "/implement"),
    ("MCP prompt", MessageOrigin.SLASH_EXPANSION, "/server:prompt value"),
    ("attachments/images", MessageOrigin.USER, "inspect @./image.png"),
    ("paste expansion", MessageOrigin.USER, "inspect [Image #1]"),
    ("steer/queued input", MessageOrigin.USER, "steer now"),
    ("serve send/steer", MessageOrigin.USER, None),
    ("-p", MessageOrigin.USER, None),
    ("inbox wake", MessageOrigin.NOTIFICATION, None),
    ("automation prompt", MessageOrigin.AUTOMATION_PROMPT, None),
)


@pytest.mark.parametrize(
    ("path", "origin", "display_text"),
    _MODEL_INPUT_AUDIT,
    ids=[case[0] for case in _MODEL_INPUT_AUDIT],
)
def test_model_input_audit_preserves_authorship_and_nested_user_input(
    path: str, origin: MessageOrigin, display_text: str | None
) -> None:
    metadata: dict[str, object] = {MESSAGE_ORIGIN_METADATA: origin.value}
    if display_text is not None:
        metadata["zeta.user_display_text"] = display_text
    message = Message(
        MessageRole.USER,
        [TextContent(f"model input from {path}")],
        metadata=metadata,
    )

    rendered = _rendered(_message_row(message))

    assert message.metadata[MESSAGE_ORIGIN_METADATA] == origin.value
    expected_authorship = (
        "harness_unknown"
        if origin is MessageOrigin.NOTIFICATION
        else origin.value
    )
    assert rendered["authorship"] == expected_authorship
    if origin in {MessageOrigin.SKILL_EXPANSION, MessageOrigin.SLASH_EXPANSION}:
        assert rendered["user_authored_input"] == {
            "authorship": MessageOrigin.USER.value,
            "text": display_text,
        }
    else:
        assert "user_authored_input" not in rendered


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


def _unmarked_user_message() -> Message:
    return Message(MessageRole.USER, [TextContent("unattributed text")])


def test_new_user_message_without_origin_is_rejected_by_store_append_paths(
    tmp_path: Path,
) -> None:
    for method_name in ("append_message", "append_message_with_approval_requests"):
        store = ConversationStore(tmp_path / method_name)
        before = store.path.read_bytes()
        with pytest.raises(ValueError, match="origin"):
            getattr(store, method_name)(_unmarked_user_message())
        assert store.entries == []
        assert store.path.read_bytes() == before


@pytest.mark.parametrize("origin", ["not-an-origin", MessageOrigin.UNKNOWN.value])
def test_new_user_message_with_invalid_origin_is_rejected(
    tmp_path: Path, origin: str
) -> None:
    store = ConversationStore(tmp_path)
    message = Message(
        MessageRole.USER,
        [TextContent("unattributed text")],
        metadata={MESSAGE_ORIGIN_METADATA: origin},
    )

    with pytest.raises(ValueError, match="origin"):
        store.append_message(message)


def test_new_user_message_without_origin_is_rejected_by_steer(tmp_path: Path) -> None:
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path),
        skill_catalog=SkillCatalog.empty(),
    )

    with pytest.raises(ValueError, match="origin"):
        loop.steer(_unmarked_user_message())

    assert loop.has_pending_steering is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin", "message"),
    [
        (MessageOrigin.USER, _unmarked_user_message()),
        (MessageOrigin.UNKNOWN, None),
    ],
    ids=["missing-metadata", "unknown-origin"],
)
async def test_new_user_message_without_origin_is_rejected_by_run_turn(
    tmp_path: Path, origin: MessageOrigin, message: Message | None
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())

    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="origin"):
        async for _event in loop.run_turn(
            "unattributed text", origin=origin, user_message=message
        ):
            pass

    assert store.entries == []
    assert store.path.read_bytes() == before
    await loop.close()


def test_historical_unmarked_user_row_still_replays_and_reconciles_as_harness_unknown(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="historical-session")
    store.append_message(
        Message(
            MessageRole.USER,
            [TextContent("historical text")],
            metadata={MESSAGE_ORIGIN_METADATA: MessageOrigin.USER.value},
        )
    )
    store.close()
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    rows[-1]["data"]["message"].pop("metadata")
    store.path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    reopened = ConversationStore(tmp_path, session_id="historical-session")

    assert reopened.messages() == [
        Message(MessageRole.USER, [TextContent("historical text")])
    ]
    assert _label(reopened.entries[-1].to_dict()) == "harness_unknown"
