import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.branch_reads import branch_messages
from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.agent.background_lifecycle import recover_agent_children
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.core.transcript_recovery.runtime import (
    append_child_repair_notification,
    repair_transcript_for_runtime,
)
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    with_message_origin,
)
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.storage_format import WRITER_EPOCH
from zeta.tui.slash_handlers import SlashHandlerMixin


@pytest.mark.asyncio
async def test_corrupt_child_stays_unavailable_until_tui_repair_then_reopens(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = await await_or_fail(
        asyncio.to_thread(
            manager.create, provider="codex", model="offline", cwd=tmp_path
        ),
        what="create root session",
    )
    root = opened.store
    child = ConversationStore(
        root.session_dir / "agents", session_id="1", cwd=tmp_path
    )
    child.append_message(
        with_message_origin(
            Message(MessageRole.USER, [TextContent("retained")]),
            MessageOrigin.USER,
        )
    )
    child.close()
    child_path = root.session_dir / "agents" / "1"
    transcript = child_path / "conversation.jsonl"
    transcript.write_bytes(transcript.read_bytes() + b"{not json}\n")
    damaged = transcript.read_bytes()

    root.register_agent_child(
        ToolCall("child-call", "agent", {}),
        child_session_path=str(child_path),
        description="worker",
        agent_type="worker",
        child_instance_id=f"{root.session_id}:1",
        background=True,
        accepts_follow_ups=True,
    )
    restarted = AgentLoop(FakeBackend([]), root, skill_catalog=SkillCatalog.empty())
    recover_agent_children(restarted, WRITER_EPOCH)
    assert root.agent_notifications() == []
    assert transcript.read_bytes() == damaged
    assert len(root.agent_children()) == 1
    assert root.agent_notifications() == []

    tui = SimpleNamespace(loop=SimpleNamespace(manager=manager, store=root))
    output = await SlashHandlerMixin.slash_repair_transcript(
        tui, f"{root.session_id} {root.session_id}:1 confirm"
    )
    assert "repaired transcript" in output

    reopened = ConversationStore(
        root.session_dir / "agents", session_id="1", cwd=tmp_path
    )
    try:
        reopened.append_message(
            with_message_origin(
                Message(MessageRole.USER, [TextContent("usable")]),
                MessageOrigin.USER,
            )
        )
        assert [message.content[0].text for message in branch_messages(reopened)] == [
            "retained",
            "usable",
        ]
    finally:
        reopened.close()

    notices = root.agent_notifications()
    assert len(notices) == 1
    recovery = root.agent_notifications()[0]
    assert recovery.data["status"] == "error"
    assert "child transcript repaired" in recovery.data["text"]


@pytest.mark.asyncio
async def test_child_repair_notice_deduplicates_after_confirmed_repair(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    opened = await await_or_fail(
        asyncio.to_thread(
            manager.create, provider="codex", model="offline", cwd=tmp_path
        ),
        what="create root session",
    )
    root = opened.store
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    child.close()
    transcript = child.path
    transcript.write_bytes(transcript.read_bytes() + b"{not json}\n")

    result = await repair_transcript_for_runtime(
        manager,
        root.session_id,
        f"{root.session_id}:1",
        parent_store=root,
    )
    _, appended = append_child_repair_notification(
        root, result, child_session_path=child.session_dir
    )
    assert appended is False
    assert len(root.agent_notifications()) == 1
