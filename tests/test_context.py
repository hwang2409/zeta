from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from zeta.context import BudgetExceeded, ContextAssembler
from zeta.fake import FakeBackend, ScriptedTurn
from zeta.store import ConversationStore
from zeta.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


@pytest.fixture(scope="module")
def context_root() -> Path:
    with TemporaryDirectory(prefix="zeta-context-") as directory:
        yield Path(directory)


def text(role: MessageRole, value: str) -> Message:
    return Message(role, [TextContent(value)])


def count(message: Message) -> int:
    return sum(
        len(block.text)
        for block in message.content
        if isinstance(block, TextContent)
    ) + 1


def compact_count(message: Message) -> int:
    return 1 if message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION} or message.metadata.get("compaction_summary") else count(message)


@pytest.mark.asyncio
async def test_retained_tail_is_verbatim(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(text(MessageRole.ASSISTANT, "recent one"))
    store.append_message(text(MessageRole.USER, "recent two"))

    assembler = ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=2,
        system_prompt="system",
    )

    messages = await assembler.assemble()

    assert [message.to_dict() for message in messages[-2:]] == [
        text(MessageRole.ASSISTANT, "recent one").to_dict(),
        text(MessageRole.USER, "recent two").to_dict(),
    ]


@pytest.mark.asyncio
async def test_compaction_requires_non_tail_content(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "tail"))
    assembler = ContextAssembler(
        store,
        token_budget=2,
        retained_tail=1,
        token_counter=lambda _: 2,
    )

    with pytest.raises(BudgetExceeded, match="retained tail"):
        await assembler.assemble()


@pytest.mark.asyncio
async def test_tool_call_and_result_force_tail_extension(context_root: Path) -> None:
    call = ToolCall("call-1", "read", {"path": "a"})
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("result")],
            tool_result=ToolResult(call.id, "result"),
        )
    )
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=35,
        retained_tail=2,
        token_counter=count,
        backend=backend,
    )

    messages = await assembler.assemble()

    assert any(
        any(
            isinstance(block, ToolUseContent) and block.tool_call.id == call.id
            for block in message.content
        )
        for message in messages
    )
    assert any(
        message.tool_result is not None and message.tool_result.tool_call_id == call.id
        for message in messages
    )


@pytest.mark.asyncio
async def test_summary_completion_has_no_tools(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old content"))
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=8,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    await assembler.assemble()

    assert backend.calls[0][1] == []


@pytest.mark.asyncio
async def test_failed_summary_leaves_store_unchanged(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old content"))
    store.append_message(text(MessageRole.USER, "tail"))
    before = store.path.read_bytes()
    backend = FakeBackend([ScriptedTurn()])
    assembler = ContextAssembler(
        store,
        token_budget=8,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    with pytest.raises(RuntimeError):
        await assembler.assemble()

    assert store.path.read_bytes() == before
    assert not any(entry.type == "compaction" for entry in store.entries)


@pytest.mark.asyncio
async def test_marker_replays_as_digest_not_source_range(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old content"))
    store.append_message(text(MessageRole.ASSISTANT, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=8,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    messages = await assembler.assemble()
    replayed = await assembler.assemble()

    assert [entry.type for entry in store.replay()] == ["message", "message", "compaction"]
    assert [message.role for message in replayed] == [
        MessageRole.SYSTEM,
        MessageRole.COMPACTION,
        MessageRole.ASSISTANT,
        MessageRole.ASSISTANT,
    ]
    assert all(
        "old content" not in (block.text if isinstance(block, TextContent) else "")
        for message in replayed
        for block in message.content
    )
    assert messages[-1] == replayed[-1]
    assert assembler.digest is not None


@pytest.mark.asyncio
async def test_provider_usage_informs_next_assembly(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=20,
        retained_tail=1,
        token_counter=lambda _: 1,
        backend=backend,
    )
    await assembler.assemble()
    assembler.record_usage({"total_tokens": 100})

    with pytest.raises(BudgetExceeded):
        await assembler.assemble()


@pytest.mark.asyncio
async def test_compaction_is_idempotent_for_same_store_state(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old content"))
    store.append_message(text(MessageRole.ASSISTANT, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=8,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    await assembler.assemble()
    before = store.path.read_bytes()
    await assembler.assemble()

    assert store.path.read_bytes() == before
    assert len([entry for entry in store.entries if entry.type == "compaction"]) == 1
