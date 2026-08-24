from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pytest

from zeta.core.context import (
    BudgetExceeded,
    CompactionPolicy,
    ContextAssembler,
    StaleBranchError,
    SummaryInputTooLarge,
)
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    StreamEvent,
    StreamEventType,
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
    if (
        message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION}
        or message.metadata.get("compaction_summary")
    ):
        return 1
    return 30


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
        token_budget=1,
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
        token_budget=40,
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
        token_budget=40,
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
        token_budget=40,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    messages = await assembler.assemble()
    replayed = await assembler.assemble()

    assert [entry.type for entry in store.replay()] == ["message", "message", "compaction"]
    assert [message.role for message in replayed] == [
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
async def test_cached_provider_usage_triggers_compaction(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=30,
        retained_tail=1,
        token_counter=lambda _: 1,
        backend=backend,
    )

    await assembler.assemble()
    assembler.record_usage(
        {
            "input_tokens": 18,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "output_tokens": 7,
        }
    )

    compacted = await assembler.assemble_context()

    assert compacted.compacted is True
    assert assembler.tokens_used_this_session == 40
    assert assembler.cache_read_input_tokens_this_session == 10
    assert assembler.cache_creation_input_tokens_this_session == 5
    assert assembler.uncached_input_tokens_this_session == 18
    assert assembler.output_tokens_this_session == 7


@pytest.mark.parametrize(
    ("usage", "expected_total"),
    [
        ({"input_tokens": 8, "cache_read_input_tokens": 5}, 13),
        ({"cache_read_input_tokens": 5, "output_tokens": 2}, 7),
    ],
)
def test_partial_provider_usage_counts_known_tokens(
    context_root: Path,
    usage: dict[str, int],
    expected_total: int,
) -> None:
    assembler = ContextAssembler(ConversationStore(context_root))

    assembler.record_usage(usage)

    assert assembler.tokens_used_this_session == expected_total


@pytest.mark.asyncio
async def test_compaction_is_idempotent_for_same_store_state(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old content"))
    store.append_message(text(MessageRole.ASSISTANT, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=40,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    await assembler.assemble()
    before = store.path.read_bytes()
    await assembler.assemble()

    assert store.path.read_bytes() == before
    assert len([entry for entry in store.entries if entry.type == "compaction"]) == 1


@pytest.mark.asyncio
async def test_over_budget_compaction_does_not_persist_marker(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])

    def output_count(message: Message) -> int:
        if message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION}:
            return 1
        if message.metadata.get("compaction_summary"):
            return 100
        return 30

    assembler = ContextAssembler(
        store,
        token_budget=40,
        retained_tail=1,
        token_counter=output_count,
        backend=backend,
    )
    before = store.path.read_bytes()

    with pytest.raises(BudgetExceeded, match="compacted context"):
        await assembler.assemble()

    assert store.path.read_bytes() == before
    assert not any(entry.type == "compaction" for entry in store.entries)


@pytest.mark.asyncio
async def test_repeated_compaction_replays_flattened_marker_range(
    context_root: Path,
) -> None:
    store = ConversationStore(context_root)
    for index in range(5):
        store.append_message(text(MessageRole.USER, f"old {index}"))
    store.append_message(text(MessageRole.USER, "first tail"))
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first summary")]),
            ScriptedTurn([TextContent("second summary")]),
        ]
    )

    def repeat_count(message: Message) -> int:
        if (
            message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION}
            or message.metadata.get("compaction_summary")
        ):
            return 1
        return 200

    assembler = ContextAssembler(
        store,
        token_budget=500,
        retained_tail=1,
        token_counter=repeat_count,
        backend=backend,
    )

    await assembler.assemble()
    for value in ("new one", "new two", "new tail"):
        store.append_message(text(MessageRole.USER, value))
    await assembler.assemble()

    markers = [entry for entry in store.entries if entry.type == "compaction"]
    assert [(entry.data["source_seq_start"], entry.data["source_seq_end"]) for entry in markers] == [
        (1, 5),
        (1, 9),
    ]
    assert markers[0].data["replaces"] == []
    assert markers[1].data["replaces"] == [markers[0].id]

    reopened = ConversationStore(context_root, session_id=store.session_id)
    fresh = ContextAssembler(
        reopened,
        token_budget=500,
        retained_tail=1,
        token_counter=repeat_count,
    )
    messages = await fresh.assemble()

    assert [message.role for message in messages] == [
        MessageRole.COMPACTION,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert [
        message.metadata.get("compaction_summary")
        for message in messages
        if message.role is MessageRole.ASSISTANT
    ] == [True]
    assert all(
        value not in (block.text if isinstance(block, TextContent) else "")
        for message in messages
        for block in message.content
        for value in ("old 0", "old 4")
    )


@pytest.mark.asyncio
async def test_repeated_compaction_replacement_is_idempotent(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_compaction_marker("summary", 1, 1)
    store.append_message(text(MessageRole.USER, "tail"))
    assembler = ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=1,
        token_counter=compact_count,
    )

    first = [message.to_dict() for message in await assembler.assemble()]
    before = store.path.read_bytes()
    second = [message.to_dict() for message in await assembler.assemble()]

    assert second == first
    assert store.path.read_bytes() == before


@pytest.mark.asyncio
async def test_same_state_recompaction_survives_cold_reload(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first summary")]),
            ScriptedTurn([TextContent("second summary")]),
        ]
    )

    def same_state_count(message: Message) -> int:
        if (
            message.role in {MessageRole.SYSTEM, MessageRole.COMPACTION}
            or message.metadata.get("compaction_summary")
        ):
            return 1
        return 100

    assembler = ContextAssembler(
        store,
        token_budget=160,
        retained_tail=1,
        token_counter=same_state_count,
        backend=backend,
    )

    await assembler.assemble()
    first_marker = next(entry for entry in store.entries if entry.type == "compaction")
    assembler.record_usage({"total_tokens": 300})
    await assembler.assemble()
    markers = [entry for entry in store.entries if entry.type == "compaction"]

    assert markers[1].data["replaces"] == [first_marker.id]
    reopened = ConversationStore(context_root, session_id=store.session_id)
    fresh = ContextAssembler(
        reopened,
        token_budget=160,
        retained_tail=1,
        token_counter=same_state_count,
    )

    messages = await fresh.assemble()

    assert [message.role for message in messages] == [
        MessageRole.COMPACTION,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]


@pytest.mark.asyncio
async def test_marker_replay_chain_scales_linearly(context_root: Path) -> None:
    store = ConversationStore(context_root)
    previous_id: str | None = None
    chain_length = 100
    for index in range(chain_length):
        marker = store.append_compaction_marker(
            f"summary {index}",
            1,
            1,
            replaces=[] if previous_id is None else [previous_id],
        )
        previous_id = marker.id

    assembler = ContextAssembler(store, token_budget=10_000, retained_tail=1)
    started = perf_counter()
    messages = await assembler.assemble()
    elapsed = perf_counter() - started

    assert elapsed < chain_length / 100
    assert [message.role for message in messages] == [
        MessageRole.COMPACTION,
        MessageRole.ASSISTANT,
    ]


def test_zero_retained_tail_is_rejected(context_root: Path) -> None:
    store = ConversationStore(context_root)

    with pytest.raises(ValueError, match="at least one"):
        ContextAssembler(store, retained_tail=0)


@pytest.mark.asyncio
async def test_stale_branch_discards_summary(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "old"))
    store.append_message(text(MessageRole.USER, "tail"))

    class RebranchingBackend:
        async def complete(self, messages: object, tool_schemas: object):
            store.append_message(text(MessageRole.USER, "rebranched"))
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, "summary"),
            )

    assembler = ContextAssembler(
        store,
        token_budget=40,
        retained_tail=1,
        token_counter=compact_count,
        backend=RebranchingBackend(),
    )

    with pytest.raises(StaleBranchError):
        await assembler.assemble()

    assert not any(entry.type == "compaction" for entry in store.entries)


@pytest.mark.asyncio
async def test_summary_source_bound_rejects_large_input(context_root: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("unused")])])
    policy = CompactionPolicy(backend)

    with pytest.raises(SummaryInputTooLarge):
        await policy.summarize(
            [text(MessageRole.USER, "x" * 200)],
            max_source_tokens=10,
        )

    assert backend.calls == []
