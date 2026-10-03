from pathlib import Path
import asyncio
import base64
from tempfile import TemporaryDirectory
from time import perf_counter

import pytest

from zeta.core.context import (
    BudgetExceeded,
    CompactionPolicy,
    ContextAssembler,
    StaleBranchError,
    SummaryInputTooLarge,
    SummaryCompletionError,
)
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex_errors import CodexStreamError
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.ollama import _messages as build_ollama_messages
from zeta.protocol.types import (
    CompletionBackend,
    ErrorInfo,
    Message,
    MessageRole,
    ImageContent,
    RedactedThinkingContent,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    StreamEvent,
    StreamEventType,
    ThinkingContent,
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
async def test_failed_assistant_output_is_excluded_from_retry_context(
    context_root: Path,
) -> None:
    store = ConversationStore(context_root)
    store.append_message(text(MessageRole.USER, "prompt"))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("partial")],
            metadata={
                "turn_failed": True,
                "turn_error": {"code": "stream_error", "message": "disconnected"},
            },
        )
    )

    assembled = await ContextAssembler(store).assemble()

    assert [message.role for message in assembled] == [MessageRole.USER]


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
async def test_oversized_retained_tail_shrinks_at_tool_group_boundary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    latest_user = text(MessageRole.USER, "latest request must remain verbatim")
    store.append_message(latest_user)
    for index in range(2):
        call = ToolCall(f"call-{index}", "bash", {"command": f"job {index}"})
        store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
        store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult(call.id, str(index) * 400),
            )
        )
    backend = FakeBackend(
        [ScriptedTurn([TextContent("summary")]) for _ in range(20)]
    )
    assembler = ContextAssembler(
        store,
        token_budget=290,
        retained_tail=8,
        backend=backend,
    )

    assembled = await assembler.assemble_context()

    assert assembled.compacted
    assert assembled.token_count <= assembler.token_budget
    assert [message.role for message in assembled.messages] == [
        MessageRole.COMPACTION,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL_RESULT,
    ]
    assert assembled.messages[2].to_dict() == latest_user.to_dict()
    assert [
        block.tool_call.id
        for message in assembled.messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    ] == ["call-1"]
    assert assembled.messages[-1].tool_result == ToolResult("call-1", "1" * 400)

    reopened = ContextAssembler(
        ConversationStore(tmp_path, session_id=store.session_id),
        token_budget=290,
        retained_tail=8,
    )
    replayed = await reopened.assemble_context()
    assert [message.to_dict() for message in replayed.messages] == [
        message.to_dict() for message in assembled.messages
    ]

    previous_marker = next(entry for entry in store.entries if entry.type == "compaction")
    next_request = text(MessageRole.USER, "next request")
    store.append_message(next_request)
    replacement = await ContextAssembler(
        store,
        token_budget=1_000,
        retained_tail=1,
        backend=FakeBackend([ScriptedTurn([TextContent("replacement summary")])]),
    ).assemble_context(force=True)
    markers = [entry for entry in store.entries if entry.type == "compaction"]
    assert markers[-1].data["replaces"] == [previous_marker.id]
    assert replacement.messages[-1].to_dict() == next_request.to_dict()
    assert all(
        "latest request must remain verbatim"
        not in (block.text if isinstance(block, TextContent) else "")
        for message in replacement.messages
        for block in message.content
    )


@pytest.mark.asyncio
async def test_adaptive_compaction_reuses_marker_with_request_only_truncation(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    old_call = ToolCall("old", "bash", {"command": "old"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(old_call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(old_call.id, "old" * 100),
        )
    )
    pinned = text(MessageRole.USER, "PINNED-VERBATIM-UNIQUE")
    store.append_message(pinned)
    latest_call = ToolCall("latest", "bash", {"command": "latest"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(latest_call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(latest_call.id, "latest" * 500),
        )
    )
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("FIRST")]),
            ScriptedTurn([TextContent("SECOND")]),
            ScriptedTurn([TextContent("THIRD")]),
        ]
    )
    assembler = ContextAssembler(
        store,
        token_budget=500,
        retained_tail=8,
        backend=backend,
    )

    assembled = [await assembler.assemble_context() for _ in range(3)]

    assert len([entry for entry in store.entries if entry.type == "compaction"]) == 1
    assert len(backend.calls) == 1
    assert {context.digest for context in assembled} == {assembled[0].digest}
    assert all(
        [message.to_dict() for message in context.messages]
        == [message.to_dict() for message in assembled[0].messages]
        for context in assembled
    )

    reopened = await ContextAssembler(
        ConversationStore(tmp_path, session_id=store.session_id),
        token_budget=500,
        retained_tail=8,
    ).assemble_context()

    assert reopened.digest == assembled[0].digest
    assert [message.to_dict() for message in reopened.messages] == [
        message.to_dict() for message in assembled[0].messages
    ]


@pytest.mark.asyncio
async def test_parallel_mixed_tool_results_keep_flags_and_pairing_when_truncated(
    tmp_path: Path,
) -> None:
    image = {
        "type": "image",
        "data": (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
            "+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg=="
        ),
        "mimeType": "image/png",
    }
    calls = [
        ToolCall("call-error", "bash", {"command": "fail"}),
        ToolCall("call-canceled", "bash", {"command": "wait"}),
        ToolCall("call-image", "read", {"path": "image.png"}),
    ]
    originals = [
        ToolResult(calls[0].id, "error " + "x" * 4_000, is_error=True),
        ToolResult(calls[1].id, "canceled " + "y" * 4_000, is_canceled=True),
        ToolResult(
            calls[2].id,
            "image receipt " + "z" * 4_000,
            content_blocks=[image],
            structured_content={"kind": "image", "count": 1},
        ),
    ]
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "inspect these tool results"))
    store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call) for call in calls])
    )
    for result in originals:
        store.append_message(Message(MessageRole.TOOL_RESULT, tool_result=result))
    assembler = ContextAssembler(
        store,
        token_budget=500,
        retained_tail=8,
        backend=FakeBackend([ScriptedTurn([TextContent("generic summary")])] * 20),
    )

    assembled = await assembler.assemble_context()

    results = [
        message.tool_result
        for message in assembled.messages
        if message.tool_result is not None
    ]
    assert len(results) == 3
    assert all("[output truncated for context:" in result.content for result in results)
    assert [(result.is_error, result.is_canceled) for result in results] == [
        (True, False),
        (False, True),
        (False, False),
    ]

    anthropic = build_messages_payload(
        assembled.messages, [], model="claude-sonnet-4-5", max_tokens=10_000
    )
    anthropic_results = [
        block
        for message in anthropic["messages"]
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert [result["tool_use_id"] for result in anthropic_results] == [
        call.id for call in calls
    ]
    assert [result["is_error"] for result in anthropic_results] == [True, False, False]

    codex = build_responses_payload(
        assembled.messages, [], model="gpt-5.6-luna"
    )["input"]
    assert [
        item["call_id"] for item in codex if item.get("type") == "function_call_output"
    ] == [call.id for call in calls]

    ollama = build_ollama_messages(assembled.messages)
    assert [message["tool_name"] for message in ollama if message["role"] == "tool"] == [
        call.name for call in calls
    ]


@pytest.mark.asyncio
async def test_minimal_tool_tail_is_truncated_without_mutating_store(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "old request"))
    call = ToolCall("call-large", "bash", {"command": "noisy"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    original_result = ToolResult(call.id, "head" + "x" * 4_000 + "tail")
    store.append_message(Message(MessageRole.TOOL_RESULT, tool_result=original_result))
    backend = FakeBackend(
        [ScriptedTurn([TextContent("summary")]) for _ in range(20)]
    )
    assembler = ContextAssembler(
        store,
        token_budget=240,
        retained_tail=8,
        backend=backend,
    )

    first = await assembler.assemble_context()
    second = await assembler.assemble_context()

    assert first.messages == second.messages
    assert first.digest == second.digest
    assembled_result = first.messages[-1].tool_result
    assert assembled_result is not None
    assert assembled_result.tool_call_id == call.id
    assert assembled_result.content.startswith("head")
    assert "tail\n[output truncated for context:" in assembled_result.content
    assert "full output is in the session log at seq 3" in assembled_result.content
    assert store.messages()[2].tool_result == original_result

    anthropic = build_messages_payload(
        first.messages, [], model="claude-sonnet-4-5", max_tokens=10_000
    )
    codex = build_responses_payload(first.messages, [], model="gpt-5.6-luna")
    ollama = build_ollama_messages(first.messages)
    assert any(
        block.get("tool_use_id") == call.id
        for message in anthropic["messages"]
        for block in message["content"]
        if block.get("type") == "tool_result"
    )
    assert any(item.get("call_id") == call.id for item in codex["input"])
    assert any(message.get("role") == "tool" for message in ollama)


@pytest.mark.asyncio
async def test_adaptive_compaction_uses_configured_budget(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.ASSISTANT, "older context"))
    store.append_message(text(MessageRole.USER, "latest request"))
    call = ToolCall("call-budget", "bash", {"command": "noisy"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(call.id, "result" * 200),
        )
    )

    def adaptive_count(message: Message) -> int:
        if message.role is MessageRole.USER:
            return 190
        if (
            message.role is MessageRole.COMPACTION
            or message.metadata.get("compaction_summary")
        ):
            return 5
        if message.tool_result is not None:
            return max(20, len(message.tool_result.content) // 10)
        return 20

    assembler = ContextAssembler(
        store,
        token_budget=260,
        retained_tail=8,
        token_counter=adaptive_count,
        backend=FakeBackend([ScriptedTurn([TextContent("summary")])] * 20),
    )

    assembled = await assembler.assemble_context()

    assert assembled.compacted
    assert assembled.token_count <= assembler.token_budget
    assert any(
        block.text == "latest request"
        for message in assembled.messages
        for block in message.content
        if isinstance(block, TextContent)
    )


@pytest.mark.asyncio
async def test_post_compaction_overflow_truncates_tool_result(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.ASSISTANT, "old" * 300))
    store.append_message(text(MessageRole.USER, "current request"))
    call = ToolCall("call-post", "bash", {"command": "noisy"})
    store.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(call)]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(call.id, "result" * 80),
        )
    )
    backend = FakeBackend(
        [ScriptedTurn([TextContent("summary" * 20)]) for _ in range(20)]
    )
    assembler = ContextAssembler(
        store,
        token_budget=260,
        retained_tail=2,
        backend=backend,
    )

    assembled = await assembler.assemble_context()

    assert assembled.compacted
    assert assembled.token_count <= assembler.token_budget
    result = assembled.messages[-1].tool_result
    assert result is not None
    assert "[output truncated for context:" in result.content
    assert store.messages()[-1].tool_result == ToolResult(call.id, "result" * 80)


@pytest.mark.asyncio
async def test_system_prompt_over_budget_still_raises(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "request"))
    assembler = ContextAssembler(
        store,
        token_budget=10,
        retained_tail=1,
        system_prompt="system prompt is much too large",
        token_counter=count,
    )

    with pytest.raises(BudgetExceeded, match="system prompt and retained tail"):
        await assembler.assemble()


@pytest.mark.asyncio
async def test_fitting_compaction_output_is_byte_identical(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.USER, "old" * 100))
    tail = text(MessageRole.USER, "current")
    store.append_message(tail)
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        backend=FakeBackend(
            [ScriptedTurn([TextContent("summary")]) for _ in range(10)]
        ),
    )

    assembled = await assembler.assemble()

    assert [message.to_dict() for message in assembled] == [
        Message(
            MessageRole.COMPACTION,
            [TextContent("[compaction marker: entries 1–1]")],
            metadata={"source_seq_start": 1, "source_seq_end": 1},
        ).to_dict(),
        Message(
            MessageRole.ASSISTANT,
            [TextContent("summary")],
            metadata={
                "compaction_summary": True,
                "source_seq_start": 1,
                "source_seq_end": 1,
            },
        ).to_dict(),
        tail.to_dict(),
    ]


@pytest.mark.asyncio
async def test_forced_compaction_excludes_pinned_user_from_summary_source(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(text(MessageRole.ASSISTANT, "older answer"))
    pinned = text(MessageRole.USER, "PINNED-VERBATIM-UNIQUE")
    store.append_message(pinned)
    store.append_message(text(MessageRole.ASSISTANT, "answer tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])

    assembled = await ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=1,
        backend=backend,
    ).assemble_context(force=True)

    source_prompt = backend.calls[0][0][-1].content[0]
    assert isinstance(source_prompt, TextContent)
    assert "older answer" in source_prompt.text
    assert "PINNED-VERBATIM-UNIQUE" not in source_prompt.text
    assert [message.to_dict() for message in assembled.messages][-2:] == [
        pinned.to_dict(),
        text(MessageRole.ASSISTANT, "answer tail").to_dict(),
    ]


@pytest.mark.asyncio
async def test_forced_compaction_uses_empty_source_when_only_pinned_user_precedes_tail(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    pinned = text(MessageRole.USER, "PINNED-VERBATIM-UNIQUE")
    tail = text(MessageRole.ASSISTANT, "answer tail")
    store.append_message(pinned)
    store.append_message(tail)
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])

    assembled = await ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=1,
        backend=backend,
    ).assemble_context(force=True)

    source_prompt = backend.calls[0][0][-1].content[0]
    assert isinstance(source_prompt, TextContent)
    assert source_prompt.text.endswith("\n\n[]")
    assert "PINNED-VERBATIM-UNIQUE" not in source_prompt.text
    assert [message.to_dict() for message in assembled.messages][-2:] == [
        pinned.to_dict(),
        tail.to_dict(),
    ]
    assert assembled.compacted
    assert len(backend.calls) == 1
    assert len([entry for entry in store.entries if entry.type == "compaction"]) == 1


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
async def test_compaction_strips_thinking_from_input_and_summary(context_root: Path) -> None:
    store = ConversationStore(context_root)
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [
                ThinkingContent("private plan", "signature-secret"),
                RedactedThinkingContent("redacted-secret"),
                TextContent("visible fact"),
            ],
            metadata={"codex_output_items": [{"encrypted_content": "opaque-secret"}]},
        )
    )
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend(
        [
            ScriptedTurn(
                [
                    ThinkingContent("summary private", "summary-signature"),
                    RedactedThinkingContent("summary-redacted"),
                    TextContent("summary visible"),
                ]
            )
        ]
    )
    assembler = ContextAssembler(
        store,
        token_budget=40,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    compacted = await assembler.assemble()

    source_prompt = backend.calls[0][0][-1].content[0].text
    assert "private plan" not in source_prompt
    assert "signature-secret" not in source_prompt
    assert "redacted-secret" not in source_prompt
    assert "opaque-secret" not in source_prompt
    assert '"thinking"' not in source_prompt
    assert '"redacted_thinking"' not in source_prompt
    assert all(
        isinstance(block, TextContent)
        for message in compacted
        for block in message.content
    )
    assert "summary visible" in "".join(
        block.text
        for message in compacted
        for block in message.content
        if isinstance(block, TextContent)
    )

    replayed = await assembler.assemble()

    assert [message.to_dict() for message in replayed] == [
        message.to_dict() for message in compacted
    ]
    assert all(
        isinstance(block, TextContent)
        for message in replayed
        for block in message.content
    )


@pytest.mark.asyncio
async def test_compaction_summary_replaces_image_base64_with_placeholder(
    context_root: Path,
) -> None:
    image_data = base64.b64encode(b"\xff\xd8\xff\xd9").decode()
    store = ConversationStore(context_root)
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            tool_result=ToolResult(
                "call-1",
                "stale",
                content_blocks=[
                    {
                        "type": "image",
                        "data": image_data,
                        "mimeType": "image/jpeg",
                        "caption": "a test image",
                    }
                ],
            ),
        )
    )
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=90,
        retained_tail=1,
        token_counter=lambda message: 100 if message.role is MessageRole.TOOL_RESULT else 1,
        backend=backend,
    )

    await assembler.assemble()

    summary_prompt = backend.calls[0][0][0].content[0].text
    assert image_data not in summary_prompt
    assert (
        "[image block] media_type=image/jpeg bytes=4 caption=a test image"
        in summary_prompt
    )


@pytest.mark.asyncio
async def test_compaction_summary_replaces_message_image_with_placeholder(
    context_root: Path,
) -> None:
    raw_image = b"\xff\xd8\xffimage bytes"
    image_data = base64.b64encode(raw_image).decode()
    store = ConversationStore(context_root)
    store.append_message(
        Message(
            MessageRole.USER,
            [
                ImageContent(
                    image_data, "image/jpeg", "/tmp/reference.png", len(raw_image)
                )
            ],
        )
    )
    store.append_message(text(MessageRole.USER, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=90,
        retained_tail=1,
        token_counter=lambda message: (
            100
            if message.role is MessageRole.USER
            and any(isinstance(block, ImageContent) for block in message.content)
            else 1
        ),
        backend=backend,
    )

    await assembler.assemble()

    summary_prompt = backend.calls[0][0][0].content[0].text
    assert image_data not in summary_prompt
    assert (
        "[image attachment] filename=reference.png media_type=image/jpeg bytes=14"
        in summary_prompt
    )


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
    store.append_message(text(MessageRole.ASSISTANT, "old content"))
    store.append_message(text(MessageRole.USER, "current request"))
    store.append_message(text(MessageRole.ASSISTANT, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=70,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )

    messages = await assembler.assemble()
    replayed = await assembler.assemble()

    assert [entry.type for entry in store.replay()] == [
        "message",
        "message",
        "message",
        "compaction",
    ]
    assert [message.role for message in replayed] == [
        MessageRole.COMPACTION,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert all(
        "old content" not in (block.text if isinstance(block, TextContent) else "")
        for message in replayed
        for block in message.content
    )
    assert replayed[2] == text(MessageRole.USER, "current request")
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
    store.append_message(text(MessageRole.ASSISTANT, "old content"))
    store.append_message(text(MessageRole.USER, "current request"))
    store.append_message(text(MessageRole.ASSISTANT, "tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    assembler = ContextAssembler(
        store,
        token_budget=70,
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


@pytest.mark.asyncio
async def test_compaction_summarizes_large_source_in_bounded_requests(
    context_root: Path,
) -> None:
    store = ConversationStore(context_root)
    old_parts = [f"fact-{index}-" + chr(65 + index) * 2_000 for index in range(4)]
    for part in old_parts:
        store.append_message(text(MessageRole.USER, part))
    store.append_message(text(MessageRole.USER, "current request"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])] * 20)
    assembler = ContextAssembler(
        store,
        token_budget=2_000,
        retained_tail=1,
        token_counter=lambda message: (
            1
            if message.role is MessageRole.COMPACTION
            or message.metadata.get("compaction_summary")
            else 500
        ),
        backend=backend,
    )

    await assembler.assemble()

    assert store.compaction_marker_count() == 1
    assert len(backend.calls) > 1
    sources = [call[0][-1].content[0].text.split("\n\n", 1)[1] for call in backend.calls]
    assert all(len(source) <= 4_000 for source in sources)
    assert all(any(part in source for source in sources) for part in old_parts)


@pytest.mark.asyncio
async def test_empty_summary_is_retried_with_stronger_instruction() -> None:
    telemetry: list[dict[str, object]] = []
    backend = FakeBackend(
        [ScriptedTurn([TextContent("   ")]), ScriptedTurn([TextContent("summary")])]
    )

    result = await CompactionPolicy(backend).summarize_chunked(
        [text(MessageRole.USER, "source")], on_telemetry=telemetry.append
    )

    assert result == "summary"
    assert len(backend.calls) == 2
    retry_prompt = backend.calls[1][0][-1].content[0]
    assert isinstance(retry_prompt, TextContent)
    assert "Return a non-empty summary" in retry_prompt.text
    assert telemetry[-1]["retries"] == 1


@pytest.mark.asyncio
async def test_summary_stream_error_is_retried() -> None:
    telemetry: list[dict[str, object]] = []

    class StreamErrorBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tool_schemas):
            self.calls += 1
            if self.calls == 1:
                raise CodexStreamError("Codex output item completed with open blocks")
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, "summary"),
            )

    backend = StreamErrorBackend()
    result = await CompactionPolicy(backend).summarize_chunked(
        [text(MessageRole.USER, "source")], on_telemetry=telemetry.append
    )

    assert result == "summary"
    assert backend.calls == 2
    assert telemetry[-1]["retries"] == 1


@pytest.mark.asyncio
async def test_summary_aclose_error_propagates_after_success() -> None:
    class ClosingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "sent"):
                raise StopAsyncIteration
            self.sent = True
            return StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, "summary"),
            )

        async def aclose(self):
            raise RuntimeError("close failed")

    class Backend:
        def complete(self, messages, tool_schemas):
            return ClosingStream()

    with pytest.raises(RuntimeError, match="close failed"):
        await CompactionPolicy(Backend()).summarize([text(MessageRole.USER, "source")])


@pytest.mark.asyncio
async def test_summary_aclose_cancellation_propagates_after_success() -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class ClosingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "sent"):
                raise StopAsyncIteration
            self.sent = True
            return StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, "summary"),
            )

        async def aclose(self):
            close_started.set()
            await release_close.wait()

    class Backend:
        def complete(self, messages, tool_schemas):
            return ClosingStream()

    task = asyncio.create_task(
        CompactionPolicy(Backend()).summarize([text(MessageRole.USER, "source")])
    )
    await close_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_close.set()


@pytest.mark.asyncio
async def test_summary_provider_error_stays_primary_when_aclose_fails() -> None:
    class ClosingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return StreamEvent(
                StreamEventType.ERROR,
                error=ErrorInfo("provider_failed", "provider failed"),
            )

        async def aclose(self):
            raise RuntimeError("close failed")

    class Backend:
        def complete(self, messages, tool_schemas):
            return ClosingStream()

    with pytest.raises(SummaryCompletionError, match="provider failed") as raised:
        await CompactionPolicy(Backend()).summarize([text(MessageRole.USER, "source")])
    assert raised.value.code == "provider_failed"


@pytest.mark.asyncio
async def test_chunked_compaction_maps_chunks_with_bounded_overlap_and_order() -> None:
    class OverlapBackend(CompletionBackend):
        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0
            self.sources: list[str] = []

        async def complete(self, messages, tool_schemas):
            source = messages[-1].content[0].text.split("\n\n", 1)[1]
            self.sources.append(source)
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.01 * (5 - len(self.sources)))
            self.active -= 1
            label = (
                source.split("chunk-")[1].split("-")[0]
                if "chunk-" in source else "reduce"
            )
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, f"summary-{label}"),
            )

    backend = OverlapBackend()
    result = await CompactionPolicy(backend).summarize_chunked(
        [text(MessageRole.USER, f"chunk-{index}-" + "x" * 40) for index in range(8)],
        max_source_tokens=30,
    )

    assert backend.maximum <= 3
    assert backend.maximum > 1
    assert result.startswith("summary-")
    map_sources = [source for source in backend.sources if "chunk-" in source]
    assert map_sources == sorted(
        map_sources, key=lambda source: int(source.split("chunk-")[1].split("-")[0])
    )
    reduction = [source for source in backend.sources if "summary-" in source]
    assert reduction
    assert [reduction[-1].index(f"summary-{index}") for index in range(8)] == sorted(
        reduction[-1].index(f"summary-{index}") for index in range(8)
    )


@pytest.mark.asyncio
async def test_chunked_compaction_cancellation_cleans_up_map_tasks() -> None:
    started: list[int] = []
    finalized: list[int] = []

    class CancelBackend(CompletionBackend):
        async def complete(self, messages, tool_schemas):
            index = len(started)
            started.append(index)
            try:
                await asyncio.sleep(10)
            finally:
                finalized.append(index)
            yield StreamEvent(StreamEventType.MESSAGE_END, message=text(MessageRole.ASSISTANT, "ok"))

    task = asyncio.create_task(
        CompactionPolicy(CancelBackend()).summarize_chunked(
            [text(MessageRole.USER, "x" * 80) for _ in range(8)], max_source_tokens=30
        )
    )
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(finalized) == set(started)
    assert len(started) <= 3


@pytest.mark.asyncio
async def test_chunked_error_closes_all_started_streams_and_does_not_start_queued() -> None:
    started: list[int] = []
    closed: list[int] = []

    class ErrorBackend(CompletionBackend):
        async def complete(self, messages, tool_schemas):
            index = len(started)
            started.append(index)
            try:
                if index == 0:
                    yield StreamEvent(StreamEventType.RETRY)
                    yield StreamEvent(
                        StreamEventType.ERROR,
                        error=ErrorInfo("failed", "map failed"),
                    )
                    return
                await asyncio.sleep(10)
                yield StreamEvent(StreamEventType.MESSAGE_END, message=text(MessageRole.ASSISTANT, "ok"))
            finally:
                closed.append(index)

    with pytest.raises(Exception, match="map failed"):
        await CompactionPolicy(ErrorBackend()).summarize_chunked(
            [text(MessageRole.USER, f"chunk-{i}-" + "x" * 40) for i in range(8)],
            max_source_tokens=30,
        )
    assert set(closed) == set(started)
    assert len(started) <= 3


@pytest.mark.asyncio
async def test_chunked_telemetry_uses_provider_model_and_bounded_sanitized_usage() -> None:
    telemetry: list[dict] = []

    class ProviderBackend(CompletionBackend):
        async def complete(self, messages, tool_schemas):
            yield StreamEvent(StreamEventType.MESSAGE_START, data={"model": "model-a"})
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="x")
            yield StreamEvent(StreamEventType.MESSAGE_END,
                              message=text(MessageRole.ASSISTANT, "summary"),
                              data={"usage": {"input_tokens": 2, "output_tokens": 3,
                                               "secret": "nope", "content": {"bad": 1}}})

    await CompactionPolicy(ProviderBackend()).summarize_chunked(
        [text(MessageRole.USER, "x" * 80) for _ in range(4)],
        max_source_tokens=30,
        on_telemetry=telemetry.append,
    )
    assert telemetry
    assert telemetry[-1]["models"] == ["model-a"]
    assert telemetry[-1]["output_tokens"] == 3 * (telemetry[-1]["chunk_count"] + 1)
    assert set(telemetry[-1]) == {"source_size", "chunk_count", "map_seconds", "reduce_seconds", "total_seconds", "retries", "output_tokens", "models"}


@pytest.mark.asyncio
async def test_chunked_compaction_counts_retry_events_in_telemetry() -> None:
    telemetry: list[dict[str, object]] = []
    calls = 0

    class RetryBackend(CompletionBackend):
        async def complete(self, messages, tool_schemas):
            nonlocal calls
            call = calls
            calls += 1
            source = messages[-1].content[0].text.split("\n\n", 1)[1]
            if call in {0, 2} or "summary" in source:
                yield StreamEvent(StreamEventType.RETRY)
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, "summary"),
            )

    await CompactionPolicy(RetryBackend()).summarize_chunked(
        [text(MessageRole.USER, "x" * 80) for _ in range(4)],
        max_source_tokens=30,
        on_telemetry=telemetry.append,
    )

    assert telemetry[-1]["retries"] == 3


@pytest.mark.asyncio
async def test_chunked_compaction_reports_non_content_telemetry() -> None:
    telemetry: list[dict[str, object]] = []
    backend = FakeBackend([ScriptedTurn([TextContent("summary")], usage={"output_tokens": 2})] * 40)
    await CompactionPolicy(backend).summarize_chunked(
        [text(MessageRole.USER, "x" * 80) for _ in range(8)],
        max_source_tokens=30,
        on_telemetry=telemetry.append,
    )
    assert telemetry
    assert telemetry[-1]["chunk_count"] > 1
    assert telemetry[-1]["source_size"] > 0
    assert telemetry[-1]["map_seconds"] >= 0
    assert telemetry[-1]["total_seconds"] >= telemetry[-1]["map_seconds"]


@pytest.mark.asyncio
async def test_compaction_reduces_chunk_size_after_provider_context_error() -> None:
    class LimitedBackend(CompletionBackend):
        def __init__(self) -> None:
            self.sources: list[str] = []

        async def complete(self, messages, tool_schemas):
            assert tool_schemas == []
            source = messages[-1].content[0].text.split("\n\n", 1)[1]
            self.sources.append(source)
            if len(source) > 220:
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("context_length_exceeded", "stream error"),
                )
                return
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=text(MessageRole.ASSISTANT, "summary"),
            )

    backend = LimitedBackend()
    policy = CompactionPolicy(backend)

    result = await policy.summarize_chunked(
        [text(MessageRole.USER, "x" * 280)], max_source_tokens=100
    )

    assert result == "summary"
    assert len(backend.sources) > 2
    assert len(backend.sources[0]) > 220
    assert all(len(source) <= 220 for source in backend.sources[1:])
